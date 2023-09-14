# from turtle import forward
from torchvision import models,transforms
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2,3],keepdim=keepdim)

class LPIPS(nn.Module):
    def __init__(self,use_dropout=True):
        super(LPIPS, self).__init__()
        self.chns = [64,128,256,512,512]
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.lins = [self.lin0,self.lin1,self.lin2,self.lin3,self.lin4]
        self.lins = nn.ModuleList(self.lins)

    def forward(self, outs0, outs1):
        feats0, feats1, diffs = {}, {}, {}
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = lpips.normalize_tensor(outs0[kk]), lpips.normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk]-feats1[kk])**2

        res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1,len(self.chns)):
            val += res[l]

        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030,-.088,-.188])[None,:,None,None])
        self.register_buffer('scale', torch.Tensor([.458,.448,.450])[None,:,None,None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def gram_matrix(feat):

    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)

    return gram
class SPADE(nn.Module):
    def __init__(self, in_chan,out_chan,reduce_ratio=1):
        super().__init__()
        param_free_norm_type = 'instance'
        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(in_chan, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(in_chan, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)
        ks = 3
        pw = (ks - 1) // 2
        self.conv_gamma = nn.Conv2d(out_chan//reduce_ratio, in_chan, kernel_size=ks, padding=pw)
        self.conv_beta = nn.Conv2d(out_chan//reduce_ratio, in_chan, kernel_size=ks, padding=pw)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(out_chan,out_chan//reduce_ratio, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.blending_gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.blending_beta = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.mlp_gamma = nn.Conv2d(out_chan//reduce_ratio, in_chan, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(out_chan//reduce_ratio, in_chan, kernel_size=ks, padding=pw)

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_chan,out_chan//reduce_ratio, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.relu = nn.ReLU()
        kw=1
        self.w1 = nn.Sequential(nn.Conv2d(out_chan,1,kernel_size=kw, padding=(kw - 1) // 2,bias=False),
                    nn.Sigmoid())
        self.w2 = nn.Sequential(nn.Conv2d(out_chan,out_chan,kernel_size=kw, padding=(kw - 1) // 2,bias=False),
                    nn.Sigmoid())
        
        self.conv_1 = nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1)
        self.actvn = nn.ReLU()
    def forward(self, f):

        kernel_size =3
        pad = (kernel_size - 1) // 2
        # cpm_x = f.mean(1,keepdim=True)

        m1 = F.avg_pool2d(f,kernel_size=kernel_size,stride=1,padding=pad)
        rm1 = F.avg_pool2d((f-m1)**2,kernel_size=kernel_size,stride=1,padding=pad)
        smask = self.w1(rm1)
        swx = smask*f

        m2=f.mean([2,3],keepdim=True)
        rm2 = ((f-m2)**2).mean([2,3], keepdim=True)
        cmask = self.w2(rm2)
        cwx = cmask*f

        normalized = self.param_free_norm(f)

        middle_avg = self.conv3(swx)

        gamma_avg = self.conv_gamma(middle_avg)
        beta_avg = self.conv_beta(middle_avg)

        active = self.mlp_shared(cwx)

        gamma_spade, beta_spade = self.mlp_gamma(active), self.mlp_beta(active)

        gamma_alpha = F.sigmoid(self.blending_gamma)
        beta_alpha = F.sigmoid(self.blending_beta)

        gamma_final = gamma_alpha * gamma_avg + (1 - gamma_alpha) * gamma_spade
        beta_final = beta_alpha * beta_avg + (1 - beta_alpha) * beta_spade
        out = normalized * (1 + gamma_final) + beta_final
        out = self.conv_1(out)
        return out

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.chns = [64,128,256,512,512]
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        # Rewrite the output layer of every block in the VGG network: maxpool->l2pool
        for x in range(0,4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])
        for param in self.parameters():
            param.requires_grad = False
        self.scaling_layer = ScalingLayer()
        self.LPIPS = LPIPS()
        self.load_state_dict(torch.load('./models/LPIPS_KADID10k.pth'))
        for param in self.parameters():
            param.requires_grad = False

        ## JND inference
        self.spade1 = SPADE(self.chns[0],self.chns[0])
        self.load_state_dict(torch.load('./models/FLRE+LPIPS.pth'))

    def forward_once(self, x):
        h = self.stage1(x)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def recon(self,ref):
        h = ref[0]
        h_relu1_2 = self.spade1(h)

        h = self.stage2(h_relu1_2)
        h_relu2_2 = h
        
        h = self.stage3(h_relu2_2)
        h_relu3_3 = h
        
        h = self.stage4(h_relu3_3)
        h_relu4_3 = h
        
        h = self.stage5(h_relu4_3)
        h_relu5_3 = h
        jnd_feats = [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]
        return jnd_feats

    def search(self,jnd_feats,ref_feats,dist_feats):
        new_ref_feats = []
        for i in range(len(self.chns)):
            ref0, dist0 = ref_feats[i], dist_feats[i]
            ## learn disturbance
            jnd = jnd_feats[i]
            res = (jnd - ref0).abs()
            ref0_up = torch.nn.functional.relu(ref0+res)
            ref0_low = torch.nn.functional.relu(ref0-res)
            diff_ori = (ref0-dist0).abs()
            diff_up = (ref0_up-dist0).abs()
            diff_low = (ref0_low-dist0).abs()
            logit = (res>diff_ori)
            href = logit*dist0 + torch.logical_not(logit)*((diff_low<=diff_up)*ref0_low+(diff_low>diff_up)*ref0_up)
            new_ref_feats.append(href)

        return new_ref_feats

    def forward(self,x,y):
        new_ref_feats = []
        x, y = self.scaling_layer(x), self.scaling_layer(y)
        with torch.no_grad():
            ref_feats = self.forward_once(x)
            dist_feats = self.forward_once(y)
        jnds = self.recon(ref_feats)
        new_ref_feats = self.search(jnds,ref_feats,dist_feats)
        score = self.LPIPS(new_ref_feats,dist_feats)
        return score
    
def prepare_image(image, resize=True):
    if resize and min(image.size)>256:
        image = transforms.functional.resize(image,256)
    image = transforms.ToTensor()(image)
    return image

if __name__ == '__main__':
    from PIL import Image
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, default='./images/I04.BMP')
    parser.add_argument('--dist', type=str, default='./images/i04_06_4.bmp')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ref = prepare_image(Image.open(args.ref).convert("RGB")).to(device)
    dist = prepare_image(Image.open(args.dist).convert("RGB")).to(device)

    model = Model().to(device)
    model.eval()
    score = model(ref, dist)
    print('score: %.4f' % score.item())
