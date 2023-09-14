# from turtle import forward
from torchvision import models,transforms
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ot.lp import wasserstein_1d
def ws_distance(X,Y,P=2,win=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    chn_num = X.shape[1]
    X_sum = X.sum().sum()
    Y_sum = Y.sum().sum()

    X_patch   = torch.reshape(X,[win,win,chn_num,-1])
    Y_patch   = torch.reshape(Y,[win,win,chn_num,-1])
    patch_num = (X.shape[2]//win) * (X.shape[3]//win)

    X_1D = torch.reshape(X_patch,[-1,chn_num*patch_num])
    Y_1D = torch.reshape(Y_patch,[-1,chn_num*patch_num])

    X_1D_pdf = X_1D / (X_sum + 1e-6)
    Y_1D_pdf = Y_1D / (Y_sum + 1e-6)

    interval = np.arange(0, X_1D.shape[0], 1)
    all_samples = torch.from_numpy(interval).to(device).repeat([patch_num*chn_num,1]).t()

    X_pdf = X_1D * X_1D_pdf
    Y_pdf = Y_1D * Y_1D_pdf

    wsd   = wasserstein_1d(all_samples, all_samples, X_pdf, Y_pdf, P)

    L2 = ((X_1D - Y_1D) ** 2).sum(dim=0)
    w  =  (1 / ( torch.sqrt(torch.exp( (- 1/(wsd+10) ))) * (wsd+10)**2))

    final = wsd + L2 * w
    # final = wsd

    return final.sum()

class Downsample(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        self.padding = (filter_size - 2 )//2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:,None]*a[None,:])
        g = g/torch.sum(g)
        self.register_buffer('filter', g[None,None,:,:].repeat((self.channels,1,1,1)))
        # print (g)

    def forward(self, input):
        input = input**2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out+1e-12).sqrt()
def to_patch(x,ptch_size = 4):
        x_patch = torch.stack(torch.split(x, ptch_size, dim=-2))
        ptchs = torch.concat(torch.split(x_patch, ptch_size, dim=-1))
        return ptchs
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
        
        kw=1
        self.w1 = nn.Sequential(nn.Conv2d(out_chan,1,kernel_size=kw, padding=(kw - 1) // 2,bias=False),
                    nn.Sigmoid())
        self.w2 = nn.Sequential(nn.Conv2d(out_chan,out_chan,kernel_size=kw, padding=(kw - 1) // 2,bias=False),
                    nn.Sigmoid())
        
        self.conv_1 = nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1)
       # self.actvn = nn.ReLU()
    def forward(self, f):

        kernel_size =3
        pad = (kernel_size - 1) // 2

        m1 = F.avg_pool2d(f,kernel_size=kernel_size,stride=1,padding=pad)
        rm1 = F.avg_pool2d((f-m1)**2,kernel_size=kernel_size,stride=1,padding=pad)
        smask = self.w1(rm1)
        swx = smask*f

        m2=f.mean([2,3],keepdim=True)
        rm2 = ((f-m2)**2).mean([2,3], keepdim=True)
        cmask = self.w2(rm2)
        cwx = cmask*f
#
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

        self.chns = [3,64,128,256,512,512]
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        # Rewrite the output layer of every block in the VGG network: maxpool->l2pool
        for x in range(0,4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), Downsample(channels=64))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), Downsample(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), Downsample(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), Downsample(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])
        for param in self.parameters():
            param.requires_grad = False


        ## JND inference
        self.spade1 = SPADE(self.chns[1],self.chns[1])
        self.load_state_dict(torch.load('./models/FRLE+DeepWSD.pth'))

    def DeepWSD(self, ref_feats, dist_feats,window):
        score = 0
        for k in range(len(self.chns)):
            row_padding = round(ref_feats[k].size(2) / window) * window - ref_feats[k].size(2)
            column_padding = round(ref_feats[k].size(3) / window) * window - ref_feats[k].size(3)

            pad = nn.ZeroPad2d((column_padding, 0, 0, row_padding))
            feats0_k = pad(ref_feats[k])
            feats1_k = pad(dist_feats[k])
            tmp = ws_distance(feats0_k, feats1_k, win=window) # the k-th score of the ii-th image
            score = score + tmp
        score = score / (k+1)
        score =  torch.log(score + 1)**2
        return score
    def forward_once(self, x):
        h = x
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [x,h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]
    
    def recon(self,ref):
        x = ref[0]
        h = ref[1]
        h_relu1_2 = self.spade1(h)

        h = self.stage2(h_relu1_2)
        h_relu2_2 = h
        
        h = self.stage3(h_relu2_2)
        h_relu3_3 = h
        
        h = self.stage4(h_relu3_3)
        h_relu4_3 = h
        
        h = self.stage5(h_relu4_3)
        h_relu5_3 = h
        jnd_feats = [x,h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]
        return jnd_feats

    def search(self,jnd_feats,ref_feats,dist_feats):
        new_ref_feats = []
        for i in range(len(self.chns)):
            ref0, dist0 = ref_feats[i], dist_feats[i]
            if (i !=0):
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
            else:
                new_ref_feats.append(ref0)
        return new_ref_feats
    def downsample(self,img1, img2, maxSize = 256):
        _,channels,H,W = img1.shape
        f = int(max(1,np.round(max(H,W)/maxSize)))

        aveKernel = (torch.ones(channels,1,f,f)/f**2).to(img1.device)
        img1 = F.conv2d(img1, aveKernel, stride=f, padding = 0, groups = channels)
        img2 = F.conv2d(img2, aveKernel, stride=f, padding = 0, groups = channels)
        # For an extremely Large image, the larger window will use to increase the receptive field.
        if f >= 5:
            win = 16
        else:
            win = 4
        return img1, img2, win, f
    def forward(self,x,y):
        new_ref_feats = []
        x, y,windows,_ = self.downsample(x, y)
        with torch.no_grad():
            ref_feats = self.forward_once(x)
            dist_feats = self.forward_once(y)

        jnds = self.recon(ref_feats)
        new_ref_feats = self.search(jnds,ref_feats,dist_feats)
        score = self.DeepWSD(new_ref_feats,dist_feats,windows)
        return score
    
def prepare_image(image, repeatNum = 1): ##deepwsd 
    H, W = image.size
    if max(H,W)>512 and max(H,W)<1000:
        image = transforms.functional.resize(image,[256,256])
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0)

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

