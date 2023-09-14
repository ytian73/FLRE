# from turtle import forward
from torchvision import models,transforms
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
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
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))
        # DISTS
        self.register_parameter("alpha", nn.Parameter(torch.randn(1, sum(self.chns),1,1)))
        self.register_parameter("beta", nn.Parameter(torch.randn(1, sum(self.chns),1,1)))
        self.alpha.data.normal_(0.1,0.01)
        self.beta.data.normal_(0.1,0.01)
        weights = torch.load('./models/weights.pt')

        self.alpha.data = weights['alpha']
        self.beta.data = weights['beta']

        for param in self.parameters():
            param.requires_grad = False

        ## JND inference
        self.spade1 = SPADE(self.chns[1],self.chns[1])
        self.load_state_dict(torch.load('./models/FLRE+DISTS.pth'))
    def DISTS(self,feats0,feats1):
        dist1 = 0
        dist2 = 0
        c1 = 1e-6
        c2 = 1e-6
        w_sum = self.alpha.sum() + self.beta.sum()
        alpha = torch.split(self.alpha/w_sum, self.chns, dim=1)
        beta = torch.split(self.beta/w_sum, self.chns, dim=1)
        for k in range(len(self.chns)):
            x_mean = feats0[k].mean([2,3], keepdim=True)
            y_mean = feats1[k].mean([2,3], keepdim=True)
            S1 = (2*x_mean*y_mean+c1)/(x_mean**2+y_mean**2+c1)
            dist1 = dist1+(alpha[k]*S1).sum(1,keepdim=True)

            x_var = ((feats0[k]-x_mean)**2).mean([2,3], keepdim=True)
            y_var = ((feats1[k]-y_mean)**2).mean([2,3], keepdim=True)
            xy_cov = (feats0[k]*feats1[k]).mean([2,3],keepdim=True) - x_mean*y_mean
            S2 = (2*xy_cov+c2)/(x_var+y_var+c2)
            dist2 = dist2+(beta[k]*S2).sum(1,keepdim=True)

        score = 1 - (dist1+dist2)
        return score
    def forward_once(self, x):
        h = (x-self.mean)/self.std
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
        return img1, img2
    def forward(self,x,y):
        new_ref_feats = []
        with torch.no_grad():
            ref_feats = self.forward_once(x)
            dist_feats = self.forward_once(y)

        jnds = self.recon(ref_feats)
        new_ref_feats = self.search(jnds,ref_feats,dist_feats)

        score = self.DISTS(new_ref_feats,dist_feats)
        return score
    
  
def prepare_image(image, resize=True):
    if resize and min(image.size)>256:
        image = transforms.functional.resize(image,256)
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

