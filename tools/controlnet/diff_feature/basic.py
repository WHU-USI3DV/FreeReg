from ..utils.share import *
import cv2
import torch
import random
import einops
import numpy as np
from PIL import Image
from copy import deepcopy
import torchvision.transforms as tvtf
from pytorch_lightning import seed_everything
from ..cldm.model import create_model, load_state_dict


class capture():
    def __init__(self, 
                 load_model = True, 
                 basic = './tools/controlnet/models', 
                 yaml = 'control_v11f1p_sd15_depth.yaml', 
                 sd_ckpt = 'v1-5-pruned.ckpt', 
                 cn_ckpt = 'control_v11f1p_sd15_depth.pth', 
                 seed = -1, 
                 t = 150) -> None:
        # basic
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # diffusion related
        self.basic = basic
        self.yaml = yaml
        self.sd_ckpt = sd_ckpt
        self.cn_ckpt = cn_ckpt
        self.seed = seed

        # else:
        self.strength = 1.0
        self.eta = 1.0
        self.uncond_scale = 5.0
        self.steps = 20
        self.seed = seed
        self.guess_mode = False
        self.only_mid_control = False
        self.a_prompt = 'best quality'
        self.n_prompt = 'lowres, bad anatomy, bad hands, cropped, worst quality'

        if load_model:
            self.load_model()
            self.prepare()

        # input parses
        self.img_resolution = 512
        self.dpt_resolution = 512
        
        # update t
        if load_model:
            self.t = t
            index = round((1000 - self.t)/(1000/self.steps))
            self.t = int((self.steps-index)*(1000/self.steps))
            print(f'We force to use step-{self.t} (~{t} rather than {t}) for our control process use {self.steps} steps!')
            self.tlist = torch.tensor([self.t]).to(self.device)


    def prepare(self):
        if self.seed == -1:
            self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        # to cuda
        self.model.to(self.device)
        self.model.control_model.to(self.device)
        self.model.first_stage_model.to(self.device)
        self.model.cond_stage_model.to(self.device)
        self.control_scales = [self.strength * (0.825 ** float(12 - i)) for i in range(13)] if self.guess_mode else ([self.strength] * 13)
        self.model.control_scales = self.control_scales

    def load_model(self):
        basic_yaml = f'{self.basic}/{self.yaml}'
        basic_sd_pth = f'{self.basic}/{self.sd_ckpt}'
        basic_cn_pth = f'{self.basic}/{self.cn_ckpt}'
        self.model = create_model(basic_yaml).cpu()
        self.model.load_state_dict(load_state_dict(basic_sd_pth, location=self.device), strict=False)
        self.model.load_state_dict(load_state_dict(basic_cn_pth, location=self.device), strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()

    def pca_feat(self, X, n_components = 3):
        # x should be c*{any shape}
        # conduct normalization
        X = X/torch.norm(X,dim=0,keepdim=True)
        # fit
        X = X.cuda()
        c, *size = X.shape
        X = X.reshape(c,-1).T
        n, c = X.shape
        mean = torch.mean(X, axis=0)
        X = X - mean
        covariance_matrix = 1 / n * torch.matmul(X.T, X)
        eigenvalues, eigenvectors = torch.linalg.eig(covariance_matrix)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
        idx = torch.argsort(-eigenvalues)
        eigenvectors = eigenvectors[:, idx]
        proj_mat = eigenvectors[:, 0:n_components]
        # project
        X = X.matmul(proj_mat).T
        X = X.reshape(tuple([-1] + size))
        return X.cpu()
    
    def pca_feats(self, xlist, n_components = 256):
        # [tensor: c*h*w]
        hws = []
        split = [0]
        flatten_xlist = []
        outlist = []
        for item in xlist:
            # the final output feature shape
            c,*size = item.shape
            hws.append(tuple([n_components] + size))
            item = item.reshape(c,-1)
            split.append(split[-1]+item.shape[1])
            flatten_xlist.append(item)
        flatten_x = torch.cat(flatten_xlist,dim=1)
        feat = self.pca_feat(flatten_x, n_components)
        # reshape back
        for i in range(len(xlist)):
            feat_i = feat[:,split[i]:split[i+1]]
            feat_i = feat_i.reshape(hws[i])
            outlist.append(feat_i)
        return outlist

    def fmap_resize(self, flist, target_h = None, target_w = None):
        if (target_h is None) or (target_w is None):
            target_h = 0
            target_w = 0
            for f in flist:
                c,h,w = f.shape
                if h>target_h:target_h=h
                if w>target_w:target_w=w
        # reisze
        outlist = []
        for item in flist:
            item = tvtf.Resize(size=(target_h, target_w))(item)
            outlist.append(item)
        return outlist

    def visualize_img(self, img):
        # input should be h*w*3
        assert img.shape[-1] == 3 
        img = Image.fromarray(img)
        img.show()

    def visualize_chw_feat(self, imgin):
        # input should be c*h*w feature map
        assert len(imgin.shape) == 3        
        img = deepcopy(imgin)
        if img.shape[0] > 3:
            img = self.pca_feat(img)
        img = einops.rearrange(img, 'c h w -> h w c').clone()

        # normalize to 0-255
        def visual_normalize(img):
            img = img.astype(np.float64)
            vmin = np.percentile(img, 2)
            vmax = np.percentile(img, 85)
            img -= vmin
            img /= vmax - vmin
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
            return img

        img = visual_normalize(img)
        img = Image.fromarray(img)
        img.show()

    def visualize_diffusion_result(self, img):
        # input should be c*h*w, stable diffusion output results [-1,1] -> [0.255]
        img = einops.rearrange(img, 'c h w -> h w c').clone()
        img = img.cpu().numpy()
        img = img*127.5+127.5
        img = img.clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img)
        img.show()
        img.save('./diff_result.png')
    
    def merge_feat(self, featlist_s:list, featlist_t:list, checklayers:list, pca_dim = None, h=None, w=None):
        # merge which features
        featlist_s = [featlist_s[i] for i in checklayers]
        featlist_t = [featlist_t[i] for i in checklayers]
        # conduct pca per-layer
        if pca_dim is not None:
            pcalist_s, pcalist_t = [], []
            for i in range(len(featlist_s)):
                sfeat = featlist_s[i]
                tfeat = featlist_t[i]
                # conduct pca
                sfeat,tfeat = self.pca_feats([sfeat,tfeat],pca_dim)
                pcalist_s.append(sfeat)
                pcalist_t.append(tfeat)
        else:
            pcalist_s = featlist_s
            pcalist_t = featlist_t
        # map to the largest
        pcalist_s = self.fmap_resize(pcalist_s, h, w)
        pcalist_t = self.fmap_resize(pcalist_t, h, w)
        # to a larget map
        fs = torch.cat(pcalist_s,dim=0)
        ft = torch.cat(pcalist_t,dim=0)
        # fs: (len(checklayers)*pca_dim)*h*w,torch.Tensor
        # ft: (len(checklayers)*pca_dim)*h*w,torch.Tensor
        return fs, ft

    def uv_back_to_origin(self, uv, ho, wo, hd, wd):
        # reproject to origin rgb
        scale_x, scale_y = int(ho/hd), int(wo/wd)
        uv[:,1] *= scale_x
        uv[:,0] *= scale_y
        uv[:,1] += int(scale_x/2.0)
        uv[:,0] += int(scale_y/2.0)
        return uv


