from ..utils.share import *
import torch
import einops
import numpy as np
from PIL import Image
from copy import deepcopy
from ..utils.util import resize_image, HWC3

class img_processor():
    def __init__(self, 
                 capturer,
                 ) -> None:
        # basic
        self.capturer = capturer

    def load_rgb(self, rgb_fn):
        rgb = np.array(Image.open(rgb_fn)).astype(np.uint8) #0-255
        img = deepcopy(rgb)
        img = HWC3(img)
        img = resize_image(img, self.capturer.img_resolution)
        return rgb, img

    def add_noise(self, img):
        # input img(feat_map) should be b*c*h*w
        img = self.capturer.model.q_sample(img, self.capturer.tlist)
        return img

    def sd_input(self, img):
        # rgb loading
        self.H, self.W, self.C = img.shape
        img = (torch.from_numpy(np.array(img).astype(np.float32))-127.5)/ 127.5  # must be [-1,1]
        img = einops.rearrange(img[None], 'b h w c -> b c h w').clone()
        img = img.to(self.capturer.device)
        return img

    def sd_process(self, img, prompt = ''):
        # diffusion encoding
        img = self.capturer.model.encode_first_stage(img)
        # encoder comes f -> 1*4*64*88
        img = self.capturer.model.get_first_stage_encoding(img).detach() 
        # add noise
        noise_img = self.add_noise(img)
        # diffusion u-net
        cond = {"c_crossattn": [self.capturer.model.get_learned_conditioning([prompt + ', ' + self.capturer.a_prompt])]}
        cond_txt = torch.cat(cond['c_crossattn'], 1).to(self.capturer.device)
        with torch.no_grad():
            _, inter_feats = self.capturer.model.model.diffusion_model(x=noise_img, 
                                timesteps=self.capturer.tlist, 
                                context=cond_txt, 
                                control=None, 
                                only_mid_control=self.capturer.only_mid_control,
                                per_layers = True)
        # 1*c*h*w -> c*h*w
        inter_feats = [i[0].detach() for i in inter_feats]
        return inter_feats

    def sd_single_img(self, img_fn:str, prompt = ''):
        # sd input:
        rgb, img = self.load_rgb(img_fn)
        # diffusion features
        img = self.sd_input(img)
        # a list of c*h*w layer feature maps at self.capturer.t
        chw_feat_list = self.sd_process(img, prompt=prompt)
        return rgb, chw_feat_list

    def img_match(self, img_fn_source, img_fn_target, check_layers, pca_dim = 256):
        # get features
        rgbs, featlist_s = self.sd_single_img(img_fn_source)
        rgbt, featlist_t = self.sd_single_img(img_fn_target)
        # feature list to feature map
        fs, ft = self.capturer.merge_feat(featlist_s, featlist_t, check_layers, pca_dim = pca_dim)
        # conduct feature match
        uvs, uvt = self.capturer.chw_img_match(fs, ft)
        # uvst back to origin image size
        uvs = self.capturer.uv_back_to_origin(uvs, rgbs.shape[0], rgbs.shape[1], fs.shape[1], fs.shape[2])
        uvt = self.capturer.uv_back_to_origin(uvt, rgbt.shape[0], rgbt.shape[1], ft.shape[1], ft.shape[2])
        # draw
        self.capturer.draw_match(rgbs, rgbt, uvs, uvt)
    

