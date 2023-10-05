# contruct dense feature maps:
# for rgb images:
# (16* down sampled diffusion features): uv + dfs
# for dpt images:
# (16* down sampled diffusion features): uv + dfs

# we thus need:
# rgb_df_uv: (32*44)*2 ~ the default keypoints
# rgb_df_feat: (32*44)*{} ~ need-upsampling to original size after pca
# rgb_kpts_uv : nkpts*2
# rgb_kpts_xyz: nkpts*3 (based on zoe)
# rgb_gf_feat:  nkpts*32

# dpt_df_uv: (32*44)*2 ~ the default keypoints
# dpt_df_feat: (32*44)*{} ~ need-upsampling to original size after pca
# dpt_kpts_uv : nkpts*2
# dpt_kpts_xyz: nkpts*3 (based on zoe)
# dpt_gf_feat:  nkpts*32

# to test:
# rgb_features 2 rgb_features (on zoe keypoints)
# dpt_features 2 dpt_features (on dpt keypoints)
# rgb_features 2 dpt_features (on zoe-dpt keypoints)


import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from tools.fcgf.fcgf_feat import fcgf_extractor
from tools.controlnet.cn_feat import control_extractor
from utils.utils import nn_match, dpt_3d_convert, suppress_stdout_stderr


class pipeline_feat():
    def __init__(self, cfg, update_df_feat=False, update_gf_feat=False):
        self.cfg = cfg
        self.nkpts = self.cfg.feat.merge.nkpts
        self.rgb_intrinsic = np.array(self.cfg.meta.rgb_intrinsic)
        self.dpt_intrinsic = np.array(self.cfg.meta.dpt_intrinsic)
        self.save_layers = cfg.feat.cn.save_layers
        self.check_layers = cfg.feat.cn.check_layers
        self.spconv = fcgf_extractor(pth = cfg.feat.fcgf.pth,
                                     vs = cfg.feat.fcgf.vs)
        # load_model = False if not update_df_feat else True # GPU memory friendly
        load_model=True
        self.control = control_extractor(load_model=load_model,
                                         seed=cfg.feat.cn.seed, 
                                         t = cfg.feat.cn.step, 
                                         basic= cfg.feat.cn.ckpts.basic,
                                         yaml=cfg.feat.cn.ckpts.yaml,
                                         sd_ckpt=cfg.feat.cn.ckpts.sd_ckpt,
                                         cn_ckpt=cfg.feat.cn.ckpts.cn_ckpt,
                                         prompt=cfg.feat.cn.prompt)
        self.convertor = dpt_3d_convert()
        self.nn_searcher = nn_match(nn_max_n=-1)
        self.update_df_feat = update_df_feat
        self.update_gf_feat = update_gf_feat
        self.dpt_scale = cfg.meta.dpt_scale
        self.nkpts = self.cfg.feat.merge.nkpts

    def spconv_feature_extract(self, 
                               dpt, 
                               dpt_scale = 1,
                               intrinsic = None, 
                               extrinsic = np.eye(4),
                               zoe = False):
        # projection and extract
        # all pc here under the identity pose
        if zoe:
            uv, d, pc = self.convertor.proj_depth(dpt, 
                                            intrinsic=intrinsic, 
                                            extrinsic=extrinsic, 
                                            depth_unit=dpt_scale, 
                                            filter_edge=self.cfg.proj.filter_edge, 
                                            window_s=self.cfg.proj.win_length,
                                            max_range=self.cfg.proj.max_range_inwin, 
                                            filter_far=self.cfg.proj.filter_far, 
                                            far_thres=self.cfg.proj.far_thres,
                                            filter_near=self.cfg.proj.filter_near, 
                                            near_thres=self.cfg.proj.near_thres,
                                            return_uv=True)
            # mask off out-of-range points
            mask = (d<self.cfg.eval.mask.max) & (d>self.cfg.eval.mask.min)
            pc = pc[mask]
        else:
            uv, d, pc = self.convertor.proj_depth(dpt, 
                                            intrinsic=intrinsic, 
                                            extrinsic=extrinsic, 
                                            depth_unit=dpt_scale, 
                                            filter_edge=False,
                                            filter_far=self.cfg.proj.filter_far, 
                                            far_thres=self.cfg.proj.far_thres,
                                            filter_near=self.cfg.proj.filter_near, 
                                            near_thres=self.cfg.proj.near_thres,
                                            return_uv=True)
            # mask off out-of-range points
            mask = (d<self.cfg.eval.mask.max) & (d>self.cfg.eval.mask.min)
            pc = pc[mask]
        idx, feat = self.spconv.run(pc, numpy=False)
        return {'pc':pc[idx], 'feat':feat}
        
    def cn_feature_extract(self, img_fn, type = 'rgb'):
        if type in ['rgb']:
            img, feats = self.control.rgb_feature(img_fn)
        elif type in ['dpt', 'zoe']:
            img, feats = self.control.dpt_feature(img_fn)
        else:
            raise TypeError(f'cn cannot process type {type}!')
        if type in ['rgb']:
            target_w, target_h = self.cfg.meta.rgb_size
        else:
            target_w, target_h = self.cfg.meta.dpt_size
        # select a subset of intermediate feature maps
        def return_uv(target_h, target_w, feat):
            hf, wf = feat.shape[1:]
            u = np.arange(wf)[None,:,None].repeat(hf,axis=0)
            v = np.arange(hf)[:,None,None].repeat(wf,axis=1)
            uv = np.concatenate([u, v],axis=-1)
            uv = uv.reshape(-1,2)
            uv = self.control.capturer.uv_back_to_origin(uv, target_h, target_w, hf, wf)
            return uv
        # check layers  cuda tensor -> cpu tensor
        feats = {i:feats[i].cpu() for i in self.save_layers}
        # uv in origin img: n*2
        uv = return_uv(target_h, target_w, feats[max(self.check_layers)]).reshape(-1,2)
        return {'uv':uv, 'feat':feats}

    def kpts_sampling(self, item, type='rgb'):
        if type in ['rgb']:
            uv = None
            if self.nkpts<0:
                uv = item['rgb_df_uv']
            else:
                w,h = self.cfg.meta.rgb_size
                # we sample keypoints on rgb and dpt using random sampling
                u = np.arange(w)[:,None,None].repeat(h,axis=1)
                v = np.arange(h)[None,:,None].repeat(w,axis=0)
                uv = np.concatenate([u,v],axis=-1)
                uv = uv.reshape(-1,2)
                uv = np.random.permutation(uv)[0:self.nkpts]
            zoe = np.load(item['zoe_fn'])
            zoe = zoe[uv[:,1],uv[:,0]]
            xyz = self.convertor.proj_2to3(uv, zoe, self.rgb_intrinsic, np.eye(4), depth_unit=1)
            return uv, xyz
        else:
            uv = None
            if self.nkpts<0:
                uv = item['dpt_df_uv']
            else:
                w,h = self.cfg.meta.dpt_size
                # we sample keypoints on rgb and dpt using random sampling
                u = np.arange(w)[:,None,None].repeat(h,axis=1)
                v = np.arange(h)[None,:,None].repeat(w,axis=0)
                uv = np.concatenate([u,v],axis=-1)
                uv = uv.reshape(-1,2)
                uv = np.random.permutation(uv)[0:self.nkpts]
            dpt = np.array(Image.open(item['dpt_fn']))/self.cfg.meta.dpt_scale
            dpt = dpt[uv[:,1],uv[:,0]]
            xyz = self.convertor.proj_2to3(uv, dpt, self.dpt_intrinsic, np.eye(4), depth_unit=1)
            return uv, xyz
        
    def feature_transfer_xyz(self, query, source, source_f, mask = True):
        # nn
        query = torch.from_numpy(query.astype(np.float32))
        source = torch.from_numpy(source.astype(np.float32))
        idx,dist = self.nn_searcher.find_nn_gpu(query, source, nn_max_n=-1, return_distance=True, dist_type='SquareL2')
        qf = source_f[idx]
        if mask:
            mask = dist[:,0]>self.cfg.feat.fcgf.trans_thres**2
            qf[mask] *= 0.0
        return qf

    def frame_cn(self, item):
        # rgb features
        if ('rgb_fn' in item):
            rgb_fn = item['rgb_fn']
            rgb_df_f = self.cn_feature_extract(rgb_fn, type = 'rgb') # a dict
            # diffsuion features
            item['rgb_df_uv'] = rgb_df_f['uv']
            item['rgb_df_f'] = rgb_df_f['feat'] # a dict
        if ('dpt_fn' in item):
            # dpt features
            dpt_fn = item['dpt_fn']
            dpt_df_f = self.cn_feature_extract(dpt_fn, type = 'dpt') # a dict
            # dpt diffusion features
            item['dpt_df_uv'] = dpt_df_f['uv']
            item['dpt_df_f'] = dpt_df_f['feat'] # a dict

    def frame_fcgf(self, item):
        if ('zoe_fn' in item):
            # zoe for rgb
            zoe_fn = item['zoe_fn']
            zoe = np.load(zoe_fn)
            zoe_spconv_f = self.spconv_feature_extract(zoe, 
                                                    dpt_scale=1,
                                                    intrinsic=self.rgb_intrinsic, 
                                                    extrinsic=np.eye(4),
                                                    zoe=True) # {pc:,feat:}
            # we only preserve geo-feats on sampled key-points
            kpts_uv, kpts_xyz = self.kpts_sampling(item,type='rgb')
            item['rgb_kpts_uv'] = kpts_uv
            item['rgb_kpts_xyz'] = kpts_xyz
            rgb_gf_f = self.feature_transfer_xyz(kpts_xyz,zoe_spconv_f['pc'],zoe_spconv_f['feat'])
            item['rgb_gf_f'] = rgb_gf_f

        if ('dpt_fn' in item):
            # dpt fcgf features
            dpt_fn = item['dpt_fn']
            # extract_fcgf_features
            dpt = np.array(Image.open(dpt_fn))/self.dpt_scale
            dpt_spconv_f = self.spconv_feature_extract(dpt, 
                                                dpt_scale=1,
                                                intrinsic=self.dpt_intrinsic, 
                                                extrinsic=np.eye(4),
                                                zoe=False) # {pc:,feat:}
            # we only preserve geo-feats on sampled key-points
            kpts_uv, kpts_xyz = self.kpts_sampling(item,type='dpt')
            item['dpt_kpts_uv'] = kpts_uv
            item['dpt_kpts_xyz'] = kpts_xyz
            dpt_gf_f = self.feature_transfer_xyz(kpts_xyz,dpt_spconv_f['pc'],dpt_spconv_f['feat'])
            item['dpt_gf_f'] = dpt_gf_f
        
    def process_frame(self, item):
        cond = os.path.exists(item['to_fn']) 
        if cond and (not self.update_df_feat) and (not self.update_gf_feat):
            return 0
        else:
            if cond:
                item = torch.load(item['to_fn'])
            if (not cond) or self.update_df_feat:
                print(f'Extracting diffusion features.')
                self.frame_cn(item)
            if (not cond) or self.update_gf_feat:
                print(f'Extracting geometric features.')
                self.frame_fcgf(item)
            torch.save(item, item['to_fn'])
        
    def process_meta(self, meta):
        for id, frame in tqdm(meta['frames'].items()):
            with suppress_stdout_stderr():
                self.process_frame(frame)

    def run(self, metas):
        for scene, meta in metas.items():
            print(f'Extracting features on {scene}...')
            self.process_meta(meta)