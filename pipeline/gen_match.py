import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from utils.utils import nn_match, random_se3
from tools.controlnet.cn_feat import capture
from tools.pose import pnpor, ransacor, essentialor

class pipeline_match():
    def __init__(self,
                 cfg,
                 rgb_ftype = ['rgb_df','rgb_gf'],
                 dpt_ftype = ['dpt_df','dpt_gf'],
                 processor_type = 'se3',
                 weight = [1,1,1,1,1,1,1,1],
                 update_pca_feat = False):
        self.cfg = cfg
        self.rgb_intrinsic = np.array(self.cfg.meta.rgb_intrinsic)
        self.dpt_intrinsic = np.array(self.cfg.meta.dpt_intrinsic)
        self.rgb_ftype = rgb_ftype
        self.dpt_ftype = dpt_ftype
        # feature merge
        self.weight = weight[0:len(rgb_ftype)]
        self.weight = np.array(self.weight)/np.sum(np.array(self.weight))
        # type check
        self.check()
        self.matcher = nn_match()
        self.processor_type = processor_type
        self.determine_processor()
        # else
        self.update_pca_feat = update_pca_feat
        self.check_layers = self.cfg.feat.cn.check_layers
        self.pca_dim = self.cfg.feat.cn.pca_dim
        self.capturor = capture(load_model = False) # we only need pca

    def merge_conflict_check(self):
        # note that for rgb and dpt features merging, only occurs when rgb-d input is given (rgb_intrinsic = dpt_intrinsic, rgb_size = dpt_size)
        if (not np.sum(self.rgb_intrinsic-self.dpt_intrinsic)<1e-5) or \
                    (not self.cfg.meta.rgb_size == self.cfg.meta.dpt_size):
            raise TypeError('NOT RGBD INPUT! CANNOT FUSE RGB AND DEPTH FEATURES!')

    def check(self):
        for ftypes in [self.rgb_ftype, self.dpt_ftype]:
            for ft in ftypes:
                if not ft in ['rgb_df','rgb_gf','dpt_df','dpt_gf']:
                    raise TypeError(f'We only support: rgb_df, rgb_gf, dpt_df, dpt_gf, but got {ft}!')
        # guarantee cn2cn spconv2spconv
        for i in range(len(self.rgb_ftype)):
            rgbf = self.rgb_ftype[i]
            dptf = self.dpt_ftype[i]
            rgbf_proce = str.split(rgbf,'_')[-1]
            dptf_proce = str.split(dptf,'_')[-1]
            if not rgbf_proce == dptf_proce:
                raise TypeError(f'{rgbf_proce} - {rgbf_proce} is not support. {rgbf_proce} and {dptf_proce} come from different embedding spaces!')
        # determine source and target for processor
        self.source_input = 'rgb'
        self.target_input = 'dpt'
        if ('dpt_df' in self.rgb_ftype) or ('dpt_gf' in self.rgb_ftype):
            self.source_input = 'dpt'
        if ('rgb_df' in self.dpt_ftype) or ('rgb_gf' in self.dpt_ftype):
            self.target_input = 'rgb'
        # merge conflict check if ['rgb***, dpt***] or [zoe*** dpt***]
        sft = [str.split(s,'_')[0] for s in self.rgb_ftype]
        tft = [str.split(t,'_')[0] for t in self.dpt_ftype]
        if ('rgb' in sft) and ('dpt' in  sft): self.merge_conflict_check()
        if ('rgb' in tft) and ('dpt' in  tft): self.merge_conflict_check()

    def determine_processor(self):
        # processor
        print(f'source-feat:{self.rgb_ftype}')
        print(f'target-feat:{self.dpt_ftype}')
        print(f'weight: {self.weight}')
        if (self.processor_type == 'se3'):
            #we must use zoe
            if (self.source_input in ['rgb']) or (self.target_input in ['rgb']):
                print(f'we use zoe-ransac solver for source-{self.source_input} and target-{self.target_input}!')
                self.processor = ransacor(self.cfg.reg.ransac.zoe_ird,
                                          iters = self.cfg.reg.ransac.iters)
                self.scaling = True
            else:
                print(f'we use dpt-ransac solver for source-{self.source_input} and target-{self.target_input}!')
                self.processor = ransacor(self.cfg.reg.ransac.dpt_ird,
                                          iters = self.cfg.reg.ransac.iters)
                self.scaling = False
        elif (self.processor_type == 'pnp'):
            # means source is also given dpt
            print(f'we use pnp solver for source-{self.source_input} and target-{self.target_input}!')
            self.processor = pnpor(self.cfg.reg.pnp.ird,
                                   iters = self.cfg.reg.pnp.iters,
                                   intrinsic = self.rgb_intrinsic)
        elif (self.processor_type == 'se3_force'):
                print(f'we force to use dpt-ransac solver for source-rgbd and target-rgbd!')
                self.processor = ransacor(self.cfg.reg.ransac.dpt_ird,
                                          iters = self.cfg.reg.ransac.iters)
                self.scaling = False
        elif (self.processor_type == 'essential'):
            print(f'we use essential solver for source-{self.source_input} and target-{self.target_input}!')
            self.processor = essentialor(3.0,
                                   intrinsic = self.rgb_intrinsic)
        else:
            raise TypeError('Wrong se3 solver type! use se3/pnp/se3_force/essential')

    def feature_pca_and_aggregate(self, sitem, titem, sid, tid):
        sf_merge, tf_merge = [], []
        for i in range(len(self.rgb_ftype)):
            rgbf = self.rgb_ftype[i]
            dptf = self.dpt_ftype[i]
            if self.cfg.feat.cn.pca_dim>0:
                # we can load for fast eval
                pca_feat_dir = str.rfind(sitem['to_fn'],'/')
                pca_feat_dir = sitem['to_fn'][:pca_feat_dir] + f'_pca/{rgbf}-{dptf}'
                # save df pca results
                rgbf_proce = str.split(rgbf,'_')[-1]
                if rgbf_proce in ['df']:
                    if not os.path.exists(pca_feat_dir): os.makedirs(pca_feat_dir)
                pca_feat_fn = f'{pca_feat_dir}/{sid}-{tid}.pth'
                if (os.path.exists(pca_feat_fn)) and (not self.update_pca_feat):
                    content = torch.load(pca_feat_fn)
                    sf = content['sf']
                    tf = content['tf']
                else:
                    rgbf_proce = str.split(rgbf,'_')[-1]
                    if rgbf_proce in ['df']:
                        sf = sitem[f'{rgbf}_f']
                        tf = titem[f'{dptf}_f']
                        # now we need to conduct pca on each feature map and conduct concatenate
                        # here checklayers already be applied when saving items
                        # the feature map size is c*h*w [*16 downsampling]
                        sf, tf = self.capturor.merge_feat(sf, tf, checklayers = self.check_layers, pca_dim = self.pca_dim)
                        content = {'sf':sf, 'tf':tf}
                        torch.save(content, pca_feat_fn)
                    else:
                        sf, tf = sitem[f'{rgbf}_f'], titem[f'{dptf}_f']
            else:
                sf, tf = sitem[f'{rgbf}_f'], titem[f'{dptf}_f']
            sf_merge.append(sf)
            tf_merge.append(tf)
        # we donnot concatenate here for df and gf lie on different kpts, we finally resample keypoints
        return sf_merge, tf_merge
    
    def feature_merge_on_kpts(self, feats, item, type = 'rgb'):
        w,h  = self.cfg.meta.rgb_size if type in ['rgb'] else self.cfg.meta.dpt_size
        kpts_uv = item['rgb_kpts_uv'] if type in ['rgb'] else item['dpt_kpts_uv']
        ftps = self.rgb_ftype if type in ['rgb'] else self.dpt_ftype
        for i in range(len(ftps)):
            ftype = ftps[i]
            proce = str.split(ftype,'_')[-1]
            if proce in ['df']:
                sf = feats[i] # c*h*w
                # up_sampling feature maps and select
                sf = torch.nn.functional.interpolate(sf[None], (h, w), mode='bilinear', align_corners=True)[0]
                c = sf.shape[0]
                sf = sf[:,kpts_uv[:,1],kpts_uv[:,0]]
                feats[i] = sf.T
        # concate merge
        for i in range(len(feats)):
            sf = feats[i]
            # normalize + reweight following: a tale of two features
            sf = sf/(torch.norm(sf, dim=-1, keepdim=True)+1e-5)
            sf *= self.weight[i]
            feats[i] = sf
        feats = torch.cat(feats,dim=-1)
        return feats
    
    def match_mask(self, ftype, item):
        # the holes in depth-map will harm registration, we thus remove out them if use dpt
        if ftype in ['rgb']:
            # rgb will not be disturbed by dpt-holes
            uv = item['rgb_kpts_uv']
            return np.arange(uv.shape[0])
        else:
            dpt = np.array(Image.open(item['dpt_fn']))/self.cfg.meta.dpt_scale
            dpt_uv = item['dpt_kpts_uv']  # densified dpt for feature extraction
            dpt = dpt[dpt_uv[:,1],dpt_uv[:,0]]
            mask = (dpt>self.cfg.match.dpt_mask.min) & (dpt<self.cfg.match.dpt_mask.max)
            return np.where(mask)[0]  

    def feature_match(self, sitem, titem, sf, tf):  
        torch.cuda.empty_cache()
        smask = self.match_mask(self.source_input, sitem)
        tmask = self.match_mask(self.target_input, titem)
        sf = sf[smask]
        tf = tf[tmask]
        # conduct match
        id0, id1 = self.matcher.find_corr(sf.cuda(), tf.cuda(), mutual=True)
        id0 = smask[id0]
        id1 = tmask[id1]
        pps = np.concatenate([id0[:,None], id1[:,None]],axis=1)
        return pps

    def match(self, sitem, titem, sid, tid):
        # pca on diffusion features
        sf, tf = self.feature_pca_and_aggregate(sitem, titem, sid, tid) 
        # merge features
        sf = self.feature_merge_on_kpts(sf,sitem,self.source_input)
        tf = self.feature_merge_on_kpts(tf,titem,self.target_input)
        # conduct feature matching
        pps = self.feature_match(sitem, titem, sf, tf)
        # conduct se3 solving
        if self.processor_type in ['pnp']:
            if self.source_input in ['rgb']:
                skpts = sitem['rgb_kpts_uv']
                tkpts = titem['dpt_kpts_xyz']
            elif self.target_input in ['rgb']:
                skpts = sitem['dpt_kpts_xyz']
                tkpts = titem['rgb_kpts_uv']
            else:
                skpts = sitem['rgb_kpts_uv']
                tkpts = titem['dpt_kpts_xyz']
            if pps.shape[0]<8:
                T = random_se3()
            else:
                _, T = self.processor.run(skpts, tkpts, pps)
        elif self.processor_type in ['se3_force']:
            # must be rgb-d
            skpts = sitem['dpt_kpts_xyz']
            tkpts = titem['dpt_kpts_xyz']
            if pps.shape[0]<5:
                T = random_se3()
            else:
                _, T = self.processor.run(skpts, tkpts, pps, self.scaling)
        elif self.processor_type in ['essential']:
            # Rgb-rgb note that essential decomposition cannot recover the absolute translation (scale is unknown)
            # Here t is |t|=1, thus RR is meaning-less!
            skpts = sitem['rgb_kpts_uv']
            tkpts = titem['rgb_kpts_uv']
            if pps.shape[0]<5:
                T = random_se3()
            else:
                _, T = self.processor.run(skpts, tkpts, pps)
        else: # se3
            skpts = sitem['rgb_kpts_xyz'] if self.source_input in ['rgb'] else sitem['dpt_kpts_xyz']
            tkpts = titem['rgb_kpts_xyz'] if self.target_input in ['rgb'] else titem['dpt_kpts_xyz']
            if pps.shape[0]<5:
                T = random_se3()
            else:
                _, T = self.processor.run(skpts, tkpts, pps, self.scaling)
        return pps, T


    def process_meta(self, meta):
        for pair in tqdm(meta['pairs']):
            id0, id1 = pair['q_id'], pair['d_id']
            sitem = meta['frames'][id0]        
            titem = meta['frames'][id1]  
            sitem = torch.load(sitem['to_fn'])  
            titem = torch.load(titem['to_fn'])  # with features
            # gtT: source->target
            # sext@s = text@q
            # inv(text)@sext @s = q  source to target
            gt = pair['gt']
            pps, T = self.match(sitem, titem, id0, id1)
            np.savez(pair['to_fn'], 
                     gt = gt, 
                     pps = pps, 
                     Tpre = T, 
                     source_type = self.source_input, 
                     target_type = self.target_input)

    def run(self,metas):
        for scene, meta in metas.items():
            print(f'Conducting registration on {scene}...')
            self.process_meta(meta)

