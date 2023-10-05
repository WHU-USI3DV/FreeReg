##############################
# introduction:
# processes: projection, densify, feature extraction, match, registration
##############################

import os
import numpy as np
import open3d as o3d
from PIL import Image
from config import gen_config
from tools.dense import name2densefunc
from pipeline.gen_zoe import pipeline_zoe
from pipeline.gen_feat import pipeline_feat
from pipeline.gen_match import pipeline_match
from pipeline.gen_eval import pipeline_eval
from utils.utils import dpt_3d_convert, save_depth

class rgb2dpt:
    def __init__(self,
                 cfg,
                 rgb_ftype = ['rgb_df','rgb_gf'],
                 dpt_ftype = ['dpt_df','dpt_gf'],
                 processor_type = 'se3',
                 ):
        self.cfg = cfg
        self.zoe = pipeline_zoe(self.cfg,
                                update_dpt=False)
        self.extractor = pipeline_feat(self.cfg,
                                       update_df_feat=True, 
                                       update_gf_feat=True)
        self.regor = pipeline_match(self.cfg,
                                    rgb_ftype=rgb_ftype,
                                    dpt_ftype=dpt_ftype,
                                    processor_type=processor_type,
                                    update_pca_feat=True,)
        self.basic = [rgb_ftype, dpt_ftype, np.ones(10)[:len(rgb_ftype)], processor_type]
        self.evalor = pipeline_eval(self.cfg, self.basic)
    
    def load_data(self):
        # rgb, dpt
        # for eval:
        # rgb_gtd, dpt_gtd, pose
        self.meta = {}
        self.meta['frames'] = {}
        self.meta['pairs'] = []
        # load source data (rgb only)
        frame_rgb = {
            'rgb_fn':f'{self.cfg.meta.base}/source_rgb.png',
            # here zoe is just a name, we will generate a zoe to save to this file
            'zoe_fn':f'{self.cfg.meta.base}/source_rgb.zoe.npy',
            'to_fn': f'{self.cfg.meta.feat_base}/demo/feat/0.feat.pth',
            # for eval, not need if no eval is needed
            'rgb_gtd':np.array(Image.open(f'{self.cfg.meta.base}/just_for_eval/source_rgb.gtd.png'))/self.cfg.meta.dpt_scale
        }

        # load target pc
        pc = o3d.io.read_point_cloud(f'{self.cfg.meta.base}/target_pc.ply')
        # project and densify
        # project pc to dpt camera center and save it to sparse depth
        projector = dpt_3d_convert()
        dpt = projector.proj_pc2dpt(pc,
                                    extrinsic=np.eye(4),  # sensor center of the point cloud
                                    intrinsic=np.array(self.cfg.meta.dpt_intrinsic),
                                    h=self.cfg.meta.dpt_size[1],
                                    w=self.cfg.meta.dpt_size[0])
        proj_dpt_fn = f'{self.cfg.meta.base}/target_dpt.proj.png'
        save_depth(proj_dpt_fn,dpt,scale=self.cfg.meta.dpt_scale)
        # then we densify it for network processing and save densified dpt
        dpt = name2densefunc[self.cfg.meta.densefunc](dpt.astype(np.float32))
        dpt_fn = f'{self.cfg.meta.base}/target_dpt.densify.png'
        save_depth(dpt_fn,dpt,self.cfg.meta.dpt_scale)
        # target meta
        frame_dpt = {
            'dpt_fn': dpt_fn,
            'proj_dpt_fn': proj_dpt_fn,
            'to_fn': f'{self.cfg.meta.feat_base}/demo/feat/1.feat.pth',
            # for eval, not need if no eval is needed
            'dpt_gtd':np.array(Image.open(f'{self.cfg.meta.base}/just_for_eval/target_dpt.gtd.png'))/self.cfg.meta.dpt_scale
        }
        self.meta['frames']['0'] = frame_rgb
        self.meta['frames']['1'] = frame_dpt
        
        # pair for registration evaluation
        pair = {
                'q_id': '0',
                'd_id': '1',
                'overlap': 0.5, # no use
                'to_fn': f'{self.cfg.meta.feat_base}/demo/match/0-1.trans.npz',
                # for eval, set to random if no eval is needed
                'gt': np.loadtxt(f'{self.cfg.meta.base}/just_for_eval/rgb2pc.gt.pose.txt')
            }
        self.meta['pairs'].append(pair)
        # create fn for feat and match saving
        if not os.path.exists(f'{self.cfg.meta.feat_base}/demo/feat'): os.makedirs(f'{self.cfg.meta.feat_base}/demo/feat')
        if not os.path.exists(f'{self.cfg.meta.feat_base}/demo/match'): os.makedirs(f'{self.cfg.meta.feat_base}/demo/match')
    
    def preprocess(self):
        # only frames with rgb_fn(zoe_fn) will be processed.
        self.zoe.run({'demo':self.meta})
 
    def extraction(self):
        # extract features on rgb_fn/zoe_fn and dpt_fn once given
        self.extractor.process_meta(self.meta)
    
    def match(self):
        self.regor.process_meta(self.meta)
    
    def eval(self):
        self.evalor.run({'demo':self.meta})
    
    def run(self):
        # load feature
        self.load_data()
        # densify and zoe
        self.preprocess()
        # extract features
        self.extraction()
        # registration
        self.match()
        # evaluation
        self.eval()

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',default='config/demo.yaml',type=str,help='config file name')
    parser.add_argument('--type',default='cz',type=str,help='c/z/cz')
    args = parser.parse_args()
    
    if args.type in ['cz']:
        rgb_ftype = ['rgb_df','rgb_gf']
        dpt_ftype = ['dpt_df','dpt_gf']
        processor_type = 'se3'
    elif args.type in ['c']:
        rgb_ftype = ['rgb_df']
        dpt_ftype = ['dpt_df']
        processor_type = 'pnp'
    elif args.type in ['z']:
        rgb_ftype = ['rgb_gf']
        dpt_ftype = ['dpt_gf']
        processor_type = 'se3'
    else:
        raise TypeError('wrong type! use c/z/cz, or you can modify demo.py to diy your registration type.')
    
    cfg = gen_config(args.cfg)    
    mm_reg = rgb2dpt(cfg=cfg,
                     rgb_ftype=rgb_ftype,
                     dpt_ftype=dpt_ftype,
                     processor_type=processor_type)
    mm_reg.run()

    from utils.drawer import visualizor
    visor = visualizor(mm_reg.cfg)
    pair = mm_reg.meta['pairs'][0]
    visor.draw_demo(pair, save_pth = './demo.png')
    