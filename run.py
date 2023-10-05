from config import gen_config
from dataops.metas import gen_meta

class pipeline_run():
    def __init__(self,
                 cfg,
                 rgb_ftype,
                 dpt_ftype,
                 weight=[1,1,1,1,1,1,1,1],
                 processor_type='se3',
                 update_zoe = False,
                 update_df = False,
                 update_gf = True,
                 update_pca = False) -> None:
        self.cfg = cfg
        self.metas = gen_meta(self.cfg).run()
        self.rgb_ftype = rgb_ftype
        self.dpt_ftype = dpt_ftype
        self.weight = weight
        self.processor_type = processor_type
        # for result-saving
        self.basic = [rgb_ftype, dpt_ftype, weight[:len(rgb_ftype)], processor_type]
        # for fast evaluation
        self.update_zoe = update_zoe
        self.update_df = update_df
        self.update_gf = update_gf
        self.update_pca = update_pca

    def zoe(self):
        from pipeline.gen_zoe import pipeline_zoe
        pipeline_zoe(self.cfg,
                    update_dpt=self.update_zoe
                    ).run(self.metas)

    def feat(self):
        from pipeline.gen_feat import pipeline_feat
        pipeline_feat(self.cfg, 
                      update_df_feat=self.update_df,
                      update_gf_feat=self.update_gf
                      ).run(self.metas)
    
    def match(self):
        from pipeline.gen_match import pipeline_match
        pipeline_match(self.cfg, 
                       self.rgb_ftype,
                       self.dpt_ftype,
                       self.processor_type,
                       self.weight,
                       update_pca_feat=self.update_pca
                       ).run(self.metas)
    
    def eval(self):
        from pipeline.gen_eval import pipeline_eval
        pipeline_eval(self.cfg, 
                      self.basic
                      ).run(self.metas)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',default='3dmatch',type=str,help='3dmatch/scannet/kitti')
    parser.add_argument('--type',default='dg',type=str,help='d/g/dg')
    parser.add_argument('--nkpts',default=-1,type=int,help='number of keypoints per frame, -1 means *16 downsample kpts')
    parser.add_argument('--update_zoe',action='store_true')
    parser.add_argument('--update_df',action='store_true')
    parser.add_argument('--update_gf',action='store_true')
    parser.add_argument('--update_pca',action='store_true')
    args = parser.parse_args()
    
    name2dataset={
        '3dmatch':f'config/3dmatch.yaml',
        'scannet':f'config/scannet.yaml',
        'kitti':f'config/kitti.yaml'
    }
    
    if args.type in ['dg']:
        rgb_ftype = ['rgb_df','rgb_gf']
        dpt_ftype = ['dpt_df','dpt_gf']
        processor_type = 'se3'
    elif args.type in ['d']:
        rgb_ftype = ['rgb_df']
        dpt_ftype = ['dpt_df']
        processor_type = 'pnp'
    elif args.type in ['g']:
        rgb_ftype = ['rgb_gf']
        dpt_ftype = ['dpt_gf']
        processor_type = 'se3'
    else:
        raise TypeError('wrong type! use d/g/dg, or you can modify run.py to diy your registration type.')
    
    # for fast evaluation 
    update_zoe = True if args.update_zoe else False
    update_df  = True if args.update_df  else False
    update_gf  = True if args.update_gf  else False # change only when use different kpts number
    update_pca = True if args.update_pca else False # change only when use different pca dimension

    cfg = gen_config(name2dataset[args.dataset])
    cfg.feat.merge.nkpts = args.nkpts
    runner = pipeline_run(cfg=cfg,
                          rgb_ftype=rgb_ftype,
                          dpt_ftype=dpt_ftype,
                          processor_type=processor_type,
                          update_zoe = update_zoe,
                          update_df = update_df,
                          update_gf = update_gf,
                          update_pca = update_pca)
    runner.zoe()
    runner.feat()
    runner.match()
    runner.eval()