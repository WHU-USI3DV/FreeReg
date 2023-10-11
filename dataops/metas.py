import os
import numpy as np
from glob import glob
from PIL import Image
from utils.utils import trans_gt_for_kitti
from tools.dense.depth_map_utils import name2densefunc


class gen_meta(): 
    def __init__(self,cfg):
        self.cfg = cfg
        # basedir with :
        # scene/0.depth.png\0.color.png\0_pose.txt\0.ply
        # scene/eval/
        # scene/query_database_overlap.txt
        self.base = cfg.meta.base
        self.save_base = cfg.meta.feat_base
        # check pairs > gt overlap
        self.pair_type = cfg.meta.pair_type
        self.overlap_check = cfg.meta.overlap_pair
        self.n_seq_pair = cfg.meta.seq_n_pair
        # others
        self.rgb_intrinsic = np.array(cfg.meta.rgb_intrinsic)
        self.dpt_intrinsic = np.array(cfg.meta.dpt_intrinsic)
        self.dpt_scale = cfg.meta.dpt_scale

    def makedirs(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def run(self):
        scenes = glob(f'{self.base}/*')
        scenes.sort()
        metas = {}
        for scene in scenes:
            name = str.split(scene,'/')[-1]
            # init
            metas[name] = {}
            metas[name]['frames'] = {}
            metas[name]['pairs'] = []
            # frames
            query_fns = glob(f'{scene}/query/*.color.png')
            for q in query_fns:
                q_id = str.split(q,'/')[-1]
                q_id = str.split(q_id,'.')[0]
                
                # densify the input sparse depth map
                if self.cfg.meta.densify:
                    dpt_fn = f'{self.base}/{name}/query/{q_id}.dense.depth.png'
                    if not os.path.exists(dpt_fn):
                        dpt = np.array(Image.open(f'{self.base}/{name}/query/{q_id}.depth.png')) / self.dpt_scale
                        dpt = name2densefunc[self.cfg.meta.densefunc](dpt.astype(np.float32))
                        dpt = dpt * self.dpt_scale
                        dpt = dpt.astype(np.uint16)
                        dpt = Image.fromarray(dpt)
                        dpt.save(dpt_fn)
                else:
                    dpt_fn = f'{self.base}/{name}/query/{q_id}.depth.png'
                
                # gt depth for evaluaiton
                if os.path.exists(f'{self.base}/{name}/query/{q_id}.color.gtd.png'):
                    rgb_gt_dpt = np.array(Image.open(f'{self.base}/{name}/query/{q_id}.color.gtd.png'))/self.dpt_scale
                else:
                    if self.cfg.meta.densify:
                        rgb_gt_dpt = np.array(Image.open(f'{self.base}/{name}/query/{q_id}.dense.depth.png'))/self.dpt_scale
                    else:
                        rgb_gt_dpt = np.array(Image.open(f'{self.base}/{name}/query/{q_id}.depth.png'))/self.dpt_scale
                        assert (rgb_gt_dpt.shape[1] == self.cfg.meta.rgb_size[0]) and (rgb_gt_dpt.shape[0] == self.cfg.meta.rgb_size[1])

                if os.path.exists(f'{self.base}/{name}/query/{q_id}.depth.gtd.png'):
                    dpt_gt_dpt = np.array(Image.open(f'{self.base}/{name}/query/{q_id}.depth.gtd.png'))/self.dpt_scale
                else:
                    if self.cfg.meta.densify:
                        dpt_gt_dpt = np.array(Image.open(f'{self.base}/{name}/query/{q_id}.dense.depth.png'))/self.dpt_scale
                    else:
                        dpt_gt_dpt = np.array(Image.open(f'{self.base}/{name}/query/{q_id}.depth.png'))/self.dpt_scale
                
                item = {
                    'q_id': q_id,
                    # conform to rgb intrinsic and rgb image size
                    'rgb_fn': f'{self.base}/{name}/query/{q_id}.color.png',
                    'zoe_fn': f'{self.base}/{name}/eval/query/{q_id}.depth.gen.npy',
                    # conform to dpt intrinsic and dpt image size
                    'dpt_fn': dpt_fn,
                    'dpt_scale': self.dpt_scale,
                    # for evaluation
                    'rgb_gtd': rgb_gt_dpt,
                    'dpt_gtd': dpt_gt_dpt,
                    # extrinsic
                    'ext':np.loadtxt(f'{self.base}/{name}/query/{q_id}_pose.txt'),
                    # feature fn
                    'to_fn': f'{self.save_base}/{name}/feat/{q_id}.feat.pth',
                }
                metas[name]['frames'][q_id] = item
            # pair
            if self.pair_type in ['overlap']:
                over = np.loadtxt(f'{scene}/query/overlap.txt')
                over = over[over[:,-1]>self.overlap_check]
                for p in over:
                    i, j, o = int(p[0]), int(p[1]), p[2]
                    sext = metas[name]['frames'][f'{i}']['ext']
                    text = metas[name]['frames'][f'{j}']['ext']
                    gt = np.linalg.inv(text)@sext
                    item = {
                        'q_id': f'{i}',
                        'd_id': f'{j}',
                        'overlap': o,
                        'to_fn': f'{self.save_base}/{name}/match/{i}-{j}.trans.npz',
                        'gt': gt
                    }
                    metas[name]['pairs'].append(item)

            elif self.pair_type in ['seq']:
                for i in range(len(query_fns)-self.n_seq_pair):
                    sext = metas[name]['frames'][f'{i}']['ext']
                    for j in range(1,self.n_seq_pair):
                        text = metas[name]['frames'][f'{i+j}']['ext']
                        gt = np.linalg.inv(text)@sext
                        if self.cfg.name in ['kitti']:
                            gt = trans_gt_for_kitti(gt)
                        item = {
                            'q_id': f'{i}',
                            'd_id': f'{i+j}',
                            'overlap': 0.8, # no use
                            'to_fn': f'{self.save_base}/{name}/match/{i}-{i+j}.trans.npz',
                            'gt':gt
                        }
                        metas[name]['pairs'].append(item)
            else:
                raise TypeError('Wrong pairing type!: cfg.meta.pair_type:overlap/seq')
            # makedirs
            self.makedirs(f'{self.save_base}/{name}/feat')
            self.makedirs(f'{self.save_base}/{name}/match')
        return metas
