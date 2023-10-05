import time
import torch
import numpy as np
from tqdm import tqdm
from utils.r_eval import compute_T_err
from utils.utils import transform_points, dpt_3d_convert

class pipeline_eval():
    def __init__(self,
                 cfg,
                 basic = [['rgb_df'], ['dpt_df'], [1], 'pnp']):
        self.cfg = cfg
        self.ir_thres = self.cfg.eval.ir.thres
        self.rr_r_thres = self.cfg.eval.rr.r_thres
        self.rr_t_thres = self.cfg.eval.rr.t_thres
        self.rgb_intrinsic = np.array(self.cfg.meta.rgb_intrinsic)
        self.dpt_intrinsic = np.array(self.cfg.meta.dpt_intrinsic)
        self.convertor = dpt_3d_convert()
        # basic information
        self.rgb_ftype, self.dpt_ftype, self.weight, self.process_type = basic

    def ir_cal(self, sxyz, txyz, gt):
            if len(sxyz)<3:
                 return np.array([0])
            sxyz = transform_points(sxyz, gt)
            disp = sxyz - txyz
            dist = np.sqrt(np.sum(np.square(disp),axis=-1))
            return dist
    
    def eval_mask(self, ftype, item):
        if ftype in ['rgb']:
            uv = item['rgb_kpts_uv']
            gtd = item['rgb_gtd']
            gtd = gtd[uv[:,1],uv[:,0]]
            intrinsic = self.rgb_intrinsic
        else:
            uv = item['dpt_kpts_uv']
            gtd = item['dpt_gtd']
            gtd = gtd[uv[:,1],uv[:,0]]
            intrinsic = self.dpt_intrinsic
        gt_xyz = self.convertor.proj_2to3(uv,gtd,intrinsic,np.eye(4),depth_unit=1)
        mask = (gtd>self.cfg.eval.mask.min) & (gtd<self.cfg.eval.mask.max)
        return gt_xyz, mask  

    def eval_pair(self, source_type, target_type, sitem, titem, pps):
        gts, smask = self.eval_mask(source_type,sitem)
        gtt, tmask = self.eval_mask(target_type,titem)
        smask = smask[pps[:,0]]
        tmask = tmask[pps[:,1]]
        ppsmask = np.where(smask & tmask)[0]
        pps_eval = pps[ppsmask]
        gt_xyz_s = gts[pps_eval[:,0]]
        gt_xyz_t = gtt[pps_eval[:,1]]
        return gt_xyz_s, gt_xyz_t

    def run(self, metas):
        irs, ins, rrs, res, tes = [], [], [], [], []
        for scene, meta in metas.items():
            print(f'Evaling on {scene}...')
            for pair in tqdm(meta['pairs']):
                item = np.load(pair['to_fn'])
                # ir
                sid, tid = pair['q_id'], pair['d_id']
                sitem = torch.load(meta['frames'][sid]['to_fn'])
                titem = torch.load(meta['frames'][tid]['to_fn'])
                gt, stype, ttype, pps = item['gt'], item['source_type'], item['target_type'], item['pps']
                smatch_xyz, tmatch_xyz = self.eval_pair(stype, ttype, sitem, titem, pps)
                ird = self.ir_cal(smatch_xyz, tmatch_xyz, gt)
                ins.append(np.sum(ird<self.ir_thres))
                irs.append(np.mean(ird<self.ir_thres))
                # rr
                re, te = compute_T_err(item['gt'], item['Tpre'])
                res.append(re)
                tes.append(te)
                if re<self.rr_r_thres and te<self.rr_t_thres:
                    rrs.append(1)
                else:
                    rrs.append(0)
        irs, ins, rrs, res, tes = np.array(irs), np.array(ins), np.array(rrs), np.array(res), np.array(tes)
        # basic info
        msg = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\n'
        msg += f'Dataset: {self.cfg.name}\n'
        msg += f'Source feature: {self.rgb_ftype}\n'
        msg += f'Target feature: {self.dpt_ftype}\n'
        msg += f'Feature weight: {self.weight}\n'
        msg += f'SE(3) Solver: {self.process_type}\n'
        # feature match recall
        fmr = np.mean(irs>0.05)
        msg += f'Feature matching ratio at ir_thres {self.ir_thres}: {fmr}\n'
        # inlier ratio
        ir = np.mean(irs)
        msg += f'Inlier ratio at ir_thres {self.ir_thres}: {ir}\n'
        # inlier number
        inliner_n = np.mean(ins)
        msg += f'Inlier number at ir_thres {self.ir_thres}: {inliner_n}\n'
        # registration related
        re = np.mean(res)
        te = np.mean(tes)
        msg += f'Average rotation error: {re}; Average translation error: {te}\n'
        re_success = np.mean(res[rrs>0.5])
        te_success = np.mean(tes[rrs>0.5])
        msg += f'On successful pairs: Average rotation error: {re_success}; Average translation error: {te_success}\n'
        rr = np.mean(rrs.astype(np.int32))
        msg += f'Registration recall at r_thres {self.rr_r_thres} and t_thres {self.rr_t_thres}: {rr}\n'
        # print
        print(msg)
        # log
        with open(self.cfg.logfile, "a") as f:
             f.write(msg)