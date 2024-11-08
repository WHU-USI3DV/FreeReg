import cv2
import copy
import torch
import numpy as np
from .utils import transform_points,random_se3, dpt_3d_convert

class kabsch_refine():
    def __init__(self,scaling):
        self.scaling = scaling

    def single_kabsch(self,spts,tpts,ws):
        ws = ws/np.mean(ws)
        wdiag = np.diag(ws)
        scenter=np.mean(spts*ws[:,None],0,keepdims=True)
        tcenter=np.mean(tpts*ws[:,None],0,keepdims=True)
        m = (spts-scenter).T @ wdiag @ (tpts-tcenter)
        if np.linalg.det(m)<1e-5:
            return None
        U,S,VT = np.linalg.svd(m)
        rotation = VT.T @ U.T   #predicted RT
        offset = tcenter - (scenter @ rotation.T)
        transform=np.concatenate([rotation,offset.T],1)
        out = np.eye(4)
        out[0:3] = transform
        return out 

    def sse3_cal(self,spts,tpts,ws=None):
        if ws is None:
            ws = np.ones([spts.shape[0]])
        ws = ws/np.mean(ws)
        if self.scaling:
            # scale estimation
            srange = spts-np.mean(spts*ws[:,None],axis=0,keepdims=True)
            trange = tpts-np.mean(tpts*ws[:,None],axis=0,keepdims=True)
            srange = np.mean(np.sqrt(np.sum(srange**2,axis=-1))*ws)
            trange = np.mean(np.sqrt(np.sum(trange**2,axis=-1))*ws)
            scale = trange/srange
            # rescale spt to tpt
            spts = spts*scale
        else:scale=1
        T = self.single_kabsch(spts,tpts,ws=ws)
        return scale,T

    def resolve_trans(self,spts,tpts,s,T,thres):
        # get inliers
        spts_T = transform_points(copy.deepcopy(spts)*s,T)
        disp = np.sum(np.square(spts_T-tpts),axis=-1)
        mask = disp<(thres**2)
        if np.sum(mask)<6:return None,None
        s,Tref = self.sse3_cal(spts[mask],tpts[mask])
        return s,Tref

    def run(self,spts,tpts,pps,scale,T,thres=[]):
        mspts = spts[pps[:,0]]
        mtpts = tpts[pps[:,1]]
        Tinit = copy.deepcopy(T)
        for thre in thres:
            scale,T = self.resolve_trans(mspts,mtpts,scale,T,thre)
            if T is None: return Tinit
        return T

class essentialor():
    def __init__(self, cfg):
        self.inlinerd = cfg.reg.pnp.ird # pixels
        self.convertor = dpt_3d_convert()
        self.intrinsic = np.array(self.cfg.meta.rgb_intrinsic)
    
    def run(self, kpts_q, kpts_d, matches = None):
        if len(kpts_q)<6:return np.array([[0,0],[1,1]]).astype(np.int32), random_se3()
        if matches is None:
            assert kpts_q.shape[0] == kpts_d.shape[0]
            kpts_qm = kpts_q
            kpts_tm = kpts_d
        else:
            kpts_qm = kpts_q[matches[:,0]]
            kpts_tm = kpts_d[matches[:,1]]
        # essential matrix
        essential_matrix, _ = cv2.findEssentialMat(kpts_qm, kpts_tm, cameraMatrix=self.intrinsic, 
                                                   method=cv2.RANSAC, prob=0.99, threshold=self.inlinerd)
        _, R, t, _ = cv2.recoverPose(essential_matrix, kpts_qm, kpts_tm, self.intrinsic)
        T = np.eye(4)
        T[0:3,0:3] = R
        T[0:3,-1:] = t
        return 0, T

class pnpor():
    def __init__(self, cfg):
        self.cfg = cfg
        self.inlinerd = cfg.reg.pnp.ird # pixels
        self.iters = cfg.reg.pnp.iters
        self.convertor = dpt_3d_convert()
        self.intrinsic = np.array(cfg.meta.rgb_intrinsic)
    
    def run3d3d(self, kpts_q, kpts_d, matches = None):
        if len(kpts_q)<6:return np.array([[0,0],[1,1]]).astype(np.int32), random_se3()
        if matches is None:
            assert kpts_q.shape[0] == kpts_d.shape[0]
            kpts_qm = kpts_q
            pair_xyz = kpts_d
        else:
            kpts_qm  = kpts_q[matches[:,0]]
            pair_xyz = kpts_d[matches[:,1]]
        pair_uv, _ = self.convertor.proj_3to2(kpts_qm, self.intrinsic, np.eye(4))
        # conduct ransac pnp -- cv2 obtaion invT
        ret, rvec, tvec, inlier = cv2.solvePnPRansac(objectPoints=pair_xyz, 
                                            imagePoints=pair_uv, 
                                            cameraMatrix=self.intrinsic.astype(np.float32), 
                                            distCoeffs=np.zeros((4,1)).astype(np.float32), 
                                            useExtrinsicGuess=False, 
                                            iterationsCount=self.iters, 
                                            reprojectionError=self.inlinerd, 
                                            confidence=0.999,
                                            flags=cv2.SOLVEPNP_ITERATIVE)
        if inlier is None:
            return np.array([[0,0],[1,1]]).astype(np.int32), random_se3()
        inlier = inlier[:,0]
        # inv T
        R,_ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[0:3,-1:] = tvec
        T[0:3,0:3] = R
        T = np.linalg.inv(T)
        return inlier, T

    def run(self, kpts_q, kpts_d, matches = None):
        if len(kpts_q)<6:return np.array([[0,0],[1,1]]).astype(np.int32), random_se3()
        if matches is None:
            assert kpts_q.shape[0] == kpts_d.shape[0]
            pair_uv = kpts_q
            pair_xyz = kpts_d
        else:
            pair_uv  = kpts_q[matches[:,0]]
            pair_xyz = kpts_d[matches[:,1]]
        # conduct ransac pnp -- cv2 obtaion invT
        pair_uv = pair_uv.astype(np.float32)
        pair_xyz = pair_xyz.astype(np.float32)
        try:
            ret, rvec, tvec, inlier = cv2.solvePnPRansac(objectPoints=pair_xyz, 
                                                imagePoints=pair_uv, 
                                                cameraMatrix=self.intrinsic.astype(np.float32), 
                                                distCoeffs=np.zeros((4,1)).astype(np.float32), 
                                                useExtrinsicGuess=False, 
                                                iterationsCount=self.iters, 
                                                reprojectionError=self.inlinerd, 
                                                confidence=0.999,
                                                flags=cv2.SOLVEPNP_ITERATIVE)
        except:
            return random_se3()
        # inv T
        R,_ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[0:3,-1:] = tvec
        T[0:3,0:3] = R
        T = np.linalg.inv(T)  
        return T

class kabschor():
    def __init__(self, 
                 ird_3d = 0.3,
                 ird_2d = 10,
                 iters = 50000,
                 max_T_pre_batch=5000):

        self.convertor = dpt_3d_convert()
        self.max_T_pre_batch = max_T_pre_batch # for memory-friendly
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.scaling = True
        self.w_2d = 0.5
        self.ird_3d = ird_3d
        self.ird_2d = ird_2d
        self.iters = iters
        self.np_per_hypo = 4
        self.refinement = True
        self.refiner = kabsch_refine(self.scaling)
        
    def set_intrinsic(self, intrinsic):
        self.intrincpu = intrinsic
        self.intrinsic = torch.from_numpy(self.intrincpu.astype(np.float32)).cuda()

    def det_check(self,rots):
        # rot: h*3*3
        det  = rots[:,0,0] * (rots[:,1,1]*rots[:,2,2]-rots[:,1,2]*rots[:,2,1])
        det -= rots[:,1,0] * (rots[:,0,1]*rots[:,2,2]-rots[:,0,2]*rots[:,2,1])
        det += rots[:,2,0] * (rots[:,0,1]*rots[:,1,2]-rots[:,0,2]*rots[:,1,1])
        return det

    def kabsch_hypos(self,
                     beforeT,
                     afterT,
                     n_hypo,
                     np_per_hypo=4,
                     cov_det_check = True,
                     rot_det_check = True):
        # beforeT: m*3 zoe-pts
        # afterT : m*3 dpt-pts
        # sample idxs:
        m = afterT.shape[0]
        per_iter = int(m/np_per_hypo)
        triplets = []
        n_tri = 0
        while n_tri<n_hypo:
            idx = np.random.permutation(m)[:int(per_iter*np_per_hypo)]
            triplets.append(idx.reshape(per_iter,np_per_hypo))
            n_tri+=per_iter
        triplets = np.concatenate(triplets,axis=0)[0:n_hypo]
        triplets = triplets.reshape(-1)
        # lift tensor to GPU
        beforeT_gpu = torch.from_numpy(beforeT.astype(np.float32)).to(self.device)
        afterT_gpu = torch.from_numpy(afterT.astype(np.float32)).to(self.device)
        # get triplets
        beforeT_tri = beforeT_gpu[triplets].reshape(n_hypo,np_per_hypo,3)
        afterT_tri = afterT_gpu[triplets].reshape(n_hypo,np_per_hypo,3)
        # centralization
        beforeT_center = torch.mean(beforeT_tri,dim=1)
        afterT_center = torch.mean(afterT_tri,dim=1)
        beforeT_gpu = beforeT_tri - beforeT_center[:,None]
        afterT_gpu = afterT_tri - afterT_center[:,None]
        # scale
        if self.scaling:
            beforeT_radius = torch.norm(beforeT_gpu,dim=-1)
            afterT_radius = torch.norm(afterT_gpu,dim=-1)
            beforeT_std = torch.mean(torch.abs(beforeT_radius),dim=-1)
            afterT_std = torch.mean(torch.abs(afterT_radius),dim=-1)
            scale = afterT_std/(beforeT_std+1e-3)
            # scale back
            beforeT_gpu = beforeT_gpu*scale[:,None,None]
            beforeT_center = beforeT_center*scale[:,None]
        else:
            scale = torch.ones([n_hypo]).to(self.device)

        #### calculate rotations
        # covmatrix
        cov = torch.einsum(f'hpa,hpb->hab',beforeT_gpu,afterT_gpu)
        if cov_det_check:
            # cov bad check for svd divation
            det_sample = torch.abs(self.det_check(cov))>1e-6
            beforeT_tri = beforeT_tri[det_sample]
            beforeT_center = beforeT_center[det_sample]
            afterT_tri = afterT_tri[det_sample]
            afterT_center = afterT_center[det_sample]
            scale = scale[det_sample]
            cov = cov[det_sample]
        # rotation estimation
        U,S,VT = torch.linalg.svd(cov)
        rot = torch.einsum('hba,hcb->hac',VT,U)
        if rot_det_check:
            # we need to remove out bad Ts with det(R)=-1
            det_check = self.det_check(rot)
            det_sample = det_check>1e-3
            beforeT_tri = beforeT_tri[det_sample]
            beforeT_center = beforeT_center[det_sample]
            afterT_tri = afterT_tri[det_sample]
            afterT_center = afterT_center[det_sample]
            scale = scale[det_sample]
            rot = rot[det_sample]
        #### translation estimation
        trans = afterT_center - torch.einsum(f'hab,hb->ha',rot,beforeT_center)
        Ts = torch.cat([rot,trans[:,:,None]],dim=-1)
        Ts = torch.cat([Ts,torch.zeros_like(Ts)[:,0:1,:]],dim=1)
        Ts[:,-1,-1] +=1.
        return beforeT_tri, afterT_tri, scale, Ts
    
    def gen_hypos(self, beforeT,afterT,n_hypo,thres,np_per_hypo=4):
        # beforeT: m*3 zoe-pts
        # afterT : m*3 dpt-pts
        beforeT_tri, afterT_tri, scales, Ts = self.kabsch_hypos(beforeT, afterT,n_hypo,np_per_hypo)
        # to camera coor
        hpts_c = torch.einsum(f'hab,hnb->hna',Ts[:,0:3,0:3],beforeT_tri*scales[:,None,None])+Ts[:,0:3,-1][:,None]
        # disp
        disp = torch.norm(hpts_c-afterT_tri,dim=-1)
        rmse_sq = torch.mean(disp**2,dim=-1)
        # here use half-3d_ird-thres
        thres_sq = (thres/2.)**2
        # kept transformations
        scales = scales[rmse_sq<thres_sq].to('cpu').numpy()
        Ts = Ts[rmse_sq<thres_sq].to('cpu').numpy()
        return scales,Ts

    def proj_3to2(self,Ts,pts):
        # Ts: torch.Tensor.CUDA N_hypo*4*4
        # pts: torch.Tensor.CUDA N_pts*3
        # pts->harmonic pts
        hpts = torch.cat([pts,torch.ones_like(pts)[:,0:1]],dim=1)
        # transform back
        invTs = torch.linalg.inv(Ts)
        # to camera coor
        hpts_c = torch.einsum(f'hab,nb->hna',invTs,hpts)[:,:,0:3]
        # to camera plain
        hpts_cp = torch.einsum(f'ba,hna->hnb',self.intrinsic,hpts_c)
        # depth
        proj_dpts = hpts_cp[:,:,-1] # h*n
        # pixel
        proj_uvs = hpts_cp[:,:,0:2]/(proj_dpts[:,:,None]+1e-5) # h*n*2
        return proj_uvs,proj_dpts

    def score_hypos_2d(self,uvs,proj_uvs,proj_dpts,thres):
        # uvs: image coor in correspondences torch.Tensor.CUDA n*2
        # proj_uvs: h*n*2
        # proj_dpts: h*n
        distance = torch.norm(proj_uvs-uvs[None],dim=-1)
        # filter outliers
        scores = torch.sum(distance<thres,dim=-1)
        return scores

    def score_hypos_3d(self,spts,tpts,scale,Ts,thres):
        # spts: matches*3
        # tpts: matches*3
        # scale: kept Transformations
        # Ts: kept Transformations*4*4
        h = len(scale)
        spts_hypo = spts[None].repeat(h,1,1)
        spts_hypo = spts_hypo*scale[:,None,None]
        # to harmonic
        spts_hypo = torch.cat([spts_hypo,torch.ones_like(spts_hypo)[:,:,0:1]],dim=-1)
        # align
        spts_hypo = torch.einsum(f'hac,hnc->hna',Ts,spts_hypo)[:,:,0:3]
        # distance
        distance = torch.norm(spts_hypo-tpts[None],dim=-1)
        # proj_dpts normalization
        scores = torch.sum(distance<thres,dim=-1)
        return scores

    def ransac(self,suvs,spts,tpts,scale,Ts,thres2d,thres3d):
        # to tensor
        # uvs: matches*2
        # pts: matches*3
        # Ts: hypotheses*4*4
        suvs_gpu = torch.from_numpy(suvs.astype(np.float32)).to(self.device)
        spts_gpu = torch.from_numpy(spts.astype(np.float32)).to(self.device)
        tpts_gpu = torch.from_numpy(tpts.astype(np.float32)).to(self.device)
        scale_gpu = torch.from_numpy(scale.astype(np.float32)).to(self.device)
        Ts_gpu = torch.from_numpy(Ts.astype(np.float32)).to(self.device)
        # we batch process
        scores_2d, scores_3d = [],[]
        for i in range(int(Ts.shape[0]/self.max_T_pre_batch)+1):
            Ts_gpu_b = Ts_gpu[i*self.max_T_pre_batch:(i+1)*self.max_T_pre_batch]
            scale_b = scale_gpu[i*self.max_T_pre_batch:(i+1)*self.max_T_pre_batch]
            # proj 3 to 2
            proj_uvs, proj_dpts = self.proj_3to2(Ts_gpu_b,tpts_gpu)
            # scores
            scores_b_2d = self.score_hypos_2d(suvs_gpu,proj_uvs,proj_dpts,thres2d).to('cpu').numpy()
            # scores on 3d
            scores_b_3d = self.score_hypos_3d(spts_gpu,tpts_gpu,scale_b,Ts_gpu_b,thres3d).to('cpu').numpy()
            # score normalziation
            scores_2d.append(scores_b_2d)
            scores_3d.append(scores_b_3d)
        scores_2d = np.concatenate(scores_2d,axis=0)
        scores_3d = np.concatenate(scores_3d,axis=0)
        scores_2d = scores_2d/(1+np.mean(scores_2d))
        scores_3d = scores_3d/(1+np.mean(scores_3d))
        self.w_2d = max(self.w_2d,0)
        self.w_2d = min(self.w_2d,1)
        scores = self.w_2d*scores_2d+(1-self.w_2d)*scores_3d
        if len(scores)<2:
            return 1,random_se3()
        # final
        return scale[np.argmax(scores)],Ts[np.argmax(scores)]

    def run(self,suvs,skpts,tkpts,pps):
        # suvs: kpts uv on source img
        # skpts: kpts xyz on source img
        # tkpts: kpts xyz on target img
        # pps: m*2
        # generate hypos
        smuvs=suvs[pps[:,0]]
        smkpts=skpts[pps[:,0]]
        tmkpts=tkpts[pps[:,1]]
        scales,hypos = self.gen_hypos(smkpts,tmkpts,self.iters,self.ird_3d,np_per_hypo=self.np_per_hypo)
        scale, out = self.ransac(smuvs,smkpts,tmkpts,scales,hypos,thres2d=self.ird_2d,thres3d=self.ird_3d)
        # refinement
        if self.refinement:
            out=self.refiner.run(skpts,tkpts,pps,scale,out,thres=[self.ird_3d*2,self.ird_3d])
        return out
    
