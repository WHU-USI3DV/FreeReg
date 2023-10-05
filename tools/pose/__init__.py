import cv2
import copy
import numpy as np
import open3d as o3d
from utils.utils import transform_points,random_se3, dpt_3d_convert

class refiner():
    def __init__(self):
        pass

    def center_cal(self,key_m0,key_m1,scores):
        key_m0=key_m0*scores[:,None]
        key_m1=key_m1*scores[:,None]
        key_m0=np.sum(key_m0,axis=0)
        key_m1=np.sum(key_m1,axis=0)
        return key_m0,key_m1

    def SVDR_w(self,beforerot,afterrot,scores):# beforerot afterrot Scene2,Scene1
        weight=np.diag(scores)
        H=np.matmul(np.matmul(np.transpose(afterrot),weight),beforerot)
        U,Sigma,VT=np.linalg.svd(H)
        return np.matmul(U,VT)

    def s_cal(self,key_m0,key_m1,center0,center1,scores):
        key_m0=(key_m0-center0[None,:])*scores[:,None]
        key_m1=(key_m1-center1[None,:])*scores[:,None]
        scale_0 = np.std(key_m0)
        scale_1 = np.std(key_m1)
        return scale_0/scale_1

    def R_cal(self,key_m0,key_m1,center0,center1,scores):
        key_m0=key_m0-center0[None,:]
        key_m1=key_m1-center1[None,:]
        return self.SVDR_w(key_m1,key_m0,scores)

    def t_cal(self,center0,center1,R):
        return center0-center1@R.T

    def Rt_cal(self,key_m0,key_m1,scores):
        scores=scores/np.sum(scores)
        center0,center1=self.center_cal(key_m0,key_m1,scores)
        R=self.R_cal(key_m0,key_m1,center0,center1,scores)
        t=self.t_cal(center0,center1,R)
        return 1,R,t

    def sRt_cal(self,key_m0,key_m1,scores, scaling = False):
        scores=scores/np.sum(scores)
        center0,center1=self.center_cal(key_m0,key_m1,scores)
        # scale
        if scaling:
            s = self.s_cal(key_m0,key_m1,center0,center1,scores)
            key_m1, center1 = key_m1*s, center1*s
        else:
            s = 1
        # rotation
        R=self.R_cal(key_m0,key_m1,center0,center1,scores)
        t=self.t_cal(center0,center1,R)
        return s,R,t
    
    def Refine_trans(self,key_m0,key_m1,T,scores,inlinerdist=None, scaling = False):
        key_m1_t=transform_points(key_m1,T)
        disp = key_m0-key_m1_t
        dist=np.sum(np.square(disp),axis=-1)
        overlap=np.where(dist<inlinerdist*inlinerdist)[0]
        if len(overlap)<4:return 1 ,copy.deepcopy(T), None
        scores=scores[overlap]
        key_m0=key_m0[overlap]
        key_m1=key_m1[overlap]
        s,R,t=self.sRt_cal(key_m0, key_m1, scores, scaling=scaling)
        Tnew=np.eye(4)
        Tnew[0:3,0:3]=s*R
        Tnew[0:3,3]=t
        return s, Tnew, overlap

class essentialor():
    def __init__(self, inlierd = 5.0, intrinsic = np.array([[585,0,320],[0,585,240],[0,0,1]])):
        self.inlinerd = inlierd # pixels
        self.convertor = dpt_3d_convert()
        self.intrinsic = intrinsic
    
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
    def __init__(self, inlierd = 10, iters = 50000, intrinsic = np.array([[585,0,320],[0,585,240],[0,0,1]])):
        self.inlinerd = inlierd # pixels
        self.iters = iters
        self.convertor = dpt_3d_convert()
        self.intrinsic = intrinsic
    
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
            return 0,random_se3()
        
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

class ransacor():
    def __init__(self, inlierd, iters = 50000, refinement = True):
        self.inlierd = inlierd
        self.iters = iters
        self.refinement = refinement
        self.refiner = refiner()

    def o3d_ransac(self, keys0, keys1, match, scaling = False):
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(keys0)
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(keys1)
        coors = o3d.utility.Vector2iVector(match)

        if hasattr(o3d, 'pipelines'):
            regmodule = o3d.pipelines
        else:
            regmodule = o3d

        result = regmodule.registration.registration_ransac_based_on_correspondence( \
        source = source_pcd, \
        target = target_pcd, \
        corres = coors,\
        max_correspondence_distance = self.inlierd,\
        estimation_method = regmodule.registration.TransformationEstimationPointToPoint(with_scaling = scaling), \
        ransac_n = 4,\
        criteria = regmodule.registration.RANSACConvergenceCriteria(self.iters, 1000))     
        trans = result.transformation
        return trans

    def run(self, keys0, keys1, match, scaling=False):          
        trans = self.o3d_ransac(keys0, keys1, match, scaling=scaling)
        scale = 1
        if self.refinement:
            # refine:
            Keys_m0 = keys0[match[:,0]]
            Keys_m1 = keys1[match[:,1]]
            scale, trans, _=self.refiner.Refine_trans(Keys_m1,Keys_m0,trans,np.ones(match.shape[0]),inlinerdist=self.inlierd*2.0, scaling=scaling)
            scale, trans, inlier=self.refiner.Refine_trans(Keys_m1,Keys_m0,trans,np.ones(match.shape[0]),inlinerdist=self.inlierd,scaling=scaling)
            # note that our final pose should not hold scale
            if not scale>0:
                scale = 1
            trans[0:3,0:3] = trans[0:3,0:3]/scale
        return inlier, trans

