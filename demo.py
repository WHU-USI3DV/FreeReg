import os
import torch
import numpy as np
import open3d as o3d
from PIL import Image
from Utils.pose import kabschor
from Utils.utils import dpt_3d_convert,edge_filter,visual_pcd
from tools.DepthPro.command_pro_dpt import apple_pro_depth
from tools.YOHO.yoho_extract import yoho_extractor,nn_match

class Pipe():
    def __init__(self,nkpts=3000,vs=None):
        self.vs = vs
        self.nkpts = nkpts
        self._load_model()
        self._load_solver()
    
    def _load_model(self):
        self.depthpro = apple_pro_depth(ckpt = f'tools/DepthPro/checkpoints/depth_pro.pt', device = 'cuda')
        self.yoho = yoho_extractor(fcgf_ckpt = 'tools/YOHO/model/Backbone/best_val_checkpoint.pth',
                                   yoho_ckpt = 'tools/YOHO/model/PartI_train/model_best.pth', device = 'cuda')
    
    def _load_solver(self):
        self.projector = dpt_3d_convert()
        self.matcher = nn_match()
        self.solver = kabschor()

    def _load_img(self,img_fn):
        return np.array(Image.open(img_fn))/255.

    def _load_pcd(self,pcd_fn):
        if pcd_fn[-3:] == 'npy':
            return np.load(pcd_fn)
        else:
            pcd = o3d.io.read_point_cloud(pcd_fn)
            pcd = np.array(pcd.points)[:,0:3]
            return pcd

    @torch.no_grad()
    def _image_to_3d(self, image):
        depth,_,_,intrinsic = self.depthpro(image)
        sky = depth > 199.
        xyz = self.projector.proj_depth(depth,intrinsic,depth_unit=1.)
        edg = edge_filter(depth,sky,times=0.05).reshape(-1)
        xyz = xyz[(~edg)&(~sky.reshape(-1))]
        xyz = np.random.permutation(xyz)[:50000]
        return xyz, intrinsic
    
    def _determine_vs(self, image_pc, pc):
        if self.vs is not None: return 0
        delta = 0
        for pc in [image_pc,pc]:
            pc_delta = np.max(pc,axis=0) - np.min(pc,axis=0)
            delta += np.mean(pc_delta)
        delta /= 2
        self.vs = delta / 3 * 0.025
    
    @torch.no_grad()
    def _extract_yoho(self, pc, nkpts=5000):
        '''
        pc numpy array n*3
        '''
        kpts, yoho_desc, yoho_eqv = self.yoho.run(pc, self.vs, nkpts)
        return kpts, yoho_desc
    
    def _match(self, sf, tf):
        # des0 --> n0*32
        # des1 --> n1*32
        msid,mtid = self.matcher.find_corr(sf,tf)
        pps = np.concatenate([msid[:,None],mtid[:,None]],axis=-1)
        return pps
    
    def _pose(self, suvs, skpts, tkpts, pps, intrinsic):
        self.solver.set_intrinsic(intrinsic)
        T = self.solver.run(suvs,skpts,tkpts,pps)
        return T
    
    def _check(self, image, pc, intrinsic, T):
        H,W = image.shape[0:2]
        uvs,_ = self.projector.proj_3to2(pc,intrinsic,T)
        msk = (uvs[:,0]>0) & (uvs[:,1]>0) & (uvs[:,0]<W) & (uvs[:,1]<H)
        rgb = np.ones_like(pc)*0.5
        valid_uvs = uvs[msk].astype(np.int32)
        rgb[msk] = image[valid_uvs[:,1],valid_uvs[:,0]]
        visual_pcd(pc,rgb,normal=True)
        
    def __call__(self, img, pc):
        print('[Mono Depth Estimation] Project image to 3D with DepthPro...')
        img_pc,intrinsic = self._image_to_3d(img)
        self._determine_vs(img_pc,pc)
        print('[YOHO Features] Extract YOHO features on recon point cloud...')
        img_kpts, img_feats = self._extract_yoho(img_pc,self.nkpts)
        img_uvs,_ = self.projector.proj_3to2(img_pc,intrinsic,np.eye(4))
        print('[YOHO Features] Extract YOHO features on given point cloud...')
        pcd_kpts, pcd_feats = self._extract_yoho(pc,self.nkpts)
        print('[Match Corresponses] Match the constructed YOHO features to matches...')
        pps = self._match(img_feats,pcd_feats).astype(np.int16)
        print('[Kabsch] Solve the final pose...')
        T = self._pose(img_uvs,img_kpts,pcd_kpts,pps,intrinsic)
        self._check(img,pc,intrinsic,T)
        print("[Result] Estimated intrinsic of image:\n", intrinsic)
        print("[Result] Estimated extrinsic of image:\n", np.linalg.inv(T))
        
if __name__=='__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img',default="demo/outdoor/000183.png",type=str,help='dir of image file')
    parser.add_argument('--pcd',default="demo/outdoor/000189.npy",type=str,help='dir of point cloud file, pcd/ply/npy')
    # The following parameters do not need setting if you just want to use F-FreeReg on-the-fly.
    parser.add_argument('--nkpts',default=5000,type=int,help='extract 5k kpts on each pc')
    parser.add_argument('--vs',default=None,type=float,help='voxel size for pc voxelization, referring https://github.com/HpWang-whu/YOHO/tree/master/others,\
                                                             if set to None, we will auto-set it.')
    args = parser.parse_args()

    pipe = Pipe(args.nkpts,args.vs)
    img = pipe._load_img(args.img)
    pcd = pipe._load_pcd(args.pcd)
    pipe(img,pcd)