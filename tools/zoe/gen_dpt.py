import os
import torch
import numpy as np
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
from tools.zoe.zoedepth.utils.config import get_config
from tools.zoe.zoedepth.models.builder import build_model


class img2dpt():
    def __init__(self, 
                 model_type = 'zoedepth',
                 model_fn = '/mnt/proj/SOTAs/ZoeDepth-main/checkpoints/ZoeD_M12_N.pt',
                 config_version = None,
                 intrinsic = np.array([[585,0,320],[0,585,240],[0,0,1]])):
        self.model_type = model_type
        self.model_fn = model_fn
        self.config_version = config_version
        self.intrinsic = intrinsic
        self.load_model()

    def o3d_show(self, pcs):
        pcds = []
        for pc in pcs:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=20))
            pcd.paint_uniform_color(np.random.rand(3))
            pcds.append(pcd)
        o3d.visualization.draw_geometries(pcds)
        return pcd

    def to_harmonic(self, input):
        M = input.shape[0]
        input = np.concatenate([input, np.ones([M,1])],axis=1)
        return input

    def proj_2to3(self, uv, depth, intrinsic, extrinsic, depth_unit = 1000):
        # input:
        # uv            M*2     the image coordinates of predicted pairs on sample image
        # depth         M       the depth of the matched voxels of sample image
        # intrinsic     3*3     the intrinsic matrix
        # extrinsic     4*4     the extrinsic matrix the the sample/depth image
        # output:
        # the corresponding depth of the matched pixels on the sample image
        # formula       xyz = extrinsic@(inv(intrinsic)@uvd)
        uv_harmonic = self.to_harmonic(uv)
        uv_harmonic = uv_harmonic * depth[:,None]/depth_unit
        camera_coor = (np.linalg.inv(intrinsic) @ uv_harmonic.T).T
        camera_coor = self.to_harmonic(camera_coor)
        world_coor  = (extrinsic @ camera_coor.T).T
        return world_coor[:,0:3]

    def proj_3to2(self, xyz, intrinsic, extrinsic):
        # input:
        # xyz           M*3     the xyz points
        # depth         M       the depth of the matched voxels of sample image
        # intrinsic     3*3     the intrinsic matrix
        # extrinsic     4*4     the extrinsic matrix the the sample/depth image
        # output:
        # the corresponding depth of the matched pixels on the sample image
        # formula       uvd=intrinsic(inv(extrinsic)@xyz)
        xyz = self.to_harmonic(xyz)
        xyz = np.linalg.inv(extrinsic) @ xyz.T
        uvd = intrinsic @ xyz[0:3]
        uvd = uvd.T
        uv, d = uvd[:,0:2]/uvd[:,-1:], uvd[:,-1]
        return uv, d
    
    def proj_depth(self, depth, intrinsic, extrinsic = np.eye(4), depth_unit = 1000,
                   window_s = 3, max_range = 0.2):
        h, w = depth.shape[0:2]
        u = np.arange(w)[None,:,None].repeat(h,axis=0)
        v = np.arange(h)[:,None,None].repeat(w,axis=1)
        uvd = np.concatenate([u, v, depth[:,:,None]],axis=-1)

        # condeuct mask
        mask = np.zeros_like(depth)
        for i in range(window_s, h):
            for j in range(window_s, w):
                check = depth[(i-window_s):(i+window_s), (j-window_s):(j+window_s)] / depth_unit
                check = np.max(check) - np.min(check)
                if check < max_range:
                    mask[i,j] = 1

        uvd = uvd[mask>0.5]
        uvd = uvd.reshape(-1,3)
        pc = self.proj_2to3(uvd[:,0:2], uvd[:,-1], intrinsic, extrinsic, depth_unit)
        return pc

    def load_model(self):
        # ZoeD_N
        if self.config_version is None:
            conf = get_config(self.model_type, "infer")
        else:
            conf = get_config(self.model_type, "infer", config_version = self.config_version)
        # use the weights downloaded
        if os.path.exists(self.model_fn):
            conf['pretrained_resource'] = f'local::{self.model_fn}'
        self.model_zoe = build_model(conf)
        ##### sample prediction
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.zoe = self.model_zoe.to(DEVICE)

    def predict(self, image):
        depth_numpy = self.zoe.infer_pil(image)  # as numpy
        # print(np.min(depth_numpy), np.max(depth_numpy))
        return depth_numpy
    
    def run(self, image):
        # get-depth
        dpt = self.predict(image)
        return dpt

    def run_fn(self, img_fn):
        image = Image.open(img_fn).convert('RGB')
        # get-depth
        dpt = self.predict(image)
        return dpt

    def run_show_gt(self, base_fn):
        # img2dpt
        dpt = self.run_fn(f'{base_fn}.color.png')
        pc = self.proj_depth(dpt, self.intrinsic, depth_unit= 1)
        # gt-depth
        dpt_gt = Image.open(f'{base_fn}.depth.png')
        dpt_gt = np.array(dpt_gt)
        pc_gt = self.proj_depth(dpt_gt, self.intrinsic, depth_unit=1000)
        # show depth
        rgb = np.array(Image.open(f'{base_fn}.color.png').convert('RGB'))
        plt.imshow(rgb)
        # plt.imshow(dpt)
        plt.show()
        # shwo projected point cloud
        pcds = self.o3d_show([pc, pc_gt])
        return dpt, pc


if __name__ == '__main__':
    projector = img2dpt(model_type='zoedepth_nk', model_fn='/mnt/proj/SOTAs/ZoeDepth-main/checkpoints/ZoeD_M12_NK.pt')
    scene = 'sun3d-hotel_uc-scan3'
    base_fn = f'/mnt/proj/Datasets/3DMatch/data/3DMatch_raw/{scene}/seq-01/frame-000000'
    projector.run_show_gt(base_fn)
