import os
import cv2
import time
import torch
import numpy as np
import open3d as o3d

class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0.0
    self.sq_sum = 0.0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count
    self.sq_sum += val**2 * n
    self.var = self.sq_sum / self.count - self.avg**2

class Timer(object):
  """A simple timer."""

  def __init__(self, binary_fn=None, init_val=0):
    self.total_time = 0.
    self.calls = 0
    self.start_time = 0.
    self.diff = 0.
    self.binary_fn = binary_fn
    self.tmp = init_val

  def reset(self):
    self.total_time = 0
    self.calls = 0
    self.start_time = 0
    self.diff = 0

  @property
  def avg(self):
    return self.total_time / self.calls

  def tic(self):
    # using time.time instead of time.clock because time time.clock
    # does not normalize for multithreading
    self.start_time = time.time()

  def toc(self, average=True):
    self.diff = time.time() - self.start_time
    self.total_time += self.diff
    self.calls += 1
    if self.binary_fn:
      self.tmp = self.binary_fn(self.tmp, self.diff)
    if average:
      return self.avg
    else:
      return self.diff

class dpt_3d_convert():
    def __init__(self):
        # we reimplement here for speed up
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        uv, d = uvd[:,0:2]/(uvd[:,-1:]+1e-5), uvd[:,-1]
        return uv, d

    def proj_depth(self, depth, 
                   intrinsic, 
                   extrinsic = np.eye(4), 
                   depth_unit = 1000,
                   filter_edge = False, window_s = 3, max_range = 0.2, 
                   return_uv = False,
                   filter_far = False, far_thres = 80,
                   filter_near = False, near_thres = 0.01):
        if depth.ndim>2:
            depth = depth[:,:,0]
        # get uvs
        h, w = depth.shape[0:2]
        u = np.arange(w)[None,:,None].repeat(h,axis=0)
        v = np.arange(h)[:,None,None].repeat(w,axis=1)
        uvd = np.concatenate([u, v, depth[:,:,None]],axis=-1)
        # depth
        # conduct mask
        if filter_edge: 
            kernel = np.ones((int(window_s*2+1),int(window_s*2+1)), dtype=np.uint16)
            nei_max = cv2.dilate(depth,kernel,iterations=1)
            nei_min =  cv2.erode(depth,kernel,iterations=1)
            nei_check = nei_max - nei_min
            uvd = uvd[nei_check<max_range]
        uvd = uvd.reshape(-1,3)
        depth = uvd[:,-1] / depth_unit
        if filter_far:
            uvd = uvd[depth<far_thres]
            depth = depth[depth<far_thres]
        if filter_near:
            uvd = uvd[depth>near_thres]
            depth = depth[depth>near_thres]
        pc = self.proj_2to3(uvd[:,0:2], depth, intrinsic, extrinsic, depth_unit=1)
        if return_uv:
            return uvd[:,0:2], depth, pc
        else:
            return pc

import torch.nn.functional as F
def nei_delta(input,pad=2):
    if not type(input) is torch.Tensor:
        input = torch.from_numpy(input.astype(np.float32))
    if len(input.shape) < 3:
        input = input[:,:,None]
    h,w,c = input.shape
    # reshape
    input = input.permute(2,0,1)[None]
    input = F.pad(input, pad=(pad,pad,pad,pad), mode='replicate')
    kernel = 2*pad + 1
    input = F.unfold(input,[kernel,kernel],padding=0)
    input = input.reshape(c,-1,h,w).permute(2,3,0,1).squeeze() # hw(3)*25
    return torch.amax(input,dim=-1),torch.amin(input,dim=-1),input

def edge_filter(metric_dpt,sky=None,times=0.1):
    sky = np.zeros_like(metric_dpt,bool) if sky is None else sky
    _max = np.percentile(metric_dpt[~sky],95)
    _min = np.percentile(metric_dpt[~sky], 5)
    _range = _max - _min
    nei_max,nei_min,_ = nei_delta(metric_dpt)
    delta = (nei_max-nei_min).numpy()
    edge = delta > times*_range
    return edge

def points_to_hpoints(points):
    n,_=points.shape
    return np.concatenate([points,np.ones([n,1])],1)

def hpoints_to_points(hpoints):
    return hpoints[:,:-1]/hpoints[:,-1:]

def transform_points(pts,transform):
    h,w=transform.shape
    if h==3 and w==3:
        return pts @ transform.T
    if h==3 and w==4:
        return pts @ transform[:,:3].T + transform[:,3:].T
    elif h==4 and w==4:
        return hpoints_to_points(points_to_hpoints(pts) @ transform.T)
    else: raise NotImplementedError

def random_rotation_matrix():
    """
    Generates a random 3D rotation matrix from axis and angle.

    Args:
        numpy_random_state: numpy random state object

    Returns:
        Random rotation matrix.
    """
    rng = np.random.RandomState()
    axis = rng.rand(3) - 0.5
    axis /= np.linalg.norm(axis) + 1E-8
    theta = np.pi * rng.uniform(0.0, 1.0)
    thetas=axis*theta
    alpha=thetas[0]
    beta=thetas[1]
    gama=thetas[2]
    Rzalpha=np.array([[np.cos(alpha),np.sin(alpha),0],
                      [-np.sin(alpha),np.cos(alpha),0],
                      [0,0,1]])

    Rybeta=np.array([[np.cos(beta),0,-np.sin(beta)],
                     [0,1,0],
                     [np.sin(beta),0,np.cos(beta)]])

    Rzgama=np.array([[np.cos(gama),np.sin(gama),0],
                      [-np.sin(gama),np.cos(gama),0],
                      [0,0,1]])
    R=np.matmul(Rzgama,np.matmul(Rybeta,Rzalpha))
    return R

def random_se3():
    T = np.eye(4)
    T[0:3,0:3] = random_rotation_matrix()
    t = np.random.rand(3)-0.5
    T[0:3,-1] = t*1000
    return T

def make_open3d_point_cloud(xyz, color=None):
    if not hasattr(xyz,'ndim'):
        return xyz
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd

def trans_gt_for_kitti(gt):
    r = gt[0:3,0:3]
    r = r[[0,2,1]]
    r = r[:,[0,2,1]]
    gt[0:3,0:3] = r
    t = gt[0:3,-1:]
    t[1]*=-1
    t = t[[2,1,0]]
    gt[0:3,-1:] = t 
    return gt

def single_kabsch(beforeT,afterT):
    center0=np.mean(afterT,0,keepdims=True)
    center1=np.mean(beforeT,0,keepdims=True)
    m = (beforeT-center1).T @ (afterT-center0)
    if np.linalg.det(m)<1e-5:
        return random_se3()
    U,S,VT = np.linalg.svd(m)
    rotation = VT.T @ U.T   #predicted RT
    offset = center0 - (center1 @ rotation.T)
    transform=np.concatenate([rotation,offset.T],1)
    out = np.eye(4)
    out[0:3] = transform
    return out 

def visual_pcd(xyz, color=None, normal = True):
    if hasattr(xyz,'ndim'):
        xyz_norm = np.mean(np.sqrt(np.sum(np.square(xyz),axis=1)))
        xyz = xyz / xyz_norm
        xyz = xyz.reshape(-1,3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
    else: pcd = xyz
    if color is not None:
        color = color.reshape(-1,3)
        pcd.colors = o3d.utility.Vector3dVector(color)
    if normal:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.2, 20))
    o3d.visualization.draw_geometries([pcd])
   
def visual_pcds(xyzs, normal = True):
    pcds = []
    for xyz in xyzs:
        if hasattr(xyz,'ndim'):
            # xyz_norm = np.mean(np.sqrt(np.sum(np.square(xyz),axis=1)))
            # xyz = xyz / xyz_norm
            xyz = xyz.reshape(-1,3)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.paint_uniform_color(np.random.rand(3))
        else: pcd = xyz
        if normal:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.2, 20))
        pcds.append(pcd)
    o3d.visualization.draw_geometries(pcds)
     