import os
import time
import torch
import numpy as np
import open3d as o3d
from PIL import Image

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
    
class nn_match():
    def __init__(self, nn_max_n = 500) -> None:
        self.nn_max_n = nn_max_n

    def pdist(self, A, B, dist_type='L2'):
          if dist_type == 'L2':
              D2 = torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
              return torch.sqrt(D2 + 1e-7)
          elif dist_type == 'SquareL2':
              return torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
          else:
              raise NotImplementedError('Not implemented')

    def find_nn_gpu(self, F0, F1, nn_max_n=-1, return_distance=False, dist_type='SquareL2'):
        # Too much memory if F0 or F1 large. Divide the F0
        if nn_max_n > 1:
            N = len(F0)
            C = int(np.ceil(N / nn_max_n))
            stride = nn_max_n
            dists, inds = [], []
            for i in range(C):
                dist = self.pdist(F0[i * stride:(i + 1) * stride], F1, dist_type=dist_type)
                min_dist, ind = dist.min(dim=1)
                dists.append(min_dist.detach().unsqueeze(1).cpu())
                inds.append(ind.cpu())

            if C * stride < N:
                dist = self.pdist(F0[C * stride:], F1, dist_type=dist_type)
                min_dist, ind = dist.min(dim=1)
                dists.append(min_dist.detach().unsqueeze(1).cpu())
                inds.append(ind.cpu())

            dists = torch.cat(dists)
            inds = torch.cat(inds)
            assert len(inds) == N
        else:
            dist = self.pdist(F0, F1, dist_type=dist_type)
            min_dist, inds = dist.min(dim=1)
            dists = min_dist.detach().unsqueeze(1).cpu()
            inds = inds.cpu()
        if return_distance:
            return inds, dists
        else:
            return inds
        
    def find_knn_gpu(self, F0, F1, nn_max_n=-1, k=2, return_distance=False, dist_type='SquareL2'):
      # Too much memory if F0 or F1 large. Divide the F0
      if nn_max_n > 1:
          N = len(F0)
          C = int(np.ceil(N / nn_max_n))
          stride = nn_max_n
          dists, inds = [], []
          for i in range(C):
              dist = self.pdist(F0[i * stride:(i + 1) * stride], F1, dist_type=dist_type)
              min_dist, ind = torch.topk(-dist, k, dim=1)
              dists.append(-min_dist.detach().unsqueeze(1).cpu())
              inds.append(ind.cpu())

          if C * stride < N:
              dist = self.pdist(F0[C * stride:], F1, dist_type=dist_type)
              min_dist, ind = torch.topk(-dist, k, dim=1)
              dists.append(-min_dist.detach().unsqueeze(1).cpu())
              inds.append(ind.cpu())

          dists = torch.cat(dists,dim=0)
          inds = torch.cat(inds,dim=0)
          assert len(inds) == N
      else:
          dist = self.pdist(F0, F1, dist_type=dist_type)
          min_dist, inds = torch.topk(-dist, k, dim=1)
          dists = -min_dist.detach().unsqueeze(1).cpu()
          inds = inds.cpu()
      if return_distance:
          return inds, dists
      else:
          return inds

    def find_corr(self, F0, F1, subsample_size=-1, mutual = True):
        #init
        inds0, inds1 = np.arange(F0.shape[0]), np.arange(F1.shape[0])
        if subsample_size > 0:
            N0 = min(len(F0), subsample_size)
            N1 = min(len(F1), subsample_size)
            inds0 = np.random.choice(len(F0), N0, replace=False)
            inds1 = np.random.choice(len(F1), N1, replace=False)
            F0, F1 = F0[inds0], F1[inds1]
        # Compute the nn
        nn_inds_in1 = self.find_nn_gpu(F0, F1, nn_max_n=self.nn_max_n)
        if not mutual:
          inds1 = inds1[nn_inds_in1]
        else:
          matches = []
          nn_inds_in0 = self.find_nn_gpu(F1, F0, nn_max_n=self.nn_max_n)
          for i in range(len(nn_inds_in1)):
              if i == nn_inds_in0[nn_inds_in1[i]]:
                matches.append((i, nn_inds_in1[i]))
          matches = np.array(matches).astype(np.int32)
          inds0 = inds0[matches[:,0]]
          inds1 = inds1[matches[:,1]]
        return inds0, inds1


class dpt_3d_convert():
    def __init__(self):
        pass

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
    
    def proj_depth(self, depth, intrinsic, extrinsic = np.eye(4), depth_unit = 1000,
                   filter_edge = False, window_s = 3, max_range = 0.2, 
                   return_uv = False,
                   filter_far = False, far_thres = 80,
                   filter_near = False, near_thres = 0.01):
        if depth.ndim>2:
            depth = depth[:,:,0]
        h, w = depth.shape[0:2]
        u = np.arange(w)[None,:,None].repeat(h,axis=0)
        v = np.arange(h)[:,None,None].repeat(w,axis=1)
        uvd = np.concatenate([u, v, depth[:,:,None]],axis=-1)
        # condeuct mask
        if filter_edge:
            mask = np.zeros_like(depth)
            for i in range(window_s, h):
                for j in range(window_s, w):
                    check = depth[(i-window_s):(i+window_s), (j-window_s):(j+window_s)] / depth_unit
                    check = np.max(check) - np.min(check)
                    if check < max_range:
                        mask[i,j] = 1
            uvd = uvd[mask>0.5]
        uvd = uvd.reshape(-1,3)
        if filter_far:
            uvd = uvd[uvd[:,-1]<far_thres*depth_unit]
        if filter_near:
            uvd = uvd[uvd[:,-1]>near_thres*depth_unit]
        pc = self.proj_2to3(uvd[:,0:2], uvd[:,-1], intrinsic, extrinsic, depth_unit)
        if return_uv:
            return uvd[:,0:2], uvd[:,-1], pc
        else:
            return pc
    
    def proj_pc2dpt(self, ply, extrinsic, intrinsic, h, w):
        if type(ply) is not np.ndarray:
            ply = np.array(ply.points)
        uv, dpt = self.proj_3to2(ply, intrinsic, extrinsic)
        mask_w = (uv[:,0]<w) & (uv[:,0]>=0)
        mask_h = (uv[:,1]<h) & (uv[:,1]>=0)
        # mask mask off the back-project points
        mask_d = dpt>0.05
        mask = mask_h & mask_w & mask_d
        uv = uv[mask].astype(np.int32)
        dpt = dpt[mask]
        result = np.ones([h,w])*10000
        for i in range(uv.shape[0]):
            u,v = uv[i]
            d = dpt[i]
            result[v,u] = min(result[v,u],d)
        result[result>9999] = 0.0
        return result

class suppress_stdout_stderr(object):
    '''
    Avoid terminal output of diffusion processings!
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])

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
            pcd.color = o3d.utility.Vector3dVector(color)
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

def save_depth(dpt_fn,dpt,scale=1):
    dpt = dpt * scale
    dpt = dpt.astype(np.uint16)
    dpt = Image.fromarray(dpt)
    dpt.save(dpt_fn)