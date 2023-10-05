import cv2
import torch
import numpy as np
import open3d as o3d
from PIL import Image
from copy import deepcopy
import matplotlib.pyplot as plt
from utils.utils import transform_points,make_open3d_point_cloud, dpt_3d_convert

normR = 0.5
normN = 30
t = [3,0,0]
cmap = plt.cm.plasma

def visual_pcd(xyzs:list, normal = False):
  pcds = []
  for pcd in xyzs:
    color = np.random.rand(3)
    if hasattr(pcd,'ndim'):
        pcd = make_open3d_point_cloud(pcd)
    pcd.paint_uniform_color(color)
    if normal:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.3, 20))
    pcds.append(pcd)
  o3d.visualization.draw_geometries(pcds)

def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color, text, path=None,
                            show_keypoints=False, margin=10,
                            opencv_display=False, opencv_title='',
                            small_text=[]):
    H0, W0 = image0.shape[0:2]
    H1, W1 = image1.shape[0:2]
    H, W = max(H0, H1), W0 + W1 + margin
    out = 255*np.ones((H, W, 3), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0+margin:] = image1
    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)
    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)
    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)
    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_fg, 1, cv2.LINE_AA)
    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_fg, 1, cv2.LINE_AA)
    if path is not None:
        out = out.astype(np.uint8)
        out = Image.fromarray(out)
        out.save(path)
    return out

def dist2color(disp,inlier_thres):
    result = np.zeros([disp.shape[0],3])
    inlier = np.where(disp<=inlier_thres)[0]
    outlier = np.where(disp>inlier_thres)[0]
    result[inlier] = np.array([[0,255,0]]).repeat(len(inlier),axis=0)
    result[outlier] = np.array([[255,0,0]]).repeat(len(outlier),axis=0)
    return result

def draw_matches_2d_givenuv(image0, image1, uv0, uv1, kpts0, kpts1, matches,
                    inliers_only = True, inlier_thres = 0.3, gt = np.eye(4),
                    pth = None):
    uv0_m = uv0[matches[:,0]]
    uv1_m = uv1[matches[:,1]]
    kpts0_m = kpts0[matches[:,0]]
    kpts1_m = kpts1[matches[:,1]]
    # inlier check
    kpts0_m_gt = transform_points(kpts0_m, gt)
    disp = kpts0_m_gt - kpts1_m
    disp = np.sqrt(np.sum(disp**2, axis=1))
    inlier = disp<inlier_thres
    colors = dist2color(disp,inlier_thres)
    # reorganize outlier, inlier
    inlier_n = int(np.sum(inlier))
    reorganize_index = np.concatenate([np.where(~inlier)[0], np.where(inlier)[0]],axis=0)
    uv0_m = uv0_m[reorganize_index]
    uv1_m = uv1_m[reorganize_index]
    colors = colors[reorganize_index]
    if inliers_only:
        colors = colors[-inlier_n:]
        uv0_m = uv0_m[-inlier_n:]
        uv1_m = uv1_m[-inlier_n:]
    # visual
    if image1.ndim<3:
        image1 = image1[:,:,None].repeat(3,axis=-1).astype(np.float32)  
        image1 = dpt_normalize_visual(image1).astype(np.uint8)   
    return make_matching_plot_fast(image0, image1, uv0_m, uv1_m, uv0_m, uv1_m, color=colors, text='',path=pth)

def dpt_normalize_ct(depth):
    # following controlnet  1-depth
    depth = depth.astype(np.float64)
    vmin = np.percentile(depth, 2)
    vmax = np.percentile(depth, 85)
    depth -= vmin
    depth /= vmax - vmin
    depth = 1.0 - depth
    depth_image = (depth * 255.0).clip(0, 255).astype(np.uint8)
    return depth_image

def dpt_normalize_visual(dpt, max_range = None):
    if dpt.ndim>2:
        dpt = dpt[:,:,0]
    if max_range is not None:
        dpt[dpt>max_range] = np.max(dpt)
    v_max = np.max(dpt)
    v_min = np.min(dpt)
    v_max = np.max(dpt[dpt<v_max])
    v_min = np.min(dpt[dpt>v_min])
    output = (dpt - v_min) / (v_max - v_min)
    output = 1.0-output
    output = 255 * cmap(output)  # H, W, C
    output[dpt<0.01] = 255
    output = output[:,:,0:3]
    return output.astype('uint8')

class visualizor():
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.convertor = dpt_3d_convert()
        self.rgb_intrinsic = np.array(self.cfg.meta.rgb_intrinsic)
        self.dpt_intrinsic = np.array(self.cfg.meta.dpt_intrinsic)
        self.target_w, self.target_h = self.cfg.meta.dpt_size

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
        return uv, gt_xyz, mask  

    def pps2xyz(self, item0, item1, pps):
        # mask out points without ground-truth depth//for visual evaluation
        suv, sxyz, smask = self.eval_mask(ftype='rgb',item=item0)
        tuv, txyz, tmask = self.eval_mask(ftype='dpt',item=item1)
        pmask = (smask[pps[:,0]]) & (tmask[pps[:,1]])
        pps = pps[pmask]
        return suv, tuv, sxyz, txyz, pps

    def resize(self, img, uv):
        h,w = img.shape[0:2]
        time_h, time_w = self.target_h/h, self.target_w/w
        img = cv2.resize(img,(self.target_w,self.target_h),interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.uint8)
        uv[:,0] = uv[:,0]*time_w
        uv[:,1] = uv[:,1]*time_h
        uv = uv.astype(np.int32)
        return img, uv

    def draw_demo(self, pair, save_pth = './demo.png'):
        base_fn = pair['to_fn']
        match_result = np.load(base_fn)
        pps = match_result['pps']
        gt = match_result['gt']

        loc = str.rfind(base_fn,'/')
        feat_dir = base_fn[:loc][:-5] + f'feat/'
        sid, tid = pair['q_id'], pair['d_id']
        sitem = torch.load(f'{feat_dir}/{sid}.feat.pth')
        titem = torch.load(f'{feat_dir}/{tid}.feat.pth')

        simg = np.array(Image.open(sitem['rgb_fn']))
        dpt_fn = titem['proj_dpt_fn']
        timg = np.array(Image.open(dpt_fn))[:,:,None].repeat(3,axis=-1).astype(np.float32)  
        timg = dpt_normalize_visual(timg).astype(np.float32)
        
        suv, tuv, sxyz, txyz, pps = self.pps2xyz(sitem, titem, pps)
        simg, suv = self.resize(simg, suv)
        timg, tuv = self.resize(timg, tuv)
        gt = pair['gt']
        img = draw_matches_2d_givenuv(simg, timg, suv, tuv, sxyz, txyz, pps, inliers_only=False, 
                                inlier_thres=self.cfg.eval.ir.thres, gt = gt, pth=save_pth)