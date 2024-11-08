"""
Common functions using open3d in the whole pipeline.
"""

import open3d as o3d
import copy

def read_ply(road):
    return o3d.io.read_point_cloud(road)

def make_open3d_point_cloud(xyz, color=None):
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(xyz)
  if color is not None:
    pcd.colors = o3d.utility.Vector3dVector(color)
  return pcd

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def draw_registration_result_with_keys(source, target, keys, transformation):
    keys_pc = o3d.geometry.PointCloud()
    keys_pc.points = o3d.utility.Vector3dVector(keys)
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    keys_pc.paint_uniform_color([1,0,0])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([keys_pc,source_temp, target_temp])
    