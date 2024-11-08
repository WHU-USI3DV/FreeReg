"""
Dataset class for organizing datasets with:
Pointcloud + Pointcloud_o3d
Keypointindex + Keypoint
PCpairs + pairgt


Train dataloader of PartI and PartII. 
"""


import os
import numpy as np
import abc

import torch
from torch.utils.data import Dataset
from utils.r_eval import compute_R_diff,quaternion_from_matrix
from utils.utils import read_pickle, make_non_exists_dir
import open3d as o3d

class EvalDataset(abc.ABC):
    @abc.abstractmethod
    def get_pair_ids(self):
        pass

    @abc.abstractmethod
    def get_cloud_ids(self):
        pass

    @abc.abstractmethod
    def get_pc_dir(self,cloud_id):
        pass
    
    @abc.abstractmethod
    def get_key_dir(self,cloud_id):
        pass

    @abc.abstractmethod
    def get_transform(self,id0,id1):
        # note the order!
        # target: id0, source: id1
        # R @ pts1 + t = pts0
        pass

    @abc.abstractmethod
    def get_name(self):
        pass

    @abc.abstractmethod
    def get_kps(self,cloud_id):
        pass

#The dataset class for original/ground truth datas
class ThrDMatchPartDataset(EvalDataset):
    def __init__(self,root_dir,stationnum,gt_dir=None):
        self.root=root_dir
        if gt_dir==None:
            self.gt_dir=f'{self.root}/PointCloud/gt.log'
        else:
            self.gt_dir=gt_dir
        self.kps_pc_fn=[f'{self.root}/Keypoints_PC/cloud_bin_{k}Keypoints.npy' for k in range(stationnum)]
        self.kps_fn=[f'{self.root}/Keypoints/cloud_bin_{k}Keypoints.txt' for k in range(stationnum)]
        self.pc_ply_paths=[f'{self.root}/PointCloud/cloud_bin_{k}.ply' for k in range(stationnum)]
        self.pc_txt_paths=[f'{self.root}/PointCloud/cloud_bin_{k}.txt' for k in range(stationnum)]
        self.pair_id2transform=self.parse_gt_fn(self.gt_dir)
        self.pair_ids=[tuple(v.split('-')) for v in self.pair_id2transform.keys()]
        self.pc_ids=[str(k) for k in range(stationnum)]
        self.pair_num=self.get_pair_nums()
        self.name='3dmatch/kitchen'

    #function for gt(input: gt.log)
    @staticmethod
    def parse_gt_fn(fn):
        with open(fn,'r') as f:
            lines=f.readlines()
            pair_num=len(lines)//5
            pair_id2transform={}
            for k in range(pair_num):
                id0,id1=np.fromstring(lines[k*5],dtype=np.float32,sep='\t')[0:2]
                id0=int(id0)
                id1=int(id1)
                row0=np.fromstring(lines[k*5+1],dtype=np.float32,sep=' ')
                row1=np.fromstring(lines[k*5+2],dtype=np.float32,sep=' ')
                row2=np.fromstring(lines[k*5+3],dtype=np.float32,sep=' ')
                transform=np.stack([row0,row1,row2],0)
                pair_id2transform['-'.join((str(id0),str(id1)))]=transform

            return pair_id2transform

    def get_pair_ids(self):
        return self.pair_ids

    def get_pair_nums(self):
        return len(self.pair_ids)

    def get_cloud_ids(self):
        return self.pc_ids

    def get_pc_dir(self,cloud_id):
        return self.pc_ply_paths[int(cloud_id)]

    def get_pc(self,pc_id):
        if os.path.exists(self.pc_ply_paths[int(pc_id)]):
            pc=o3d.io.read_point_cloud(self.pc_ply_paths[int(pc_id)])
            return np.array(pc.points)
        else:
            pc=np.loadtxt(self.pc_paths[int(pc_id)],delimiter=',')
            return pc
    
    def get_pc_o3d(self,pc_id):
        return o3d.io.read_point_cloud(self.pc_ply_paths[int(pc_id)])
            
    def get_key_dir(self,cloud_id):
        return self.kps_fn[int(cloud_id)]

    def get_transform(self, id0, id1):
        return self.pair_id2transform['-'.join((id0,id1))]

    def get_name(self):
        return self.name

    def get_kps(self, cloud_id):
        if not os.path.exists(self.kps_pc_fn[int(cloud_id)]):
            pc=self.get_pc(cloud_id)
            key_idxs=np.loadtxt(self.kps_fn[int(cloud_id)]).astype(np.int)
            keys=pc[key_idxs]
            make_non_exists_dir(f'{self.root}/Keypoints_PC')
            np.save(self.kps_pc_fn[int(cloud_id)],keys)
            return keys
        return np.load(self.kps_pc_fn[int(cloud_id)])

#Get dataset items with the dataset name(output: dict)
def get_dataset_name(dataset_name,origin_data_dir):

    if dataset_name=='demo':
        datasets={}
        datasets['wholesetname']=f'{dataset_name}'
        scenes=['kitchen']
        stationnums=[2]
        for i in range(len(scenes)):
            root_dir=f'{origin_data_dir}/{dataset_name}/'+scenes[i]
            datasets[scenes[i]]=ThrDMatchPartDataset(root_dir,stationnums[i])
            datasets[scenes[i]].name=f'{dataset_name}/{scenes[i]}'
        return datasets


    if dataset_name=='3dmatch':
        datasets={}
        datasets['wholesetname']=f'{dataset_name}'
        scenes=["kitchen","sun3d-home_at-home_at_scan1_2013_jan_1",
        "sun3d-home_md-home_md_scan9_2012_sep_30","sun3d-hotel_uc-scan3",
        "sun3d-hotel_umd-maryland_hotel1","sun3d-hotel_umd-maryland_hotel3",
        "sun3d-mit_76_studyroom-76-1studyroom2","sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika"]
        stationnums=[60,60,60,55,57,37,66,38]

        for i in range(len(scenes)):
            root_dir=f'{origin_data_dir}/{dataset_name}/'+scenes[i]
            datasets[scenes[i]]=ThrDMatchPartDataset(root_dir,stationnums[i])
            datasets[scenes[i]].name=f'{dataset_name}/{scenes[i]}'
        return datasets


    if dataset_name=='3dLomatch':
        datasets={}
        datasets['wholesetname']=f'{dataset_name}'
        scenes=["kitchen","sun3d-home_at-home_at_scan1_2013_jan_1",
        "sun3d-home_md-home_md_scan9_2012_sep_30","sun3d-hotel_uc-scan3",
        "sun3d-hotel_umd-maryland_hotel1","sun3d-hotel_umd-maryland_hotel3",
        "sun3d-mit_76_studyroom-76-1studyroom2","sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika"]
        stationnums=[60,60,60,55,57,37,66,38]
        for i in range(len(scenes)):
            root_dir=f'{origin_data_dir}/3dmatch/'+scenes[i]
            gt_dir=f'{root_dir}/PointCloud/gtLo.log'
            datasets[scenes[i]]=ThrDMatchPartDataset(root_dir,stationnums[i],gt_dir)
            datasets[scenes[i]].name=f'{dataset_name}/{scenes[i]}'
        return datasets


    if dataset_name=='ETH':
        datasets={}
        datasets['wholesetname']=f'{dataset_name}'
        scenes=['gazebo_summer','gazebo_winter','wood_autumn','wood_summer']
        stationnums=[32,31,32,37]
        for i in range(len(scenes)):
            root_dir=f'{origin_data_dir}/{dataset_name}/'+scenes[i]
            datasets[scenes[i]]=ThrDMatchPartDataset(root_dir,stationnums[i])
            datasets[scenes[i]].name=f'{dataset_name}/{scenes[i]}'
        return datasets


    if dataset_name=='WHU-TLS':
        datasets={}
        datasets['wholesetname']=f'{dataset_name}'
        scenes=['Park','Mountain','Campus','RiverBank','UndergroundExcavation','Tunnel']
        stationnums=[32,6,10,7,12,7]
        for i in range(len(scenes)):
            root_dir=f'{origin_data_dir}/{dataset_name}/'+scenes[i]
            datasets[scenes[i]]=ThrDMatchPartDataset(root_dir,stationnums[i])
            datasets[scenes[i]].name=f'{dataset_name}/{scenes[i]}'
        return datasets


    if dataset_name=='3dmatch_train':
        datasets={}
        datasets['wholesetname']=f'{dataset_name}'
        datasets['valscenes']=['sun3d-brown_bm_4-brown_bm_4','sun3d-harvard_c11-hv_c11_2','7-scenes-heads','rgbd-scenes-v2-scene_10','bundlefusion-office0','analysis-by-synthesis-apt2-kitchen']
        scenes=['bundlefusion-apt0', 'rgbd-scenes-v2-scene_02', 'bundlefusion-office1', 'sun3d-brown_cogsci_1-brown_cogsci_1', 'rgbd-scenes-v2-scene_06', 'analysis-by-synthesis-apt2-kitchen', 'rgbd-scenes-v2-scene_03', 'bundlefusion-apt1', 'sun3d-harvard_c8-hv_c8_3', 'bundlefusion-copyroom', 'sun3d-home_bksh-home_bksh_oct_30_2012_scan2_erika', 'rgbd-scenes-v2-scene_04', '7-scenes-pumpkin', 'rgbd-scenes-v2-scene_01', 'analysis-by-synthesis-office2-5a', 'sun3d-brown_bm_1-brown_bm_1', 'bundlefusion-apt2', 'sun3d-brown_cs_2-brown_cs2', 'bundlefusion-office2', 'sun3d-hotel_sf-scan1', 'sun3d-hotel_nips2012-nips_4', 'bundlefusion-office3', 'rgbd-scenes-v2-scene_09', 'rgbd-scenes-v2-scene_05', 'rgbd-scenes-v2-scene_07', '7-scenes-heads', 'sun3d-harvard_c3-hv_c3_1', 'rgbd-scenes-v2-scene_08', 'sun3d-mit_76_417-76-417b', 'sun3d-mit_32_d507-d507_2', 'sun3d-mit_46_ted_lab1-ted_lab_2', '7-scenes-chess', 'rgbd-scenes-v2-scene_10', 'sun3d-harvard_c11-hv_c11_2', 'analysis-by-synthesis-apt2-living', 'sun3d-mit_w20_athena-sc_athena_oct_29_2012_scan1_erika', 'analysis-by-synthesis-apt1-living', 'analysis-by-synthesis-apt1-kitchen', 'sun3d-mit_dorm_next_sj-dorm_next_sj_oct_30_2012_scan1_erika', '7-scenes-stairs', 'sun3d-brown_bm_4-brown_bm_4', 'bundlefusion-office0', 'sun3d-harvard_c6-hv_c6_1', 'rgbd-scenes-v2-scene_14', 'rgbd-scenes-v2-scene_12', 'analysis-by-synthesis-office2-5b', 'analysis-by-synthesis-apt2-luke', '7-scenes-office', 'sun3d-harvard_c5-hv_c5_1', 'sun3d-brown_cs_3-brown_cs3', '7-scenes-fire', 'rgbd-scenes-v2-scene_11', 'analysis-by-synthesis-apt2-bed', 'rgbd-scenes-v2-scene_13']
        stationnums=[85, 8, 57, 28, 10, 9, 8, 84, 10, 44, 96, 8, 54, 8, 14, 65, 38, 52, 34, 92, 62, 37, 7, 11, 9, 18, 19, 9, 77, 54, 75, 54, 7, 8, 10, 70, 15, 11, 26, 24, 32, 60, 15, 6, 7, 17, 19, 90, 20, 34, 36, 6, 10, 4]
        for i in range(len(scenes)):
            root_dir=f'{origin_data_dir}/{dataset_name}/'+scenes[i]
            datasets[scenes[i]]=ThrDMatchPartDataset(root_dir,stationnums[i])
            datasets[scenes[i]].name=f'{dataset_name}/{scenes[i]}'
        return datasets

    else:
        raise NotImplementedError


def get_dataset(cfg,training=True):
    if training:
        dataset_name=cfg.trainset_name
    else:
        dataset_name=cfg.testset_name
    origin_dir=cfg.origin_data_dir
    return get_dataset_name(dataset_name,origin_dir)
    

#train dataset 
class Enhanced_train_dataset_PartI(Dataset):
    def __init__(self,cfg,is_training=True):
        self.cfg=cfg
        self.output_dir=self.cfg.output_cache_fn
        self.is_training=is_training
        self.Rgroup=np.load(f'{self.cfg.SO3_related_files}/Rotation.npy').astype(np.float32)
        if self.is_training:
            self.name_pair_ids=read_pickle(cfg.train_pcpair_list_fn) #list: name id0 id1 pt1 pt2
        else:
            self.name_pair_ids=read_pickle(cfg.val_pppair_list_fn)[0:3000]   #list: name id0 id1 pt1 pt2

    def R2DR_id(self,R):
        min_diff=180
        best_id=0
        for R_id in range(self.Rgroup.shape[0]):
            R_diff=compute_R_diff(self.Rgroup[R_id],R)
            if R_diff<min_diff:
                min_diff=R_diff
                best_id=R_id
        return best_id

    def __getitem__(self,index):
        if self.is_training:
            item=torch.load(f'{self.output_dir}/Train_val_list/trainset/{index}.pth')
            return item
        
        else:
            item=torch.load(f'{self.cfg.output_cache_fn}/Train_val_list/valset/{index}.pth')
            return item
        

    def __len__(self):
        return len(self.name_pair_ids)


class Enhanced_train_dataset_PartII(Dataset):
    def __init__(self,cfg,is_training=True):
        self.cfg=cfg
        self.output_dir=self.cfg.output_cache_fn
        self.is_training=is_training
        self.Rgroup=np.load(f'{self.cfg.SO3_related_files}/Rotation.npy').astype(np.float32)
        if self.is_training:
            self.name_pair_ids=read_pickle(cfg.train_pcpair_list_fn) #list: name id0 id1 pt1 pt2
        else:
            self.name_pair_ids=read_pickle(cfg.val_pppair_list_fn)[0:3000]   #list: name id0 id1 pt1 pt2

    def R2DR_id(self,R):
        min_diff=180
        best_id=0
        for R_id in range(self.Rgroup.shape[0]):
            R_diff=compute_R_diff(self.Rgroup[R_id],R)
            if R_diff<min_diff:
                min_diff=R_diff
                best_id=R_id
        return best_id
    
    def DeltaR(self,R,index):
        R_anchor=self.Rgroup[index]#3*3
        #R=Rres@Ranc->Rres=R@Ranc.T
        deltaR=R@R_anchor.T
        return quaternion_from_matrix(deltaR)

    def __getitem__(self,index):
        if self.is_training:
            item=torch.load(f'{self.output_dir}/Train_val_list/trainset/{index}.pth')
            return item
        
        else:
            item=torch.load(f'{self.cfg.output_cache_fn}/Train_val_list/valset/{index}.pth')
            deltaR=self.DeltaR(item['R'].numpy(),int(item['true_idx']))
            item['deltaR']=torch.from_numpy(deltaR.astype(np.float32))
            return item
        

    def __len__(self):
        return len(self.name_pair_ids)

        
name2traindataset={
    "Enhanced_train_dataset_PartI":Enhanced_train_dataset_PartI,
    "Enhanced_train_dataset_PartII":Enhanced_train_dataset_PartII
}

