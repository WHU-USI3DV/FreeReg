import torch
import argparse
import numpy as np
from copy import deepcopy
from .utils.network import PartI_test
from .fcgf_feat import fcgf_extractor
from .utils.utils import transform_points

group_file = 'tools/YOHO/utils/group_related'

# fake config
parser = argparse.ArgumentParser()
parser.add_argument('--SO3_related_files',default=group_file,type=str)
args = parser.parse_args()

class nn_match():
    def __init__(self) -> None:
        pass

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
        
    def find_corr(self, F0, F1, subsample_size=-1, mutual = True, nn_max_n = 500):
        #init
        inds0, inds1 = np.arange(F0.shape[0]), np.arange(F1.shape[0])
        if subsample_size > 0:
            N0 = min(len(F0), subsample_size)
            N1 = min(len(F1), subsample_size)
            inds0 = np.random.choice(len(F0), N0, replace=False)
            inds1 = np.random.choice(len(F1), N1, replace=False)
            F0 = F0[inds0]
            F1 = F1[inds1]
        # Compute the nn
        nn_inds_in1 = self.find_nn_gpu(F0, F1, nn_max_n=nn_max_n)
        if not mutual:
          inds1 = inds1[nn_inds_in1]
        else:
          matches = []
          nn_inds_in0 = self.find_nn_gpu(F1, F0, nn_max_n=nn_max_n)
          for i in range(len(nn_inds_in1)):
              if i == nn_inds_in0[nn_inds_in1[i]]:
                matches.append((i, nn_inds_in1[i]))
          matches = np.array(matches).astype(np.int32)
          inds0 = inds0[matches[:,0]]
          inds1 = inds1[matches[:,1]]
        return inds0, inds1
    
class yoho_extractor():
    def __init__(self,
                 fcgf_ckpt = 'tools/YOHO/model/Backbone/best_val_checkpoint.pth',
                 yoho_ckpt = 'tools/YOHO/model/PartI_train/model_best.pth',
                 device = 'cuda'):
        # basic
        self.device = device
        self.grs = np.load(f'{group_file}/Rotation.npy')
        self.fcgf = fcgf_extractor(pth=fcgf_ckpt, device = device)
        self.yoho_ckpt = yoho_ckpt
        self.network = PartI_test(args)
        self._load_model()
        self.nn_searcher = nn_match()
        self.bs = 500

    def _load_model(self):
        pth = torch.load(self.yoho_ckpt)['network_state_dict']
        self.network.load_state_dict(pth,strict=True)
        self.network.to(self.device).eval()
        
    def _feature_transfer_xyz(self, query, source, source_f):
        # nn
        query = torch.from_numpy(query.astype(np.float32)).to(self.device)
        source = torch.from_numpy(source.astype(np.float32)).to(self.device)
        idx,dist = self.nn_searcher.find_nn_gpu(query, source, nn_max_n=1000, return_distance=True, dist_type='SquareL2')
        qf = source_f[idx]
        return qf

    def extract_features(self,pc,voxel_size,nkpts=5000):
        kpts_index = np.random.permutation(len(pc))[0:nkpts]
        kpts = pc[kpts_index]
        kpts_f = []
        # pc rotation + feature extraction
        for i in range(self.grs.shape[0]):
            kptsi = transform_points(deepcopy(kpts),self.grs[i])
            pci = transform_points(deepcopy(pc),self.grs[i])
            # fcgf extraction
            pci_ds,pci_f = self.fcgf.run(pci,voxel_size)
            kptsi_f = self._feature_transfer_xyz(kptsi,pci_ds,pci_f)
            kpts_f.append(kptsi_f[:,:,None])
        kpts_f = torch.concat(kpts_f,dim=-1).to(self.device)
        yoho_inv,yoho_eqv = [],[]
        nbatch,nlast = len(kpts_f)//self.bs, len(kpts_f)%self.bs
        with torch.no_grad():
            if nlast<2:
                nbatch -= 1
            for i in range(nbatch):
                batch = kpts_f[self.bs*i:self.bs*(i+1)]
                output = self.network(batch)
                yoho_inv.append(output['inv'])
                yoho_eqv.append(output['eqv'])
            batch = kpts_f[self.bs*nbatch:]
            output = self.network(batch)
            yoho_inv.append(output['inv'])       
            yoho_eqv.append(output['eqv'])       
            # output: 5000*32; 5000*32*60
            yoho_inv = torch.cat(yoho_inv,dim=0).cpu()
            yoho_eqv = torch.cat(yoho_eqv,dim=0).cpu()
        return kpts,yoho_inv,yoho_eqv

    def run(self, pc, voxel_size = 0.025, nkpts=5000):
        # get features. inds is the indexes in the input pc (indexes of down-sampled keypoints)
        kpts, feat_inv, feat_eqv = self.extract_features(pc, voxel_size, nkpts=nkpts)
        # feat l2 normalization
        return kpts, feat_inv, feat_eqv
