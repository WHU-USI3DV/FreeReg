"""
generate FCGF features
conduct mutmul feature matching
"""
import torch
import numpy as np
import MinkowskiEngine as ME
from tools.fcgf.unet import load_model


class fcgf_extractor():
    def __init__(self, pth = '', vs = 0.025):
        self.pth_fn = pth
        self.voxel_size = vs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        checkpoint = torch.load(self.pth_fn)
        config = checkpoint['config']
        num_feats = 1
        Model = load_model(config.model)
        self.model = Model(
            num_feats,
            config.model_n_out,
            bn_momentum=0.05,
            normalize_feature=config.normalize_feature,
            conv1_kernel_size=config.conv1_kernel_size,
            D=3)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def extract_features(self,pc,numpy=True):
        coords = np.floor(pc / self.voxel_size)
        coords = torch.from_numpy(coords.astype(np.int32)+0.0)
        coords = coords.contiguous()
        _, sel = ME.utils.sparse_quantize(coords, return_index=True)
        # Get coords
        xyz = pc[sel]
        feats=np.ones((xyz.shape[0], 1))
        coords = np.floor(xyz / self.voxel_size)
        coords_me, feats_me = ME.utils.sparse_collate([coords], [feats])
        sinput0 = ME.SparseTensor(
                feats_me.float().to(self.device),
                coordinates=coords_me.to(self.device))
        F = self.model(sinput0).F
        F = F.detach()
        F = F / torch.norm(F,dim=-1,keepdim=True)
        if numpy:
            return sel, F.cpu().numpy()
        else:
            return sel, F.cpu()

    def run(self, pc, numpy = True):
        # get features. inds is the indexes in the input pc (indexes of down-sampled keypoints)
        inds, feat = self.extract_features(pc, numpy = numpy)
        # feat l2 normalization
        return inds, feat