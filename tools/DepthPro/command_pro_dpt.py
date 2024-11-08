import depth_pro
import numpy as np
from PIL import Image
from depth_pro.depth_pro import DepthProConfig

class apple_pro_depth():
    def __init__(self,device='cuda',ckpt = '/mnt/proj/SOTAs/ml-depth-pro-main/checkpoints/depth_pro.pt'):
        self.ckpt = ckpt
        self.device = device
        self._load_model()
        
    def _load_model(self):
        cfg = DepthProConfig(
            patch_encoder_preset="dinov2l16_384",
            image_encoder_preset="dinov2l16_384",
            checkpoint_uri=self.ckpt,
            decoder_features=256,
            use_fov_head=True,
            fov_encoder_preset="dinov2l16_384",
        )
        self.model, self.transform = depth_pro.create_model_and_transforms(config=cfg,device=self.device)
        self.model.eval()
        
    def get_intrins(self, f, H, W):
        new_cu = (W / 2.0) - 0.5
        new_cv = (H / 2.0) - 0.5
        intrins = np.array([
            [f,         0,     new_cu  ],
            [0,         f,     new_cv  ],
            [0,         0,     1       ]
        ])
        return intrins
    
    def to(self,device):
        self.device = device
        self.model.to(device)
    
    def __call__(self, image,f_px=None):
        if type(image) is np.ndarray:
            if np.amax(image) < 1.1:
                image = image*255
            image = Image.fromarray(image.astype(np.uint8))
        # trans
        image = self.transform(image).to(self.device)
        # predict
        prediction = self.model.infer(image, f_px=f_px)
        depth = prediction["depth"]  # Depth in [m].
        focallength_px = prediction["focallength_px"]  # Focal length in pixels.
        # output
        H,W = depth.shape[0:2]
        depth = depth.detach().cpu().numpy()
        focallength_px = focallength_px.detach().cpu().numpy() if f_px is None else f_px
        intrisnc = self.get_intrins(focallength_px,H,W)
        return depth, False, False, intrisnc
