import numpy as np
from tools.zoe.meta_dpt import depth_generator

class pipeline_zoe():
    def __init__(self, cfg, update_dpt=False):
        self.cfg = cfg
        self.dg = depth_generator(
            model_type=cfg.zoe.zoe_type,
            model_fn=cfg.zoe.zoe_ck,
            update_dpt=update_dpt,
            base=cfg.meta.base,
            config_version=cfg.zoe.config_version,
            rgb_instrinsic=np.array(cfg.meta.rgb_intrinsic),
            gen_ply=False)
    
    def run_single_rgb(self, rgb):
        return self.dg.processor.depthor.run(rgb)
    
    def run(self, metas):
        self.dg.run(metas)