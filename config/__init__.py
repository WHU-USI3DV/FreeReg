from omegaconf import OmegaConf

def gen_config(cfg_path):
    return OmegaConf.load(cfg_path)