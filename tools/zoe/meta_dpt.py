import os
import numpy as np
from glob import glob
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm
from tools.zoe.gen_dpt import img2dpt
from utils.utils import Timer, dpt_3d_convert

timer = Timer()

class single_dpt():
    def __init__(self,
                 rgb_instrinsic,
                 model_type,
                 model_fn,
                 config_version=None,
                 ):
        self.depthor = img2dpt(model_type,model_fn,config_version,rgb_instrinsic)

    def run(self, rgb):
        # rgb(numpy) -> dpt
        dpt = self.depthor.run(rgb)
        return dpt
    
class meta_dpt():
    def __init__(self,
                 # model for pth generation
                 model_type,
                 model_fn,
                 config_version = None,
                 # projector
                 rgb_instrinsic = np.array([[585,0,320],[0,585,240],[0,0,1]]),
                 update_dpt = False):
        self.rgb_instrinsic = rgb_instrinsic
        self.depthor = single_dpt(rgb_instrinsic, 
                                  model_type, 
                                  model_fn,
                                  config_version)
        self.update_dpt = update_dpt
    
    def run_dpts(self, meta):
        # process querys
        for _, q_item in tqdm(meta['frames'].items()):
            # to save fns
            if not ('rgb_fn' in q_item):continue
            if not ('zoe_fn' in q_item):raise TypeError('identify a zoe fn!')
            to_fn = q_item['zoe_fn'] 
            # judge if exists
            if (not self.update_dpt) and os.path.exists(to_fn):continue
            basic = to_fn[:str.rfind(to_fn,'/')]
            if not os.path.exists(basic): os.makedirs(basic)
            query = Image.open(q_item['rgb_fn'])
            query = np.asarray(query, dtype=np.float32) / 255.0
            timer.tic()
            dpt = self.depthor.run(query)
            timer.toc()
            # save
            np.save(to_fn, dpt)

     
class depth_generator():
    def __init__(self,
                 # for depth
                 model_type = 'zoedepth', 
                 config_version = None,
                 model_fn = '/mnt/proj/SOTAs/ZoeDepth-main/checkpoints/ZoeD_M12_N.pt',
                 update_dpt = False,
                 # for eval
                 base = f'/mnt/proj/Datasets/3DMatch/data/3dmatch_test',
                # projector
                 rgb_instrinsic = np.array([[585,0,320],[0,585,240],[0,0,1]]),
                # ply
                 gen_ply = True
                ):
        # base dir
        self.base = base
        self.rgb_instrinsic = rgb_instrinsic
        self.update_dpt = update_dpt
        # runner
        self.processor = meta_dpt(model_type,
                                   model_fn,
                                   config_version = config_version,
                                   rgb_instrinsic = rgb_instrinsic,
                                   update_dpt = update_dpt)
        self.gen_ply = gen_ply
        self.convertor = dpt_3d_convert()
    
    def makedirs(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def process_meta(self, scene, meta):
        # run depths
        print(f'Estimating zoe-depth for rgb on {scene}:')
        self.processor.run_dpts(meta)

    def run(self, metas):
        for scene, meta in metas.items():
            self.process_meta(scene, meta)
        if self.update_dpt:
            print(f'Average depth estimation time (w/o file loading): {timer.avg}s / image')

