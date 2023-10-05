from tools.controlnet.diff_feature import capture, img_processor, dpt_processor

class null_img_processor():
    def sd_single_img(self, img_fn, prompt):
        return img_fn, prompt

class null_dpt_processor():
    def control_single_dpt(self, dpt_fn, prompt, dpt):
        return dpt_fn, prompt

class Null_model():  
    def __init__(self) -> None:
        self.prompt = ''
        self.capturer = capture(load_model=False)
        self.img_processing = null_img_processor()
        self.dpt_processing = null_dpt_processor()

class control_extractor(Null_model):
    def __init__(self, 
                 load_model = True,
                 seed = -1, 
                 t = 150, 
                 basic = './tools/controlnet/models', 
                 yaml = 'control_v11f1p_sd15_depth.yaml', 
                 sd_ckpt = 'v1-5-pruned.ckpt', 
                 cn_ckpt = 'control_v11f1p_sd15_depth.pth', 
                 prompt = 'a photo of a room and furniture') -> None:
        super().__init__()
        if load_model:
            self.prompt = prompt
            self.capturer = capture(seed = seed, basic = basic, 
                                    yaml = yaml, sd_ckpt = sd_ckpt,
                                    cn_ckpt = cn_ckpt, t = t)
            self.img_processing = img_processor(self.capturer)
            self.dpt_processing = dpt_processor(self.capturer)

    def dpt_feature(self, dpt_fn = '', dpt = None):
        img, feat_list = self.dpt_processing.control_single_dpt(dpt_fn = dpt_fn, prompt = self.prompt, dpt = dpt)
        return img, feat_list

    def rgb_feature(self, img_fn):
        img, feat_list = self.img_processing.sd_single_img(img_fn, prompt = self.prompt)
        return img, feat_list