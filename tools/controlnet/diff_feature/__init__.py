from tools.controlnet.diff_feature.basic import capture
from tools.controlnet.diff_feature.sd_img import img_processor
from tools.controlnet.diff_feature.control_dpt import dpt_processor

def run(s_fn:str, t_fn:str, type = 'img2img', check_layers = [2,5,8], pca_dim = 256, seed = -1):
    capturer = capture(seed=seed)
    if type in  ['img2img']:
        img_processing = img_processor(capturer)
        srgb, sfeat = img_processing.sd_single_img(s_fn)
        trgb, tfeat = img_processing.sd_single_img(t_fn)
    elif type in ['img2dpt']:
        img_processing = img_processor(capturer)
        dpt_processing = dpt_processor(capturer)
        srgb, sfeat = img_processing.sd_single_img(s_fn)
        trgb, tfeat = dpt_processing.control_single_dpt(t_fn)
    elif type in ['dpt2img']:
        img_processing = img_processor(capturer)
        dpt_processing = dpt_processor(capturer)
        srgb, sfeat = dpt_processing.control_single_dpt(s_fn)
        trgb, tfeat = img_processing.sd_single_img(t_fn)
    elif type in ['dpt2dpt']:
        dpt_processing = dpt_processor(capturer)
        srgb, sfeat = dpt_processing.control_single_dpt(s_fn)
        trgb, tfeat = dpt_processing.control_single_dpt(t_fn)
    else:
        raise TypeError('Wrong Type!')
    # match
    uvs, uvt = capturer.img_match(srgb, sfeat, trgb, tfeat, check_layers, pca_dim)    
    return uvs, uvt