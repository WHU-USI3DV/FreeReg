"""
Common functions used in the whole pipeline.
"""


import os
import pickle
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from tensorboardX import SummaryWriter

def make_non_exists_dir(fn):
    if not os.path.exists(fn):
        os.makedirs(fn)

def config_writer(cfg,road):
    confwriter=open(road,'w')
    confwriter.write(f'network: {cfg.train_network_type}\n')
    confwriter.write(f'dataset: {cfg.trainset_type}\n')
    confwriter.write(f'loss: {cfg.loss_type}\n')
    confwriter.write(f'lr: {cfg.lr_init} and become *{cfg.lr_decay_rate} pre {cfg.lr_decay_step} epoches\n')
    confwriter.close()

#data preprocess
def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, pkl_path):
    os.system('mkdir -p {}'.format(os.path.dirname(pkl_path)))
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

def points_to_hpoints(points):
    n,_=points.shape
    return np.concatenate([points,np.ones([n,1])],1)

def hpoints_to_points(hpoints):
    return hpoints[:,:-1]/hpoints[:,-1:]

def transform_points(pts,transform):
    h,w=transform.shape
    if h==3 and w==3:
        return pts @ transform.T
    if h==3 and w==4:
        return pts @ transform[:,:3].T + transform[:,3:].T
    elif h==4 and w==4:
        return hpoints_to_points(points_to_hpoints(pts) @ transform.T)
    else: raise NotImplementedError

def random_rotation_matrix():
    """
    Generates a random 3D rotation matrix from axis and angle.

    Args:
        numpy_random_state: numpy random state object

    Returns:
        Random rotation matrix.
    """
    rng = np.random.RandomState()
    axis = rng.rand(3) - 0.5
    axis /= np.linalg.norm(axis) + 1E-8
    theta = np.pi * rng.uniform(0.0, 1.0)
    thetas=axis*theta
    alpha=thetas[0]
    beta=thetas[1]
    gama=thetas[2]
    Rzalpha=np.array([[np.cos(alpha),np.sin(alpha),0],
                      [-np.sin(alpha),np.cos(alpha),0],
                      [0,0,1]])

    Rybeta=np.array([[np.cos(beta),0,-np.sin(beta)],
                     [0,1,0],
                     [np.sin(beta),0,np.cos(beta)]])

    Rzgama=np.array([[np.cos(gama),np.sin(gama),0],
                      [-np.sin(gama),np.cos(gama),0],
                      [0,0,1]])
    R=np.matmul(Rzgama,np.matmul(Rybeta,Rzalpha))
    return R

#train
class MultiGPUWrapper(nn.Module):
    def __init__(self,network,losses):
        super().__init__()
        self.network=network
        self.losses=losses

    def forward(self, data_gt):
        results={}
        data_pr=self.network(data_gt)
        results.update(data_pr)
        for loss in self.losses:
            results.update(loss(data_pr,data_gt,data_gt['step']))
        return results

class DummyLoss:
    def __init__(self,losses):
        self.keys=[]
        for loss in losses:
            self.keys+=loss.keys

    def __call__(self, data_pr, data_gt, step):
        return {key: data_pr[key] for key in self.keys}

def to_cuda(data):
    if type(data)==list:
        results = []
        for i, item in enumerate(data):
            if type(item).__name__ == "Tensor":
                results.append(item.cuda())
            elif type(item).__name__ == 'list':
                tensor_list = []
                for tensor in item:
                    tensor_list.append(tensor.cuda())
                results.append(tensor_list)
            else:
                raise NotImplementedError
        return results
    elif type(data)==dict:
        results={}
        for k,v in data.items():
            if type(v).__name__ == "Tensor":
                results[k]=v.cuda()
            elif type(v).__name__ == 'list':
                tensor_list = []
                for tensor in v:
                    tensor_list.append(tensor.cuda())
                results[k]=tensor_list
            else:
                raise NotImplementedError
        return results
    else:
        raise NotImplementedError

def adjust_learning_rate(optimizer, epoch, lr_decay_rate, lr_decay_epoch, min_lr=1e-6):
    if ((epoch + 1) % lr_decay_epoch) != 0:
        return

    for param_group in optimizer.param_groups:
        # print(param_group)
        lr_before = param_group['lr']
        param_group['lr'] = param_group['lr'] * lr_decay_rate
        param_group['lr'] = max(param_group['lr'], min_lr)

    print('changing learning rate {:5f} to {:.5f}'.format(lr_before, max(param_group['lr'], min_lr)))

def reset_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        # print(param_group)
        # lr_before = param_group['lr']
        param_group['lr'] = lr
    # print('changing learning rate {:5f} to {:.5f}'.format(lr_before,lr))
    return lr

class ExpDecayLR():
    def __init__(self,cfg,decay_step):
        self.lr_init=cfg.lr_init
        self.decay_step=decay_step
        self.decay_rate=cfg.lr_decay_rate

    def __call__(self, step, *args, **kwargs):
        return self.lr_init*(self.decay_rate**(step//self.decay_step))

#record
class Recorder(object):
    def __init__(self, rec_dir, rec_fn):
        self.rec_dir = rec_dir
        self.rec_fn = rec_fn
        self.data = OrderedDict()
        self.writer = SummaryWriter(log_dir=rec_dir)

    def rec_loss(self, losses_batch, step, epoch, prefix='train', dump=False):
        for k, v in losses_batch.items():
            name = '{}/{}'.format(prefix, k)
            if name in self.data:
                self.data[name].append(v)
            else:
                self.data[name] = [v]

        if dump:
            if prefix == 'train':
                msg = '{} epoch {} step {} '.format(prefix, epoch, step)
            else:
                msg = '{} epoch {} '.format(prefix, epoch)
            for k, v in self.data.items():
                if not k.startswith(prefix): continue
                if len(v) > 0:
                    msg += '{} {:.5f} '.format(k.split('/')[-1], np.mean(v))
                    self.writer.add_scalar(k, np.mean(v), step)
                self.data[k] = []

            print(msg)
            with open(self.rec_fn, 'a') as f:
                f.write(msg + '\n')

    def rec_msg(self, msg):
        print(msg)
        with open(self.rec_fn, 'a') as f:
            f.write(msg + '\n')

class Logger:
    def __init__(self, log_dir):
        self.log_dir=log_dir
        self.data = OrderedDict()
        self.writer = SummaryWriter(log_dir=log_dir)

    def log(self,data, prefix='train',step=None,verbose=False):
        msg=f'{prefix} '
        for k, v in data.items():
            msg += f'{k} {v:.5f} '
            self.writer.add_scalar(f'{prefix}/{k}',v,step)

        if verbose:
            print(msg)
        with open(os.path.join(self.log_dir,f'{prefix}.txt'), 'a') as f:
            f.write(msg + '\n')

def evaluate_the_match(kps0,kps1,matches,transform_gt,threshold=0.1):
    # evaluate the matching precision
    kpsm0=kps0[matches[:,0]]
    kpsm1=kps1[matches[:,1]]
    kpsm1_t=transform_points(kpsm1,transform_gt)
    dist=np.linalg.norm(kpsm0-kpsm1_t,2,1)
    correct_ratio=np.mean(dist<threshold)#正确比例
    return correct_ratio