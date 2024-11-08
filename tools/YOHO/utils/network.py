"""
Model in Pytorch of YOHO.
"""

import torch
import torch.nn as nn
import numpy as np


#DRnet
class Comb_Conv(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.comb_layer=nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim,out_dim,(1,13),1)
        )
    def forward(self,input):
        return self.comb_layer(input)

class Residual_Comb_Conv(nn.Module):
    def __init__(self,in_dim,middle_dim,out_dim,Nei_in_SO3):
        super().__init__()
        self.Nei_in_SO3=Nei_in_SO3
        self.comb_layer_in=nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim,middle_dim,(1,13),1)
        )
        self.comb_layer_out=nn.Sequential(
            nn.BatchNorm2d(middle_dim),
            nn.ReLU(),
            nn.Conv2d(middle_dim,out_dim,(1,13),1)
        )
        self.short_cut=False
        if not in_dim==out_dim:
            self.short_cut=True
            self.short_cut_layer=nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim,out_dim,(1,13),1)
            )
    
    def data_process(self,data):
        data=torch.squeeze(data)
        if len(data.size())==2:
            data=data[None,:,:]
        data=data[:,:,self.Nei_in_SO3]
        data=torch.reshape(data,[data.shape[0],data.shape[1],60,13])
        return data

    def forward(self,feat_input):#feat:bn*f*60
        feat=self.data_process(feat_input)
        feat=self.comb_layer_in(feat)
        feat=self.data_process(feat)
        feat=self.comb_layer_out(feat)[:,:,:,0]
        if self.short_cut:
            feat_sc=self.data_process(feat_input)
            feat_sc=self.short_cut_layer(feat_sc)[:,:,:,0]
        else:
            feat_sc=feat_input
        
        return feat+feat_sc #output:bn*f*60

class PartI_network(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg=cfg

        self.Nei_in_SO3=torch.from_numpy(np.load(f'{self.cfg.SO3_related_files}/Nei_Index_in_SO3_ordered_13.npy').astype(np.int).reshape([-1])).cuda()    #nei 60*12 readin
        self.Rgroup_npy=np.load(f'{self.cfg.SO3_related_files}/Rotation.npy').astype(np.float32)
        self.Rgroup=torch.from_numpy(self.Rgroup_npy).cuda()

        self.Conv_in=nn.Sequential(nn.Conv2d(32,256,(1,13),1))
        self.SO3_Conv_layers=nn.ModuleList([Residual_Comb_Conv(256,512,256,self.Nei_in_SO3)])
        self.Conv_out=Comb_Conv(256,32)

    def data_process(self,data):
        data=torch.squeeze(data)
        data=data[:,:,self.Nei_in_SO3]
        data=torch.reshape(data,[data.shape[0],data.shape[1],60,13])
        return data

    def SO3_Conv(self,data):#data:bn,f,gn
        data=self.data_process(data)
        data=self.Conv_in(data)[:,:,:,0]
        for layer in range(len(self.SO3_Conv_layers)):
            data=self.SO3_Conv_layers[layer](data)
        data=self.data_process(data)
        data=self.Conv_out(data)[:,:,:,0]
        return data

        
    def forward(self, feats):
        feats_eqv=self.SO3_Conv(feats)# bn,f,gn
        feats_eqv=feats_eqv+feats
        feats_inv=torch.mean(feats_eqv,dim=-1)# bn,f

        #before conv for partII
        feats_eqv=feats_eqv/torch.clamp_min(torch.norm(feats_eqv,dim=1,keepdim=True),min=1e-4)
        feats_inv=feats_inv/torch.clamp_min(torch.norm(feats_inv,dim=1,keepdim=True),min=1e-4)

        return {'inv':feats_inv,'eqv':feats_eqv}

class PartI_train(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg=cfg
        self.PartI_net=PartI_network(self.cfg)
        self.R_index_permu=torch.from_numpy(np.load(f'{self.cfg.SO3_related_files}/60_60.npy').astype(np.int)).cuda() 
        
    
    def Des2DR(self,Des1,Des2):#before_rot after_rot
        Des1=Des1[:,:,torch.reshape(self.R_index_permu,[-1])].reshape([Des1.shape[0],Des1.shape[1],60,60])
        cor=torch.einsum('bfag,bfg->ba',Des1,Des2)
        return torch.argmax(cor,dim=1)

    def forward(self,data):
        feats0=torch.squeeze(data['feats0']) # bn,32,60
        feats1=torch.squeeze(data['feats1']) # bn,32,60
        true_idxs=torch.squeeze(data['true_idx']) # bn
        yoho_0=self.PartI_net(feats0)
        yoho_1=self.PartI_net(feats1)
        pre_idxs=self.Des2DR(yoho_0['eqv'],yoho_1['eqv'])
        #pre_idxs=self.Des2DR(feats0,feats1)
        part1_ability=torch.mean((pre_idxs==true_idxs).type(torch.float32))

        return {'feats0_eqv_bf_conv':feats0,
                'feats1_eqv_bf_conv':feats1,
                'feats0_eqv_af_conv':yoho_0['eqv'],
                'feats1_eqv_af_conv':yoho_1['eqv'],
                'feats0_inv':yoho_0['inv'],
                'feats1_inv':yoho_1['inv'],
                'DR_pre_ability':part1_ability, # no use for partI
                'DR_true_index':true_idxs,
                'DR_pre_index':pre_idxs}        # no use for partI

class PartI_test(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg=cfg
        self.PartI_net=PartI_network(self.cfg)

    def forward(self,group_feat):
        return self.PartI_net(group_feat)

class PartII_train(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg=cfg

        self.Rgroup_npy=np.load(f'{self.cfg.SO3_related_files}/Rotation.npy').astype(np.float32)
        self.Rgroup=torch.from_numpy(self.Rgroup_npy).cuda()
        self.Nei_in_SO3=torch.from_numpy(np.load(f'{self.cfg.SO3_related_files}/Nei_Index_in_SO3_ordered_13.npy').astype(np.int).reshape([-1])).cuda()    #nei 60*12 readin
        self.R_index_permu=torch.from_numpy(np.load(f'{self.cfg.SO3_related_files}/60_60.npy').astype(np.int)).cuda() 


        self.PartI_net=PartI_train(self.cfg)
        self.Conv_init=Comb_Conv(32*4,256)
        self.PartII_SO3_Conv_layers=nn.ModuleList([Residual_Comb_Conv(256,512,256,self.Nei_in_SO3)])
        
        self.PartII_To_R_dims=[256,512,128,4]
        self.PartII_To_R_FC=nn.Sequential(
            nn.Conv2d(self.PartII_To_R_dims[0],self.PartII_To_R_dims[1],1,1),
            nn.BatchNorm2d(self.PartII_To_R_dims[1]),
            nn.ReLU(),
            nn.Conv2d(self.PartII_To_R_dims[1],self.PartII_To_R_dims[2],1,1),
            nn.BatchNorm2d(self.PartII_To_R_dims[2]),
            nn.ReLU(),
            nn.Conv2d(self.PartII_To_R_dims[2],self.PartII_To_R_dims[3],1,1)
        )
        
    def data_process(self,data):
        data=torch.squeeze(data)
        data=data[:,:,self.Nei_in_SO3]
        data=torch.reshape(data,[data.shape[0],data.shape[1],60,13])
        return data

    def PartII_SO3_Conv(self,data):#data:bn,f,gn
        data=self.data_process(data)
        data=self.Conv_init(data)[:,:,:,0]
        for layer in self.PartII_SO3_Conv_layers:
            data=layer(data)
        return data #data:bn,f,gn
    
    def forward(self, data):
        true_idxs=torch.squeeze(data['true_idx']) # bn
        self.PartI_net.eval()
        with torch.no_grad():
            PartI_output=self.PartI_net(data)
        feats0=PartI_output['feats0_eqv_bf_conv'].detach()
        feats1=PartI_output['feats1_eqv_bf_conv'].detach()
        feats0_eqv=PartI_output['feats0_eqv_af_conv'].detach()
        feats1_eqv=PartI_output['feats1_eqv_af_conv'].detach()
        part1_ability=PartI_output['DR_pre_ability'].detach()
        pre_idxs=PartI_output['DR_pre_index'].detach()
        for i in range(feats0.shape[0]):
            feats0[i]=feats0[i,:,self.R_index_permu[true_idxs[i]]]
            feats0_eqv[i]=feats0_eqv[i,:,self.R_index_permu[true_idxs[i]]]
        feats_eqv=torch.cat([feats0,feats1,feats0_eqv,feats1_eqv],dim=1)
        
        feats_eqv=self.PartII_SO3_Conv(feats_eqv)#bn f gn

        feats_inv=torch.mean(feats_eqv,dim=-1)
        feats_inv=feats_eqv.unsqueeze(-1)#bn f 1 1
        feats_inv=feats_eqv.unsqueeze(-1)
        feats_inv=self.PartII_To_R_FC(feats_inv)#bn 4 1 1
        quaternion_pre=feats_inv[:,:,0,0]
        #quaternion_pre=quaternion_pre/torch.norm(quaternion_pre,dim=1)[:,None]
        
        return {'quaternion_pre':quaternion_pre,
                'part1_ability':part1_ability,
                'pre_idxs':pre_idxs,
                'true_idxs':true_idxs}

class PartII_test(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg=cfg

        self.Rgroup_npy=np.load(f'{self.cfg.SO3_related_files}/Rotation.npy').astype(np.float32)
        self.Rgroup=torch.from_numpy(self.Rgroup_npy).cuda()
        self.Nei_in_SO3=torch.from_numpy(np.load(f'{self.cfg.SO3_related_files}/Nei_Index_in_SO3_ordered_13.npy').astype(np.int).reshape([-1])).cuda()    #nei 60*12 readin
        self.R_index_permu=torch.from_numpy(np.load(f'{self.cfg.SO3_related_files}/60_60.npy').astype(np.int)).cuda() 

        self.Conv_init=Comb_Conv(32*4,256)
        self.PartII_SO3_Conv_layers=nn.ModuleList([Residual_Comb_Conv(256,512,256,self.Nei_in_SO3)])
        
        self.PartII_To_R_dims=[256,512,128,4]
        self.PartII_To_R_FC=nn.Sequential(
            nn.Conv2d(self.PartII_To_R_dims[0],self.PartII_To_R_dims[1],1,1),
            nn.BatchNorm2d(self.PartII_To_R_dims[1]),
            nn.ReLU(),
            nn.Conv2d(self.PartII_To_R_dims[1],self.PartII_To_R_dims[2],1,1),
            nn.BatchNorm2d(self.PartII_To_R_dims[2]),
            nn.ReLU(),
            nn.Conv2d(self.PartII_To_R_dims[2],self.PartII_To_R_dims[3],1,1)
        )
        

    def data_process(self,data):
        data=torch.squeeze(data)
        if len(data.size())==2:
            data=data[None,:,:]
        data=data[:,:,self.Nei_in_SO3]
        data=torch.reshape(data,[data.shape[0],data.shape[1],60,13])
        return data

    def PartII_SO3_Conv(self,data):#data:bn,f,gn
        data=self.data_process(data)
        data=self.Conv_init(data)[:,:,:,0]
        for layer in self.PartII_SO3_Conv_layers:
            data=layer(data)
        return data #data:bn,f,gn
    
    
    def forward(self, data):
        feats0_eqv_bf_conv=data['before_eqv0']
        feats1_eqv_bf_conv=data['before_eqv1']
        feats0_eqv_af_conv=data['after_eqv0']
        feats1_eqv_af_conv=data['after_eqv1']
        pre_idxs=data['pre_idx']
        
        for i in range(feats0_eqv_bf_conv.shape[0]):
            feats0_eqv_bf_conv[i]=feats0_eqv_bf_conv[i,:,self.R_index_permu[pre_idxs[i]]]
            feats0_eqv_af_conv[i]=feats0_eqv_af_conv[i,:,self.R_index_permu[pre_idxs[i]]]
        feats_eqv=torch.cat([feats0_eqv_bf_conv,feats1_eqv_bf_conv,feats0_eqv_af_conv,feats1_eqv_af_conv],dim=1)

        feats_eqv=self.PartII_SO3_Conv(feats_eqv)#bn f gn
        feats_inv=torch.mean(feats_eqv,dim=-1)#bn f
        feats_inv=feats_eqv.unsqueeze(-1)#bn f 1 1
        feats_inv=feats_eqv.unsqueeze(-1)
        feats_inv=self.PartII_To_R_FC(feats_inv)#bn 4 1 1
        quaternion_pre=feats_inv[:,:,0,0]
        quaternion_pre=quaternion_pre/torch.norm(quaternion_pre,dim=1)[:,None]
        return {'quaternion_pre':quaternion_pre,'pre_idxs':pre_idxs}



name2network={  
    'PartI_train':PartI_train,
    'PartI_test':PartI_test,
    'PartII_train':PartII_train,
    'PartII_test':PartII_test
}

