"""Descriptor-local response bases with bounded, blockwise conditional gates.

Chemical structural locality, signed initial-well morphology and observed
amplitude descriptors remain explicit up to a linear nine-coordinate readout.
This module changes only a frozen-HR conditional mean, not its covariance or
the experimental endpoint; these bases are not verified target/pathway laws.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .geometry_kernel import DescriptorBank
from .geometry_kernel_replacement import ReplacementBasisBank
from .hierarchical_geometry import RidgeResidualMean,_positive_integer


MODES=('static_structured','conditional_generic','conditional_structured')


class LocalResponseBank(nn.Module):
    """Three explicit responses per descriptor, with TRAIN-only block scales."""
    SCALE_FLOOR=1e-6

    def __init__(self,descriptor_bank):
        super().__init__()
        if not isinstance(descriptor_bank,DescriptorBank):
            raise TypeError('A fitted DescriptorBank is required')
        self.descriptor_bank=deepcopy(descriptor_bank)
        self.input_dim=descriptor_bank.input_dim
        self.chemical_dim=descriptor_bank.chemical_dim
        self.anchor_count=descriptor_bank.anchor_count
        self.descriptor_dim=descriptor_bank.descriptor_dim
        self.basis_size=3*self.descriptor_dim
        reference=descriptor_bank.descriptor_center
        for name in ('chemical_generic_scale','chemical_structured_scale','morphology_scale','scalar_scale'):
            self.register_buffer(name,reference.new_ones(()))
        self.config=dict(schema_version=1,descriptor_config=deepcopy(descriptor_bank.config),
            basis_shape=[self.descriptor_dim,3],basis_size=self.basis_size,
            chemical_structured='[T,T^2,T^4]',
            chemical_generic=dict(centers=[0.,.5,1.],width=.25),
            morphology='[q,tanh(2q),q*abs(q)] using unstandardized initial-direction cosine',
            scalar='[v,tanh(v),v/hypot(v,1)] using final three standardized observed descriptors',
            normalization='one available-TRAIN uncentered RMS per complete block; no object normalization',
            scale_floor=self.SCALE_FLOOR,raw_descriptor_bypass=False,
            support_gate='availability only; similarity support is diagnostic')

    @classmethod
    def fit(cls,x_all,chem_all,mask,ids,fit_ids,metadata=None,max_anchors=64):
        # Reuse exactly the existing fitting identities, landmarks and chemical
        # block RMS; only the signed morphology/scalar responses differ.
        previous=ReplacementBasisBank.fit(x_all,chem_all,mask,ids,fit_ids,metadata,max_anchors)
        bank=cls(previous.descriptor_bank)
        bank.chemical_generic_scale.copy_(previous.chemical_generic_scale)
        bank.chemical_structured_scale.copy_(previous.chemical_structured_scale)
        requested=bank.descriptor_bank.config['fitting_ids']
        lookup={str(unit):i for i,unit in enumerate(ids)}
        rows=np.asarray([lookup[unit] for unit in requested])
        x=torch.as_tensor(np.asarray(x_all)[rows],dtype=bank.morphology_scale.dtype)
        chem=torch.as_tensor(np.asarray(chem_all)[rows],dtype=x.dtype)
        available=torch.as_tensor(np.asarray(mask)[rows],dtype=torch.bool)
        with torch.no_grad():
            description=bank(x,chem,available,ids=requested)
            raw=bank._unscaled_blocks(description,'structured')
            active=description['availability']
            values={}
            for name in ('morphology','scalar'):
                fitting=raw[name][active]
                value=float(fitting.square().mean().sqrt()) if fitting.numel() else 0.
                if not math.isfinite(value):
                    raise ValueError('A fitting local response block has nonfinite RMS')
                getattr(bank,name+'_scale').fill_(max(value,cls.SCALE_FLOOR))
                values[name]=value
        bank.config.update(fitting_ids=requested,available_fitting_rows=int(active.sum()),
            fitting_block_rms=dict(chemical_generic=previous.config['fitting_block_rms']['chemical_generic'],
                chemical_structured=previous.config['fitting_block_rms']['chemical_structured'],**values),
            applied_block_scales={name:float(getattr(bank,name+'_scale')) for name in
                ('chemical_generic','chemical_structured','morphology','scalar')})
        return bank

    def forward(self,x,chem,mask,ids=None):
        return self.descriptor_bank(x,chem,mask,ids=ids)

    def _unscaled_blocks(self,description,mode):
        if mode in ('static_structured','conditional_structured'):mode='structured'
        if mode=='conditional_generic':mode='generic'
        if mode not in ('generic','structured'):
            raise ValueError('Unknown local response basis')
        d,t=description['descriptors'],description['tanimoto']
        if d.ndim!=2 or d.shape[1]!=self.descriptor_dim or t.shape!=(len(d),self.anchor_count):
            raise ValueError('Descriptor geometry differs from the fitted local basis')
        if mode=='structured':chemical=torch.stack((t,t.square(),t.pow(4)),-1)
        else:chemical=torch.stack([torch.exp(-.5*((t-center)/.25).square()) for center in (0.,.5,1.)],-1)
        unscaled=d*self.descriptor_bank.descriptor_scale+self.descriptor_bank.descriptor_center
        q=unscaled[:,self.anchor_count:2*self.anchor_count]
        morphology=torch.stack((q,torch.tanh(2*q),q*q.abs()),-1)
        v=d[:,-3:]
        scalar=torch.stack((v,torch.tanh(v),v/torch.hypot(v,torch.ones_like(v))),-1)
        active=description['availability'][:,None,None]
        return {key:torch.where(active,value,torch.zeros_like(value)) for key,value in
            dict(chemical=chemical,morphology=morphology,scalar=scalar).items()}

    def basis_blocks(self,description,mode):
        chemical_mode='generic' if mode in ('generic','conditional_generic') else 'structured'
        blocks=self._unscaled_blocks(description,mode)
        scales=dict(chemical=getattr(self,f'chemical_{chemical_mode}_scale'),
                    morphology=self.morphology_scale,scalar=self.scalar_scale)
        if any(not torch.isfinite(scale) or scale<=0 for scale in scales.values()):
            raise ValueError('Local basis scales must be positive and finite')
        result={key:value/scales[key] for key,value in blocks.items()}
        if any(not torch.isfinite(value).all() for value in result.values()):
            raise FloatingPointError('A local response basis overflowed; no feature is clipped')
        return result

    def basis(self,description,mode):
        blocks=self.basis_blocks(description,mode)
        return torch.cat((blocks['chemical'],blocks['morphology'],blocks['scalar']),dim=1)

    @classmethod
    def from_config(cls,config):
        config=deepcopy(dict(config))
        if config.get('schema_version')!=1:raise ValueError('Unsupported local response bank schema')
        bank=cls(DescriptorBank.from_config(config['descriptor_config']))
        for key in ('basis_shape','basis_size','scale_floor','chemical_structured','chemical_generic','morphology','scalar'):
            if config.get(key)!=bank.config[key]:raise ValueError('Saved local response definition changed: '+key)
        bank.config=config
        return bank

    def save(self,path):
        torch.save(dict(config=self.config,state_dict=self.state_dict(),
            dtype=str(self.scalar_scale.dtype).removeprefix('torch.')),Path(path))

    @classmethod
    def load(cls,path):
        saved=torch.load(Path(path),map_location='cpu',weights_only=True)
        if saved['dtype'] not in ('float32','float64'):raise ValueError('Unsupported local bank dtype')
        bank=cls.from_config(saved['config']).to(dtype=getattr(torch,saved['dtype']))
        bank.load_state_dict(saved['state_dict'])
        return bank


class ConditionalResponseKernelMean(nn.Module):
    """Local response mixing followed only by a bias-free linear readout.

    A conditional gate receives the descriptors but can affect predictions only
    through the explicit response bases. Its last layer and the final readout
    start at zero, so gradients reach those layers in successive updates while
    the initial predictor exactly equals the immutable historical HR mean.
    """
    def __init__(self,base_hr,bank,mode='conditional_structured',incremental_penalty=.1,hidden_dim=16):
        super().__init__()
        if not isinstance(base_hr,RidgeResidualMean) or not isinstance(bank,LocalResponseBank):
            raise TypeError('A complete frozen HR mean and LocalResponseBank are required')
        if mode not in MODES or base_hr.input_dim!=bank.input_dim:
            raise ValueError('Local response mode or initial-well dimensions differ')
        if not math.isfinite(incremental_penalty) or incremental_penalty<0:
            raise ValueError('Incremental penalty must be finite and nonnegative')
        hidden_dim=_positive_integer(hidden_dim,'hidden_dim')
        self.base_hr=deepcopy(base_hr).eval().requires_grad_(False)
        self.bank=deepcopy(bank).to(dtype=base_hr.coefficient.dtype,device=base_hr.coefficient.device)
        self.mode=mode;self.incremental_penalty_weight=float(incremental_penalty)
        self.local_coefficients=nn.Parameter(torch.full((bank.descriptor_dim,3),1/3,
            dtype=base_hr.coefficient.dtype,device=base_hr.coefficient.device))
        if mode=='static_structured':
            self.condition_logits=nn.Parameter(torch.zeros(3,3))
            self.conditioner=None
        else:
            self.register_parameter('condition_logits',None)
            self.conditioner=nn.Sequential(nn.Linear(bank.descriptor_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,9))
            nn.init.zeros_(self.conditioner[-1].weight);nn.init.zeros_(self.conditioner[-1].bias)
        self.output=nn.Linear(bank.descriptor_dim,9,bias=False)
        self.register_buffer('descriptor_block',torch.tensor([0]*bank.anchor_count+[1]*bank.anchor_count+[2]*3,dtype=torch.long))
        self.to(dtype=base_hr.coefficient.dtype,device=base_hr.coefficient.device)
        nn.init.zeros_(self.output.weight)
        self.config=dict(schema_version=1,base_hr_config=deepcopy(base_hr.config),bank_config=deepcopy(bank.config),
            mode=mode,incremental_penalty=float(incremental_penalty),hidden_dim=hidden_dim,
            dtype=str(base_hr.coefficient.dtype).removeprefix('torch.'),
            gate_formula='1 + 0.5*tanh(three-block by three-basis logits)',
            coefficient_shape=[bank.descriptor_dim,3],coefficient_initialization='1/3, trainable, unconstrained',
            mean_formula='ridge + inherited_bound*tanh(frozen_HR_raw + availability*linear_local_response)',
            raw_descriptor_bypass=False,output_bias=False,nonlinear_downstream_readout=False,
            covariance_updated=False,object_normalization=False,
            parameter_count_scope='conditional generic/structured matched; static removes the conditioning MLP')

    def train(self,mode=True):
        super().train(mode);self.base_hr.eval()
        return self

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def _details(self,x,chem,mask,ids=None):
        self.base_hr._check_inputs(x)
        with torch.no_grad():
            ridge=self.base_hr.base_mean(x);hr_raw=self.base_hr.network(x)
            baseline=ridge+self.base_hr.correction_bound*torch.tanh(hr_raw)
        description=self.bank(x,chem,mask,ids=ids)
        blocks=self.bank.basis_blocks(description,self.mode)
        basis=torch.cat((blocks['chemical'],blocks['morphology'],blocks['scalar']),dim=1)
        logits=(self.condition_logits[None].expand(len(x),-1,-1) if self.conditioner is None else
                self.conditioner(description['descriptors']).reshape(len(x),3,3))
        gate=1+.5*torch.tanh(logits)
        descriptor_gate=gate[:,self.descriptor_block,:]
        local=(basis*self.local_coefficients[None]*descriptor_gate).sum(-1)
        raw=self.output(local)
        availability=description['availability'][:,None]
        gated=availability*raw;total=hr_raw+gated
        mean=ridge+self.base_hr.correction_bound*torch.tanh(total)
        if not torch.isfinite(raw).all() or not torch.isfinite(mean).all():
            raise FloatingPointError('Local response predictor produced a nonfinite mean')
        contributions=torch.stack([F.linear(local*(self.descriptor_block==block)[None],self.output.weight)
            for block in range(3)],dim=1)
        return dict(mean=mean,base_hr_mean=baseline,ridge_mean=ridge,increment=mean-baseline,
            kernel_raw=raw,path_output=raw,output_bias=torch.zeros_like(raw),
            gated_kernel_raw=gated,hr_raw=hr_raw,total_raw=total,
            basis_values=basis.flatten(1),local_basis=basis,local_coefficients=self.local_coefficients,
            block_gate=gate,gate=gate,descriptor_gate=descriptor_gate,condition_logits=logits,
            local_activation=local,block_readout_contributions=contributions,block_contributions=contributions,
            chemical_readout=contributions[:,0],morphology_readout=contributions[:,1],scalar_readout=contributions[:,2],
            chemical_basis=blocks['chemical'].flatten(1),morphology_basis=blocks['morphology'].flatten(1),
            scalar_basis=blocks['scalar'].flatten(1),**description)

    def forward(self,x,chem,mask):
        return self._details(x,chem,mask)['mean']

    def loss(self,x,chem,mask,target):
        details=self._details(x,chem,mask);predicted=details['mean']
        if (not isinstance(target,torch.Tensor) or target.shape!=predicted.shape or target.dtype!=predicted.dtype
                or target.device!=predicted.device or not torch.isfinite(target).all()):
            raise ValueError('Target must be finite and aligned to the standardized nine-coordinate prediction')
        mse=F.mse_loss(predicted,target);increment=details['increment'].square().mean()
        penalty=self.incremental_penalty_weight*increment
        return dict(loss=mse+penalty,mean_mse=mse,incremental_mse=increment,incremental_penalty=penalty)

    @torch.no_grad()
    def diagnostics(self,x,chem,mask,ids=None):
        details=self._details(x,chem,mask,ids=ids)
        details.update(saturation_fraction=(torch.tanh(details['total_raw']).abs()>=.95).to(x.dtype).mean(),
            inherited_hr_saturation_fraction=(torch.tanh(details['hr_raw']).abs()>=.95).to(x.dtype).mean(),
            incremental_mse=details['increment'].square().mean(),
            total_correction_max=(details['mean']-details['ridge_mean']).abs().max())
        return details

    @classmethod
    def from_config(cls,config):
        config=deepcopy(dict(config))
        if config.get('schema_version')!=1 or config.get('dtype') not in ('float32','float64'):
            raise ValueError('Unsupported conditional response model configuration')
        bc=dict(config['base_hr_config']);dtype=getattr(torch,config['dtype'])
        hr=RidgeResidualMean.from_config(bc,coefficient=torch.zeros(bc['input_dim'],9,dtype=dtype),
                                       intercept=torch.zeros(9,dtype=dtype))
        bank=LocalResponseBank.from_config(config['bank_config'])
        model=cls(hr,bank,**{key:config[key] for key in ('mode','incremental_penalty','hidden_dim')})
        for key in ('gate_formula','coefficient_shape','raw_descriptor_bypass','output_bias','nonlinear_downstream_readout','object_normalization'):
            if config.get(key)!=model.config[key]:raise ValueError('Saved conditional response architecture changed: '+key)
        return model

    def save(self,path):
        config=deepcopy(self.config);config['dtype']=str(self.base_hr.coefficient.dtype).removeprefix('torch.')
        torch.save(dict(config=config,state_dict=self.state_dict()),Path(path))

    @classmethod
    def load(cls,path):
        saved=torch.load(Path(path),map_location='cpu',weights_only=True)
        model=cls.from_config(saved['config']);model.load_state_dict(saved['state_dict'])
        model.eval()
        return model
