"""Single-path descriptor MLP or explicit-basis KAN on a frozen HR mean.

Generic and structured kernels differ only in their chemical response block.
Morphology and scalar blocks are identical. Each block has one TRAIN-fitted
RMS scale, not per-object normalization; no raw-descriptor bypass or coefficient
L1 normalization enters the kernel path. These are structural feature priors,
not verified drug-target/pathway laws or new covariance models.
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
from .hierarchical_geometry import RidgeResidualMean, _positive_integer
from .kernels import SplineKANLinear


class ReplacementBasisBank(nn.Module):
    """Original fit-only descriptors plus fixed, block-scaled response bases."""
    SCALE_FLOOR = 1e-6

    def __init__(self,descriptor_bank):
        super().__init__()
        if not isinstance(descriptor_bank,DescriptorBank):
            raise TypeError('A fitted DescriptorBank is required')
        self.descriptor_bank=deepcopy(descriptor_bank)
        self.input_dim=descriptor_bank.input_dim
        self.chemical_dim=descriptor_bank.chemical_dim
        self.anchor_count=descriptor_bank.anchor_count
        self.descriptor_dim=descriptor_bank.descriptor_dim
        self.basis_size=6*self.anchor_count+9
        reference=descriptor_bank.descriptor_center
        for name in ('chemical_generic_scale','chemical_structured_scale','morphology_scale','scalar_scale'):
            self.register_buffer(name,reference.new_ones(()))
        self.config=dict(schema_version=1,descriptor_config=deepcopy(descriptor_bank.config),
            chemical_structured='concatenate(T,T^2,T^4); no constant basis',
            chemical_generic=dict(kind='per-T fixed RBF',centers=[0.,.5,1.],width=.25),
            morphology=dict(kind='per-cosine fixed RBF',centers=[-1.,0.,1.],width=.5),
            scalar='concatenate(v,tanh(v),v^2/(1+v^2)); final three standardized descriptors',
            normalization='one RMS per entire block, available TRAIN rows only; no per-object normalization',
            scale_floor=self.SCALE_FLOOR,basis_size=self.basis_size,
            support_gate='availability only; continuous support diagnostics do not alter predictions')

    @classmethod
    def fit(cls,x_all,chem_all,mask,ids,fit_ids,metadata=None,max_anchors=64):
        descriptor=DescriptorBank.fit(x_all,chem_all,mask,ids,fit_ids,metadata,max_anchors)
        bank=cls(descriptor)
        names=list(map(str,ids));lookup={unit:i for i,unit in enumerate(names)}
        rows=np.asarray([lookup[unit] for unit in descriptor.config['fitting_ids']])
        x=torch.as_tensor(np.asarray(x_all)[rows],dtype=descriptor.descriptor_center.dtype)
        chem=torch.as_tensor(np.asarray(chem_all)[rows],dtype=x.dtype)
        available=torch.as_tensor(np.asarray(mask)[rows],dtype=torch.bool)
        rms={}
        with torch.no_grad():
            description=descriptor(x,chem,available,ids=descriptor.config['fitting_ids'])
            active=description['availability']
            for mode in ('generic','structured'):
                blocks=bank._unscaled_blocks(description,mode)
                for name,values in blocks.items():
                    key=f'chemical_{mode}' if name=='chemical' else name
                    if key in rms:
                        continue
                    fitting=values[active]
                    value=float(fitting.square().mean().sqrt()) if fitting.numel() else 0.
                    if not math.isfinite(value):
                        raise ValueError('A fitting basis block has nonfinite RMS')
                    rms[key]=value
                    getattr(bank,key+'_scale').fill_(max(value,cls.SCALE_FLOOR))
            bank.config.update(fitting_ids=descriptor.config['fitting_ids'],
                available_fitting_rows=int(active.sum()),fitting_block_rms=rms,
                applied_block_scales={key:max(value,cls.SCALE_FLOOR) for key,value in rms.items()})
        return bank

    def forward(self,x,chem,mask,ids=None):
        return self.descriptor_bank(x,chem,mask,ids=ids)

    def _unscaled_blocks(self,description,mode):
        if mode not in ('generic','structured'):
            raise ValueError('An explicit kernel basis is generic or structured')
        d,t=description['descriptors'],description['tanimoto']
        if d.ndim!=2 or d.shape[1]!=self.descriptor_dim or t.shape!=(len(d),self.anchor_count):
            raise ValueError('The descriptor and anchor geometry differs from the fitted bank')
        if mode=='structured':
            chemical=torch.cat((t,t.square(),t.pow(4)),-1)
        else:
            chemical=torch.cat([torch.exp(-.5*((t-center)/.25).square())
                                for center in (0.,.5,1.)],-1)
        # Recover original initial-well cosines from the saved fitting scaler.
        original=d*self.descriptor_bank.descriptor_scale+self.descriptor_bank.descriptor_center
        cosine=original[:,self.anchor_count:2*self.anchor_count]
        morphology=torch.cat([torch.exp(-.5*((cosine-center)/.5).square())
                              for center in (-1.,0.,1.)],-1)
        v=d[:,-3:]
        # hypot avoids overflow in v^2/(1+v^2), without clipping v itself.
        saturation=(v/torch.hypot(v,torch.ones_like(v))).square()
        scalar=torch.cat((v,torch.tanh(v),saturation),-1)
        active=description['availability'][:,None]
        return {name:torch.where(active,value,torch.zeros_like(value)) for name,value in
                dict(chemical=chemical,morphology=morphology,scalar=scalar).items()}

    def basis_blocks(self,description,mode):
        raw=self._unscaled_blocks(description,mode)
        scales=dict(chemical=getattr(self,f'chemical_{mode}_scale'),
                    morphology=self.morphology_scale,scalar=self.scalar_scale)
        if any(not torch.isfinite(scale) or scale<=0 for scale in scales.values()):
            raise ValueError('Basis RMS scales must remain positive and finite')
        values={name:value/scales[name] for name,value in raw.items()}
        if any(not torch.isfinite(value).all() for value in values.values()):
            raise FloatingPointError('An explicit basis overflowed; no coordinates are clipped')
        return values

    def basis(self,description,mode):
        blocks=self.basis_blocks(description,mode)
        return torch.cat((blocks['chemical'],blocks['morphology'],blocks['scalar']),-1)

    @classmethod
    def from_config(cls,config):
        config=deepcopy(dict(config))
        if config.get('schema_version')!=1:
            raise ValueError('Unsupported replacement basis schema')
        bank=cls(DescriptorBank.from_config(config['descriptor_config']))
        if config.get('basis_size')!=bank.basis_size or config.get('scale_floor')!=cls.SCALE_FLOOR:
            raise ValueError('The saved explicit basis dimensions or scale rule changed')
        bank.config=config
        return bank

    def save(self,path):
        torch.save(dict(config=self.config,state_dict=self.state_dict(),
            dtype=str(self.morphology_scale.dtype).removeprefix('torch.')),Path(path))

    @classmethod
    def load(cls,path):
        saved=torch.load(Path(path),map_location='cpu',weights_only=True)
        bank=cls.from_config(saved['config']).to(dtype=getattr(torch,saved['dtype']))
        bank.load_state_dict(saved['state_dict'])
        return bank


class GeometryKernelReplacementMean(nn.Module):
    """One learned path, inside the unchanged frozen-HR correction bound.

    The MLP receives standardized descriptors directly. A kernel receives only
    the concatenated explicit basis. Neither has an additive descriptor bypass.
    All trainable pathways share a zero-initialized final nine-coordinate head.
    """
    def __init__(self,base_hr,bank,mode='structured',incremental_penalty=.1,
                 hidden_dim=32,kan_hidden_dim=24):
        super().__init__()
        if not isinstance(base_hr,RidgeResidualMean) or not isinstance(bank,ReplacementBasisBank):
            raise TypeError('A complete fitted HR mean and ReplacementBasisBank are required')
        if mode not in ('mlp','generic','structured') or base_hr.input_dim!=bank.input_dim:
            raise ValueError('Single-path mode or complete initial-well input dimension differs')
        if not math.isfinite(incremental_penalty) or incremental_penalty<0:
            raise ValueError('Incremental penalty must be finite and nonnegative')
        hidden_dim=_positive_integer(hidden_dim,'hidden_dim')
        kan_hidden_dim=_positive_integer(kan_hidden_dim,'kan_hidden_dim')
        self.base_hr=deepcopy(base_hr).eval()
        self.base_hr.requires_grad_(False)
        dtype,device=base_hr.coefficient.dtype,base_hr.coefficient.device
        self.bank=deepcopy(bank).to(dtype=dtype,device=device)
        self.mode,self.incremental_penalty_weight=mode,float(incremental_penalty)
        if mode=='mlp':
            self.path=nn.Sequential(nn.Linear(bank.descriptor_dim,hidden_dim),nn.GELU(),
                                    nn.Linear(hidden_dim,hidden_dim),nn.GELU())
        else:
            self.path=nn.Sequential(SplineKANLinear(bank.basis_size,kan_hidden_dim),nn.Tanh(),
                                    SplineKANLinear(kan_hidden_dim,hidden_dim),nn.Tanh())
        self.output=nn.Linear(hidden_dim,9)
        self.to(dtype=dtype,device=device)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.config=dict(schema_version=1,base_hr_config=deepcopy(base_hr.config),
            bank_config=deepcopy(bank.config),mode=mode,incremental_penalty=float(incremental_penalty),
            hidden_dim=hidden_dim,kan_hidden_dim=kan_hidden_dim,dtype=str(dtype).removeprefix('torch.'),
            mean_formula='ridge + inherited_bound*tanh(frozen_hr_raw + availability*single_path_raw)',
            descriptor_bypass=False,coefficient_l1_scaling=False,covariance_updated=False,
            parameter_count_scope='generic and structured matched; MLP architecture is not parameter matched')

    def train(self,mode=True):
        super().train(mode)
        self.base_hr.eval()
        return self

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def _details(self,x,chem,mask,ids=None):
        self.base_hr._check_inputs(x)
        with torch.no_grad():
            ridge=self.base_hr.base_mean(x)
            hr_raw=self.base_hr.network(x)
            baseline=ridge+self.base_hr.correction_bound*torch.tanh(hr_raw)
        description=self.bank(x,chem,mask,ids=ids)
        if self.mode=='mlp':
            path_input=description['descriptors']
            blocks={name:path_input.new_empty((len(x),0)) for name in ('chemical','morphology','scalar')}
            basis=path_input.new_empty((len(x),0))
        else:
            blocks=self.bank.basis_blocks(description,self.mode)
            basis=torch.cat((blocks['chemical'],blocks['morphology'],blocks['scalar']),-1)
            path_input=basis
        hidden=self.path(path_input)
        raw=self.output(hidden)
        path_output=F.linear(hidden,self.output.weight,None)
        head_bias=self.output.bias[None].expand_as(raw)
        gated=description['availability'][:,None]*raw
        total=hr_raw+gated
        mean=ridge+self.base_hr.correction_bound*torch.tanh(total)
        if not torch.isfinite(raw).all() or not torch.isfinite(mean).all():
            raise FloatingPointError('The single-path geometry mean overflowed')
        return dict(mean=mean,base_hr_mean=baseline,ridge_mean=ridge,increment=mean-baseline,
            kernel_raw=raw,gated_kernel_raw=gated,hr_raw=hr_raw,total_raw=total,
            path_output=path_output,output_bias=head_bias,basis_values=basis,
            chemical_basis=blocks['chemical'],morphology_basis=blocks['morphology'],
            scalar_basis=blocks['scalar'],**description)

    def forward(self,x,chem,mask):
        return self._details(x,chem,mask)['mean']

    def loss(self,x,chem,mask,target):
        details=self._details(x,chem,mask)
        predicted=details['mean']
        if (not isinstance(target,torch.Tensor) or target.shape!=predicted.shape
                or target.dtype!=predicted.dtype or target.device!=predicted.device
                or not torch.isfinite(target).all()):
            raise ValueError('The target must be finite, aligned standardized nine-coordinate output')
        mse=F.mse_loss(predicted,target)
        increment=details['increment'].square().mean()
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
        config=dict(config)
        if config.get('schema_version')!=1 or config.get('dtype') not in ('float32','float64'):
            raise ValueError('Unsupported replacement model configuration')
        dtype=getattr(torch,config['dtype']);base_config=dict(config['base_hr_config'])
        hr=RidgeResidualMean.from_config(base_config,
            coefficient=torch.zeros(base_config['input_dim'],9,dtype=dtype),
            intercept=torch.zeros(9,dtype=dtype))
        bank=ReplacementBasisBank.from_config(config['bank_config'])
        return cls(hr,bank,**{key:config[key] for key in
            ('mode','incremental_penalty','hidden_dim','kan_hidden_dim')})

    def save(self,path):
        config=deepcopy(self.config)
        config['dtype']=str(self.base_hr.coefficient.dtype).removeprefix('torch.')
        torch.save(dict(config=config,state_dict=self.state_dict()),Path(path))

    @classmethod
    def load(cls,path):
        saved=torch.load(Path(path),map_location='cpu',weights_only=True)
        model=cls.from_config(saved['config'])
        model.load_state_dict(saved['state_dict'])
        model.eval()
        return model
