"""Training-bound chemical/initial-well kernels on a frozen geometry mean.

The added branch changes the nine-coordinate conditional mean only. It is a
chemical-locality inductive bias, not a target/pathway mechanism or an identified
biological law. Both bases receive the same descriptors and have exactly the
same trainable mixing network. Neither the covariance nor the original endpoint
is part of this module.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .biology_kernel import fit_chemical_anchors, validate_anchor_data
from .hierarchical_geometry import RidgeResidualMean, _positive_integer
from .kernels import SplineKANLinear


class DescriptorBank(nn.Module):
    """Fit-only landmarks/scales; prediction takes X and chemistry, never Y.

    X is the caller's full ridge input: the final coordinate is its observed
    log-norm coordinate. The preceding coordinates are normalized to a direction
    for landmark cosines, while their magnitude is retained as log1p(norm).
    Support diagnostics use all available TRAIN fingerprints. When query IDs
    are provided their identical TRAIN identity is excluded from those support
    diagnostics, but no cutoff or support-based gate changes predictions.
    """
    def __init__(self, config):
        super().__init__()
        self.config = deepcopy(dict(config))
        if self.config.get('schema_version') != 1:
            raise ValueError('Unsupported geometry descriptor schema')
        anchor = validate_anchor_data(self.config['anchor_data'])
        self.input_dim = _positive_integer(self.config['input_dim'], 'input_dim')
        if self.input_dim < 2:
            raise ValueError('The complete input needs morphology and its log-norm coordinate')
        self.chemical_dim = anchor['chemical_dim']
        self.anchor_count = len(anchor['anchor_ids'])
        self.descriptor_dim = 2*self.anchor_count+3
        self.basis_size = 1+3*self.anchor_count
        self.validity_index = anchor['validity_index']
        self.train_ids = list(anchor['train_ids'])
        self._train_lookup = {unit:i for i,unit in enumerate(self.train_ids)}
        dtype = torch.float64
        bits = len(anchor['fingerprint_indices'])
        self.register_buffer('fingerprint_indices', torch.tensor(anchor['fingerprint_indices'],dtype=torch.long))
        self.register_buffer('anchor_fingerprints', torch.tensor(anchor['anchor_fingerprints'],dtype=dtype).reshape(-1,bits))
        self.register_buffer('training_fingerprints',torch.tensor(anchor['training_fingerprints'],dtype=dtype))
        self.register_buffer('training_available',torch.tensor(anchor['training_available'],dtype=torch.bool))
        self.register_buffer('anchor_directions',torch.zeros(self.anchor_count,self.input_dim-1,dtype=dtype))
        self.register_buffer('descriptor_center',torch.zeros(self.descriptor_dim,dtype=dtype))
        self.register_buffer('descriptor_scale',torch.ones(self.descriptor_dim,dtype=dtype))
        self.register_buffer('generic_centers',torch.zeros(self.anchor_count,self.descriptor_dim,dtype=dtype))
        self.register_buffer('generic_width_scale',torch.ones((),dtype=dtype))
        self.register_buffer('generic_bandwidth_multipliers',torch.tensor([.5,1.,2.],dtype=dtype))

    @classmethod
    def fit(cls,x_all,chem_all,mask,ids,fit_ids,metadata=None,max_anchors=64):
        """Index fitting IDs BEFORE estimating any landmark or numeric scale."""
        x,chem,available=np.asarray(x_all),np.asarray(chem_all),np.asarray(mask)
        names=list(map(str,ids)); requested=list(map(str,fit_ids))
        if (x.ndim!=2 or x.shape[1]<2 or chem.ndim!=2 or len(chem)!=len(x)
                or len(names)!=len(x) or available.shape!=(len(x),) or available.dtype!=bool):
            raise ValueError('Compound-aligned full X, chemistry and boolean availability required')
        if len(set(names))!=len(names) or not requested or len(set(requested))!=len(requested):
            raise ValueError('Fitting and source identities must be unique')
        lookup={unit:i for i,unit in enumerate(names)}
        if not set(requested).issubset(lookup):
            raise ValueError('A fitting identity is absent')
        rows=np.asarray([lookup[unit] for unit in requested])
        fit_x=np.asarray(x[rows],dtype=np.float64)
        fit_chem=np.asarray(chem[rows],dtype=np.float64)
        if not np.isfinite(fit_x).all():
            raise ValueError('Fitting X must be finite; no clipping or imputation is applied')
        anchors=fit_chemical_anchors(fit_chem,available[rows],requested,requested,
                                    chemical_metadata=metadata,max_anchors=max_anchors)
        k=len(anchors['anchor_ids'])
        config=dict(schema_version=1,input_dim=x.shape[1],anchor_data=anchors,
            descriptor_names=([f'chemical_T:{unit}' for unit in anchors['anchor_ids']]
                +[f'initial_well_cosine:{unit}' for unit in anchors['anchor_ids']]
                +['input_log_norm_coordinate','log1p_initial_input_norm','fingerprint_bit_density']),
            fitting_ids=requested,scope='initial well plus chemical structural locality, not a biological mechanism',
            support_gate='chemical availability only; support statistics are diagnostics, never cutoffs',
            generic_basis='constant plus fixed RBFs at TRAIN anchor descriptors, three bandwidths',
            generic_bandwidths=[.5,1.,2.],scalar_scale_floor=1e-6)
        bank=cls(config)
        anchor_rows=np.asarray([requested.index(unit) for unit in anchors['anchor_ids']],dtype=int)
        tx=torch.tensor(fit_x,dtype=torch.float64)
        tc=torch.tensor(fit_chem,dtype=torch.float64)
        tm=torch.tensor(available[rows],dtype=torch.bool)
        with torch.no_grad():
            norms=torch.linalg.vector_norm(tx[:,:-1],dim=-1)
            if not torch.isfinite(norms).all():
                raise ValueError('The fitting input norm overflowed')
            directions=tx[:,:-1]/torch.where(norms>0,norms,torch.ones_like(norms))[:,None]
            if k:
                bank.anchor_directions.copy_(directions[anchor_rows])
            raw,_,active,_=bank._raw(tx,tc,tm)
            if active.any():
                fitted=raw[active]
                bank.descriptor_center.copy_(fitted.mean(0))
                scale=fitted.std(0,unbiased=False)
                bank.descriptor_scale.copy_(torch.where(scale<1e-6,torch.ones_like(scale),scale))
                normalized=(raw-bank.descriptor_center)/bank.descriptor_scale
                if k:
                    bank.generic_centers.copy_(normalized[anchor_rows])
                    distances=(normalized[active,None]-bank.generic_centers[None]).square().mean(-1).sqrt()
                    positive=distances[distances>0]
                    if len(positive):
                        bank.generic_width_scale.copy_(torch.quantile(positive,.5))
            diagnostic=bank(tx,tc,tm,ids=requested)
            bank.config['training_support_summary']=dict(
                available=int(active.sum()),self_identity_excluded=True,
                max_similarity_mean=float(diagnostic['max_similarity'][active].mean()) if active.any() else None,
                effective_support_mean=float(diagnostic['effective_support'][active].mean()) if active.any() else None,
                max_similarity_quantiles=(torch.quantile(diagnostic['max_similarity'][active],
                    torch.tensor([0.,.25,.5,.75,1.],dtype=torch.float64)).tolist() if active.any() else []),
                scale_fit='available fitting chemical rows only',
                generic_width_scale=float(bank.generic_width_scale))
        return bank

    @classmethod
    def from_config(cls,config):
        return cls(config)

    def _raw(self,x,chem,mask):
        if (not isinstance(x,torch.Tensor) or x.ndim!=2 or x.shape[1]!=self.input_dim or not len(x)
                or x.dtype!=self.descriptor_center.dtype or x.device!=self.descriptor_center.device):
            raise ValueError('Full X must match descriptor bank dtype/device and input dimension')
        if (not isinstance(chem,torch.Tensor) or chem.shape!=(len(x),self.chemical_dim)
                or chem.dtype!=x.dtype or chem.device!=x.device
                or not isinstance(mask,torch.Tensor) or mask.shape!=(len(x),)
                or mask.dtype!=torch.bool or mask.device!=x.device):
            raise ValueError('Chemical tensor and boolean availability must match X')
        if not torch.isfinite(x).all():
            raise ValueError('Initial-well inputs must be finite')
        active=mask.clone()
        if self.validity_index is not None:
            valid=chem[:,self.validity_index]
            if torch.any(active & (~torch.isfinite(valid)|((valid!=0)&(valid!=1)))):
                raise ValueError('Available chemical validity flags must be binary and finite')
            active &= valid==1
        bits=chem.index_select(-1,self.fingerprint_indices)
        if not torch.isfinite(bits[active]).all() or torch.any((bits[active]!=0)&(bits[active]!=1)):
            raise ValueError('Available chemical fingerprints must be binary and finite')
        bits=torch.where(active[:,None],bits,torch.zeros_like(bits))
        mass=bits.sum(-1)
        active &= (mass>0)&(self.anchor_count>0)
        bits=torch.where(active[:,None],bits,torch.zeros_like(bits))
        intersection=bits@self.anchor_fingerprints.T
        union=mass[:,None]+self.anchor_fingerprints.sum(-1)[None]-intersection
        similarity=intersection/union.clamp_min(1.)
        norms=torch.linalg.vector_norm(x[:,:-1],dim=-1)
        if not torch.isfinite(norms).all():
            raise FloatingPointError('Initial-well norm overflowed')
        direction=x[:,:-1]/torch.where(norms>0,norms,torch.ones_like(norms))[:,None]
        morphology=direction@self.anchor_directions.T
        raw=torch.cat((similarity,morphology,x[:,-1:],torch.log1p(norms)[:,None],
                       (mass/len(self.fingerprint_indices))[:,None]),-1)
        return torch.where(active[:,None],raw,torch.zeros_like(raw)),similarity,active,bits

    def forward(self,x,chem,mask,ids=None):
        raw,similarity,active,bits=self._raw(x,chem,mask)
        normalized=(raw-self.descriptor_center)/self.descriptor_scale
        normalized=torch.where(active[:,None],normalized,torch.zeros_like(normalized))
        intersection=bits@self.training_fingerprints.T
        union=bits.sum(-1)[:,None]+self.training_fingerprints.sum(-1)[None]-intersection
        support=intersection/union.clamp_min(1.)
        support=torch.where(self.training_available[None]&active[:,None],support,torch.zeros_like(support))
        excluded=torch.zeros(len(x),dtype=torch.bool,device=x.device)
        if ids is not None:
            names=list(map(str,ids))
            if len(names)!=len(x):
                raise ValueError('Diagnostic IDs must align with the query objects')
            query,reference=[],[]
            for i,unit in enumerate(names):
                if unit in self._train_lookup:
                    query.append(i);reference.append(self._train_lookup[unit])
            if query:
                support=support.clone()
                support[query,reference]=0.
                excluded[query]=True
        weight_sum=support.sum(-1);weight_square=support.square().sum(-1)
        ess=weight_sum.square()/torch.where(weight_square>0,weight_square,torch.ones_like(weight_square))
        return dict(descriptors=normalized,tanimoto=similarity,availability=active,
            max_similarity=support.max(-1).values,effective_support=ess,similarity_mass=weight_sum,
            support_self_excluded=excluded)

    def basis(self,description,mode):
        d,t=description['descriptors'],description['tanimoto']
        one=torch.ones_like(d[:,:1])
        if mode=='structured':
            return torch.cat((one,t,t.square(),t.pow(4)),-1)
        if mode!='generic':
            raise ValueError('Kernel basis must be structured or generic')
        distance=(d[:,None]-self.generic_centers[None]).square().mean(-1)
        width=self.generic_width_scale*self.generic_bandwidth_multipliers
        responses=[torch.exp(-.5*distance/w.square()) for w in width]
        return torch.cat((one,*responses),-1)

    def save(self,path):
        torch.save(dict(config=self.config,state_dict=self.state_dict()),Path(path))

    @classmethod
    def load(cls,path):
        saved=torch.load(Path(path),map_location='cpu',weights_only=True)
        bank=cls.from_config(saved['config'])
        bank.load_state_dict(saved['state_dict'])
        return bank


class GeometryKernelMean(nn.Module):
    """Bounded, zero-initialized addition inside the frozen HR tanh.

    mu = ridge(X) + b*tanh(f_H(X) + availability*f_K(descriptors)).
    The original total bound b is unchanged. Penalization is on the actual
    increment relative to the frozen HR prediction, not on raw f_K magnitude.
    """
    def __init__(self,base_hr,bank,mode='structured',incremental_penalty=.1,
                 hidden_dim=32,kan_hidden_dim=24):
        super().__init__()
        if not isinstance(base_hr,RidgeResidualMean) or not isinstance(bank,DescriptorBank):
            raise TypeError('A full fitted RidgeResidualMean and DescriptorBank are required')
        if mode not in ('structured','generic') or bank.input_dim!=base_hr.input_dim:
            raise ValueError('Basis mode or full-input dimensions differ')
        if not math.isfinite(incremental_penalty) or incremental_penalty<0:
            raise ValueError('Incremental penalty must be finite and nonnegative')
        hidden_dim=_positive_integer(hidden_dim,'hidden_dim')
        kan_hidden_dim=_positive_integer(kan_hidden_dim,'kan_hidden_dim')
        self.base_hr=deepcopy(base_hr).eval()
        for parameter in self.base_hr.parameters():
            parameter.requires_grad_(False)
        dtype,device=base_hr.coefficient.dtype,base_hr.coefficient.device
        self.bank=deepcopy(bank).to(dtype=dtype,device=device)
        self.mode,self.incremental_penalty_weight=mode,float(incremental_penalty)
        d,b=self.bank.descriptor_dim,self.bank.basis_size
        self.coefficients=nn.Sequential(SplineKANLinear(d,kan_hidden_dim),nn.Tanh(),
                                       SplineKANLinear(kan_hidden_dim,b))
        self.descriptor_path=nn.Sequential(nn.Linear(d,hidden_dim),nn.GELU(),
                                           nn.Linear(hidden_dim,hidden_dim),nn.GELU())
        self.basis_lift=nn.Linear(b,hidden_dim,bias=False)
        self.output=nn.Linear(hidden_dim,9)
        self.to(dtype=dtype,device=device)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.config=dict(schema_version=1,base_hr_config=deepcopy(base_hr.config),bank_config=deepcopy(bank.config),
            mode=mode,incremental_penalty=float(incremental_penalty),hidden_dim=hidden_dim,
            kan_hidden_dim=kan_hidden_dim,dtype=str(dtype).removeprefix('torch.'),
            mean_formula='ridge + inherited_bound * tanh(frozen_hr_raw + availability * kernel_raw)',
            generic_parameters='fixed TRAIN centers and bandwidths; same trainable parameter count as structured',
            covariance_updated=False)

    def train(self,mode=True):
        super().train(mode)
        self.base_hr.eval()  # In particular, copied HR dropout must stay off.
        return self

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def _details(self,x,chem,mask,ids=None):
        self.base_hr._check_inputs(x)
        with torch.no_grad():
            ridge=self.base_hr.base_mean(x)
            hr_raw=self.base_hr.network(x)
            base=ridge+self.base_hr.correction_bound*torch.tanh(hr_raw)
        description=self.bank(x,chem,mask,ids=ids)
        basis=self.bank.basis(description,self.mode)
        coefficients=torch.tanh(self.coefficients(description['descriptors']))
        coefficients=coefficients/(1.+coefficients.abs().sum(-1,keepdim=True))
        hidden=self.descriptor_path(description['descriptors'])+self.basis_lift(coefficients*basis)
        raw=self.output(hidden)
        gated=description['availability'][:,None]*raw
        total=hr_raw+gated
        mean=ridge+self.base_hr.correction_bound*torch.tanh(total)
        if not torch.isfinite(mean).all() or not torch.isfinite(raw).all():
            raise FloatingPointError('The geometry kernel mean overflowed')
        return dict(mean=mean,base_hr_mean=base,ridge_mean=ridge,increment=mean-base,
            kernel_raw=raw,gated_kernel_raw=gated,hr_raw=hr_raw,total_raw=total,
            basis_values=basis,**description)

    def forward(self,x,chem,mask):
        return self._details(x,chem,mask)['mean']

    def loss(self,x,chem,mask,target):
        details=self._details(x,chem,mask)
        predicted=details['mean']
        if (not isinstance(target,torch.Tensor) or target.shape!=predicted.shape
                or target.dtype!=predicted.dtype or target.device!=predicted.device
                or not torch.isfinite(target).all()):
            raise ValueError('Finite standardized nine-coordinate targets must match the mean')
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
            raise ValueError('Unsupported geometry kernel configuration')
        dtype=getattr(torch,config['dtype']);base_cfg=dict(config['base_hr_config'])
        base=RidgeResidualMean.from_config(base_cfg,
            coefficient=torch.zeros(base_cfg['input_dim'],9,dtype=dtype),
            intercept=torch.zeros(9,dtype=dtype))
        bank=DescriptorBank.from_config(config['bank_config'])
        return cls(base,bank,**{key:config[key] for key in
            ('mode','incremental_penalty','hidden_dim','kan_hidden_dim')})

    def save(self,path):
        torch.save(dict(config=self.config,state_dict=self.state_dict()),Path(path))

    @classmethod
    def load(cls,path):
        saved=torch.load(Path(path),map_location='cpu',weights_only=True)
        model=cls.from_config(saved['config'])
        model.load_state_dict(saved['state_dict'])
        model.eval()
        return model
