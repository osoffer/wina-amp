import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'utils'))

import types
from torch import nn

from utils.utils import ActivationModule, Distribution, SparsifyFn, get_module_device

import torch


def _monkeypatch_mlp(mlp, file_path, grabbing_mode=False, sparse_mode=None, mask_by=None):
    mlp.forward_old = mlp.forward
    if 'phi' in mlp.__class__.__name__.lower():
        mlp.forward = types.MethodType(_mlp_forward_phi, mlp)
    else:
        mlp.forward = types.MethodType(_mlp_forward, mlp)

    mlp.add_sparse_fns = types.MethodType(add_sparse_fns, mlp)
    
    mlp.file_path = file_path
    mlp.grabbing_mode = grabbing_mode
    
    mlp.sparse_mode = sparse_mode
    if sparse_mode == 'wina':
        if 'phi' in mlp.__class__.__name__.lower():
            mlp.gate_up_norm_by_column = mlp.gate_up_proj.weight.norm(dim=0)
            mlp.down_norm_by_column = torch.tensor(1.)
        else:    
            mlp.gate_norm_by_column = mlp.gate_proj.weight.norm(dim=0)
            mlp.up_norm_by_column = torch.tensor(1.)
            mlp.down_norm_by_column = torch.tensor(1.)
            
                    
    if not grabbing_mode:
        if sparse_mode == 'wina':
            mlp.distrs = {}
            if 'phi' in mlp.__class__.__name__.lower():
                mlp.distrs['gate_up'] = Distribution(file_path, hidden_type='gate_up') if mask_by == 'threshold' else None
                mlp.distrs['down'] = Distribution(file_path, hidden_type='down') if mask_by == 'threshold' else None
            
                mlp.sparse_fns = nn.ModuleDict({
                    'gate_up': SparsifyFn(mlp.distrs['gate_up'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(mlp)),
                    'down': SparsifyFn(mlp.distrs['down'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(mlp)),
                })
            else:
                mlp.distrs['gate'] = Distribution(file_path, hidden_type='gate') if mask_by == 'threshold' else None
                mlp.distrs['up'] = Distribution(file_path, hidden_type='up') if mask_by == 'threshold' else None
                mlp.distrs['down'] = Distribution(file_path, hidden_type='down') if mask_by == 'threshold' else None
            
                mlp.sparse_fns = nn.ModuleDict({
                    'gate': SparsifyFn(mlp.distrs['gate'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(mlp)),
                    'up': SparsifyFn(mlp.distrs['up'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(mlp)),
                    'down': SparsifyFn(mlp.distrs['down'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(mlp)),
                })
                
        elif sparse_mode == 'teal':
            mlp.distrs = {}
            mlp.distrs['h1'] = Distribution(file_path, hidden_type='h1') if mask_by == 'threshold' else None
            mlp.distrs['h2'] = Distribution(file_path, hidden_type='h2') if mask_by == 'threshold' else None

            if 'phi' in mlp.__class__.__name__.lower():
                mlp.sparse_fns = nn.ModuleDict({
                    'gate_up': SparsifyFn(mlp.distrs['h1'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(mlp)),
                    'down': SparsifyFn(mlp.distrs['h2'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(mlp)),
                })
            else:
                mlp.sparse_fns = nn.ModuleDict({
                    'gate': SparsifyFn(mlp.distrs['h1'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(mlp)),
                    'up': SparsifyFn(mlp.distrs['h1'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(mlp)),
                    'down': SparsifyFn(mlp.distrs['h2'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(mlp)),
                })

    mlp.activation_module = ActivationModule(file_path)

    return mlp

def add_sparse_fns(self, sparsity=0.25, mask_by=None):
    mlp = self
    mlp.grabbing_mode = False
    sparse_mode = mlp.sparse_mode
    file_path = mlp.file_path
    if sparse_mode == 'wina':
        mlp.distrs = {}
        if 'phi' in mlp.__class__.__name__.lower():
            mlp.distrs['gate_up'] = Distribution(file_path, hidden_type='gate_up') if mask_by == 'threshold' else None
            mlp.distrs['down'] = Distribution(file_path, hidden_type='down') if mask_by == 'threshold' else None
        
            mlp.sparse_fns = nn.ModuleDict({
                'gate_up': SparsifyFn(mlp.distrs['gate_up'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(mlp)),
                'down': SparsifyFn(mlp.distrs['down'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(mlp)),
            })
        else:
            mlp.distrs['gate'] = Distribution(file_path, hidden_type='gate') if mask_by == 'threshold' else None
            mlp.distrs['up'] = Distribution(file_path, hidden_type='up') if mask_by == 'threshold' else None
            mlp.distrs['down'] = Distribution(file_path, hidden_type='down') if mask_by == 'threshold' else None
        
            mlp.sparse_fns = nn.ModuleDict({
                'gate': SparsifyFn(mlp.distrs['gate'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(mlp)),
                'up': SparsifyFn(mlp.distrs['up'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(mlp)),
                'down': SparsifyFn(mlp.distrs['down'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(mlp)),
            })
    elif sparse_mode == 'teal':
        mlp.distrs = {}
        mlp.distrs['h1'] = Distribution(file_path, hidden_type='h1') if mask_by == 'threshold' else None
        mlp.distrs['h2'] = Distribution(file_path, hidden_type='h2') if mask_by == 'threshold' else None

        if 'phi' in mlp.__class__.__name__.lower():
            mlp.sparse_fns = nn.ModuleDict({
                'gate_up': SparsifyFn(mlp.distrs['h1'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(mlp)),
                'down': SparsifyFn(mlp.distrs['h2'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(mlp)),
            })
        else:
            mlp.sparse_fns = nn.ModuleDict({
                'gate': SparsifyFn(mlp.distrs['h1'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(mlp)),
                'up': SparsifyFn(mlp.distrs['h1'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(mlp)),
                'down': SparsifyFn(mlp.distrs['h2'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(mlp)),
            })
    
def _mlp_forward(self, x, activation_module=None):
    if hasattr(self, 'config') and self.config.pretraining_tp > 1:
        # TODO: UNTESTED

        assert 1 == 0, "Pretraining TP > 1 not implemented yet"
    else:       
        if self.grabbing_mode:
            if self.sparse_mode == 'wina':
                self.activation_module.grab_activations(x * self.gate_norm_by_column.to(x.device), 'gate')
                self.activation_module.grab_activations(x * self.up_norm_by_column.to(x.device), 'up')
            elif self.sparse_mode == 'teal':
                self.activation_module.grab_activations(x, 'h1')
            
            intermediate_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            
            if self.sparse_mode == 'wina':
                self.activation_module.grab_activations(intermediate_states * self.down_norm_by_column.to(intermediate_states.device), 'down')
            elif self.sparse_mode == 'teal':
                self.activation_module.grab_activations(intermediate_states, 'h2')
            
            down_proj = self.down_proj(intermediate_states)
        else:                
            if self.sparse_mode == 'wina':
                norm_dtype, x_dtype = self.gate_norm_by_column.dtype, x.dtype
                x_gate = x * self.sparse_fns['gate'](x.to(norm_dtype) * self.gate_norm_by_column.to(x.device)).to(x_dtype)
                x_up = x * self.sparse_fns['up'](x.to(norm_dtype) * self.up_norm_by_column.to(x.device)).to(x_dtype)
            elif self.sparse_mode == 'teal':
                x_gate = x * self.sparse_fns['gate'](x)
                x_up = x * self.sparse_fns['up'](x)

            intermediate_states = self.act_fn(self.gate_proj(x_gate)) * self.up_proj(x_up)
            
            if self.sparse_mode == 'wina':
                intermediate_states = intermediate_states * self.sparse_fns['down'](intermediate_states.to(norm_dtype) * self.down_norm_by_column.to(intermediate_states.device)).to(x_dtype)
            elif self.sparse_mode == 'teal':
                intermediate_states = intermediate_states * self.sparse_fns['down'](intermediate_states)

            down_proj = self.down_proj(intermediate_states)
            
    return down_proj

def _mlp_forward_phi(self, hidden_states, activation_module=None):
    if self.grabbing_mode:
        if self.sparse_mode == 'wina':
            self.activation_module.grab_activations(hidden_states * self.gate_up_norm_by_column.to(hidden_states.device), 'gate_up')
        elif self.sparse_mode == 'teal':
            self.activation_module.grab_activations(hidden_states, 'h1')
        
        up_states = self.gate_up_proj(hidden_states)
        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * self.activation_fn(gate)
                
        if self.sparse_mode == 'wina':
            self.activation_module.grab_activations(up_states * self.down_norm_by_column.to(up_states.device), 'down')
        elif self.sparse_mode == 'teal':
            self.activation_module.grab_activations(up_states, 'h2')
        
        down_proj = self.down_proj(up_states)
    else:                
        if self.sparse_mode == 'wina':
            norm_dtype = self.gate_up_norm_by_column.dtype
            x_dtype = hidden_states.dtype
            hidden_states = hidden_states * self.sparse_fns['gate_up'](hidden_states.to(norm_dtype) * self.gate_up_norm_by_column.to(hidden_states.device)).to(x_dtype)
        elif self.sparse_mode == 'teal':
            hidden_states = hidden_states * self.sparse_fns['gate_up'](hidden_states)

        up_states = self.gate_up_proj(hidden_states)
        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * self.activation_fn(gate)
        
        if self.sparse_mode == 'wina':
            up_states = up_states * self.sparse_fns['down'](up_states.to(norm_dtype) * self.down_norm_by_column.to(up_states.device)).to(x_dtype)
        elif self.sparse_mode == 'teal':
            up_states = up_states * self.sparse_fns['down'](up_states)

        down_proj = self.down_proj(up_states)

    return down_proj