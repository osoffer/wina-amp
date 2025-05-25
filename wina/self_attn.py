import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'utils'))

import types

import torch
import torch.nn as nn

from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb, repeat_kv
)

from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as apply_rotary_pos_emb_qwen2

from utils.utils import ActivationModule, Distribution, SparsifyFn, get_module_device

from transformers.modeling_flash_attention_utils import _flash_attention_forward

from transformers.utils import logging


logger = logging.get_logger(__name__)


def _monkeypatch_self_attn(self_attn, file_path, grabbing_mode=False, sparse_mode=None, mask_by=None):
    self_attn.forward_old = self_attn.forward

    if 'qwen' in self_attn.__class__.__name__.lower():
        self_attn.forward = types.MethodType(_sdpa_forward_qwen, self_attn)
    elif 'phi' in self_attn.__class__.__name__.lower():
        self_attn.forward = types.MethodType(_sdpa_forward_phi, self_attn)
    else:
        self_attn.forward = types.MethodType(_sdpa_forward, self_attn)

    self_attn.add_sparse_fns = types.MethodType(add_sparse_fns, self_attn)

    self_attn.file_path = file_path
    self_attn.grabbing_mode = grabbing_mode
    
    self_attn.sparse_mode = sparse_mode
    if sparse_mode == 'wina':
        if 'phi' in self_attn.__class__.__name__.lower():
            self_attn.qkv_norm_by_column = self_attn.qkv_proj.weight.norm(dim=0)
            self_attn.o_norm_by_column = torch.tensor(1.)
        else:
            self_attn.q_norm_by_column = torch.tensor(1.)
            self_attn.k_norm_by_column = self_attn.k_proj.weight.norm(dim=0)
            self_attn.v_norm_by_column = torch.tensor(1.)
            self_attn.o_norm_by_column = torch.tensor(1.)
                
    if not grabbing_mode:
        if sparse_mode == 'teal':
            self_attn.distrs = {}
            self_attn.distrs['h1'] = Distribution(self_attn.file_path, hidden_type='h1') if mask_by == 'threshold' else None
            self_attn.distrs['h2'] = Distribution(self_attn.file_path, hidden_type='h2') if mask_by == 'threshold' else None
            if 'phi' in self_attn.__class__.__name__.lower():
                self_attn.sparse_fns = nn.ModuleDict({
                    'qkv': SparsifyFn(self_attn.distrs['h1'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(self_attn)),
                    'o': SparsifyFn(self_attn.distrs['h2'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(self_attn))
                })
            else:
                self_attn.sparse_fns = nn.ModuleDict({
                    'q': SparsifyFn(self_attn.distrs['h1'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(self_attn)),
                    'k': SparsifyFn(self_attn.distrs['h1'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(self_attn)),
                    'v': SparsifyFn(self_attn.distrs['h1'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(self_attn)),
                    'o': SparsifyFn(self_attn.distrs['h2'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(self_attn))
                })
        elif sparse_mode == 'wina':
            self_attn.distrs = {}
            if 'phi' in self_attn.__class__.__name__.lower():
                self_attn.distrs['qkv'] = Distribution(self_attn.file_path, hidden_type='qkv') if mask_by == 'threshold' else None
                self_attn.distrs['o'] = Distribution(self_attn.file_path, hidden_type='o') if mask_by == 'threshold' else None
                
                self_attn.sparse_fns = nn.ModuleDict({
                    'qkv': SparsifyFn(self_attn.distrs['qkv'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(self_attn)),
                    'o': SparsifyFn(self_attn.distrs['o'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(self_attn))
                })
            else:
                self_attn.distrs = {}
                self_attn.distrs['q'] = Distribution(self_attn.file_path, hidden_type='q') if mask_by == 'threshold' else None
                self_attn.distrs['k'] = Distribution(self_attn.file_path, hidden_type='k') if mask_by == 'threshold' else None
                self_attn.distrs['v'] = Distribution(self_attn.file_path, hidden_type='v') if mask_by == 'threshold' else None
                self_attn.distrs['o'] = Distribution(self_attn.file_path, hidden_type='o') if mask_by == 'threshold' else None
                
                self_attn.sparse_fns = nn.ModuleDict({
                    'q': SparsifyFn(self_attn.distrs['q'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(self_attn)),
                    'k': SparsifyFn(self_attn.distrs['k'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(self_attn)),
                    'v': SparsifyFn(self_attn.distrs['v'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(self_attn)),
                    'o': SparsifyFn(self_attn.distrs['o'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(self_attn))
                })
                
    self_attn.activation_module = ActivationModule(self_attn.file_path)

    return self_attn
    
def add_sparse_fns(self, sparsity=0.25, mask_by=None):
    self_attn = self
    self_attn.grabbing_mode = False
    sparse_mode = self_attn.sparse_mode
    file_path = self_attn.file_path
    if sparse_mode == 'teal':
        self_attn.distrs = {}
        self_attn.distrs['h1'] = Distribution(self_attn.file_path, hidden_type='h1') if mask_by == 'threshold' else None
        self_attn.distrs['h2'] = Distribution(self_attn.file_path, hidden_type='h2') if mask_by == 'threshold' else None
        if 'phi' in self_attn.__class__.__name__.lower():
            self_attn.sparse_fns = nn.ModuleDict({
                'qkv': SparsifyFn(self_attn.distrs['h1'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(self_attn)),
                'o': SparsifyFn(self_attn.distrs['h2'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(self_attn))
            })
        else:
            self_attn.sparse_fns = nn.ModuleDict({
                'q': SparsifyFn(self_attn.distrs['h1'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(self_attn)),
                'k': SparsifyFn(self_attn.distrs['h1'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(self_attn)),
                'v': SparsifyFn(self_attn.distrs['h1'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(self_attn)),
                'o': SparsifyFn(self_attn.distrs['h2'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(self_attn))
            })
    elif sparse_mode == 'wina':
        self_attn.distrs = {}
        if 'phi' in self_attn.__class__.__name__.lower():
            self_attn.distrs['qkv'] = Distribution(self_attn.file_path, hidden_type='qkv') if mask_by == 'threshold' else None
            self_attn.distrs['o'] = Distribution(self_attn.file_path, hidden_type='o') if mask_by == 'threshold' else None
            
            self_attn.sparse_fns = nn.ModuleDict({
                'qkv': SparsifyFn(self_attn.distrs['qkv'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(self_attn)),
                'o': SparsifyFn(self_attn.distrs['o'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(self_attn))
            })
        else:
            self_attn.distrs = {}
            self_attn.distrs['q'] = Distribution(self_attn.file_path, hidden_type='q') if mask_by == 'threshold' else None
            self_attn.distrs['k'] = Distribution(self_attn.file_path, hidden_type='k') if mask_by == 'threshold' else None
            self_attn.distrs['v'] = Distribution(self_attn.file_path, hidden_type='v') if mask_by == 'threshold' else None
            self_attn.distrs['o'] = Distribution(self_attn.file_path, hidden_type='o') if mask_by == 'threshold' else None
            
            self_attn.sparse_fns = nn.ModuleDict({
                'q': SparsifyFn(self_attn.distrs['q'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(self_attn)),
                'k': SparsifyFn(self_attn.distrs['k'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(self_attn)),
                'v': SparsifyFn(self_attn.distrs['v'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(self_attn)),
                'o': SparsifyFn(self_attn.distrs['o'], sparse_mode=sparse_mode, mask_by=mask_by).to(get_module_device(self_attn))
            })
    
def _FA2_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask = None, #: Optional[torch.LongTensor] = None,
    position_ids = None, #: Optional[torch.LongTensor] = None,
    past_key_value = None, #: Optional[Cache] = None,
    output_attentions = False, #: bool = False,
    use_cache = False, #: bool = False,
    cache_position = None, #: Optional[torch.LongTensor] = None,
    activation_module = None,
    **kwargs,
): 
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    # MONKEYPATCH HERE
    
    if self.grabbing_mode:
        self.activation_module.grab_activations(hidden_states, 'h1')
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    else: 
        x_q = self.sparse_fns['q'](hidden_states)

        x_k = self.sparse_fns['k'](hidden_states)
        x_v = self.sparse_fns['v'](hidden_states)

        query_states = self.q_proj(x_q)
        key_states = self.k_proj(x_k)
        value_states = self.v_proj(x_v)
        
    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)


    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float16:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        
        print(f"Casting input hidden states to {target_dtype} (this should not be happening)")

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # NOTE: sliding window isn't tested for Mistral, please create an issue if something goes wrong
    # However, we don't ever utilize sequence lengths of more than 4096 for the methodology + evals
    attn_output = _flash_attention_forward(
        query_states, key_states, value_states, attention_mask, q_len, position_ids=position_ids, dropout=dropout_rate, sliding_window=getattr(self, "sliding_window", None), is_causal=True
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()

    # MONKEYPATCH HERE
    if self.grabbing_mode:
        self.activation_module.grab_activations(attn_output, 'h2')
        attn_output = self.o_proj(attn_output)
    else:
        attn_output = self.sparse_fns['o'](attn_output)
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def _sdpa_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask = None, #: Optional[torch.LongTensor] = None,
    position_ids = None, #: Optional[torch.LongTensor] = None,
    past_key_value = None, #: Optional[Cache] = None,
    output_attentions = False, #: bool = False,
    use_cache = False, #: bool = False,
    cache_position = None, #: Optional[torch.LongTensor] = None,
    position_embeddings = None,
    **kwargs,
): 
    bsz, q_len, _ = hidden_states.size()

    # MONKEYPATCH HERE
    if self.grabbing_mode:
        if self.sparse_mode == 'wina':
            self.activation_module.grab_activations(hidden_states * self.q_norm_by_column.to(hidden_states.device), 'q')
            self.activation_module.grab_activations(hidden_states * self.k_norm_by_column.to(hidden_states.device), 'k')
            self.activation_module.grab_activations(hidden_states * self.v_norm_by_column.to(hidden_states.device), 'v')
        elif self.sparse_mode == 'teal':
            self.activation_module.grab_activations(hidden_states, 'h1')
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    else: 
        if self.sparse_mode == 'wina':
            norm_dtype, x_dtype = self.k_norm_by_column.dtype, hidden_states.dtype
            criterion_q = hidden_states.to(norm_dtype) * self.q_norm_by_column.to(hidden_states.device)
            criterion_k = hidden_states.to(norm_dtype) * self.k_norm_by_column.to(hidden_states.device)
            criterion_v = hidden_states.to(norm_dtype) * self.v_norm_by_column.to(hidden_states.device)
            x_q = hidden_states * self.sparse_fns['q'](criterion_q).to(x_dtype)
            x_k = hidden_states * self.sparse_fns['k'](criterion_k).to(x_dtype)
            x_v = hidden_states * self.sparse_fns['v'](criterion_v).to(x_dtype)
        elif self.sparse_mode == 'teal':
            x_q = hidden_states * self.sparse_fns['q'](hidden_states)
            x_k = hidden_states * self.sparse_fns['k'](hidden_states)
            x_v = hidden_states * self.sparse_fns['v'](hidden_states)

        query_states = self.q_proj(x_q)
        key_states = self.k_proj(x_k)
        value_states = self.v_proj(x_v)
        
        
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    
    
    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    
    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )
    
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, -1)
    
    
    # MONKEYPATCH HERE
    if self.grabbing_mode:
        if self.sparse_mode == 'wina':
            self.activation_module.grab_activations(attn_output * self.o_norm_by_column.to(attn_output.device), 'o')
        elif self.sparse_mode == 'teal':
            self.activation_module.grab_activations(attn_output, 'h2')
        attn_output = self.o_proj(attn_output)
    else:        
        if self.sparse_mode == 'wina':
            attn_output = attn_output * self.sparse_fns['o'](attn_output.to(norm_dtype) * self.o_norm_by_column.to(attn_output.device)).to(x_dtype)
        elif self.sparse_mode == 'teal':
            attn_output = attn_output * self.sparse_fns['o'](attn_output)
        attn_output = self.o_proj(attn_output)
            

    return attn_output, None, past_key_value

def _sdpa_forward_qwen(
    self,
    hidden_states: torch.Tensor,
    attention_mask = None, #: Optional[torch.LongTensor] = None,
    position_ids = None, #: Optional[torch.LongTensor] = None,
    past_key_value = None, #: Optional[Cache] = None,
    output_attentions = False, #: bool = False,
    use_cache = False, #: bool = False,
    cache_position = None, #: Optional[torch.LongTensor] = None,
    position_embeddings = None,
    **kwargs,
): 
    bsz, q_len, _ = hidden_states.size()

    # MONKEYPATCH HERE
    
    if self.grabbing_mode:
        if self.sparse_mode == 'wina':
            self.activation_module.grab_activations(hidden_states * self.q_norm_by_column.to(hidden_states.device), 'q')
            self.activation_module.grab_activations(hidden_states * self.k_norm_by_column.to(hidden_states.device), 'k')
            self.activation_module.grab_activations(hidden_states * self.v_norm_by_column.to(hidden_states.device), 'v')
        elif self.sparse_mode == 'teal':
            self.activation_module.grab_activations(hidden_states, 'h1')
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    else: 
        if self.sparse_mode == 'wina':
            norm_dtype = self.q_norm_by_column.dtype
            x_dtype = hidden_states.dtype
            x_q = hidden_states * self.sparse_fns['q'](hidden_states.to(norm_dtype) * self.q_norm_by_column.to(hidden_states.device)).to(x_dtype)
            x_k = hidden_states * self.sparse_fns['k'](hidden_states.to(norm_dtype) * self.k_norm_by_column.to(hidden_states.device)).to(x_dtype)
            x_v = hidden_states * self.sparse_fns['v'](hidden_states.to(norm_dtype) * self.v_norm_by_column.to(hidden_states.device)).to(x_dtype)
        elif self.sparse_mode == 'teal':
            x_q = hidden_states * self.sparse_fns['q'](hidden_states)
            x_k = hidden_states * self.sparse_fns['k'](hidden_states)
            x_v = hidden_states * self.sparse_fns['v'](hidden_states)

        query_states = self.q_proj(x_q)
        key_states = self.k_proj(x_k)
        value_states = self.v_proj(x_v)
        
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    query_states, key_states = apply_rotary_pos_emb_qwen2(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )
    
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, -1)
    
    # MONKEYPATCH HERE
    if self.grabbing_mode:
        if self.sparse_mode == 'wina':
            self.activation_module.grab_activations(attn_output * self.o_norm_by_column.to(attn_output.device), 'o')
        elif self.sparse_mode == 'teal':
            self.activation_module.grab_activations(attn_output, 'h2')
        attn_output = self.o_proj(attn_output)
    else:
        if self.sparse_mode == 'wina':
            norm_dtype = self.q_norm_by_column.dtype
            x_dtype = attn_output.dtype
            attn_output = attn_output * self.sparse_fns['o'](attn_output.to(norm_dtype) * self.o_norm_by_column.to(attn_output.device)).to(x_dtype)
        elif self.sparse_mode == 'teal':
            attn_output = attn_output * self.sparse_fns['o'](attn_output)
        attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value

def _sdpa_forward_phi(
    self,
    hidden_states: torch.Tensor,
    attention_mask = None, #: Optional[torch.LongTensor] = None,
    position_ids = None, #: Optional[torch.LongTensor] = None,
    past_key_value = None, #: Optional[Cache] = None,
    output_attentions = False, #: bool = False,
    use_cache = False, #: bool = False,
    cache_position = None, #: Optional[torch.LongTensor] = None,
    position_embeddings = None,
    **kwargs,
): 
    bsz, q_len, _ = hidden_states.size()

    # MONKEYPATCH HERE
    
    if self.grabbing_mode:
        if self.sparse_mode == 'wina':
            self.activation_module.grab_activations(hidden_states * self.qkv_norm_by_column.to(hidden_states.device), 'qkv')
        elif self.sparse_mode == 'teal':
            self.activation_module.grab_activations(hidden_states, 'h1')
        
        qkv = self.qkv_proj(hidden_states)
    else: 
        if self.sparse_mode == 'wina':
            norm_dtype = self.qkv_norm_by_column.dtype
            x_dtype = hidden_states.dtype
            x_qkv = hidden_states * self.sparse_fns['qkv'](hidden_states.to(norm_dtype) * self.qkv_norm_by_column.to(hidden_states.device)).to(x_dtype)
        elif self.sparse_mode == 'teal':
            x_qkv = hidden_states * self.sparse_fns['qkv'](hidden_states)
            
        qkv = self.qkv_proj(x_qkv)
    
    query_pos = self.num_heads * self.head_dim
    query_states = qkv[..., :query_pos]
    key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
    value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    # MONKEYPATCH HERE
    if self.grabbing_mode:
        if self.sparse_mode == 'wina':
            self.activation_module.grab_activations(attn_output * self.o_norm_by_column.to(attn_output.device), 'o')
        elif self.sparse_mode == 'teal':
            self.activation_module.grab_activations(attn_output, 'h2')
        attn_output = self.o_proj(attn_output)
    else:
        if self.sparse_mode == 'wina':
            norm_dtype = self.qkv_norm_by_column.dtype
            x_dtype = attn_output.dtype
            attn_output = attn_output * self.sparse_fns['o'](attn_output.to(norm_dtype) * self.o_norm_by_column.to(attn_output.device)).to(x_dtype)
        elif self.sparse_mode == 'teal':
            attn_output = attn_output * self.sparse_fns['o'](attn_output)
        attn_output = self.o_proj(attn_output)
            
    return attn_output, None, past_key_value