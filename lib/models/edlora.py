from cmath import log
import math
from tkinter import N

import torch
import torch.nn as nn
from diffusers.models.attention_processor import AttnProcessor
from diffusers.utils.import_utils import is_xformers_available
import torch.nn.functional as F
import copy
if is_xformers_available():
    import xformers
from torch import einsum

def remove_edlora_unet_attention_forward(unet):
    def change_forward(unet):  # omit proceesor in new diffusers
        for name, layer in unet.named_children():
            if layer.__class__.__name__ == 'Attention' and name == 'attn2':
                layer.set_processor(AttnProcessor())
            else:
                change_forward(layer)
    change_forward(unet)


class EDLoRA_Control_AttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """
    def __init__(self, cross_attention_idx, place_in_unet, controller, attention_op=None):
        self.cross_attention_idx = cross_attention_idx
        self.place_in_unet = place_in_unet
        self.controller = controller
        self.attention_op = attention_op

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        if encoder_hidden_states is None:
            is_cross = False
            encoder_hidden_states = hidden_states
        else:
            is_cross = True
            if len(encoder_hidden_states.shape) == 4:  # multi-layer embedding
                encoder_hidden_states = encoder_hidden_states[:, self.cross_attention_idx, ...]
            else:  # single layer embedding
                encoder_hidden_states = encoder_hidden_states

        assert not attn.norm_cross

        batch_size, sequence_length, _ = encoder_hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        if is_xformers_available() and not is_cross:
            hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
            hidden_states = hidden_states.to(query.dtype)
        else:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            attention_probs = self.controller(attention_probs, is_cross, self.place_in_unet)
            hidden_states = torch.bmm(attention_probs, value)

        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class EDLoRA_AttnProcessor:
    def __init__(self, cross_attention_idx, attention_op=None):
        self.attention_op = attention_op
        self.cross_attention_idx = cross_attention_idx

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            if len(encoder_hidden_states.shape) == 4:  # multi-layer embedding
                encoder_hidden_states = encoder_hidden_states[:, self.cross_attention_idx, ...]
            else:  # single layer embedding
                encoder_hidden_states = encoder_hidden_states

        assert not attn.norm_cross

        batch_size, sequence_length, _ = encoder_hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        if is_xformers_available():
            hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
            hidden_states = hidden_states.to(query.dtype)
        else:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)

        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class EDLoRA_FusionAttnProcessor:
    def __init__(self, cross_attention_idx, classifier_weights=None, task_indexes=[]):
        self.cross_attention_idx = cross_attention_idx
        self.classifier_weights=classifier_weights
        if self.classifier_weights is not None:
            self.classifier_weights=self.classifier_weights/self.classifier_weights.norm(p=2, dim=-1, keepdim=True)
        self.task_indexes=task_indexes

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        **cross_attention_kwargs
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            if len(encoder_hidden_states.shape) == 4:  # multi-layer embedding
                encoder_hidden_states = encoder_hidden_states[:, self.cross_attention_idx, ...]
            else:  # single layer embedding
                encoder_hidden_states = encoder_hidden_states
        assert not attn.norm_cross
        batch_size, sequence_length, _ = encoder_hidden_states.shape

        id = cross_attention_kwargs['lora_id']

        if self.classifier_weights is not None:
            hidden_states_norm=encoder_hidden_states.chunk(2)[1]
            hidden_states_norm=hidden_states_norm / hidden_states_norm.norm(p=2, dim=-1, keepdim=True)
            classifier_weights=self.classifier_weights.unsqueeze(0).repeat(hidden_states_norm.shape[0],1,1)
            lora_weights=einsum('b w n, b c n -> b w c', classifier_weights, hidden_states_norm)
            lora_weights,_=torch.max(lora_weights, dim=-1)
            weights=[]
            for j in range(1,len(self.task_indexes)):
                mean=torch.mean(lora_weights[:,self.task_indexes[j-1]:self.task_indexes[j]], dim=-1)
                weights.append(mean.unsqueeze(1))
            weights=torch.pow(torch.cat(weights,dim=1),4)
            weights=weights/weights.norm(p=1, dim=-1, keepdim=True)
            weights=torch.cat((weights,weights),dim=0)
            weights=weights.to(hidden_states.device)
        else:
            weights=None

        
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, weights, lora_id= id)
        key = attn.to_k(encoder_hidden_states, weights, lora_id= id)
        value = attn.to_v(encoder_hidden_states, weights, lora_id= id)

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        if is_xformers_available():
            hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
            hidden_states = hidden_states.to(query.dtype)
        else:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)

        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, weights, lora_id= id)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class EDLoRA_Attend_AttnProcessor:
    def __init__(self,attnstore, cross_attention_idx, place_in_unet, attention_op=None):
        self.attention_op = attention_op
        self.cross_attention_idx = cross_attention_idx
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
            
        is_cross = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            if len(encoder_hidden_states.shape) == 4:  # multi-layer embedding
                encoder_hidden_states = encoder_hidden_states[:, self.cross_attention_idx, ...]
            else:  # single layer embedding
                encoder_hidden_states = encoder_hidden_states

        assert not attn.norm_cross

        batch_size, sequence_length, _ = encoder_hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        # if is_xformers_available():
        #     hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
        #     hidden_states = hidden_states.to(query.dtype)
        # else:
        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        self.attnstore(attention_probs, is_cross, self.place_in_unet, hidden_states.shape[1])

        hidden_states = torch.bmm(attention_probs, value)

        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

def revise_edlora_unet_attention_forward(unet):
    def change_forward(unet, count):
        for name, layer in unet.named_children():
            if layer.__class__.__name__ == 'Attention' and 'attn2' in name:
                layer.set_processor(EDLoRA_AttnProcessor(count))
                count += 1
            else:
                count = change_forward(layer, count)
        return count

    # use this to ensure the order
    cross_attention_idx = change_forward(unet.down_blocks, 0)
    cross_attention_idx = change_forward(unet.mid_block, cross_attention_idx)
    cross_attention_idx = change_forward(unet.up_blocks, cross_attention_idx)
    print(f'Number of attention layer registered {cross_attention_idx}')


def revise_edlora_unet_attention_controller_forward(unet, controller):
    class DummyController:
        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def change_forward(unet, count, place_in_unet):
        for name, layer in unet.named_children():
            if layer.__class__.__name__ == 'Attention' and 'attn2' in name:  # only register controller for cross-attention
                layer.set_processor(EDLoRA_Control_AttnProcessor(count, place_in_unet, controller))
                count += 1
            else:
                count = change_forward(layer, count, place_in_unet)
        return count

    # use this to ensure the order
    cross_attention_idx = change_forward(unet.down_blocks, 0, 'down')
    cross_attention_idx = change_forward(unet.mid_block, cross_attention_idx, 'mid')
    cross_attention_idx = change_forward(unet.up_blocks, cross_attention_idx, 'up')
    print(f'Number of attention layer registered {cross_attention_idx}')
    controller.num_att_layers = cross_attention_idx


class LoRALinearLayer(nn.Module):
    def __init__(self, name, original_module, rank=4, alpha=1):
        super().__init__()

        self.name = name

        if original_module.__class__.__name__ == 'Conv2d':
            in_channels, out_channels = original_module.in_channels, original_module.out_channels
            self.lora_down = torch.nn.Conv2d(in_channels, rank, (1, 1), bias=False)
            self.lora_up = torch.nn.Conv2d(rank, out_channels, (1, 1), bias=False)
        else:
            in_features, out_features = original_module.in_features, original_module.out_features
            self.lora_down = nn.Linear(in_features, rank, bias=False)
            self.lora_up = nn.Linear(rank, out_features, bias=False)

        self.register_buffer('alpha', torch.tensor(alpha))

        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        self.original_forward = original_module.forward
        original_module.forward = self.forward

    def forward(self, hidden_states):
        hidden_states = self.original_forward(hidden_states) + self.alpha * self.lora_up(self.lora_down(hidden_states))
        return hidden_states

class MultiLoRALinearLayer(nn.Module):
    def __init__(self, name, original_module, weights, alpha=1.0, rank=4, method='none'):
        super().__init__()

        self.name = name
        self.method = method
        self.compose=len(weights)
        self.lora_down=nn.ModuleList([])
        self.lora_up=nn.ModuleList([])

        for i in range(len(weights)):
            if original_module.__class__.__name__ == 'Conv2d':
                in_channels, out_features= original_module.in_channels, original_module.out_channels
                lora_down = torch.nn.Conv2d(in_channels, rank, (1, 1), bias=False)
                lora_up = torch.nn.Conv2d(rank, out_features, (1, 1), bias=False)
            else:
                in_features, out_features = original_module.in_features, original_module.out_features
                lora_down = nn.Linear(in_features, rank, bias=False).requires_grad_(False)
                lora_down.weight.data.copy_(weights[i][name+'.lora_down.weight'])
                lora_up = nn.Linear(rank, out_features, bias=False).requires_grad_(False)
                lora_up.weight.data.copy_(weights[i][name+'.lora_up.weight'])
            self.lora_down.append(lora_down)
            self.lora_up.append(lora_up)

        self.alpha=alpha
        # if gate_weithts is not None:
        self.original_forward = original_module.forward
        original_module.forward = self.forward

    def forward(self, hidden_states, weights=None, lora_id = None):
        lora_outputs=[]
        for i in range(self.compose):
            lora_output=self.alpha*self.lora_up[i](self.lora_down[i](hidden_states)).unsqueeze(1)
            lora_outputs.append(lora_output)
        lora_outputs=torch.cat(lora_outputs,dim=1)
        weights=weights.unsqueeze(2).unsqueeze(3)
        lora_outputs=torch.sum(weights*lora_outputs,dim=1)
        hidden_states = self.original_forward(hidden_states) + lora_outputs
        return hidden_states

    def init_gate_function(self, dim, weights=None):
        self.gate=nn.Linear(dim, self.compose, bias=False)
        torch.nn.init.ones_(self.gate.weight)
        self.init_gate=True


class LagrangeTerm(nn.Module):
    def __init__(self, weights, lr=0.0001, rank=4, alpha=1):
        super().__init__()
        self.task_order=len(weights)
        weights=list(weights[0].values())
        self.lr=lr
        self.share=nn.ParameterList()
        for weight in weights:
            share=nn.Parameter(torch.zeros(weight.shape[1],weight.shape[0]))
            self.share.append(share)

        self.mapping=nn.ModuleList()
        for j in range(5):
            mappings=nn.ParameterList()
            for weight in weights:
                mapping=nn.Parameter(torch.zeros(weight.shape[1],weight.shape[0]))
                mappings.append(mapping)
            self.mapping.append(mappings)

        self.lagrangemulti=nn.ModuleList()
        for j in range(5):
            lagrangemultis=nn.ParameterList()
            for weight in weights:
                lagrangemulti=nn.Parameter(torch.zeros(weight.shape[1],weight.shape[0]))
                lagrangemultis.append(lagrangemulti)
            self.lagrangemulti.append(lagrangemultis)

        self.affine_all=nn.ModuleList()
        for j in range(5):
            affines=nn.ParameterList()
            for i in range(len(weights)):
                affine=nn.Parameter(torch.zeros(rank,rank))
                affines.append(affine)
            self.affine_all.append(affines)
        
    def forward(self, weight, order, task_id):
        share=torch.matmul(self.affine_all[task_id][order], self.share[order])
        loss = torch.abs(weight-share).mean()

        return loss
    
    def lag_backward(self, weight, order, task_id):

        #update affine
        item1=torch.matmul(weight-self.mapping[task_id][order], self.share[order].t())
        item2=torch.matmul(self.lagrangemulti[task_id][order], self.share[order].t())
        item3=torch.pinverse(torch.matmul(self.share[order], self.share[order].t()))
        grad1=torch.matmul(item1-item2, item3)
        self.affine_all[task_id][order]= grad1

        #update share
        item1=torch.pinverse(torch.matmul(self.affine_all[task_id][order].t(), self.affine_all[task_id][order]))
        item2=torch.matmul(self.affine_all[task_id][order].t(), weight-self.mapping[task_id][order])
        item3=torch.matmul(self.affine_all[task_id][order].t(), self.lagrangemulti[task_id][order])
        grad2=torch.matmul(item1, item2-item3)
        self.share[order]= grad2

        #update mapping
        grad3=0.1*(weight-torch.matmul(self.affine_all[task_id][order], self.share[order])-self.lagrangemulti[task_id][order])
        self.mapping[task_id][order]=grad3

        #update lagrangemulti
        grad4=weight-torch.matmul(self.affine_all[task_id][order], self.share[order])-self.mapping[task_id][order]
        self.lagrangemulti[task_id][order]=grad4



