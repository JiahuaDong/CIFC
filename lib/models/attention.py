from diffusers.utils.import_utils import is_xformers_available
import math
from einops import rearrange
from torch import einsum
if is_xformers_available():
    import xformers
import abc
import torch
from typing import List
import numpy as np
from PIL import Image

class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    # @property
    # def num_uncond_att_layers(self):
    #     return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str, seq_lenth: int):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            self.forward(attn, is_cross, place_in_unet, seq_lenth)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = 16
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str, seq_lenth: int):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if seq_lenth == 512:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        self.attention_store = self.step_store
        if self.save_global_store:
            with torch.no_grad():
                if len(self.global_store) == 0:
                    self.global_store = self.step_store
                else:
                    for key in self.global_store:
                        for i in range(len(self.global_store[key])):
                            self.global_store[key][i] += self.step_store[key][i].detach()
        self.step_store = self.get_empty_store()
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def get_average_global_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.global_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}

    def __init__(self, save_global_store=False):
        '''
        Initialize an empty AttentionStore
        :param step_index: used to visualize only a specific step in the diffusion process
        '''
        super(AttentionStore, self).__init__()
        self.save_global_store = save_global_store
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}
        self.curr_step_index = 0
        self.num_uncond_att_layers = 0

class RegionT2I_AttnProcessor:
    def __init__(self,attnstore, cross_attention_idx, place_in_unet, attention_op=None):
        self.attention_op = attention_op
        self.cross_attention_idx = cross_attention_idx
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet


    def region_rewrite(self, attn, hidden_states, query, region_list, height, width):

        def get_region_mask(region_list, feat_height, feat_width):
            exclusive_mask = torch.zeros((feat_height, feat_width))
            for region in region_list:
                start_h, start_w, end_h, end_w = region[-1]
                start_h, start_w, end_h, end_w = math.ceil(start_h * feat_height), math.ceil(
                    start_w * feat_width), math.floor(end_h * feat_height), math.floor(end_w * feat_width)
                exclusive_mask[start_h:end_h, start_w:end_w] += 1
            return exclusive_mask

        dtype = query.dtype
        seq_lens = query.shape[1]
        downscale = math.sqrt(height * width / seq_lens)

        # 0: context >=1: may be overlap
        feat_height, feat_width = int(height // downscale), int(width // downscale)
        region_mask = get_region_mask(region_list, feat_height, feat_width)

        query = rearrange(query, 'b (h w) c -> b h w c', h=feat_height, w=feat_width)
        hidden_states = rearrange(hidden_states, 'b (h w) c -> b h w c', h=feat_height, w=feat_width)

        new_hidden_state = torch.zeros_like(hidden_states)
        new_hidden_state[:, region_mask == 0, :] = hidden_states[:, region_mask == 0, :]

        replace_ratio = 1.0
        new_hidden_state[:, region_mask != 0, :] = (1 - replace_ratio) * hidden_states[:, region_mask != 0, :]
        
        attention_region_list=[]
        for region in region_list:
            region_key, region_value, region_box = region

            if attn.upcast_attention:
                query = query.float()
                region_key = region_key.float()

            start_h, start_w, end_h, end_w = region_box
            start_h, start_w, end_h, end_w = math.ceil(start_h * feat_height), math.ceil(
                start_w * feat_width), math.floor(end_h * feat_height), math.floor(end_w * feat_width)

            attention_region = einsum('b h w c, b n c -> b h w n', query[:, start_h:end_h, start_w:end_w, :], region_key) * attn.scale
            if attn.upcast_softmax:
                attention_region = attention_region.float()

            attention_region = attention_region.softmax(dim=-1)
            attention_region = attention_region.to(dtype)
            attention_region_list.append(attention_region)
            hidden_state_region = einsum('b h w n, b n c -> b h w c', attention_region, region_value)
            new_hidden_state[:, start_h:end_h, start_w:end_w, :] += \
                replace_ratio * (hidden_state_region / (
                    region_mask.reshape(
                        1, *region_mask.shape, 1)[:, start_h:end_h, start_w:end_w, :]
                ).to(query.device))

        new_hidden_state = rearrange(new_hidden_state, 'b h w c -> b (h w) c')
        return new_hidden_state, attention_region_list

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **cross_attention_kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            is_cross = False
            encoder_hidden_states = hidden_states
        else:
            is_cross = True

            if len(encoder_hidden_states.shape) == 4:  # multi-layer embedding
                encoder_hidden_states = encoder_hidden_states[:, self.cross_attention_idx, ...]
            else:
                encoder_hidden_states = encoder_hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        if is_xformers_available() and not is_cross:
            hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
            hidden_states = hidden_states.to(query.dtype)
        else:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)

        if is_cross:
            region_list = []
            for region in cross_attention_kwargs['region_list']:
                if len(region[0].shape) == 4:
                    region_key = attn.to_k(region[0][:, self.cross_attention_idx, ...])
                    region_value = attn.to_v(region[0][:, self.cross_attention_idx, ...])
                else:
                    region_key = attn.to_k(region[0])
                    region_value = attn.to_v(region[0])
                region_key = attn.head_to_batch_dim(region_key)
                region_value = attn.head_to_batch_dim(region_value)
                region_list.append((region_key, region_value, region[1]))

            hidden_states, attention_region_list = self.region_rewrite(
                attn=attn,
                hidden_states=hidden_states,
                query=query,
                region_list=region_list,
                height=cross_attention_kwargs['height'],
                width=cross_attention_kwargs['width'])

            self.attnstore(attention_region_list, is_cross, self.place_in_unet, sequence_length)

        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def aggregate_attention(controller: AttentionStore,
                        res: int,
                        from_where: List[str],
                        is_cross: bool,
                        select: int) -> torch.Tensor:
    """ Aggregates the attention across the different layers and heads at the specified resolution. """
    out = []
    attention_maps = controller.get_average_attention()

    num_pixels = res 
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            # if item[0].shape[1] <= num_pixels:
            cross_maps = item
            out.append(cross_maps)
    outs=[]
    region_num=len(out[0])
    for i in range(region_num):
        region=[out[j][i] for j in range(len(out))]
        region= torch.cat(region, dim=0)
        region= region.sum(0) / region.shape[0]
        outs.append(region)
    return outs

def aggregate_attention_box(controller: AttentionStore,
                        res: int,
                        from_where: List[str],
                        is_cross: bool,
                        select: int) -> torch.Tensor:
    """ Aggregates the attention across the different layers and heads at the specified resolution. """
    out = []
    attention_maps = controller.get_average_attention()

    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            # if item.shape[1] == num_pixels:
            cross_maps = rearrange(item, 'b (h w) c -> b h w c', b=item.shape[0], h=res, w=res*2)
            out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out
def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)

def show_self_attention_comp(attention_store, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    view_images(np.concatenate(images, axis=1))