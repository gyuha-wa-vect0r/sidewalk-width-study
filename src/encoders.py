"""Image normalization and frozen DINOv2 and ResNet-18 encoders."""
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from safetensors.torch import load_file

from metrics import sha

DINO_SHA256 = '7a6f7b3b9fa4b8732e707476a03cd6cdce210048582f21aafb7991c17d98e362'
RESNET_SHA256 = 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'


def transform(image, size):
    # Read native pixels directly: never call baseline preprocess (forces256 first).
    image = image.convert('RGB').resize((size,size), Image.Resampling.BICUBIC)
    pad = (-size) % 14
    image = ImageOps.expand(image, (pad//2,pad//2,pad-pad//2,pad-pad//2), fill=0)
    a = np.asarray(image, dtype=np.float32).transpose(2,0,1).copy() / 255
    t = torch.from_numpy(a)
    return (t-t.new_tensor([.485,.456,.406])[:,None,None])/t.new_tensor([.229,.224,.225])[:,None,None]


def convert_hf_state(state):
    """Lossless parameter rename/QKV concatenate to pinned upstream DINOv2."""
    mapped = {}
    consumed = set()
    def take(key):
        consumed.add(key)
        return state[key]
    names = {
        'cls_token':'embeddings.cls_token', 'mask_token':'embeddings.mask_token',
        'register_tokens':'embeddings.register_tokens', 'pos_embed':'embeddings.position_embeddings',
        'patch_embed.proj.weight':'embeddings.patch_embeddings.projection.weight',
        'patch_embed.proj.bias':'embeddings.patch_embeddings.projection.bias',
        'norm.weight':'layernorm.weight', 'norm.bias':'layernorm.bias',
    }
    for dst, src in names.items():
        mapped[dst] = take(src)
    for i in range(12):
        src, dst = f'encoder.layer.{i}.', f'blocks.{i}.'
        for end in ['weight','bias']:
            mapped[dst+'attn.qkv.'+end] = torch.cat([take(src+'attention.attention.'+q+'.'+end) for q in ['query','key','value']])
            mapped[dst+'attn.proj.'+end] = take(src+'attention.output.dense.'+end)
            for norm in ['norm1','norm2']:
                mapped[dst+norm+'.'+end] = take(src+norm+'.'+end)
            for fc in ['fc1','fc2']:
                mapped[dst+'mlp.'+fc+'.'+end] = take(src+'mlp.'+fc+'.'+end)
        for j in [1,2]:
            mapped[dst+f'ls{j}.gamma'] = take(src+f'layer_scale{j}.lambda1')
    if consumed != set(state):
        raise ValueError(f'Unmapped weights: {set(state)-consumed}')
    return mapped


def sdpa_forward(self, x, attn_bias=None):
    if attn_bias is not None:
        raise ValueError('Nested attention unsupported in this pilot')
    b,n,c = x.shape
    q,k,v = self.qkv(x).reshape(b,n,3,self.num_heads,c//self.num_heads).permute(2,0,3,1,4).unbind(0)
    z = F.scaled_dot_product_attention(q,k,v,dropout_p=0.)
    return self.proj_drop(self.proj(z.transpose(1,2).reshape(b,n,c)))


def load_dino(checkpoint, upstream):
    os.environ['XFORMERS_DISABLED'] = '1'
    sys.path.insert(0, str(upstream.resolve()))
    from dinov2.models.vision_transformer import vit_base

    if sha(checkpoint) != DINO_SHA256:
        raise ValueError('Generic pretrained model checksum mismatch')
    model = vit_base(img_size=518, patch_size=14, init_values=1., num_register_tokens=4,
                     block_chunks=0, interpolate_antialias=True, interpolate_offset=0.)
    model.load_state_dict(convert_hf_state(load_file(str(checkpoint))), strict=True)
    model.requires_grad_(False).eval()
    for block in model.blocks:
        block.attn.forward = types.MethodType(sdpa_forward, block.attn)
    return model.cuda()



def encoder_model(name, weights, upstream):
    if name == 'dino':
        return load_dino(weights / 'dinov2.safetensors', upstream)
    from torchvision.models import resnet18

    state = weights / 'resnet18-f37072fd.pth'
    if sha(state) != RESNET_SHA256:
        raise ValueError('ResNet-18 checkpoint checksum mismatch')
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(state, map_location='cpu', weights_only=True), strict=True)
    model.fc = torch.nn.Identity()
    return model.eval().cuda()


@torch.inference_mode()
def encode(name, encoder, images):
    batch = torch.stack([transform(Image.fromarray(img), 224) for img in images]).cuda()
    if name == 'dino':
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = encoder.forward_features(batch)['x_norm_clstoken']
    else:
        output = encoder(batch)
    return output.float().cpu().numpy()
