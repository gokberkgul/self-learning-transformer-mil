import torch
import torch.nn as nn
from torchvision import transforms
import math
from functools import partial

from utils import trunc_normal_


class PatchEmbed(nn.Module):
    """Given an image, divides into patches, applies embedding and flattens.

    Parameters
    ----------
    img_size : int
        Image size, assumed to be square. If non-square image is passed, cropping is applied first.
    patch_size : int
        Patch size, assumed to be square.
    channel_size : int
        Channel size of the image.
    embed_dim : int
        Dimension of embeddings. Kept fixed throughout the layers.

    Attributes
    ----------
    grid_size : int
        Image will be divided into (grid_size, grid_size) patches
    num_patches : int
        Number of patches generated from the image.
    proj : nn.Conv2d
        Divides image into patches and applies embedding.
    transform : nn.Sequential
        Center crops the image if dimensions of the input doesn't match

    """
    def __init__(self, img_size=224, patch_size=8, channel_size=3, embed_dim=384):
        """Constructor"""
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size**2

        self.proj = nn.Conv2d(channel_size, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.transform = torch.nn.Sequential(transforms.CenterCrop(img_size),)

    def forward(self, x):
        """Forward pass"""
        num_samples, channel_size, H, W = x.shape
        if H != self.img_size or W != self.img_size:
            import warnings
            warnings.warn("There was a input mismatch, center crop will be applied", category=RuntimeWarning)
            x = self.transform(x)  #: (num_samples, channel_size, img_size, img_size)
        x = self.proj(x)  #: (num_samples, embed_dim, grid_size, grid_size)
        x = x.reshape(num_samples, self.embed_dim, self.num_patches)
        x = x.transpose(1, 2)  #: (num_samples, num_patches, embed_dim)
        return x


class Attention(nn.Module):
    """Attention Module of the Vision Transformer

    Parameters
    ----------
    embed_dim : int
        Dimension of embeddings. Kept fixed throughout the layers.
    num_heads : int
        Number of heads of Attention
    qkv_bias : bool
        Flag for whether QKV Linear Transform will contain bias term or not.
    attn_drop : float
        Dropout probability of first linear projection.
    proj_drop : float
        Dropout probability of last linear projection.

    Attributes
    ----------
    head_dim : int
        Dimension of the attention head.
    scale : float
        Normalizing constant of softmax function.
    qkv : nn.Linear
        Linear projection for the Query, Key and Value matrices.
    proj : nn.Linear
        Takes concatanated attention heads and maps into new space.
    attn_drop, proj_drop : nn.Dropout
        Dropout layers.

    """
    def __init__(self, embed_dim, num_heads=6, qkv_bias=False, attn_drop=.0, proj_drop=.0):
        """Constructor"""
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = (embed_dim // num_heads)
        self.scale = self.head_dim ** (-0.5)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """Forward pass"""
        num_samples, num_tokens, embed_dim = x.shape  #: (num_samples, num_tokens = num_patches + cls_token, embed_dim)
        if embed_dim != self.embed_dim:
            raise ValueError("Dimension mismatch")
        qkv = self.qkv(x)  #: (num_samples, num_tokens, 3 * embed_dim)
        qkv = qkv.reshape(num_samples, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  #: (3, num_samples, num_heads, num_tokens, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  #: (num_samples, num_heads, num_tokens, head_dim)

        k_T = k.transpose(-2, -1)  #: (num_samples, num_heads, head_dim, num_tokens)
        attn = (q @ k_T) * self.scale  #: (num_samples, num_heads, num_tokens, num_tokens)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v  #: (num_samples, num_heads, num_tokens, head_dim)
        x = x.transpose(1, 2)  #: (num_samples, num_tokens, num_heads, head_dim)
        x = x.reshape(num_samples, num_tokens, embed_dim)
        x = self.proj(x)  #: (num_samples, num_tokens, embed_dim)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP Module of the Vision Transformer

    Parameters
    ----------
    in_features : int
        Dimension of input layer.
    hidden_features : int
        Dimension of hidden layer.
    out_features : int
        Dimension of output layer.
    act_layer : nn.* (Non-linear Activations)
        Activation function of hidden layer.
    drop : float
        Dropout probability of last linear projection.

    Attributes
    ----------
    fc1, fc2 : nn.Linear
        Linear layers.
    drop : nn.Dropout
        Dropout layer.

    """
    def __init__(self, in_features, act_layer=nn.GELU, drop=.0, hidden_features=None, out_features=None,):
        """Constructor"""
        super().__init__()
        out_features = out_features or in_features  #: If no out_features passed, set it to in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """Forward pass"""
        x = self.fc1(x)  #: (n_samples, n_patches + 1, hidden_features)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)  #: (n_samples, n_patches + 1, embed_dim)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Stochastic Path

    Parameters
    ----------
    drop_prob : float
        Probabilty of dropping the path.

    """
    def __init__(self, drop_prob=.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """Forward pass"""
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        x = x.div(keep_prob) * random_tensor
        return x


class Block(nn.Module):
    """Transformer encoder block

    Parameters
    ----------
    embed_dim : int
        Dimension of embeddings. Kept fixed throughout the layers.
    num_heads : int
        Number of heads of Attention.
    mlp_ratio : float
        Ratio between input and hidden layer of the MLP module.
    qkv_bias : bool
        Flag for whether QKV Linear Transform will contain bias term or not.
    proj_drop : float
        Dropout probability of MLP layers and last linear projection of Attention.
    attn_drop : float
        Dropout probability of first linear projection of Attention.
    drop_prob : float
        Probabilty of dropping the path.
    act_layer : nn.* (Non-linear Activations)
        Activation function of hidden layer.
    norm_layer : nn.* (Normalization Layers)
        Normalization applied to data

    Attributes
    ----------
    attn : Attention
        Attention block.
    drop_path : DropPath
        DropPath block.
    mlp : MLP
        MLP block.

    """
    def __init__(self, embed_dim, num_heads=6, mlp_ratio=4., qkv_bias=False, proj_drop=.0, attn_drop=.0,
                 drop_prob=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """Constructor"""
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.attn = Attention(embed_dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                              proj_drop=proj_drop)
        self.drop_path = DropPath(drop_prob)
        self.norm2 = norm_layer(embed_dim)
        self.mlp = MLP(in_features=embed_dim, hidden_features=int(mlp_ratio * embed_dim), out_features=None,
                       act_layer=act_layer, drop=proj_drop)

    def forward(self, x):
        """Forward pass"""
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer

    Parameters
    ----------
    img_size : int
        Image size, assumed to be square. If non-square image is passed, cropping is applied first.
    patch_size : int
        Patch size, assumed to be square.
    channel_size : int
        Channel size of the image.
    embed_dim : int
        Dimension of embeddings. Kept fixed throughout the layers.
    num_classes : int
        Number of classes in dataset. If left as 0, classifier head will be identity function.
    depth : int
        Number of blocks in Transformer.
    num_heads : int
        Number of heads of Attention.
    qkv_bias : bool
        Flag for whether QKV Linear Transform will contain bias term or not.
    attn_drop : float
        Dropout probability of first linear projection.
    proj_drop : float
        Dropout probability of last linear projection.
    mlp_ratio : float
        Ratio between input and hidden layer of the MLP module.
    drop_prob : float
        Probabilty of dropping the path.
    act_layer : nn.* (Non-linear Activations)
        Activation function of hidden layer.
    norm_layer : nn.* (Normalization Layers)
        Normalization applied to data.

    Attributes
    ----------
    patch_embed : PatchEmbed
        Patch embedding layer.
    cls_token : nn.Parameter
        Learnable CLS token.
    pos_embed : nn.Parameter
        Learnable positional embeddings.
    pos_drop : nn.Dropout
        Dropout layer for position embeddings.
    blocks : nn.ModuleList
        List of Blocks with length depth.
    norm : nn.* (Normalization Layers)
        Normalization layer.
    head : nn.Linear
        Classifier head of Transformer. Left as identity if num_classes is zero.

    """
    def __init__(self, img_size=224, patch_size=8, channel_size=3, embed_dim=384, num_classes=0, depth=12,
                 num_heads=6, qkv_bias=False, attn_drop=0., proj_drop=0., mlp_ratio=4., drop_prob=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """Constructor"""
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, channel_size=channel_size,
                                      embed_dim=embed_dim)  #: Patch embedding module

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=proj_drop)

        dpr = [x.item() for x in torch.linspace(0, drop_prob, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, proj_drop=proj_drop,
                  attn_drop=attn_drop, drop_prob=dpr[i], act_layer=act_layer, norm_layer=norm_layer)
            for i in range(depth)
        ])  #: Transformer blocks
        self.norm = norm_layer(embed_dim)  #: Last normalizing layer

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()  #: Classifier on top
        self.apply(self.init_weights)

    def init_weights(self, m):
        """Initializes weights of layers in the Transformer"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, W, H):
        """Interpolates positional embeddings"""
        num_patches = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if num_patches == N and W == H:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = W // self.patch_embed.patch_size
        h0 = H // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        """Divides image into patches, appends cls token and adds positional embeddings to patches"""
        num_samples, channel_size, H, W = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(num_samples, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, W, H)

        return self.pos_drop(x)

    def forward(self, x):
        """Forward pass"""
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]  #: Return cls token
    
    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


## Below here is copy-pasted from https://github.com/facebookresearch/dino

def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
