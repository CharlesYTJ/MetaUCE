import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import warnings
from torch.nn import functional as F


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)  # 输入C为4*dim, 输出为2*dim
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x = self.norm(x)

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)
        x= self.norm(x)

        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
        scale_learnable=True, bias_learnable=True,
        mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
            requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
            requires_grad=bias_learnable)
    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias


class poolc(nn.Module):

    def __init__(self, k=3):
        super(poolc, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=k, stride=3)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        return x


# class dynamic_photon_search(nn.Module):
#     """ CRN (channel Response Normalization) layer
#     """
#
#     def __init__(self, dim):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(1, 1, dim)*1e-5)
#         self.bias = nn.Parameter(torch.zeros(1, 1, dim))
#         self.pool = poolc()
#
#     def forward(self, x):
#         # B, L, C
#         y = self.pool(x)
#         # print('y', y.size())
#         Gx = torch.norm(y, p=2, dim=1, keepdim=True)
#         Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
#         # print('Nx', Nx.size())
#         # return self.gamma * (x * Nx) + self.beta + x
#         return x + Nx*self.weight + self.bias


class CRN(nn.Module):
    """ CRN (channel Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=-1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class new_SPP(nn.Module):
    def __init__(self, dim, k=5):
        super(new_SPP, self).__init__()

        self.crn = CRN(dim)
        self.m = nn.MaxPool1d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            x = x + self.m(x)
            x = x.permute(0, 2, 1)
            x = self.crn(x)
            return x



class MetaMlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0., bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class ConvNeXtV2Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)  # depthwise conv, p=(kernel_size - 1) // 2不会改变输入大小
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):

        input = x
        x = x.permute(0, 2, 1)
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = input + self.drop_path(x)
        return x


class MetaFormerBlock(nn.Module):
    """
    Implementation of one MetaFormer block.
    """
    def __init__(self, dim,
                 token_mixer=nn.Identity, mlp=MetaMlp,
                 norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0.,
                 layer_scale_init_value=None, res_scale_init_value=None
                 ):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()
        
    def forward(self, x):

        x = self.res_scale1(x) + \
            self.layer_scale1(
                self.drop_path1(
                    self.token_mixer(self.norm1(x))
                )
            )
        x = self.res_scale2(x) + \
            self.layer_scale2(
                self.drop_path2(
                    self.mlp(self.norm2(x))
                )
            )
        return x

# class MetaBasicLayer(nn.Module):
#     """ A basic Swin Transformer layer for one stage.
#     Args:
#         dim (int): Number of input channels.
#         input_resolution (tuple[int]): Input resolution.
#         depth (int): Number of blocks.
#         num_heads (int): Number of attention heads.
#         window_size (int): Local window size.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
#         norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
#         downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
#         use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
#     """
#     def __init__(self, dim, depth, block2_depth, input_resolution, token_mixer=new_SPP, mlp=MetaMlp, Encoder_blocks=[MetaFormerBlock, ConvNeXtV2Block],
#                     drop=0., drop_path=0., drop_path2=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
#
#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.depth = depth
#         self.block2_depth = block2_depth
#         self.use_checkpoint = use_checkpoint
#
#         # build blocks
#         self.blocks = nn.ModuleList([
#             Encoder_blocks[0](dim=dim,
#                                  token_mixer=token_mixer,
#                                  mlp=mlp,
#                                  drop=drop,
#                                  layer_scale_init_value=None,
#                                  res_scale_init_value=None,
#                                  drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#                                  norm_layer=norm_layer)
#             for i in range(depth)])
#
#         self.blocks2 = nn.ModuleList([
#             Encoder_blocks[1](dim=dim,
#                   drop_path=drop_path2[i] if isinstance(drop_path2, list) else drop_path2)
#             for i in range(block2_depth)])
#
#         # patch merging layer
#         if downsample is not None:
#             self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
#         else:
#             self.downsample = None
#
#     def forward(self, x):
#
#         for blk in self.blocks:
#             if self.use_checkpoint:
#                 x1 = checkpoint.checkpoint(blk, x)
#             else:
#                 x1 = blk(x)
#
#         for blk in self.blocks2:
#             if self.use_checkpoint:
#                 x2 = checkpoint.checkpoint(blk, x)
#             else:
#                 x2 = blk(x)
#         x = x2 + x1
#         # print('x', x.size())
#         if self.downsample is not None:
#             x = self.downsample(x)
#
#         return x
#
#     def extra_repr(self) -> str:
#         return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"
#
#     def flops(self):
#         flops = 0
#         for blk in self.blocks:
#             flops += blk.flops()
#         if self.downsample is not None:
#             flops += self.downsample.flops()
#         return flops

##########################################################################################
class MetaBasicLayer1(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """
    def __init__(self, dim, depth, block2_depth, input_resolution, token_mixer=new_SPP, mlp=MetaMlp,
                 # fetus_blocks_a=[MetaFormerBlock, MetaFormerBlock, MetaFormerBlock, MetaFormerBlock],
                 fetus_blocks_b=[ConvNeXtV2Block, ConvNeXtV2Block, ConvNeXtV2Block, ConvNeXtV2Block],
                 twins_layer_number=4, drop=0., drop_path=0., drop_path2=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.block2_depth = block2_depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        #for number in range(twins_layer_number):
            #self.blocks = nn.ModuleList([
                #fetus_blocks_a[number](dim=dim,
                                     #token_mixer=token_mixer,
                                     #mlp=mlp,
                                     #drop=drop,
                                     #layer_scale_init_value=None,
                                     #res_scale_init_value=None,
                                     #drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                     #norm_layer=norm_layer)
                #for i in range(depth)])
        for number in range(twins_layer_number):
            self.blocks2 = nn.ModuleList([
                fetus_blocks_b[number](dim=dim,
                      drop_path=drop_path2[i] if isinstance(drop_path2, list) else drop_path2)
                for i in range(block2_depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):

        #for blk in self.blocks:
            #if self.use_checkpoint:
                #x1 = checkpoint.checkpoint(blk, x)
            #else:
                #x1 = blk(x)

        #if self.downsample is not None:
            #x1 = self.downsample(x1)

        for blk in self.blocks2:
            if self.use_checkpoint:
                x2 = checkpoint.checkpoint(blk, x)
            else:
                x2 = blk(x)
        #
        if self.downsample is not None:
            x2 = self.downsample(x2)

        # x = x1 + x2
        # print('x', x.size())
        # if self.downsample is not None:
            # x = self.downsample(x)
        return x2

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class MetaBasicLayer2(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """
    def __init__(self, dim, depth, block2_depth, input_resolution, token_mixer=new_SPP, mlp=MetaMlp,
                 fetus_blocks_a=[MetaFormerBlock, MetaFormerBlock, MetaFormerBlock, MetaFormerBlock],
                 fetus_blocks_b=[ConvNeXtV2Block, ConvNeXtV2Block, ConvNeXtV2Block, ConvNeXtV2Block],
                 twins_layer_number=4, drop=0., drop_path=0., drop_path2=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.block2_depth = block2_depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        for number in range(twins_layer_number):
            self.blocks = nn.ModuleList([
                fetus_blocks_a[number](dim=dim,
                                     token_mixer=token_mixer,
                                     mlp=mlp,
                                     drop=drop,
                                     layer_scale_init_value=None,
                                     res_scale_init_value=None,
                                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                     norm_layer=norm_layer)
                for i in range(depth)])
        for number in range(twins_layer_number):
            self.blocks2 = nn.ModuleList([
                fetus_blocks_b[number](dim=dim,
                      drop_path=drop_path2[i] if isinstance(drop_path2, list) else drop_path2)
                for i in range(block2_depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):

        for blk in self.blocks:
            if self.use_checkpoint:
                x1 = checkpoint.checkpoint(blk, x)
            else:
                x1 = blk(x)

        if self.downsample is not None:
            x1 = self.downsample(x1)

        for blk in self.blocks2:
            if self.use_checkpoint:
                x2 = checkpoint.checkpoint(blk, x)
            else:
                x2 = blk(x)

        if self.downsample is not None:
            x2 = self.downsample(x2)

        x = x1 + x2
        # print('x', x.size())
        # if self.downsample is not None:
            # x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops
###########################################################################################

class MetaBasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,  dim, depth, input_resolution,token_mixer=new_SPP, mlp=MetaMlp,
                 drop=0.,drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            MetaFormerBlock(dim=dim,
                                 token_mixer=token_mixer,
                                 mlp=mlp,
                                 drop=drop, 
                                 layer_scale_init_value=None, 
                                 res_scale_init_value=None,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class PatchExpand_two(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x = self.norm(x)

        return x


##########################################################################
class MetaUCE(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        out_chans (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    # 修改前drop_path_rate=0.1
    def __init__(self, img_size=224, patch_size=4, in_chans=3, out_chans=3,
                 embed_dim=96,
                 fetus_blocks_number=[3, 3, 9, 3],
                 depths=[3, 3, 9, 3], block2_depths=[3, 3, 9, 3], depths_decoder=[1, 2, 2, 2],
                 single_layer=True,
                 mlp_ratio=4.,
                 drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()

        print(
            "MetaformerSys expand initial----Encoder1_depths:{};Encoder2_depths:{};depths_decoder:{};drop_path_rate:{};out_chans:{}".format(
                depths, block2_depths,
                depths_decoder, drop_path_rate, out_chans))
        
        self.out_chans = out_chans
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
########################################################################################
        if single_layer:
            # build encoder and bottleneck layers
            self.layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = MetaBasicLayer1(dim=int(embed_dim * 2 ** i_layer),
                                        depth=fetus_blocks_number[i_layer],
                                        block2_depth=block2_depths[i_layer],
                                        input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                          patches_resolution[1] // (2 ** i_layer)),
                                        drop=drop_rate,
                                        drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                        norm_layer=norm_layer,
                                        downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                        use_checkpoint=use_checkpoint)
                self.layers.append(layer)

            numbers = [1, 2]
            self.add_layer = nn.ModuleList()
            self.crn = nn.ModuleList()
            for add in numbers:
                add_up = PatchExpand_two(input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - add)),
                                                           patches_resolution[1] // (2 ** (self.num_layers - 1 - add))),
                                         dim=2 * int(embed_dim * 2 ** (self.num_layers - 1 - add)), dim_scale=2,
                                         norm_layer=norm_layer)
                crn = CRN(dim=int(embed_dim * 2 ** (self.num_layers - 1 - add)))
                self.crn.append(crn)
                self.add_layer.append(add_up)

            # build decoder layers
            self.layers_up = nn.ModuleList()
            self.concat_back_dim = nn.ModuleList()
            for i_layer in range(self.num_layers):
                concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                          int(embed_dim * 2 ** (
                                                  self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
                if i_layer == 0:
                    layer_up = PatchExpand(
                        input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                          patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                        dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
                else:
                    layer_up = MetaBasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                                 depth=depths[(self.num_layers - 1 - i_layer)],
                                                 input_resolution=(
                                                     patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                                     patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                                 drop=drop_rate,
                                                 drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                                     depths[:(self.num_layers - 1 - i_layer) + 1])],
                                                 norm_layer=norm_layer,
                                                 upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                                 use_checkpoint=use_checkpoint)
                self.layers_up.append(layer_up)
                self.concat_back_dim.append(concat_linear)
        else:
            # build encoder and bottleneck layers
            self.layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = MetaBasicLayer2(dim=int(embed_dim * 2 ** i_layer),
                                       depth=fetus_blocks_number[i_layer],
                                       block2_depth=block2_depths[i_layer],
                                       input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                         patches_resolution[1] // (2 ** i_layer)),
                                       drop=drop_rate,
                                       drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                       norm_layer=norm_layer,
                                       downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                       use_checkpoint=use_checkpoint)
                self.layers.append(layer)

            numbers = [1, 2]
            self.add_layer = nn.ModuleList()
            self.crn = nn.ModuleList()
            for add in numbers:
                add_up = PatchExpand_two(input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - add)),
                                                           patches_resolution[1] // (2 ** (self.num_layers - 1 - add))),
                                         dim=2 * int(embed_dim * 2 ** (self.num_layers - 1 - add)), dim_scale=2,
                                         norm_layer=norm_layer)
                crn = CRN(dim=int(embed_dim * 2 ** (self.num_layers - 1 - add)))
                self.crn.append(crn)
                self.add_layer.append(add_up)

            # build decoder layers
            self.layers_up = nn.ModuleList()
            self.concat_back_dim = nn.ModuleList()
            for i_layer in range(self.num_layers):
                concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                          int(embed_dim * 2 ** (
                                                      self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
                if i_layer == 0:
                    layer_up = PatchExpand(
                        input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                          patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                        dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
                else:
                    layer_up = MetaBasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                                 depth=depths[(self.num_layers - 1 - i_layer)],
                                                 input_resolution=(
                                                 patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                                 patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                                 drop=drop_rate,
                                                 drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                                     depths[:(self.num_layers - 1 - i_layer) + 1])],
                                                 norm_layer=norm_layer,
                                                 upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                                 use_checkpoint=use_checkpoint)
                self.layers_up.append(layer_up)
                self.concat_back_dim.append(concat_linear)
#################################################################################################

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size),
                                          dim_scale=4, dim=embed_dim)
            self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.out_chans, kernel_size=1, bias=False)
            self.dp = nn.Sequential(
                #nn.AdaptiveMaxPool2d((1,1)),
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1,bias=True),
                nn.Tanh()
                
            )
            # self.dp = nn.Sequential(
                # #nn.AdaptiveMaxPool2d((1,1)),
                # nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1,bias=True),
                # nn.Tanh()
                
            # )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):

        # convnext_list = self.convnext(x)
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []

        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)

        # return x, x_downsample, convnext_list
        return x, x_downsample

    def forward_up_features(self, x, x_downsample):
        # 不写为for循环
        # print('x', x.size())
        x = self.layers_up[0](x)
        # print('xx1', x.size())
        x1 = torch.cat([x, x_downsample[2]], -1)
        x = self.concat_back_dim[1](x1)
        x = self.layers_up[1](x)
        # print('xx2', x.size())
        x2 = torch.cat([x, x_downsample[1]], -1)
        # print('x1', x1.size())
        x1 = self.add_layer[0](x1)
        # print('x11', x1.size())
        # print('x2', x1.size())
        x1 = self.crn[0](x1)
        x = x2 + x1
        x = self.concat_back_dim[2](x)
        x = self.layers_up[2](x)
        # print('xx3', x.size())
        x3 = torch.cat([x, x_downsample[0]], -1)
        x2 = self.add_layer[1](x2)
        x2 = self.crn[1](x2)
        x = x3 + x2
        x = self.concat_back_dim[3](x)
        x = self.layers_up[3](x)
        # print('xx4', x.size())
        x = self.norm_up(x)
        return x

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            x = self.output(x)
            
        return x

    def forward(self, img):
        x, x_downsample = self.forward_features(img)
        # print('已完成')
        x = self.forward_up_features(x, x_downsample)
        x = self.up_x4(x)
        x = torch.tanh(x)
        imgd = self.dp(img)
        r1, r2, r3 = torch.split(x+0.1*torch.log1p(imgd), 1, dim=1)
        # r1, r2, r3 = torch.split(x, 1, dim=1)
        img = img + r1 * (torch.pow(img, 2) - img)
        img = img + r2 * (torch.pow(img, 2) - img)
        img = img + r2 * (torch.pow(img, 2) - img)
        enhance_image = img + r3 * (torch.pow(img, 2) - img)

        r = torch.cat([r1, r2, r3], 1)

        # return enhance_image, r, torch.log1p(imgd)
        return enhance_image, r, torch.log1p(imgd)

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.out_chans
        return flops
        
def metauace3v(img_size=224):
    model = MetaUCE(img_size=img_size, patch_size=4, in_chans=3, out_chans=3,
                 embed_dim=32,
                 blocks_number=[2, 2, 6, 2],
                 depths=[1, 2, 3, 1],
                 mlp_ratio=4.,
                 drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first",
                 )
    return model
