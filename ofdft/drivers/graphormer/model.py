# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Adapt from graphormer/models/graphormer_3d.py

from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import argparse

def softmax_dropout(input, dropout_prob: float, is_training: bool):
    return F.dropout(F.softmax(input, -1), dropout_prob, is_training)

class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None

class SelfMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        scaling_factor=1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = (self.head_dim * scaling_factor) ** -0.5

        self.in_proj: Callable[[Tensor], Tensor] = nn.Linear(
            embed_dim, embed_dim * 3, bias=bias
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query: Tensor,
        attn_bias: Tensor = None,
    ) -> Tensor:
        n_node, n_graph, embed_dim = query.size()
        q, k, v = self.in_proj(query).chunk(3, dim=-1)

        _shape = (-1, n_graph * self.num_heads, self.head_dim)
        q = q.contiguous().view(_shape).transpose(0, 1) * self.scaling
        k = k.contiguous().view(_shape).transpose(0, 1)
        v = v.contiguous().view(_shape).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) + attn_bias
        attn_probs = softmax_dropout(attn_weights, self.dropout, self.training)

        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(n_node, n_graph, embed_dim)
        attn = self.out_proj(attn)
        return attn


class Graphormer3DEncoderLayer(nn.Module):
    """
    Implements a Graphormer-3D Encoder Layer.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout

        self.dropout = dropout
        self.activation_dropout = activation_dropout

        self.self_attn = SelfMultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
        )
        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: Tensor,
        attn_bias: Tensor = None,
    ):
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(
            query=x,
            attn_bias=attn_bias,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = F.gelu(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        return x


class RBF(nn.Module):
    def __init__(self, K, edge_types):
        super().__init__()
        self.K = K
        self.means = nn.parameter.Parameter(torch.empty(K))
        self.temps = nn.parameter.Parameter(torch.empty(K))
        self.mul: Callable[..., Tensor] = nn.Embedding(edge_types, 1)
        self.bias: Callable[..., Tensor] = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means, 0, 3)
        nn.init.uniform_(self.temps, 0.1, 10)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1) 

    def forward(self, x: Tensor, edge_types):
        mul = self.mul(edge_types)
        bias = self.bias(edge_types)
        x = mul * x.unsqueeze(-1) + bias
        mean = self.means.float()
        temp = self.temps.float().abs()
        return ((x - mean).square() * (-temp)).exp().type_as(self.means)


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_types):
        mul = self.mul(edge_types)
        bias = self.bias(edge_types)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)

class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()
        if hidden is None:
            hidden = input
        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, output_size)

    def reset_parameters(self):
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()

    def forward(self, x):
        x = F.gelu(self.layer1(x))
        x = self.layer2(x)
        return x


class Scalarizer(nn.Module):
    def __init__(self, init_scale=0.1, shrunk=False, coeff_dim=207, outer_scale=10):
        super(Scalarizer, self).__init__()
        inner_factor = torch.ones(coeff_dim) * init_scale
        outer_factor = torch.ones(coeff_dim) * outer_scale
        self.register_parameter('inner_factor', nn.Parameter(inner_factor))
        self.register_parameter('outer_factor', nn.Parameter(outer_factor))
        self.shrunk = shrunk
    
    def forward(self, coeff):
        '''
        coeff: B, N, 207
        scalar_coeff: 1, 1, 207
        '''
        if self.shrunk:
            return torch.tanh(coeff * self.inner_factor.reshape(1, 1, -1)) * self.outer_factor.reshape(1, 1, -1)
        else:
            return coeff * self.inner_factor.reshape(1, 1, -1)
        
class GaussianMLP(nn.Module):
    def __init__(self, input, hidden, alpha=40, learnable=False):
        super().__init__()
        self.input = input
        self.hidden = hidden
        self.layer1 = nn.Linear(input, hidden, bias=False)
        self.layer2 = nn.Linear(input, hidden, bias=False)
        # self.weight1 = nn.Parameter(torch.empty((hidden, input)))
        # self.weight2 = nn.Parameter(torch.empty((hidden, hidden)))
        # torch.nn.init.trunc_normal_(self.weight1, a=-1, b=1, std=math.sqrt(hidden)) 
        self.register_parameter('alpha', nn.Parameter(torch.tensor([alpha]), requires_grad=learnable))
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.trunc_normal_(self.layer1.weight, mean=0, std=1, a=-1, b=1)
        torch.nn.init.trunc_normal_(self.layer2.weight, mean=0, std=1, a=-1, b=1)
        self.layer1.weight = torch.nn.parameter.Parameter(self.layer1.weight / math.sqrt(self.input))
        self.layer2.weight = torch.nn.parameter.Parameter(self.layer2.weight / math.sqrt(self.hidden))

    def forward(self, coeff: Tensor, padding_mask: Tensor):
        '''
        coeff: [B, N, C]
        sigma: [1,] -> [1, 1, 1]
        '''
        x = coeff / torch.sqrt(torch.mean(coeff ** 2, dim=-1, keepdim=True) + 1e-5) #RMSnorm
        x = self.layer1(x)
        x = torch.exp(-self.alpha * x ** 2)
        x = self.layer2(x)
        x = x + coeff
        x = x * (~padding_mask[:, :, None])
        return x

class GaussianEncoder(nn.Module):
    def __init__(self, input, hidden, output, hidden_layers=5, alpha=40, learnable=False, grad_rescale=0.1):
        super().__init__()
        self.input = input
        self.hidden = hidden
        self.hidden_layers = hidden_layers
        self.embed_layer = nn.Linear(input, hidden, bias=False)
        self.gaussians = nn.ModuleList(
            [GaussianMLP(hidden, hidden, alpha=alpha, learnable=learnable) for _ in range(hidden_layers)]
            )
        self.out_layer = nn.Linear(hidden, output, bias=False)
        self.reset_parameters()
        self.grad_rescale = grad_rescale

    def reset_parameters(self):
        torch.nn.init.trunc_normal_(self.embed_layer.weight, mean=0, std=1, a=-1, b=1)
        torch.nn.init.trunc_normal_(self.out_layer.weight, mean=0, std=1, a=-1, b=1)
        self.embed_layer.weight = torch.nn.parameter.Parameter(self.embed_layer.weight / math.sqrt(self.input))
        self.out_layer.weight = torch.nn.parameter.Parameter(self.out_layer.weight / math.sqrt(self.hidden))

    def forward(self, coeff, padding_mask):
        x = self.embed_layer(coeff)
        x = GradMultiply.apply(x, 1 / self.grad_rescale)
        for layer in self.gaussians:
            x = layer(x, padding_mask)
        x = GradMultiply.apply(x, self.grad_rescale)
        out = self.out_layer(x) * (~padding_mask[:, :, None])
        return out


class Graphormer3D(nn.Module):
    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument("--blocks", type=int, metavar="L", help="num blocks")
        parser.add_argument(
            "--embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--node-loss-weight",
            type=float,
            metavar="D",
            help="loss weight for node fitting",
        )
        parser.add_argument(
            "--min-node-loss-weight",
            type=float,
            metavar="D",
            help="loss weight for node fitting",
        )
        parser.add_argument(
            "--num-kernel",
            type=int,
        )
        parser.add_argument(
            "--init-scale",
            type=float,
            help="0: no pre-scale, >0: scale using \lambda * tanh(\gamma*coeff)"
        )
        parser.add_argument(
            "--outer-scale",
            type=float,
            default=10,
            help="outer scale \lambda: scale using \lambda * tanh(\gamma*coeff)"
        )
        parser.add_argument(
            '--shrunk', 
            action=argparse.BooleanOptionalAction, 
            help='shrunk the input coeffs (output grads) or not'
        )
        parser.add_argument(
            "--kernel-type",
            type=str,
        )
        parser.add_argument(
            '--coeff-dim', 
            type=int,
            default=207,
            help='coeff dimension'
        )
        parser.add_argument(
            '--coeff-encoder-type', 
            type=str,
            default='mlp',
            help='coeff encoder type'
        )
        parser.add_argument(
            '--gauss-alpha', 
            type=float,
            default=10.0,
            help='sigma of gaussian activation function'
        )
        parser.add_argument(
            '--gauss-layers', 
            type=int,
            default=5,
            help='sigma of gaussian activation function'
        )
        parser.add_argument(
            '--gauss-learn', 
            action=argparse.BooleanOptionalAction, 
            help='learn gaussian sigma or not'
        )
        parser.add_argument(
            '--gauss-grad-scale', 
            type=float, 
            help='learn gaussian sigma or not'
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture(args)
        return cls(args)

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.atom_types = 64
        self.edge_types = 64 * 64
        self.atom_encoder = nn.Embedding(
            self.atom_types, self.args.embed_dim, padding_idx=0
        )
        self.init_scale = self.args.init_scale
        self.shrunk = self.args.shrunk
        self.coeff_dim = self.args.coeff_dim
        self.encoder_type  = self.args.coeff_encoder_type
        if self.encoder_type == 'mlp':
            outer_scale = self.args.outer_scale
            self.scalarizer = Scalarizer(init_scale=self.init_scale, shrunk=self.shrunk, coeff_dim=self.coeff_dim, outer_scale=outer_scale) if self.init_scale > 0 else lambda x: x
            self.coeff_encoder = NonLinear(self.coeff_dim, self.args.embed_dim, hidden=self.args.embed_dim)
        elif self.encoder_type == 'ga_mlp':
            alpha = self.args.gauss_alpha
            gauss_layers = self.args.gauss_layers
            learnable = self.args.gauss_learn
            grad_rescale = self.args.gauss_grad_scale
            self.scalarizer = Scalarizer(init_scale=2, shrunk=self.shrunk, coeff_dim=self.coeff_dim, outer_scale=1)
            self.coeff_encoder = GaussianEncoder(self.coeff_dim, self.args.embed_dim, self.args.embed_dim, hidden_layers=gauss_layers, alpha=alpha, 
                learnable=learnable, grad_rescale=grad_rescale)
        else:
            raise NotImplementedError('unrecognized coeff encoder type')

        self.input_dropout = self.args.input_dropout
        self.layers = nn.ModuleList(
            [
                Graphormer3DEncoderLayer(
                    self.args.embed_dim,
                    self.args.ffn_embed_dim,
                    num_attention_heads=self.args.attention_heads,
                    dropout=self.args.dropout,
                    attention_dropout=self.args.attention_dropout,
                    activation_dropout=self.args.activation_dropout,
                )
                for _ in range(self.args.layers)
            ]
        )

        self.final_ln: Callable[[Tensor], Tensor] = nn.LayerNorm(self.args.embed_dim)

        self.engergy_proj: Callable[[Tensor], Tensor] = NonLinear(
            self.args.embed_dim, 1
        )
        self.ground_energy_proj: Callable[[Tensor], Tensor] = NonLinear(
            self.args.embed_dim, 1
        )
        self.coeff_offset_proj: Callable[[Tensor], Tensor] = NonLinear(
            self.args.embed_dim, self.coeff_dim
        )

        K = self.args.num_kernel
        if self.args.kernel_type == 'rbf':
            self.dist_encoder = RBF(K, self.edge_types)
        elif self.args.kernel_type == 'gbf':
            self.dist_encoder = GaussianLayer(K, self.edge_types)
        else:
            raise NotImplementedError()

        self.bias_proj: Callable[[Tensor], Tensor] = NonLinear(
            K, self.args.attention_heads
        )
        self.edge_proj: Callable[[Tensor], Tensor] = nn.Linear(K, self.args.embed_dim)

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates
        return super().set_num_updates(num_updates)

    def forward(self, batched_data):
        atoms = batched_data["x"] # B x T x 1
        pos = batched_data['pos'].float()
        node_attr = batched_data['node_attr']
        padding_mask = ~batched_data['padding_mask']

        n_graph, n_node = atoms.size()[:2]
        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist: Tensor = delta_pos.norm(dim=-1) + 1e-5

        edge_type = atoms.view(n_graph, n_node, 1) * self.atom_types + atoms.view(
            n_graph, 1, n_node
        )

        dist_feature = self.dist_encoder(dist, edge_type)
        edge_features = dist_feature.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(-1), 0.0
        )

        node_features = self.scalarizer(node_attr)
        if self.encoder_type == 'mlp':
            node_features = self.coeff_encoder(node_features)
        else:
            node_features = self.coeff_encoder(node_features, padding_mask)

        graph_node_feature = (
            node_features
            + self.atom_encoder(atoms.squeeze(-1))
            + self.edge_proj(edge_features.sum(dim=-2))
        )

        # ===== MAIN MODEL =====
        output = F.dropout(
            graph_node_feature, p=self.input_dropout, training=self.training
        )
        output = output.transpose(0, 1).contiguous()

        graph_attn_bias = self.bias_proj(dist_feature).permute(0, 3, 1, 2).contiguous()
        graph_attn_bias.masked_fill_(
            padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
        )

        graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
        for _ in range(self.args.blocks):
            for enc_layer in self.layers:
                output = enc_layer(output, attn_bias=graph_attn_bias)

        output = self.final_ln(output)
        output = output.transpose(0, 1)
        final_output = F.dropout(output, p=0.1, training=self.training)

        # predict energy for current SCF step
        eng_output = (
            self.engergy_proj(final_output)
        ).flatten(-2)
        eng_output = eng_output * (~padding_mask)
        eng_output = eng_output.sum(dim=-1)

        # predict ground energy
        ground_eng_output = (
            self.ground_energy_proj(final_output)
        ).flatten(-2)
        ground_eng_output = ground_eng_output * (~padding_mask)
        ground_eng_output = ground_eng_output.sum(dim=-1)
        
        # predict delta coefficients
        coeff_offset_out = self.coeff_offset_proj(final_output)

        return eng_output, ground_eng_output, coeff_offset_out


def base_architecture(args):
    args.blocks = getattr(args, "blocks", 1)
    args.layers = getattr(args, "layers", 12)
    args.embed_dim = getattr(args, "embed_dim", 768)
    args.ffn_embed_dim = getattr(args, "ffn_embed_dim", 768)
    args.attention_heads = getattr(args, "attention_heads", 48)
    args.input_dropout = getattr(args, "input_dropout", 0.0)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.node_loss_weight = getattr(args, "node_loss_weight", 15)
    args.min_node_loss_weight = getattr(args, "min_node_loss_weight", 1)
    args.eng_loss_weight = getattr(args, "eng_loss_weight", 1)
    args.num_kernel = getattr(args, "num_kernel", 128)
    args.init_scale = getattr(args, "init_scale", 0)
    args.outer_scale = getattr(args, "outer_scale", 10)
    args.shrunk = getattr(args, "shrunk", False)
    args.coeff_dim = getattr(args, "coeff_dim", 207)
    args.coeff_encoder_type = getattr(args, "coeff_encoder_type", 'mlp')
    args.gauss_alpha = getattr(args, "gauss_alpha", 40.0)
    args.gauss_layers = getattr(args, "gauss_layers", 5)
    args.gauss_learn = getattr(args, "gauss_learn", False)
    args.gauss_grad_scale = getattr(args, "gauss_grad_scale", 1)
    args.kernel_type = getattr(args, "kernel_type", 'rbf')

def load_model(model_ckpt_path, use_ema=False):
    ckpt = torch.load(model_ckpt_path, map_location='cpu')
    model = Graphormer3D(ckpt['cfg']['model'])
    if use_ema:
        model.load_state_dict(ckpt['extra_state']['ema'])
    else:
        model.load_state_dict(ckpt['model'])
    model.eval()
    return model
