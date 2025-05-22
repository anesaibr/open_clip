# Copyright (c) Meta Platforms, Inc. and affiliates.
import logging # added this
from logging import getLogger
import math
from typing import Optional
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import torch.distributed as dist

from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor.parallel import parallelize_module
from torch.cuda.amp import autocast

from .colwise_embedding_bag import ColwiseEmbeddingBag, xFormerEmbeddingBag

logger = getLogger()


@dataclass
class ProductKeyArgs:
    is_enabled: bool = False
    layers: str = (
        ""  # Which layers to have the memory with product key on (example "6,12")
    )
    mem_n_keys: int = 1024  # The number of keys in the memory.
    mem_heads: int = 4  # Number of memory reading heads
    mem_knn: int = 32  # Number of memory slots to read / update - k-NN to the query
    mem_share_values: bool = True  # Share values across memories
    mem_k_dim: int = 512  # Memory keys dimension
    mem_v_dim: int = -1  # Memory values dimension (-1 for automatic output dimension)
    swilu_projection: bool = True
    value_fixed_lr: Optional[
        float
    ] = 0.001  # The learning rate for the value of the PK network
    mem_gated: bool = False
    peer_variant: bool = False


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class HashingMemory(nn.Module):

    VALUES = None
    EVAL_MEMORY = True

    @classmethod
    def reset_shared_state(cls):
        """Call this before creating a new model instance in the same process."""
        cls.VALUES = None
        logging.info("Reset HashingMemory.VALUES to None.")

    def __init__(
        self,
        input_dim,
        output_dim,
        value_fixed_lr=0.001,  # added for xlformers and replaces mem_value_optimizer, set to None to use the same learning rate as the rest of the model
        # global parameters
        mem_k_dim=512,  # Memory keys dimension
        mem_v_dim=-1,  # Memory values dimension (-1 for automatic output dimension)
        mem_heads=4,  # Number of memory reading heads
        mem_knn=32,  # Number of memory slots to read / update - k-NN to the query
        mem_share_values=True,  # Share values across memories
        # keys
        mem_n_keys=1024,  # Number of keys
        # queries
        mem_query_bias=True,  # Query MLP bias
        mem_query_batchnorm=False,  # Query MLP batch norm
        # gating
        mem_gated=False,  # gated memory
        # values initialization
        # dropout
        mem_input_dropout=0.0,  # Input dropout
        mem_query_dropout=0.0,  # Query dropout
        mem_value_dropout=0.0,  # Value dropout
        # architecture
        peer_variant=False,  # Replaces the PK memory with the PEER variant (Parameter Efficient Expert Retrieval)
        swilu_projection=True,
    ):
        # Check parameters
        # even number of key dimensions for product quantization
        assert mem_k_dim >= 2

        # dropout
        assert 0 <= mem_input_dropout < 1
        assert 0 <= mem_query_dropout < 1
        assert 0 <= mem_value_dropout < 1

        # PEER variant
        assert not (
            peer_variant and mem_v_dim > 0
        ), f"Cannot use PEER variant with a value dimension different from the input dimension (mem_v_dim=-1)"

        # dimensions
        assert mem_k_dim % 2 == 0
        assert mem_heads >= 2

        # query batchnorm
        if mem_query_batchnorm:
            logger.warning(
                "WARNING: if you use batch normalization, be sure that you use batches of sentences with the same size at training time. Otherwise, the padding token will result in incorrect mean/variance estimations in the BatchNorm layer."
            )

        # initialize
        super().__init__()
        self.use_peer_variant = peer_variant

        # global parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        # number of indices / entries in the memory
        self.size = mem_n_keys**2
        self.k_dim = mem_k_dim

        self.v_dim = mem_v_dim if mem_v_dim > 0 else output_dim

        # values initialization
        self.swilu_proj = swilu_projection
        self.v_proj = mem_v_dim > 0 or self.swilu_proj
        self.heads = mem_heads
        self.knn = mem_knn

        # dropout
        self.input_dropout = mem_input_dropout
        self.query_dropout = mem_query_dropout
        self.value_dropout = mem_value_dropout

        # initialize keys
        self.keys = nn.Parameter(
            torch.empty(2 * self.heads * int(self.size**0.5), self.k_dim // 2)
        )
        nn.init.normal_(self.keys, mean=0.0, std=0.02)  # ADDED THIS to fix NaN error, can adjust std=1.0 for strong variance
        print("[DEBUG:init] keys initialized:")
        print("  stats:", self.keys.min(), self.keys.max(), self.keys.mean(), self.keys.std())



        # optionally use the same values for all memories
        self.mem_share_values = mem_share_values

        self.original = not self.mem_share_values or HashingMemory.VALUES is None

        print(f"[HashingMemory] Creating memory layer. "
              f"Original={self.original}. mem_share_values={self.mem_share_values}. "
              f"Global VAL={HashingMemory.VALUES}")
        
        # initialize the values
        if self.original: # actually constructs an Embedding table and sets the global reference
            if not self.use_peer_variant:  # PK
                self.values = xFormerEmbeddingBag(self.size, self.v_dim)
                print("[DEBUG] Initializing self.values...")
                print("  values weight stats:", self.values.weight.min(), self.values.weight.max(), self.values.weight.mean(), self.values.weight.std())
                HashingMemory.VALUES = self.values
            else:  # PEER
                self.values_u = nn.Embedding(self.size, self.v_dim)
                self.values_v = nn.Embedding(self.size, self.v_dim)
                HashingMemory.VALUES = self.values_u, self.values_v
        else:  # rely on the “original” layer’s global reference and set values to None initially
            if not self.use_peer_variant:  # PK
                self.values = None
            else:  # PEER
                self.values_u = None
                self.values_v = None
        self.value_fixed_lr = value_fixed_lr

        if self.v_proj:
            proj_input = mem_v_dim
            if self.swilu_proj and proj_input < 0:
                proj_input = output_dim
            self.value_proj = torch.nn.Linear(proj_input, output_dim)
        if self.swilu_proj:
            self.swilu_projection = torch.nn.Linear(self.input_dim, proj_input)
        # gated memory
        self.gating = None
        if mem_gated:
            self.gating = torch.nn.Linear(input_dim, 1)

        # query network
        # layer sizes / number of features
        l_sizes = (self.input_dim, self.heads * self.k_dim)

        self.query_proj = QueryMLP(
            self.input_dim,
            self.heads,
            self.k_dim,
            l_sizes,
            bias=mem_query_bias,
            batchnorm=mem_query_batchnorm,
        )

    def mp_parallelize(self, mesh, model_args, distributed_args, param_dtype):
        # fsdp_config = dict(
        #     mp_policy=(
        #         MixedPrecisionPolicy(
        #             param_dtype=param_dtype,
        #             # reduce_dtype=torch.float32,
        #             reduce_dtype=torch.bfloat16,
        #         )
        #     ),
        #     mesh=mesh["dp_replicate"],
        # )
        # # parallelize the module
        # if distributed_args.memory_parallel_size > 1:
        #     assert (
        #         not self.use_peer_variant
        #     ), f"The PEER variant does not have a memory parallel implementation"
        #     if self.original:
        #         layer_plan = {"values": ColwiseEmbeddingBag()}
        #         parallelize_module(
        #             self,
        #             mesh["memory_parallel"],
        #             layer_plan,
        #         )

        print(f"[HashingMemory] mp_parallelize called. original={self.original}. "
              f"mem_share_values={self.mem_share_values}. "
              f"HashingMemory.VALUES={HashingMemory.VALUES}")
        
        # print(f"[HashingMemory] Sharding memory table with shape {self.values.weight.shape}."
        #       f"[HashingMemory] device = {self.values.weight.device}")
        
        # share the parameters
        # if self.original:
        #     if not self.use_peer_variant:
        #         self.values = fully_shard(
        #             self.values, **fsdp_config, reshard_after_forward=False
        #         )
        #     else:
        #         self.values_u = fully_shard(
        #             self.values_u, **fsdp_config, reshard_after_forward=False
        #         )
        #         self.values_v = fully_shard(
        #             self.values_v, **fsdp_config, reshard_after_forward=False
        #         )

        # --- Rank-Local Value Sharing Logic (Retained) ---
        # This section ensures layers within the same rank use the same 'values' object.
        if self.mem_share_values and self.original:
            if not self.use_peer_variant:
                HashingMemory.VALUES = self.values  # store reference globally
            else:
                HashingMemory.VALUES = self.values_u, self.values_v
            # Broadcast the actual memory values from rank 0
            # if dist.is_initialized():
            #     dist.broadcast(self.values.weight.data, src=0)

        if self.mem_share_values and not self.original:
            # If this is NOT the 'original' layer, retrieve the shared reference
            # stored previously by the 'original' layer *in this same rank*.
            print("[HashingMemory] Re-assigning self.values from global!")
            if not self.use_peer_variant:
                self.values = HashingMemory.VALUES # store reference from global xFormerEmbeddingBag + ensure it's on the current device!
                # if self.values.device != torch.cuda.current_device():
                self.values = self.values.to(torch.cuda.current_device())
                # if dist.is_initialized():
                #     dist.broadcast(self.values.weight.data, src=0)
                #     dist.barrier()  # Ensure all ranks have received the shared memory
                #     print(f"[DEBUG] broadcasted shared memory to rank {dist.get_rank()}")
                #     if torch.isnan(self.values.weight).any() or torch.isinf(self.values.weight).any():
                #         print("[DEBUG] self.values.weight has NaNs/Infs after broadcast")
                #         print("weight stats:", self.values.weight.min(), self.values.weight.max(), self.values.weight.mean(), self.values.weight.std())
                
            else:
                self.values_u, self.values_v = HashingMemory.VALUES
                self.values_u = self.values_u.to(torch.cuda.current_device())
                self.values_v = self.values_v.to(torch.cuda.current_device())
        
        if self.values is None:
            print("[HashingMemory] self.values is still None? skipping shape print.")
            return
        
        # Now we can safely do the shape print
        print(f"[HashingMemory] Sharding memory table with shape {self.values.weight.shape}.")
        print(f"[HashingMemory] device = {self.values.weight.device}")

    def reset_parameters(self, init_std=None, factor=1.0):
        # keys
        bound = 1 / math.sqrt(self.k_dim)
        nn.init.uniform_(self.keys, a=-bound, b=bound)
        # values
        if self.original:
            if not self.use_peer_variant:
                nn.init.normal_(self.values.weight, mean=0, std=self.v_dim**-0.5)
            else:
                nn.init.normal_(self.values_u.weight, mean=0, std=self.v_dim**-0.5)
                nn.init.normal_(self.values_v.weight, mean=0, std=self.v_dim**-0.5)
        # queries
        nn.init.xavier_uniform_(self.query_proj.query_mlps[0].weight)
        # value projection
        if self.v_proj:
            nn.init.normal_(self.value_proj.weight, mean=0, std=self.output_dim**-0.5)
        if self.swilu_proj:
            nn.init.normal_(
                self.swilu_projection.weight, mean=0, std=self.output_dim**-0.5
            )
        # fixed learning rate:
        if self.original:
            if self.use_peer_variant:
                for p in self.values_u.parameters():
                    p.fixed_lr = self.value_fixed_lr
                    p.pk_value_param = True
                for p in self.values_v.parameters():
                    p.fixed_lr = self.value_fixed_lr
                    p.pk_value_param = True
            else:
                for p in self.values.parameters():
                    p.fixed_lr = self.value_fixed_lr
                    p.pk_value_param = True
        if self.gating is not None:
            nn.init.normal_(self.gating.weight, mean=0, std=self.input_dim**-0.5)

    def forward(self, input):
        """
        Read from the memory.
        """
        #------------------------(Casting-Necessary if used in the Autocast)----------------------------------------
        # dtype = input.dtype  # Get input dtype once

        # # Cast all parameters to input.dtype
        # keys = self.keys.to(dtype)
        # if self.values is not None:
        #     values_weight = self.values.weight.to(dtype)
        # if self.value_proj is not None:
        #     value_proj_weight = self.value_proj.weight.to(dtype)
        #     if self.value_proj.bias is not None:
        #         value_proj_bias = self.value_proj.bias.to(dtype)
        # if self.swilu_projection is not None:
        #     swilu_projection_weight = self.swilu_projection.weight.to(dtype)
        #     if self.swilu_projection.bias is not None:
        #         swilu_projection_bias = self.swilu_projection.bias.to(dtype)
        # if self.gating is not None:
        #     gating_weight = self.gating.weight.to(dtype)
        #     if self.gating.bias is not None:
        #         gating_bias = self.gating.bias.to(dtype)
        #----------------------------------------------------------------
        B, T, C = input.shape
        input = input.view(-1, self.input_dim)

        # input dimensions
        assert input.shape[-1] == self.input_dim
        prefix_shape = input.shape[:-1]

        # compute query / store it
        bs = np.prod(prefix_shape)
        #---------------(Autocast)-------------------
        # with autocast():
        #     input = F.dropout(
        #         input, p=self.input_dropout, training=self.training
        #     )  # input shape
        #     query = self.query_proj(input)  # (bs * heads, k_dim)
        #     query = F.dropout(
        #         query, p=self.query_dropout, training=self.training
        #     )  # (bs * heads, k_dim)
        #     assert query.shape == (bs * self.heads, self.k_dim)

        #     # get indices
        #     knn = self.knn
        #     scores, indices = self.get_indices(query, knn)  # (bs * heads, knn) ** 2

        #     # store indices / scores (eval mode only - for usage statistics)
        #     if not self.training and HashingMemory.EVAL_MEMORY:
        #         self.last_indices = indices.view(bs, self.heads, knn).detach().cpu()
        #         self.last_scores = scores.view(bs, self.heads, knn).detach().cpu().float()

        #     # re-scoring
        #     scores = F.softmax(scores.float(), dim=-1).type_as(scores)  # (bs * heads, knn)

        #     # merge heads / knn (since we sum heads)
        #     indices = indices.view(bs, self.heads * knn)  # (bs, heads * knn)
        #     scores = scores.view(bs, self.heads * knn)  # (bs, heads * knn)

        #     if not self.use_peer_variant:
        #         output = self.values(indices, scores).to(dtype)  # Cast to input.dtype  # (bs, v_dim)
        #         if self.v_proj and not self.swilu_proj:
        #             # output = self.value_proj(output)
        #             output = F.Linear(output, value_proj_weight, value_proj_bias)
        #         if self.swilu_proj:
        #             # output = self.value_proj(output * F.silu(self.swilu_projection(input)))
        #             swilu_out = F.linear(input, swilu_projection_weight, swilu_projection_bias)
        #             output = F.linear(output * F.silu(swilu_out), value_proj_weight, value_proj_bias)
        #     else:
        #         u = self.values_u(indices).to(dtype)
        #         x = torch.einsum(
        #             "bh, blh->bl", input, u
        #         )  # (bs, v_dim) , (bs, heads * knn, v_dim) -> (bs, heads * knn)
        #         x = F.gelu(x)  # This can be either GeLU or ReLU
        #         v = self.values_v(indices)
        #         x = x * scores  # (bs, heads * knn)
        #         output = torch.einsum(
        #             "bl, blh->bh", x, v
        #         )  # (bs, heads * knn) , (bs, heads * knn, v_dim) -> (bs, v_dim)

        #     output = F.dropout(
        #         output, p=self.value_dropout, training=self.training
        #     )  # (bs, v_dim)

        #     # reshape output
        #     if len(prefix_shape) >= 2:
        #         output = output.view(prefix_shape + (self.v_dim,))  # (..., v_dim)

        #     if self.gating:
        #         # output = F.sigmoid(self.gating(input)) * output
        #         gate = F.sigmoid(F.linear(input, gating_weight, gating_bias))
        #         output = gate * output
        #     output = output.view(B, T, -1)
        #     return output
        #------------------------------------------------
        input = F.dropout(
            input, p=self.input_dropout, training=self.training
        )  # input shape
        # Debugging
        if torch.isnan(input).any() or torch.isinf(input).any():
            print("[DEBUG] input to query_proj has NaNs/Infs!")
            print("input stats:", input.min(), input.max(), input.mean(), input.std())

        query = self.query_proj(input)  # (bs * heads, k_dim)
        # Debugging
        if torch.isnan(query).any() or torch.isinf(query).any():
            print("[DEBUG] query_proj output contains NaNs/Infs!")
            print("query stats:", query.min(), query.max(), query.mean(), query.std())
        
        query = F.dropout(
            query, p=self.query_dropout, training=self.training
        )  # (bs * heads, k_dim)
        assert query.shape == (bs * self.heads, self.k_dim)

        # get indices
        knn = self.knn
        scores, indices = self.get_indices(query, knn)  # (bs * heads, knn) ** 2

        # store indices / scores (eval mode only - for usage statistics)
        if not self.training and HashingMemory.EVAL_MEMORY:
            self.last_indices = indices.view(bs, self.heads, knn).detach().cpu()
            self.last_scores = scores.view(bs, self.heads, knn).detach().cpu().float()
        
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            print("[DEBUG] Softmax input has NaNs/Infs before F.softmax")
        # re-scoring
        scores = F.softmax(scores.float(), dim=-1).type_as(scores)  # (bs * heads, knn)

        if torch.isnan(scores).any() or torch.isinf(scores).any():
            print("[DEBUG] Softmax output has NaNs/Infs after F.softmax")
            print("scores stats:", scores.min(), scores.max(), scores.mean(), scores.std())

        # merge heads / knn (since we sum heads)
        indices = indices.view(bs, self.heads * knn)  # (bs, heads * knn)
        scores = scores.view(bs, self.heads * knn)  # (bs, heads * knn)

        if not self.use_peer_variant:
            output = self.values(indices, scores)  # (bs, v_dim)
            if torch.isnan(output).any() or torch.isinf(output).any():
                print("[DEBUG] self.values() returned NaNs/Infs")
                print("values output stats:", output.min(), output.max(), output.mean(), output.std())
            if self.v_proj and not self.swilu_proj:
                output = self.value_proj(output)
            if self.swilu_proj:
                swilu_input = self.swilu_projection(input)
                if torch.isnan(swilu_input).any() or torch.isinf(swilu_input).any():
                    print("[DEBUG] swilu_projection(input) produced NaNs/Infs")
                    print("swilu_input stats:", swilu_input.min(), swilu_input.max(), swilu_input.mean(), swilu_input.std())
                activation = F.silu(swilu_input)
                if torch.isnan(activation).any() or torch.isinf(activation).any():
                    print("[DEBUG] silu output has NaNs/Infs")
                output = self.value_proj(output * F.silu(self.swilu_projection(input)))

        else:
            u = self.values_u(indices)
            x = torch.einsum(
                "bh, blh->bl", input, u
            )  # (bs, v_dim) , (bs, heads * knn, v_dim) -> (bs, heads * knn)
            x = F.gelu(x)  # This can be either GeLU or ReLU
            v = self.values_v(indices)
            x = x * scores  # (bs, heads * knn)
            output = torch.einsum(
                "bl, blh->bh", x, v
            )  # (bs, heads * knn) , (bs, heads * knn, v_dim) -> (bs, v_dim)

        output = F.dropout(
            output, p=self.value_dropout, training=self.training
        )  # (bs, v_dim)

        # reshape output
        if len(prefix_shape) >= 2:
            output = output.view(prefix_shape + (self.v_dim,))  # (..., v_dim)

        if self.gating:
            gating_out = self.gating(input)
            if torch.isnan(gating_out).any() or torch.isinf(gating_out).any():
                print("[DEBUG] gating(input) produced NaNs/Infs!")
                print("gating stats:", gating_out.min(), gating_out.max(), gating_out.mean(), gating_out.std())
            output = F.sigmoid(self.gating(input)) * output
        output = output.view(B, T, -1)

        if torch.isnan(output).any() or torch.isinf(output).any():
            print("[DEBUG] HashingMemory FINAL output has NaNs/Infs!")
            print("output stats:", output.min(), output.max(), output.mean(), output.std())


        return output

    def get_indices(self, query, knn):
        assert query.dim() == 2 and query.size(1) == self.k_dim
        bs = len(query) // self.heads
        query = query.view(-1, self.heads, self.k_dim)
        # === DEBUG: check for NaNs in the incoming query
        if torch.isnan(query).any() or torch.isinf(query).any():
            print("[DEBUG:get_indices] query contains NaNs or Infs!")
            print("  query stats:", query.min(), query.max(), query.mean(), query.std())

        half = self.k_dim // 2
        # keys : (heads, 2, n_keys, half)
        # keys1 : (heads, n_keys, half)
        
        # keys = self.keys.to(query.dtype) # <- unify dtypes during einsum,casting keys to query!
        # keys = keys.view(self.heads, 2, -1, half)
        keys = self.keys.view(self.heads, 2, -1, half)
        keys1 = keys[:, 0, :, :]
        keys2 = keys[:, 1, :, :]
        n_keys = len(keys[0][0])

        # === DEBUG: check for NaNs in the keys
        if torch.isnan(keys).any() or torch.isinf(keys).any():
            print("[DEBUG:get_indices] keys contain NaNs or Infs!")
            print("  keys stats:", keys.min(), keys.max(), keys.mean(), keys.std())

        # split query for product quantization
        q1 = query[:, :, :half]  # (bs, heads, half)
        q2 = query[:, :, half:]  # (bs, heads, half)

        # === DEBUG: sanity check on q1/q2
        if torch.isnan(q1).any() or torch.isnan(q2).any():
            print("[DEBUG:get_indices] q1 or q2 contains NaNs!")

        # compute indices with associated scores
        scores1 = torch.einsum(
            "blh, lkh->blk", q1, keys1
        )  # (bs , heads, n_keys ** 0,5)
        scores2 = torch.einsum(
            "blh, lkh->blk", q2, keys2
        )  # (bs , heads, n_keys ** 0,5)

        if torch.isnan(scores1).any() or torch.isnan(scores2).any():
            print("[DEBUG:get_indices] scores1 or scores2 has NaNs before topk!")

        scores1, indices1 = scores1.topk(knn, dim=2, largest=True)  # (bs, heads, knn)
        scores2, indices2 = scores2.topk(knn, dim=2, largest=True)  # (bs, heads, knn)

        # cartesian product on best candidate keys
        all_scores = (
            scores1.view(bs, self.heads, knn, 1).expand(bs, self.heads, knn, knn)
            + scores2.view(bs, self.heads, 1, knn).expand(bs, self.heads, knn, knn)
        ).view(
            bs, self.heads, -1
        )  # (bs, heads, knn ** 2)
        all_indices = (
            indices1.view(bs, self.heads, knn, 1).expand(bs, self.heads, knn, knn)
            * n_keys
            + indices2.view(bs, self.heads, 1, knn).expand(bs, self.heads, knn, knn)
        ).view(
            bs, self.heads, -1
        )  # (bs, heads, knn ** 2)

        # === DEBUG: check combined scores before final topk
        if torch.isnan(all_scores).any():
            print("[DEBUG:get_indices] all_scores has NaNs before final topk!")

        # select overall best scores and indices
        scores, best_indices = torch.topk(
            all_scores, k=knn, dim=2, largest=True, sorted=True
        )  # (bs, heads, knn)
        indices = all_indices.gather(2, best_indices)  # (bs, knn)

        # return scores with indices
        assert scores.shape == indices.shape == (bs, self.heads, knn)
        # === DEBUG: check final scores + indices
        if torch.isnan(scores).any() or torch.isnan(indices).any():
            print("[DEBUG:get_indices] final scores/indices has NaNs")
            print("  scores stats:", scores.min(), scores.max(), scores.mean(), scores.std())

        return scores.view(bs * self.heads, knn), indices.view(bs * self.heads, knn)


class QueryMLP(nn.Module):
    def __init__(self, input_dim, heads, k_dim, sizes, bias=False, batchnorm=False):
        super().__init__()
        self.input_dim = input_dim
        self.heads = heads
        self.k_dim = k_dim
        self.sizes = sizes
        assert sizes[0] == input_dim
        assert sizes[-1] == (heads * k_dim)

        # MLPs
        sizes_ = list(sizes)
        sizes_[-1] = sizes_[-1]
        self.query_mlps = QueryMLP.mlp(sizes_, bias=bias, batchnorm=batchnorm)

    @staticmethod
    def mlp(sizes, bias=True, batchnorm=True):
        """
        Generate a feedforward neural network.
        """
        assert len(sizes) >= 2
        pairs = [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        layers = []

        for i, (dim_in, dim_out) in enumerate(pairs):
            layers.append(nn.Linear(dim_in, dim_out, bias=bias))
            if batchnorm:
                layers.append(nn.BatchNorm1d(dim_out))
            if i < len(pairs) - 1:
                layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def forward(self, input):
        """
        Compute queries using either grouped 1D convolutions or ModuleList + concat.
        """
        assert input.shape[-1] == self.input_dim
        input = (
            input.contiguous().view(-1, self.input_dim) if input.dim() > 2 else input
        )
        bs = len(input)

        outputs = [m(input) for m in self.query_mlps]
        query = torch.cat(outputs, 1) if len(outputs) > 1 else outputs[0]

        assert query.shape == (bs, self.heads * self.k_dim)
        return query.view(bs * self.heads, self.k_dim)