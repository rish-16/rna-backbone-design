# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

from rna_backbone_design.tools.rhofold_api.rhofold.model.primitives import (
    Linear, 
    LayerNorm,
    Attention, 
    GlobalAttention, 
    _attention_chunked_trainable,
)
from rna_backbone_design.tools.rhofold_api.rhofold.utils.chunk_utils import chunk_layer
from rna_backbone_design.tools.rhofold_api.rhofold.utils.tensor_utils import (
    permute_final_dims,
)


class LearnedPositionalEmbedding(nn.Embedding):
    """
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        if padding_idx is not None:
            num_embeddings_ = num_embeddings + padding_idx + 1
        else:
            num_embeddings_ = num_embeddings
        super().__init__(num_embeddings_, embedding_dim, padding_idx)
        self.max_positions = num_embeddings

    def forward(self, input: torch.Tensor):
        ''' '''
        mask = input.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + self.padding_idx
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class MSANet(nn.Module):
    def __init__(self,
                 d_model = 64,
                 d_msa = 21,
                 padding_idx = None,
                 max_len = 4096,
                 is_pos_emb = True,
                 **unused):
        super(MSANet, self).__init__()
        self.is_pos_emb = is_pos_emb

        self.embed_tokens = nn.Embedding(d_msa, d_model, padding_idx = padding_idx)
        if self.is_pos_emb:
            self.embed_positions = LearnedPositionalEmbedding(max_len, d_model, padding_idx)

    def forward(self, tokens):
        '''

        '''

        B, K, L = tokens.shape
        msa_fea = self.embed_tokens(tokens)

        if self.is_pos_emb:
            msa_fea += self.embed_positions(tokens.reshape(B * K, L)).view(msa_fea.size())

        return msa_fea

    def get_emb_weight(self):
        return self.embed_tokens.weight


class MSAAttention(nn.Module):
    def __init__(
        self,
        c_in,
        c_hidden,
        no_heads,
        pair_bias=False,
        c_z=None,
        inf=1e9,
    ):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            pair_bias:
                Whether to use pair embedding bias
            c_z:
                Pair embedding channel dimension. Ignored unless pair_bias
                is true
            inf:
                A large number to be used in computing the attention mask
        """
        super(MSAAttention, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.pair_bias = pair_bias
        self.c_z = c_z
        self.inf = inf

        self.layer_norm_m = LayerNorm(self.c_in)

        self.layer_norm_z = None
        self.linear_z = None
        if self.pair_bias:
            self.layer_norm_z = LayerNorm(self.c_z)
            self.linear_z = Linear(self.c_z, self.no_heads, bias=False)
        
        self.mha = Attention(
            self.c_in, 
            self.c_in, 
            self.c_in, 
            self.c_hidden, 
            self.no_heads,
        )

    @torch.jit.ignore
    def _chunk(self, 
        m: torch.Tensor,
        biases: List[torch.Tensor],
        chunk_size: int,
        use_memory_efficient_kernel: bool,
    ) -> torch.Tensor:
        def fn(m, biases):
            m = self.layer_norm_m(m)
            return self.mha(
                q_x=m, 
                kv_x=m, 
                biases=biases,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
            )

        return chunk_layer(
            fn,
            {
                "m": m, 
                "biases": biases, 
            },
            chunk_size=chunk_size,
            no_batch_dims=len(m.shape[:-2])
        )

    def _prep_inputs(self,
        m: torch.Tensor,
        z: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
        n_seq, n_res = m.shape[-3:-1]
        if mask is None:
            # [*, N_seq, N_res]
            mask = m.new_ones(
                m.shape[:-3] + (n_seq, n_res),
            )

        # [*, N_seq, 1, 1, N_res]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        if (self.pair_bias and 
            z is not None and                       # For the 
            self.layer_norm_z is not None and       # benefit of
            self.linear_z is not None               # TorchScript
        ):
            chunks = []

            for i in range(0, z.shape[-3], 256):
                z_chunk = z[..., i: i + 256, :, :]

                # [*, N_res, N_res, C_z]
                z_chunk = self.layer_norm_z(z_chunk)
            
                # [*, N_res, N_res, no_heads]
                z_chunk = self.linear_z(z_chunk)

                chunks.append(z_chunk)
            
            z = torch.cat(chunks, dim=-3)
            
            # [*, 1, no_heads, N_res, N_res]
            z = permute_final_dims(z, (2, 0, 1)).unsqueeze(-4)

        return m, mask_bias, z

    @torch.jit.ignore
    def _chunked_msa_attn(self,
        m: torch.Tensor,
        z: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        chunk_logits: int,
        checkpoint: bool,
        inplace_safe: bool = False
    ) -> torch.Tensor:
        """ 
        MSA attention with training-time chunking of the softmax computation.
        Saves memory in the extra MSA stack. Probably obviated by our fused 
        attention kernel, which is now used by default.
        """
        MSA_DIM = -4

        def _get_qkv(m, z):
            m, mask_bias, z = self._prep_inputs(
                m, z, mask, inplace_safe=inplace_safe
            )
            q, k, v = self.mha._prep_qkv(m, m)
            return m, q, k, v, mask_bias, z

        m, q, k, v, mask_bias, z = _get_qkv(m, z)
       
        o = _attention_chunked_trainable(
            query=q, 
            key=k, 
            value=v, 
            biases=[mask_bias, z], 
            chunk_size=chunk_logits, 
            chunk_dim=MSA_DIM,
            checkpoint=checkpoint,
        )

        m = self.mha._wrap_up(o, m)

        return m

    def forward(self, 
        m: torch.Tensor, 
        z: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None, 
        chunk_size: Optional[int] = None,
        use_memory_efficient_kernel: bool = False,
        inplace_safe: bool = False,
        _chunk_logits: Optional[int] = None,
        _checkpoint_chunks: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding. Required only if
                pair_bias is True
            mask:
                [*, N_seq, N_res] MSA mask
            chunk_size:
                Size of chunks into which the inputs are split along their
                batch dimensions. A low value decreases memory overhead at the 
                cost of slower execution. Chunking is not performed by default.
                
        """
        if(_chunk_logits is not None):
            return self._chunked_msa_attn(
                m=m, z=z, mask=mask, 
                chunk_logits=_chunk_logits, 
                checkpoint=_checkpoint_chunks,
                inplace_safe=inplace_safe,
            )           

        m, mask_bias, z = self._prep_inputs(
            m, z, mask, inplace_safe=inplace_safe
        )

        biases = [mask_bias]
        if(z is not None):
            biases.append(z)

        if chunk_size is not None:
            m = self._chunk(
                m, 
                biases, 
                chunk_size,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
            )
        else:
            m = self.mha(
                q_x=m, 
                kv_x=m, 
                biases=biases,
            )

        return m


class MSARowAttentionWithPairBias(MSAAttention):
    """
    Implements Algorithm 7.
    """

    def __init__(self, c_m, c_z, c_hidden, no_heads, inf=1e9):
        """
        Args:
            c_m:
                Input channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            inf:
                Large number used to construct attention masks
        """
        super(MSARowAttentionWithPairBias, self).__init__(
            c_m,
            c_hidden,
            no_heads,
            pair_bias=True,
            c_z=c_z,
            inf=inf,
        )


class MSAColumnAttention(nn.Module):
    """
    Implements Algorithm 8.

    By rights, this should also be a subclass of MSAAttention. Alas,
    most inheritance isn't supported by TorchScript.
    """

    def __init__(self, c_m, c_hidden, no_heads, inf=1e9):
        """
        Args:
            c_m:
                MSA channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            inf:
                Large number used to construct attention masks
        """
        super(MSAColumnAttention, self).__init__()
        
        self.c_m = c_m
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf

        self._msa_att = MSAAttention(
            c_in=c_m,
            c_hidden=c_hidden,
            no_heads=no_heads,
            pair_bias=False,
            c_z=None,
            inf=inf,
        )

    def forward(self, 
        m: torch.Tensor, 
        mask: Optional[torch.Tensor] = None, 
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            mask:
                [*, N_seq, N_res] MSA mask
            chunk_size:
                Size of chunks into which the inputs are split along their
                batch dimensions. A low value decreases memory overhead at the 
                cost of slower execution. Chunking is not performed by default.
        """ 
        # [*, N_res, N_seq, C_in]
        m = m.transpose(-2, -3)
        if mask is not None:
            mask = mask.transpose(-1, -2)

        m = self._msa_att(m, mask=mask, chunk_size=chunk_size)

        # [*, N_seq, N_res, C_in]
        m = m.transpose(-2, -3)
        if mask is not None:
            mask = mask.transpose(-1, -2)

        return m


class MSAColumnGlobalAttention(nn.Module):
    def __init__(
        self, c_in, c_hidden, no_heads, inf=1e9, eps=1e-10,
    ):
        super(MSAColumnGlobalAttention, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf
        self.eps = eps

        self.layer_norm_m = nn.LayerNorm(c_in)

        self.global_attention = GlobalAttention(
            c_in=c_in,
            c_hidden=c_hidden,
            no_heads=no_heads,
            inf=inf,
            eps=eps,
        )

    @torch.jit.ignore
    def _chunk(self,
        m: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        mha_input = {
            "m": m,
            "mask": mask,
        }

        def fn(m, mask):
            m = self.layer_norm_m(m)
            return self.global_attention(m, mask)

        return chunk_layer(
            fn,
            mha_input,
            chunk_size=chunk_size,
            no_batch_dims=len(m.shape[:-2]),
        )

    def forward(
        self, 
        m: torch.Tensor, 
        mask: Optional[torch.Tensor] = None, 
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:

        if mask is None:
            # [*, N_seq, N_res]
            mask = torch.ones(
                m.shape[:-1],
                dtype=m.dtype,
                device=m.device,
            ).detach()

        # [*, N_res, N_seq, C_in]
        m = m.transpose(-2, -3)
        mask = mask.transpose(-1, -2)

        if chunk_size is not None:
            m = self._chunk(m, mask, chunk_size)
        else:
            m = self.layer_norm_m(m)
            m = self.global_attention(m=m, mask=mask)

        # [*, N_seq, N_res, C_in]
        m = m.transpose(-2, -3)

        return m
