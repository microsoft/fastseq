# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Apply the EL Attention optimizations to fairseq-v0.10.2"""

import math
from typing import Optional, List, NamedTuple, Dict, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor
from fairseq import utils
from fairseq.models.transformer import TransformerEncoder, TransformerDecoder, TransformerModel
from fairseq.modules.multihead_attention import MultiheadAttention
from fairseq.sequence_generator import SequenceGenerator, EnsembleModel
from fairseq.models.fairseq_model import FairseqEncoderDecoderModel
from fairseq.modules.transformer_layer import TransformerDecoderLayer
from fairseq.tasks.fairseq_task import FairseqTask
from fastseq.utils.api_decorator import replace
from fastseq.ops.ngram_repeat_block import NGramRepeatBlock
from fastseq.config import USE_EL_ATTN

@replace(FairseqTask, USE_EL_ATTN)
class FairseqTask(FairseqTask):
    def transpose_enc_dec_kv_proj(self, models):
        for model in models:
            model.transpose_enc_dec_kv_proj()

@replace(TransformerDecoderLayer, USE_EL_ATTN)
class TransformerDecoderLayer(TransformerDecoderLayer):
    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_out_v: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).
        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        # torch.cuda.nvtx.range_push('self attn')
        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        # torch.cuda.nvtx.range_pop()
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            # torch.cuda.nvtx.range_push('enc dec attn')
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out_v,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            # torch.cuda.nvtx.range_pop()
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None


@replace(TransformerEncoder, USE_EL_ATTN)
class TransformerEncoder(TransformerEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """
    @classmethod
    def create_named_tuple(cls):
        EncoderOut = NamedTuple(
            "TransformerEncoderOut",
            [
                ("encoder_out", Tensor),  # T x B x C
                ("encoder_out_v", Tensor), # B x T x C
                ("encoder_padding_mask", Optional[Tensor]),  # B x T
                ("encoder_embedding", Optional[Tensor]),  # B x T x C
                ("encoder_states", Optional[List[Tensor]]),  # List[T x B x C]
                ("src_tokens", Optional[Tensor]),  # B x T
                ("src_lengths", Optional[Tensor]),  # B x 1
            ]
        )
        return EncoderOut

    def forward(
        self,
        src_tokens,
        src_lengths,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings
        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """ 
        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        EncoderOut = self.create_named_tuple()
        return EncoderOut(
            encoder_out=x.permute(1, 2, 0).contiguous(),  # B x C x T
            encoder_out_v=x.permute(1, 0, 2).contiguous(), # B x T x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )


    @torch.jit.export
    def reorder_encoder_out(self, encoder_out, new_order, beam_size):
        """
        Reorder encoder output according to *new_order*.
        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order
        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        """
        Since encoder_padding_mask and encoder_embedding are both of type
        Optional[Tensor] in EncoderOut, they need to be copied as local
        variables for Torchscript Optional refinement
        """
        encoder_padding_mask: Optional[Tensor] = encoder_out.encoder_padding_mask
        encoder_embedding: Optional[Tensor] = encoder_out.encoder_embedding
        encoder_states = encoder_out.encoder_states

        new_encoder_out = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(
                            0,
                            new_order.reshape(-1, beam_size)[:, 0] // beam_size)
        )
        new_encoder_out_v = (
            encoder_out.encoder_out_v
            if encoder_out.encoder_out_v is None
            else encoder_out.encoder_out_v.index_select(
                0,
                new_order.reshape(-1, beam_size)[:, 0] // beam_size)
        )
        new_encoder_padding_mask = (
            encoder_padding_mask
            if encoder_padding_mask is None
            else encoder_padding_mask.index_select(
                            0,
                            new_order.reshape(-1, beam_size)[:, 0] // beam_size)
        )
        src_tokens = encoder_out.src_tokens
        if src_tokens is not None:
            src_tokens = src_tokens.index_select(0, new_order)

        src_lengths = encoder_out.src_lengths
        if src_lengths is not None:
            src_lengths = src_lengths.index_select(0, new_order)

        EncoderOut = TransformerEncoder.create_named_tuple()
        return EncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            encoder_out_v=new_encoder_out_v, # T x B x C
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,  # B x T
            src_lengths=src_lengths,  # B x 1
        )


@replace(EnsembleModel, USE_EL_ATTN)
class EnsembleModel(EnsembleModel):
    """A wrapper around an ensemble of models."""

    def transpose_enc_dec_kv_proj(self):
        for model in self.models:
            model.transpose_enc_dec_kv_proj()

    # @torch.jit.export
    def reorder_encoder_out(self, encoder_outs, new_order, beam_size):
        new_outs = []
        if not self.has_encoder():
            return new_outs
        for i, model in enumerate(self.models):
            assert encoder_outs is not None
            new_outs.append(
                model.encoder.reorder_encoder_out(encoder_outs[i], new_order, beam_size)
            )
        return new_outs


@replace(TransformerDecoder, USE_EL_ATTN)
class TransformerDecoder(TransformerDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).
        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_out_v if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}


@replace(FairseqEncoderDecoderModel, USE_EL_ATTN)
class FairseqEncoderDecoderModel(FairseqEncoderDecoderModel):
    """class for encoder-decoder models.
    Args:
        encoder (FairseqEncoder): the encoder
        decoder (FairseqDecoder): the decoder
    """

    def transpose_enc_dec_kv_proj (self):
        for i in range (0, len(self.decoder.layers)):
            self.num_heads = self.decoder.layers[i].encoder_attn.num_heads
            self.head_dim = self.decoder.layers[i].encoder_attn.head_dim

            self.decoder.layers[i].encoder_attn.k_proj_weight_t = (
                    self.decoder.layers[i].encoder_attn.k_proj.weight
                    .view(self.num_heads,
                        self.head_dim, self.num_heads * self.head_dim)
                    ).cuda()
            self.decoder.layers[i].encoder_attn.k_proj_bias_t = (
                    self.decoder.layers[i].encoder_attn.k_proj.bias
                    .view(self.num_heads, self.head_dim, 1)
                    ).cuda()

            self.decoder.layers[i].encoder_attn.v_proj_weight_t = (
                self.decoder.layers[i].encoder_attn.v_proj.weight
                .view(self.num_heads, self.head_dim,
                    self.num_heads * self.head_dim)
                .transpose(1, 2)
                .contiguous()
                ).cuda()
            self.decoder.layers[i].encoder_attn.v_proj_bias_t = (
                self.decoder.layers[i].encoder_attn.v_proj.bias
                .view(1, 1, self.num_heads * self.head_dim)
                ).cuda()

            del self.decoder.layers[i].encoder_attn.k_proj
            del self.decoder.layers[i].encoder_attn.v_proj


@replace(TransformerModel, USE_EL_ATTN)
class TransformerModel(TransformerModel):
    """ Represent the BART model."""
    def make_generation_fast_(self, **kwargs):
        super().make_generation_fast_(**kwargs)  # pylint: disable=bad-super-call

@replace(MultiheadAttention, USE_EL_ATTN)
class MultiheadAttention(MultiheadAttention):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self,
                 embed_dim,
                 num_heads,
                 kdim=None,
                 vdim=None,
                 dropout=0.,
                 bias=True,
                 add_bias_kv=False,
                 add_zero_attn=False,
                 self_attention=False,
                 encoder_decoder_attention=False,
                 q_noise=0.0,
                 qn_block_size=8,):
        super().__init__(embed_dim, num_heads, kdim, vdim, dropout, bias,
                         add_bias_kv, add_zero_attn, self_attention,
                         encoder_decoder_attention, q_noise, qn_block_size)

        self.beam_size = 1
        self.tpu = False
        self.k_proj_weight_t = torch.empty([0])
        self.k_proj_bias_t = torch.empty([0])
        self.v_proj_weight_t = torch.empty([0])
        self.v_proj_bias_t = torch.empty([0])

    def apply_sparse_mask(
        self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel
        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        kv_bsz = -1
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if (
            not self.onnx_trace
            and not self.tpu  # don't use PyTorch version on TPUs
            and incremental_state is None
            and not static_kv
            # A workaround for quantization to work. Otherwise JIT compilation
            # treats bias in linear module as method.
            and not torch.jit.is_scripting()
        ):
            assert key is not None and value is not None
            return F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                torch.empty([0]),
                torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout_module.p,
                self.out_proj.weight,
                self.out_proj.bias,
                self.training or self.dropout_module.apply_during_inference,
                key_padding_mask,
                need_weights,
                attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
            )

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
        else:
            saved_state = None

        # Get q, k, v
        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            assert key is not None
            kv_bsz = key.size(0)
            embed_dim = key.size(1)
            tgt_len = 1
            # torch.cuda.nvtx.range_push('Q reshape')
            q = torch.addmm(self.q_proj.bias.view(1, -1),
                    query.view(-1,embed_dim), self.q_proj.weight.T,
                    beta=self.scaling, alpha=self.scaling)
            q = (
                q
                .view(bsz, self.num_heads, self.head_dim)
                .transpose(0, 1)
                )
            # torch.cuda.nvtx.range_pop()
            k, v = None, None
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        
        if not self.encoder_decoder_attention:
            q *= self.scaling
            if self.bias_k is not None:
                assert self.bias_v is not None
                k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
                v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
                if attn_mask is not None:
                    attn_mask = torch.cat(
                        [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                    )
                if key_padding_mask is not None:
                    key_padding_mask = torch.cat(
                        [
                            key_padding_mask,
                            key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                        ],
                        dim=1,
                    )
            q = (
                q.contiguous()
                .view(tgt_len, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

            if k is not None:
                kv_bsz = k.size(1)
                k = (
                    k.contiguous()
                    .view(-1, kv_bsz * self.num_heads, self.head_dim)
                    .transpose(0, 1)
                )
            if v is not None:
                assert kv_bsz
                v = (
                    v.contiguous()
                    .view(-1, kv_bsz * self.num_heads, self.head_dim)
                    .transpose(0, 1)
                )

            if saved_state is not None:
                # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
                if "prev_key" in saved_state:
                    _prev_key = saved_state["prev_key"]
                    assert _prev_key is not None
                    kv_bsz = _prev_key.size(0)
                    prev_key = _prev_key.view(kv_bsz * self.num_heads, -1, self.head_dim)
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
                if "prev_value" in saved_state:
                    _prev_value = saved_state["prev_value"]
                    assert _prev_value is not None
                    assert kv_bsz == _prev_value.size(0)
                    prev_value = _prev_value.view(kv_bsz * self.num_heads, -1, self.head_dim)
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
                prev_key_padding_mask: Optional[Tensor] = None
                if "prev_key_padding_mask" in saved_state:
                    prev_key_padding_mask = saved_state["prev_key_padding_mask"]
                assert k is not None and v is not None
                key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                    key_padding_mask=key_padding_mask,
                    prev_key_padding_mask=prev_key_padding_mask,
                    batch_size=kv_bsz,
                    src_len=k.size(1),
                    static_kv=static_kv,
                )

                saved_state["prev_key"] = k.view(kv_bsz, self.num_heads, -1, self.head_dim)
                saved_state["prev_value"] = v.view(kv_bsz, self.num_heads, -1, self.head_dim)
                saved_state["prev_key_padding_mask"] = key_padding_mask
                # In this branch incremental_state is never None
                assert incremental_state is not None
                incremental_state = self._set_input_buffer(incremental_state, saved_state)
        
        assert key is not None
        src_len = key.size(2)
        if not self.encoder_decoder_attention:
            assert k is not None
            src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if self.add_zero_attn:
            assert v is not None and k is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        if self.encoder_decoder_attention:
            # torch.cuda.nvtx.range_push('bmm_q_k_proj_weight')
            q_w = torch.bmm(q, self.k_proj_weight_t)
            # torch.cuda.nvtx.range_pop()
            # torch.cuda.nvtx.range_push('bmm_q_k_proj_bias')
            q_b = torch.bmm(q, self.k_proj_bias_t)
            # torch.cuda.nvtx.range_pop()
            # torch.cuda.nvtx.range_push('q_w_reshape')
            q_b = (q_b.view(self.num_heads, kv_bsz, self.beam_size, 1)
                    .transpose(0,1)
                    .reshape(kv_bsz, self.num_heads*self.beam_size, 1)
                  )

            q_w = (q_w.view(self.num_heads, kv_bsz, self.beam_size, embed_dim)
                    .transpose(0,1)
                    .contiguous()
                    .view(kv_bsz, self.num_heads*self.beam_size, embed_dim)
                  )
            # torch.cuda.nvtx.range_pop()

            # torch.cuda.nvtx.range_push('bmm_q_w_key')
            attn_weights = torch.bmm(q_w, key)
            # torch.cuda.nvtx.range_pop()
            # torch.cuda.nvtx.range_push('add_attn_weight_q_b')
            attn_weights = attn_weights + q_b
            # torch.cuda.nvtx.range_pop()
        else:
            assert k is not None
            # torch.cuda.nvtx.range_push('Q_K')
            attn_weights = torch.bmm(q, k.transpose(1, 2))
            # torch.cuda.nvtx.range_pop()
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not self.tpu:
                if not self.encoder_decoder_attention:
                    attn_weights = attn_weights.masked_fill(
                        key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                        float("-inf"),
                    )
                else:
                    attn_weights = attn_weights.view(kv_bsz, self.num_heads,-1,
                                            tgt_len, src_len)
                    attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(
                            torch.bool), float("-inf"))
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                attn_weights = attn_weights.transpose(0, 2)
            if not self.encoder_decoder_attention:
                attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        if self.encoder_decoder_attention:
            assert value is not None
            attn_probs = attn_probs.view(
                    kv_bsz, self.num_heads*self.beam_size*tgt_len, src_len)

            # torch.cuda.nvtx.range_push('bmm_attn_prob_value')
            attn_h = torch.bmm(attn_probs, value)
            # torch.cuda.nvtx.range_pop()

            # torch.cuda.nvtx.range_push('attn_h_reshape')
            attn_h = (attn_h.view(kv_bsz,
                self.num_heads, self.beam_size, embed_dim)
                .transpose(0,1)
                .contiguous()
                .view(self.num_heads, kv_bsz*self.beam_size, embed_dim)
               )
            # torch.cuda.nvtx.range_pop()

            # torch.cuda.nvtx.range_push('bmm_attn_h_v_proj_weight')
            attn = torch.bmm(attn_h, self.v_proj_weight_t)
            # torch.cuda.nvtx.range_pop()

            # torch.cuda.nvtx.range_push('attn reshape')
            attn = (attn
                    .transpose(0,1)
                    .contiguous()
                    .view(1, kv_bsz*self.beam_size,
                        self.num_heads*self.head_dim)
                   )
            # torch.cuda.nvtx.range_pop()

            # (1, kv_bsz*beam, self.num_heads*self.head_dim)
            # torch.cuda.nvtx.range_push('add_attn_v_proj_bias')
            attn = attn + self.v_proj_bias_t
            # torch.cuda.nvtx.range_pop()

            # torch.cuda.nvtx.range_push('attn_reshape')
            attn = attn.view(1, -1, self.head_dim).transpose(0,1).contiguous()
            # torch.cuda.nvtx.range_pop()

        else:
            assert v is not None
            # torch.cuda.nvtx.range_push('A_V')
            attn = torch.bmm(attn_probs, v)
            # torch.cuda.nvtx.range_pop()
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)
        else:
            attn_weights = None

        return attn, attn_weights

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer[k] is not None:
                    if self.encoder_decoder_attention:
                        if (input_buffer_k.size(0) * self.beam_size == new_order.size(0)):
                            return incremental_state
                        elif (input_buffer_k.size(0) == new_order.size(0)):
                            break
                        elif self.beam_size > 1:
                            input_buffer[k] = input_buffer_k.index_select(
                                0,
                                new_order.reshape(-1, self.beam_size)[:, 0] //
                                self.beam_size)
                        else:
                            input_buffer[k] = input_buffer_k.index_select(
                                0, new_order)
                    else:
                        input_buffer[k] = input_buffer_k.index_select(
                            0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def set_beam_size(self, beam_size):
        self.beam_size = beam_size

    def make_generation_fast_(self, beamable_mm_beam_size=None, **kwargs):
        if beamable_mm_beam_size is not None:
            self.set_beam_size(beamable_mm_beam_size)


@replace(SequenceGenerator, USE_EL_ATTN)
class SequenceGenerator(SequenceGenerator):
    """
    Sequence Generator is optimized by reducing the cached memory usage
    during the encoding period for beam search.
    """

    @torch.no_grad()
    def _generate(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        constraints: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(self.model.models_size)
            ],
        )
        net_input = sample["net_input"]

        if "src_tokens" in net_input:
            src_tokens = net_input["src_tokens"]
            # length of the source text being the character length except EndOfSentence and pad
            src_lengths = (
                (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
            )
        elif "source" in net_input:
            src_tokens = net_input["source"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        else:
            raise Exception("expected src_tokens or source in net input")

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimenions (i.e. audio features)
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size
        self.no_repeat_ngram_op = NGramRepeatBlock()

        if constraints is not None and not self.search.supports_constraints:
            raise NotImplementedError(
                "Target-side constraints were provided, but search method doesn't support them"
            )

        # Initialize constraints, when active
        self.search.init_constraints(constraints, beam_size)

        max_len: int = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                self.model.max_decoder_positions() - 1,
            )
        assert (
            self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"

        EncoderOut = TransformerEncoder.create_named_tuple()

        def merge_encoder_out(encoder_out_list: List[Optional[EncoderOut]]):
            encoder_out = torch.cat([
                o.encoder_out for o in encoder_out_list], dim=0)
            false_mask=None
            if not any([
                o.encoder_padding_mask != None for o in encoder_out_list]):
                encoder_padding_mask = None
            else:
                masks = [o.encoder_padding_mask
                        if o.encoder_padding_mask != None
                        else torch.zeros(
                            (o.encoder_out.size(0), o.encoder_out.size(1)),
                            dtype = torch.bool, device=encoder_out.device)
                        for o in encoder_out_list]
                encoder_padding_mask = torch.cat(masks, dim=0)#.to(encoder_out.device)

            encoder_embedding = torch.cat(
                    [o.encoder_embedding for o in encoder_out_list], dim=0)
            encoder_out_v = torch.cat([
                o.encoder_out_v for o in encoder_out_list], dim=0)

            return [EncoderOut(
                encoder_out=encoder_out,  # B x C x T
                encoder_padding_mask=encoder_padding_mask,  # B x T
                encoder_embedding=encoder_embedding,  # B x T x C
                encoder_out_v=encoder_out_v,  # B x T x C
                encoder_states=None,  # List[T x B x C]
                src_tokens=None,
                src_lengths=None,
            )]

        # compute the encoder output for each beam
        max_batch_size = math.ceil(2_147_483_647 / (src_len*src_len*16) / 4)
        sub_batch_size = 1
        while sub_batch_size * 2 <= max_batch_size:
            sub_batch_size *= 2

        loop_num = (bsz + sub_batch_size - 1) // sub_batch_size

        if loop_num > 1:
            #assert token_embeddings is None, "not support split token_embeddings yet"
            split_src_tokens = torch.split(src_tokens, sub_batch_size)
            split_src_lengths = torch.split(src_lengths, sub_batch_size)
            encoder_out_list = []
            for sub_src_tokens, sub_src_lengths in zip(
                    split_src_tokens, split_src_lengths):
                split_input = {'src_tokens': sub_src_tokens,
                               'src_lengths': sub_src_lengths}
                split_output = self.model.forward_encoder(split_input)
                encoder_out_list.append(split_output[0])
            encoder_outs = merge_encoder_out(encoder_out_list)
        else:
            encoder_outs = self.model.forward_encoder(net_input)

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        # ensure encoder_outs is a List.
        assert encoder_outs is not None

        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        )  # +1 for eos; pad is never chosen for scoring
        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
            .to(src_tokens)
            .long()
            .fill_(self.pad)
        )  # +2 for eos and pad
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        attn: Optional[Tensor] = None

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = (
            torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        finished = [
            False for i in range(bsz)
        ]  # a boolean array indicating if the sentence at the index is finished or not
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None

        original_batch_idxs: Optional[Tensor] = None
        if "id" in sample and isinstance(sample["id"], Tensor):
            original_batch_idxs = sample["id"]
        else:
            original_batch_idxs = torch.arange(0, bsz).type_as(tokens)

        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size
                    )
                    original_batch_idxs = original_batch_idxs[batch_idxs]
                self.model.reorder_incremental_state(incremental_states, reorder_state)
                encoder_outs = self.model.reorder_encoder_out(
                    encoder_outs, reorder_state, beam_size
                )

            lprobs, avg_attn_scores = self.model.forward_decoder(
                tokens[:, : step + 1],
                encoder_outs,
                incremental_states,
                self.temperature,
            )

            if self.lm_model is not None:
                lm_out = self.lm_model(tokens[:, : step + 1])
                probs = self.lm_model.get_normalized_probs(
                    lm_out, log_probs=True, sample=None
                )
                probs = probs[:, -1, :] * self.lm_weight
                lprobs += probs

            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # handle max length constraint
            if step >= max_len:
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos + 1 :] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            if (
                prefix_tokens is not None
                and step < prefix_tokens.size(1)
                and step < max_len
            ):
                lprobs, tokens, scores = self._prefix_tokens(
                    step, lprobs, scores, tokens, prefix_tokens, beam_size
                )
            elif step < self.min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.eos] = -math.inf

            # Record attention scores, only support avg_attn_scores is a Tensor
            if avg_attn_scores is not None:
                if attn is None:
                    attn = torch.empty(
                        bsz * beam_size, avg_attn_scores.size(1), max_len + 2
                    ).to(scores)
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            eos_bbsz_idx = torch.empty(0).to(
                tokens
            )  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = torch.empty(0).to(
                scores
            )  # scores of hypothesis ending with eos (finished sentences)

            if self.should_set_src_lengths:
                self.search.set_src_lengths(src_lengths)

            if self.no_repeat_ngram_size > 0:
                if (tokens.is_cuda and lprobs.is_cuda):
                    lprobs = self.no_repeat_ngram_op(tokens,lprobs, bsz, step,
                            beam_size, self.no_repeat_ngram_size)
                else:
                    lprobs = self._no_repeat_ngram(tokens, lprobs, bsz, beam_size, step)

            # Shape: (batch, cand_size)
            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
                tokens[:, : step + 1],
                original_batch_idxs,
            )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            # Shape of eos_mask: (batch size, beam size)
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

            # only consider eos when it's among the top beam_size indices
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )

            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )

                finalized_sents = self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    attn,
                    src_lengths,
                    max_len,
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if self.search.stop_on_max_len and step >= max_len:
                break
            assert step < max_len

            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(
                    bsz, dtype=torch.bool, device=cand_indices.device
                )
                batch_mask[finalized_sents] = False
                # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
                batch_idxs = torch.arange(
                    bsz, device=cand_indices.device
                ).masked_select(batch_mask)

                # Choose the subset of the hypothesized constraints that will continue
                self.search.prune_sentences(batch_idxs)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, attn.size(1), -1
                    )
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.

            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False
            )

            # update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            # Make sure there is at least one active item for each sentence in the batch.
            assert (~cands_to_ignore).any(dim=1).all()

            # update cands_to_ignore to ignore any finalized hypos

            # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
            # can be selected more than once).
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses

            # Set the tokens for each beam (can select the same row more than once)
            tokens[:, : step + 1] = torch.index_select(
                tokens[:, : step + 1], dim=0, index=active_bbsz_idx
            )
            # Select the next token for each of them
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
                cand_indices, dim=1, index=active_hypos
            )
            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx
                )
            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )

            # Update constraints based on which candidates were selected for the next beam
            self.search.update_constraints(active_hypos)

            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, : step + 2] = torch.index_select(
                    attn[:, :, : step + 2], dim=0, index=active_bbsz_idx
                )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            scores = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[sent]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, Tensor]], finalized[sent]
            )
            
        return finalized