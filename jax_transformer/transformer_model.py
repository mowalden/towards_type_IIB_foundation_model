import jax  # lax.dynamic_update_slice writes one token into the inference cache.
import jax.numpy as jnp
from flax import linen as nn


class MultiheadAttention(nn.Module):
    r"""Multiple self-attention heads in parallel.

    Attributes:
        num_heads (int): Number of self-attention heads.
        dropout (float): Probability of dropout.
    """

    num_heads: int
    dropout: float

    @nn.compact
    def __call__(
        self, x, training: bool, encoder_output=None, mask=None,
        # Keyword-only inference options; defaults preserve full-forward calls.
        # self_kv is writable loop state; cross_kv contains fixed encoder K/V.
        *, self_kv=None, cross_kv=None, decode_position=None, prepare_cross=False,
    ):
        r"""Computes multi-head attention: MultiHead(Q,K,V) = Concat(head_1,...,head_n)W^O

        Args:
            x (jax.Array or None): Query input [batch_size, seq_length, embed_dim].
                Use seq_length=1 for cached decoding; None when prepare_cross=True.
            training (bool): Controls whether dropout is applied.
            encoder_output (jax.Array or None): Encoder activations
                [encoder_batch, source_length, embed_dim] for cross-attention.
                Required for prepare_cross; omitted for cached cross-attention.
            mask (jax.Array or None): Nonzero entries allow attention. Must
                broadcast to [batch_size, num_heads, query_length, key_length].
                Cached self-attention constructs its own causal mask instead.
            self_kv (tuple or None): Self-attention (K, V) buffers, each
                [batch_size, num_heads, cache_length, head_size]. Requires a
                single-token x and decode_position. None selects no self cache.
            cross_kv (tuple or None): Preprojected encoder (K, V), each
                [encoder_batch, num_heads, source_length, head_size]. A singleton
                encoder batch broadcasts across samples. Replaces encoder_output.
            decode_position (integer scalar or None): Zero-based token position
                to update in self_kv; must lie within the cache capacity.
            prepare_cross (bool): If True, project encoder_output to K/V and
                return immediately; x, mask, and supplied caches are unused.

        Returns:
            Normally, projected activations [batch_size, seq_length, embed_dim].
            With self_kv, returns (activations, updated (K, V)).
            With prepare_cross, returns only the projected encoder (K, V).

        Raises:
            AssertionError: If embed_dim not divisible by num_heads.
        """
        if prepare_cross:
            # Project a single encoder prompt once; retain the singleton batch
            # dimension so all sampled sequences share these keys and values.
            batch_size, seq_len, embed_dim = encoder_output.shape
            # Use the original cross-attention projection and its checkpoint name.
            # The 2*embed_dim output concatenates K and V; BF16 is compute dtype,
            # while the saved parameter dtype is left at the existing default.
            kv = nn.Dense(
                2 * embed_dim, kernel_init=nn.initializers.xavier_uniform(),
                use_bias=False, name="kv_dense", dtype=jnp.bfloat16,
            )(encoder_output)
            # Split K/V, then split embedding channels into heads and transpose
            # [batch, source_length, heads, head_size] to
            # [batch, heads, source_length, head_size] for attention matmuls.
            # Return only these arrays: preparing cross K/V needs no queries.
            return tuple(
                part.reshape(batch_size, seq_len, self.num_heads,
                             embed_dim // self.num_heads).transpose(0, 2, 1, 3)
                for part in jnp.split(kv, 2, axis=-1)
            )

        batch_size, seq_len, embed_dim = x.shape
        assert (
            embed_dim % self.num_heads == 0
        ), f"Embedding dimension must be divisible by numer of heads but got embed_dim={embed_dim} and num_heads={self.num_heads}."
        head_size = embed_dim // self.num_heads

        # No encoder data means self-attention. Cached cross-attention supplies
        # cross_kv instead of encoder_output, so it must take the other branch.
        if encoder_output is None and cross_kv is None:
            QKV = nn.Dense(
                3 * embed_dim,
                kernel_init=nn.initializers.xavier_uniform(),
                use_bias=False,
                name="qkv_dense",
                dtype=jnp.bfloat16
            )(x)
            Q, K, V = jnp.split(QKV, 3, axis=-1)
            Q, K, V = map(
                lambda x: x.reshape(batch_size, seq_len, self.num_heads, head_size),
                (Q, K, V),
            )
            Q, K, V = map(lambda x: jnp.transpose(x, (0, 2, 1, 3)), (Q, K, V))
            if self_kv is not None:
                # Only the current token is projected. Prior tokens stay in
                # the fixed-size cache carried by the generation loop.
                # Python zeros become int64 under x64 mode; all start indices
                # must instead match the explicitly int32 decode position.
                start = (jnp.int32(0), jnp.int32(0),
                         jnp.asarray(decode_position, dtype=jnp.int32), jnp.int32(0))
                # Start coordinates are (batch, head, token position, channel).
                # K is currently [B, heads, 1, head_size]; replace that one slot
                # and retain the entire fixed-size key buffer as the new K.
                K = jax.lax.dynamic_update_slice(
                    self_kv[0], K, start
                )
                # Perform the identical one-slot update for values.
                V = jax.lax.dynamic_update_slice(
                    self_kv[1], V, start
                )
                # Allow all cached tokens through the CURRENT position. Future
                # slots are unfilled and must contribute zero attention weight.
                # Singleton axes broadcast this mask across batches/heads/queries.
                mask = (jnp.arange(K.shape[2]) <= decode_position)[None, None, None, :]
        else:
            Q = nn.Dense(
                embed_dim,
                kernel_init=nn.initializers.xavier_uniform(),
                use_bias=False,
                name="q_dense",
                dtype=jnp.bfloat16
            )(x)
            Q = Q.reshape(batch_size, seq_len, self.num_heads, head_size)
            Q = jnp.transpose(Q, (0, 2, 1, 3))
            if cross_kv is not None:
                # Reuse projected encoder keys/values. Their batch size may be
                # one: matmul broadcasts them against all B decoder queries.
                K, V = cross_kv
            else:
                # Original full-forward path: project the supplied encoder
                # outputs, split K/V, and arrange both as [B, heads, source, dim].
                seq_len_enc = encoder_output.shape[1]
                KV = nn.Dense(
                    2 * embed_dim,
                    kernel_init=nn.initializers.xavier_uniform(),
                    use_bias=False, name="kv_dense", dtype=jnp.bfloat16,
                )(encoder_output)
                K, V = jnp.split(KV, 2, axis=-1)
                K, V = map(
                    lambda x: x.reshape(batch_size, seq_len_enc, self.num_heads, head_size),
                    (K, V),
                )
                K, V = map(lambda x: jnp.transpose(x, (0, 2, 1, 3)), (K, V))

        logits = (Q @ jnp.swapaxes(K, -1, -2)) / jnp.sqrt(head_size)
        if mask is not None:
            logits = jnp.where(mask == 0, -1e10, logits)
        attn = nn.softmax(logits, axis=-1)
        attn = nn.Dropout(rate=self.dropout, deterministic=not training)(attn)

        out = attn @ V
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(batch_size, seq_len, embed_dim)
        out = nn.Dense(
            embed_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            use_bias=False,
            name="out_dense",
            dtype=jnp.bfloat16
        )(out)
        out = nn.Dropout(self.dropout, deterministic=not training)(out)
        # Cached self-attention returns the UPDATED full buffers for the next
        # token. Ordinary attention and cached cross-attention return only out.
        return (out, (K, V)) if self_kv is not None else out


class FeedForward(nn.Module):
    r"""Position-wise feed-forward network.

    Attributes:
        embed_dim (int): Embedding dimension.
        dropout (float): Probability of dropout.
    """

    embed_dim: int
    dropout: float

    @nn.compact
    def __call__(self, x, training: bool):
        r"""Computes forward call: FFN(x)=max(0,xW_1+b_1)W_2+b_2

        Args:
            x (jnp.ndarray([batch_size, seq_len, embed_dim])): Batched input of embedded token IDs.
            training (bool): Controls whether dropout is applied.

        Returns:
            out (jnp.ndarray([batch_size, seq_len, embed_dim])): Output.
        """

        out = nn.Dense(
            self.embed_dim * 4,
            kernel_init=nn.initializers.xavier_uniform(),
            name="ffn_dense_1",
            dtype=jnp.bfloat16
        )(x)
        out = nn.relu(out)
        out = nn.Dense(
            self.embed_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            name="ffn_dense_2",
            dtype=jnp.bfloat16
        )(out)
        out = nn.Dropout(self.dropout, deterministic=not training)(out)
        return out


class DecoderBlock(nn.Module):
    r"""A single decoder block.

    Attributes:
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        dropout (float): Propability of dropout.
    """

    embed_dim: int
    num_heads: int
    dropout: float

    @nn.compact
    def __call__(
        self, x, training: bool, encoder_output=None, mask_self=None, mask_cross=None,
        *, self_kv=None, cross_kv=None, decode_position=None, prepare_cross=False,
    ):
        r"""Performs all computations contained in a single decoder block.

        Args:
            x (jax.Array or None): Decoder activations [batch_size, seq_len,
                embed_dim]; seq_len=1 for cached decoding, None for prepare_cross.
            training (bool): Controls dropout; use False for cached inference.
            encoder_output (jax.Array or None): Encoder activations
                [batch_size, source_length, embed_dim] for ordinary cross-attention
                or cache preparation. Not needed when cross_kv is supplied.
            mask_self (jax.Array or None): Self-attention mask broadcastable to
                [batch_size, num_heads, seq_len, seq_len]. Cached self-attention
                builds its own mask from decode_position instead.
            mask_cross (jax.Array or None): Encoder attention mask broadcastable
                to [batch_size, num_heads, seq_len, source_length]. Required with
                encoder_output during full forward; optional with cross_kv.
            self_kv (tuple or None): This layer's self-attention (K, V), each
                [batch_size, num_heads, cache_length, head_size].
            cross_kv (tuple or None): This layer's projected encoder (K, V), each
                [encoder_batch, num_heads, source_length, head_size]; encoder_batch
                can be one to share a prompt across decoder rows.
            decode_position (integer scalar or None): Zero-based position of the
                single token whose keys/values are written into self_kv.
            prepare_cross (bool): Return only this layer's encoder K/V projection;
                requires encoder_output and skips the decoder computations.

        Returns:
            Normally, decoder activations [batch_size, seq_len, embed_dim].
            With self_kv, returns (activations, updated self_kv).
            With prepare_cross, returns only the projected encoder (K, V).
        """
        if prepare_cross:
            # Visit only this block's named cross-attention projection. None
            # means no decoder input is needed; False disables training behavior.
            return MultiheadAttention(self.num_heads, self.dropout, name="cross_attn")(
                None, False, encoder_output=encoder_output, prepare_cross=True,
            )
        attention = MultiheadAttention(self.num_heads, self.dropout, name="self_attn")(
            # Supply this layer's cache and position; None preserves normal use.
            x, training, mask=mask_self, self_kv=self_kv,
            decode_position=decode_position,
        )
        if self_kv is not None:
            # Separate the attention activation from the updated cache pair.
            attention, self_kv = attention
        x = nn.LayerNorm(name="ln_1")(x + attention)

        if cross_kv is not None:
            # Query fixed encoder K/V using the current token's activation.
            attention = MultiheadAttention(
                self.num_heads, self.dropout, name="cross_attn"
            )(x, training, cross_kv=cross_kv, mask=mask_cross)
            # Preserve the trained residual connection and normalization weights.
            x = nn.LayerNorm(name="ln_2")(x + attention)
        elif encoder_output is not None:
            if mask_cross is not None:
                attention = MultiheadAttention(
                    self.num_heads, self.dropout, name="cross_attn"
                )(x, training, encoder_output, mask=mask_cross)
                x = nn.LayerNorm(name="ln_2")(x + attention)
            else:
                raise AssertionError("Got encoder_output but mask_cross is None.")

        x = nn.LayerNorm(name="ln_3")(
            x + FeedForward(self.embed_dim, self.dropout)(x, training)
        )

        # Propagate this layer's updated cache alongside its decoder activation.
        return (x, self_kv) if self_kv is not None else x


class EncoderBlock(nn.Module):
    r"""A single encoder block.

    Attributes:
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        dropout (float): Probability of dropout.
    """

    embed_dim: int
    num_heads: int
    dropout: float

    @nn.compact
    def __call__(self, x, training: bool, mask=None):
        r"""Performs all computations contained in a single encoder block.

        Args:
            x (jnp.ndarray([batch_size, seq_len, embed_dim])): Batched input of embedded token IDs.
            mask (jax.Array or None): Attention mask broadcastable to [batch_size, num_heads, seq_len, seq_len]; zero entries are masked.
            training (bool): Controls whether dropout is applied.

        Returns:
            out (jnp.ndarray([batch_size, seq_len, embed_dim])): Output.
        """
        attention = MultiheadAttention(self.num_heads, self.dropout, name="self_attn")(
            x, training, mask=mask
        )
        x = nn.LayerNorm(name="ln_1")(x + attention)
        out = nn.LayerNorm(name="ln_2")(
            x + FeedForward(self.embed_dim, self.dropout)(x, training)
        )
        return out


class DecoderModel(nn.Module):
    r"""A decoder-only model.

    Attributes:
        num_layers (int): Number of layers.
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads per layer.
        vocab_size (int): Size of vocabulary.
        max_len (int): Maximum length of sequences.
        dropout (float): Probability of dropout.
    """

    num_layers: int
    embed_dim: int
    num_heads: int
    vocab_size: int
    max_len: int
    dropout: float

    @nn.compact
    def __call__(self, input_ids, mask=None, training=False):
        r"""Performs forward-pass of the model.

        Args:
            input_ids (jnp.ndarray([batch_size, seq_len])): Batched input of token IDs.
            mask (jax.Array or None): Decoder attention/padding mask; zero entries indicate masked positions.
            training (bool): Controls whether dropout is applied.

        Returns:
            out (jnp.ndarray([batch_size, seq_len, vocab_size])): Logits for next token prediction.
        """
        seq_len = input_ids.shape[1]
        causal_mask = jnp.tril(jnp.ones((self.max_len, self.max_len)))
        seq_len = input_ids.shape[-1]
        mask_self = causal_mask[None, None, :seq_len, :seq_len] * mask[:, None, None, :]
        mask_self = mask_self * mask[:, None, :, None]
        token_emb = nn.Embed(self.vocab_size, self.embed_dim, name="te")(input_ids)
        pos_emb = nn.Embed(self.max_len, self.embed_dim, name="pe")(jnp.arange(seq_len))
        x = token_emb + pos_emb
        x = nn.Dropout(self.dropout, deterministic=not training)(x)
        if mask is None:
            causal_mask = jnp.tril(jnp.ones((self.max_len, self.max_len)))
            mask = causal_mask[None, None, :seq_len, :seq_len]
        for _ in range(self.num_layers):
            x = DecoderBlock(self.embed_dim, self.num_heads, self.dropout)(
                x, training, mask_self=mask
            )
        out = nn.Dense(self.vocab_size, kernel_init=nn.initializers.xavier_uniform())(x)
        return out


class EncoderDecoderModel(nn.Module):
    r"""An encoder-decoder model.

    Attributes:
        num_layers (int): Number of layers.
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads per layer.
        vocab_size (int): Size of vocabulary.
        max_len_in (int): Maximum length of input sequences (Encoder).
        max_len_out (int): Maximum length of output sequences (Decoder).
        dropout (float): Probability of dropout.
    """

    num_layers: int
    embed_dim: int
    num_heads: int
    vocab_size: int
    dropout: float
    max_len_in: int
    max_len_out: int

    @nn.compact
    def __call__(
        self, input_ids=None, output_ids=None, mask_enc=None, mask_dec=None,
        training=False, *, prepare_cross=False, self_kv=None, cross_kv=None,
        decode_position=None,
    ):
        r"""Performs forward-pass of the model.

        Args:
            input_ids (jax.Array or None): Encoder IDs [encoder_batch, seq_len_in].
                Required for full forward/preparation; unused in cached decoding.
            output_ids (jax.Array or None): Decoder IDs [batch_size, seq_len_out]
                for full forward or [batch_size, 1] for cached decoding. Unused
                and may be None when prepare_cross=True.
            mask_enc (jax.Array or None): Encoder padding mask
                [encoder_batch, seq_len_in], with zero for padding. Required
                for full forward/preparation; optional in cached decoding when
                every encoder position is valid. encoder_batch may be one for
                a shared cached prompt.
            mask_dec (jax.Array or None): Decoder padding mask
                [batch_size, seq_len_out]. Required for full forward; unused
                in preparation and cached decoding, which builds a causal mask.
            training (bool): Controls dropout; must be False for either cache mode.
            prepare_cross (bool): Encode input_ids and return projected cross K/V
                for each decoder layer. Use the ordinary forward path to init weights.
            self_kv (tuple): Per-layer (K, V) buffers, each shaped
                [batch_size, num_heads, cache_length, head_size].
            cross_kv (tuple or None): Per-layer encoder (K, V), each shaped
                [encoder_batch, num_heads, seq_len_in, head_size], optionally with
                batch size one to share a prompt. Required for cached decoding.
            decode_position (scalar): Zero-based position of the single input
                token in cached inference. Requires self_kv and cross_kv.

        Returns:
            Full forward: logits [batch_size, seq_len_out, vocab_size].
            prepare_cross: a tuple of per-layer projected (K, V).
            Cached decode: (logits [batch_size, vocab_size], updated self_kv).
        """
        # Cached inference uses the same named modules/parameters as training.
        # The cache is explicit loop state, never a checkpoint collection.
        if decode_position is not None:
            # Cache reuse assumes deterministic inference, not training dropout.
            if training:
                raise ValueError("Cached decoding is only supported for inference.")
            # Each call consumes exactly one token at a shared position per row.
            if output_ids.shape[1] != 1:
                raise ValueError("Cached decoding expects exactly one token per row.")
            # Self K/V grow with generated tokens; cross K/V come from the prompt.
            if self_kv is None or cross_kv is None:
                raise ValueError("Cached decoding requires self- and cross-attention caches.")
            # Each decoder layer has distinct projections and needs its own pair.
            if len(self_kv) != self.num_layers or len(cross_kv) != self.num_layers:
                raise ValueError("Expected one cache per decoder layer.")
            # Look up the existing token embedding: [B, 1] -> [B, 1, embed_dim].
            x = nn.Embed(self.vocab_size, self.embed_dim, name="te_out")(output_ids)
            # Add the learned ABSOLUTE position. Using position zero on every
            # one-token call would change the model's predictions.
            x = x + nn.Embed(self.max_len_out, self.embed_dim, name="pe_out")(
                # A one-element index gives [1, embed_dim], shared across rows.
                jnp.reshape(decode_position, (1,))
            )
            # This Python list is assembled while tracing the fixed layer stack;
            # it does not cause host interaction during the device decoding loop.
            updated_cache = []
            for j in range(self.num_layers):
                # Reuse dec_block_j's existing attention, FFN, and norm weights.
                x, layer_cache = DecoderBlock(
                    self.embed_dim, self.num_heads, self.dropout, name=f"dec_block_{j}"
                )(
                    x, False, self_kv=self_kv[j], cross_kv=cross_kv[j],
                    decode_position=decode_position,
                    # Broadcast encoder padding across heads and the one query.
                    mask_cross=None if mask_enc is None else mask_enc[:, None, None, :],
                )
                # Preserve layer order when carrying updated K/V to the next step.
                updated_cache.append(layer_cache)
            # Apply the existing vocabulary projection to ONLY this one token.
            # Explicit module names keep these parameter paths checkpoint-compatible.
            logits = nn.Dense(
                self.vocab_size, kernel_init=nn.initializers.xavier_uniform(),
                name="dense_out", dtype=jnp.bfloat16,
            )(x)
            # Remove the length-one token axis, match the original FP32 logits,
            # and return a fixed-structure tuple suitable for lax.while_loop.
            return logits[:, 0, :].astype(jnp.float32), tuple(updated_cache)

        # Full forward needs encoder+decoder batches; cache preparation needs
        # only the encoder inputs, so skip decoder-input assertions in that mode.
        if not prepare_cross:
            assert len(input_ids) == len(output_ids) == len(mask_enc) == len(mask_dec), f"Forward pass got arguments of different batch sizes. Got batch sizes {len(input_ids)}, {len(output_ids)}, {len(mask_enc)}, {len(mask_dec)}."
            assert output_ids.shape[-1] <= self.max_len_out, f"Maximal allowed length of output IDs is {self.max_len_out} but got output length of {output_ids.shape[-1]}."
        elif training:
            # Preparing reusable cross K/V also requires dropout to be disabled.
            raise ValueError("Cross-attention cache preparation is inference-only.")
        assert input_ids.shape[-1] <= self.max_len_in, f"Maximal allowed length of input IDs is {self.max_len_in} but got input length of {input_ids.shape[-1]}."

        # Encoder
        seq_len_in = input_ids.shape[-1]
        mask_enc = mask_enc[:, None, None, :]
        token_emb_in = nn.Embed(self.vocab_size, self.embed_dim, name="te_in")(
            input_ids
        )
        pos_emb_in = nn.Embed(self.max_len_in, self.embed_dim, name="pe_in")(
            jnp.arange(seq_len_in)
        )
        enc_out = token_emb_in + pos_emb_in
        enc_out = nn.Dropout(self.dropout, deterministic=not training)(enc_out)
        for i in range(self.num_layers):
            enc_out = EncoderBlock(
                self.embed_dim, self.num_heads, self.dropout, name=f"enc_block_{i}"
            )(enc_out, training, mask=mask_enc)

        if prepare_cross:
            # The encoder above has run once. Each decoder layer projects its
            # own cross K/V from enc_out, then we return before any decoding.
            return tuple(
                DecoderBlock(
                    self.embed_dim, self.num_heads, self.dropout, name=f"dec_block_{j}"
                )(None, False, encoder_output=enc_out, prepare_cross=True)
                # The tuple's order matches dec_block_j in the cached path.
                for j in range(self.num_layers)
            )

        # Decoder
        causal_mask = jnp.tril(jnp.ones((self.max_len_out, self.max_len_out)))
        seq_len_out = output_ids.shape[-1]
        causal_mask = jnp.tril(jnp.ones((seq_len_out, seq_len_out)))
        mask_self = (
            causal_mask[None, None, :seq_len_out, :seq_len_out]
            * mask_dec[:, None, None, :]
        )
        mask_self = mask_self * mask_dec[:, None, :, None]
        mask_cross = mask_enc * mask_dec[:, None, :, None]
        token_emb_out = nn.Embed(self.vocab_size, self.embed_dim, name="te_out")(
            output_ids
        )
        pos_emb_out = nn.Embed(self.max_len_out, self.embed_dim, name="pe_out")(
            jnp.arange(seq_len_out)
        )
        x = token_emb_out + pos_emb_out
        x = nn.Dropout(self.dropout, deterministic=not training)(x)
        for j in range(self.num_layers):
            x = DecoderBlock(
                self.embed_dim, self.num_heads, self.dropout, name=f"dec_block_{j}"
            )(x, training, enc_out, mask_self, mask_cross)

        # Output
        out = nn.Dense(
            self.vocab_size,
            kernel_init=nn.initializers.xavier_uniform(),
            name="dense_out",
            dtype=jnp.bfloat16
        )(x)
        return out.astype(jnp.float32)
