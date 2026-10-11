import jax.numpy as jnp
import jax
from flax import linen as nn
from flax.training import train_state, orbax_utils
from flax.jax_utils import replicate, unreplicate
import optax
from tqdm import tqdm
import os
import orbax.checkpoint as ocp
import numpy as np
from jax_transformer.transformer_model import DecoderModel, EncoderDecoderModel
from functools import partial


def shard_batch(batch):
    r"""Splits a single batch into batches for each device.

    Args:
        batch (jnp.ndarray([batch_size, seq_len])): Input batch.

    Returns:
        share (jnp.ndarray([num_devices, batch_size/num_devices, seq_len])): Shards of batch.

    Raise:
        AssertionError: If batch_size is not devisible by num_devices.
    """
    n = jax.local_device_count()
    shards = batch.reshape(n, batch.shape[0] // n, *batch.shape[1:])
    return shards


@partial(jax.pmap, axis_name="data")
def train_step_decoder_only(
    state, input_ids, target_ids, mask_in, mask_loss, dropout_key
):
    r"""A single training step for a decoder-only model.

    Args:
        state (train_state.TrainState): Train state.
        input_ids (jnp.ndarray([batch_size, seq_len])): Batched input of token IDs.
        target_ids (jnp.ndarray([batch_size, seq_len])): Batched target IDs.
        mask_in (jnp.ndarray([batch_size, seq_len])): Batched padding mask for self-attention.
        mask_loss (jnp.ndarray([batch_size, seq_len])): Batched padding mask for loss.
        dropout_key (jax.prng_key): Key for dropout.

    Returns:
        loss (jnp.float): Mean loss value of batch.
        state (train_state.TrainState): Train state after a single optimization step.
    """

    def loss_fn(params):
        """Compute the loss for the enclosing step's inputs and masks.

        Args:
            params (PyTree): Candidate model weights passed to state.apply_fn.

        Returns:
            jax.Array: Scalar masked loss, reduced as required by this step.
        """
        logits = state.apply_fn(
            {"params": params},
            input_ids=input_ids,
            training=True,
            mask=mask_in,
            rngs={"dropout": dropout_key},
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, target_ids)
        loss = loss * mask_loss
        token_count = jax.lax.psum(mask_loss.sum(), axis_name="data")
        num_devices = jax.lax.psum(1.0, axis_name="data")
        denominator = jnp.maximum(token_count, 1.0)/num_devices
        return loss.sum() / denominator

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, axis_name="data")
    loss = jax.lax.pmean(loss, axis_name="data")
    state = state.apply_gradients(grads=grads)
    token_count = jax.lax.psum(mask_loss.sum(), axis_name="data")
    return loss, state, token_count


@partial(jax.pmap, axis_name="data")
def train_step_encoder_decoder(
    state,
    input_ids,
    decoder_input_ids,
    target_ids,
    mask_enc,
    mask_dec_in,
    mask_loss,
    dropout_key,
):
    r"""A single training step for an encoder-decoder model.

    Args:
        state (train_state.TrainState): Train state.
        input_ids (jnp.ndarray([batch_size, seq_len_in])): Batched input IDs.
        decoder_input_ids (jnp.ndarray([batch_size, seq_len_out-1])): Batched output IDs.
        target_ids (jnp.ndarray([batch_size, seq_len_out-1])): Batched target IDs.
        mask_enc (jnp.ndarray([batch_size, seq_len_in])): Batched padding mask for encoder.
        mask_dec_in (jnp.ndarray([batch_size, seq_len_out-1])): Batched padding mask for self-attention in decoder.
        mask_loss (jnp.ndarray([batch_size, seq_len_out-1])): Batched padding mask for loss.
        dropout_key (jax.prng_key): Key for dropout.

    Returns:
        loss (jnp.float): Mean loss value of batch.
        state (train_state.TrainState): Train state after a single optimization step.
    """

    def loss_fn(params):
        """Compute the loss for the enclosing step's inputs and masks.

        Args:
            params (PyTree): Candidate model weights passed to state.apply_fn.

        Returns:
            jax.Array: Scalar masked loss, reduced as required by this step.
        """
        logits = state.apply_fn(
            {"params": params},
            input_ids=input_ids,
            output_ids=decoder_input_ids,
            mask_enc=mask_enc,
            mask_dec=mask_dec_in,
            training=True,
            rngs={"dropout": dropout_key},
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, target_ids)
        loss = loss * mask_loss
        token_count = jax.lax.psum(mask_loss.sum(), axis_name="data")
        num_devices = jax.lax.psum(1.0, axis_name="data")
        denominator = jnp.maximum(token_count, 1.0)/num_devices
        return loss.sum() / denominator

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, axis_name="data")
    loss = jax.lax.pmean(loss, axis_name="data")
    state = state.apply_gradients(grads=grads)
    token_count = jax.lax.psum(mask_loss.sum(), axis_name="data")
    return loss, state, token_count


@partial(jax.pmap, axis_name="data")
def validation_step_decoder_only(state, input_ids, target_ids, mask_in, mask_loss):
    r"""A single validation step.

    Args:
        state (train_state.TrainState): Train state.
        input_ids (jnp.ndarray([batch_size, seq_len])): Batched input IDs.
        target_ids (jnp.ndarray([batch_size, seq_len])): Batched target IDs.
        mask_in (jnp.ndarray([batch_size, seq_len])): Batched padding mask for self-attention.
        mask_loss (jnp.ndarray([batch_size, seq_len])): Batched padding mask for loss.
        dropout_key (jax.prng_key): Key for dropout.


    Returns:
        loss (float): Mean validation loss value of batch.
    """

    def loss_fn(params):
        """Compute the loss for the enclosing step's inputs and masks.

        Args:
            params (PyTree): Candidate model weights passed to state.apply_fn.

        Returns:
            jax.Array: Scalar masked loss, reduced as required by this step.
        """
        logits = state.apply_fn(
            {"params": params},
            input_ids=input_ids,
            training=False,
            mask=mask_in,
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, target_ids)
        loss = loss * mask_loss
        return loss.sum()

    loss = loss_fn(state.params)
    loss_sum = jax.lax.psum(loss, axis_name="data")
    token_count = jax.lax.psum(mask_loss.sum(), axis_name="data")
    return loss_sum, token_count


@partial(jax.pmap, axis_name="data")
def validation_step_encoder_decoder(
    state,
    input_ids,
    decoder_input_ids,
    target_ids,
    mask_enc,
    mask_dec_in,
    mask_loss,
):
    r"""A single validation step.

    Args:
        state (train_state.TrainState): Train state.
        input_ids (jnp.ndarray([batch_size, seq_len_in])): Batched input of token IDs.
        decoder_input_ids (jnp.ndarray([batch_size, seq_len_out-1])): Batched input of token IDs.
        target_ids (jnp.ndarray([batch_size, seq_len_out-1])): One-hot encoded targets.
        mask_enc (jax.Array): Encoder padding mask [batch_size, seq_len_in].
        mask_dec_in (jnp.ndarray([batch_size, seq_len_out-1])): Batched padding mask for self-attention in decoder.
        mask_loss (jnp.ndarray([batch_size, seq_len_out-1])): Batched padding mask for loss.

    Returns:
        loss (float): Mean validation loss value of batch.
    """

    def loss_fn(params):
        """Compute the loss for the enclosing step's inputs and masks.

        Args:
            params (PyTree): Candidate model weights passed to state.apply_fn.

        Returns:
            jax.Array: Scalar masked loss, reduced as required by this step.
        """
        logits = state.apply_fn(
            {"params": params},
            input_ids=input_ids,
            output_ids=decoder_input_ids,
            mask_enc=mask_enc,
            mask_dec=mask_dec_in,
            training=False,
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, target_ids)
        loss = loss * mask_loss
        return loss.sum()

    loss = loss_fn(state.params)
    loss_sum = jax.lax.psum(loss, axis_name="data")
    token_count = jax.lax.psum(mask_loss.sum(), axis_name="data")
    return loss_sum, token_count


def init_model(model, model_type, params_key):
    r"""Initializes model.

    Args:
        model (nn.Module): The transformer model to be initialized.
        model_type (str): Model type.
        params_key (jax.prng_key): Key for param initialization.

    Returns:
        params (dict): Initialized model parameters.

    Raise:
        AssertionError: If model_type not dec or enc_dec.
    """
    assert model_type in [
        "dec",
        "enc_dec",
    ], f"Model type must be dec or enc_dec but got {model_type}."
    if model_type == "dec":
        inp = jnp.ones((1, model.max_len), dtype=jnp.int32)
        params = model.init({"params": params_key}, input_ids=inp, training=False)[
            "params"
        ]
    elif model_type == "enc_dec":
        inp = jnp.ones((1, model.max_len_in), dtype=jnp.int32)
        out = jnp.ones((1, model.max_len_out), dtype=jnp.int32)
        mask_enc = jnp.ones((1, model.max_len_in), dtype=jnp.int32)
        mask_dec = jnp.ones((1, model.max_len_out), dtype=jnp.int32)
        params = model.init(
            {"params": params_key},
            input_ids=inp,
            output_ids=out,
            mask_enc=mask_enc,
            mask_dec=mask_dec,
            training=False,
        )["params"]

    return params


def _pad(arr, length):
    """Append zero rows to an array.

    Args:
        arr (jax.Array): Input array with a leading batch dimension.
        length (int): Number of zero rows to append, not the final batch size.

    Returns:
        jax.Array: Padded array with the input dtype and trailing dimensions.
    """
    padding = jnp.zeros((length, *arr.shape[1:]), dtype=arr.dtype)
    arr = jnp.concatenate([arr, padding], axis=0)
    return arr


def _prepare_batch(batch, model_type, mask_key="masks"):
    r"""Prepare a batch for the selected model type.

    Args:
        batch (dict): Token and mask arrays. Decoder-only requires input_ids
            and mask_key; encoder-decoder requires input_ids, output_ids,
            mask_enc, and mask_dec.
        model_type (str): "dec" or "enc_dec".
        mask_key (str): Decoder-only padding-mask field; defaults to "masks".

    Returns:
        tuple: Device-sharded inputs, shifted targets, and masks, padded to a
            batch size divisible by the number of local devices.
    """
    if model_type == "dec":
        batch_ids = jnp.array(batch["input_ids"])
        mask = jnp.array(batch[mask_key])
        n = jax.local_device_count()
        if batch_ids.shape[0] % n != 0:
            padding_length = n - batch_ids.shape[0] % n
            batch_ids, mask = (_pad(x, padding_length) for x in (batch_ids, mask))
        return tuple(
            shard_batch(x)
            for x in (batch_ids[:, :-1], batch_ids[:, 1:], mask[:, :-1], mask[:, 1:])
        )
    if model_type == "enc_dec":
        input_ids = jnp.array(batch["input_ids"])
        output_ids = jnp.array(batch["output_ids"])
        mask_enc = jnp.array(batch["mask_enc"])
        mask_dec = jnp.array(batch["mask_dec"])
        n = jax.local_device_count()
        if input_ids.shape[0] % n != 0:
            padding_length = n - input_ids.shape[0] % n
            input_ids, output_ids, mask_enc, mask_dec = (
                _pad(x, padding_length)
                for x in (input_ids, output_ids, mask_enc, mask_dec)
            )
        return tuple(
            shard_batch(x)
            for x in (
                input_ids,
                output_ids[:, :-1],
                output_ids[:, 1:],
                mask_enc,
                mask_dec[:, :-1],
                mask_dec[:, 1:],
            )
        )
    raise AssertionError(
        f"Only DecoderModel and EncoderDecoderModel are supported for training but got {model_type}."
    )


def _log_epoch_metrics(epoch, train_loss, val_loss, log_freq, num_epochs, width):
    r"""Log training progress at the configured intervals.

    Args:
        epoch (int): Current one-based epoch number.
        train_loss (float): Training loss to display.
        val_loss (float): Validation loss to display.
        log_freq (int or None): Epoch interval; None disables logging.
        num_epochs (int): Final epoch, which is always logged when enabled.
        width (int): Terminal width used to calculate display-column padding.
    """
    if log_freq is None:
        return
    if epoch == 1 or epoch % log_freq == 0 or epoch == num_epochs:
        try:
            padding = width // 3 - 3
        except:
            padding = 10
        tqdm.write(
            f"Epoch: {epoch:<{padding}}Train loss: {'{0:.3f}'.format(round(train_loss, 3)):<{padding}}Val loss: {'{0:.3f}'.format(round(val_loss, 3))}"
        )


def _save_checkpoint(state, ckpt_dir, epoch, checkpointer):
    r"""Save an unreplicated checkpoint and return the replicated state.

    Args:
        state (train_state.TrainState): Device-replicated training state.
        ckpt_dir (str): Parent directory for checkpoint_<epoch>.
        epoch (int): Epoch number used in the checkpoint directory name.
        checkpointer (ocp.PyTreeCheckpointer): Object used to save the state.

    Returns:
        train_state.TrainState: State replicated across local devices again.
    """
    state = unreplicate(state)
    ckpt = {"model": state}
    save_args = orbax_utils.save_args_from_target(ckpt)
    checkpointer.save(ckpt_dir + f"/checkpoint_{epoch}", ckpt, save_args=save_args)
    return replicate(state)


def train(
    model,
    num_epochs,
    train_loader,
    val_loader,
    optimizer,
    params=None,
    save_freq=None,
    ckpt_dir=None,
    log_freq=None,
    root_key=jax.random.key(42),
):
    r"""Trains a given transformer model.

    Args:
        model (nn.Module): The transformer model to be trained.
        num_epochs (int): Number of training epochs.
        train_loader (torch.utils.data.DataLoader): Training data loader.
        val_loader (torch.utils.data.DataLoader): Validation data loader.
        optimizer (optax.GradientTransformation): Optimizer to be used in training.
        params (dict): Model parameters of pretrained model. Defaults to None.
        save_freq (int): Save the model every save_freq epochs.
        ckpt_dir (str): Path to checkpoint directory.
        log_freq (int): Log metrics every log_freq epochs.
        root_key (jax.prng_key): Key for initialization and dropout.

    Returns:
        state (train_state.TrainState): Train state of fully trained model.
        training_loss (jnp.ndarray([num_epochs])): Mean training loss per epoch.
        validation_loss (jnp.ndarray([num_epochs])): Mean validation loss per epoch.

    Raises:
        AssertionError: If a save frequency is set, a checkpoint directory must also be set.
        AssertionError: Only encoder and encoder-decoder are supported as models.
    """
    # Assertions
    if (save_freq is None) != (ckpt_dir is None):
        raise AssertionError(
            f"Got save_freq = {save_freq} but ckpt_dir = {ckpt_dir}. Both must be defined or None."
        )
    if ckpt_dir is not None:
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        checkpointer = ocp.PyTreeCheckpointer()
    if isinstance(model, EncoderDecoderModel):
        model_type = "enc_dec"
    elif isinstance(model, DecoderModel):
        model_type = "dec"
    else:
        raise AssertionError(
            f"Only DecoderModel and EncoderDecoderModel are supported for training but got {type(model)}."
        )
    n = jax.local_device_count()
    assert (
        train_loader.batch_size % n == 0
    ), f"Batch size must be divisible by device count. Got train loader batch size {train_loader.batch_size} and device count {n}."
    assert (
        val_loader.batch_size % n == 0
    ), f"Batch size must be divisible by device count. Got train loader batch size {val_loader.batch_size} and device count {n}."

    # Set keys
    params_key, dropout_key = jax.random.split(root_key)

    # Initialize model
    if params is None:
        params = init_model(model, model_type, params_key)

    # Create training state (over multiple GPUs, if applicable)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )
    state = replicate(state)

    # Print number of parameters
    param_count = sum(x.size for x in jax.tree.leaves(params))
    print(f"Number of model parameters: {param_count}")

    # Training loop
    training_loss = []
    validation_loss = []
    try:
        width = int(os.get_terminal_size().columns // 2 * 2)
    except:
        width = 75
    s = "Training"
    print(f"{s:-^{width}}")
    for epoch in tqdm(range(1, num_epochs + 1)):
        train_loss_total = 0.0
        val_loss_total = 0.0
        val_token_total = 0.0
        train_token_total = 0.0
        for batch in train_loader:
            prepared_batch = _prepare_batch(batch, model_type)
            if model_type == "dec":
                input_ids, target_ids, mask_in, mask_loss = prepared_batch
                dropout_keys = jax.random.split(dropout_key, jax.local_device_count())
                loss, state, token_count = train_step_decoder_only(
                    state=state,
                    input_ids=input_ids,
                    target_ids=target_ids,
                    mask_in=mask_in,
                    mask_loss=mask_loss,
                    dropout_key=dropout_keys,
                )
            elif model_type == "enc_dec":
                (
                    input_ids,
                    decoder_input_ids,
                    target_ids,
                    mask_enc,
                    mask_dec_in,
                    mask_loss,
                ) = prepared_batch
                dropout_keys = jax.random.split(dropout_key, jax.local_device_count())
                loss, state, token_count = train_step_encoder_decoder(
                    state=state,
                    input_ids=input_ids,
                    decoder_input_ids=decoder_input_ids,
                    target_ids=target_ids,
                    mask_enc=mask_enc,
                    mask_dec_in=mask_dec_in,
                    mask_loss=mask_loss,
                    dropout_key=dropout_keys,
                )
            _, dropout_key = jax.random.split(dropout_key)
            train_loss_total += loss[0] * token_count[0]
            train_token_total += token_count[0]
        for batch in val_loader:
            prepared_batch = _prepare_batch(
                batch, model_type
            )
            if model_type == "dec":
                input_ids, target_ids, mask_in, mask_loss = prepared_batch
                loss_sum, token_count = validation_step_decoder_only(
                    state=state,
                    input_ids=input_ids,
                    target_ids=target_ids,
                    mask_in=mask_in,
                    mask_loss=mask_loss,
                )
            elif model_type == "enc_dec":
                (
                    input_ids,
                    decoder_input_ids,
                    target_ids,
                    mask_enc,
                    mask_dec_in,
                    mask_loss,
                ) = prepared_batch
                loss_sum, token_count = validation_step_encoder_decoder(
                    state=state,
                    input_ids=input_ids,
                    decoder_input_ids=decoder_input_ids,
                    target_ids=target_ids,
                    mask_enc=mask_enc,
                    mask_dec_in=mask_dec_in,
                    mask_loss=mask_loss,
                )
            val_loss_total += loss_sum[0]
            val_token_total += token_count[0]
        train_loss = train_loss_total / jnp.maximum(train_token_total, 1.0)
        val_loss = val_loss_total / jnp.maximum(val_token_total, 1.0)
        training_loss.append(train_loss)
        validation_loss.append(val_loss)
        _log_epoch_metrics(epoch, train_loss, val_loss, log_freq, num_epochs, width)
        if save_freq is not None and epoch % save_freq == 0:
            state = _save_checkpoint(state, ckpt_dir, epoch, checkpointer)
    try:
        width = int(os.get_terminal_size().columns // 2 * 2)
    except:
        width = 10
    s = "Done"
    print(f"{s:-^{width}}")
    state = unreplicate(state)
    return state, jnp.array(training_loss), jnp.array(validation_loss)


@jax.jit
def get_logits(state, input_ids, output_ids=None):
    r"""A single forward-pass to obtain logits.

    Args:
        state (train_state.TrainState): Training state.
        input_ids (jnp.ndarray([num_samples, seq_len])): Input IDs.
        output_ids (jnp.ndarray([num_samples, seq_len_out])): Output IDs (encoder-decoder only).
    Returns:
        logits (jnp.ndarray([num_samples, vocab_size])): Logits for next token.
    """
    if output_ids is None:
        logits = state.apply_fn(
            {"params": state.params}, input_ids=input_ids, training=False
        )[:, -1, :]
    else:
        mask_enc = jnp.ones_like(input_ids)
        mask_dec = jnp.ones_like(output_ids)
        logits = state.apply_fn(
            {"params": state.params},
            input_ids=input_ids,
            output_ids=output_ids,
            mask_enc=mask_enc,
            mask_dec=mask_dec,
            training=False,
        )[:, -1, :]
    return logits


# Compile the entire generation routine. These four arguments control Python
# branching, the callable, or array shapes, so their values are compile-time
# constants. Changing them (or an input shape/dtype) may compile another version;
# changing weight values, the random key, or temperature normally reuses it.
@partial(jax.jit, static_argnames=("apply_fn", "max_len", "use_kv_cache", "num_samples"))
def _sample_ids(
    params, apply_fn, input_ids, root_key, sampling_temp, bos_id, eos_id, max_len,
    use_kv_cache=False, num_samples=None,
):
    """Generate a fixed-size batch on device, returning IDs and first-EOS lengths.

    Cached decoding accepts a single shared encoder prompt. The fallback path
    supports other apply functions using fixed-size padded forward passes.

    Args:
        params (PyTree): Trained model parameters, without optimizer state.
        apply_fn (callable): Model apply function accepting {"params": params}.
            Cached mode requires EncoderDecoderModel's cache keyword arguments.
            Static for JIT compilation.
        input_ids (jax.Array): Unpadded encoder token IDs [1, prompt_length] for
            a shared cached prompt, or [batch_size, prompt_length] without caching.
        root_key (jax.Array): Initial JAX random key; split once per sampled step.
        sampling_temp (float): Positive finite divisor applied to sampling logits.
        bos_id (int): Vocabulary ID placed at the start of every decoder row.
        eos_id (int): End-of-sequence ID, also used to prefill unused output slots.
        max_len (int): Static output capacity including BOS/EOS, at least two.
            The final slot is forced EOS; self caches have max_len-2 positions.
        use_kv_cache (bool): Static switch enabling single-token cached decoding.
            False runs the fixed-length padded full-forward compatibility path.
        num_samples (int or None): Static decoder batch size. None uses the
            encoder input batch size. Without caching, must match that batch size.

    Returns:
        tuple: Device arrays (output_ids, lengths), both int32. output_ids has
            shape [batch_size, max_len]; lengths has shape [batch_size] and counts
            through the first EOS, including BOS. Encoder prompt IDs are excluded.
    """
    # B = number of generated sequences; L = max_len, including BOS and EOS.
    # With caching, input_ids has just one row even though we generate B rows.
    batch_size = input_ids.shape[0] if num_samples is None else num_samples
    # The last slot remains EOS, matching the previous forced termination.
    output_ids = jnp.full((batch_size, max_len), eos_id, dtype=jnp.int32)
    # Every decoder sequence starts with BOS at column zero.
    output_ids = output_ids.at[:, 0].set(bos_id)
    # One device-side flag per row replaces the Python set of EOS indices.
    finished = jnp.zeros(batch_size, dtype=jnp.bool_)
    # Default to the full length; overwrite once a row first emits EOS.
    lengths = jnp.full(batch_size, max_len, dtype=jnp.int32)
    # L=2 already contains [BOS, EOS], so there is nothing to predict.
    if max_len == 2:
        return output_ids, lengths
    # The supplied encoder prompt has no padding: every prompt token is valid.
    mask_enc = jnp.ones_like(input_ids)
    # Column indices used to hide the unfilled suffix in the uncached path.
    positions = jnp.arange(max_len - 1)
    # An empty tuple keeps the same loop-state structure when no cache is used.
    self_kv = ()
    if use_kv_cache:
        # Encoder and cross-attention projections run once, before the loop.
        # Cross K/V keep batch size 1 and broadcast across sampled sequences.
        cross_kv = apply_fn(
            # Supply existing trained weights; this does not initialize weights.
            {"params": params}, input_ids=input_ids, mask_enc=mask_enc,
            # Disable dropout and request only encoder/cross-attention work.
            training=False, prepare_cross=True,
        )
        # Allocate one (K, V) pair per decoder layer. Each buffer is
        # [B, heads, L-2, head_size], with the projection's dtype (BF16).
        # There are L-2 model calls: the last output slot is forced EOS, and
        # predicting the token in slot L-2 consumes the token in slot L-3.
        self_kv = tuple(
            tuple(jnp.zeros(
                (batch_size, kv.shape[1], max_len - 2, kv.shape[3]), dtype=kv.dtype
            ) for kv in layer)
            for layer in cross_kv
        )

    def condition(carry):
        """Decide whether another token should be generated.

        Args:
            carry (tuple): (position, output_ids, finished, lengths, key, self_kv)
                loop state. position is the next output column; finished contains
                one completion flag per row. Other entries are unused here.

        Returns:
            jax.Array: Scalar boolean indicating remaining work and capacity.
        """
        # Extract only the next output column and per-row completion flags.
        position, _, finished, _, _, _ = carry
        # A device boolean keeps this decision inside the compiled loop.
        # Stop before the reserved final EOS slot, or when every row is done.
        return (position < max_len - 1) & ~jnp.all(finished)

    def step(carry):
        """Generate one token per row and update the fixed-shape loop state.

        Args:
            carry (tuple): Next-column int32 position, int32 [B, L] output_ids,
                boolean [B] finished flags, int32 [B] first-EOS lengths, PRNG key,
                and per-layer self-attention (K, V) buffers (empty when uncached).

        Returns:
            tuple: Updated values in the same order, with unchanged shapes/dtypes.
        """
        # Carry every changing value explicitly; all shapes/dtypes stay fixed.
        position, output_ids, finished, lengths, key, self_kv = carry
        if use_kv_cache:
            # Read the previous token for all rows as [B, 1]. The explicit
            # int32 zero matches position even when JAX's x64 mode is enabled.
            current_token = jax.lax.dynamic_slice(
                output_ids, (jnp.int32(0), position - 1), (batch_size, 1)
            )
            # Return [B, vocabulary] logits and updated per-layer self K/V.
            # Shared cross K/V are loop-invariant; they are never recomputed here.
            logits, self_kv = apply_fn(
                {"params": params}, output_ids=current_token, mask_enc=mask_enc,
                training=False, self_kv=self_kv, cross_kv=cross_kv,
                decode_position=position - 1,  # Position of the consumed token.
            )
        else:
            # Compatibility path for apply functions without cache support.
            mask_dec = jnp.broadcast_to(
                # True for already written tokens; repeat the mask over B rows.
                positions < position, (batch_size, max_len - 1)
            )
            logits = apply_fn(
                {"params": params}, input_ids=input_ids,
                # Keep a fixed input length, excluding the reserved final EOS.
                output_ids=output_ids[:, :-1], mask_enc=mask_enc,
                mask_dec=mask_dec, training=False,
            )[:, position - 1, :]  # Select the last valid token, not padding.
        # Draw one ID per row from temperature-scaled logits. No explicit
        # softmax is needed: categorical accepts unnormalized log probabilities.
        new_ids = jax.random.categorical(key, logits / sampling_temp, axis=-1)
        # Preserve the previous sampler's per-step PRNG sequence.
        _, key = jax.random.split(key)
        # Completed rows remain in the fixed-size batch but only receive EOS.
        # Cast categorical's result to int32 so x64 mode cannot change the buffer.
        new_ids = jnp.where(finished, eos_id, new_ids).astype(jnp.int32)
        # Mark only the FIRST EOS for a row; subsequent EOS must not extend it.
        newly_finished = ~finished & (new_ids == eos_id)
        # position is zero-based, so position+1 is the length including EOS/BOS.
        lengths = jnp.where(newly_finished, position + 1, lengths)
        # A finished flag stays true for the rest of the generation call.
        finished = finished | newly_finished
        # Write one column without concatenating or changing the output shape.
        output_ids = output_ids.at[:, position].set(new_ids)
        # Advance to the next output column and carry the updated state forward.
        return position + 1, output_ids, finished, lengths, key, self_kv

    # Run condition/step on device, starting at column 1 (BOS already occupies 0).
    # Discard the final position, flags, key, and caches: callers need IDs/lengths.
    _, output_ids, _, lengths, _, _ = jax.lax.while_loop(
        condition, step,
        (jnp.int32(1), output_ids, finished, lengths, root_key, self_kv),
    )
    return output_ids, lengths  # Device arrays; numeric results stay on device.


def sample(
    state,
    model_type,
    num_samples,
    max_len,
    tokenizer,
    prompt,
    root_key=jax.random.key(42),
    sampling_temp=1.0,
    return_tokens=False,
    use_kv_cache=True,
):
    r"""Samples new tokens based on inputs using a trained model.

    Args:
        state (train_state.TrainState): Training state to be used in sampling.
        model_type (str): Must be "encoder-decoder"; decoder-only is unsupported.
        num_samples (int): Number of samples per prompt.
        max_len (int): Maximum output length including BOS and EOS (at least 2).
        tokenizer (DictTokenizer): Supplies to_ids, to_token, and dict entries
            for "<BOS>", "<EOS>", and "<PAD>" (for numeric output).
        prompt (array-like): One encoder prompt in the tokenizer's token format.
        root_key (jax.Array): Initial JAX random key. Defaults to key(42); pass
            distinct keys for independent sampling calls.
        sampling_temp (float): Positive finite sampling temperature; defaults to 1.
        return_tokens (bool): True returns unpadded CPU token-string arrays;
            False (default) returns a padded 2D JAX array of int32 token IDs.
        use_kv_cache (bool): Use cached decoding for EncoderDecoderModel (default).
            False keeps padded decoding for comparison. Other apply functions
            use the padded path automatically.

    Returns:
        samples (jax.Array or list[np.ndarray]): Numeric output has shape
            [num_samples, prompt_length + max_len]. Each row contains the prompt,
            BOS, completion, first EOS, then PAD tokens. String output is a list
            of CPU arrays truncated after the first EOS, without padding.

    Raises:
        AssertionError: If the model is not encoder-decoder.
        ValueError: If lengths, sample count, temperature, or prompt shape are invalid.
    """
    assert (
        model_type == "encoder-decoder"
    ), f"Only encoder-decoder is currently supported as model type but got {model_type}."

    # Validate on the host before compilation: BOS/EOS require two slots.
    if not isinstance(max_len, (int, np.integer)) or max_len < 2:
        raise ValueError("max_len must be an integer of at least 2 (BOS and EOS).")
    # Array dimensions must be nonnegative integers; accept NumPy integers too.
    if not isinstance(num_samples, (int, np.integer)) or num_samples < 0:
        raise ValueError("num_samples must be a nonnegative integer.")
    # Prevent undefined/nonfinite scaling of logits inside the compiled loop.
    if not np.isfinite(sampling_temp) or sampling_temp <= 0:
        raise ValueError("sampling_temp must be finite and greater than zero.")
    # Empty string output needs no tokenization or device work.
    if num_samples == 0 and return_tokens:
        return []

    # Tokenize on CPU as [1, prompt_length]; fix the integer dtype explicitly.
    prompt_ids = np.asarray(tokenizer.to_ids(np.atleast_2d(prompt)), dtype=np.int32)
    # This API generates many continuations of ONE prompt, not a prompt batch.
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("sample expects a single prompt.")
    # Preserve both dimensions even when no numeric samples were requested.
    if num_samples == 0:
        return jnp.empty((0, prompt_ids.shape[1] + int(max_len)), dtype=jnp.int32)
    # For model.apply, __self__ is the model instance owning the bound method.
    # Other callables have no such owner and use the compatibility path.
    model = getattr(state.apply_fn, "__self__", None)
    cached = use_kv_cache and isinstance(model, EncoderDecoderModel)
    # The largest consumed position is L-3; it must fit the learned embeddings.
    if cached and max_len - 2 > model.max_len_out:
        raise ValueError("Requested generation exceeds the model's position embeddings.")
    input_ids = jnp.asarray(prompt_ids)  # Move the single prompt to JAX's device.
    if not cached:
        # Only full-forward decoding needs one encoder input row per sample.
        input_ids = jnp.repeat(input_ids, num_samples, axis=0)
    # Pass weights rather than the whole TrainState: optimizer state is unused.
    # Python ints make the two shape-setting static arguments unambiguous.
    output_ids, lengths = _sample_ids(
        state.params, state.apply_fn, input_ids, root_key, sampling_temp,
        tokenizer.dict["<BOS>"], tokenizer.dict["<EOS>"], max_len=int(max_len),
        use_kv_cache=cached, num_samples=int(num_samples),
    )
    if not return_tokens:
        # lengths includes the first EOS: keep it and replace only later slots.
        # Broadcasting compares [1, max_len] positions with [num_samples, 1].
        padded_ids = jnp.where(
            jnp.arange(max_len)[None, :] < lengths[:, None],
            output_ids,
            jnp.asarray(tokenizer.dict["<PAD>"], dtype=output_ids.dtype),
        )
        return padded_ids
    else:
        # Token strings cannot be represented by JAX arrays. Transfer once and format
        # the existing unpadded string output on CPU.
        output_ids, lengths = jax.device_get((output_ids, lengths))
        # Pair each row with its first-EOS length; prepend the original CPU prompt.
        # These ragged arrays can differ in length without triggering JAX compilation.
        samples = [
            np.concatenate([prompt_ids[0], row[:length]])
            for row, length in zip(output_ids, lengths)
        ]

        # The tokenizer expects 2D input; flatten its one-row result for callers.
        samples = [
            tokenizer.to_token(np.atleast_2d(token_ids)).flatten()
            for token_ids in samples
        ]
        return samples


def load_model(target_state, path):
    r"""Load a saved checkpoint.

    Args:
        target_state (train_state.TrainState): Initialized target state to map checkpoint onto.
        path (str): Path to checkpoint.

    Returns:
        restored_state (train_state.TrainState): Loaded checkpoint.
    """
    checkpointer = ocp.PyTreeCheckpointer()
    restore_args = orbax_utils.restore_args_from_target({"model": target_state})
    restored_state = checkpointer.restore(
        path, item={"model": target_state}, restore_args=restore_args
    )["model"]
    return restored_state
