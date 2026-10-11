from jax_transformer.transformer_model import DecoderModel, EncoderDecoderModel
from jax_transformer.utils import load_model
from jax_transformer.data_loader import DictTokenizer
from data_scripts.tokenizer_utils import token_to_data
from flax.training import train_state
import optax
from scipy.optimize import root
import jax
import jax.numpy as jnp
import json
import numpy as np
from tqdm import tqdm
import os


def load_config(RUN_ID, cwd):
    r"""Loads config of trained model.

    Args:
        RUN_ID (int): Specifies the trained model.
        cwd (str): Path to directory containing the folder 'logs'.

    Returns:
        config (dict): Model config.

    Raise:
        AssertionError: If config['optimizer'] contains unsupported optimizer.
    """
    with open(cwd + f"/logs/run_{RUN_ID}/run_{RUN_ID}.json") as f:
        config = json.load(f)
    assert config["optimizer"] in [
        "adam",
        "adamw",
    ], f"Currently only adam and adamw are possible as optimizers, but got {config['optimizer']}."
    return config


def load_model_from_config(
    RUN_ID, config, load_ckpt, cwd, override_opt=False, GRPO_ID=None
):
    r"""Loads trained model from respective config.

    Args:
        RUN_ID (int): Specifies the trained model.
        config (dict): Model config.
        load_ckpt (str): Name of checkpoint to be loaded.
        cwd (str): Path to directory containing the folder 'checkpoints'.
        override_opt (bool): Needed for compatibility with older checkpoints. Defaults to False.

    Returns:
        state (train_state.TrainState): Loaded train state.
        model (nn.Module): Loaded model.
        tokenizer (utils.DictTokenizer): Tokenizer used during training.
    """
    models = {"decoder": DecoderModel, "encoder-decoder": EncoderDecoderModel}
    optimizers = {"adam": optax.adam, "adamw": optax.adamw}
    schedules = {"cosine": optax.warmup_cosine_decay_schedule}

    tokenizer = DictTokenizer(config["token_dict"])
    optimizer = optimizers[config["optimizer"]]
    if config["lr_schedule"] is not None:
        schedule = schedules[config["lr_schedule"]]
        decay_steps = config["decay_steps"]
        lr_schdl = schedule(
            init_value=0.0,
            peak_value=config["lr"],
            warmup_steps=config["warmup_steps"],
            decay_steps=decay_steps,
        )
        opt = optimizer(learning_rate=lr_schdl, b1=0.9, b2=0.98, eps=1e-9)
    else:
        opt = optimizer(learning_rate=config["lr"], b1=0.9, b2=0.98, eps=1e-9)

    # Needed for compatibility with older checkpoints
    if override_opt:
        if RUN_ID <= 35:
            opt = optax.adamw(
                learning_rate=1e-5,
                weight_decay=0.00,
            )
        else:
            with open(cwd + f"/logs/run_{RUN_ID}/GRPO_{GRPO_ID}/log.json") as f:
                grpo_config = json.load(f)
            lr = grpo_config["lr"]
            opt = optax.adam(learning_rate=lr)
            if RUN_ID >= 46 or (RUN_ID == 45 and int(GRPO_ID) == 3):
                opt = optax.chain(optax.clip_by_global_norm(1.0), opt)

    # Setting args for model
    args_model = {
        "num_layers": config["num_layers"],
        "embed_dim": config["embed_dim"],
        "num_heads": config["num_heads"],
        "vocab_size": tokenizer.vocab_size,
        "dropout": config["dropout"],
    }
    if config["model"] == "encoder-decoder":
        max_len_in = config["max_len_in"]
        max_len_out = config["max_len_out"]
        update = {"max_len_in": max_len_in, "max_len_out": max_len_out}
        args_model.update(update)
    elif config["model"] == "decoder":
        max_len = config["max_len"]
        update = {"max_len": max_len}
        args_model.update(update)

    # Loading model
    model = models[config["model"]](**args_model)
    params_key = jax.random.key(42)
    if config["model"] == "decoder":
        inp = jnp.ones((1, model.max_len), dtype=jnp.int32)
        params = model.init({"params": params_key}, input_ids=inp, training=False)[
            "params"
        ]
    elif config["model"] == "encoder-decoder":
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
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=opt)
    state = load_model(state, cwd + f"/checkpoints/run_{RUN_ID}/" + load_ckpt)
    return state, model, tokenizer


def to_old_a_matrix(model):
    r"""Map a-matrix of jaxvacua model to previously used convention. Needed to verify old models.

    Args:
        model (jvc.FluxVacuaFinder): Jaxvacua model specifying geometry.

    Returns:
        model (jvc.FluxVacuaFinder): Jaxvacua model with old a-matrix.
    """
    a_matrix = np.zeros((model.h12, model.h12))
    for I1 in range(model.h12):
        for I2 in range(model.h12):
            a_matrix[I1][I2] = model.lcs_tree.intnums[I1][I2][I2] / 2.0
    model.lcs_tree.a_matrix = a_matrix
    return model


def solve(
    fluxes, moduli0, tau0, model, tol, use_scipy=True, verbose=False, debug=False
):
    r"""Solving the F-term conditions for given input and model.

    Args:
        fluxes (jnp.ndarray([batch_size, 4*(model.h12+1)])): Batch of fluxes.
        moduli0 (jnp.ndarray([batch_size, model.h12])): Batch of moduli used as initial guesses.
        tau0 (jnp.ndarray([batch_size])): Batch of axio-dilaton used as initial guesses.
        model (jvc.FluxVacuaFinder): Jaxvacua model specifying geometry.
        use_scipy (bool): Use scipy.optimize.root() to find solutions. Defaults to True.
        verbose (bool): Show solving progress. Defaults to False.
        debug (bool): Log if terminated.

    Returns:
        fluxes (jnp.ndarray([solved_size, 4*(model.h12+1)])): Fluxes with found solution.
        moduli (jnp.ndarray([solved_size, model.h12])): Moduli at solution.
        tau (jnp.ndarray([solved_size])): Axio-dilaton at solution.
        idx_solved (jnp.ndarray([batch_size])): Boolean mask over input batch, indicating position of solved fluxes.
    """
    if debug:
        print("Solving...")

    if moduli0.ndim == 1:
        if use_scipy:
            raise NotImplementedError
        else:
            solver = jax.vmap(
                lambda fluxes0: model.newton_method_flux_vacua(
                    moduli0,
                    tau0,
                    fluxes0,
                    mode="SUSY",
                    solver_mode="complex",
                    tol=tol,
                    step_size_Newton=1.0,
                )
            )
            moduli, tau, res = solver(fluxes)
    elif moduli0.ndim == 2:
        if use_scipy:
            moduli0_np = np.asarray(moduli0)
            tau0_np = np.asarray(tau0)
            fluxes_np = np.asarray(fluxes)
            batch_size = len(moduli0_np)
            moduli0_np = np.stack([moduli0_np.real, moduli0_np.imag], axis=-1).reshape(
                batch_size, -1
            )
            tau0_np = np.stack([tau0_np.real, tau0_np.imag], axis=-1)
            x0_np = np.concatenate([moduli0_np, tau0_np], axis=-1)
            x = np.empty_like(x0_np)
            idx_solved = np.empty(batch_size, dtype=bool)
            if verbose:
                for i in tqdm(range(len(fluxes_np))):
                    flux_iter = fluxes_np[i]
                    x0_iter = x0_np[i]
                    sol = root(
                        model.DW_x,
                        x0_iter,
                        args=(flux_iter,),
                        jac=model.dDW_x,
                        method="hybr",
                        tol=tol * 1e-3,
                    )
                    res = np.max(np.abs(model.DW_x(sol.x, flux_iter)))
                    idx_solved[i] = sol.success and (res < tol)
                    x[i] = sol.x
            else:
                for i, (flux_iter, x0_iter) in enumerate(zip(fluxes_np, x0_np)):
                    sol = root(
                        model.DW_x,
                        x0_iter,
                        args=(flux_iter,),
                        jac=model.dDW_x,
                        method="hybr",
                        tol=tol * 1e-3,
                    )
                    res = np.max(np.abs(model.DW_x(sol.x, flux_iter)))
                    idx_solved[i] = sol.success and (res < tol)
                    x[i] = sol.x
            moduli = x[:, 0:-2:2] + 1.0j * x[:, 1:-2:2]
            tau = x[:, -2] + 1.0j * x[:, -1]
        else:
            with jax.default_device(jax.devices("cpu")[0]):
                solver = lambda moduli0, tau0, fluxes0: model.newton_method_flux_vacua(
                    moduli0,
                    tau0,
                    fluxes0,
                    mode="SUSY",
                    solver_mode="complex",
                    max_iters=50,
                    tol=tol,
                    step_size_Newton=1.0,
                )
                moduli, tau, res = solver(moduli0, tau0, fluxes)
                idx_solved = jnp.abs(res) < tol
    else:
        raise AssertionError(
            f"moduli0 must be have one or two axis but got shape {moduli0}."
        )
    # Filter for solved
    fluxes, moduli, tau = (
        fluxes[idx_solved],
        moduli[idx_solved],
        tau[idx_solved],
    )
    if debug:
        print("Solving terminated.")
    return (
        jnp.asarray(fluxes),
        jnp.asarray(moduli),
        jnp.asarray(tau),
        jnp.asarray(idx_solved),
    )


def map_to_FD(fluxes, moduli, tau, model, debug, debug_deep):
    r"""Maps fluxes, moduli and axio-dilaton to fundamental domain of SL(2,Z) and monodromy transformations. If the mapping is not possible, the original data points are returned.

    Args:
        fluxes (jnp.ndarray([batch_size, 4*(model.h12+1)])): Batch of fluxes.
        moduli (jnp.ndarray([batch_size, model.h12])): Batch of moduli.
        tau (jnp.ndarray([batch_size])): Batch of axio-dilaton.
        model (jvc.FluxVacuaFinder): Jaxvacua model specifying geometry.
        debug (bool): Log if terminated.
        debug_deep (bool): Print progress.

    Returns:
        fluxes (jnp.ndarray([batch_size, 4*(model.h12+1)])): Fluxes within fundamental domain.
        moduli (jnp.ndarray([batch_size, model.h12])): Moduli within fundamental domain.
        tau (jnp.ndarray([batch_size])): Axio-dilaton within fundamental domain.
    """
    moduli_FD_lst = []
    tau_FD_lst = []
    fluxes_FD_lst = []
    if debug:
        print("Mapping to FD...")
    if debug_deep:
        for i in tqdm(range(len(fluxes))):
            moduli_iter, tau_iter, flux_iter = moduli[i], tau[i], fluxes[i]
            moduli_FD, tau_FD, flux_FD = model.map_to_fd(
                moduli_iter, tau_iter, flux_iter
            )
            tau_FD_lst.append(np.asarray(tau_FD))
            moduli_FD_lst.append(np.asarray(moduli_FD)[None, :])
            fluxes_FD_lst.append(np.asarray(flux_FD[None, :]))
    else:
        for moduli_iter, tau_iter, flux_iter in zip(moduli, tau, fluxes):
            moduli_FD, tau_FD, flux_FD = model.map_to_fd(
                moduli_iter, tau_iter, flux_iter
            )
            tau_FD_lst.append(np.asarray(tau_FD))
            moduli_FD_lst.append(np.asarray(moduli_FD)[None, :])
            fluxes_FD_lst.append(np.asarray(flux_FD[None, :]))

    fluxes = np.concatenate(fluxes_FD_lst, axis=0)
    moduli = np.concatenate(moduli_FD_lst, axis=0)
    tau = np.array(tau_FD_lst)
    if debug:
        print("Mapping terminated.")
    return jnp.asarray(fluxes), jnp.asarray(moduli), jnp.asarray(tau)


def check_tadpole(model, fluxes, debug=False):
    r"""Checks tadpole bound for set of fluxes.

    Args:
        model (jvc.FluxVacuaFinder): Jaxvacua model specifying geometry.
        fluxes (jnp.ndarray([batch_size, 4*(model.h12+1)])): Batch of fluxes.
        debug (bool): Log if terminated.

    Returns:
        fluxes (jnp.ndarray([valid_size, 4*(model.h12+1)])): Fluxes within tadpole bound.
        idx_valid (jnp.ndarray([batch_size])): Boolean mask over input batch, indicating position of valid fluxes.
    """
    if debug:
        print("Checking tadpole...")
    Q3 = model.Q()
    nflux = model.tadpole(fluxes)
    idx_valid = (nflux > 0) & (nflux <= Q3)
    fluxes = fluxes[idx_valid]
    if debug:
        print("Check terminated.")
    return fluxes, idx_valid


def check_FD(fluxes, moduli, tau, model=None, debug=False):
    r"""Checks if input in fundamental domain of SL(2,Z).

    Args:
        fluxes (jnp.ndarray([batch_size, 4*(model.h12+1)])): Batch of fluxes.
        moduli (jnp.ndarray([batch_size, model.h12])): Batch of moduli.
        tau (jnp.ndarray([batch_size])): Batch of axio-dilaton.
        debug (bool): Log if terminated.
        model (jvc.FluxVacuaFinder): Needed for consistency (will be changed). Defaults to None.

    Returns:
        fluxes (jnp.ndarray([valid_size, 4*(model.h12+1)])): Fluxes within fundamental domain.
        moduli (jnp.ndarray([valid_size, model.h12])): Moduli within fundamental domain.
        tau (jnp.ndarray([valid_size])): Axio-dilaton within fundamental domain.
        idx_valid (jnp.ndarray([batch_size])): Boolean mask over input batch, indicating position of valid fluxes.
    """
    if debug:
        print("Checking tau FD...")
    idx_tau = (tau.imag > 0.83) & (jnp.abs(tau.real) < 0.5)
    fluxes, moduli, tau = fluxes[idx_tau], moduli[idx_tau], tau[idx_tau]
    if debug:
        print("Check terminated.")
    return fluxes, moduli, tau, idx_tau


def check_LCS(fluxes, moduli, tau, model, debug=False):
    r"""Checks if input in LCS region.

    Args:
        fluxes (jnp.ndarray([batch_size, 4*(model.h12+1)])): Batch of fluxes.
        moduli (jnp.ndarray([batch_size, model.h12])): Batch of moduli.
        tau (jnp.ndarray([batch_size])): Batch of axio-dilaton.
        model (jvc.FluxVacuaFinder): Jaxvacua model specifying geometry.
        debug (bool): Log if terminated.

    Returns:
        fluxes (jnp.ndarray([valid_size, 4*(model.h12+1)])): Fluxes within LCS region.
        moduli (jnp.ndarray([valid_size, model.h12])): Moduli within LCS region.
        tau (jnp.ndarray([valid_size])): Axio-dilaton within LCS region.
        idx_cone (jnp.ndarray([batch_size])): Boolean mask over input batch, indicating position of valid fluxes.
    """
    if debug:
        print("Checking LCS region...")
    idx_cone = jnp.all(moduli.imag @ model.lcs_tree.hyperplanes.T > 0, axis=1)
    moduli, tau, fluxes = moduli[idx_cone], tau[idx_cone], fluxes[idx_cone]
    if debug:
        print("Check terminated.")
    return fluxes, moduli, tau, idx_cone


def check_monodromy_FD(
    fluxes,
    moduli,
    tau,
    model=None,
    debug=False,
):
    r"""Checks if input in fundamental domain of monodromy transformations.

    Args:
        fluxes (jnp.ndarray([batch_size, 4*(model.h12+1)])): Batch of fluxes.
        moduli (jnp.ndarray([batch_size, model.h12])): Batch of moduli.
        tau (jnp.ndarray([batch_size])): Batch of axio-dilaton.
        debug (bool): Log if terminated.
        model (jvc.FluxVacuaFinder): Needed for consistency (will be changed). Defaults to None.

    Returns:
        fluxes (jnp.ndarray([valid_size, 4*(model.h12+1)])): Fluxes within fundamental domain.
        moduli (jnp.ndarray([valid_size, model.h12])): Moduli within fundamental domain.
        tau (jnp.ndarray([valid_size])): Axio-dilaton within fundamental domain.
        idx_domain (jnp.ndarray([batch_size])): Boolean mask over input batch, indicating position of valid fluxes.
    """
    if debug:
        print("Checking monodromy FD...")
    idx_domain = jnp.all(jnp.abs(moduli.real) < 0.5, axis=-1)
    moduli, tau, fluxes = (
        moduli[idx_domain],
        tau[idx_domain],
        fluxes[idx_domain],
    )
    if debug:
        print("Check terminated.")
    return fluxes, moduli, tau, idx_domain


def get_unique(fluxes, moduli, tau, idx_samples, debug=False):
    r"""Return numerically unique flux vacua.

    Args:
        fluxes (jnp.ndarray([batch_size, 4*(model.h12+1)])): Batch of fluxes.
        moduli (jnp.ndarray([batch_size, model.h12])): Batch of moduli.
        tau (jnp.ndarray([batch_size])): Batch of axio-dilaton.
        idx_samples (jnp.ndarray([batch_size])): Boolean mask recording original position in sample array.
        debug (bool): Log if terminated.

    Returns:
        fluxes (jnp.ndarray([unique_size, 4*(model.h12+1)])): Unique fluxes.
        moduli (jnp.ndarray([unique_size, model.h12])): Unique moduli.
        tau (jnp.ndarray([unique_size])): Unique axio-dilaton
        idx_samples (jnp.ndarray([batch_size])): Updated boolean mask indicating unique original position in sample array.
    """
    if debug:
        print("Filtering unique...")
    fluxes, idx_unique = jnp.unique(fluxes, axis=0, return_index=True)
    idx_samples = idx_samples[idx_unique]
    moduli, tau = moduli[idx_unique], tau[idx_unique]
    if debug:
        print("Filtering terminated.")
    return fluxes, moduli, tau, idx_samples


def validate_vacua(
    fluxes,
    moduli,
    tau,
    idx_samples,
    model,
    out,
    tol,
    use_scipy,
    plot_trajectory=False,
    debug=False,
    debug_deep=False,
):
    # Check tadpole
    fluxes, idx_valid_tadpole = check_tadpole(model, fluxes, debug)
    idx_samples = idx_samples[idx_valid_tadpole]
    out["failed_tadpole"] = jnp.sum(~idx_valid_tadpole)

    if jnp.any(idx_valid_tadpole):
        # Solve F-term conditions
        moduli, tau = moduli[idx_valid_tadpole], tau[idx_valid_tadpole]
        z0 = moduli
        tau0 = tau
        fluxes, moduli, tau, idx_solved = solve(
            fluxes, z0, tau0, model, tol, use_scipy, debug_deep, debug
        )
        out["failed_to_solve"] = jnp.sum(~idx_solved)

        if plot_trajectory:
            plot_trajectory_dict = {"z0": z0[idx_solved], "solver_raw_z": moduli, "solver_is_valid": np.zeros(len(moduli), dtype=bool)}

        # If solutions were found, check conditions
        if jnp.any(idx_solved):
            idx_samples = idx_samples[idx_solved]
            # Map to FD
            fluxes, moduli, tau = map_to_FD(
                fluxes, moduli, tau, model, debug, debug_deep
            )

            # Perform checks
            check_fcts = [check_FD, check_LCS, check_monodromy_FD]
            check_failed = []

            if plot_trajectory:
                idx_check_lst = []

            for fct in check_fcts:
                fluxes, moduli, tau, idx_check = fct(fluxes, moduli, tau, model, debug)
                idx_samples = idx_samples[idx_check]
                check_failed.append(jnp.sum(~idx_check))
                if plot_trajectory:
                    idx_check_lst.append(idx_check)

            if plot_trajectory:
                valid_indices = np.arange(len(plot_trajectory_dict["solver_is_valid"]))
                for idx_check in idx_check_lst:
                    valid_indices = valid_indices[np.asarray(idx_check, dtype=bool)]
                plot_trajectory_dict["solver_is_valid"][valid_indices] = True
                np.savez(os.getcwd() + f"/data/sol_trajectories/{model.lcs_tree.h12}_{model.lcs_tree.h11}_trajectory.npz", z0=plot_trajectory_dict["z0"], solver_raw_z=plot_trajectory_dict["solver_raw_z"], solver_is_valid=plot_trajectory_dict["solver_is_valid"])

            # Update console output
            out["valid"] = len(fluxes)
            check_keys = ["tau_FD", "LCS", "m_FD"]
            for key, fail_rate in zip(check_keys, check_failed):
                out["failed_" + key] = fail_rate

            # Filter for unique
            fluxes, moduli, tau, idx_samples = get_unique(
                fluxes, moduli, tau, idx_samples, debug
            )
            out["valid_unique"] = len(fluxes)

            if len(fluxes) == 0:
                return None, None, None, None, out
            else:
                return fluxes, moduli, tau, idx_samples, out
        else:
            out["valid_unique"] = 0
            out["valid"] = 0
            return None, None, None, None, out
    else:
        out["valid_unique"] = 0
        out["valid"] = 0
        return None, None, None, None, out


def convert_samples_to_data(samples, discr, gs_inv=None):
    fluxes = samples[0]
    moduli = samples[1]
    moduli_real, moduli_imag = jnp.split(moduli, 2, axis=-1)
    moduli = (moduli_real + 1.0j * moduli_imag) / discr
    tau = samples[2]
    if gs_inv is not None:
        tau = (tau[:, 0] + 1.0j * gs_inv) / discr
    else:
        tau = (tau[:, 0] + 1.0j * tau[:, 1]) / discr
    return fluxes, moduli, tau


def process_sequence(sequence, tokenizer, conds, split_tokens=None):
    sequence = np.array(sequence)
    r"""Processes single token sequence.
                                
        Args:
            sequence (np.ndarray([seq_len])): One-dimensional array containing sequence of tokens as strings.
            tokenizer (utils.DictTokenizer): Tokenizer.
            conds (int or np.ndarray([num_conds])): Correct lengths of each part of sequence.
            split_tokens (np.ndarray([num_conds-1])): Split tokens indicating where to split the sequence. Defaults to None.

        Returns:
            data (list): List of different parts of sequence. Returns None if sequence invalid.
        """
    try:
        bos_idx = np.argwhere(sequence == "<BOS>").item()
    except:
        return None
    if split_tokens is not None:
        try:
            split_idx = np.array(
                [
                    np.argwhere(sequence == split_token).item()
                    for split_token in split_tokens
                ]
            )

            split_idx = np.insert(split_idx, 0, bos_idx)
            split_idx = np.append(split_idx, np.argwhere(sequence == "<EOS>").item())
            data = []
            for i in range(len(split_idx) - 1):
                data_iter, valid_mask = token_to_data(
                    sequence[None, split_idx[i] + 1 : split_idx[i + 1]],
                    tokenizer,
                    return_invalid=True,
                )
                if valid_mask.item():
                    data.append(data_iter.flatten())
                else:
                    return None
            lens = [len(data_iter) for data_iter in data]
            if np.all(lens == conds):
                return data
            else:
                return None
        except:
            return None
    else:
        data = token_to_data(sequence[None, bos_idx + 1 :], tokenizer).flatten()
        if len(data) == conds:
            return data
        else:
            return None



@jax.jit
def process_sample_batch(tokens, token_value, is_separator_token, PLUS_ID, MINUS_ID, EOS_ID):
    values = token_value[tokens]
    is_number = values >= 0
    is_plus  = tokens == PLUS_ID
    is_minus = tokens == MINUS_ID
    is_sign  = is_plus | is_minus
    is_eos = tokens == EOS_ID
    is_sep = is_separator_token[tokens]
    segment = jnp.cumsum(is_sep, -1) - is_sep.astype(jnp.int32)
    xs = (
        values.T,
        is_plus.T,
        is_minus.T,
        is_number.T,
        is_sign.T,
        is_sep.T,
        is_eos.T,
    )

    def step(carry, x):
        acc, sign, has_number, has_sign, valid = carry

        value, is_plus, is_minus, is_number, is_sign, is_sep, is_eos = x

        boundary = is_sign | is_sep | is_eos

        # -----------------------------------------
        # Emit previous number
        # -----------------------------------------

        emit = boundary & has_number & has_sign

        emitted_value = jnp.where(
            emit,
            sign * acc,
            0,
        )

        # -----------------------------------------
        # Validate
        # -----------------------------------------

        # Numeric chunks are only allowed after a sign.
        valid &= ~(is_number & ~has_sign)

        # A sign shouldn't terminate an empty number.
        #
        # You may or may not want this depending on BOS etc.
        valid &= ~(is_sign & has_sign & ~has_number)

        # Separator/EOS shouldn't terminate an empty signed number.
        valid &= ~((is_sep | is_eos) & has_sign & ~has_number)

        # -----------------------------------------
        # Finish old number
        # -----------------------------------------

        acc = jnp.where(boundary, 0, acc)

        has_number = jnp.where(
            boundary,
            False,
            has_number,
        )

        has_sign = jnp.where(
            boundary,
            False,
            has_sign,
        )

        # -----------------------------------------
        # Start new number
        # -----------------------------------------

        sign = jnp.where(is_plus, 1, sign)
        sign = jnp.where(is_minus, -1, sign)

        has_sign |= is_sign

        # -----------------------------------------
        # Accumulate
        # -----------------------------------------

        acc = jnp.where(
            is_number,
            acc * 1000 + value,
            acc,
        )

        has_number |= is_number

        return (
            acc,
            sign,
            has_number,
            has_sign,
            valid,
        ), (
            emitted_value,
            emit,
        )

    B = values.shape[0]
    initial = (
        jnp.zeros(B, dtype=jnp.int64),  # acc
        jnp.ones(B, dtype=jnp.int64),   # sign
        jnp.zeros(B, dtype=jnp.bool_),  # has_number
        jnp.zeros(B, dtype=jnp.bool_),  # has_sign
        jnp.ones(B, dtype=jnp.bool_),   # structurally valid
    )


    final_carry, (emitted_values, emit_mask) = jax.lax.scan(
        step,
        initial,
        xs,
    )
    _, _, _, _, scan_valid = final_carry

    emitted_values = emitted_values.T
    emit_mask = emit_mask.T


    return emitted_values, emit_mask, segment, scan_valid


def process_samples(
    samples_lst, tokenizer, conds, split_tokens=None, return_valid_idx=False
):
    r"""Processes generated samples for one or multiple prompts.

    Args:
        samples_lst (list): List of sampled sequences for each prompt.
        tokenizer (utils.DictTokenizer): Tokenizer.
        conds (np.ndarray([num_prompts, num_conds])): Correct lengths of each part of sequence for each prompt.
        split_tokens (np.ndarray([num_conds-1])): Split tokens indicating where to split the sequence. Defaults to None.
        return_valid_idx (bool): Whether to additionally return indices of valid samples. Defaults to False.

    Returns:
        samples_valid (list): Processed samples for each prompt. If return_valid_idx is True, also returns indices of valid samples.
    """
    if split_tokens is None:
        samples_valid = []
        for i in range(len(samples_lst)):
            fluxes_valid = [
                out[None, :]
                for out in (
                    process_sequence(sequence, tokenizer, conds[i])
                    for sequence in samples_lst[i]
                )
                if out is not None
            ]
            samples_valid.append(jnp.concatenate(fluxes_valid, axis=0))
        return samples_valid
    else:
        samples_valid = []
        valid_idx_per_prompt = []
        token_value = jnp.full(tokenizer.vocab_size, -1, dtype=int)
        token_value = token_value.at[-tokenizer.base:].set(jnp.arange(tokenizer.base))
        SEP_IDS = [tokenizer.dict[sep] for sep in split_tokens]
        is_separator_token = jnp.zeros(tokenizer.vocab_size, dtype=jnp.bool_)
        is_separator_token = is_separator_token.at[
            jnp.array(SEP_IDS)
        ].set(True)

        import sys
        jnp.set_printoptions(threshold=sys.maxsize)

        
        for samples, cond in zip(samples_lst, conds):
            emitted_values, emit_mask, segment, scan_valid = process_sample_batch(samples, token_value, is_separator_token, tokenizer.dict["+"], tokenizer.dict["-"], tokenizer.dict["<EOS>"])

            B, L = emitted_values.shape

            valid = scan_valid.copy()
            parts = []

            for seg_id, expected_len in enumerate(cond):
                expected_len = int(expected_len)

                # Which emitted numbers belong to this segment?
                # shape: [B, L]
                belongs = emit_mask & (segment == seg_id)

                # How many integers did each sample generate for this segment?
                # shape: [B]
                actual_len = jnp.sum(belongs, axis=-1)

                # This sample is valid only if this segment has exactly
                # the expected number of integers.
                valid &= actual_len == expected_len

                # Position of each emitted number within this segment.
                #
                # Example:
                #
                # belongs = [F, F, T, F, T, T, F]
                # pos     = [-1,-1, 0, 0, 1, 2, 2]
                #
                pos = jnp.cumsum(belongs, axis=-1) - 1

                # Allocate fixed-size output for this segment.
                part = jnp.zeros(
                    (B, expected_len),
                    dtype=emitted_values.dtype,
                )

                # Scatter emitted values into their position.
                batch_idx = jnp.arange(B)[:, None]

                safe_pos = jnp.clip(
                    pos,
                    0,
                    expected_len - 1,
                )

                contribution = jnp.where(
                    belongs & (pos < expected_len),
                    emitted_values,
                    0,
                )

                part = part.at[
                    batch_idx,
                    safe_pos,
                ].add(contribution)

                parts.append(part)

             # Require exactly the right number of separators.
            num_separators = jnp.sum(
                is_separator_token[samples],
                axis=-1,
            )

            valid &= num_separators == len(cond) - 1

            # Data-dependent operation: keep this outside @jax.jit
            valid_idx = jnp.where(valid)[0]

            parts = [
                part[valid]
                for part in parts
            ]
            samples_valid.append(parts)
            valid_idx_per_prompt.append(valid_idx)
        if return_valid_idx:
            return samples_valid, valid_idx_per_prompt
        else:
            return samples_valid

