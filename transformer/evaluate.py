from jax_transformer.utils import sample
from evaluation_scripts.evaluation_utils import *
from evaluation_scripts.output_utils import *
import os
import jax
import jax.numpy as jnp
import pickle
from data_scripts.tokenizer_utils import int_to_token
import numpy as np
from tqdm.auto import tqdm
import jaxvacua as jvc  # type: ignore
from data_scripts.tokenizer_utils import (
    hodge_numbers_to_token,
    int_nums_to_token,
    a_matrix_to_token,
    b_vector_to_token,
    chi_to_token,
)
import argparse
import warnings
import time

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--RUN_ID", type=int)
parser.add_argument("--load_ckpt", type=str)
parser.add_argument("--mod", action="store_true")
parser.add_argument("--single", action="store_true")
parser.add_argument("--nflux", action="store_true")
parser.add_argument("--w0", action="store_true")
parser.add_argument("--gs_inv", action="store_true")
parser.add_argument("--num_samples", type=int, default=1000)
parser.add_argument("--tol", type=float, default=1e-10)
parser.add_argument("--sampling_temp", type=float, default=0.6)
parser.add_argument("--discr", type=float, default=10.0)
parser.add_argument("--only_sample", action="store_true")
parser.add_argument("--samples_file", type=str)
parser.add_argument("--only_evaluate", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--use_newton", action="store_true")
parser.add_argument("--save_valid", action="store_true")

cwd = os.getcwd()
args = parser.parse_args()

assert (
    args.RUN_ID is not None and args.load_ckpt is not None
), "--RUN_ID and --load_ckpt always have to be set."
assert (
    np.sum([args.mod, args.nflux, args.w0, args.gs_inv]) == 1
), "Invalid option set. Needs to be --mod --nflux --gs_inv OR --w0."

if args.only_evaluate:
    assert (
        args.samples_file is not None
    ), "--only_evaluate is set but --samples_file is None."
    
if args.only_sample:
    assert (
        args.samples_file is not None
    ), "--only_sample is set but --samples_file is None."
    
    if os.path.isfile(cwd + f"/data/samples/{args.samples_file}.p"):
        raise (
            FileExistsError(
                f"--samples_file is {args.samples_file} but file already exists."
            )
        )

# Configure JAX compilation cache
# jax.config.update("jax_compilation_cache_max_size", int(3e10))

###################
#   DEFINITIONS   #
###################
geometry_log_path = cwd + "/data/data_multiple/geometry_log_degree_10.p"

use_scipy = not args.use_newton


# Ensure compatibility with older checkpoints
if "GRPO" in args.load_ckpt:
    override_opt = True
    GRPO_ID = args.load_ckpt.split("/")[0].split("_")[-1]
else:
    override_opt = False
    GRPO_ID = None
config = load_config(args.RUN_ID, cwd)
state, model, tokenizer = load_model_from_config(
    args.RUN_ID, config, args.load_ckpt, cwd, override_opt=override_opt, GRPO_ID=GRPO_ID
)
max_len = np.min([80, model.max_len_out])

# Set prompts
all_hodge_pairs = np.array(
    [
        [[2, 120], [2, 128], [2, 132], [2, 144], [2, 272]],
        [[3, 99], [3, 107], [3, 115], [3, 119], [3, 243]],
        [[4, 98], [4, 118], [4, 122], [4, 130], [4, 142]],
        [[5, 89], [5, 149], [5, 153], [5, 165], [5, 213]],
    ]
)


prompts = all_hodge_pairs.reshape(-1, 2)

# Load data
train_data_file = "_".join(
    config["train_data_flux"].split("/")[-1].split(".")[0].split("_")[2:]
)
h12_lst = all_hodge_pairs[:, 0, 0]
raw_data_paths = {
    h12: cwd + f"/data/data_multiple/train_data_raw_{train_data_file}_{h12}.npy"
    for h12 in h12_lst
}
with open(geometry_log_path, "rb") as f:
    geometry_log = pickle.load(f)

if args.w0:
    prompt_with_w0 = []
    aux_prompts = np.array([1.1])
    for prompt in prompts:
        prompt = np.repeat(prompt[None, :], len(aux_prompts), axis=0)
        prompt = np.concatenate(
            [prompt, np.round(args.discr * aux_prompts)[:, None].astype(np.int16)],
            axis=-1,
        )
        prompt_with_w0.append(prompt)
    prompts = np.concatenate(prompt_with_w0, axis=0)

if args.nflux:
    prompt_with_nflux = []
    aux_prompts_len = []
    for prompt in prompts:
        geometry_key = f"{prompt[0]} {prompt[1]}"
        h12 = prompt[0]
        model_ID = geometry_log[geometry_key]["model_ID"]
        jvc_model = jvc.FluxVacuaFinder(h12=h12, model_ID=model_ID, maximum_degree=10)
        aux_prompts = np.array([40])
        aux_prompts_len.append(len(aux_prompts))
        prompt = np.repeat(prompt[None, :], len(aux_prompts), axis=0)
        prompt = np.concatenate([prompt, aux_prompts[:, None]], axis=-1)
        prompt_with_nflux.append(prompt)
    prompts = np.concatenate(prompt_with_nflux, axis=0)

if args.gs_inv:
    prompt_with_gs_inv = []
    aux_prompts = np.array([4.0])
    for prompt in prompts:
        prompt = np.repeat(prompt[None, :], len(aux_prompts), axis=0)
        prompt = np.concatenate(
            [prompt, np.round(args.discr * aux_prompts)[:, None].astype(np.int16)],
            axis=-1,
        )
        prompt_with_gs_inv.append(prompt)
    prompts = np.concatenate(prompt_with_gs_inv, axis=0)


prompts_tok = []
for prompt in prompts:
    prompt_tok, _ = int_to_token(
        np.atleast_2d(prompt), config["base"], do_padding=False
    )
    prompts_tok.append(prompt_tok[:, 1:].flatten())

# Define conditions

if args.w0 or args.gs_inv:
    mult = len(aux_prompts)
elif args.nflux:
    mult = aux_prompts_len
else:
    mult = 1
num_per_row = all_hodge_pairs.shape[-2]
conds = 4 * (np.repeat(np.array(h12_lst), num_per_row) + 1)
conds = np.repeat(conds, mult)[:, None]
if args.gs_inv:
    t_conds = 1 * np.ones(mult * len(h12_lst) * num_per_row)[:, None]
elif args.nflux:
    t_conds = 2 * np.ones(np.sum(mult))[:, None]
else:
    t_conds = 2 * np.ones(mult * len(h12_lst) * num_per_row)[:, None]
z_conds = np.repeat(2 * np.array(h12_lst), num_per_row)
z_conds = np.repeat(z_conds, mult)[:, None]
conds = np.concatenate([conds, z_conds, t_conds], axis=-1)
split_tokens = ["<M_START>", "<T_START>"]

###################
#     Sampling    #
###################

if args.only_evaluate:
    assert not args.only_sample, "Both only_evaluate and only_sample were set to True."
    if os.path.isfile(cwd + f"/data/samples/{args.samples_file}.p"):
        samples = pickle.load(
            open(
                cwd + f"/data/samples/{args.samples_file}.p",
                "rb",
            )
        )
    else:
        raise FileNotFoundError(
            f"only_evaluate was set to True but couldn't find samples at {cwd}/data/samples/{args.samples_file}.p"
        )
else:
    # Maximum length of samples
    samples_lst = []
    print(f"Sampling...")
    time0 = time.time()
    for i, prompt in enumerate(tqdm(prompts_tok)):
        samples = sample(
            state=state,
            model_type=config["model"],
            num_samples=args.num_samples,
            max_len=max_len,
            tokenizer=tokenizer,
            prompt=prompt,
            sampling_temp=args.sampling_temp,
        )
        samples_lst.append(samples)
    # WARNING: This does not return unique values

    print(f"Processing samples...")
    jax.config.update("jax_enable_x64", True)
    samples = process_samples(samples_lst, tokenizer, conds, split_tokens)
    time1 = time.time()
    print(
        f"Sampling time in seconds (including processing): {np.round(time1-time0, 1)}"
    )
if args.only_sample:
    print(f"Saving...")
    pickle.dump(
        samples,
        open(
            cwd + f"/data/samples/{args.samples_file}.p",
            "wb",
        ),
    )
    exit()

# DEBUG: CLEAR CACHES
jax.clear_caches()
###################
#    Evaluation   #
###################

fluxes_train_dict = {}
mean_validity = []
if args.nflux or args.w0 or args.gs_inv:
    valid_aux = {}
if args.save_valid:
    assert args.single is not None, "--save_valid only supported for --single."
    valid_dict = {}
print(f"Validating...")
time0 = time.time()
# Needed since N_flux prompts have different sizes
if args.nflux:
    start_idx = 0
for k in range(len(all_hodge_pairs)):
    h12 = all_hodge_pairs[k][0][0]
    # Load original training data
    fluxes_train = jnp.load(
        raw_data_paths[h12],
        mmap_mode="r",
    )
    training_data = set(map(tuple, np.array(fluxes_train)))
    for i in range(len(all_hodge_pairs[k])):

        # Print header for geometry
        print("##################################")
        print(f"Prompted geometry: {all_hodge_pairs[k][i]}")

        # Check all geometries with the same h12 for each prompt
        if args.mod:
            num_valid_total = 0
            fluxes_prev = set()
            print(f"Num samples: {len(samples[k * 5 + i][0])}")
            log_dict = {}

        for j in range(len(all_hodge_pairs[k])):
            if (args.single or args.nflux or args.w0 or args.gs_inv) and not np.all(
                all_hodge_pairs[k][j] == all_hodge_pairs[k][i]
            ):
                continue
            # Prepare dict for console output
            out = {}
            geometry_key = f"{all_hodge_pairs[k][j][0]} {all_hodge_pairs[k][j][1]}"
            out["geometry"] = geometry_key
            # Prepare model
            model_ID = geometry_log[geometry_key]["model_ID"]
            model = jvc.FluxVacuaFinder(h12=h12, model_ID=model_ID, maximum_degree=10)
            # Needed for compatibility with older checkpoints
            if args.RUN_ID <= 35:
                model = to_old_a_matrix(model)

            if args.w0 or args.gs_inv:
                len_aux_iter = len(aux_prompts)
            elif args.nflux:
                len_aux_iter = len(aux_prompts)
            else:
                len_aux_iter = 1
            for l in range(len_aux_iter):
                if args.nflux and (l + 1) % 100 == 0:
                    jax.clear_caches()
                if args.mod:
                    samples_iter = samples[k * 5 + i].copy()
                elif args.nflux:
                    idx = start_idx + l
                    samples_iter = samples[idx]
                else:
                    if (
                        args.gs_inv
                        and (l + 1) % 10 == 0
                        and all_hodge_pairs[k][j][0] == 5
                    ):
                        jax.clear_caches()
                    idx = (k * 5 + i) * len(aux_prompts) + l
                    samples_iter = samples[idx]
                if args.gs_inv:
                    gs_inv = int(prompts[idx, -1])
                else:
                    gs_inv = None
                if samples_iter is not None:
                    fluxes, moduli, tau = convert_samples_to_data(
                        samples_iter, args.discr, gs_inv
                    )
                else:
                    fluxes = None

                if fluxes is not None:
                    idx_samples = jnp.arange(len(fluxes))
                    fluxes, moduli, tau, idx_samples, out = validate_vacua(
                        fluxes,
                        moduli,
                        tau,
                        idx_samples,
                        model,
                        out,
                        args.tol,
                        use_scipy,
                        False,
                        False,
                        False,
                    )

                else:
                    out["valid"] = 0
                    out["valid_unique"] = 0
                mean_validity.append(out["valid_unique"])
                if fluxes is not None:
                    if args.nflux:
                        valid_aux[
                            f"{prompts[idx][0]} {prompts[idx][1]} {prompts[idx][2]}"
                        ] = {
                            "dist": model.tadpole(fluxes),
                            "validity": out["valid_unique"],
                        }
                    if args.w0:
                        valid_aux[
                            f"{prompts[idx][0]} {prompts[idx][1]} {prompts[idx][2]}"
                        ] = {
                            "dist": jnp.abs(
                                model.superpotential_gauge_invariant(
                                    moduli, tau, fluxes
                                )
                            ),
                            "validity": out["valid_unique"],
                        }
                    if args.gs_inv:
                        valid_aux[
                            f"{prompts[idx][0]} {prompts[idx][1]} {prompts[idx][2]}"
                        ] = {"dist": jnp.imag(tau), "validity": out["valid_unique"]}

                    # Check if in training data
                    fluxes_np = np.array(fluxes)
                    num_in_train_data = sum(
                        tuple(row) in training_data for row in fluxes_np
                    )
                    out["num_in_train_data"] = num_in_train_data

                    # Update genuinely new solutions
                    if args.mod and not args.single:
                        num_new = 0
                        for idx_sample in np.array(idx_samples):
                            if idx_sample not in fluxes_prev:
                                fluxes_prev.add(idx_sample)
                                num_new += 1
                        num_valid_total += num_new

                    if args.save_valid and np.all(
                        all_hodge_pairs[k][j] == all_hodge_pairs[k][i]
                    ):
                        valid_dict[geometry_key] = {
                            "fluxes": fluxes,
                            "moduli": moduli,
                            "tau": tau,
                        }
                # Print console output
                if args.nflux or args.w0 or args.gs_inv:
                    aux = prompts[idx][-1]
                else:
                    aux = None
                print_console_output(out, aux, args.verbose)
            if args.nflux:
                start_idx = idx + 1

            if args.nflux or args.w0 or args.gs_inv:
                jax.clear_caches()
        if args.mod and not args.single:
            print("------------------------")
            print(f"Total valid: {num_valid_total}")
            jax.block_until_ready(fluxes_prev)
            del fluxes_prev
        jax.clear_caches()
    jax.block_until_ready(fluxes_train)
    del fluxes_train
if args.nflux or args.w0 or args.gs_inv:
    if args.nflux:
        file_str = "nflux"
    if args.w0:
        file_str = "w0"
    if args.gs_inv:
        file_str = "gs_inv"
    if not os.path.isfile(
        cwd
        + f"/data/samples/{file_str}_valid_{args.RUN_ID}_{args.load_ckpt.replace('/', '_')}.p"
    ):
        pickle.dump(
            valid_aux,
            open(
                cwd
                + f"/data/samples/{file_str}_valid_{args.RUN_ID}_{args.load_ckpt.replace('/', '_')}.p",
                "wb",
            ),
        )
    else:
        raise FileExistsError(
            f"Tried to save sample distributions at {cwd}/data/samples/{file_str}_valid_{args.RUN_ID}_{args.load_ckpt.replace('/', '_')}.p. File already exists."
        )
if args.single:
    print("------------------------")
    print(
        f"Mean validity: {jnp.mean(jnp.array(mean_validity)*100.)/float(args.num_samples)}"
    )
    print("------------------------")


time1 = time.time()
print(f"Evaluation time in seconds: {np.round(time1-time0, 1)}")
