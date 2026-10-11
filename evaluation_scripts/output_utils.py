import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from math import ceil
from pathlib import Path
from matplotlib.collections import LineCollection


def print_console_output(out, aux=None, verbose=False):
    r"""Prints details about solving iteration.

    Args:
        out (dict): Dictionary containing run data.
        verbose (bool): Controls print details. Defaults to False.
        aux (str): Auxiliary prompt (i.e. nflux, w0 or gs_inv).
    """
    print("------------------------")
    if aux is None:
        print(f"Geometry: {out['geometry']}")
    else:
        print(f"Aux prompt: {aux}")
    print("------------------------")
    if verbose:
        print(f"Failed tadpole: {out['failed_tadpole']}")
        if "failed_to_solve" in out:
            print(f"Failed to solve: {out['failed_to_solve']}")
            if "failed_tau_FD" in out:
                print(f"Failed tau FD: {out['failed_tau_FD']}")
                print(f"Failed LCS: {out['failed_LCS']}")
                print(f"Failed FD monodromies: {out['failed_m_FD']}")
    print(f"Valid (non-unique): {out['valid']}")
    print(f"Valid (unique): {out['valid_unique']}")
    if "num_in_train_data" in out:
        print(f"In training data: {out['num_in_train_data']}")


def write_latex_table(RUN_ID, all_hodge_pairs, log_dict_all_pairs, num_samples):
    r"""Writes table in latex summarizing evaluation. Saves file as 'latex_table_{RUN_ID}.txt'.

    Args:
        RUN_ID (int): Indicating evaluated model.
        all_hodge_pairs (list): All hodge pairs that were used in the prompts.
        log_dict_all_pairs (dict): Dictionary containing evaluation metrics for each prompt.
        num_samples (int): Number of samples per prompt.
    """
    with open(f"latex_table_{RUN_ID}.txt", "w") as f:
        f.write(r"""\begin{table}[!t]
                \centering
                \resizebox{\textwidth}{!}{
                \begin{tabular}{c|ccc|cccccc}
                """)
        for k in range(len(all_hodge_pairs)):
            f.write(r"prompt & $\#$sampled & $\#$valid & $\#$valid new & $h^{1,1}=")
            for i in range(len(all_hodge_pairs[k])):
                if i == 0:
                    f.write(rf"{all_hodge_pairs[k][i][1]}$ & ")
                else:
                    f.write(rf"${all_hodge_pairs[k][i][1]}$ & ")
            f.write(r"""$\Sigma$ \\
                    \hline
                    """)

            col_keys = [
                f"{col_geometry[0]} {col_geometry[1]}"
                for col_geometry in all_hodge_pairs[k]
            ]

            for i in range(len(all_hodge_pairs[k])):
                row_geometry = all_hodge_pairs[k][i]
                row_geometry_key = rf"{row_geometry[0]} {row_geometry[1]}"
                f.write(
                    f"$h^{{1,2}}={row_geometry[0]}, h^{{1,1}}={row_geometry[1]}$ & {num_samples}  & {log_dict_all_pairs[row_geometry_key]["total valid"]} & "
                )
                num_new = log_dict_all_pairs[row_geometry_key]["total valid"]
                for col_key in col_keys:
                    if log_dict_all_pairs[row_geometry_key][col_key]["in_train"] != 0:
                        num_new = "-"
                f.write(rf"{num_new} & ")

                valid_sum = 0
                for j in range(len(all_hodge_pairs[k])):
                    if i == j:
                        f.write(r"\cellcolor{yellow!25} ")
                    f.write(
                        rf"{log_dict_all_pairs[row_geometry_key][col_keys[j]]["valid"]} & "
                    )
                    valid_sum += log_dict_all_pairs[row_geometry_key][col_keys[j]][
                        "valid"
                    ]
                f.write(rf"""{valid_sum} \\ 
                        """)
            f.write(r"""\hline
                    """)
        f.write(r"""\end{tabular}}
                \caption{}
                \label{}
                \end{table}
                """)


def plot_hist(
    all_hodge_pairs, prompts, aux_prompts, valid_dict, x, filename, discr=None
):
    
    sns.set_theme(rc={"text.usetex": True})
    fig, ax = plt.subplots(
        len(all_hodge_pairs), 5, figsize=(len(all_hodge_pairs) * 5, 16)
    )
    for row in range(len(all_hodge_pairs)):
        for col in range(5):
            dataframes = []
            for i in range(len(aux_prompts)):
                idx = row * 5 * len(aux_prompts) + col * len(aux_prompts) + i
                key = f"{prompts[idx][0]} {prompts[idx][1]} {prompts[idx][2]}"
                if key in valid_dict:
                    if discr is None:
                        prompt_value = float(prompts[idx][-1])
                    else:
                        prompt_value = float(prompts[idx][-1]) / discr
                    df = pd.DataFrame(
                        {
                            x: valid_dict[key]["dist"],
                            "Prompt": prompt_value,
                        }
                    )
                    dataframes.append(df)
            dataframes = pd.concat(dataframes, ignore_index=True)
            prompt_values = sorted(dataframes["Prompt"].unique())
            color_map = dict(
                zip(prompt_values, sns.color_palette(n_colors=len(prompt_values)))
            )

            if discr is None:
                binwidth = 1
            else:
                binwidth = 0.1

            sns.histplot(
                data=dataframes,
                x=x,
                ax=ax[row, col],
                discrete=discr is None,
                binwidth=binwidth,
                hue="Prompt",
                stat="probability",
                palette=color_map,
                legend=False,
            )

            for i in range(len(aux_prompts)):
                idx = row * 5 * len(aux_prompts) + col * len(aux_prompts) + i
                key = f"{prompts[idx][0]} {prompts[idx][1]} {prompts[idx][2]}"
                if key in valid_dict:
                    if discr is None:
                        prompt_value = float(prompts[idx][-1])
                    else:
                        prompt_value = float(prompts[idx][-1]) / discr
                    ax[row, col].axvline(
                        prompt_value - 0.05,
                        ymax=100,
                        color=color_map.get(prompt_value, "red"),
                        linestyle="--",
                    )
            ax[row, col].set_ylabel(" ")
            ax[row, col].set_title(
                r"$\{h^{1,2}, h^{1,1}\}= \{$"
                + str(prompts[idx][0])
                + ", "
                + str(prompts[idx][1])
                + r"$\}$"
            )
    fig.subplots_adjust(wspace=0.2, hspace=0.35)
    plt.savefig(filename, bbox_inches="tight", dpi=300)


def plot_cont(
    all_hodge_pairs, prompts, aux_prompts, valid_dict, x, filename, discr=None):
    r"""Plot prompted versus sampled counts as a heatmap, alongside validity.
    
    Args:
        all_hodge_pairs (list): All hodge pairs that were used in the prompts. Shape is (#h12, #h11).
        prompts (list): List of tokenized prompts. 
        aux_prompts (np.ndarray): Array of additional prompts, e.g. $|W_0|$.
        valid_dict (dict): Dictionary containing distributions of the additional prompts together with validities.
        x (str): Name of axis. Depends on which additional prompt was chosen.
        filename (str): File name to be used for the figures.
        discr (float): Value used for discretizing. Defaults to None for input that was not discretized (only $N_{\rm flux}$).
   
    """

    if isinstance(aux_prompts, list):
        start_index = 0
    num_rows = len(all_hodge_pairs)*len(all_hodge_pairs[0])
    for row in range(num_rows):
        dataframes_dist = []
        validity = []
        prompt_values = []

        if isinstance(aux_prompts, list):
            iter_len = aux_prompts[row]
        else:
            iter_len = aux_prompts

        sns.set_theme(rc={"text.usetex": True})
        fig, ax = plt.subplots(
                1, 2, figsize=(10, 5)
            )

        for i in range(iter_len):
            
            if isinstance(aux_prompts, list):
                idx = start_index + i
            else:
                idx = row * aux_prompts + i
            key = f"{prompts[idx][0]} {prompts[idx][1]} {prompts[idx][2]}"
            if key in valid_dict:
                if discr is None:
                    prompt_value = float(prompts[idx][-1])
                else:
                    prompt_value = float(prompts[idx][-1]) / discr - 0.05
                df_dist = pd.DataFrame(
                    {
                        x + str(" sampled"): valid_dict[key]["dist"],
                        x + str(" prompted"): prompt_value
                    }
                )

                validity.append(valid_dict[key]["validity"]/1000.)
                prompt_values.append(prompt_value)   
                dataframes_dist.append(df_dist)
        if isinstance(aux_prompts, list):
            start_index = idx + 1
        dataframes_dist = pd.concat(dataframes_dist, ignore_index=True)
        dataframes_validity = pd.DataFrame({"Validity": validity, x: prompt_values})

        

        # Center unit-sized cells on integer values; use finer bins for
        # continuous quantities, as in plot_hist().

        if discr is not None:
            bins_x = np.arange(0.0,6.2,0.1)
            bins_y = np.arange(0.0, 9.1, 0.1)
            bins = (bins_x, bins_y)
        else:
            bins = 'auto'

        sns.histplot(
            data=dataframes_dist,
            x=x + str(" prompted"),
            y=x + str(" sampled"),
            discrete=discr is None,
            bins=bins,
            stat="count",
            cmap="viridis",
            cbar=True,
            thresh=0,
            ax=ax[0],
        )

        sns.lineplot(
            data=dataframes_dist,
            x = x + str(" prompted"),
            y = x + str(" prompted"),
            c="red",
            linestyle="--",
            linewidth=.5,
            ax=ax[0],
        )

        sns.lineplot(
            data=dataframes_validity,
            x = x,
            y = "Validity",
            ax=ax[1]
        )

        ax[0].set_title(
            r"$\{h^{1,2}, h^{1,1}\}= \{$"
            + str(prompts[idx][0])
            + ", "
            + str(prompts[idx][1])
            + r"$\}$"
        )
        ax[1].set_title("Validity")
        ax[1].set_ylabel(" ")
        ax[1].set_ylim([0.,1.])
        fig.subplots_adjust(wspace=0.2, hspace=0.35)
        plt.savefig(filename+f"_{prompts[idx][0]}_{prompts[idx][1]}.png", bbox_inches="tight", dpi=300)


def plot_cont_dual(
    all_hodge_pairs, prompts, aux_prompts, valid_dict_pre, valid_dict_grpo, x, filename, discr=None):
    r"""Plot prompted versus sampled counts as a heatmap, alongside validity.
    
    Args:
        all_hodge_pairs (list): All hodge pairs that were used in the prompts. Shape is (#h12, #h11).
        prompts (list): List of tokenized prompts. 
        aux_prompts (np.ndarray): Array of additional prompts, e.g. $|W_0|$.
        valid_dict_pre (dict): Dictionary containing distributions of the additional prompts together with validities for the pre-trained model.
        valid_dict_grpo (dict): Dictionary containing distributions of the additional prompts together with validities for the GRPO model.
        x (str): Name of axis. Depends on which additional prompt was chosen.
        filename (str): File name to be used for the figures.
        discr (float): Value used for discretizing. Defaults to None for input that was not discretized (only $N_{\rm flux}$).
   
    """

    if isinstance(aux_prompts, list):
        start_index = 0
    for h12_i, h12  in enumerate(all_hodge_pairs):
        if isinstance(aux_prompts, list):
            sharex = False
            height = 18
            height = 8
        else:
            sharex = True
            height = 17
            height = 8
        sns.set_theme(rc={"text.usetex": True})
        fig, ax = plt.subplots(
            2, 3, figsize=(15, height), sharex=sharex
        )
        for h11_i in range(len(h12)):
            geometry_i = h12_i * 5 + h11_i
            dataframes_dist_pre = []
            dataframes_dist_grpo = []
            validity_pre = []
            validity_none_unique_pre = []
            validity_grpo = []
            validity_none_unique_grpo = []
            prompt_values_pre = []
            prompt_values_grpo = []
            prompt_values = []
            if isinstance(aux_prompts, list):
                iter_len = aux_prompts[geometry_i]
            else:
                iter_len = aux_prompts

            for i in range(iter_len):
                
                if isinstance(aux_prompts, list):
                    idx = start_index + i
                else:
                    idx = geometry_i * aux_prompts + i
                key = f"{prompts[idx][0]} {prompts[idx][1]} {prompts[idx][2]}"
                if discr is None:
                    prompt_value = float(prompts[idx][-1])
                else:
                    prompt_value = float(prompts[idx][-1]) / discr - 0.05
                if key in valid_dict_pre:
                    if x == f"$g_s$":
                        df_dist_pre = pd.DataFrame(
                            {
                                x + str(" sampled"): 1/valid_dict_pre[key]["dist"],
                                x + str(" prompted"): 1/prompt_value,
                            }
                        )
                    else:
                        df_dist_pre = pd.DataFrame(
                            {
                                x + str(" sampled"): valid_dict_pre[key]["dist"],
                                x + str(" prompted"): prompt_value,
                            }
                        )
                    dataframes_dist_pre.append(df_dist_pre)
                    validity_pre.append(valid_dict_pre[key]["validity"]/1000.)
                    validity_none_unique_pre.append(valid_dict_pre[key]["validity_non_unique"]/1000.)
                    prompt_values_pre.append(prompt_value)
                else:
                    validity_pre.append(0.)
                    validity_none_unique_pre.append(0.)
                if key in valid_dict_grpo:
                    if x == r"$g_s$":
                        df_dist_grpo = pd.DataFrame(
                            {
                                x + str(" sampled"): 1/valid_dict_grpo[key]["dist"],
                                x + str(" prompted"): 1/prompt_value
                            }
                        )
                    else:
                        df_dist_grpo = pd.DataFrame(
                            {
                                x + str(" sampled"): valid_dict_grpo[key]["dist"],
                                x + str(" prompted"): prompt_value
                            }
                        )
                    dataframes_dist_grpo.append(df_dist_grpo)
                    validity_grpo.append(valid_dict_grpo[key]["validity"]/1000.)
                    validity_none_unique_grpo.append((valid_dict_grpo[key]["validity_non_unique"]/1000.))
                    prompt_values_grpo.append(prompt_value)
                else:
                    validity_grpo.append(0.)
                    validity_none_unique_grpo.append(0.)
                prompt_values.append(prompt_value) 
                
            if isinstance(aux_prompts, list):
                start_index = idx + 1
            dataframes_dist_pre = pd.concat(dataframes_dist_pre, ignore_index=True)
            dataframes_dist_grpo = pd.concat(dataframes_dist_grpo, ignore_index=True)
            if x == r"$g_s$":
                prompt_values = 1/np.flip(prompt_values)

            dataframes_validity_pre = pd.DataFrame({"Validity": validity_pre, x: prompt_values, "type": "pre_trained (unique)"})
            dataframes_validity_grpo = pd.DataFrame({"Validity": validity_grpo, x: prompt_values, "type": "GRPO (unique)"})
            dataframes_validity_non_unique_grpo = pd.DataFrame({"Validity": validity_none_unique_grpo, x: prompt_values, "type": "GRPO"})
            dataframes_validity = pd.concat(
                [dataframes_validity_pre, dataframes_validity_grpo, dataframes_validity_non_unique_grpo],
                ignore_index=True,
            )

            

            # Center unit-sized cells on integer values; use finer bins for
            # continuous quantities, as in plot_hist().

            if discr is not None:
                if x == r"$1/g_s$":
                    bins_x = np.arange(1.0,6.2,0.1)
                    bins_y = np.arange(1.0, 9.1, 0.1)
                elif x == r"$g_s$":
                    bins_x = 1/np.flip(np.arange(.9,6.2,0.1))
                    bins_y = 1/np.flip(np.arange(.9, 9.1, 0.1))
                else:
                    bins_x = np.arange(0.0,6.2,0.1)
                    bins_y = np.arange(0.0, 9.1, 0.1)
                bins = (bins_x, bins_y)
            else:
                bins = 'auto'

            sns.histplot(
                data=dataframes_dist_pre,
                x=x + str(" prompted"),
                y=x + str(" sampled"),
                discrete=discr is None,
                bins=bins,
                stat="count",
                cmap="viridis",
                cbar=True,
                thresh=0,
                ax=ax[h11_i, 0],
            )

            sns.lineplot(
                data=dataframes_dist_pre,
                x = x + str(" prompted"),
                y = x + str(" prompted"),
                c="lightcoral",
                linestyle="--",
                linewidth=1.,
                ax=ax[h11_i, 0],
            )

            sns.histplot(
                data=dataframes_dist_grpo,
                x=x + str(" prompted"),
                y=x + str(" sampled"),
                discrete=discr is None,
                bins=bins,
                stat="count",
                cmap="viridis",
                cbar=True,
                thresh=0,
                ax=ax[h11_i, 1],
            )

            sns.lineplot(
                data=dataframes_dist_grpo,
                x = x + str(" prompted"),
                y = x + str(" prompted"),
                c="lightcoral",
                linestyle="--",
                linewidth=1.,
                ax=ax[h11_i, 1],
            )

            colors = sns.color_palette()
            sns.lineplot(
                data=dataframes_validity,
                hue="type",
                style="type",
                palette={
                    "pre_trained (unique)": colors[0],
                    "GRPO (unique)": colors[1],
                    "GRPO": colors[1],
                },
                dashes={
                    "pre_trained (unique)": "",
                    "GRPO (unique)": "",
                    "GRPO": (5, 2),
                },
                x = x,
                y = "Validity",
                ax=ax[h11_i, 2]
            )

            ax[h11_i, 0].set_title(
                r"$\{h^{1,2}, h^{1,1}\}= \{$"
                + str(prompts[idx][0])
                + ", "
                + str(prompts[idx][1])
                + r"$\}$, pre-trained"
            )
            ax[h11_i, 1].set_title(
                        r"$\{h^{1,2}, h^{1,1}\}= \{$"
                        + str(prompts[idx][0])
                        + ", "
                        + str(prompts[idx][1])
                        + r"$\}$, GRPO"
                    )
            ax[h11_i, 2].set_title(r"$\{h^{1,2}, h^{1,1}\}= \{$"
                                    + str(prompts[idx][0])
                                    + ", "
                                    + str(prompts[idx][1])
                                    + r"$\}$, Validity")

            if isinstance(aux_prompts, list) and h11_i != 4:
                ax[h11_i, 0].set_xlabel(" ")
                ax[h11_i, 1].set_xlabel(" ")
                ax[h11_i, 2].set_xlabel(" ")
            ax[h11_i, 2].set_ylabel(" ")
            ax[h11_i, 2].set_ylim([0.,1.])
            handles, labels = ax[h11_i, 2].get_legend_handles_labels()
            ax[h11_i, 2].legend(handles=handles, labels=labels)
            if isinstance(aux_prompts, list):
                hspace=0.25
            else:
                hspace=0.15
            fig.subplots_adjust(wspace=0.15, hspace=hspace)
        plt.savefig(filename+f"_all_h12_{prompts[idx][0]}.png", bbox_inches="tight", dpi=300)


def plot_imaginary_moduli_correlation_matrix(
    npz_path,
    h11,
    sample_fraction=0.40,
    random_seed=42,
    solution="raw",
    padding=0.08,
    save_path=None,
    dpi=300,
):
    """
    Plot a lower-triangular matrix of all pairwise Im(z_i)-Im(z_j)
    initializer-to-solution displacements.

    The diagonal marginal distributions are omitted. Only solver-valid
    samples are considered.

    Parameters
    ----------
    npz_path : str or Path
        Evaluation NPZ containing z0, solver outputs, and solver_is_valid.
    sample_fraction : float, default=0.40
        Fraction of valid samples to display.
    random_seed : int, default=42
        Random seed used to select the displayed subset.
    solution : {"raw", "canonical"}, default="raw"
        "raw" uses solver_raw_z, the direct SciPy root.
        "canonical" uses solver_z, the symmetry-mapped solution.
    padding : float, default=0.08
        Fractional padding added to each modulus range.
    save_path : str or Path, optional
        Figure output path. SVG or PDF gives the sharpest result.
    dpi : int, default=300
        Resolution when saving a raster image.

    Returns
    -------
    fig, axes, selected_indices
    """
    npz_path = Path(npz_path)

    # Resolve repository-relative paths from the notebook directory.
    if not npz_path.is_file() and not npz_path.is_absolute():
        resolved = next(
            (
                parent / npz_path
                for parent in (Path.cwd(), *Path.cwd().parents)
                if (parent / npz_path).is_file()
            ),
            None,
        )
        if resolved is not None:
            npz_path = resolved

    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)

    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must lie in (0, 1].")

    if solution not in {"raw", "canonical"}:
        raise ValueError("solution must be 'raw' or 'canonical'.")

    solution_key = (
        "solver_raw_z" if solution == "raw" else "solver_z"
    )
    solution_label = (
        "SciPy root solution"
        if solution == "raw"
        else "canonical solution"
    )

    # load samples from a saved NPZ archive. The solver_is_valid mask is used to filter out invalid samples.
    with np.load(npz_path, allow_pickle=False) as data:
        required = {"z0", solution_key, "solver_is_valid"}
        missing = sorted(required.difference(data.files))

        if missing:
            raise KeyError(f"{npz_path} is missing arrays: {missing}")

        initial_z = np.asarray(data["z0"], dtype=np.complex128)
        final_z = np.asarray(data[solution_key], dtype=np.complex128)
        solver_valid = np.asarray(
            data["solver_is_valid"], dtype=bool
        ).reshape(-1)

        if initial_z.ndim != 2:
            raise ValueError(
                f"z0 must have shape (samples, h12), found {initial_z.shape}."
            )

        if final_z.shape != initial_z.shape:
            raise ValueError(
                f"{solution_key} has shape {final_z.shape}, "
                f"expected {initial_z.shape}."
            )

        n_samples, h12 = initial_z.shape

        if solver_valid.size != n_samples:
            raise ValueError(
                "solver_is_valid is not aligned with the moduli arrays."
            )

        geometry_id = (
            int(np.asarray(data["trajectory_geometry_id"]).reshape(-1)[0])
            if "trajectory_geometry_id" in data
            else npz_path.stem
        )

    if h12 < 2:
        raise ValueError(
            "At least two complex-structure moduli are required."
        )

    # Keep finite, solver-valid rows.
    finite = (
        np.all(np.isfinite(initial_z.real), axis=1)
        & np.all(np.isfinite(initial_z.imag), axis=1)
        & np.all(np.isfinite(final_z.real), axis=1)
        & np.all(np.isfinite(final_z.imag), axis=1)
    )

    valid_indices = np.flatnonzero(solver_valid & finite)

    if valid_indices.size == 0:
        raise ValueError("No finite solver-valid samples were found.")

    number_to_plot = max(
        1,
        int(ceil(sample_fraction * valid_indices.size)),
    )

    rng = np.random.default_rng(random_seed)
    selected_indices = np.sort(
        rng.choice(
            valid_indices,
            size=number_to_plot,
            replace=False,
        )
    )

    initial_imag = initial_z[selected_indices].imag
    final_imag = final_z[selected_indices].imag

    # Calculate one content-adaptive range for each modulus. This makes a
    # modulus use the same scale wherever it appears in the matrix.
    modulus_limits = []

    for modulus_index in range(h12):
        values = np.concatenate(
            [
                initial_imag[:, modulus_index],
                final_imag[:, modulus_index],
            ]
        )

        low = float(values.min())
        high = float(values.max())
        span = high - low

        if span == 0:
            span = max(abs(low), 1.0) * 0.1

        modulus_limits.append(
            (
                low - padding * span,
                high + padding * span,
            )
        )

    # A compact strict-lower-triangle layout:
    # rows represent z_2,...,z_h12 and columns z_1,...,z_(h12-1).
    grid_size = h12 - 1

    initial_color = "#8A8A8A"
    final_color = "#F28E2B"
    line_color = "#9CA3AF"

    with sns.axes_style("darkgrid"), sns.plotting_context(
        "talk",
        font_scale=0.80,
    ):
        fig, axes = plt.subplots(
            grid_size,
            grid_size,
            figsize=(
                3.5 * grid_size + 2.4,
                3.35 * grid_size + 0.8,
            ),
            squeeze=False,
            facecolor="white",
        )

        for row in range(grid_size):
            # y variable is z_j, beginning with z_2.
            j = row + 1

            for column in range(grid_size):
                ax = axes[row, column]
                i = column

                # Keep only pairs with i < j.
                if i >= j:
                    ax.set_visible(False)
                    continue

                initial_x = initial_imag[:, i]
                initial_y = initial_imag[:, j]
                final_x = final_imag[:, i]
                final_y = final_imag[:, j]

                # Shape: (samples, two endpoints, x/y coordinates).
                segments = np.stack(
                    [
                        np.column_stack([initial_x, initial_y]),
                        np.column_stack([final_x, final_y]),
                    ],
                    axis=1,
                )

                ax.add_collection(
                    LineCollection(
                        segments,
                        colors=line_color,
                        linewidths=1,
                        alpha=0.4,
                        zorder=1,
                    )
                )

                ax.scatter(
                    initial_x,
                    initial_y,
                    s=17,
                    color=initial_color,
                    alpha=0.50,
                    edgecolors="none",
                    zorder=2,
                )

                ax.scatter(
                    final_x,
                    final_y,
                    s=21,
                    color=final_color,
                    alpha=0.85,
                    edgecolors="white",
                    linewidths=0.25,
                    zorder=3,
                )

                ax.set_xlim(modulus_limits[i])
                ax.set_ylim(modulus_limits[j])
                ax.set_aspect("auto")

                # Only the bottom row needs x labels.
                if row == grid_size - 1:
                    ax.set_xlabel(
                        rf"$\mathrm{{Im}}(z_{{{i + 1}}})$"
                    )
                else:
                    ax.tick_params(labelbottom=False)

                # Only the first column needs y labels.
                if column == 0:
                    ax.set_ylabel(
                        rf"$\mathrm{{Im}}(z_{{{j + 1}}})$"
                    )
                else:
                    ax.tick_params(labelleft=False)

                ax.grid(
                    True,
                    color="white",
                    linewidth=1.0,
                    alpha=0.92,
                )

                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color("#D1D5DB")
                    spine.set_linewidth(0.9)


        plot_left = 0.09
        plot_right = 0.81
        plot_center = (plot_left + plot_right) / 2

        fig.suptitle(
            r"$\{h^{1,2}, h^{1,1}\}= \{$" + str(h12) + ", " + str(h11) + r"$\}$",
            x=plot_center,
            y=0.99,
            ha="center",
            fontsize=15,
            fontweight="semibold",
        )

        fig.subplots_adjust(
            left=plot_left,
            right=plot_right,
            bottom=0.08,
            top=0.95,
            wspace=0.10,
            hspace=0.10,
        )

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            fig.savefig(
                save_path,
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
            )
            print(f"Saved to {save_path}")
