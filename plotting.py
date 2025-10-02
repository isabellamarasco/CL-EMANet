import os
import argparse
from typing import List, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt


########################################
############# LABEL MAP ################
########################################
RUN_LABEL_MAP = {
    "no_no_resmlp": "No Normalization",
    "global_no_resmlp": "Global",
    "local_no_resmlp": "Local",
    "CN_no_resmlp": "CN",
    "ema_no_resmlp": "EMANet",
    "no_er_resmlp": "No Normalization",
    "global_er_resmlp": "Global",
    "local_er_resmlp": "Local",
    "CN_er_resmlp": "CN",
    "ema_er_resmlp": "EMANet",
    "no_reservoir_resmlp": "No Normalization",
    "global_reservoir_resmlp": "Global",
    "local_reservoir_resmlp": "Local",
    "CN_reservoir_resmlp": "CN",
    "ema_reservoir_resmlp": "EMANet",
    "no_agem_resmlp": "No Normalization",
    "global_agem_resmlp": "Global",
    "local_agem_resmlp": "Local",
    "CN_agem_resmlp": "CN",
    "ema_agem_resmlp": "EMANet",
    "no_ogd_resmlp": "No Normalization",
    "global_ogd_resmlp": "Global",
    "local_ogd_resmlp": "Local",
    "CN_ogd_resmlp": "CN",
    "ema_ogd_resmlp": "EMANet",
    "no_der_resmlp": "No Normalization",
    "global_der_resmlp": "Global",
    "local_der_resmlp": "Local",
    "CN_der_resmlp": "CN",
    "ema_der_resmlp": "EMANet",
    "no_ewc_resmlp": "No Normalization",
    "global_ewc_resmlp": "Global",
    "local_ewc_resmlp": "Local",
    "CN_ewc_resmlp": "CN",
    "ema_ewc_resmlp": "EMANet",
    "no_lwf_resmlp": "No Normalization",
    "global_lwf_resmlp": "Global",
    "local_lwf_resmlp": "Local",
    "CN_lwf_resmlp": "CN",
    "ema_lwf_resmlp": "EMANet",
}

# Fixed legend/curve order
ORDER = ["No Normalization", "Global", "Local", "CN", "EMANet"]


########################################
########### PARSE ARGUMENTS ############
########################################
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare multiple runs on a single sequential timeline (seen & current)."
    )

    # Output
    parser.add_argument(
        "--save_plot_dir",
        type=str,
        default="./plots",
        help="Directory to save generated plots.",
    )

    # Inputs
    parser.add_argument(
        "--compare_paths",
        nargs="+",
        default=None,
        help="Explicit list of .pt result files to compare (all curves on same figures).",
    )
    parser.add_argument(
        "--all_with",
        type=str,
        default=None,
        help="Substring to match; include all .pt in --results_dir whose names contain this substring (case-insensitive by default).",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="Directory to search when using --all_with.",
    )
    parser.add_argument(
        "--case_sensitive",
        action="store_true",
        help="If set, --all_with matching is case-sensitive.",
    )

    # Plot smoothing
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=1,
        help="Moving-average window (same-length) for smoothing sequential curves (1 = no smoothing).",
    )

    return parser.parse_args()


########################################
############# PLOTTING UTILS ###########
########################################
def _moving_average_same(arr: np.ndarray, win: int) -> np.ndarray:
    """Same-length moving average (centered, pad edges) so boundaries remain aligned."""
    if win <= 1:
        return arr
    if win % 2 == 0:
        win += 1  # prefer odd window for centered smoothing
    pad = win // 2
    pad_left = np.repeat(arr[:1], pad)
    pad_right = np.repeat(arr[-1:], pad)
    padded = np.concatenate([pad_left, arr, pad_right])
    kernel = np.ones(win) / float(win)
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed  # length preserved


def _common_experience_lengths(
    curves_per_run: List[List[List[float]]],
) -> Tuple[int, List[int]]:
    """
    Given multiple runs' per-epoch curves (list over runs -> list over experiences -> list of epochs),
    return:
      - E_common: number of experiences common to all runs (min over runs)
      - Lk: per-experience common epoch lengths (min length across runs for each experience k)
    """
    if not curves_per_run:
        return 0, []
    E_common = min(len(run) for run in curves_per_run)
    if E_common == 0:
        return 0, []
    Lk = []
    for k in range(E_common):
        Lk.append(min(len(run[k]) for run in curves_per_run))
    return E_common, Lk


def _build_seq_with_boundaries(
    exp_lists: List[List[float]], Lk: List[int]
) -> Tuple[np.ndarray, List[int]]:
    """
    Truncate each experience to Lk[k], then concatenate across experiences.
    Return sequential curve and cumulative boundaries (start indices).
    """
    seq_vals = []
    boundaries = []
    total = 0
    for k, lst in enumerate(exp_lists[: len(Lk)]):
        boundaries.append(total)
        take = Lk[k]
        arr = np.array(lst[:take], dtype=np.float32)
        seq_vals.append(arr)
        total += len(arr)
    if total == 0:
        return np.array([], dtype=np.float32), boundaries
    seq = np.concatenate(seq_vals, axis=0)
    return seq, boundaries


def _plot_compare_sequential(
    title: str,
    ylabel: str,
    outfile: str,
    runs: List[Tuple[str, np.ndarray]],  # (label, seq)
    boundaries: List[int],
    smooth: int = 1,
):
    if not runs:
        return

    plt.figure()
    for label, seq in runs:
        if seq.size == 0:
            continue
        y = _moving_average_same(seq, smooth) if smooth > 1 else seq
        x = np.arange(len(y))
        plt.plot(x, y, label=label, linewidth=1.75)

    # dashed vertical lines + labels (from boundaries computed on common truncation)
    if runs and any(seq.size > 0 for _, seq in runs):
        # pick y-limits from combined data (post-smoothing)
        all_y = np.concatenate(
            [
                _moving_average_same(seq, smooth) if smooth > 1 else seq
                for _, seq in runs
                if seq.size > 0
            ]
        )
        ymin, ymax = float(np.min(all_y)), float(np.max(all_y))
        ytop = ymax + (ymax - ymin + 1e-6) * 0.05
        vline_style = dict(color="#BBBBBB", linestyle="--", linewidth=1.0, alpha=0.6)
        label_color = "#666666"
        for i, bx in enumerate(boundaries):
            plt.axvline(bx, **vline_style)
            plt.text(
                bx,
                ytop,
                f"{i}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=label_color,
            )
        plt.ylim(0.59, 1.01)  # fixed vis range as per your previous choice

    # plt.title(title)  # optional
    plt.xlabel("Global epoch (concatenated across experiences)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend()
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, bbox_inches="tight", dpi=400)
    plt.close()


########################################
################ MAIN ##################
########################################
def main(cfg):
    # Collect files
    paths = cfg.compare_paths
    if paths is None and cfg.all_with is not None:
        needle = cfg.all_with if cfg.case_sensitive else cfg.all_with.lower()
        paths = []
        if not os.path.isdir(cfg.results_dir):
            print(f"[ERROR] results_dir not found: {cfg.results_dir}")
            return
        for fname in os.listdir(cfg.results_dir):
            if not fname.endswith(".pt"):
                continue
            hay = fname if cfg.case_sensitive else fname.lower()
            if needle in hay:
                paths.append(os.path.join(cfg.results_dir, fname))
        if not paths:
            print(
                f"[WARN] No .pt files in {cfg.results_dir} matching substring: {cfg.all_with}"
            )
            return

    if not paths:
        print(
            "[ERROR] Provide --compare_paths file1.pt file2.pt ... or --all_with SUBSTRING"
        )
        return

    os.makedirs(cfg.save_plot_dir, exist_ok=True)

    # Load runs
    loaded: List[Tuple[str, List[List[float]], List[List[float]]]] = (
        []
    )  # (label, es, ec)
    for f in paths:
        try:
            d = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[WARN] Failed to load {f}: {e}")
            continue

        run_name = d.get("run_name", None) or os.path.basename(f)
        label = RUN_LABEL_MAP.get(run_name, run_name)  # pretty label mapping

        es = d.get("epoch_avg_acc_seen", [])
        ec = d.get("epoch_acc_current", [])
        if len(es) == 0 and len(ec) == 0:
            print(f"[WARN] No per-epoch curves in {f}, skipping.")
            continue

        loaded.append((label, es, ec))

    if not loaded:
        print("[ERROR] No valid results to plot.")
        return

    # Reorder runs by fixed ORDER
    def order_key(item):
        lbl = item[0]
        return ORDER.index(lbl) if lbl in ORDER else len(ORDER)

    loaded.sort(key=order_key)

    labels = [lbl for (lbl, _, _) in loaded]
    seen_runs_raw = [es for (_, es, _) in loaded]
    curr_runs_raw = [ec for (_, _, ec) in loaded]

    # Compute common experience lengths (truncate per experience so curves align)
    E_seen, Lk_seen = (
        _common_experience_lengths(seen_runs_raw)
        if any(len(r) > 0 for r in seen_runs_raw)
        else (0, [])
    )
    E_curr, Lk_curr = (
        _common_experience_lengths(curr_runs_raw)
        if any(len(r) > 0 for r in curr_runs_raw)
        else (0, [])
    )

    # Build sequential curves and a single set of boundaries for each metric
    seen_seq_runs: List[Tuple[str, np.ndarray]] = []
    curr_seq_runs: List[Tuple[str, np.ndarray]] = []
    seen_boundaries: List[int] = []
    curr_boundaries: List[int] = []

    if E_seen > 0:
        for lbl, es in zip(labels, seen_runs_raw):
            seq, boundaries = _build_seq_with_boundaries(es, Lk_seen)
            seen_seq_runs.append((lbl, seq))
            if not seen_boundaries:
                seen_boundaries = boundaries  # same for all due to Lk
    if E_curr > 0:
        for lbl, ec in zip(labels, curr_runs_raw):
            seq, boundaries = _build_seq_with_boundaries(ec, Lk_curr)
            curr_seq_runs.append((lbl, seq))
            if not curr_boundaries:
                curr_boundaries = boundaries

    # Plot comparison figures (all methods overlaid)
    if seen_seq_runs:
        out_seen = os.path.join(cfg.save_plot_dir, "compare_sequential_seen.png")
        _plot_compare_sequential(
            title="AvgAcc(seen) — Sequential (all runs)",
            ylabel="AvgAcc(seen)",
            outfile=out_seen,
            runs=seen_seq_runs,
            boundaries=seen_boundaries,
            smooth=cfg.smooth_window,
        )
        print(f"[OK] Saved {out_seen}")
    else:
        print("[INFO] No 'seen' curves found.")

    if curr_seq_runs:
        out_curr = os.path.join(cfg.save_plot_dir, "compare_sequential_current.png")
        _plot_compare_sequential(
            title="Acc(current) — Sequential (all runs)",
            ylabel="Acc(current)",
            outfile=out_curr,
            runs=curr_seq_runs,
            boundaries=curr_boundaries,
            smooth=cfg.smooth_window,
        )
        print(f"[OK] Saved {out_curr}")
    else:
        print("[INFO] No 'current' curves found.")


if __name__ == "__main__":
    cfg = parse_args()
    if not hasattr(cfg, "save_plot_dir"):
        cfg.save_plot_dir = "./plots"
    main(cfg)
