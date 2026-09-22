import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import networkx as nx
    import numpy as np
    import csv
    import ast
    from collections import Counter
    from collections import defaultdict
    import argparse
    from matplotlib.patches import Patch
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import os
    import matplotlib
    import matplotlib.pyplot as plt
    import os, json, pickle
    import pandas as pd
    from typing import Any, Tuple
    from dataclasses import dataclass
    from typing import Optional, List, Tuple
    import community as community_louvain

    import HanziSemanticAnalyzer as HSA
except ImportError:
    print("Import Error! Some packages aren't installed so import failed."
          "List of imports: networkx, numpy, csv, ast, argparse, "
          "collections, matplotlib, os, json, pickle, pandas, dataclasses, "
          "typing, community.")

# strokes that have no semantic meaning
# these are filtered out so that graph edges represent meaningful shared parts
meaningless_strokes = {"㇙", "㇛", "㇂", "㇈", "㇗", "㇉", "㇏", "㇒", "㇀", "㇝",
                       "㇜", "㇎", "㇇", "龴", "㇆", "㇅", "?", "丨", "丿", "丶",
                       "一"}
# map for radicals from traditional to simplified
t_t_s = {
    '糸': '纟', '見': '见', '言': '讠', '貝': '贝', '足': '⻊', '車': '车',
    '長': '长', '門': '门', '韋': '韦', '頁': '页', '風': '风', '飛': '飞',
    '食': '饣', '馬': '马', '魚': '鱼', '鳥': '鸟', '麥': '麦', '黽': '黾',
    '鼠': '鼡', '齊': '齐', '齒': '齿', '龍': '龙', '龜': '龟', '金': '钅',
    '鹵': '卤', "戶": "户", "艸": "⺾", "黃": "黄"
}

# used for file-only plotting
os.environ.setdefault("MPLBACKEND", "Agg")
# force-safe fallback if something else was set
try:
    matplotlib.use(os.environ.get("MPLBACKEND", "Agg"), force=True)
except Exception:
    matplotlib.use("Agg", force=True)

# cache directories
CACHE_DIR = str(DATA_DIR / "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CSV_RESULTS = os.path.join(CACHE_DIR, "semantic_community_analysis.csv")
PARQUET_RESULTS = os.path.join(CACHE_DIR,
                               "semantic_community_analysis.parquet")
DETAILS_JSON = os.path.join(CACHE_DIR, "detailed_results.json")
PARTITION_PKL = os.path.join(CACHE_DIR, "louvain_partition.pkl")
RADICALS_PKL = os.path.join(CACHE_DIR, "char_to_radicals.pkl")
BUBBLES_JSON = os.path.join(CACHE_DIR, "top3_bubbles.json")


# ********************** DUMMY FUNCS ******************

@dataclass
class RandomBaselineResult:
    run_id: int
    communities_analyzed: int
    mean_avg_similarity: float
    mean_coherence: float
    df: pd.DataFrame  # per-community table


def _partition_to_dict(partition: dict[str, int]) -> dict[int, list[str]]:
    """
    convert {char -> community_id} to {community_id -> [chars]}
    """
    out = defaultdict(list)
    for ch, cid in partition.items():
        out[int(cid)].append(ch)
    return dict(out)


def make_random_partition_like(partition: dict[str, int],
                               seed: int | None = 42) -> dict[str, int]:
    """
    creates a random dummy partition with the SAME number of communities and
    the SAME size distribution as the real Louvain partition, but with characters
    shuffled uniformly at random across those sizes.
    """
    rng = np.random.default_rng(seed)
    # real sizes per community (keep original community ids for easy mapping)
    real = _partition_to_dict(partition)
    comm_ids = list(real.keys())
    sizes = [len(real[cid]) for cid in comm_ids]

    # pool of all characters
    all_chars = [ch for chars in real.values() for ch in chars]
    rng.shuffle(all_chars)

    # slice the shuffled pool into the original size buckets
    new_partition: dict[str, int] = {}
    idx = 0
    for cid, size in zip(comm_ids, sizes):
        slice_chars = all_chars[idx: idx + size]
        idx += size
        for ch in slice_chars:
            new_partition[ch] = int(cid)  # reuse the same ids for easier joins
    return new_partition


def communities_from_partition(partition: dict[str, int]) -> dict[
    int, list[str]]:
    """
    Convenience: {char -> cid} → {cid -> [chars]}.
    """
    return _partition_to_dict(partition)


def _analyze_one_partition_with_hanzi(
        analyzer,
        partition_dict: dict[int, list[str]],
        hanzi_meanings: dict[str, str]
) -> tuple[pd.DataFrame, list]:
    """
    Uses the existing pipeline to compute avg_similarity & coherence per community.
    Small communities (<15) are skipped inside analyzer, same as real run.
    """
    return analyzer.analyze_louvain_communities(partition_dict, hanzi_meanings)


def run_random_baseline(
        hanzi_csv_path: str,
        real_partition: dict[str, int],
        runs: int = 10,
        seed: int = 1234
) -> tuple[list[RandomBaselineResult], pd.DataFrame]:
    """
    Run N randomizations, analyze each with HanziSemanticAnalyzer,
    and return per-run summaries plus a concatenated long table.
    """
    analyzer = HanziCommunityAnalyzer()
    hanzi_meanings = analyzer.parse_hanzi_data(hanzi_csv_path)

    per_run: list[RandomBaselineResult] = []
    long_rows = []

    # deterministically vary the seed per run
    base_rng = np.random.SeedSequence(seed)
    child_seeds = base_rng.spawn(runs)

    for i in range(runs):
        s = int(
            np.random.Generator(np.random.PCG64(child_seeds[i])).integers(0,
                                                                          2 ** 31 - 1))

        rnd_partition = make_random_partition_like(real_partition, seed=s)
        rnd_comm = communities_from_partition(rnd_partition)

        df, _details = _analyze_one_partition_with_hanzi(analyzer, rnd_comm,
                                                         hanzi_meanings)
        if df.empty:
            # nothing cleared min size, record zeros for transparency
            per_run.append(RandomBaselineResult(
                run_id=i,
                communities_analyzed=0,
                mean_avg_similarity=0.0,
                mean_coherence=0.0,
                df=df
            ))
            continue

        res = RandomBaselineResult(
            run_id=i,
            communities_analyzed=len(df),
            mean_avg_similarity=float(df["avg_similarity"].mean()),
            mean_coherence=float(df["coherence_score"].mean()),
            df=df.assign(run_id=i)
        )
        per_run.append(res)
        long_rows.append(res.df)

    long_df = pd.concat(long_rows,
                        ignore_index=True) if long_rows else pd.DataFrame()
    return per_run, long_df

def compare_real_vs_random(real_df: pd.DataFrame,
                           random_long_df: pd.DataFrame) -> dict[str, Any]:
    """
    Quick numeric comparison between your real Louvain results and the random baseline.
    Returns a dict with a few headline numbers; extend as you like.
    """
    out: dict[str, Any] = {}

    if real_df is not None and not real_df.empty:
        out["real_mean_avg_similarity"] = float(
            real_df["avg_similarity"].mean())
        out["real_mean_coherence"] = float(real_df["coherence_score"].mean())
        out["real_n_communities"] = int(len(real_df))
    else:
        out["real_mean_avg_similarity"] = None
        out["real_mean_coherence"] = None
        out["real_n_communities"] = 0

    if random_long_df is not None and not random_long_df.empty:
        grp = random_long_df.groupby("run_id")
        out["random_runs"] = int(grp.ngroups)
        out["random_mean_of_run_means_avg_similarity"] = float(
            grp["avg_similarity"].mean().mean()
        )
        out["random_mean_of_run_means_coherence"] = float(
            grp["coherence_score"].mean().mean()
        )
        # optional: 5th/95th percentile envelopes across runs
        out["random_run_mean_coherence_p05"] = float(
            grp["coherence_score"].mean().quantile(0.05))
        out["random_run_mean_coherence_p95"] = float(
            grp["coherence_score"].mean().quantile(0.95))
    else:
        out["random_runs"] = 0
        out["random_mean_of_run_means_avg_similarity"] = None
        out["random_mean_of_run_means_coherence"] = None
        out["random_run_mean_coherence_p05"] = None
        out["random_run_mean_coherence_p95"] = None

    # quick delta, if both present
    if out["real_mean_coherence"] is not None and out[
        "random_mean_of_run_means_coherence"] is not None:
        out["delta_real_minus_random_coherence"] = (
                out["real_mean_coherence"] - out[
            "random_mean_of_run_means_coherence"]
        )
    else:
        out["delta_real_minus_random_coherence"] = None

    return out

def run_dummy_baseline_experiment(hanzi_csv_path: str,
                                  real_partition: dict[str, int],
                                  real_results_df: pd.DataFrame | None,
                                  out_dir: str = "cache",
                                  runs: int = 10,
                                  seed: int = 1234) -> dict[str, Any]:
    """
    taking care of all the dummy model runs
    :param hanzi_csv_path:  path to hanzi dataset
    :param real_partition: real louvain communities
    :param real_results_df: results with real louvain communities
    :param out_dir: dir for the files to be saved to
    :param runs: how many runs of dummy model?
    :param seed: seed for randomize
    :return:
    """
    os.makedirs(out_dir, exist_ok=True)

    per_run, long_df = run_random_baseline(
        hanzi_csv_path=hanzi_csv_path,
        real_partition=real_partition,
        runs=runs,
        seed=seed
    )

    rnd_csv = os.path.join(out_dir, "random_baseline_per_community.csv")
    if not long_df.empty:
        long_df.to_csv(rnd_csv, index=False, encoding="utf-8")
        print(f"[random-baseline] wrote {rnd_csv}")

    summary = compare_real_vs_random(real_results_df, long_df)
    # json
    rnd_json = os.path.join(out_dir, "random_baseline_summary.json")
    with open(rnd_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[random-baseline] wrote {rnd_json}")

    # print to screen
    print("\n=== Random-baseline vs Real (headline) ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    per_run2, long_df2 = run_random_baseline_multinomial(
        hanzi_csv_path=hanzi_csv_path,
        real_partition=real_partition,
        runs=runs,
        seed=seed,
        concentration=1.0
    )

    rnd_csv = os.path.join(out_dir,
                           "random_baseline_per_community_randomize.csv")
    if not long_df2.empty:
        long_df2.to_csv(rnd_csv, index=False, encoding="utf-8")
        print(f"[random-baseline] wrote {rnd_csv}")

    summary2 = compare_real_vs_random(real_results_df, long_df2)
    # small json summary
    rnd_json = os.path.join(out_dir, "random_baseline_summary_randomize.json")
    with open(rnd_json, "w", encoding="utf-8") as f:
        json.dump(summary2, f, ensure_ascii=False, indent=2)
    print(f"[random-baseline] wrote {rnd_json}")

    # print to screen
    print("\n=== Random-baseline-randomize vs Real (headline) ===")
    for k, v in summary2.items():
        print(f"{k}: {v}")

    return summary


def make_random_partition_multinomial_sizes(real_partition: dict[str, int],
                                            seed: int | None = 123,
                                            concentration: float = 1.0) -> \
dict[str, int]:
    """

    :param real_partition:
    :param seed:
    :param concentration:
    :return:
    """
    """
    Null #2: keep the number of communities (k) but redraw sizes at random.
    - Draw p ~ Dirichlet(alpha=concentration * 1_k)  -> adds variability
    - Draw counts ~ Multinomial(n_chars, p)
    - Assign shuffled characters to those buckets.
    NOTE: This can create many small buckets; your analyzer may drop <15.
    """
    rng = np.random.default_rng(seed)
    # real k and all chars
    real = defaultdict(list)
    for ch, cid in real_partition.items():
        real[int(cid)].append(ch)
    comm_ids = list(real.keys())
    k = len(comm_ids)
    all_chars = [ch for chars in real.values() for ch in chars]
    n = len(all_chars)
    rng.shuffle(all_chars)

    # probabilities and sizes
    alpha = np.full(k, float(concentration))
    p = rng.dirichlet(alpha)  # random proportions
    counts = rng.multinomial(n, p)  # integer bucket sizes

    # ensure no zero size buckets
    # by move one char from the largest bucket to zeros
    zeros = np.where(counts == 0)[0]
    if zeros.size:
        for z in zeros:
            largest = int(np.argmax(counts))
            if counts[largest] > 1:
                counts[largest] -= 1
                counts[z] += 1

    # slice assignment
    new_partition: dict[str, int] = {}
    idx = 0
    for cid, size in zip(comm_ids, counts):
        slice_chars = all_chars[idx: idx + size]
        idx += size
        for ch in slice_chars:
            new_partition[ch] = int(cid)
    return new_partition


def run_random_baseline_multinomial(hanzi_csv_path: str,
                                    real_partition: dict[str, int],
                                    runs: int = 100,
                                    seed: int = 2024,
                                    concentration: float = 1.0):
    analyzer = HanziCommunityAnalyzer()
    hanzi_meanings = analyzer.parse_hanzi_data(hanzi_csv_path)

    per_run, long_rows = [], []
    for i in range(runs):
        s = seed + i
        rnd_partition = make_random_partition_multinomial_sizes(
            real_partition, seed=s, concentration=concentration)
        rnd_comm = communities_from_partition(rnd_partition)
        df, _details = _analyze_one_partition_with_hanzi(analyzer, rnd_comm,
                                                         hanzi_meanings)
        if df.empty:
            per_run.append(RandomBaselineResult(i, 0, 0.0, 0.0, df))
            continue
        res = RandomBaselineResult(
            run_id=i,
            communities_analyzed=len(df),
            mean_avg_similarity=float(df["avg_similarity"].mean()),
            mean_coherence=float(df["coherence_score"].mean()),
            df=df.assign(run_id=i, null_model="multinomial")
        )
        per_run.append(res);
        long_rows.append(res.df)
    long_df = pd.concat(long_rows, ignore_index=True) \
        if long_rows else pd.DataFrame()
    return per_run, long_df

# ***************DUMMY FUNCS ****************

# ********** CACHE FUNCS *********

def _convert_for_json(obj):
    """recursively convert numpy types to python types"""
    if isinstance(obj, dict):
        return {k: _convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        return obj


def save_results(results_df: pd.DataFrame,
                 detailed_results: list,
                 partition: dict = None,
                 char_to_radicals: dict = None) -> None:
    """
    save results to cache for later
    :param results_df: result data frame
    :param detailed_results: list of results
    :param partition: louvainpartition
    :param char_to_radicals:  dictionary of chars and radicals
    :return:
    """
    results_df.to_csv(CSV_RESULTS, index=False, encoding="utf-8")
    try:
        results_df.to_parquet(PARQUET_RESULTS, index=False)
    except Exception:
        pass

    # convert before json dump
    safe_details = _convert_for_json(detailed_results)
    with open(DETAILS_JSON, "w", encoding="utf-8") as f:
        json.dump(safe_details, f, ensure_ascii=False, indent=2)

    # partition + rad as pickle (no problem with numpy here)
    if partition is not None:
        with open(PARTITION_PKL, "wb") as f:
            pickle.dump(partition, f)
    if char_to_radicals is not None:
        with open(RADICALS_PKL, "wb") as f:
            pickle.dump(char_to_radicals, f)


def load_results(prefer_parquet: bool = True
                 ) -> Tuple[pd.DataFrame, list, dict | None, dict | None]:
    """
    load cache results
    :param prefer_parquet: true if prefer parquet
    :return: results_df, detailed_results, partition, char_to_radicals
    """
    if prefer_parquet and os.path.exists(PARQUET_RESULTS):
        results_df = pd.read_parquet(PARQUET_RESULTS)
    elif os.path.exists(CSV_RESULTS):
        results_df = pd.read_csv(CSV_RESULTS)
    else:
        raise FileNotFoundError("No cached results found")

    # load detailed results
    detailed_results = []
    if os.path.exists(DETAILS_JSON):
        with open(DETAILS_JSON, "r", encoding="utf-8") as f:
            detailed_results = json.load(f)

    # load partition and radicals if available
    partition = None
    if os.path.exists(PARTITION_PKL):
        with open(PARTITION_PKL, "rb") as f:
            partition = pickle.load(f)

    char_to_radicals = None
    if os.path.exists(RADICALS_PKL):
        with open(RADICALS_PKL, "rb") as f:
            char_to_radicals = pickle.load(f)

    return results_df, detailed_results, partition, char_to_radicals


def cache_exists() -> bool:
    return os.path.exists(CSV_RESULTS) or os.path.exists(PARQUET_RESULTS)


def export_cluster_radicals_csv(
        analyses_by_cid: dict[int, dict],
        out_csv_counts: str = os.path.join(CACHE_DIR,
                                           "cluster_radicals_counts.csv"),
        out_csv_percent: str = os.path.join(CACHE_DIR,
                                            "cluster_radicals_percent.csv"),
        *,
        disambiguate_groups_with_cid: bool = False,
) -> tuple[str, str]:
    """
    write 2 csv with columns group (with community id), subgroup (rad),
    count (count or percantage)
    :param analyses_by_cid: analysis
    :param out_csv_counts: csv path
    :param out_csv_percent: csv path
    :param disambiguate_groups_with_cid: if true keep c-id in group
    :return:
    """

    os.makedirs(os.path.dirname(out_csv_counts), exist_ok=True)
    rows_count = [("Group", "Subgroup", "Count")]
    rows_pct = [("Group", "Subgroup", "Count")]

    for cid, analysis in analyses_by_cid.items():
        for detail in analysis.get("semantic_cluster_details", []):
            theme = detail.get("theme") or ", ".join(detail.get("theme_terms",
                                                                [])[:3])
            size = int(detail.get("size", 0)) or 1
            group_name = f"{theme}" if disambiguate_groups_with_cid else theme

            # prefer full list if available, else fall back
            # to top_radicals from analyzer
            rad_info = detail.get("radical_analysis") or {}
            top_rads = rad_info.get("top_radicals") or []
            #
            # radicals_full = None
            # if use_full_if_available:
            #     # If we've already computed full counts somewhere else and stashed them,
            #     # you could read them here; otherwise we stay with top_rads.
            #     # (If you want true 'full', pass char_to_radicals to build_top3_bubble_cache above.)
            #     pass

            iterable = top_rads  # radicals_full if radicals_full else
            for rad, cnt in iterable:
                rows_count.append((group_name, rad, int(cnt)))
                rows_pct.append(
                    (group_name, rad, round(100.0 * cnt / size, 2)))

    with open(out_csv_counts, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows_count)

    with open(out_csv_percent, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows_pct)

    print(f"wrote Canva CSV (counts)   → {out_csv_counts}")
    print(f"wrote Canva CSV (percent%) → {out_csv_percent}")
    return os.path.abspath(out_csv_counts), os.path.abspath(out_csv_percent)


# *********** END OF CACHE RELATED FUNCS *********


# *********** PLOTS FUNCS **************

# helper: confidence band using residual std around the fitted line
def _add_conf_band(ax, xs, ys, xfit, y, m, b, is_log=False):
    """
    xs, ys are the line coordinates already computed for drawing.
    xfit is the x used for fitting residuals (x or log10(x)).
    """
    residuals = y - (m * xfit + b)
    s = residuals.std(ddof=2) if len(residuals) > 2 else 0.0
    # lighter band per your request (very transparent)
    ax.fill_between(xs, ys - s, ys + s, color="gray", alpha=0.08,
                    label="±1 std dev")


def plot_coherence_variants(results_df: pd.DataFrame,
                            out_dir: str = str(ASSETS_DIR / "scatterplots")):
    """
    generate 4 versions of the community size vs coherence scatterplots
    1) log + confidence band
    2) only log
    3) only confidence band (no log)
    4) only correlation coefficient r (instead of slope)
    :param results_df: results dataframe
    :param out_dir: "scatterplots" dir
    :return:
    """
    os.makedirs(out_dir, exist_ok=True)

    # basic sanity -
    df = results_df.copy()
    if df.empty:
        raise ValueError("results_df is empty")
    # drop NaNs / inf just in case
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["num_characters", "coherence_score"]
    )

    x = df["num_characters"].values.astype(float)
    y = df["coherence_score"].values.astype(float)

    # LOG + CONFIDENCE BAND
    # filter out non-positive for log x
    mask_pos = x > 0
    x_pos, y_pos = x[mask_pos], y[mask_pos]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(x_pos, y_pos, alpha=0.7, edgecolor="none")
    ax.set_xscale("log")
    ax.set_xlabel("Community size (log scale)")
    ax.set_ylabel("Semantic coherence")
    ax.set_title("Community size and Semantic Coherence correlation ("
                 "confidence band)")

    if len(x_pos) >= 2:
        # set exact axis limits first, then span the line across them
        xmin, xmax = x_pos.min(), x_pos.max()
        ax.set_xlim(xmin, xmax * 1.1)  # in log plots, cannot start at 0
        ax.set_ylim(bottom=0.1)
        # fit in log-space
        xlog = np.log10(x_pos)
        m, b = np.polyfit(xlog, y_pos, 1)
        xs = np.logspace(np.log10(xmin), np.log10(xmax), 200)
        ys = m * np.log10(xs) + b
        ax.plot(xs, ys, linewidth=2, color="navy")
        _add_conf_band(ax, xs, ys, xlog, y_pos, m, b, is_log=True)
    plt.suptitle(
        f"Trend slope ≈ {m:.10f} (slightly negative -> larger "
        f"communities are a bit less coherent)",
        fontsize=10, y=0.98, color="#555555")
    plt.tight_layout()
    f1 = os.path.join(out_dir, "log_confband.png")
    plt.savefig(f1, dpi=200)
    plt.close()

    # 2) ONLY LOG
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(x_pos, y_pos, alpha=0.7, edgecolor="none")
    ax.set_xscale("log")
    ax.set_xlabel("Community size (log scale)")
    ax.set_ylabel("Semantic coherence")
    ax.set_title("Community size and Semantic Coherence correlation ("
                 "confidence band)")

    if len(x_pos) >= 2:
        xmin, xmax = x_pos.min(), x_pos.max()
        ax.set_xlim(xmin, xmax * 1.1)
        ax.set_ylim(bottom=0.1)
        xlog = np.log10(x_pos)
        m, b = np.polyfit(xlog, y_pos, 1)
        xs = np.logspace(np.log10(xmin), np.log10(xmax), 200)
        ys = m * np.log10(xs) + b
        ax.plot(xs, ys, linewidth=2, color="navy")

    plt.suptitle(
        f"Trend slope ≈ {m:.10f} (slightly negative ⇒ larger communities "
        f"are a bit less coherent)",
        fontsize=10, y=0.98, color="#555555")
    plt.tight_layout()
    f2 = os.path.join(out_dir, "log_only.png")
    plt.savefig(f2, dpi=200)
    plt.close()

    # ONLY CONFIDENCE BAND (NO LOG)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(x, y, alpha=0.7, edgecolor="none")
    # make x-y-axis start at 0 and match the line to axis length
    xmin, xmax = 0.0, float(np.nanmax(x)) * 1.03
    ymin, ymax = 0.1, float(np.nanmax(y)) * 1.03
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Community size (# characters)")
    ax.set_ylabel("Semantic coherence")
    ax.set_title("Community size and Semantic Coherence correlation ("
                 "confidence band)")

    if len(x) >= 2:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(xmin, xmax, 200)  # spans full axis
        ys = m * xs + b
        ax.plot(xs, ys, linewidth=2, color="navy")
        _add_conf_band(ax, xs, ys, x, y, m, b, is_log=False)

    plt.suptitle(
        f"Trend slope ≈ {m:.10f} (slightly negative ⇒ larger communities "
        f"are a bit less coherent)",
        fontsize=9, y=0.98, color="#555555")
    plt.tight_layout()
    f3 = os.path.join(out_dir, "confband_only.png")
    plt.savefig(f3, dpi=200)
    plt.close()

    # ONLY CORRELATION (R) INSTEAD OF SLOPE
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(x, y, alpha=0.7, edgecolor="none")
    xmin, xmax = 0.0, float(np.nanmax(x)) * 1.03
    ymin, ymax = 0.1, float(np.nanmax(y)) * 1.03
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Community size (# characters)")
    ax.set_ylabel("Semantic coherence")
    ax.set_title("Community size and Semantic Coherence correlation ("
                 "confidence band)")

    if len(x) >= 2:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(xmin, xmax, 200)
        ys = m * xs + b
        ax.plot(xs, ys, linewidth=2, color="navy")
        r = np.corrcoef(x, y)[0, 1]
        ax.text(0.02, 0.98, f"Correlation r = {r:.3f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    plt.suptitle(
        f"Trend slope ≈ {m:.10f} (slightly negative ⇒ larger " f"communities "
        f"are a bit less coherent)",
        fontsize=9, y=0.98, color="#555555")

    plt.tight_layout()
    f4 = os.path.join(out_dir, "correlation_only.png")
    plt.savefig(f4, dpi=200)
    plt.close()

    return {"log+conf": f1, "log_only": f2, "conf_only": f3, "correlation": f4}

def plot_all_similarity_bars(results_df: pd.DataFrame,
                             save_path: str = str(ASSETS_DIR / "all_similarity_bars.png")):
    """
    horizontal bar chart of all communities similarity score
    :param results_df: results dataframe
    :param save_path: path to save the plot
    :return: path to plot
    """
    df = results_df.copy()
    if df.empty:
        raise ValueError("results_df is empty")

    # sort all communities ascending by similarity
    df = df.sort_values("avg_similarity", ascending=True)

    sim = df["avg_similarity"].values
    sizes = df["num_characters"].values
    ids = [f"{int(cid)}" for cid in df["community_id"]]

    # colormap for community size
    cmap = plt.cm.viridis
    norm = plt.Normalize(sizes.min(), sizes.max())
    colors = cmap(norm(sizes))

    # figure height scaled to number of communities
    fig_h = max(5, 0.25 * len(df))
    fig, ax = plt.subplots(figsize=(9, fig_h))

    bars = ax.barh(ids, sim, color=colors)

    # x-axis: start at 0, extend ~8% beyond max
    max_val = float(np.nanmax(sim)) if len(sim) else 1.0
    ax.set_xlim(left=0, right=max_val * 1.08)

    # add labels: size inside, score at tip
    for bar, n, val in zip(bars, sizes, sim):
        y_mid = bar.get_y() + bar.get_height() / 2.0
        w = bar.get_width()
        inside_color = "white" if w > (0.55 * max_val) else "black"
        ax.text(w * 0.5, y_mid, f"n={int(n)}",
                va="center", ha="center", fontsize=8,
                fontweight="bold", color=inside_color)
        ax.text(w + max_val * 0.006, y_mid, f"{val:.3f}",
                va="center", ha="left", fontsize=8,
                fontweight="bold")

    ax.set_xlabel("Average similarity score", fontweight="bold")
    ax.set_ylabel("Community id", fontweight="bold")
    ax.set_title("All communities by average similarity", fontweight="bold")

    # add colorbar for sizes
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Community size (n)", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return os.path.abspath(save_path)

def plot_all_coherence_bars(results_df: pd.DataFrame,
                             save_path: str = str(ASSETS_DIR / "all_coherence_bars.png")):
    """
    horizontal bar chart of all communities coherence scores
    :param results_df: results dataframe
    :param save_path: path to save the plot
    :return: path to plot
    """
    df = results_df.copy()
    if df.empty:
        raise ValueError("results_df is empty")

    # sort all communities ascending by coherence
    df = df.sort_values("coherence_score", ascending=True)

    coh = df["coherence_score"].values
    sizes = df["num_characters"].values
    ids = [f"{int(cid)}" for cid in df["community_id"]]

    # colormap for community size
    cmap = plt.cm.viridis
    norm = plt.Normalize(sizes.min(), sizes.max())
    colors = cmap(norm(sizes))

    # figure height scaled to number of communities
    fig_h = max(5, 0.25 * len(df))
    fig, ax = plt.subplots(figsize=(9, fig_h))

    bars = ax.barh(ids, coh, color=colors)

    # x-axis: start at 0, extend ~8% beyond max
    max_val = float(np.nanmax(coh)) if len(coh) else 1.0
    ax.set_xlim(left=0, right=max_val * 1.08)

    # add labels: size inside, score at tip
    for bar, n, val in zip(bars, sizes, coh):
        y_mid = bar.get_y() + bar.get_height() / 2.0
        w = bar.get_width()
        inside_color = "white" if w > (0.55 * max_val) else "black"
        ax.text(w * 0.5, y_mid, f"n={int(n)}",
                va="center", ha="center", fontsize=8,
                fontweight="bold", color=inside_color)
        ax.text(w + max_val * 0.006, y_mid, f"{val:.3f}",
                va="center", ha="left", fontsize=8,
                fontweight="bold")

    ax.set_xlabel("Average coherence score", fontweight="bold")
    ax.set_ylabel("Community id", fontweight="bold")
    ax.set_title("All communities by average coherence", fontweight="bold")

    # # add colorbar for sizes
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Community size (n)", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return os.path.abspath(save_path)

def make_eval_figures(results_df: pd.DataFrame,
                      out_dir: str = "eval_figs",
                      top_n: int = 10,
                      label_top_k: int = 3):
    """
    Convenience wrapper: makes both figures and returns their paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    p1 = plot_coherence_variants(results_df, out_dir="eval_figs_variants")
    # p2 = plot_top_coherent_compact_binned(results_df, top_n=10, bins=5)
    p3 = plot_all_similarity_bars(results_df)
    p4 = plot_all_coherence_bars(results_df)

    # return {"scatter": p1, "top_bars": p2}

# ****** END PLOTS FUNCS ********



class HanziCommunityAnalyzer:
    """functions to analyse the communities"""

    def __init__(self):
        """
        encapsulates the external semantic analyzer so callers don't need to
        manage it directly
        """
        self.semantic_analyzer = HSA.HanziSemanticAnalyzer()

    @staticmethod
    def parse_hanzi_data(csv_file_path):
        """
        parses hanzi csv data into {character -> english meaning}.
        :param csv_file_path: expected csv columns (at least):
        - 0: character (single hanzi)
        - 3: english meaning (assumed per current dataset)
        :return: dict mapping character to its meaning string
        """

        hanzi_data = {}
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                # only accept rows with the expected columns
                if len(row) >= 4:
                    char = row[0]
                    meaning = row[3]  # english meaning is in 4th column
                    hanzi_data[char] = meaning
        return hanzi_data

    @staticmethod
    def analyze_communities(partition):
        """
        print size stats of discovered communities
        :param partition: dict {node -> community_id} from louvain
        :return:
        """
        if not partition:
            return

        community_sizes = defaultdict(int)
        for node, community in partition.items():
            community_sizes[community] += 1

        print(f"\nCommunity Analysis:")
        print(f"Number of communities: {len(community_sizes)}")
        print(f"Largest community: {max(community_sizes.values())} characters")
        print(
            f"Smallest community: {min(community_sizes.values())} characters")
        print(
            f"Average community size: "
            f"{np.mean(list(community_sizes.values())):.2f}")

    def analyze_louvain_communities(self, communities_dict, hanzi_meanings):
        """
        compute semantic coherence for each louvain community
        :param communities_dict: dict of {community_id -> list[str chars]}
        :param hanzi_meanings: dict of {char -> english meaning}
        :return:
        results_df: pandas dataframe
        results: list of detailed dicts per community with coherence+sim
        """
        # convert character communities to meaning communities
        meaning_communities = {}
        character_to_meaning = {}

        for community_id, characters in communities_dict.items():
            # skips communities smaller than 15 to keep scores stable
            if len(characters) < 15:
                continue
            meanings = []
            chars_with_meanings = []

            for char in characters:
                # only keep characters that exist in our meaning map
                if char in hanzi_meanings:
                    meaning = hanzi_meanings[char]
                    meanings.append(meaning)
                    chars_with_meanings.append(char)
                    character_to_meaning[char] = meaning

            if meanings:  # only include communities with valid meanings
                meaning_communities[community_id] = {
                    'meanings'  : meanings,
                    'characters': chars_with_meanings
                }

        # analyze semantic coherence
        results = []
        for community_id, data in meaning_communities.items():
            meanings = data['meanings']
            characters = data['characters']

            # analyzer returns avg_similarity and coherence_score
            analysis = \
                self.semantic_analyzer.analyze_community_semantic_coherence(
                    meanings)

            results.append({
                'community_id'   : community_id,
                'characters'     : characters,
                'meanings'       : meanings,
                'num_characters' : len(meanings),
                'avg_similarity' : analysis['avg_similarity'],
                'coherence_score': analysis['coherence_score']
            })

        # sort by coherence score and
        # creates a dataframe with summary scores per community
        results_df = pd.DataFrame([{
            'community_id'   : r['community_id'],
            'num_characters' : r['num_characters'],
            'avg_similarity' : r['avg_similarity'],
            'coherence_score': r['coherence_score']
        } for r in results]).sort_values('coherence_score', ascending=False)

        return results_df, results

    def detailed_community_analysis(self, community_data,
                                    char_to_radicals=None):
        """
        analyzes the semantics of the clusters and
        print a deeper look into one community, with/without radicals
        this is meant mainly for on screen inspection
        :param community_data:
        :param char_to_radicals:
        :return: analysis
        """
        meanings = community_data['meanings']
        characters = community_data['characters']

        print(f"\n=== Community "
              f"{community_data.get('community_id', 'Unknown')} Analysis ===")
        print(f"Number of characters: {len(characters)}")

        # community-wide radical analysis
        if char_to_radicals:
            radical_counts = Counter()
            for char in characters:
                if char in char_to_radicals:
                    radicals = char_to_radicals[char]
                    radical_counts.update(radicals)

            if radical_counts:
                print(f"\nCommunity-Wide Radical Analysis:")
                top_radicals = radical_counts.most_common(5)
                for radical, count in top_radicals:
                    percentage = (count / len(characters)) * 100
                    print(
                        f"  {radical}: {count} characters ({percentage:.1f}%)")

        # enhanced semantic analysis with optional radical tracking
        analysis = \
            self.semantic_analyzer.analyze_community_semantic_coherence_with_radicals(
                meanings, characters, char_to_radicals)

        print(f"\nSemantic Analysis:")
        print(f"  Average similarity: {analysis['avg_similarity']:.3f}")
        print(f"  Coherence score: {analysis['coherence_score']:.3f}")
        print(f"  Semantic clusters: {analysis['semantic_clusters']}")
        print(f"  Dominant themes: {'#'.join(analysis['dominant_themes'])}")

        # optional: per-cluster radical summaries (only for clusters sized ≥ 3)
        if 'semantic_cluster_details' in analysis and char_to_radicals:
            print(f"\nSemantic Cluster Breakdown:")
            for detail in analysis['semantic_cluster_details']:
                # only 3+ characters
                if detail['size'] >= 3:
                    print(f"\n  🎯 Cluster: '{detail['theme']}' "
                          f"({detail['size']} characters)")

                    radical_info = detail['radical_analysis']
                    if radical_info['top_radicals']:
                        print(f"     Dominant radicals:")
                        for radical, count in radical_info[
                                                  'top_radicals'][:3]:
                            percentage = (count / detail['size']) * 100
                            print(f"       {radical}: {count} chars "
                                  f"({percentage:.1f}%)")

                        # show if high radical concentration or not
                        if radical_info['dominant_radical_percentage'] > 60:
                            print(f"     → High radical coherence: "
                                  f"{radical_info['dominant_radical_percentage']:.1f}"
                                  f"% share same radical")

                    # sample a few chars for quick look
                    if detail['size'] < 15:
                        sample_chars = detail['characters']
                    else:
                        sample_chars = detail['characters'][:10]
                    if sample_chars:
                        print(
                            f"     Sample chars: {' '.join(sample_chars)}")

        # when small community - also show pairwise similarities
        if len(meanings) <= 10:
            print("\nPairwise similarities:")
            sim_matrix = analysis['similarity_matrix']
            for i, meaning1 in enumerate(meanings):
                for j, meaning2 in enumerate(meanings[i + 1:], i + 1):
                    print(f"  '{meaning1}' <-> '{meaning2}': "
                          f"{sim_matrix[i][j]:.3f}")

        return analysis



def parse_radicals(radicals_str):
    """
    parse the all_radicals column which contains list literals as strings.

    :param radicals_str: example: "['艹', '氵', '匚', '木']"
    :return: list of str, of radicals (empty list if missing/invalid data)
    """
    if pd.isna(radicals_str):
        return []

    try:
        # the column contains string representation of python lists
        radicals_list = ast.literal_eval(str(radicals_str).strip())

        # ensure it's a list and filter out empties
        if isinstance(radicals_list, list):
            return [r for r in radicals_list if r and str(r).strip()]
        else:
            return []
    except (ValueError, SyntaxError) as e:
        # keep running but warn, corrupted cells shouldn't crash a run...
        print(f"Warning: Could not parse radicals '{radicals_str}': {e}")
        return []


class HanziRadicalGraph:
    """
    build and analyze a radical-sharing graph over hanzi characters

    expected columns: char, total_strokes, radical, meaning, category,
                         components, base_components, all_radicals
    """

    def __init__(self, csv_file_path):
        """
        initialize the graph builder with a CSV file containing Hanzi data
        and initialize other parameters
        :param csv_file_path: CSV file containing Hanzi data
        """
        self.df = pd.read_csv(csv_file_path)
        self.graph = nx.Graph()
        self.radical_to_chars = defaultdict(set)
        self.char_to_radicals = {}

    def analyze_community_radicals(self, partition, top_k=5):
        """
        for each community, find the most common radicals among its characters
        :param partition: communities (louvain)
        :param top_k: initialized as 5 if not asked otherwise
        :return: result: size, top radicals, sample chars
        """
        community_to_chars = defaultdict(list)
        for char, comm in partition.items():
            community_to_chars[comm].append(char)

        results = {}
        for comm_id, chars in community_to_chars.items():
            radical_counts = Counter()
            for char in chars:
                radicals = self.char_to_radicals.get(char, [])
                radical_counts.update(radicals)

            most_common = radical_counts.most_common(top_k)
            results[comm_id] = {
                "size"        : len(chars),
                "top_radicals": most_common,
                "sample_chars": chars[:10]
            }

        return results

    def build_radical_mappings(self):
        """
        build per-character radical sets and reverse maps

        it also:
        - ignores characters that are themselves meaningless strokes
        - filters out meaningless radicals
        - normalizes traditional radicals using t_t_s
        - fills both char_to_radicals and radical_to_chars
        :return:
        """
        print("Building radical mappings...")

        for idx, row in self.df.iterrows():
            char = row['char']
            radicals = parse_radicals(row['all_radicals'])

            if radicals:  # only process characters with radicals
                self.char_to_radicals[char] = set()

                # only process characters that are meaningful
                if char not in meaningless_strokes:
                    # use fixed meaningful radicals only
                    for radical in radicals:
                        if radical not in meaningless_strokes:
                            if radical in t_t_s:
                                radical = t_t_s[radical]
                            self.char_to_radicals[char].add(radical)
                            self.radical_to_chars[radical].add(char)

        print(f"Processed {len(self.char_to_radicals)} characters")
        print(f"Found {len(self.radical_to_chars)} unique radicals")

    def build_graph(self, min_shared_radicals=1, require_same_strokes=False):
        """
        add nodes for characters and edges for shared radicals.
        :param min_shared_radicals: minimum size of the intersection between
        two characters' meaningful radical sets for an edge to exist
        :param require_same_strokes: when true, only connect characters whose
        stroke counts are exactly equal; this helps reduce noisy links
        :return: networkx graph object held by the instance (also stored on self)
        """

        print(
            f"Building graph with minimum {min_shared_radicals} "
            f"shared radical(s)...")
        if require_same_strokes:
            print("Also requiring same stroke count for connections.")

        # create a mapping for quick stroke count lookup
        char_to_strokes = {}
        for idx, row in self.df.iterrows():
            char = row['char']
            if char in self.char_to_radicals:
                strokes = int(row.get('total_strokes', 0))
                char_to_strokes[char] = strokes

        # add all characters as nodes with their attributes
        for char in self.char_to_radicals.keys():
            # find the row data for this character
            char_data = self.df[self.df['char'] == char]
            if not char_data.empty:
                row = char_data.iloc[0]
                # create ascii label for better visibility in gephi
                ascii_label = f"U+{ord(char):04X}"  # unicode code point
                meaning = str(row.get('meaning', '')).strip()
                if meaning:
                    ascii_label += \
                        f"_{meaning.split(',')[0].replace(' ', '_')[:10]} "

                # add node attributes that will be useful in gephi
                self.graph.add_node(char,
                                    char=char,
                                    Label=char,  # gephi default label
                                    ascii_label=ascii_label,
                                    display_name=f"{char} ({ascii_label})",
                                    total_strokes=row.get('total_strokes', ''),
                                    radical=row.get('radical', ''),
                                    meaning=row.get('meaning', ''),
                                    category=row.get('category', ''),
                                    radicals_list=','.join(
                                        self.char_to_radicals[char]),
                                    radical_count=len(
                                        self.char_to_radicals[char]))

        # add edges between characters that share radicals
        characters = list(self.char_to_radicals.keys())
        edges_added = 0
        skipped_different_strokes = 0

        for i, char1 in enumerate(characters):
            for char2 in characters[i + 1:]:
                radicals1 = self.char_to_radicals[char1]
                radicals2 = self.char_to_radicals[char2]

                shared_radicals = radicals1.intersection(radicals2)
                meaningful_shared_radicals = set()
                for rad in shared_radicals:
                    if rad not in meaningless_strokes:
                        meaningful_shared_radicals.add(rad)

                if len(meaningful_shared_radicals) >= min_shared_radicals:
                    # check stroke count requirement if enabled
                    if require_same_strokes:
                        strokes1 = char_to_strokes.get(char1, 0)
                        strokes2 = char_to_strokes.get(char2, 0)
                        if not (strokes2 == strokes1):
                            skipped_different_strokes += 1
                            continue

                    # add edge with attributes
                    self.graph.add_edge(char1, char2,
                                        weight=len(meaningful_shared_radicals),
                                        shared_radicals=','.join(
                                            meaningful_shared_radicals),
                                        shared_count=len(
                                            meaningful_shared_radicals))
                    edges_added += 1

        print(f"Graph built with {self.graph.number_of_nodes()} "
              f"nodes and {self.graph.number_of_edges()} edges")
        if require_same_strokes and skipped_different_strokes > 0:
            print(f"Skipped {skipped_different_strokes} potential edges due "
                  f"to different stroke counts")
        return self.graph

    def build_graph_stroke_groups(self, min_shared_radicals=1,
                                  require_same_strokes=False):
        """
        build the graph where characters are connected if they share radicals
        :param min_shared_radicals: minimum number of shared radicals to
        create an edge
        :param require_same_strokes: if True, only connect characters with
        the same stroke count group
        :return: the graph
        """
        print(f"Building graph with minimum {min_shared_radicals} "
              f"shared radical(s)...")
        if require_same_strokes:
            print("Also requiring same stroke count for connections.")

        # create a mapping for quick stroke count lookup
        char_to_strokes = {}
        for idx, row in self.df.iterrows():
            char = row['char']
            if char in self.char_to_radicals:
                strokes = int(row.get('total_strokes', 0))
                char_to_strokes[char] = strokes

        # add all characters as nodes with their attributes
        for char in self.char_to_radicals.keys():
            # find the row data for this character
            char_data = self.df[self.df['char'] == char]
            if not char_data.empty:
                row = char_data.iloc[0]
                # create ascii label for better visibility in gephi
                ascii_label = f"U+{ord(char):04X}"  # unicode code point
                meaning = str(row.get('meaning', '')).strip()
                if meaning:
                    ascii_label += \
                        f"_{meaning.split(',')[0].replace(' ', '_')[:10]} "

                # add node attributes that will be useful in gephi
                self.graph.add_node(char,
                                    char=char,
                                    Label=char,  # for gephi's default label
                                    ascii_label=ascii_label,
                                    display_name=f"{char} ({ascii_label})",
                                    total_strokes=row.get('total_strokes', ''),
                                    radical=row.get('radical', ''),
                                    meaning=row.get('meaning', ''),
                                    category=row.get('category', ''),
                                    radicals_list=','.join(
                                        self.char_to_radicals[char]),
                                    radical_count=len(
                                        self.char_to_radicals[char]))

        # add edges between characters that share radicals
        characters = list(self.char_to_radicals.keys())
        edges_added = 0
        skipped_different_strokes = 0

        for i, char1 in enumerate(characters):
            for char2 in characters[i + 1:]:
                radicals1 = self.char_to_radicals[char1]
                radicals2 = self.char_to_radicals[char2]

                shared_radicals = radicals1.intersection(radicals2)
                meaningful_shared_radicals = set()
                for rad in shared_radicals:
                    if rad not in meaningless_strokes:
                        meaningful_shared_radicals.add(rad)

                if len(meaningful_shared_radicals) >= min_shared_radicals:
                    # check stroke count requirement if enabled
                    if require_same_strokes:
                        strokes1 = char_to_strokes.get(char1, 0)
                        strokes2 = char_to_strokes.get(char2, 0)
                        if (strokes1 < 7 and strokes2 < 7) \
                                or (strokes1 in range(7, 10) and
                                    strokes2 in range(7, 10)) \
                                or (strokes1 in range(11, 13) and
                                    strokes2 in range(11, 13)) \
                                or (strokes1 in range(14, 17) and
                                    strokes2 in range(14, 17)) \
                                or (strokes1 > 17 and strokes2 > 17):
                            # Add edge with attributes
                            self.graph.add_edge(char1, char2,
                                                weight=len(
                                                    meaningful_shared_radicals),
                                                shared_radicals=','.join(
                                                    meaningful_shared_radicals),
                                                shared_count=len(
                                                    meaningful_shared_radicals))
                            edges_added += 1
                        else:
                            skipped_different_strokes += 1

        print(f"Graph built with {self.graph.number_of_nodes()} "
              f"nodes and {self.graph.number_of_edges()} edges")
        if require_same_strokes and skipped_different_strokes > 0:
            print(f"Skipped {skipped_different_strokes} "
                  f"potential edges due to different stroke counts")
        return self.graph

    def export_to_gephi(self, filename="hanzi_radical_graph",
                        format="graphml"):
        """
        export the graph to a format that Gephi can read
        :param filename: base filename (without extension)
        :param format: export format ('graphml', 'gml')
        :return:
        """
        if self.graph.number_of_nodes() == 0:
            print("Graph is empty. Build the graph first using build_graph().")
            return

        output_file = f"{filename}.{format}"

        try:
            if format == "graphml":
                # GraphML is the most reliable format for Gephi
                nx.write_graphml(self.graph, output_file, encoding='utf-8')
            elif format == "gml":
                nx.write_gml(self.graph, output_file)
            else:
                print(
                    f"Unsupported format: {format}. Use 'graphml', or 'gml'")
                return

            print(f"Graph exported to {output_file} for Gephi")
            print(f"File contains {self.graph.number_of_nodes()} "
                  f"nodes and {self.graph.number_of_edges()} edges")

            # print some info about node attributes
            if self.graph.nodes():
                sample_node = list(self.graph.nodes())[0]
                node_attrs = list(self.graph.nodes[sample_node].keys())
                print(f"Node attributes: {node_attrs}")

                if self.graph.edges():
                    sample_edge = list(self.graph.edges())[0]
                    edge_attrs = list(self.graph.edges[sample_edge].keys())
                    print(f"Edge attributes: {edge_attrs}")
        except:
            print("export to gephi failed")

    def add_community_data_to_graph(self, partition,
                                    community_name="community"):
        """
        Add community information as node attributes in the graph.
        This is used for coloring nodes by community in Gephi.
        :param partition: dict mapping node -> community ID
        :param community_name: name for the community attribute
        :return:
        """

        if not partition:
            print("No partition data provided.")
            return

        for node in self.graph.nodes():
            if node in partition:
                self.graph.nodes[node][community_name] = partition[node]
            else:
                self.graph.nodes[node][community_name] = -1  # unassigned

        print(f"Added {community_name} data to {len(partition)} nodes")

    def get_graph_statistics(self):
        """Get basic statistics about the graph."""
        stats = {
            'nodes'                 : self.graph.number_of_nodes(),
            'edges'                 : self.graph.number_of_edges(),
            'density'               : nx.density(self.graph),
            'connected_components'  : nx.number_connected_components(
                self.graph),
            'largest_component_size': len(
                max(nx.connected_components(self.graph), key=len)),
        }

        # Degree statistics
        degrees = [d for n, d in self.graph.degree()]
        if degrees:
            stats['avg_degree'] = np.mean(degrees)
            stats['max_degree'] = max(degrees)
            stats['min_degree'] = min(degrees)

        return stats

    def find_radical_communities_resolution(self, resolution=1.0, min_size=10):
        """
        find communities in the graph using Louvain, with options to:
          - adjust resolution
          - filter/merge very small communities
        """
        print(f"Finding communities using Louvain method...")

        # run Louvain
        partition = community_louvain.best_partition(self.graph,
                                                     resolution=resolution)

        # separate communities
        comm_to_nodes = defaultdict(list)
        for node, comm in partition.items():
            comm_to_nodes[comm].append(node)

        # isolate nodes (degree = 0)
        isolates = [n for n in self.graph.nodes if self.graph.degree(n) == 0]

        # filter small communities
        if min_size > 1:
            for comm, nodes in list(comm_to_nodes.items()):
                if len(nodes) < min_size:
                    del comm_to_nodes[comm]
        return partition, isolates


def parse_args():
    p = argparse.ArgumentParser(description="Hanzi communities pipeline")
    p.add_argument("--use-cache", action="store_true",
                   help="Skip heavy recompute and load cached results.")
    p.add_argument("--recompute", action="store_true",
                   help="Force recompute and overwrite cache.")
    p.add_argument("--plot-only", action="store_true",
                   help="Load cache and only produce plots (no prints).")
    p.add_argument("--out-dir", default="eval_figs",
                   help="Directory for plots.")
    p.add_argument("--top-n", type=int, default=10,
                   help="Top-N communities by coherence for bar chart.")
    return p.parse_args()



if __name__ == "__main__":
    # build graph -> find communities -> analyze semantics -> export

    # args = parse_args()
    #
    # if args.use_cache and cache_exists() and not args.recompute:
    #     # Fast path: plotting only (or light work)
    #     results_df, detailed_results, partition, char_to_radicals = \
    #         load_results()
    #     make_eval_figures(results_df, out_dir=args.out_dir,
    #                       top_n=args.top_n, label_top_k=3)
    #     # bubble (only if the json exists; otherwise skip silently)
    #     # if os.path.exists(BUBBLES_JSON):
    #     #     plot_top3_communities_bubbles_from_cache(
    #     #         cache_path=BUBBLES_JSON,
    #     #         save_path=os.path.join(args.out_dir, "top3_bubbles.png")
    #     #     )
    #     print("Loaded cached results. ")
    #     if not args.plot_only:
    #         # optional: print a short summary without the massive logs
    #         print("\nCached summary (head):")
    #         print(results_df.head(10).to_string(index=False))
    #     raise SystemExit(0)

    # 1) initialize the graph builder with your hanzi csv
    graph_builder = HanziRadicalGraph(str(DATA_DIR / 'chinese_characters_data3.csv'))

    # 2) build radical mappings and connect characters into a graph
    graph_builder.build_radical_mappings()
    graph = graph_builder.build_graph(min_shared_radicals=1,
                                      require_same_strokes=True)

    gephi_file_name = str(DATA_DIR / 'hanzi_radical_graph')

    # 3) quick stats before community detection to ensure graph looks sane
    stats = graph_builder.get_graph_statistics()
    print("\nGraph Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n=== COMMUNITY DETECTION ===")

    # 4) run louvain and get a partition (+ isolates)
    print("\nFinding Louvain Communities...")
    communities_louvain, isolates = \
        graph_builder.find_radical_communities_resolution(resolution=1,
                                                          # todo
                                                          min_size=10)

    if communities_louvain:
        # set up the semantic analyzer and load meanings
        analyzer = HanziCommunityAnalyzer()
        analyzer.analyze_communities(communities_louvain)
        print(f"\nIsolates count (unconnected nodes): {len(isolates)}")

        # store community ids on nodes for coloring in gephi
        graph_builder.add_community_data_to_graph(communities_louvain,
                                                  "louvain_community")

        print("\n=== SEMANTIC COMMUNITY ANALYSIS ===")

        # 5) reshape partition into {community_id -> [chars]} for semantic
        louvain_communities_dict = defaultdict(list)
        for character, community_id in communities_louvain.items():
            louvain_communities_dict[community_id].append(character)
        louvain_communities_dict = dict(louvain_communities_dict)

        print("Loading hanzi meanings for semantic analysis...")
        hanzi_meanings = analyzer.parse_hanzi_data(
            str(DATA_DIR / 'chinese_characters_data3.csv'))
        print(f"Loaded {len(hanzi_meanings)} hanzi characters with meanings")

        # 6) compute coherence scores per community and print a ranked table
        print("\nAnalyzing semantic coherence of communities...")
        results_df, detailed_results = analyzer.analyze_louvain_communities(
            louvain_communities_dict, hanzi_meanings)

        print("\n=== COMMUNITY SEMANTIC COHERENCE RESULTS ===")
        print(results_df.to_string(index=False, max_colwidth=80))

        print("\n=== DETAILED SEMANTIC ANALYSIS OF TOP COMMUNITIES ===")
        detailed_results_sorted = sorted(detailed_results,
                                         key=lambda x: x['coherence_score'],
                                         reverse=True)
        analysis = []

        # sort and take top communities
        detailed_results_sorted = sorted(detailed_results,
                                         key=lambda x: x['coherence_score'],
                                         reverse=True)
        top_communities = detailed_results_sorted[:5]

        analyses_by_cid: dict[int, dict] = {}

        for i, community_data in enumerate(top_communities, 1):
            print(f"\n--- Top Community #{i} ---")
            a = analyzer.detailed_community_analysis(community_data,
                                                     graph_builder.char_to_radicals)

            cid = int(community_data['community_id'])
            analyses_by_cid[
                cid] = a  # keep the full dict; we'll pick the bits we need
            # below

            if i < len(top_communities):
                print("\n" + "=" * 60)

        # write summary caches (existing)
        save_results(results_df, detailed_results,
                     partition=communities_louvain,
                     char_to_radicals=graph_builder.char_to_radicals)

        # export csvs for canva plotting (counts + percentages)
        csv_counts, csv_percent = export_cluster_radicals_csv(
            analyses_by_cid,
            out_csv_counts=os.path.join(CACHE_DIR,
                                        "cluster_radicals_counts.csv"),
            out_csv_percent=os.path.join(CACHE_DIR,
                                         "cluster_radicals_percent.csv"),
            disambiguate_groups_with_cid=True
        )
        print("Canva CSVs:", csv_counts, csv_percent)

        # also drop a csv for later inspection
        results_df.to_csv(str(DATA_DIR / 'semantic_community_analysis.csv'), index=False,
                          encoding='utf-8')
        print(f"\nSemantic analysis results exported to "
              f"semantic_community_analysis.csv")

    # 7) export to gephi for visualization
    print("\n=== EXPORTING TO GEPHI ===")
    graph_builder.export_to_gephi(gephi_file_name, "graphml")
