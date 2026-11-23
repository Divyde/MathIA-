"""
run_regions.py  --  FIRST DRAFT (AI-assisted, to be rewritten by the student)

Runs every test on the three regions and saves everything to ./results :
    log.txt            everything that was printed
    summary.csv        one row per region / variant / method
    <region>_<variant>_box.csv, _ruler.csv   the raw N(eps) and L(eps) curves
    <region>_<variant>.png                   log-log fits + local slopes
    summary_D.png      D (box vs ruler) for every region

Variants: "all lines" (islands included) and "longest line only" (mainland proxy).
Needs coastline_dimension.py and region_testing.py in the same folder.
"""
import matplotlib
matplotlib.use("Agg")                     # no pop-up windows, so it can run unattended

import csv
import math
import os
import sys
import time
import traceback

import numpy as np
import matplotlib.pyplot as plt

import coastline_dimension as cd
from region_testing import SEGMENTS, PATH, load_coast_segment

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

EXCLUDE = ("Po delta",)                    # segments to skip (cut from the IA)
REGIONS = {n: bb for n, bb in SEGMENTS.items() if not any(x in n for x in EXCLUDE)}


# ------------------------------------------------------------- logging helper
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def slug(s):
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")


# ------------------------------------------------------------------- helpers
def fit_on(eps, vals, fit_range, ruler):
    """Fit inside fit_range. Returns dict or None if fewer than 3 points."""
    m = (eps >= fit_range[0]) & (eps <= fit_range[1])
    e, v = eps[m], vals[m]
    if len(e) < 3:
        return None
    x = np.log(e) if ruler else np.log(1.0 / e)
    slope, icpt, se, r2 = cd.linear_fit(x, np.log(v))
    D = 1 - slope if ruler else slope
    return dict(D=D, se=se, r2=r2, slope=slope, icpt=icpt, n=len(e),
                eps_lo=float(e.min()), eps_hi=float(e.max()))


def measure(lines, eps_floor, eps_list):
    """Box counting (base 2) and ruler method on a list of lines."""
    _, _, s = cd.bounding_square(lines)
    max_level = int(max(3, min(14, math.floor(math.log2(s / eps_floor)))))
    t = time.time()
    eps_b, N = cd.box_counts(lines, b=2, max_level=max_level)
    t_box = time.time() - t
    t = time.time()
    eps_r = np.array(sorted(eps_list), float)
    L = np.array([sum(cd.ruler_length(l, e) for l in lines) for e in eps_r])
    t_rul = time.time() - t
    return eps_b, N, eps_r, L, max_level, t_box, t_rul


def plot_variant(title, eps_b, N, eps_r, L, fb, fr, fit_range, path):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    # box counting
    a = ax[0, 0]
    a.loglog(1 / eps_b, N, "o-", label="data")
    if fb:
        xs = np.array([1 / fit_range[1], 1 / fit_range[0]])
        a.loglog(xs, np.exp(fb["icpt"]) * xs ** fb["slope"], "r--",
                 label=f"fit D_B = {fb['D']:.3f}")
    a.set_xlabel("1 / eps  (1/m)")
    a.set_ylabel("N(eps)")
    a.set_title("Box counting")
    a.legend()
    a.grid(True, which="both", alpha=0.3)
    # ruler
    a = ax[0, 1]
    a.loglog(eps_r, L, "o-", label="data")
    if fr:
        xs = np.array([fit_range[0], fit_range[1]])
        a.loglog(xs, np.exp(fr["icpt"]) * xs ** fr["slope"], "r--",
                 label=f"fit D = {fr['D']:.3f}")
    a.set_xlabel("ruler length eps (m)")
    a.set_ylabel("L(eps) (m)")
    a.set_title("Ruler method")
    a.legend()
    a.grid(True, which="both", alpha=0.3)
    # local slopes
    for a, (e, y, ruler, fit, name) in zip(
            (ax[1, 0], ax[1, 1]),
            ((eps_b, N, False, fb, "box"), (eps_r, L, True, fr, "ruler"))):
        ls = cd.local_slopes(e, y, window=3, ruler=ruler)
        if ls:
            a.semilogx([p[0] for p in ls], [p[1] for p in ls], "o-")
        if fit:
            a.axhline(fit["D"], color="r", linestyle="--")
        a.axvspan(fit_range[0], fit_range[1], alpha=0.1)
        a.set_xlabel("eps (m)")
        a.set_ylabel("local D")
        a.set_title(f"Local slopes ({name}); shaded = fit range")
        a.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ----------------------------------------------------------------------- main
def main():
    log = open(os.path.join(OUT, "log.txt"), "w")
    sys.stdout = Tee(sys.__stdout__, log)
    t0 = time.time()
    print("RUN STARTED", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("Regions:", list(REGIONS))
    print("Data file:", PATH)

    # 1) Koch validation
    print("\n=== Koch validation ===")
    koch_ok = cd.run_koch_validation()
    if not koch_ok:
        print("WARNING: Koch validation failed. Treat all numbers with care.")

    # 2) load all regions
    print("\n=== Loading regions ===")
    data = {}
    for name, bb in REGIONS.items():
        t = time.time()
        data[name] = load_coast_segment(PATH, bb)
        print(f"{name}: loaded in {time.time() - t:.0f} s")

    print("\n=== Data summary ===")
    spacings = {}
    lengths = {}
    for name, lines in data.items():
        lens = np.array(sorted([cd.line_length(l) for l in lines], reverse=True))
        spacings[name] = cd.median_spacing(lines)
        lengths[name] = lens.sum()
        worst = max(np.hypot(*np.diff(np.asarray(l, float), axis=0).T).max() for l in lines)
        print(f"{name}: {len(lines)} lines, {lens.sum() / 1000:.1f} km, "
              f"longest line {lens[0] / 1000:.1f} km ({lens[0] / lens.sum():.0%}), "
              f"{int((lens < 100).sum())} lines under 100 m, "
              f"median spacing {spacings[name]:.0f} m, largest jump {worst:.0f} m")

    # 3) common eps range
    eps_floor = 2 * max(spacings.values())
    eps_cap = min(lengths.values()) / 10
    eps_list = []
    e = eps_floor
    while e <= eps_cap:
        eps_list.append(e)
        e *= 2
    fit_range = (eps_floor, eps_list[-1])
    print(f"\nCommon eps range: {eps_floor:.0f} m to {eps_list[-1]:.0f} m "
          f"({len(eps_list)} ruler lengths): {[round(x) for x in eps_list]}")

    # 4) run everything
    summary_rows = []
    header = ["region", "variant", "method", "D", "std_error", "R2", "n_points",
              "eps_lo_m", "eps_hi_m", "length_km", "lines", "pct_longer_when_ruler_halved"]
    results = {}                                           # for the summary plot
    for name, lines in data.items():
        variants = {"all lines": lines,
                    "longest line only": [max(lines, key=cd.line_length)]}
        for vname, vlines in variants.items():
            tag = f"{slug(name)}_{slug(vname)}"
            print(f"\n=== {name} | {vname} ===")
            try:
                eps_b, N, eps_r, L, max_level, t_box, t_rul = measure(vlines, eps_floor, eps_list)
                print(f"box counting: {t_box:.0f} s (levels 0-{max_level}) | ruler: {t_rul:.0f} s")
                fb = fit_on(eps_b, N, fit_range, ruler=False)
                fr = fit_on(eps_r, L, fit_range, ruler=True)
                length_km = sum(cd.line_length(l) for l in vlines) / 1000
                for method, f in (("box", fb), ("ruler", fr)):
                    if f is None:
                        print(f"  {method}: fewer than 3 points in the fit range")
                        continue
                    pct = (2 ** (f["D"] - 1) - 1) * 100
                    print(f"  {method:5s} D = {f['D']:.3f} +- {f['se']:.3f}  R2 = {f['r2']:.3f}  "
                          f"({f['n']} points, eps {f['eps_lo']:.0f}-{f['eps_hi']:.0f} m)  "
                          f"halving the ruler adds {pct:.1f}%")
                    summary_rows.append([name, vname, method, round(f["D"], 4), round(f["se"], 4),
                                         round(f["r2"], 4), f["n"], round(f["eps_lo"]),
                                         round(f["eps_hi"]), round(length_km, 1), len(vlines),
                                         round(pct, 2)])
                    results[(name, vname, method)] = f
                if fb and fr:
                    diff = abs(fb["D"] - fr["D"]) / math.hypot(fb["se"], fr["se"])
                    print(f"  box vs ruler differ by {diff:.1f} combined standard errors")
                print("  local box slopes  :", [(round(a), round(b, 3)) for a, b in
                                               cd.local_slopes(eps_b, N)])
                print("  local ruler slopes:", [(round(a), round(b, 3)) for a, b in
                                               cd.local_slopes(eps_r, L, ruler=True)])
                write_csv(os.path.join(OUT, tag + "_box.csv"), ["eps_m", "N"],
                          zip(eps_b.tolist(), N.tolist()))
                write_csv(os.path.join(OUT, tag + "_ruler.csv"), ["eps_m", "L_m"],
                          zip(eps_r.tolist(), L.tolist()))
                plot_variant(f"{name} | {vname}", eps_b, N, eps_r, L, fb, fr, fit_range,
                             os.path.join(OUT, tag + ".png"))
            except Exception:
                print("FAILED:", name, vname)
                traceback.print_exc()
            write_csv(os.path.join(OUT, "summary.csv"), header, summary_rows)   # save as we go

    # 5) summary figure
    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        keys = [(n, v) for n in data for v in ("all lines", "longest line only")]
        x = np.arange(len(keys))
        for off, method, colour in ((-0.2, "box", "tab:blue"), (0.2, "ruler", "tab:orange")):
            ys = [results.get((n, v, method), {}).get("D", np.nan) for n, v in keys]
            es = [results.get((n, v, method), {}).get("se", 0) for n, v in keys]
            ax.bar(x + off, ys, 0.4, yerr=es, label=method, color=colour, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n.split(' (')[0]}\n{v}" for n, v in keys], fontsize=7)
        ax.set_ylabel("fractal dimension D")
        ax.set_ylim(1.0, None)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "summary_D.png"), dpi=130)
        plt.close(fig)
    except Exception:
        traceback.print_exc()

    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min. Results in: {OUT}")


if __name__ == "__main__":
    main()