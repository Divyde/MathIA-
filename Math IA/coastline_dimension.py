"""
coastline_dimension.py  --  FIRST DRAFT (AI-assisted, to be rewritten by the student)

Functions: koch, densify, bounding_square, boxtree_counts, box_counts,
           ruler_length, linear_fit, box_dimension, ruler_dimension,
           local_slopes, run_koch_validation, main

A "line" is an (n, 2) array of x, y points in metres.
A "lines" argument is a list of such arrays (one coast segment can have several).
"""
import math
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- Koch curve
def koch(n, p0=(0.0, 0.0), p1=(1.0, 0.0)):
    """Recursive Koch curve of depth n from p0 to p1. Returns (4**n + 1, 2) array."""
    p0 = np.array(p0, float)
    p1 = np.array(p1, float)
    if n == 0:
        return np.array([p0, p1])
    d = (p1 - p0) / 3.0
    a = p0 + d
    b = p0 + 2.0 * d
    c, s = 0.5, math.sqrt(3) / 2                      # rotate d by +60 degrees
    peak = a + np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
    parts = []
    for q0, q1 in ((p0, a), (a, peak), (peak, b), (b, p1)):
        parts.append(koch(n - 1, q0, q1)[:-1])         # drop last point (duplicate)
    parts.append(p1[None, :])
    return np.vstack(parts)


# ------------------------------------------------------------------ helpers
def densify(line, max_step):
    """Insert points so no gap between neighbours exceeds max_step."""
    line = np.asarray(line, float)
    seg = np.diff(line, axis=0)
    d = np.hypot(seg[:, 0], seg[:, 1])
    k = np.maximum(1, np.ceil(d / max_step).astype(int))   # pieces per segment
    idx = np.repeat(np.arange(len(k)), k)
    start = np.cumsum(k) - k
    frac = (np.arange(k.sum()) - np.repeat(start, k)) / np.repeat(k, k)
    pts = line[idx] + frac[:, None] * seg[idx]
    return np.vstack([pts, line[-1]])


def bounding_square(lines):
    """Smallest square (x0, y0, side) containing all points (tiny pad added)."""
    allp = np.vstack([np.asarray(l, float) for l in lines])
    xmin, ymin = allp.min(axis=0)
    xmax, ymax = allp.max(axis=0)
    s = max(xmax - xmin, ymax - ymin) * 1.0001
    return xmin, ymin, s


def line_length(line):
    seg = np.diff(np.asarray(line, float), axis=0)
    return float(np.hypot(seg[:, 0], seg[:, 1]).sum())


def median_spacing(lines):
    steps = np.concatenate([np.hypot(*np.diff(np.asarray(l, float), axis=0).T) for l in lines])
    return float(np.median(steps))


# -------------------------------------------------------------- box counting
def boxtree_counts(pts, x0, y0, s, b, max_level, level=0, counts=None):
    """Recursive subdivision. counts[k] = number of occupied boxes of side s / b**k."""
    if counts is None:
        counts = [0] * (max_level + 1)
    counts[level] += 1                                   # this box is occupied
    if level == max_level:
        return counts
    h = s / b
    ix = np.clip(((pts[:, 0] - x0) // h).astype(int), 0, b - 1)
    iy = np.clip(((pts[:, 1] - y0) // h).astype(int), 0, b - 1)
    key = ix * b + iy
    order = np.argsort(key, kind="stable")
    key, pts = key[order], pts[order]
    cuts = np.flatnonzero(np.diff(key)) + 1
    firsts = key[np.r_[0, cuts]]
    for chunk, kk in zip(np.split(pts, cuts), firsts):
        boxtree_counts(chunk, x0 + (kk // b) * h, y0 + (kk % b) * h, h, b,
                       max_level, level + 1, counts)
    return counts


def box_counts(lines, b=2, max_level=8):
    """Return eps array (box sides) and N(eps) for levels 0..max_level."""
    x0, y0, s = bounding_square(lines)
    finest = s / b ** max_level
    pts = np.vstack([densify(l, finest / 2.0) for l in lines])   # so long straight bits count
    counts = boxtree_counts(pts, x0, y0, s, b, max_level)
    eps = np.array([s / b ** k for k in range(max_level + 1)])
    return eps, np.array(counts, float)


# --------------------------------------------------------------- ruler method
def ruler_length(line, eps):
    """Divider walk (iterative). Returns L(eps) = n * eps + r."""
    P = np.asarray(line, float)
    xs, ys = P[:, 0].tolist(), P[:, 1].tolist()
    N = len(xs)
    cx, cy = xs[0], ys[0]
    i, n = 0, 0
    while i < N - 1:
        ax, ay = cx, cy
        j, hit = i, None
        while j < N - 1:
            bx, by = xs[j + 1], ys[j + 1]
            if math.hypot(bx - cx, by - cy) >= eps:      # segment leaves the circle
                dx, dy = bx - ax, by - ay
                fx, fy = ax - cx, ay - cy
                A = dx * dx + dy * dy
                B = 2.0 * (fx * dx + fy * dy)
                C = fx * fx + fy * fy - eps * eps
                t = (-B + math.sqrt(max(B * B - 4 * A * C, 0.0))) / (2 * A)   # "+" root
                t = min(max(t, 0.0), 1.0)
                hit = (ax + t * dx, ay + t * dy)
                break
            ax, ay = bx, by
            j += 1
        if hit is None:
            break
        cx, cy = hit
        i = j
        n += 1
    r = math.hypot(xs[-1] - cx, ys[-1] - cy)
    return n * eps + r


# -------------------------------------------------------------------- fitting
def linear_fit(x, y):
    """Least squares line. Returns slope, intercept, standard error of slope, R^2."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    xm, ym = x.mean(), y.mean()
    sxx = np.sum((x - xm) ** 2)
    slope = np.sum((x - xm) * (y - ym)) / sxx
    icpt = ym - slope * xm
    resid = y - (icpt + slope * x)
    se = math.sqrt(np.sum(resid ** 2) / (n - 2)) / math.sqrt(sxx) if n > 2 else float("nan")
    ss_tot = np.sum((y - ym) ** 2)
    r2 = 1 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else float("nan")
    return slope, icpt, se, r2


def _select(eps, vals, fit_range):
    if fit_range is None:
        return eps, vals
    lo, hi = fit_range
    m = (eps >= lo) & (eps <= hi)
    return eps[m], vals[m]


def box_dimension(lines, b=2, max_level=8, min_level=2, fit_range=None):
    """D_B = gradient of ln N against ln(1/eps). Returns D, se, R^2, eps, N."""
    eps, N = box_counts(lines, b, max_level)
    e, n = eps[min_level:], N[min_level:]
    e_fit, n_fit = _select(e, n, fit_range)
    D, _, se, r2 = linear_fit(np.log(1 / e_fit), np.log(n_fit))
    return D, se, r2, eps, N


def ruler_dimension(lines, eps_list, fit_range=None):
    """D = 1 - gradient of ln L against ln eps. Returns D, se, R^2, eps, L."""
    eps = np.array(sorted(eps_list), float)
    L = np.array([sum(ruler_length(l, e) for l in lines) for e in eps])
    e_fit, L_fit = _select(eps, L, fit_range)
    slope, _, se, r2 = linear_fit(np.log(e_fit), np.log(L_fit))
    return 1 - slope, se, r2, eps, L


def local_slopes(eps, y, window=3, ruler=False):
    """Sliding-window slope. Returns list of (centre eps, local D)."""
    eps, y = np.asarray(eps, float), np.asarray(y, float)
    order = np.argsort(eps)[::-1]                        # large -> small eps
    eps, y = eps[order], y[order]
    out = []
    for i in range(len(eps) - window + 1):
        ex, ey = eps[i:i + window], y[i:i + window]
        if ruler:
            D = 1 - np.polyfit(np.log(ex), np.log(ey), 1)[0]
        else:
            D = np.polyfit(np.log(1 / ex), np.log(ey), 1)[0]
        out.append((float(np.exp(np.log(ex).mean())), float(D)))
    return out


# ------------------------------------------------------------------- plotting
def plot_loglog(eps, vals, D, title, ruler=False, fit_range=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    if ruler:
        ax.loglog(eps, vals, "o-")
        ax.set_ylabel("L(eps) (m)")
        ax.set_xlabel("ruler length eps (m)")
    else:
        ax.loglog(1 / eps, vals, "o-")
        ax.set_ylabel("N(eps)")
        ax.set_xlabel("1 / eps (1/m)")
    ax.set_title(f"{title}: D = {D:.3f}")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    return fig


# ----------------------------------------------------------------- validation
def run_koch_validation(depth=9, tol=0.01):
    """Both methods on the Koch curve. Theory: ln4 / ln3 = 1.2619."""
    theory = math.log(4) / math.log(3)
    line = koch(depth)
    D_box, se_b, r2_b, _, _ = box_dimension([line], b=3, max_level=8, min_level=4)
    eps_list = [1 / (2 * 3 ** k) for k in range(1, 7)]       # 1/6 ... 1/1458
    D_rul, se_r, r2_r, _, _ = ruler_dimension([line], eps_list)
    print(f"Koch depth {depth}: {len(line)} points, theory {theory:.4f}")
    print(f"  box   D = {D_box:.4f} +- {se_b:.4f}  R2 = {r2_b:.5f}")
    print(f"  ruler D = {D_rul:.4f} +- {se_r:.4f}  R2 = {r2_r:.5f}")
    ok = abs(D_box - theory) < tol and abs(D_rul - theory) < tol
    print("  PASS" if ok else "  FAIL (difference above tolerance)")
    return ok


# ----------------------------------------------------------------------- main
def main():
    ok = run_koch_validation()
    if not ok:
        print("Fix the Koch validation before trusting any coastline numbers.")

    from region_testing import SEGMENTS, PATH, load_coast_segment   # needs geopandas

    data = {name: load_coast_segment(PATH, bb) for name, bb in SEGMENTS.items()}

    # common eps floor: twice the coarsest point spacing among all segments
    spacings = {n: median_spacing(l) for n, l in data.items()}
    eps_floor = 2 * max(spacings.values())
    eps_list = [eps_floor * 2 ** k for k in range(6)]
    print(f"\nCommon eps floor {eps_floor:.0f} m; ruler lengths {[round(e) for e in eps_list]}")

    for name, lines in data.items():
        length_km = sum(line_length(l) for l in lines) / 1000
        print(f"\n{name}: {len(lines)} lines, {length_km:.1f} km, "
              f"median spacing {spacings[name]:.0f} m")
        _, _, s = bounding_square(lines)
        max_level = max(3, min(10, int(math.log2(s / eps_floor))))
        fr = (eps_floor, eps_list[-1])
        Db, se_b, r2_b, e_b, N_b = box_dimension(lines, b=2, max_level=max_level, fit_range=fr)
        Dr, se_r, r2_r, e_r, L_r = ruler_dimension(lines, eps_list)
        print(f"  box   D = {Db:.3f} +- {se_b:.3f}  R2 = {r2_b:.3f}  (eps {fr[0]:.0f}-{fr[1]:.0f} m)")
        print(f"  ruler D = {Dr:.3f} +- {se_r:.3f}  R2 = {r2_r:.3f}  (eps {fr[0]:.0f}-{fr[1]:.0f} m)")
        print("  local box slopes  :", [(round(e), round(d, 3)) for e, d in local_slopes(e_b, N_b)])
        print("  local ruler slopes:", [(round(e), round(d, 3)) for e, d in local_slopes(e_r, L_r, ruler=True)])
        plot_loglog(e_b, N_b, Db, f"{name} box")
        plot_loglog(e_r, L_r, Dr, f"{name} ruler", ruler=True)
    plt.show()


if __name__ == "__main__":
    main()