"""
Maths IA - fractal dimension of the Italian coastline.

Contents
  1. koch()                    recursive Koch curve generator (validation curve)
  2. boxtree_counts()          recursive box counting (each box splits into b x b)
  3. ruler_length()            divider / ruler method (Richardson)
  4. linear_fit()              least squares gradient, standard error, R^2
  5. load_italy_coast()        clip the coastline to Italy and project to metres
  6. plot_analysis()           map and labelled log-log graphs

Run this file directly to save italian_coastline_analysis.png beside this script.
"""
import math
from pathlib import Path
from region_testing import SEGMENTS, PATH, CRS_METRIC, load_coast_segment
import numpy as np
import geopandas as gpd
import pyogrio


# ----------------------------------------------------------------------
# 1. Koch curve (recursive)
# ----------------------------------------------------------------------
def koch(p1, p2, depth):
    """
    """
    if depth == 0:
        return [p1, p2] #base case

    (x1, y1), (x2, y2) = p1, p2
    dx, dy = (x2 - x1) / 3, (y2 - y1) / 3
    a = (x1 + dx, y1 + dy)
    b = (x1 + 2 * dx, y1 + 2 * dy)
    c, s = 0.5, math.sqrt(3) / 2
    peak = (a[0] + dx * c - dy * s, a[1] + dx * s + dy * c)

    points = []
    for q1, q2 in [(p1, a), (a, peak), (peak, b), (b, p2)]:
        points.extend(koch(q1, q2, depth - 1)[:-1])   # drop shared end point
    points.append(p2)
    return points

# extras

def densify(points, step):
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        raise ValueError("At least two points are required.")
    if not np.isfinite(pts).all() or step <= 0:
        raise ValueError("Points must be finite and step must be positive.")
    out = []
    for p, q in zip(pts[:-1], pts[1:]):
        n = max(1, int(math.ceil(np.hypot(*(q - p)) / step)))
        t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
        out.append(p + t * (q - p))
    out.append(pts[-1:])
    return np.vstack(out)

def bounding_square(pts):
    if len(pts) == 0:
        raise ValueError("At least one point is required.")
    x0, y0 = pts.min(axis=0)
    side = float((pts.max(axis=0) - pts.min(axis=0)).max()) * (1 + 1e-9)
    if side <= 0:
        raise ValueError("The points do not span a non-zero area.")
    return float(x0), float(y0), side


def as_curves(points):
    """Normalize one polyline or a collection of polylines to validated arrays."""
    try:
        array = np.asarray(points, dtype=float)
    except (TypeError, ValueError):
        array = None
    if array is not None and array.ndim == 2:
        curves = [array]
    elif array is not None and array.ndim == 3:
        curves = list(array)
    else:
        curves = [np.asarray(curve, dtype=float) for curve in points]
    if not curves:
        raise ValueError("At least one curve is required.")
    for curve in curves:
        if curve.ndim != 2 or curve.shape[1] != 2 or len(curve) < 2:
            raise ValueError("Each curve must contain at least two (x, y) points.")
        if not np.isfinite(curve).all():
            raise ValueError("Curve coordinates must be finite.")
    return curves


def boxtree_counts(points, x0, y0, side, max_level, counts, level=0, b=2):
    if len(points) == 0:
        return
    counts[level] += 1
    if level == max_level:
        return                   # Base case: no points in this box (nothing to count) or max_level reached.

    cell = side / b
    ix = np.minimum(((points[:, 0] - x0) // cell).astype(int), b - 1)
    iy = np.minimum(((points[:, 1] - y0) // cell).astype(int), b - 1)
    for i in range(b):
        for j in range(b):
            mask = (ix == i) & (iy == j)
            if mask.any():
                boxtree_counts(points[mask], x0 + i * cell, y0 + j * cell,
                               cell, max_level, counts, level + 1, b)

def box_counts(points, max_level, b=2):
    curves = as_curves(points)
    if max_level < 1 or b < 2:
        raise ValueError("max_level must be positive and b must be at least 2.")
    pts = np.vstack(curves)
    x0, y0, side = bounding_square(pts)
    dense = np.vstack([
        densify(curve, step=side / b ** (max_level + 1)) for curve in curves
    ])
    counts = [0] * (max_level + 1)
    boxtree_counts(dense, x0, y0, side, max_level, counts, 0, b)
    eps = np.array([side / b ** k for k in range(max_level + 1)])
    return eps, np.array(counts, dtype=float)

def ruler_length(points, eps):
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        raise ValueError("points must be an array of at least two (x, y) pairs.")
    if not np.isfinite(pts).all() or not np.isfinite(eps) or eps <= 0:
        raise ValueError("points must be finite and eps must be positive.")
    pts = pts[np.r_[True, np.any(np.diff(pts, axis=0) != 0, axis=1)]]
    if len(pts) < 2:
        raise ValueError("The curve must contain at least one non-zero segment.")

    cur = pts[0].copy()
    segment = 0
    steps = 0
    while segment < len(pts) - 1:
        found = False
        for j in range(segment, len(pts) - 1):
            a = cur if j == segment else pts[j]
            b = pts[j + 1]
            d = b - a
            A = float(d @ d)
            if A == 0:
                continue
            f = a - cur
            B = 2 * float(d @ f)
            C = float(f @ f - eps ** 2)
            discriminant = B * B - 4 * A * C
            if discriminant >= 0:
                root = math.sqrt(discriminant)
                intersections = sorted(
                    t for t in ((-B - root) / (2 * A), (-B + root) / (2 * A))
                    if 1e-12 < t <= 1 + 1e-12
                )
                if intersections:
                    t = min(intersections)
                    cur = a + min(t, 1.0) * d
                    segment = j
                    steps += 1
                    found = True
                    break
        if not found:
            break
    remainder = float(np.hypot(*(pts[-1] - cur)))
    return steps * eps + remainder, steps


def ruler_lengths(points, eps_list):
    curves = as_curves(points)
    return np.array([
        sum(ruler_length(curve, e)[0] for curve in curves) for e in eps_list
    ])


# ----------------------------------------------------------------------
# 4. Least squares regression
# ----------------------------------------------------------------------
def linear_fit(x, y):
    """Gradient m, intercept c, standard error of m, and R^2 for y = m x + c."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if x.ndim != 1 or y.ndim != 1 or len(y) != n or n < 3:
        raise ValueError("At least three paired x and y values are required.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("x and y must contain only finite values.")
    sxx = np.sum((x - x.mean()) ** 2)
    if sxx == 0:
        raise ValueError("x values must not all be equal.")
    m = np.sum((x - x.mean()) * (y - y.mean())) / sxx
    c = y.mean() - m * x.mean()
    resid = y - (m * x + c)
    s = math.sqrt(np.sum(resid ** 2) / (n - 2))
    se_m = s / math.sqrt(sxx)
    syy = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 if syy == 0 and np.sum(resid ** 2) == 0 else (
        1 - np.sum(resid ** 2) / syy if syy else float("nan")
    )
    return m, c, se_m, r2


def box_dimension(points, kmin, kmax, b=2):
    """D_B from the gradient of ln N against ln(1/eps) over levels kmin..kmax."""
    if kmin < 0 or kmax <= kmin:
        raise ValueError("Require 0 <= kmin < kmax.")
    eps, N = box_counts(points, kmax, b)
    x = np.log(1 / eps[kmin:kmax + 1])
    y = np.log(N[kmin:kmax + 1])
    m, c, se, r2 = linear_fit(x, y)
    return m, se, r2, (x, y)


def ruler_dimension(points, eps_list):
    """D from Richardson: ln L = ln F + (1 - D) ln eps, so D = 1 - gradient."""
    eps_list = np.asarray(eps_list, dtype=float)
    if eps_list.ndim != 1 or len(eps_list) < 3 or np.any(eps_list <= 0):
        raise ValueError("eps_list must contain at least three positive ruler lengths.")
    L = ruler_lengths(points, eps_list)
    x, y = np.log(eps_list), np.log(L)
    m, c, se, r2 = linear_fit(x, y)
    return 1 - m, se, r2, (x, y)



def find_country_file():
    """Find a Natural Earth country-boundary file, including the local test fixture."""
    base = Path(__file__).resolve().parent
    candidates = [
        base / "data/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp",
        base / "data/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp",
        Path(pyogrio.__file__).resolve().parent
        / "tests/fixtures/naturalearth_lowres/naturalearth_lowres.shp",
    ]
    return next((path for path in candidates if path.is_file()), None)


def load_italy_coast(coast_path, country_path, epsg=32632):
    """Clip Natural Earth coastline to Italy and return its longest connected line."""
    from shapely.ops import linemerge

    if not Path(coast_path).is_file():
        raise FileNotFoundError(f"Coastline shapefile not found: {coast_path}")
    if not Path(country_path).is_file():
        raise FileNotFoundError(
            "Italy's boundary shapefile was not found. Put Natural Earth's "
            "ne_10m_admin_0_countries.shp and its companion files in "
            "'data/ne_10m_admin_0_countries/'."
        )

    coast = gpd.read_file(coast_path)
    countries = gpd.read_file(country_path)
    name_column = next(
        (column for column in ("ADMIN", "name", "NAME") if column in countries.columns),
        None,
    )
    if name_column is None:
        raise ValueError("Country data must contain an ADMIN, name, or NAME column.")
    italy = countries[countries[name_column].astype(str).str.casefold() == "italy"]
    if italy.empty and "iso_a3" in countries.columns:
        italy = countries[countries["iso_a3"] == "ITA"]
    if italy.empty:
        raise ValueError("The country-boundary file does not contain Italy.")

    if coast.crs is None:
        raise ValueError("The coastline file must declare its coordinate reference system.")
    # The packaged low-resolution fallback needs a wider margin to retain
    # detailed shoreline that its simplified country polygon cuts off.
    buffer_m = 15000 if Path(country_path).name == "naturalearth_lowres.shp" else 5000
    mask_metric = italy.to_crs(epsg).geometry.buffer(buffer_m)
    mask = gpd.GeoSeries(mask_metric, crs=epsg).to_crs(coast.crs).union_all()
    clipped = coast.clip(mask).to_crs(epsg)
    lines = []
    for geometry in clipped.geometry:
        if geometry is None or geometry.is_empty:
            continue
        if geometry.geom_type == "LineString":
            lines.append(geometry)
        elif geometry.geom_type == "MultiLineString":
            lines.extend(geometry.geoms)
    if not lines:
        raise ValueError("No coastline intersected the Italian country boundary.")
    merged = linemerge(lines)
    components = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
    components = [line for line in components if line.geom_type == "LineString"]
    if not components:
        raise ValueError("Could not extract a connected Italian coastline.")
    return [np.asarray(line.coords) for line in components], coast.crs


# ----------------------------------------------------------------------
# 6. Graph for the write-up
# ----------------------------------------------------------------------
def plot_fit(x, y, xlabel, ylabel, title, outfile):
    import matplotlib.pyplot as plt
    m, c, se, r2 = linear_fit(x, y)
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, label="data")
    xs = np.linspace(min(x), max(x), 50)
    plt.plot(xs, m * xs + c, label=f"fit: gradient = {m:.3f} ± {se:.3f}, R² = {r2:.4f}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def plot_analysis(coastline, box_result, ruler_result, output):
    import matplotlib.pyplot as plt

    box_d, box_se, box_r2, (box_x, box_y) = box_result
    ruler_d, ruler_se, ruler_r2, (ruler_x, ruler_y) = ruler_result
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for component in coastline:
        axes[0].plot(component[:, 0], component[:, 1], color="steelblue", linewidth=0.8)
    axes[0].set_title("Italian coastline")
    axes[0].set_xlabel("Easting (m)")
    axes[0].set_ylabel("Northing (m)")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].grid(alpha=0.2)

    for ax, x, y, title, xlabel, ylabel, dimension, error, r2 in (
        (axes[1], box_x, box_y, "Box counting", "ln(1 / box size)", "ln(box count)",
         box_d, box_se, box_r2),
        (axes[2], ruler_x, ruler_y, "Ruler method", "ln(ruler size)", "ln(measured length)",
         ruler_d, ruler_se, ruler_r2),
    ):
        slope = dimension if title == "Box counting" else 1 - dimension
        intercept = float(np.mean(y) - slope * np.mean(x))
        ax.scatter(x, y, color="steelblue", label="measurements")
        fit_x = np.linspace(x.min(), x.max(), 100)
        ax.plot(fit_x, slope * fit_x + intercept, color="darkorange", label="linear fit")
        ax.set_title(f"{title}: D = {dimension:.3f} +/- {error:.3f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(title=f"$R^2$ = {r2:.4f}")
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)

def load_coast_segment(coast_path, lonlat_box, crs=CRS_METRIC):
    """lonlat_box = (lon_min, lat_min, lon_max, lat_max)"""
    import geopandas as gpd
    from shapely.geometry import box
    gdf = gpd.read_file(coast_path)
    clip = box(*lonlat_box)
    seg = gpd.clip(gdf, clip).to_crs(crs)
    seg = seg.explode(index_parts=False)          # split MultiLineStrings
    lines = [list(g.coords) for g in seg.geometry if g.length > 0]
    return lines


# if __name__ == "__main__":
#     base = Path(__file__).resolve().parent
#     coast_path = base / "data/ne_10m_coastline/ne_10m_coastline.shp"
#     country_path = find_country_file()
#     if country_path is None:
#         raise FileNotFoundError(
#             "Italy's Natural Earth country boundary is missing. Place the "
#             "Natural Earth countries shapefile in data/ne_10m_admin_0_countries/."
#         )
#
#     italy_coast, _ = load_italy_coast(coast_path, country_path)
#     box_result = box_dimension(italy_coast, kmin=2, kmax=8, b=2)
#     _, _, side = bounding_square(np.vstack(italy_coast))
#     ruler_sizes = side / (2.0 ** np.arange(2, 9))
#     ruler_result = ruler_dimension(italy_coast, ruler_sizes)
#     output = base / "italian_coastline_analysis.png"
#     plot_analysis(italy_coast, box_result, ruler_result, output)
#
#     print(f"Italian coast: {len(italy_coast)} connected segments, "
#           f"{sum(len(line) for line in italy_coast)} points")
#     print(f"Country boundary data: {country_path}")
#     print(f"Box-counting dimension: {box_result[0]:.4f} +/- {box_result[1]:.4f} "
#           f"(R^2={box_result[2]:.4f})")
#     print(f"Ruler-method dimension: {ruler_result[0]:.4f} +/- {ruler_result[1]:.4f} "
#           f"(R^2={ruler_result[2]:.4f})")
#     print(f"Saved graphs to {output}")

if __name__ == '__main__':
    coast_gdf = gpd.read_file(PATH)
    plot_segments(coast_gdf, SEGMENTS)

curve = koch((0.0, 0.0), (1.0, 0.0), depth=9)
D_box = box_dimension(curve, kmin=5, kmax=8, b=3)
D_ruler = ruler_dimension(curve, np.array([3.0 ** -n for n in range(1, 6)]) * 0.9)
print(math.log(4) / math.log(3), D_box[0], D_ruler[0])