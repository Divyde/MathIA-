import matplotlib.pyplot as plt
from shapely.geometry import box
import geopandas as gpd
CRS_METRIC = "EPSG:32633"
PATH = "/home/divyde/PycharmProjects/MathIA /Math IA/data/ne_10m_coastline.zip"
SEGMENTS = {
    "Puglia (smooth)": (16.4, 39.8, 18.6, 42.0),
    "Liguria (rugged)":  (7.5, 43.7, 10.0, 44.5),
    "Po delta (lobed)":  (12.1, 44.7, 12.7, 45.2),
    "Sardinia Gallura":  (9.0, 40.8, 9.8, 41.3),
}
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

def plot_segments(coast_gdf, segments, crs=CRS_METRIC):
    italy = box(6, 36, 19, 47.5)                      # lon/lat window around Italy
    base = gpd.clip(coast_gdf, italy).to_crs(crs)

    fig, ax = plt.subplots(figsize=(8, 10))
    base.plot(ax=ax, color="lightgrey", linewidth=0.5)

    colours = plt.cm.tab10.colors
    for i, (name, bb) in enumerate(segments.items()):
        lines = load_coast_segment(PATH, bb)
        for j, ln in enumerate(lines):
            xs, ys = zip(*ln)
            ax.plot(xs, ys, color=colours[i], linewidth=1.5,
                    label=name if j == 0 else None)   # one legend entry per segment
        b = gpd.GeoSeries([box(*bb)], crs=4326).to_crs(crs)
        b.boundary.plot(ax=ax, color=colours[i], linestyle="--", linewidth=0.8)

    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.legend()
    plt.show()

coast_gdf = gpd.read_file(PATH)               # load the map once
plot_segments(coast_gdf, SEGMENTS)            # draw all segments
lines = load_coast_segment(PATH, SEGMENTS["Puglia (smooth)"])   # one segment's points