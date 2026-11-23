"""
region_testing.py  --  FIRST DRAFT (AI-assisted, to be rewritten by the student)

Defines the coast segments and loads them from the OSM coastline file.
Importing this file does nothing heavy. Run it directly to plot the segments.
"""
import os
os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")      # rebuild a missing .shx if needed

from functools import lru_cache

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import pyogrio
from shapely.geometry import box
from shapely.ops import linemerge

CRS_METRIC = "EPSG:32633"                               # one projection for every segment
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "lines.shp")

# (lon_min, lat_min, lon_max, lat_max)
SEGMENTS = {
    "Puglia (smooth)":  (16.4, 39.8, 18.6, 42.0),
    "Liguria (rugged)": (7.5, 43.7, 10.0, 44.5),
    "Po delta (lobed)": (12.1, 44.7, 12.7, 45.2),
    "Sardinia Gallura": (9.0, 40.8, 9.8, 41.3),
}


@lru_cache(maxsize=None)
def load_coast_segment(coast_path, lonlat_box, crs=CRS_METRIC):
    """Lines (numpy arrays in metres, CRS_METRIC) for the coast inside a lon/lat box."""
    file_crs = pyogrio.read_info(coast_path)["crs"]
    window = gpd.GeoSeries([box(*lonlat_box)], crs=4326).to_crs(file_crs)
    gdf = gpd.read_file(coast_path, bbox=tuple(window.total_bounds))    # reads only the window
    gdf = gpd.clip(gdf, window.iloc[0]).to_crs(crs)
    merged = linemerge(list(gdf.geometry.explode(index_parts=False)))   # rejoin split pieces
    parts = [merged] if merged.geom_type == "LineString" else list(merged.geoms)
    return [np.array(p.coords) for p in parts if p.length > 0]


def plot_segments(coast_gdf, segments, crs=CRS_METRIC):
    italy = box(6, 36, 19, 47.5)
    base = gpd.clip(coast_gdf, italy).to_crs(crs)

    fig, ax = plt.subplots(figsize=(8, 10))
    base.plot(ax=ax, color="lightgrey", linewidth=0.5)

    colours = plt.cm.tab10.colors
    for i, (name, bb) in enumerate(segments.items()):
        lines = load_coast_segment(PATH, bb)
        for j, ln in enumerate(lines):
            xs, ys = ln[:, 0], ln[:, 1]
            ax.plot(xs, ys, color=colours[i], linewidth=1.5, label=name if j == 0 else None)
        b = gpd.GeoSeries([box(*bb)], crs=4326).to_crs(crs)
        b.boundary.plot(ax=ax, color=colours[i], linestyle="--", linewidth=0.8)

    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    italy = box(6, 36, 19, 47.5)
    file_crs = pyogrio.read_info(PATH)["crs"]
    win = gpd.GeoSeries([italy], crs=4326).to_crs(file_crs)
    coast_gdf = gpd.read_file(PATH, bbox=tuple(win.total_bounds))       # Italy window only
    print("File:", PATH)
    print("Features read:", len(coast_gdf), "| file CRS:", file_crs)
    plot_segments(coast_gdf, SEGMENTS)