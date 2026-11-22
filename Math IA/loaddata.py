import os
from functools import lru_cache
import numpy as np
import geopandas as gpd
import pyogrio
from shapely.geometry import box
from shapely.ops import linemerge

CRS_METRIC = "EPSG:32633"
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "coastline.shp")


@lru_cache(maxsize=None)
def load_coast_segment(coast_path, lonlat_box, crs=CRS_METRIC):
    """Lines (metres, CRS_METRIC) for the coast inside a lon/lat box."""
    file_crs = pyogrio.read_info(coast_path)["crs"]
    window = gpd.GeoSeries([box(*lonlat_box)], crs=4326).to_crs(file_crs)
    gdf = gpd.read_file(coast_path, bbox=tuple(window.total_bounds))   # reads only the window
    gdf = gpd.clip(gdf, window.iloc[0]).to_crs(crs)
    merged = linemerge(list(gdf.geometry.explode(index_parts=False)))  # rejoin split pieces
    parts = [merged] if merged.geom_type == "LineString" else list(merged.geoms)
    return [np.array(p.coords) for p in parts if p.length > 0]