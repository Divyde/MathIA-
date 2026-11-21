import inspect
import numpy as np
import coastline_dimension as cd
from region_testing import SEGMENTS, PATH, load_coast_segment

# Show exactly how your measuring functions expect to be called
for fn in ("box_dimension", "ruler_dimension", "load_italy_coast"):
    if hasattr(cd, fn):
        print(fn, inspect.signature(getattr(cd, fn)))

print()
for name, bbox in SEGMENTS.items():
    lines = load_coast_segment(PATH, bbox)
    steps = np.concatenate([np.hypot(*np.diff(np.array(l), axis=0).T) for l in lines])
    length_km = sum(np.sum(np.hypot(*np.diff(np.array(l), axis=0).T)) for l in lines) / 1000
    print(f"{name}: {len(lines)} lines, {length_km:.1f} km, "
          f"median spacing {np.median(steps):.0f} m, largest jump {steps.max():.0f} m")