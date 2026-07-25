"""Independent-observable extension using replaceable synthetic proxies.

The built-in values are not ray-traced observables and are not physical results.
They provide an interface and identifiability experiment until real geodesic
shooting outputs are supplied.
"""

from .proxy_models import GEODESIC_FEATURES, proxy_observables
from .loader import attach_geodesic_observables, load_shooting_csv

__all__ = ["GEODESIC_FEATURES", "proxy_observables",
           "attach_geodesic_observables", "load_shooting_csv"]
