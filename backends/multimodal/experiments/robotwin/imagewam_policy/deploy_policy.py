"""Forward the historical simulator import to the JAM adapter."""
from importlib import import_module
import sys
_canonical = import_module(__package__.rsplit(".", 1)[0] + ".jam_policy.deploy_policy" if "." in __package__ else "jam_policy.deploy_policy")
sys.modules[__name__] = _canonical
