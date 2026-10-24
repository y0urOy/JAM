"""Compatibility for pre-JAM environment variables; explicit JAM values take precedence."""
import os

def adopt_legacy_environment():
    for prefix in ('IMAGEWAM_', 'FASTWAM_'):
        for key, value in tuple(os.environ.items()):
            if key.startswith(prefix):
                os.environ.setdefault('JAM_' + key[len(prefix):], value)
