"""JAM namespaces, with narrow compatibility for older recorded recipes."""
from functools import lru_cache
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
BACKEND_PACKAGES = {'wan': 'jam_wan', 'multimodal': 'jam_multimodal'}
BACKEND_ALIASES = {'fastwam': 'wan', 'imagewam': 'multimodal'}


def canonical_backend(name):
    return BACKEND_ALIASES.get(name, name)


def backend_package(name):
    return BACKEND_PACKAGES[canonical_backend(name)]


@lru_cache(maxsize=1)
def migration():
    return json.loads((ROOT / 'reproduction/naming.json').read_text())


def canonical_reference(value):
    """Translate exact old task/model names and dotted Python targets only.

    No numeric settings, external asset IDs or arbitrary substrings are changed.
    """
    if not isinstance(value, str):
        return value
    mapping = migration()
    names = {**mapping['config_names'], **mapping['modules'], **mapping['symbols']}
    def translate(match):
        token = match[0]
        if token in names:
            return names[token]
        module, dot, symbol = token.rpartition('.')
        if dot and module in mapping['modules']:
            return mapping['modules'][module] + '.' + mapping['symbols'].get(symbol, symbol)
        return token
    return re.sub(r'[A-Za-z_][A-Za-z0-9_.-]*', translate, value)


def canonical_recipe_spec(spec):
    if not isinstance(spec, dict):
        return spec
    result = dict(spec)
    result['backend'] = canonical_backend(result.get('backend'))
    result['task'] = canonical_reference(result.get('task'))
    result['overrides'] = [canonical_reference(x) for x in result.get('overrides', [])]
    return result


def adopt_legacy_environment(env, backend):
    prefixes = ('FASTWAM_',) if canonical_backend(backend) == 'wan' else ('IMAGEWAM_', 'FASTWAM_')
    for prefix in prefixes:
        for key, value in tuple(env.items()):
            if key.startswith(prefix):
                env.setdefault('JAM_' + key[len(prefix):], value)
    return env
