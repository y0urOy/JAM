#!/usr/bin/env python3
"""Resolve a training recipe on CPU without importing models or loading weights."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

from jam import ROOT, recipes, paths, base_overrides
from naming import backend_package, canonical_recipe

def compose_recipe(name, data_root=None, model_root=None, cache_root=None, overrides=()):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    r=recipes()[canonical_recipe(name)]
    args=argparse.Namespace(data_root=data_root,model_root=model_root,cache_root=cache_root)
    p=paths(args)
    repo=ROOT/'backends'/r['backend']
    spec=importlib.util.spec_from_file_location('_jam_resolvers',repo/'src'/backend_package(r['backend'])/'utils/config_resolvers.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.register_default_resolvers()
    with initialize_config_dir(config_dir=str(repo/'configs'),version_base='1.3'):
        cfg=compose(config_name='train',overrides=base_overrides(r,p)+list(overrides))
    return cfg

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',required=True)
    p.add_argument('--data-root');p.add_argument('--model-root');p.add_argument('--cache-root')
    p.add_argument('--override',action='append',default=[])
    a=p.parse_args()
    from omegaconf import OmegaConf
    cfg=compose_recipe(a.recipe,a.data_root,a.model_root,a.cache_root,a.override)
    print(OmegaConf.to_yaml(cfg,resolve=True))

if __name__=='__main__':main()
