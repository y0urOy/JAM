from pathlib import Path
import argparse
import importlib.util
import json
import sys

import pytest
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from jam import recipes, parser, command_plan
from compose import compose_recipe
from evaluate_libero import task_list
from paired_statistics import statistics

@pytest.mark.parametrize('name',list(recipes()))
def test_paper_configs_have_no_missing_fields(name):
    cfg=compose_recipe(name,data_root='/dataset with spaces',model_root='/model with spaces',cache_root='/cache with spaces')
    data=OmegaConf.to_container(cfg,resolve=True,throw_on_missing=True)
    assert data['max_steps']==recipes()[name]['steps']
    assert data['wandb']['enabled'] is False
    assert str(data['data']['train']['dataset_dirs'][0]).startswith('/dataset with spaces/')

@pytest.mark.parametrize('name',list(recipes()))
def test_distributed_training_preserves_global_batch(name):
    a=parser().parse_args(['train',name,'--nodes','8','--gpus-per-node','1','--micro-batch','16','--dry-run'])
    plan,_=command_plan(a)
    command=plan['commands'][0]
    accum=int(next(s.split('=',1)[1] for s in command if s.startswith('++gradient_accumulation_steps=')))
    assert accum*8*16==recipes()[name]['global_batch']
    assert 'PBS_JOBID' not in ' '.join(command)

def test_reject_batch_that_changes_recipe():
    a=parser().parse_args(['train','wan-jam','--gpus-per-node','3'])
    with pytest.raises(ValueError,match='global batch'):command_plan(a)

def test_reject_cross_robot_evaluation():
    a=parser().parse_args(['eval','wan-jam','--benchmark','robotwin','--checkpoint','x.pt','--stats','s.json'])
    with pytest.raises(ValueError,match='matching training recipe'):command_plan(a)

def test_fixed_plus_protocol():
    rows=task_list('plus')
    assert len(rows)==len(set(rows))==1506
    assert {s:sum(x.startswith(s+',') for x in rows) for s in ['libero_10','libero_goal','libero_spatial','libero_object']}=={
        'libero_10':378,'libero_goal':389,'libero_spatial':361,'libero_object':378}
    assert len(task_list('libero'))==40

def test_statistics_identical_policies():
    x={(f't{i}',str(j)):j%2 for i in range(5) for j in range(4)}
    s=statistics(x,x,resamples=1000)
    assert s['task_mean_delta_pp']==0 and s['bootstrap95_pp']==[0,0]
    assert s['mcnemar_two_sided_p']==1

def test_statistics_reject_unpaired_episodes():
    with pytest.raises(ValueError,match='identical'):
        statistics({('x','0'):1},{('x','1'):1},resamples=100)

def test_statistics_full_improvement_and_repeatability():
    a={(f't{i}','0'):0 for i in range(10)}
    b={k:1 for k in a}
    s=statistics(a,b,resamples=1000,seed=42)
    assert s==statistics(a,b,resamples=1000,seed=42)
    assert s['bootstrap95_pp']==[100,100]
    assert s['mcnemar_two_sided_p']==pytest.approx(2/(2**10))
