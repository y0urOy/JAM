from pathlib import Path
import argparse
import json
import subprocess
import sys

import pytest
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from jam import recipes, parser, command_plan
from compose import compose_recipe
import evaluate_libero
from evaluate_libero import task_list

LIBERO_RECIPES=['libero-wan-jam','libero-ovis-jam','libero-qwen-jam','libero-dino-b-jam','libero-dino-l-jam','libero-flux-jam']
ROBOTWIN_RECIPES=['robotwin-wan-jam','robotwin-ovis-jam','robotwin-flux-jam','robotwin-qwen-jam']

def test_release_names_exactly_the_jam_recipes():
    assert list(recipes())==LIBERO_RECIPES+ROBOTWIN_RECIPES
    fingerprints=json.loads((ROOT/'reproduction/config_fingerprints.json').read_text())['sha256']
    assert list(fingerprints)==list(recipes())
    assert {recipes()[name]['benchmark'] for name in LIBERO_RECIPES}=={'libero'}
    assert {recipes()[name]['benchmark'] for name in ROBOTWIN_RECIPES}=={'robotwin'}

@pytest.mark.parametrize('name',list(recipes()))
def test_recipe_configs_have_no_missing_fields(name):
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

def planned(mode,name,*extra):
    plan,_=command_plan(parser().parse_args([mode,name,*extra]))
    return Path(plan['cwd']),plan['commands'][0]

@pytest.mark.parametrize('name',list(recipes()))
def test_launcher_entrypoints_exist(name,tmp_path):
    r=recipes()[name]
    repo=ROOT/'backends'/r['backend']
    cwd,command=planned('train',name)
    assert cwd==repo and (repo/'scripts/train.py').is_file() and 'scripts/train.py' in command
    accelerate=OmegaConf.load(ROOT/f'environments/accelerate_zero{r["zero_stage"]}.yaml')
    deepspeed=json.loads((repo/accelerate.deepspeed_config.deepspeed_config_file).read_text())
    assert deepspeed['zero_optimization']['stage']==r['zero_stage'] and 'gradient_clipping' not in deepspeed
    if r['family'] in ('wan','flux','dino'):
        _,command=planned('text-cache',name)
        assert (repo/command[1]).is_file(), command[1]
    else:
        with pytest.raises(ValueError,match='no .*cache is needed'):
            planned('text-cache',name)
    if r['backend']=='wan':
        _,command=planned('cache',name)
        assert (repo/command[1]).is_file() and any(x.startswith('++extract.out_roots=') for x in command)
    else:
        with pytest.raises(ValueError,match='Wan foresight recipes'):
            planned('cache',name)
    if r['benchmark']=='robotwin':
        _,command=planned('eval',name,'--checkpoint','w.pt','--stats','s.json')
        commands=[command]
        assert (repo/'experiments/robotwin/jam_policy/deploy_policy.py').is_file()
    else:
        commands=[evaluate_libero.build_plan(argparse.Namespace(recipe=name,benchmark=benchmark,
            checkpoint='w.pt',stats='s.json',gpus='0',output=str(tmp_path/benchmark),
            data_root=None,model_root=None,cache_root=None,override=[]))['workers'][0]['command']
            for benchmark in ('libero','plus')]
    for command in commands:
        assert (repo/command[1]).is_file(), command[1]
        assert (repo/'configs'/(command[command.index('--config-name')+1]+'.yaml')).is_file()

@pytest.mark.parametrize('family,dims',[('wan',[7]),('ovis',[7,14]),('flux',[7,14])])
def test_action_initialization_entrypoints_exist(family,dims):
    for dim in dims:
        out=subprocess.run([sys.executable,str(ROOT/'tools/init_actions.py'),family,'--action-dim',str(dim),
                            '--model-root','/models','--dry-run'],capture_output=True,text=True,check=True)
        plan=json.loads(out.stdout)
        cwd,command=Path(plan['cwd']),plan['command']
        assert (cwd/command[1]).is_file()
        assert (cwd/command[command.index('--model-config')+1]).is_file()
        assert plan['output'].startswith('/models/init/')

def strings(value):
    if isinstance(value,dict):
        value=list(value.values())
    if isinstance(value,list):
        for item in value:
            yield from strings(item)
    elif isinstance(value,str):
        yield value

def test_downloadable_assets_are_exactly_those_the_recipes_use():
    assets=json.loads((ROOT/'reproduction/assets.json').read_text())['assets']
    roots={'model':'/models/','dataset':'/data/'}
    used=set()
    for name in recipes():
        for mode in ('train','text-cache'):
            try:
                _,command=planned(mode,name,'--model-root','/models','--data-root','/data')
            except ValueError:
                continue
            for argument in command:
                if argument.startswith('++') and '=' in argument:
                    used.update(strings(json.loads(argument.split('=',1)[1])))
        # Wan resolves its base-model and tokenizer IDs below DIFFSYNTH_MODEL_BASE_PATH.
        cfg=OmegaConf.to_container(compose_recipe(name,'/data','/models','/cache'),resolve=True)
        used.update('/models/'+s for s in strings(cfg) if s.startswith('Wan-AI/'))
    # tools/setup_simulators.py unpacks the simulator assets from the data root.
    import setup_simulators
    used.update('/data/'+x for x in [setup_simulators.LIBERO_PLUS_ASSETS]+setup_simulators.ROBOTWIN_ASSETS)
    def covered(path,item):
        prefix=roots[item.get('repo_type','model')]+item['directory']
        return path==prefix or path.startswith(prefix+'/')
    for path in used:
        if path.startswith(('/models/','/data/')) and not path.startswith('/models/init/'):
            assert any(covered(path,item) for item in assets.values()), path
    for key,item in assets.items():
        assert any(covered(path,item) for path in used), key

def test_one_environment_pins_every_requirement():
    out=subprocess.run([sys.executable,str(ROOT/'tools/bootstrap.py'),'--venv','/nonexistent/jam-env','--libero','--dry-run'],
                       capture_output=True,text=True,check=True).stdout
    commands=[line.split() for line in out.splitlines() if not line.startswith('#')]
    norm=lambda name:name.lower().replace('_','-').replace('.','-')
    lines=lambda name:[l.strip() for l in (ROOT/'environments'/name).read_text().splitlines() if l.strip() and not l.startswith('#')]
    pins=dict((norm(l.split('==')[0]),l.split('==')[1]) for l in lines('constraints.txt'))
    for line in lines('requirements.txt')+lines('libero.txt'):
        name,_,version=line.partition('==')
        assert norm(name) in pins and version in ('',pins[norm(name)]), line
    assert sum(name.startswith('opencv') for name in pins)==1
    installs=[c for c in commands if c[1:4]==['-m','pip','install']]
    assert all(c[-2:]==['-c',str(ROOT/'environments/constraints.txt')] for c in installs if '--index-url' not in c and '-e' not in c)
    assert '--no-build-isolation' in installs[-1] and {'deepspeed','flash-attn','causal-conv1d'}<={c.split('==')[0] for c in installs[-1]}
    assert commands[-1][1:]==['-m','pip','check']

def test_reject_batch_that_changes_recipe():
    a=parser().parse_args(['train','libero-wan-jam','--gpus-per-node','3'])
    with pytest.raises(ValueError,match='global batch'):command_plan(a)

def test_reject_cross_robot_evaluation():
    a=parser().parse_args(['eval','libero-wan-jam','--benchmark','robotwin','--checkpoint','x.pt','--stats','s.json'])
    with pytest.raises(ValueError,match='matching training recipe'):command_plan(a)
    a=parser().parse_args(['eval','robotwin-qwen-jam','--benchmark','plus','--checkpoint','x.pt','--stats','s.json'])
    with pytest.raises(ValueError,match='matching training recipe'):command_plan(a)

def test_fixed_plus_protocol():
    rows=task_list('plus')
    assert len(rows)==len(set(rows))==1506
    assert {s:sum(x.startswith(s+',') for x in rows) for s in ['libero_10','libero_goal','libero_spatial','libero_object']}=={
        'libero_10':378,'libero_goal':389,'libero_spatial':361,'libero_object':378}
    assert len(task_list('libero'))==40
