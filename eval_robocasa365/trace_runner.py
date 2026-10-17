"""Collect and replay trusted local TRACE-MM pilot episodes.

Examples (from Xiaomi-Robotics-1, in robocasa_365):
  python eval_robocasa365/trace_runner.py collect --output /path/run --episodes 5
  python eval_robocasa365/trace_runner.py replay --episode /path/run/CloseBlenderLid/seed_7
  python eval_robocasa365/trace_runner.py replay --episode ... --start-step 7 --mode policy
"""
from __future__ import annotations

import argparse
from collections import deque
import copy
import gc
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import shutil
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import numpy as np
import torch

import entry
from trace_state import (BlobStore, SUPPORTED_TASKS, controller_digest, digest,
                         integration_state, load_pickle, local_rng, restore_local_rng,
                         restore_snapshot, save_pickle, save_snapshot)

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
SETTINGS = {'obs_history': 4, 'obs_interval': 2, 'replan_steps': 16, 'crop_ratio': .95,
            'split': 'pretrain', 'task_set': 'target50', 'robot_type': 'robocasa365',
            'render_offsamples': 0, 'render_note': 'MSAA disabled for exact RGB replay; differs from upstream renderer default'}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def software_identity():
    files = [*sorted((ROOT/'eval_robocasa365').glob('*.py')),
             *sorted((ROOT/'deploy').glob('*.py'))]
    for repo in ('robocasa', 'robosuite'):
        tree = WORKSPACE/repo/repo
        spec = importlib.util.find_spec(repo)
        if spec is None or not (spec.origin or '').endswith('__init__.py'):
            raise RuntimeError(f'{repo} does not resolve to a regular package (origin {spec and spec.origin}); '
                               'a directory on sys.path shadows the installed package')
        imported = Path(spec.origin).resolve().parent
        if imported != tree.resolve():
            # The interpreter may import an editable install from another checkout; the identity hashes this
            # workspace's copy, so the executing copy must match it file for file.
            for source in sorted(tree.rglob('*.py')):
                other = imported/source.relative_to(tree)
                if not other.is_file() or sha_file(other) != sha_file(source):
                    raise RuntimeError(f'{repo} is imported from {imported}, which differs from {tree} '
                                       f'at {source.relative_to(tree)}')
        files += sorted(tree.rglob('*.py'))
    files += sorted((WORKSPACE/'robosuite/robosuite/controllers/config').rglob('*.json'))
    return {'packages': {p: importlib.metadata.version(p) for p in
                         ('robocasa', 'robosuite', 'mujoco', 'gymnasium', 'numpy', 'torch', 'transformers')},
            'files': {str(p.relative_to(WORKSPACE)): sha_file(p) for p in files},
            'commits': {p: subprocess.check_output(['git', '-C', str(WORKSPACE/p), 'rev-parse', 'HEAD'],
                                                  text=True).strip()
                        for p in ('Xiaomi-Robotics-1', 'robocasa', 'robosuite')}}


def deterministic_counter_regions():
    """RoboCasa's Counter.get_reset_regions de-duplicates the candidate counter-top geoms with set(). xml
    Elements hash by address, so for `loc: left_right`/`any` placements the seeded rng.choice index names a
    random slab: the blender (with the lid and the robot's anchor) lands on either side of the sink in 9 of
    the 50 pretrain layouts used by CloseBlenderLid, differently on every construction. counter.py has no
    other set() call, so give that module an order-preserving de-duplication: one seed, one scene. Archives
    recorded before this shim keep the order their construction happened to draw."""
    import robocasa.models.fixtures.counter as counter
    if 'set' not in vars(counter):
        counter.set = lambda iterable=(): dict.fromkeys(iterable)


def nondeterministic_counter_regions():
    """Restore RoboCasa's own address-ordered de-duplication: replaying an archive recorded before the shim can
    need the construction draw that archive happened to get."""
    import robocasa.models.fixtures.counter as counter
    vars(counter).pop('set', None)


def make_env(task, seed, deterministic_regions=True):
    import gymnasium as gym
    import robocasa  # noqa: F401
    (deterministic_counter_regions if deterministic_regions else nondeterministic_counter_regions)()
    # Delayed GC of old cyclic EGL contexts during a rollout can invalidate the
    # current framebuffer. Collect only before scene construction, never mid-run.
    gc.collect()
    gc.disable()
    # No hidden global RNG differences during procedural scene construction.
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = gym.make('robocasa/' + task, split=SETTINGS['split'], seed=seed)
    obs, _ = entry.reset_env(env, seed)
    # Default MSAA resolves are not pixel-deterministic on this EGL/NVIDIA stack,
    # even for repeated renders of an unchanged scene. Freeze this explicit
    # local-pilot rendering change for all collection and replay branches.
    raw = env.unwrapped.env
    ctx = raw.sim._render_context_offscreen
    ctx.gl_ctx.make_current()
    raw.sim.model.vis.quality.offsamples = SETTINGS['render_offsamples']
    ctx.con.free()
    ctx._set_mujoco_context_and_buffers()
    obs = env.unwrapped.get_observation(raw._get_observations(force_update=True))
    return env, obs


def asset_identity(xml):
    paths = {node.get('file') for node in ET.fromstring(xml).iter() if node.get('file')}
    return {path: sha_file(path) for path in sorted(paths)}


def archive_sources(root, identity):
    for relative in identity['files']:
        dest = root/'source'/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(WORKSPACE/relative,dest)


def verify_archive(path):
    inventory=json.loads((path/'files.sha256.json').read_text())
    for relative, expected in inventory.items():
        if sha_file(path/relative)!=expected:
            raise ValueError(f'Corrupt episode artifact: {relative}')


def new_client(args):
    client = entry.EvalClient(args.model_path, args.host, args.port, 'robocasa365', .95)
    client.client.client_socket.settimeout(120)
    client.client.record_trace = True
    return client


def infer(client, runtime):
    states = entry.sample_history(runtime['state_queue'], 4, 2)
    images = {k: entry.sample_history(q, 4, 2) for k, q in runtime['image_queues'].items()}
    chunk = client.infer(states, images, runtime['instruction'])
    if chunk.ndim != 2 or chunk.shape[1] != 12 or len(chunk) < 16 or not np.isfinite(chunk).all():
        raise ValueError('Invalid XR-1 action chunk')
    return chunk, {'states': states, 'images': images,
                   'observation_indices': [max(0, runtime['step']-6+2*i) for i in range(4)]}


def append_history(runtime, obs):
    runtime['observation'] = copy.deepcopy(obs)
    runtime['state_queue'].append(entry.observation_to_state(obs))
    for key, image in entry.collect_images(obs).items():
        runtime['image_queues'][key].append(image.copy())


def physical_record(env):
    raw = env.unwrapped.env
    return {'integration': integration_state(env), 'controller_sha256': controller_digest(env),
            'qpos': raw.sim.data.qpos.copy(), 'qvel': raw.sim.data.qvel.copy(),
            'actuator_ctrl': raw.sim.data.ctrl.copy(), 'contacts': int(raw.sim.data.ncon)}


def step_env(env, action):
    from robocasa.utils.env_utils import convert_action
    raw = env.unwrapped.env
    original_step = raw.step
    received = []
    def capture(value):
        received.append(np.asarray(value).copy())
        return original_step(value)
    raw.step = capture
    try:
        result = env.step(convert_action(action))
    finally:
        del raw.step
    if len(received) != 1:
        raise RuntimeError('Expected one controller step per policy action')
    return result, received[0]


def collect_episode(args, task, seed, horizon, client, store):
    path = args.output / task / f'seed_{seed}'
    path.mkdir(parents=True, exist_ok=False)
    env, obs = make_env(task, seed)
    writer = imageio.get_writer(path/'rollout.mp4', fps=20)
    events = (path/'events.jsonl').open('w')
    try:
        runtime = {'step': 0, 'observation': obs, 'instruction': obs['annotation.human.task_description'],
                   'state_queue': deque(maxlen=7),
                   'image_queues': {k: deque(maxlen=7) for k in entry.CAMERA_KEYS},
                   'action_queue': deque(), 'chunk_id': -1, 'chunk_offset': 0,
                   'server_rng': client.client.trace_control('seed', seed=args.policy_seed+seed)}
        append_history(runtime, obs)
        (path/'scene.xml').write_text(env.unwrapped.env.sim.model.get_xml())
        write_json(path/'assets.sha256.json', asset_identity((path/'scene.xml').read_text()))
        write_json(path/'scene_meta.json', env.unwrapped.env.get_ep_meta())
        meta = {'schema': 1, 'task': task, 'seed': seed, 'policy_seed': args.policy_seed+seed,
                'horizon': horizon, 'settings': SETTINGS, 'instruction': runtime['instruction'],
                'snapshot_semantics': 'before action at zero-based step t; observations indexed 0..T',
                'status': 'collecting', 'snapshots': [], 'actor': 'frozen_xr1_no_repair',
                'controller': env.unwrapped.env.robots[0].composite_controller_config}
        write_json(path/'episode.json', meta)
        save_pickle(path/'observations/000000.pklz', obs)
        save_pickle(path/'states/000000.pklz', physical_record(env))
        writer.append_data(entry.make_video_frame(obs))
        success, started = False, time.monotonic()
        for t in range(horizon):
            if t in (0, 7) or t % args.snapshot_interval == 0:
                save_snapshot(path/f'snapshots/{t:06d}.pklz', env, runtime, store)
                meta['snapshots'].append(t)
            if not runtime['action_queue']:
                chunk, inputs = infer(client, runtime)
                runtime['chunk_id'] += 1
                runtime['chunk_offset'] = 0
                runtime['server_rng'] = client.client.last_trace['rng_after']
                request = {'step': t, 'chunk': chunk, 'inputs': inputs,
                           'instruction': runtime['instruction'], 'rpc': client.client.last_trace}
                save_pickle(path/f"requests/{runtime['chunk_id']:06d}.pklz", request)
                runtime['action_queue'].extend(chunk[:16].copy())
            action = np.asarray(runtime['action_queue'].popleft(), dtype=np.float32).copy()
            from robocasa.utils.env_utils import convert_action
            (obs, reward, done, truncated, info), received = step_env(env, action)
            record = {'step': t, 'actor': 'xr1', 'policy_action': action,
                      'converted_action': convert_action(action), 'controller_action': received,
                      'chunk_id': runtime['chunk_id'], 'chunk_offset': runtime['chunk_offset'],
                      'reward': float(reward), 'done': bool(done), 'truncated': bool(truncated),
                      'success': bool(info.get('success', False))}
            save_pickle(path/f'actions/{t:06d}.pklz', record)
            runtime['chunk_offset'] += 1
            runtime['step'] = t+1
            append_history(runtime, obs)
            save_pickle(path/f'observations/{t+1:06d}.pklz', obs)
            save_pickle(path/f'states/{t+1:06d}.pklz', physical_record(env))
            events.write(json.dumps({'step':t, 'observation_before':t, 'observation_after':t+1,
                                     'actor':'xr1', 'chunk_id':record['chunk_id'],
                                     'chunk_offset':record['chunk_offset'], 'success':record['success'],
                                     'action_file':f'actions/{t:06d}.pklz'})+'\n')
            events.flush()
            writer.append_data(entry.make_video_frame(obs))
            success = record['success']
            if success or done or truncated:
                break
        save_snapshot(path/f"snapshots/{runtime['step']:06d}.pklz", env, runtime, store)
        meta['snapshots'].append(runtime['step'])
        meta.update(status='complete', success=success, steps=runtime['step'],
                    elapsed_s=time.monotonic()-started, replay_verified=False,
                    termination='success' if success else 'environment_done' if done else
                    'environment_truncated' if truncated else 'horizon')
        write_json(path/'episode.json', meta)
        writer.close()
        events.flush()
        inventory={str(p.relative_to(path)):sha_file(p) for p in sorted(path.rglob('*')) if p.is_file()}
        write_json(path/'files.sha256.json',inventory)
        print(json.dumps({'episode':str(path),'success':success,'steps':runtime['step']}), flush=True)
    finally:
        writer.close(); events.close(); env.close()
    return meta


def collect(args):
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    args.output.mkdir(parents=True, exist_ok=False)
    client = new_client(args)
    try:
        hello = client.client.trace_control('hello')
        if hello['protocol'] != 1 or hello['training']:
            raise ValueError('Requires versioned TRACE server with model in eval mode')
        manifest = {'schema':1, 'created_utc':datetime.now(timezone.utc).isoformat(),
                    'settings':SETTINGS, 'model_path':args.model_path, 'server':hello,
                    'software':software_identity(), 'episode_seed_stride':50,
                    'snapshot_interval':args.snapshot_interval, 'episodes':[],
                    'policy_version':hello['model']['sha256'],
                    'status':'collecting', 'replay_verified':False}
        write_json(args.output/'manifest.json',manifest)
        archive_sources(args.output,manifest['software'])
        store = BlobStore(args.output/'blobs')
        all_tasks = list(TASK_SET_REGISTRY['target50'])
        for task in args.tasks:
            for i in range(args.episodes):
                seed = 7 + all_tasks.index(task)*50 + i
                meta = collect_episode(args,task,seed,args.horizon or get_task_horizon(task),client,store)
                manifest['episodes'].append({'path':f'{task}/seed_{seed}',**meta})
                write_json(args.output/'manifest.json',manifest)
        manifest['status']='complete'
        write_json(args.output/'manifest.json',manifest)
    finally:
        client.close()


def replay(args):
    path = args.episode.resolve()
    root = path.parents[1]
    meta = json.loads((path/'episode.json').read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    verify_archive(path)
    if asset_identity((path/'scene.xml').read_text()) != json.loads((path/'assets.sha256.json').read_text()):
        raise ValueError('Scene asset content changed since collection')
    current = software_identity()
    compatibility = 'exact_source_match'
    if current != manifest['software']:
        # A replay-only bug fix must not invalidate immutable historical data.
        # Permit only changes inside this replay() function; every collection,
        # model, simulation, snapshot and preprocessing definition stays pinned.
        import ast
        replay_files = {
            str(Path(__file__).resolve().relative_to(WORKSPACE)): ('replay',),
            str(Path(__file__).with_name('trace_state.py').resolve().relative_to(WORKSPACE)): ('restore_snapshot',),
        }
        def collection_ast(source, ignored=('replay',)):
            tree = ast.parse(source)
            tree.body = [node for node in tree.body
                         if not (isinstance(node, ast.FunctionDef) and node.name in ignored)]
            return ast.dump(tree, include_attributes=False)
        adjusted = copy.deepcopy(current)
        for relative, ignored in replay_files.items():
            recorded_source = root/'source'/relative
            if (sha_file(recorded_source) != manifest['software']['files'][relative]
                    or collection_ast(recorded_source.read_text(),ignored)
                    != collection_ast((WORKSPACE/relative).read_text(),ignored)):
                raise ValueError('Collection code changed outside replay/restore functions')
            adjusted['files'][relative] = manifest['software']['files'][relative]
        if adjusted != manifest['software']:
            raise ValueError('Software/version drift beyond replay-only changes; use archived collection environment')
        compatibility = 'replay_only_fix_collection_ast_and_all_other_sources_identical'
    if args.start_step not in meta['snapshots'] or args.start_step >= meta['steps']:
        raise ValueError('Choose a saved pre-terminal snapshot from episode.json')
    env, _ = make_env(meta['task'],meta['seed'])
    actual_xml = env.unwrapped.env.sim.model.get_xml()
    recorded_xml = (path/'scene.xml').read_text()
    def canonical_scene(xml):
        tree = ET.fromstring(xml)
        # MuJoCo's process-global mesh cache adds this inferred MIME attribute
        # to saved XML on cache hits. Explicit model/obj and an omitted type
        # for a .obj file are equivalent. No geometry/pose/asset values change.
        for node in tree.iter('mesh'):
            if node.get('file','').lower().endswith('.obj') and node.get('content_type') == 'model/obj':
                del node.attrib['content_type']
        return ET.tostring(tree, encoding='utf-8')
    def scene_structure(xml):
        tree = ET.fromstring(canonical_scene(xml))
        # Positions are restored from the archived full-precision compiled
        # model. Still reject any structural, geometry or asset change.
        for node in tree.iter('body'):
            node.attrib.pop('pos', None)
        # Debug/collision geom colours are also sampled during construction.
        # The archived model restores geom_rgba exactly before any replay step.
        for node in tree.iter('geom'):
            node.attrib.pop('rgba',None)
        return ET.tostring(tree,encoding='utf-8')
    scene_reconstructed_from_metadata=False
    if scene_structure(recorded_xml) != scene_structure(actual_xml):
        # Some composite tasks consume/reset procedural RNG during setup.
        # Restore the archived layout/style/object selection explicitly rather
        # than relying on the seed to regenerate the same kitchen.
        raw=env.unwrapped.env
        raw.set_ep_meta(json.loads((path/'scene_meta.json').read_text()))
        entry.reset_env(env,meta['seed'])
        ctx=raw.sim._render_context_offscreen
        ctx.gl_ctx.make_current()
        raw.sim.model.vis.quality.offsamples=SETTINGS['render_offsamples']
        ctx.con.free()
        ctx._set_mujoco_context_and_buffers()
        actual_xml=raw.sim.model.get_xml()
        scene_reconstructed_from_metadata=True
    if scene_structure(recorded_xml) != scene_structure(actual_xml):
        mismatch_path=(args.report or path/'replay_failure.json').with_suffix('.scene_mismatch.xml')
        mismatch_path.parent.mkdir(parents=True,exist_ok=True)
        mismatch_path.write_text(actual_xml)
        env.close()
        raise ValueError('Recreated scene XML differs; assets/configuration drift')
    client = None
    writer = None
    report = {'mode':args.mode,'episode':str(path),'start_step':args.start_step,
              'checked_steps':0,'request_checks':0,'max_state_error':0.,'max_action_error':0.,
              'max_pixel_error':0,'pixel_changed_fraction_max':0.,'mismatches':[], 'passed':False,
              'equality_requirement':'exact physics/actions/controller/proprioception/RGB',
              'uses_recorded_future_observations':args.mode=='policy-recorded-observations',
              'software_compatibility':compatibility,
              'replay_source_sha256':sha_file(__file__),
              'restore_source_sha256':sha_file(Path(__file__).with_name('trace_state.py')),
              'scene_xml_byte_identical':recorded_xml==actual_xml,
              'scene_xml_canonical_sha256':hashlib.sha256(canonical_scene(recorded_xml)).hexdigest(),
              'compiled_model_restored_from_archive':True,
              'scene_reconstructed_from_metadata':scene_reconstructed_from_metadata}
    try:
        if args.mode in ('policy','policy-recorded-observations'):
            args.model_path = manifest['model_path']
            client = new_client(args)
            hello = client.client.trace_control('hello')
            if hello != manifest['server']:
                raise ValueError('Server model/runtime identity differs')
        runtime = restore_snapshot(path/f'snapshots/{args.start_step:06d}.pklz',env,BlobStore(root/'blobs'))
        expected_start = load_pickle(path/f'states/{args.start_step:06d}.pklz')
        if not np.array_equal(integration_state(env),expected_start['integration']):
            raise ValueError('Restored snapshot integration state does not match trace')
        if controller_digest(env) != expected_start['controller_sha256']:
            raise ValueError('Restored controller/fixture state does not match trace')
        if digest(runtime['observation']) != digest(load_pickle(path/f'observations/{args.start_step:06d}.pklz')):
            raise ValueError('Snapshot observation does not match trace')
        if client:
            client.client.trace_control('set_rng', state=runtime['server_rng'])
        if args.video:
            args.video.parent.mkdir(parents=True,exist_ok=True)
            writer=imageio.get_writer(args.video,fps=20)
            writer.append_data(entry.make_video_frame(runtime['observation']))
        end = min(meta['steps'],args.start_step+args.steps) if args.steps else meta['steps']
        for t in range(args.start_step,end):
            recorded = load_pickle(path/f'actions/{t:06d}.pklz')
            if client:
                if not runtime['action_queue']:
                    chunk, inputs = infer(client,runtime)
                    runtime['chunk_id'] += 1
                    reference = load_pickle(path/f"requests/{runtime['chunk_id']:06d}.pklz")
                    report['request_checks'] += 1
                    if client.client.last_trace['request_sha256'] != reference['rpc']['request_sha256']:
                        report['mismatches'].append({'step':t,'kind':'processed_policy_input'})
                    if not np.array_equal(chunk,reference['chunk']):
                        report['mismatches'].append({'step':t,'kind':'proposed_action_chunk',
                                                     'max_error':float(np.max(np.abs(chunk-reference['chunk'])))})
                    if any(digest(client.client.last_trace[k]) != digest(reference['rpc'][k])
                           for k in ('rng_before','rng_after')):
                        report['mismatches'].append({'step':t,'kind':'server_rng'})
                    runtime['action_queue'].extend(chunk[:16])
                action = runtime['action_queue'].popleft()
            else:
                action = recorded['policy_action']
            err=float(np.max(np.abs(action-recorded['policy_action'])))
            report['max_action_error']=max(report['max_action_error'],err)
            if err:
                report['mismatches'].append({'step':t,'kind':'executed_policy_action','max_error':err})
            (obs,reward,done,truncated,info), received=step_env(env,action)
            if not np.array_equal(received,recorded['controller_action']):
                report['mismatches'].append({'step':t,'kind':'controller_input'})
            expected_state=load_pickle(path/f'states/{t+1:06d}.pklz')
            state_error=float(np.max(np.abs(integration_state(env)-expected_state['integration'])))
            report['max_state_error']=max(report['max_state_error'],state_error)
            if state_error != 0:
                report['mismatches'].append({'step':t,'kind':'integration_state','max_error':state_error})
            if controller_digest(env) != expected_state['controller_sha256']:
                report['mismatches'].append({'step':t,'kind':'controller_or_fixture_state'})
            expected_obs=load_pickle(path/f'observations/{t+1:06d}.pklz')
            for key in expected_obs:
                x,y=expected_obs[key],obs[key]
                if key.startswith('video.'):
                    error=int(np.max(np.abs(x.astype(np.int16)-y.astype(np.int16))))
                    fraction=float(np.mean(x!=y))
                    report['max_pixel_error']=max(report['max_pixel_error'],error)
                    report['pixel_changed_fraction_max']=max(report['pixel_changed_fraction_max'],fraction)
                    if error:
                        report['mismatches'].append({'step':t,'kind':'camera','key':key,'max_error':error})
                elif not np.array_equal(x,y):
                    report['mismatches'].append({'step':t,'kind':'observation','key':key})
            if (float(reward),bool(done),bool(truncated),bool(info.get('success',False))) != (
                    recorded['reward'],recorded['done'],recorded['truncated'],recorded['success']):
                report['mismatches'].append({'step':t,'kind':'outcome'})
            report['checked_steps'] += 1
            runtime['step']=t+1
            append_history(runtime,expected_obs if args.mode=='policy-recorded-observations' else obs)
            if writer:
                writer.append_data(entry.make_video_frame(obs))
            if report['mismatches']:
                break
        report['passed']=not report['mismatches'] and report['checked_steps']==end-args.start_step
    finally:
        if writer: writer.close()
        if client: client.close()
        env.close()
        dest=args.report or path/f'replay_{args.mode}_{args.start_step:06d}.json'
        write_json(dest,report)
    print(json.dumps(report),flush=True)
    if not report['passed']:
        raise RuntimeError('Replay did not meet exact equality checks; inspect report')


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    collect_parser=sub.add_parser('collect')
    collect_parser.add_argument('--output',type=Path,required=True)
    collect_parser.add_argument('--episodes',type=int,default=5)
    collect_parser.add_argument('--tasks',nargs='+',choices=sorted(SUPPORTED_TASKS),
                                default=['CloseBlenderLid','CoffeeSetupMug'])
    collect_parser.add_argument('--horizon',type=int)
    collect_parser.add_argument('--snapshot-interval',type=int,default=128)
    collect_parser.add_argument('--policy-seed',type=int,default=700000)
    replay_parser=sub.add_parser('replay')
    replay_parser.add_argument('--episode',type=Path,required=True)
    replay_parser.add_argument('--start-step',type=int,default=0)
    replay_parser.add_argument('--steps',type=int,default=0,help='0 = through recorded episode end')
    replay_parser.add_argument('--mode',choices=['actions','policy','policy-recorded-observations'],default='actions')
    replay_parser.add_argument('--video',type=Path)
    replay_parser.add_argument('--report',type=Path)
    for p in (collect_parser,replay_parser):
        p.add_argument('--model-path',default=str(ROOT/'checkpoints/Xiaomi-Robotics-1-RoboCasa365'))
        p.add_argument('--host',default='localhost')
        p.add_argument('--port',type=int,default=10086)
    args=parser.parse_args()
    if args.command=='collect' and (args.episodes<1 or args.snapshot_interval<1 or (args.horizon is not None and args.horizon<1)):
        parser.error('episodes, snapshot interval and horizon must be positive')
    if args.command=='replay' and (args.steps<0 or args.start_step<0):
        parser.error('replay start step and step count must be nonnegative')
    return args


if __name__=='__main__':
    os.environ.setdefault('MUJOCO_GL','egl')
    args=parse_args()
    collect(args) if args.command=='collect' else replay(args)
