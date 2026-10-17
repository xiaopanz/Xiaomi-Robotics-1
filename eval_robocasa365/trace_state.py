"""Exact native MuJoCo + Python runtime checkpoints for the two TRACE pilots.

Native MjData pickles include the model and solver/contact caches. Content-addressed
blocks deduplicate the large immutable model across checkpoints. Load only trusted
local archives. These snapshots intentionally require the recorded software/assets.
"""
from __future__ import annotations

import copy
import ctypes
import hashlib
import json
import pickle
import random
import zlib
from collections import deque
from pathlib import Path

import numpy as np
import torch

SUPPORTED_TASKS = {'CloseBlenderLid', 'CoffeeSetupMug'}
SCHEMA = 1


class BlobStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, raw):
        chunks = []
        for offset in range(0, len(raw), 1024 * 1024):
            block = raw[offset:offset + 1024 * 1024]
            key = hashlib.sha256(block).hexdigest()
            dest = self.root / (key + '.z')
            if not dest.exists():
                temporary = dest.with_suffix('.tmp')
                temporary.write_bytes(zlib.compress(block, 1))
                temporary.replace(dest)
            chunks.append(key)
        return {'size': len(raw), 'chunks': chunks}

    def get(self, reference):
        pieces = []
        for key in reference['chunks']:
            if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
                raise ValueError('Invalid blob digest')
            block = zlib.decompress((self.root / (key + '.z')).read_bytes())
            if hashlib.sha256(block).hexdigest() != key:
                raise ValueError(f'Corrupt snapshot block: {key}')
            pieces.append(block)
        raw = b''.join(pieces)
        if len(raw) != reference['size']:
            raise ValueError('Snapshot size mismatch')
        return raw


def save_pickle(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_bytes(zlib.compress(pickle.dumps(value, protocol=5), 1))
    temp.replace(path)


def load_pickle(path):
    return pickle.loads(zlib.decompress(Path(path).read_bytes()))


def local_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state()}


def restore_local_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])


def objects(env):
    """Stable live-object paths; reconstruction preserves simulator references."""
    raw = env.unwrapped.env
    result = {'task': raw, 'gym': env.unwrapped}
    wrapper = env
    i = 0
    while wrapper is not env.unwrapped:
        result[f'wrapper/{i}'] = wrapper
        wrapper = wrapper.env
        i += 1
    for i, robot in enumerate(raw.robots):
        prefix = f'robot/{i}'
        result[prefix] = robot
        result[prefix + '/composite'] = robot.composite_controller
        for key, value in robot.composite_controller.part_controllers.items():
            result[prefix + '/controller/' + key] = value
        for key, value in robot.gripper.items():
            result[prefix + '/gripper/' + key] = value
    for key, value in raw.fixtures.items():
        result['fixture/' + key] = value
    for key, value in raw._observables.items():
        result['observable/' + key] = value
    # Include buffer/interpolator objects nested inside controllers and robots.
    seen = {id(v) for v in result.values()}

    def visit(value, path):
        if id(value) in seen:
            return
        module = type(value).__module__
        if module.startswith(('robosuite.utils.buffers', 'robosuite.utils.interpolators')):
            seen.add(id(value))
            result[path] = value
        elif isinstance(value, dict):
            for k, v in value.items():
                visit(v, path + '/' + str(k))
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                visit(v, path + '/' + str(i))

    for path, obj in list(result.items()):
        for key, value in vars(obj).items():
            visit(value, path + '/' + key)
    return result


class UnsupportedValue(Exception):
    pass


def native_arrays(env):
    sim = env.unwrapped.env.sim
    result = {}
    for owner, obj in [('data', sim.data._data), ('model', sim.model._model)]:
        for key in dir(obj):
            if key.startswith('_'):
                continue
            value = getattr(obj, key)
            if isinstance(value, np.ndarray) and value.size and value.flags.c_contiguous:
                result[owner + '.' + key] = value
    return result


def encode(value, refs, memo, arrays):
    if id(value) in refs:
        return ('ref', refs[id(value)])
    if isinstance(value, np.random.Generator):
        return ('rng', copy.deepcopy(value.bit_generator.state))
    if value is None or isinstance(value, (str, bytes, bool, int, float, np.generic)):
        return ('value', value)
    if isinstance(value, np.ndarray) and not value.dtype.hasobject:
        if id(value) in memo:
            return ('alias', memo[id(value)])
        token = len(memo)
        memo[id(value)] = token
        for name, base in arrays.items():
            if value.size and np.shares_memory(value, base):
                offset = value.ctypes.data - base.ctypes.data
                return ('view', (token, name, offset, value.shape, value.strides, value.dtype.str))
        return ('array', (token, value.copy()))
    if isinstance(value, dict):
        return ('dict', [(encode(k, refs, memo, arrays), encode(v, refs, memo, arrays)) for k, v in value.items()])
    if isinstance(value, (list, tuple, set, deque)):
        return (type(value).__name__, [encode(v, refs, memo, arrays) for v in value],
                value.maxlen if isinstance(value, deque) else None)
    raise UnsupportedValue(type(value).__qualname__)


def decode(node, objs, memo, arrays, existing=None):
    kind, value = node[:2]
    if kind == 'value':
        return copy.deepcopy(value)
    if kind == 'ref':
        return objs[value]
    if kind == 'alias':
        return memo[value]
    if kind == 'array':
        token, array = value
        memo[token] = array.copy()
        return memo[token]
    if kind == 'view':
        token, name, offset, shape, strides, dtype = value
        memo[token] = np.ndarray(shape, dtype=dtype, buffer=arrays[name], offset=offset, strides=strides)
        return memo[token]
    if kind == 'rng':
        if not isinstance(existing, np.random.Generator):
            bitgen = getattr(np.random, value['bit_generator'])()
            existing = np.random.Generator(bitgen)
        existing.bit_generator.state = value
        return existing
    if kind == 'dict':
        return {decode(k, objs, memo, arrays): decode(v, objs, memo, arrays) for k, v in value}
    values = [decode(v, objs, memo, arrays) for v in value]
    if kind == 'deque':
        return deque(values, maxlen=node[2])
    return {'list': list, 'tuple': tuple, 'set': set}[kind](values)


def python_state(env):
    objs = objects(env)
    refs = {id(value): key for key, value in objs.items()}
    memo, arrays = {}, native_arrays(env)
    states, omitted = {}, {}
    for path, obj in objs.items():
        attrs = {}
        for key, value in vars(obj).items():
            try:
                previous = memo.copy()
                attrs[key] = encode(value, refs, memo, arrays)
            except UnsupportedValue:
                memo = previous
                omitted[path + '.' + key] = type(value).__qualname__
        states[path] = {'attrs': attrs, 'all_keys': list(vars(obj)),
                        'class': type(obj).__module__ + '.' + type(obj).__qualname__}
    return {'objects': states, 'omitted_structural_attributes': omitted}


def restore_python(env, state):
    objs = objects(env)
    memo, arrays = {}, native_arrays(env)
    if set(objs) != set(state['objects']):
        raise ValueError('Snapshot Python object graph does not match recreated scene')
    for path, record in state['objects'].items():
        obj = objs[path]
        if record['class'] != type(obj).__module__ + '.' + type(obj).__qualname__:
            raise ValueError(f'Object class mismatch: {path}')
        for key in set(vars(obj)) - set(record['all_keys']):
            delattr(obj, key)
        for key, value in record['attrs'].items():
            setattr(obj, key, decode(value, objs, memo, arrays, vars(obj).get(key)))


def save_snapshot(path, env, runtime, store):
    raw = env.unwrapped.env
    payload = {'schema': SCHEMA, 'native': store.put(pickle.dumps(raw.sim.data._data, protocol=5)),
               'python': python_state(env), 'runtime': copy.deepcopy(runtime), 'rng': local_rng()}
    save_pickle(path, payload)
    return payload


def restore_snapshot(path, env, store):
    payload = load_pickle(path)
    if payload['schema'] != SCHEMA:
        raise ValueError('Unsupported snapshot schema')
    raw = env.unwrapped.env
    data = pickle.loads(store.get(payload['native']))
    if (data.model.nq, data.model.nv, data.model.nbody, data.model.ngeom) != (
            raw.sim.model.nq, raw.sim.model.nv, raw.sim.model.nbody, raw.sim.model.ngeom):
        raise ValueError('Native snapshot topology does not match recreated environment')
    # MuJoCo 3.3.1's Python bindings omit mj_copyData. Use the matching bundled
    # C API to preserve the destination allocation and all live NumPy views.
    # Replacing the model breaks the renderer's cached scene/texture bindings.
    import mujoco
    library = Path(mujoco.__file__).parent / ('libmujoco.so.' + mujoco.__version__)
    lib = ctypes.CDLL(str(library))
    lib.mj_copyData.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.mj_copyData.restype = ctypes.c_void_p
    current_model = raw.sim.model._model
    for key in dir(current_model):
        if key.startswith('n') and isinstance(getattr(current_model,key), int):
            if getattr(current_model,key) != getattr(data.model,key):
                raise ValueError(f'Compiled model layout differs: {key}')
    if current_model.names != data.model.names:
        raise ValueError('Compiled model named entities differ')
    # The archived compiled model is authoritative, including static fixture
    # positions. Procedural placement dictionaries can retain process history.
    # Copy into the existing allocation to keep controllers and renderer bound.
    lib.mj_copyModel.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.mj_copyModel.restype = ctypes.c_void_p
    lib.mj_copyModel(current_model._address, data.model._address)
    for fixture in raw.fixtures.values():
        body = current_model.body(fixture.root_body)
        fixture._obj.set('pos', ' '.join(str(x) for x in body.pos))
    lib.mj_copyData(raw.sim.data._data._address, raw.sim.model._model._address, data._address)
    # Do not call mj_forward: it changes lagged derived quantities/caches.
    restore_python(env, payload['python'])
    # Sink lazily caches XML elements when water/handle checks first run.
    # Schema 1 deliberately omits structural Element objects, so a recreated
    # sink may still hold None where the recorded sink held its XML reference.
    # Resolve only these source-defined structural caches from the frozen XML;
    # preserve the recorded numeric state and the full digest comparison.
    from robocasa.models.fixtures.sink import Sink
    from robocasa.models.fixtures.fridge import Fridge
    for fixture_name, fixture in raw.fixtures.items():
        # Fridge.check_rack_contact temporarily replaces get_int_sites, then
        # writes the original class-bound method back to the instance. A cold
        # constructor lacks that instance key; schema 1 records its method
        # marker but omits the callable. Rebind the pinned class implementation.
        method_key = 'fixture/' + fixture_name + '.get_int_sites'
        if (isinstance(fixture, Fridge)
                and payload['python']['omitted_structural_attributes'].get(method_key) == 'method'):
            fixture.get_int_sites = type(fixture).get_int_sites.__get__(fixture, type(fixture))
        if not isinstance(fixture, Sink):
            continue
        for cache_name in ('_water_site', '_handle_joint'):
            omitted_key = 'fixture/' + fixture_name + '.' + cache_name
            omitted_type = payload['python']['omitted_structural_attributes'].get(omitted_key)
            if omitted_type == 'Element' and getattr(fixture, cache_name) is None:
                element = getattr(fixture, cache_name[1:])
                if type(element).__qualname__ != omitted_type or getattr(fixture, cache_name) is not element:
                    raise ValueError(f'Cannot reconstruct sink XML cache: {omitted_key}')
                owner = current_model.site if cache_name == '_water_site' else current_model.joint
                owner(element.get('name'))  # The XML reference must exist in the archived model.
    # PandaOmron's WheeledRobot.control pushes robot.torques, which is None for
    # its composite controller, into DeltaBuffer. np.array(None) has object
    # dtype and is omitted by the schema-1 numeric serializer. Reconstruct only
    # this source-defined sentinel; never guess an omitted numerical state.
    for index, robot in enumerate(raw.robots):
        prefix = f'robot/{index}'
        record = payload['python']['objects'][prefix + '/recent_torques']
        for name in ('current', 'last'):
            key = prefix + '/recent_torques.' + name
            if key in payload['python']['omitted_structural_attributes']:
                # Reset itself can execute settling control ticks, so these
                # sentinels may already be present at policy step zero.
                if robot.name != 'PandaOmron' or robot.torques is not None:
                    raise ValueError(f'Cannot reconstruct omitted torque-buffer state: {key}')
                setattr(robot.recent_torques, name, np.array(None, dtype=object))
    restore_local_rng(payload['rng'])
    return payload['runtime']


def integration_state(env):
    import mujoco
    sim = env.unwrapped.env.sim
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(sim.model._model, spec))
    mujoco.mj_getState(sim.model._model, sim.data._data, state, spec)
    return state


def digest(value):
    """Stable digest for nested numerical evidence, including controller caches."""
    h = hashlib.sha256()

    def add(v):
        if isinstance(v, np.ndarray):
            h.update(str((str(v.dtype), v.shape)).encode())
            h.update(np.ascontiguousarray(v).tobytes())
        elif isinstance(v, torch.Tensor):
            add(v.cpu().numpy())
        elif isinstance(v, dict):
            for key in sorted(v, key=repr):
                add(key); add(v[key])
        elif isinstance(v, (tuple, list, deque)):
            h.update(type(v).__name__.encode())
            for item in v:
                add(item)
        elif isinstance(v, set):
            for item in sorted(v, key=repr):
                add(item)
        else:
            h.update(repr(v).encode() + b'\0')
    add(value)
    return h.hexdigest()


def controller_digest(env):
    state = python_state(env)['objects']
    return digest({k: v for k, v in state.items() if k.startswith(('robot/', 'fixture/'))})
