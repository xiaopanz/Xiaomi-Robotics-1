"""Versioned, opt-in RNG RPCs for local TRACE-MM collection and replay."""
import hashlib
import importlib.metadata
import platform
import random
from pathlib import Path

import numpy as np
import torch

PROTOCOL = 1


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def model_identity(path):
    files = {p.name: file_hash(p) for p in sorted(Path(path).iterdir()) if p.is_file()}
    return {'files': files, 'sha256': hashlib.sha256(repr(sorted(files.items())).encode()).hexdigest()}


def get_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state_all()}


def set_rng(state):
    if len(state['torch_cuda']) != torch.cuda.device_count():
        raise ValueError('CUDA device count differs from recorded server RNG')
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in state['torch_cuda']])


def seed_rng(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def request_hash(request):
    """Hash actual processed model inputs without serializing pickle object IDs."""
    digest = hashlib.sha256()
    for key in sorted(request):
        value = request[key]
        digest.update(key.encode() + b'\0')
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous()
            digest.update(str((str(value.dtype), tuple(value.shape))).encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
        else:
            digest.update(repr(value).encode())
    return digest.hexdigest()


def runtime_identity():
    return {'python': platform.python_version(), 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
            'transformers': importlib.metadata.version('transformers'),
            'flash_attn': importlib.metadata.version('flash_attn'),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}
