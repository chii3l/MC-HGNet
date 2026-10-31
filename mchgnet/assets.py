"""Paths are supplied by the user; importing this module never downloads files."""
import hashlib
import json
import os
from pathlib import Path


def asset(name):
    override = os.environ.get('MCHGNET_ASSET_CONFIG')
    if override:
        mapping = json.loads(Path(override).read_text('utf-8'))
        if name in mapping:
            return Path(mapping[name]).expanduser().resolve()
    return Path(os.environ.get('MCHGNET_ASSETS', 'assets')).expanduser().resolve()/name


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()
