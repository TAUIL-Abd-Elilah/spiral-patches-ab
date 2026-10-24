"""Merge Spiral config override files (left to right, then --set key=<json value>) and print them as one compact JSON
line for FIT_SPIRAL_CONFIG_OVERRIDES. Every key and value is validated by the fitter's own Config class (villa
0fb38c45 config.py), which the CLI itself does not do for env overrides.

usage: make_overrides.py <spiral dir> <a.json> [<b.json> ...] [--set optimizer_num_training_steps=200 ...]
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('spiral_dir')
    ap.add_argument('files', nargs='+')
    ap.add_argument('--set', action='append', default=[], metavar='KEY=JSON')
    a = ap.parse_args()
    sys.path.insert(0, a.spiral_dir)
    from config import Config

    merged = {}
    for path in a.files:
        with open(path, encoding='utf-8') as f:
            merged.update(json.load(f))
    for item in a.set:
        key, _, value = item.partition('=')
        merged[key] = json.loads(value)
    Config(merged)   # raises on unknown keys, wrong types or out-of-range values
    print(json.dumps(merged, separators=(',', ':'), sort_keys=True))


if __name__ == '__main__':
    main()
