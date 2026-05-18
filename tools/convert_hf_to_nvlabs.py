"""
Convert HuggingFace nvidia/mit-b* weights to NVlabs SegFormer format.

HuggingFace key format (actual, verified):
  segformer.encoder.patch_embeddings.{i}.proj.weight/bias
  segformer.encoder.patch_embeddings.{i}.layer_norm.weight/bias
  segformer.encoder.block.{stage}.{layer}.layer_norm_1.weight/bias
  segformer.encoder.block.{stage}.{layer}.attention.self.query.weight/bias
  segformer.encoder.block.{stage}.{layer}.attention.self.key.weight/bias    <-- separate K
  segformer.encoder.block.{stage}.{layer}.attention.self.value.weight/bias  <-- separate V
  segformer.encoder.block.{stage}.{layer}.attention.self.sr.weight/bias       (sr_ratio>1)
  segformer.encoder.block.{stage}.{layer}.attention.self.layer_norm.weight/bias (sr_ratio>1)
  segformer.encoder.block.{stage}.{layer}.attention.output.dense.weight/bias
  segformer.encoder.block.{stage}.{layer}.layer_norm_2.weight/bias
  segformer.encoder.block.{stage}.{layer}.mlp.dense1.weight/bias
  segformer.encoder.block.{stage}.{layer}.mlp.dwconv.dwconv.weight/bias
  segformer.encoder.block.{stage}.{layer}.mlp.dense2.weight/bias
  segformer.encoder.layer_norm.{i}.weight/bias
  classifier.weight/bias  (skipped)

NVlabs key format:
  patch_embed{i+1}.proj.weight/bias
  patch_embed{i+1}.norm.weight/bias
  block{stage+1}.{layer}.norm1.weight/bias
  block{stage+1}.{layer}.attn.q.weight/bias
  block{stage+1}.{layer}.attn.kv.weight/bias    <-- fused K+V (concat along dim 0)
  block{stage+1}.{layer}.attn.proj.weight/bias
  block{stage+1}.{layer}.attn.sr.weight/bias        (sr_ratio>1)
  block{stage+1}.{layer}.attn.norm.weight/bias      (sr_ratio>1)
  block{stage+1}.{layer}.norm2.weight/bias
  block{stage+1}.{layer}.mlp.fc1.weight/bias
  block{stage+1}.{layer}.mlp.dwconv.dwconv.weight/bias
  block{stage+1}.{layer}.mlp.fc2.weight/bias
  norm{i+1}.weight/bias

Usage:
  python tools/convert_hf_to_nvlabs.py /path/to/pytorch_model.bin /path/to/output/mit_b5.pth
"""

import argparse
import re
import torch


def strip_prefix(key):
    if key.startswith('segformer.encoder.'):
        return key[len('segformer.encoder.'):]
    if key.startswith('segformer.'):
        return key[len('segformer.'):]
    return key


def convert_simple_key(key):
    """Convert keys that don't need cross-tensor merging. Returns None to skip."""

    # patch_embeddings.{i}.proj.* -> patch_embed{i+1}.proj.*
    m = re.match(r'^patch_embeddings\.(\d+)\.proj\.(weight|bias)$', key)
    if m:
        return f'patch_embed{int(m.group(1)) + 1}.proj.{m.group(2)}'

    # patch_embeddings.{i}.layer_norm.* -> patch_embed{i+1}.norm.*
    m = re.match(r'^patch_embeddings\.(\d+)\.layer_norm\.(weight|bias)$', key)
    if m:
        return f'patch_embed{int(m.group(1)) + 1}.norm.{m.group(2)}'

    # layer_norm.{i}.* -> norm{i+1}.*
    m = re.match(r'^layer_norm\.(\d+)\.(weight|bias)$', key)
    if m:
        return f'norm{int(m.group(1)) + 1}.{m.group(2)}'

    # block.{stage}.{layer}.xxx -> block{stage+1}.{layer}.xxx (with sub-renames)
    m = re.match(r'^block\.(\d+)\.(\d+)\.(.*)$', key)
    if m:
        stage = int(m.group(1)) + 1
        layer = int(m.group(2))
        rest = m.group(3)

        rest = rest.replace('layer_norm_1', 'norm1')
        rest = rest.replace('layer_norm_2', 'norm2')

        rest = rest.replace('attention.self.query', 'attn.q')
        rest = rest.replace('attention.output.dense', 'attn.proj')
        # NOTE: order matters — replace longer 'layer_norm' before 'sr' is fine,
        # but both must come before any 'attention.self' fallbacks.
        rest = rest.replace('attention.self.layer_norm', 'attn.norm')
        rest = rest.replace('attention.self.sr', 'attn.sr')

        rest = rest.replace('mlp.dense1', 'mlp.fc1')
        rest = rest.replace('mlp.dense2', 'mlp.fc2')
        # mlp.dwconv.dwconv stays the same

        # K/V are handled separately (returned as None here)
        if 'attention.self.key' in rest or 'attention.self.value' in rest:
            return None

        return f'block{stage}.{layer}.{rest}'

    return 'UNHANDLED'


def merge_kv(hf_state):
    """Find every (key, value) pair in HF state, concat into NVlabs `kv`.
    Returns dict of NVlabs-format kv tensors."""
    pat = re.compile(
        r'^(?:segformer\.encoder\.)?block\.(\d+)\.(\d+)\.attention\.self\.(key|value)\.(weight|bias)$'
    )
    pairs = {}  # (stage, layer, kind) -> {'key': tensor, 'value': tensor}
    for k, v in hf_state.items():
        m = pat.match(k)
        if not m:
            continue
        stage, layer, which, kind = m.group(1), m.group(2), m.group(3), m.group(4)
        slot = pairs.setdefault((stage, layer, kind), {})
        slot[which] = v

    out = {}
    for (stage, layer, kind), slot in pairs.items():
        if 'key' not in slot or 'value' not in slot:
            raise RuntimeError(f'incomplete K/V at block.{stage}.{layer} ({kind})')
        kv = torch.cat([slot['key'], slot['value']], dim=0)
        nv_key = f'block{int(stage) + 1}.{layer}.attn.kv.{kind}'
        out[nv_key] = kv
    return out


def convert(src_path, dst_path):
    print(f'Loading HuggingFace checkpoint from: {src_path}')
    hf_state = torch.load(src_path, map_location='cpu')

    nvlabs_state = {}
    skipped_classifier = []
    skipped_kv = 0
    unhandled = []

    for hf_key, value in hf_state.items():
        if hf_key.startswith('classifier.'):
            skipped_classifier.append(hf_key)
            continue

        stripped = strip_prefix(hf_key)
        nv_key = convert_simple_key(stripped)
        if nv_key is None:
            # K/V — handled by merge_kv
            skipped_kv += 1
            continue
        if nv_key == 'UNHANDLED':
            unhandled.append(hf_key)
            continue
        nvlabs_state[nv_key] = value

    # merge K + V into kv
    kv_dict = merge_kv(hf_state)
    nvlabs_state.update(kv_dict)

    print(f'\nConverted simple keys: {len(nvlabs_state) - len(kv_dict)}')
    print(f'Merged K+V into kv:    {len(kv_dict)}  (from {skipped_kv} HF tensors)')
    print(f'Skipped classifier:    {len(skipped_classifier)}')
    if unhandled:
        print(f'!! UNHANDLED keys ({len(unhandled)}):')
        for k in unhandled:
            print(f'   {k}')
        raise RuntimeError('unhandled keys present — aborting')

    # sanity: print a few mappings
    print('\nSample NVlabs keys:')
    for k in list(nvlabs_state.keys())[:8]:
        print(f'  {k}\t{tuple(nvlabs_state[k].shape)}')
    kv_sample = next(k for k in nvlabs_state if '.kv.weight' in k)
    print(f'  {kv_sample}\t{tuple(nvlabs_state[kv_sample].shape)}  (K|V concat)')

    torch.save(nvlabs_state, dst_path)
    print(f'\nSaved NVlabs format checkpoint to: {dst_path}')
    print(f'Total parameters saved: {len(nvlabs_state)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert HuggingFace MIT weights to NVlabs SegFormer format')
    parser.add_argument('src', help='Path to HuggingFace pytorch_model.bin')
    parser.add_argument('dst', help='Output path for NVlabs format .pth')
    args = parser.parse_args()

    convert(args.src, args.dst)
