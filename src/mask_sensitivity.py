"""Replay the ablation after eroding or dilating the inferred sidewalk mask."""
import argparse

import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion
from sklearn.preprocessing import normalize

from encoders import encode, encoder_model
from metrics import ROOT, SPEC, json_file, metrics, sha
from run_study import (ARMS, ablation_designs, add_input_arguments, analysis_frame, fit_predict,
                       image_mask, pictures)

a, b = np.ogrid[-2:3, -2:3]
DISK = a * a + b * b <= 4


def masks_for(mask):
    return (mask, binary_erosion(mask, structure=DISK, border_value=1),
            binary_dilation(mask, structure=DISK, border_value=0))


def features(name, encoder, frame, variants, index, prepared):
    collected = []
    for start in range(0, len(frame), 8):
        inputs = []
        for filename in frame.iloc[start:start + 8].filename:
            with np.load(prepared / f'{filename}.npz') as source:
                rgb = source['rgb']
            inputs.extend(pictures(rgb, image_mask(variants[filename][index])))
        collected.append(encode(name, encoder, inputs))
    result = normalize(np.concatenate(collected).astype(np.float64)).reshape(len(frame), 4, -1)
    return np.moveaxis(result, 1, 0)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    frame = analysis_frame()
    variants = {}
    for name in frame.loc[frame.eligible, 'filename']:
        with np.load(args.prepared / f'{name}.npz') as source:
            mask = source['region'] == 1
        masks = masks_for(mask)
        if all(current.any() for current in masks):
            variants[name] = masks

    eligible = frame.filename.isin(variants)
    frame.assign(eligible=eligible)[['filename', 'site', 'tile', 'fresh', 'eligible']].to_csv(
        args.output / 'cohort.csv', index=False
    )
    frame = frame[eligible].reset_index(drop=True)
    result = {'eligible': len(frame), 'evaluation': int(frame.fresh.sum()),
              'protocol_sha256': sha(ROOT / 'protocols/mask_sensitivity.md'),
              'area_fraction': {}, 'models': {}}
    for name in SPEC['models']:
        encoder = encoder_model(name, args.weights, args.upstream)
        for index, variant in enumerate(('canonical', 'erode', 'dilate')):
            fractions = [variants[filename][index].mean() for filename in frame.loc[frame.fresh, 'filename']]
            result['area_fraction'][variant] = float(np.mean(fractions))
            g, rev, inside, outside = features(name, encoder, frame, variants, index, args.prepared)
            pred = fit_predict(frame, ablation_designs(g, inside - g, outside - rev))
            pred.to_csv(args.output / f'{variant}_{name}_predictions.csv', index=False)
            summary = metrics(pred, ARMS)
            result['models'].setdefault(name, {})[variant] = summary
            print(variant, name, summary['effects']['context_given_pavement'], flush=True)
        del encoder
        torch.cuda.empty_cache()
    json_file(args.output / 'metrics.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(parser)
    run(parser.parse_args())
