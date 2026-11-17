"""Run the four fixed-geometry conditions and the regression baselines."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from threadpoolctl import threadpool_limits

from encoders import encode, encoder_model
from metrics import RESULTS, SPEC, json_file, metrics

ARMS = ('geometry', 'geometry_pavement', 'geometry_context', 'geometry_both')
BASELINES = ('raw_rgb', 'row_widths', 'area_only', 'train_mean')
GRAY = (124, 116, 104)
CACHE_KEYS = (('silhouette', 'constant'), ('sidewalk_only', 'constant'),
              ('context_only', 'constant'), ('original', 'original'))


def analysis_frame():
    # Use the frozen groups and pilot exclusions, so the GPS table is unnecessary.
    folds = pd.read_csv(RESULTS / 'main/folds.csv')
    cohort = folds.loc[folds.held_out_site.eq(folds.held_out_site.min()),
                       ['filename', 'site', 'tile', 'fresh', 'eligible']]
    reference = pd.read_csv(RESULTS / 'main/dino_predictions.csv', float_precision='round_trip')
    return cohort.merge(reference[['filename', 'width']], on='filename', how='left', validate='one_to_one')


def image_mask(labels):
    return np.asarray(Image.fromarray(labels).resize((224, 224), Image.Resampling.NEAREST)) == 1


def pictures(rgb, mask):
    shape = np.full_like(rgb, GRAY)
    shape[mask] = 216
    inverse = np.full_like(rgb, 216)
    inverse[mask] = GRAY
    inside = shape.copy()
    inside[mask] = rgb[mask]
    outside = rgb.copy()
    outside[mask] = GRAY
    return shape, inverse, inside, outside


def feature_arrays(model, encoder, frame, prepared, feature_cache):
    names = list(frame.filename)
    masks = []
    direct, inverse = [], []
    for start in range(0, len(names), 32):
        images, inverted = [], []
        for name in names[start:start + 32]:
            with np.load(prepared / f'{name}.npz') as source:
                mask = image_mask(source['region'])
                masks.append(mask)
                shape, rev, inside, outside = pictures(source['rgb'], mask)
                inverted.append(rev)
                if feature_cache is None:
                    images.extend((shape, inside, outside, source['rgb']))
        inverse.append(encode(model, encoder, inverted))
        for index in range(0, len(images), 32):
            direct.append(encode(model, encoder, images[index:index + 32]))

    reversed_geometry = normalize(np.concatenate(inverse).astype(np.float64))
    if feature_cache is None:
        arrays = normalize(np.concatenate(direct).astype(np.float64)).astype(np.float32)
        geometry, inside, outside, original = arrays.reshape(len(names), 4, -1).transpose(1, 0, 2).astype(np.float64)
    else:
        with np.load(feature_cache / f'{model}_features.npz') as saved:
            features = {tuple(key): value for key, value in zip(saved['keys'], saved['values'], strict=True)}
        geometry, inside, outside, original = (
            np.stack([features[(name, *key)] for name in names]).astype(np.float64) for key in CACHE_KEYS
        )
    rows = np.stack(masks).reshape(-1, 16, 14, 224).mean(axis=(2, 3))
    return geometry, inside - geometry, outside - reversed_geometry, original, rows


def ablation_designs(geometry, pavement, context):
    zeros = np.zeros_like(geometry)
    branches = ((zeros, zeros), (pavement, zeros), (zeros, context), (pavement, context))
    return {label: np.concatenate((geometry, *branch), axis=1)
            for label, branch in zip(ARMS, branches, strict=True)}


def fit_predict(frame, designs):
    truth = frame.width.to_numpy()
    selected = frame[['filename', 'site', 'tile', 'fresh', 'width']].copy()
    for label, x in designs.items():
        selected[label] = np.nan
        for site in sorted(frame.site.unique()):
            test = frame.site.eq(site).to_numpy()
            base = Ridge(alpha=SPEC['ridge_alpha'])
            head = make_pipeline(StandardScaler(), base) if label in {'row_widths', 'area_only'} else base
            with threadpool_limits(limits=4):
                head.fit(x[~test], truth[~test])
                selected.loc[test, label] = np.maximum(0, head.predict(x[test]))
    if not np.isfinite(selected[list(designs)]).all().all():
        raise ValueError('Non-finite predictions')
    return selected


def add_input_arguments(parser):
    parser.add_argument('--prepared', type=Path, required=True, help='Folder containing <filename>.npz')
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True, help='Pinned sidewalk-tagger-ai checkout')
    parser.add_argument('--output', type=Path, required=True, help='New output folder')


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    universe = analysis_frame()
    frame = universe.loc[universe.eligible].reset_index(drop=True)
    result = {'role': SPEC['analysis_role'], 'input': len(universe),
              'eligible': len(frame), 'score': int(frame.fresh.sum())}
    for model in SPEC['models']:
        encoder = encoder_model(model, args.weights, args.upstream)
        geometry, pavement, context, original, rows = feature_arrays(
            model, encoder, frame, args.prepared, args.feature_cache
        )
        del encoder
        designs = ablation_designs(geometry, pavement, context)
        designs.update(raw_rgb=original, row_widths=rows, area_only=rows.mean(axis=1)[:, None])
        pred = fit_predict(frame, designs)
        pred['train_mean'] = pred.site.map({site: frame.loc[frame.site.ne(site), 'width'].mean()
                                          for site in frame.site.unique()})
        pred.to_csv(args.output / f'{model}_predictions.csv', index=False)
        result[model] = metrics(pred, (*ARMS, *BASELINES))
        print(f"{model}: {result['score']} evaluation photos", flush=True)
    json_file(args.output / 'metrics.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(parser)
    parser.add_argument('--feature-cache', type=Path, help='Reuse historical <model>_features.npz when available')
    run(parser.parse_args())
