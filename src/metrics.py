"""Metrics and fixed-seed spatial-block bootstrap used in the thesis."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
SPEC = json.loads((ROOT / 'protocols/seoul_v1.json').read_text())
BOOT_SEED = 20261113


def ci_by_tile(image_rows, value):
    rng = np.random.default_rng(BOOT_SEED)
    by_site = image_rows.groupby(['site','tile'])[value].agg(['sum','count'])
    sums = np.zeros(12000)
    counts = np.zeros(12000)
    for _, block in by_site.groupby(level=0):
        indices = rng.integers(len(block), size=(len(sums), len(block)))
        sums += block['sum'].to_numpy()[indices].sum(axis=1)
        counts += block['count'].to_numpy()[indices].sum(axis=1)
    return np.quantile(sums / counts, [.025,.975]).tolist()


def errors_by_threshold(rows, prediction, threshold):
    actual_narrow = rows.width.lt(threshold)
    predicted_narrow = rows[prediction].lt(threshold)
    return {
        'narrow': int(actual_narrow.sum()),
        'missed_narrow': int((actual_narrow & ~predicted_narrow).sum()),
        'false_narrow': int((~actual_narrow & predicted_narrow).sum()),
    }


def review(rows, prediction, threshold):
    budgets = []
    for site, group in rows.groupby('site'):
        budget = math.ceil(SPEC['review_fraction'] * len(group))
        chosen = group.sort_values([prediction, 'filename']).iloc[:budget]
        budgets.append({
            'site': int(site), 'budget': budget,
            'found': int(chosen.width.lt(threshold).sum()),
            'random_expected': budget * float(group.width.lt(threshold).mean()),
        })
    return {key: sum(row[key] for row in budgets) for key in ('budget', 'found', 'random_expected')}


def metrics(pred, labels):
    rows = pred.loc[pred.fresh].copy()
    truth = rows.width
    metrics = {'n': len(rows), 'conditions': {}, 'effects': {}}
    for name in labels:
        signed = rows[name] - truth
        metrics['conditions'][name] = {
            'MAE_m': float(signed.abs().mean()),
            'bias_m': float(signed.mean()),
            'site_MAE_m': signed.abs().groupby(rows.site).mean().to_dict(),
            'thresholds': {str(t): {**errors_by_threshold(rows, name, t),
                                    'review': review(rows, name, t)}
                           for t in SPEC['narrow_thresholds_m']},
        }
    contrasts = {'context_given_pavement': SPEC['primary_contrast'], **SPEC['secondary_contrasts']}
    for name, (left, right) in contrasts.items():
        rows['delta'] = (rows[left] - truth).abs() - (rows[right] - truth).abs()
        metrics['effects'][name] = {
            'gain_m': float(rows.delta.mean()),
            'conditional_95_m': ci_by_tile(rows, 'delta'),
            'site_gain_m': rows.groupby('site').delta.mean().to_dict(),
        }
    return metrics


def json_file(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()
