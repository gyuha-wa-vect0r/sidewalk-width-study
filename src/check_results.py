"""Validate the thesis aggregates from the bundled prediction CSVs on CPU."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import RESULTS, SPEC, metrics

MODELS = SPEC['models']
EXPERIMENTS = ('canonical', 'erode', 'dilate')
IDENTITY = ['filename', 'site', 'tile', 'fresh']


def read_prediction(folder, name):
    return pd.read_csv(folder / f'{name}_predictions.csv', float_precision='round_trip')


def read_json(path):
    return json.loads(path.read_text())


def compare(actual, expected, field):
    if isinstance(expected, dict):
        actual = {str(key): value for key, value in actual.items()}
        if actual.keys() != expected.keys():
            raise ValueError(f'{field} has different metric keys')
        for key, value in expected.items():
            compare(actual[key], value, f'{field}.{key}')
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f'{field} has different length')
        for index, (first, second) in enumerate(zip(actual, expected, strict=True)):
            compare(first, second, f'{field}[{index}]')
    elif not math.isclose(actual, expected, abs_tol=1e-10, rel_tol=0.0):
        raise ValueError(f'{field}: recomputed={actual}; stored={expected}')


def verify_predictions(folder, name, expected, cohort):
    data = read_prediction(folder, name)
    pd.testing.assert_frame_equal(data[IDENTITY], cohort[IDENTITY].reset_index(drop=True))
    if not data.filename.is_unique or not np.isfinite(data.iloc[:, 4:]).all().all():
        raise ValueError(f'{name} contains invalid predictions or measurements')
    recomputed = metrics(data, list(expected['conditions']))
    compare(recomputed, expected, name)
    return data


def verify_folds(folds, expected):
    sites = sorted(folds.site.unique())
    cohorts = []
    for site in sites:
        fold = folds.loc[folds.held_out_site.eq(site)].reset_index(drop=True)
        roles = np.select([~fold.eligible, fold.site.ne(site), ~fold.fresh],
                          ['empty_mask', 'fit', 'held_out_pilot'], default='score')
        if not np.array_equal(roles, fold.role) or not fold.filename.is_unique:
            raise ValueError(f'Invalid roles in holdout {site}')
        cohorts.append(fold[IDENTITY + ['eligible']])
    for other in cohorts[1:]:
        pd.testing.assert_frame_equal(cohorts[0], other)
    cohort = cohorts[0]
    compare(len(cohort), expected['input'], 'registered photos')
    compare(int(cohort.eligible.sum()), expected['eligible'], 'eligible photos')
    compare(int((cohort.eligible & cohort.fresh).sum()), expected['score'], 'evaluated photos')
    return cohort


def verify_review(data, expected, name):
    rows = []
    for threshold in SPEC['narrow_thresholds_m']:
        for site, group in data.loc[data.fresh].groupby('site'):
            budget = math.ceil(SPEC['review_fraction'] * len(group))
            narrow = group.width.lt(threshold)
            for method in ('geometry_pavement', 'row_widths', 'raw_rgb'):
                chosen = group.sort_values([method, 'filename']).iloc[:budget]
                rows.append({'encoder': name, 'width_threshold_m': threshold, 'site': site,
                             'method': method, 'photos': len(group), 'budget': budget,
                             'narrow': int(narrow.sum()), 'found': int(chosen.width.lt(threshold).sum()),
                             'missed_at_threshold': int((narrow & group[method].ge(threshold)).sum())})
    pd.testing.assert_frame_equal(pd.DataFrame(rows), expected.reset_index(drop=True))


def verify(root):
    main = root / 'main'
    reference = read_json(main / 'metrics.json')
    cohort = verify_folds(pd.read_csv(main / 'folds.csv'), reference)
    review = pd.read_csv(root / 'site_review/site_results.csv')
    for name in MODELS:
        data = verify_predictions(main, name, reference[name], cohort.loc[cohort.eligible])
        verify_review(data, review.loc[review.encoder.eq(name)], name)

    sensitivity = root / 'mask_sensitivity'
    stable = read_json(sensitivity / 'metrics.json')
    mask_cohort = pd.read_csv(sensitivity / 'cohort.csv')
    pd.testing.assert_frame_equal(cohort[IDENTITY], mask_cohort[IDENTITY])
    if (mask_cohort.eligible & ~cohort.eligible).any():
        raise ValueError('A sensitivity sample is ineligible in the main experiment')
    compare(int(mask_cohort.eligible.sum()), stable['eligible'], 'mask eligible')
    compare(int((mask_cohort.eligible & mask_cohort.fresh).sum()), stable['evaluation'], 'mask evaluated')
    for name in MODELS:
        for variant in EXPERIMENTS:
            verify_predictions(sensitivity, f'{variant}_{name}', stable['models'][name][variant],
                               mask_cohort.loc[mask_cohort.eligible])

    print(f"PASS: {reference['score']} main and {stable['evaluation']} sensitivity evaluation photos")
    for name in MODELS:
        core = reference[name]
        mae = core['conditions']['geometry_pavement']['MAE_m']
        gain = 100 * core['effects']['context_given_pavement']['gain_m']
        print(f'{name}: condition B MAE {mae:.3f} m; B to D gain {gain:+.2f} cm')
    dino = reference['dino']['conditions']
    for threshold in SPEC['narrow_thresholds_m']:
        row = dino['row_widths']['thresholds'][str(threshold)]
        other = dino['geometry_pavement']['thresholds'][str(threshold)]
        print(f"Below {threshold:.1f} m: {row['review']['budget']} reviews, "
              f"{row['review']['found']} row-width hits, {other['review']['found']} DINO B hits "
              f"among {row['narrow']} cases")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=RESULTS)
    verify(parser.parse_args().results)
