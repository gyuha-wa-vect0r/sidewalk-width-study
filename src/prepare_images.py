"""Build 224-pixel RGB images and 96x128 sidewalk labels from the Seoul archives."""
import argparse
import hashlib
import io
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from metrics import RESULTS, sha

ARCHIVES = {
    'imagery_1.zip': '220880dafcbaceb4df1d27e12ada2687',
    'imagery_2.zip': '850ad84eca1860cca862747f45bb8e0d',
    'imagery_3.zip': '84fceef93e2980a60f1998786ccc51a5',
}
MASK2FORMER_SHA256 = '917c43cb43a68cb74983b9294d502ca8548b9d98b4727077932811954c4b1f4b'
CONTEXT_CLASSES = {'building', 'wall', 'fence', 'vegetation', 'terrain', 'sky'}


def match_members(archive, remaining):
    by_stem = {Path(name).stem.casefold(): name for name in remaining}
    matched = {}
    for item in archive.infolist():
        path = Path(item.filename)
        target = by_stem.get(path.stem.casefold())
        if target is None or item.is_dir() or '__MACOSX' in path.parts or path.name.startswith('._'):
            continue
        if target in matched:
            raise ValueError(f'Duplicate archive image: {target}')
        matched[target] = item
    return matched


def decode(payload):
    with Image.open(io.BytesIO(payload)) as src:
        image = src.convert('RGB')
        rgb = np.asarray(image.resize((224, 224), Image.Resampling.BICUBIC))
        model_image = image.resize((640, 480), Image.Resampling.BICUBIC)
    return rgb, model_image, hashlib.sha256(payload).hexdigest()


@torch.inference_mode()
def run(args):
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    archive_paths = [args.archives / name for name in ARCHIVES]
    for path in archive_paths:
        with path.open('rb') as archive:
            if hashlib.file_digest(archive, 'md5').hexdigest() != ARCHIVES[path.name]:
                raise ValueError(f'Unexpected archive checksum: {path.name}')
    if sha(args.mask2former / 'model.safetensors') != MASK2FORMER_SHA256:
        raise ValueError('Mask2Former checkpoint checksum mismatch')

    folds = pd.read_csv(RESULTS / 'main/folds.csv')
    cohort = folds.drop_duplicates('filename')
    recorded_hashes = pd.read_csv(RESULTS / 'main/prepared_hashes.csv').set_index('filename').prepared_sha256
    args.output.mkdir(parents=True, exist_ok=False)
    remaining = set(cohort.filename)
    if args.pilot_prepared:
        for name in cohort.loc[~cohort.fresh, 'filename']:
            source = args.pilot_prepared / f'{name}.npz'
            if sha(source) != recorded_hashes.loc[name]:
                raise ValueError(f'Historical pilot input checksum mismatch: {name}')
            shutil.copyfile(source, args.output / source.name)
            remaining.remove(name)

    processor = AutoImageProcessor.from_pretrained(args.mask2former, local_files_only=True, use_fast=False)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.mask2former, local_files_only=True)
    model = model.eval().to('cuda', dtype=torch.float16)
    labels = {i: name.casefold() for i, name in model.config.id2label.items()}
    sidewalk = [i for i, name in labels.items() if name == 'sidewalk']
    context = [i for i, name in labels.items() if name in CONTEXT_CLASSES]
    if sidewalk != [15] or len(context) != 6:
        raise ValueError('Unexpected Mask2Former label map')

    for path in archive_paths:
        with zipfile.ZipFile(path) as archive:
            mapped = match_members(archive, remaining)
            names = list(mapped)
            for start in range(0, len(names), 4):
                batch = names[start:start + 4]
                decoded = [decode(archive.read(mapped[name])) for name in batch]
                inputs = processor([entry[1] for entry in decoded], do_resize=False, return_tensors='pt')
                output = model(**inputs.to('cuda', torch.float16))
                masks = processor.post_process_semantic_segmentation(output, target_sizes=[(96, 128)] * len(batch))
                for name, (rgb, _, digest), mask in zip(batch, decoded, masks, strict=True):
                    labels = mask.cpu().numpy()
                    region = np.zeros((96, 128), dtype=np.uint8)
                    region[np.isin(labels, sidewalk)] = 1
                    region[np.isin(labels, context)] = 2
                    np.savez(args.output / f'{name}.npz', rgb=rgb, region=region, digest=digest, source=path.name)
            remaining.difference_update(mapped)
            print(f'{path.name}: prepared {len(names)} photos', flush=True)
    if remaining:
        raise ValueError(f'{len(remaining)} frozen photos are absent from the archives')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', type=Path, required=True)
    parser.add_argument('--mask2former', type=Path, required=True)
    parser.add_argument('--pilot-prepared', type=Path, help='Optional historical input folder for the pilot masks')
    parser.add_argument('--output', type=Path, required=True, help='New output folder')
    run(parser.parse_args())
