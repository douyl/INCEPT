#!/usr/bin/env python3
import os
import csv
import glob
import argparse
import logging
import random
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone

import mne
import numpy as np
from mne.export import export_raw
from mne.preprocessing import ICA
from mne_icalabel import label_components

warnings.filterwarnings("ignore")


# ==============================================================================
# Channel and Label Definitions
# ==============================================================================
TARGET_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
]

ALL_ARTIFACT_LABELS = ['eyem', 'chew', 'shiv', 'musc', 'elec']
TARGET_LABELS = ['eyem', 'musc', 'elec', 'bckg']
DROP_LABELS = ['chew', 'shiv']

TUAR_SUBDIRS = ['01_tcp_ar', '02_tcp_le', '03_tcp_ar_a']


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess and segment TUAR EEG dataset.')

    parser.add_argument('--input_dir', type=str, required=True,
                        help='TUAR EDF root directory containing 01_tcp_ar, 02_tcp_le, 03_tcp_ar_a.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output root directory. Results will be saved under output_dir/TUAR_250Hz/.')
    parser.add_argument('--target_freq', type=int, default=250,
                        help='Target sampling frequency after resampling.')
    parser.add_argument('--seg_len', type=float, default=5.0,
                        help='Segment length in seconds.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for subject-wise split.')
    parser.add_argument('--notch_freq', type=float, default=60.0,
                        help='Line frequency for notch filter. Use 0 to skip.')
    parser.add_argument('--l_freq', type=float, default=0.1,
                        help='High-pass cutoff.')
    parser.add_argument('--h_freq', type=float, default=75.0,
                        help='Low-pass cutoff.')
    parser.add_argument('--raw_unit', type=str, choices=['uV', 'V'], default='V',
                        help='Raw data unit. If uV, convert to volts.')
    parser.add_argument('--skip_reference', action='store_true',
                        help='Skip average re-reference.')
    parser.add_argument('--skip_ica', action='store_true',
                        help='Skip ICA artifact removal.')
    parser.add_argument('--ica_n', type=int, default=None,
                        help='Number of ICA components.')
    parser.add_argument('--save_shape', type=str, choices=['ct', 'cst'], default='cst',
                        help='ct=(channels,time), cst=(channels,seconds,freq).')

    return parser.parse_args()


def get_subject_id(filename):
    return filename.split('_')[0]


def clean_channel_name(ch):
    ch = ch.replace('EEG ', '').replace('-REF', '').replace('-LE', '')
    ch = ch.replace('FP', 'Fp').replace('Z', 'z')
    return ch


def find_edf_csv_pairs(input_dir):
    pairs = []

    for subdir in TUAR_SUBDIRS:
        cur_dir = os.path.join(input_dir, subdir)
        edf_files = sorted(glob.glob(os.path.join(cur_dir, '*.edf')))

        for edf_path in edf_files:
            csv_path = os.path.splitext(edf_path)[0] + '.csv'
            if not os.path.exists(csv_path):
                logging.warning(f'CSV not found for {os.path.basename(edf_path)}, skipping.')
                continue
            pairs.append((edf_path, csv_path, subdir))

    return pairs


def split_subjects(pairs, seed=42):
    subjects = sorted(set(get_subject_id(os.path.basename(edf)) for edf, _, _ in pairs))

    rng = random.Random(seed)
    rng.shuffle(subjects)

    n_total = len(subjects)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)

    train_subs = set(subjects[:n_train])
    val_subs = set(subjects[n_train:n_train + n_val])
    eval_subs = set(subjects[n_train + n_val:])

    subject_split = {}

    for sub in train_subs:
        subject_split[sub] = 'train'
    for sub in val_subs:
        subject_split[sub] = 'val'
    for sub in eval_subs:
        subject_split[sub] = 'eval'

    return subject_split, train_subs, val_subs, eval_subs


def read_tuar_csv(csv_path):
    events = []

    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        rows = [line for line in f if not line.lstrip().startswith('#') and line.strip()]

    if not rows:
        return events

    reader = csv.DictReader(rows)

    for row in reader:
        try:
            start = float(row['start_time'])
            stop = float(row['stop_time'])
            label = row['label'].strip().lower()
        except Exception:
            continue

        labels = [x for x in label.split('_') if x in ALL_ARTIFACT_LABELS]

        if labels:
            events.append((start, stop, set(labels)))

    return events


def get_segment_label(seg_start, seg_stop, events):
    raw_labels = set()

    for ev_start, ev_stop, ev_labels in events:
        if ev_stop <= seg_start or ev_start >= seg_stop:
            continue
        raw_labels.update(ev_labels)

    if not raw_labels:
        return 'bckg'

    labels = {lab for lab in raw_labels if lab not in DROP_LABELS}

    if len(labels) == 0:
        return 'drop'
    if len(labels) == 1:
        return next(iter(labels))

    return 'multi'


def fix_invalid_meas_date(raw, fname):
    meas_date = raw.info.get('meas_date', None)

    if meas_date is None or not (1985 <= meas_date.year <= 2084):
        dummy_date = datetime(2000, 1, 1, tzinfo=timezone.utc)
        raw.set_meas_date(dummy_date)
        logging.info(f'Fixed invalid EDF date for {fname}: {meas_date} -> {dummy_date}')

    return raw


def preprocess_raw(edf_path, args):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='ERROR')

    raw.rename_channels({ch: clean_channel_name(ch) for ch in raw.ch_names})

    missing = [ch for ch in TARGET_CHANNELS if ch not in raw.ch_names]
    if missing:
        raise RuntimeError(f'Missing channels: {missing}')

    raw.pick_channels(TARGET_CHANNELS, verbose='ERROR')
    raw.reorder_channels(TARGET_CHANNELS)

    if args.raw_unit == 'uV':
        raw._data /= 1e6

    if args.notch_freq and args.notch_freq > 0:
        freqs = [args.notch_freq]
        if 2 * args.notch_freq < raw.info['sfreq'] / 2:
            freqs.append(2 * args.notch_freq)
        raw.notch_filter(freqs=freqs, verbose='ERROR')

    if args.l_freq is not None or args.h_freq is not None:
        raw.filter(l_freq=args.l_freq, h_freq=args.h_freq, phase='zero-double', verbose='ERROR')

    if args.target_freq and raw.info['sfreq'] != args.target_freq:
        raw.resample(args.target_freq, npad='auto', verbose='ERROR')

    if not args.skip_reference:
        raw.set_eeg_reference(ref_channels='average', projection=False, verbose='ERROR')

    if not args.skip_ica:
        ica = ICA(
            n_components=args.ica_n,
            method='infomax',
            fit_params=dict(extended=True),
            random_state=97,
            max_iter='auto'
        )

        ica.fit(raw, verbose='ERROR')

        ic_labels = label_components(raw, ica, method='iclabel')
        labels = ic_labels['labels']

        exclude_idx = [idx for idx, lab in enumerate(labels) if lab not in ['brain', 'other']]
        ica.exclude = exclude_idx

        if exclude_idx:
            ex_labels = [labels[i] for i in exclude_idx]
            logging.info(
                f'-> ICA excluded {len(exclude_idx)} components: '
                f'{", ".join(map(str, exclude_idx))} '
                f'(Labels: {", ".join(ex_labels)})'
            )

        raw = ica.apply(raw, verbose='ERROR')

    return raw


def save_preprocessed_edf(raw, out_edf):
    os.makedirs(os.path.dirname(out_edf), exist_ok=True)
    export_raw(out_edf, raw, fmt='edf', overwrite=True, verbose='ERROR')


def save_segment(segment, out_path, args):
    segment = segment.astype(np.float32)

    if args.save_shape == 'cst':
        segment = segment.reshape(
            len(TARGET_CHANNELS),
            int(args.seg_len),
            args.target_freq
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, segment)


def main():
    args = parse_args()

    root_dir = os.path.join(args.output_dir, f'TUAR_{args.target_freq}Hz')
    preproc_root = os.path.join(root_dir, 'Preprocess')
    segment_root = os.path.join(root_dir, 'Segment')

    os.makedirs(preproc_root, exist_ok=True)
    os.makedirs(segment_root, exist_ok=True)

    log_file = os.path.join(root_dir, 'logs.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    logging.info('=== TUAR Dataset Preprocessing + Segmentation ===')
    logging.info(f'Input dir      : {args.input_dir}')
    logging.info(f'Output root    : {root_dir}')
    logging.info(f'Preprocess dir : {preproc_root}')
    logging.info(f'Segment dir    : {segment_root}')
    logging.info(f'Raw unit       : {args.raw_unit}')
    logging.info(f'Target freq    : {args.target_freq} Hz')
    logging.info(f'Segment length : {args.seg_len} s')
    logging.info(f'Save shape     : {args.save_shape}')
    logging.info(f'Bandpass       : {args.l_freq} - {args.h_freq} Hz')
    logging.info(f'Notch freq     : {args.notch_freq}')
    logging.info(f'Skip reference : {args.skip_reference}')
    logging.info(f'Skip ICA       : {args.skip_ica}')
    logging.info(f'Target labels  : {TARGET_LABELS}')
    logging.info(f'Dropped labels : {DROP_LABELS}')
    logging.info('=' * 50)

    pairs = find_edf_csv_pairs(args.input_dir)

    if not pairs:
        logging.error(f'No EDF-CSV pairs found under {args.input_dir}')
        return

    subject_split, train_subs, val_subs, eval_subs = split_subjects(pairs, seed=args.seed)

    logging.info(f'Total EDF-CSV pairs : {len(pairs)}')
    logging.info(f'Total subjects      : {len(subject_split)}')
    logging.info(f'Train subjects      : {len(train_subs)}')
    logging.info(f'Val subjects        : {len(val_subs)}')
    logging.info(f'Eval subjects       : {len(eval_subs)}')
    logging.info('=' * 50)

    split_file = os.path.join(root_dir, 'subject_split.csv')
    with open(split_file, 'w', encoding='utf-8') as f:
        f.write('subject,split\n')
        for sub in sorted(subject_split):
            f.write(f'{sub},{subject_split[sub]}\n')

    points_per_seg = int(args.target_freq * args.seg_len)

    total_segments = 0
    saved_segments = 0
    multi_segments = 0
    drop_segments = 0
    skipped_files = 0

    label_counter = Counter()
    split_label_counter = defaultdict(Counter)

    for edf_path, csv_path, subdir in pairs:
        fname = os.path.basename(edf_path)
        base = os.path.splitext(fname)[0]
        subject_id = get_subject_id(fname)
        split_name = subject_split[subject_id]

        try:
            events = read_tuar_csv(csv_path)

            raw = preprocess_raw(edf_path, args)
            raw = fix_invalid_meas_date(raw, fname)
            data = raw.get_data()

            if data.shape[0] != len(TARGET_CHANNELS):
                logging.warning(
                    f'{fname} has {data.shape[0]} channels, '
                    f'expected {len(TARGET_CHANNELS)}. Skipping.'
                )
                skipped_files += 1
                continue

            out_edf = os.path.join(preproc_root, subdir, fname)
            if not os.path.exists(out_edf):
                save_preprocessed_edf(raw.copy(), out_edf)

            total_pts = data.shape[1]
            num_segments = total_pts // points_per_seg

            if num_segments == 0:
                logging.warning(f'File too short: {fname}. Skipping.')
                skipped_files += 1
                continue

            file_saved = 0
            file_multi = 0
            file_drop = 0

            for seg_idx in range(num_segments):
                seg_start = seg_idx * args.seg_len
                seg_stop = seg_start + args.seg_len

                label = get_segment_label(seg_start, seg_stop, events)
                total_segments += 1

                if label == 'multi':
                    multi_segments += 1
                    file_multi += 1
                    continue

                if label == 'drop':
                    drop_segments += 1
                    file_drop += 1
                    continue

                start = seg_idx * points_per_seg
                end = start + points_per_seg

                segment = data[:, start:end]
                out_name = f'{base}_{seg_idx:05d}.npy'
                out_path = os.path.join(segment_root, split_name, subject_id, label, out_name)

                save_segment(segment, out_path, args)

                saved_segments += 1
                file_saved += 1
                label_counter[label] += 1
                split_label_counter[split_name][label] += 1

            logging.info(
                f'{fname} -> saved: {file_saved}, '
                f'multi_removed: {file_multi}, '
                f'drop_removed: {file_drop}, '
                f'split: {split_name}, '
                f'preproc_subdir: {subdir}'
            )

        except Exception as e:
            skipped_files += 1
            logging.error(f'Error processing {fname}: {e}')

    multi_ratio_all = multi_segments / total_segments if total_segments > 0 else 0.0
    multi_ratio_labeled = (
        multi_segments / (multi_segments + saved_segments)
        if (multi_segments + saved_segments) > 0 else 0.0
    )
    drop_ratio_all = drop_segments / total_segments if total_segments > 0 else 0.0

    logging.info('=' * 50)
    logging.info('All done!')
    logging.info(f'Total 5s segments scanned       : {total_segments}')
    logging.info(f'Saved 4-class segments          : {saved_segments}')
    logging.info(f'Multi-artifact segments removed : {multi_segments}')
    logging.info(f'Chew/Shiv-only segments removed : {drop_segments}')
    logging.info(f'Multi removed / all segments    : {multi_ratio_all:.4%}')
    logging.info(f'Multi removed / saved+multi     : {multi_ratio_labeled:.4%}')
    logging.info(f'Drop removed / all segments     : {drop_ratio_all:.4%}')
    logging.info(f'Skipped files                   : {skipped_files}')
    logging.info(f'Label counts                    : {dict(label_counter)}')

    for split_name in ['train', 'val', 'eval']:
        logging.info(f'{split_name} label counts: {dict(split_label_counter[split_name])}')


if __name__ == '__main__':
    main()