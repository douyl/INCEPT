import os
import argparse
import glob
import logging
import numpy as np
import mne
from mne.export import export_raw
import csv
import re
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# Channel Definitions
# ==============================================================================
FINAL_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
]

# Label Mapping from TSV to Folder names
LABEL_MAP = {
    'A': 'AD',
    'F': 'FTD',
    'C': 'HC'
}

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess and segment ADFTD dataset.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Root directory containing derivatives/ and participants.tsv.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Root directory for output (e.g., ADFTD_250Hz).')
    parser.add_argument('--raw_unit', type=str, choices=['uV', 'V'], default='V',
                        help='Unit of the raw data. If "uV", data are converted to volts.')
    parser.add_argument('--target_freq', type=int, default=250,
                        help='Target sampling frequency after resampling (Hz).')
    parser.add_argument('--seg_len', type=float, default=10,
                        help='Length of each segment in seconds (default: 10s).')
    parser.add_argument('--l_freq', type=float, default=None,
                        help='Lower pass-band edge in Hz. If None, no high-pass filtering.')
    parser.add_argument('--h_freq', type=float, default=None,
                        help='Upper pass-band edge in Hz. If None, no low-pass filtering.')
    parser.add_argument('--notch_freq', type=float, default=None,
                        help='Frequency to notch filter. If None, no notch filter.')
    return parser.parse_args()

def natural_sort_key(s):
    """Fallback for natsort using pure Python."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def get_file_split_map(input_dir):
    """
    Parses participants.tsv without pandas, maps labels, 
    sorts subjects, and splits into train(60%)/eval(20%)/val(20%).
    """
    tsv_path = os.path.join(input_dir, 'participants.tsv')
    
    if not os.path.exists(tsv_path):
        logging.error(f"TSV file not found: {tsv_path}")
        return {}

    # Read TSV using native csv module
    label_dict = {'AD': [], 'FTD': [], 'HC': []}
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            sub_id = row['participant_id'].strip()
            group_code = row['Group'].strip()
            
            if group_code in LABEL_MAP:
                mapped_label = LABEL_MAP[group_code]
                label_dict[mapped_label].append(sub_id)
            else:
                logging.warning(f"Unknown group code '{group_code}' for {sub_id}, skipping.")

    file_info_map = {}
    
    # Split 6:2:2 sequentially (Train -> Eval -> Val)
    for label, subjects in label_dict.items():
        # Natural sort subjects
        subjects = sorted(subjects, key=natural_sort_key)
        
        n_total = len(subjects)
        n_train = int(n_total * 0.6)
        n_eval_val = int(n_total * 0.2) # Both eval and val take 20%
        
        train_subs = subjects[:n_train]
        eval_subs = subjects[n_train : n_train + n_eval_val]
        val_subs = subjects[n_train + n_eval_val:]
        
        logging.info(f"Class {label} -> Total: {n_total} | Train: {len(train_subs)} | Eval: {len(eval_subs)} | Val: {len(val_subs)}")
        
        for sub in train_subs: file_info_map[sub] = ('train', label)
        for sub in eval_subs:  file_info_map[sub] = ('eval', label)
        for sub in val_subs:   file_info_map[sub] = ('val', label)
        
    return file_info_map

def main():
    args = parse_args()

    preproc_dir = os.path.join(args.output_dir, 'Preprocess')
    
    # Segment dir dynamically includes the length (e.g., Segment_10s, Segment_30s)
    seg_suffix = f"{int(args.seg_len)}s" if args.seg_len.is_integer() else f"{args.seg_len}s"
    segment_dir = os.path.join(args.output_dir, f'Segment_{seg_suffix}')
    
    os.makedirs(preproc_dir, exist_ok=True)
    os.makedirs(segment_dir, exist_ok=True)

    # Setup logging
    log_file = os.path.join(args.output_dir, f'logs_segment_{seg_suffix}.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    logging.info("=== ADFTD Dataset Preprocessing + Segmentation ===")
    logging.info(f"Input dir      : {args.input_dir}")
    logging.info(f"Target freq    : {args.target_freq} Hz")
    logging.info(f"Segment length : {args.seg_len} s")
    logging.info(f"Preproc path   : {preproc_dir}")
    logging.info(f"Segment path   : {segment_dir}")
    logging.info("=" * 50)

    # ------------------------------------------------------------------
    # Parse dataset splits
    # ------------------------------------------------------------------
    logging.info("Parsing dataset splits sequentially from TSV...")
    file_split_map = get_file_split_map(args.input_dir)
    if not file_split_map:
        logging.error("Failed to parse split map. Aborting.")
        return

    # ------------------------------------------------------------------
    # Step 1: Preprocess each raw .set file and save as .edf
    # (Skips automatically if the file already exists in Preprocess/)
    # ------------------------------------------------------------------
    derivatives_dir = os.path.join(args.input_dir, 'derivatives')
    logging.info(f"Starting preprocessing step...")
    
    for sub_id, (split_name, label_name) in file_split_map.items():
        # Find .set file
        search_pattern = os.path.join(derivatives_dir, sub_id, 'eeg', '*.set')
        set_files = glob.glob(search_pattern)
        
        if not set_files:
            logging.warning(f"No .set file found for {sub_id} in {search_pattern}")
            continue
            
        fpath = set_files[0] # Assume one .set file per subject
        out_edf = os.path.join(preproc_dir, fpath.split('/')[-1].replace('.set', '.edf'))

        # If already preprocessed (e.g., from a previous run with a different seg_len), skip it!
        if os.path.exists(out_edf):
            logging.info(f"Preprocessed file exists, skipping step 1 for: {sub_id}")
            continue

        try:
            # Read EEGLAB .set file
            raw = mne.io.read_raw_eeglab(fpath, preload=True, verbose='ERROR')

            # Check and Pick Channels
            missing = [ch for ch in FINAL_CHANNELS if ch not in raw.ch_names]
            if missing:
                logging.error(f"Missing channels in {sub_id}: {missing}. Skipping.")
                continue

            raw.pick_channels(FINAL_CHANNELS, verbose='ERROR')
            raw.reorder_channels(FINAL_CHANNELS)

            # Filtering
            if args.l_freq is not None or args.h_freq is not None:
                raw.filter(l_freq=args.l_freq, h_freq=args.h_freq, verbose='ERROR')

            if args.notch_freq is not None:
                raw.notch_filter(freqs=args.notch_freq, verbose='ERROR')

            # Resample
            if raw.info['sfreq'] != args.target_freq:
                raw.resample(args.target_freq, npad='auto', verbose='ERROR')

            # Convert unit if required
            if args.raw_unit == 'uV':
                raw._data /= 1e6   # µV -> V

            # Save as EDF
            export_raw(out_edf, raw, fmt='edf', overwrite=True, verbose='ERROR')

        except Exception as e:
            logging.error(f"Error processing {sub_id}: {e}")

    # ------------------------------------------------------------------
    # Step 2: Segment all preprocessed EDFs
    # ------------------------------------------------------------------
    preproc_files = sorted(glob.glob(os.path.join(preproc_dir, '*.edf')))
    if not preproc_files:
        logging.error("No preprocessed files found. Segmentation aborted.")
        return

    points_per_seg = int(args.target_freq * args.seg_len)
    total_segments = 0

    logging.info(f"Starting segmentation of {len(preproc_files)} files into {args.seg_len}s segments...")
    
    for fpath in preproc_files:
        fname = os.path.basename(fpath)
        
        # Extract subject ID from filename (assuming filename contains sub-XXX)
        sub_match = re.search(r'(sub-\d+)', fname)
        if not sub_match:
            logging.warning(f"Could not extract sub_id from {fname}. Skipping.")
            continue
            
        sub_id = sub_match.group(1)
        base = os.path.splitext(fname)[0]

        if sub_id not in file_split_map:
            logging.warning(f"{sub_id} not in mapping dict. Skipping.")
            continue
            
        split_name, label_name = file_split_map[sub_id]

        # Create output directory dynamically: Segment_10s/train/AD/sub-001/ etc.
        save_dir = os.path.join(segment_dir, split_name, label_name, sub_id)
        os.makedirs(save_dir, exist_ok=True)

        try:
            raw = mne.io.read_raw_edf(fpath, preload=True, verbose='ERROR')
        except Exception as e:
            logging.error(f"Failed to read {fname}: {e}")
            continue

        data = raw.get_data()   # shape (channels, times)
        if data.shape[0] != len(FINAL_CHANNELS):
            logging.warning(f"{fname} has {data.shape[0]} channels, expected {len(FINAL_CHANNELS)}. Skipping.")
            continue

        total_pts = data.shape[1]
        num_segments = total_pts // points_per_seg
        
        if num_segments == 0:
            logging.warning(f"File too short: {fname} ({total_pts} pts). Skipping.")
            continue

        seg_count = 0
        for seg_idx in range(num_segments):
            start = seg_idx * points_per_seg
            end = start + points_per_seg
            segment = data[:, start:end]

            # Reshape to (19, seg_len, target_freq)
            segment_reshaped = segment.reshape(
                len(FINAL_CHANNELS),
                int(args.seg_len),
                args.target_freq
            ).astype(np.float32)

            out_name = f"{base}_{seg_idx}.npy"
            out_path = os.path.join(save_dir, out_name)
            np.save(out_path, segment_reshaped)
            seg_count += 1

        total_segments += seg_count
        logging.info(f"{sub_id} -> {seg_count} segments saved in {split_name}/{label_name}/{sub_id}")

    logging.info("=" * 50)
    logging.info(f"All done! Total segments created in {segment_dir}: {total_segments}")

if __name__ == "__main__":
    main()
