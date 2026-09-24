import os
import argparse
import glob
import logging
import numpy as np
import mne
from mne.export import export_raw
from mne.preprocessing import ICA
from mne_icalabel import label_components
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

# ==============================================================================
# Channel Definitions
# ==============================================================================
TARGET_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
]

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess and segment TUAB EEG dataset.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Root directory containing the "edf" folder.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Root directory for processed data (Preprocess/ and Segment/).')
    
    # Preprocessing parameters
    parser.add_argument('--target_freq', type=int, default=250,
                        help='Target sampling frequency after resampling (Hz).')
    parser.add_argument('--notch_freq', type=float, default=60.0,
                        help='Line frequency for notch filter (Hz). Default is 60 for TUAB.')
    parser.add_argument('--l_freq', type=float, default=0.3,
                        help='High-pass filter cutoff frequency (Hz).')
    parser.add_argument('--h_freq', type=float, default=75,
                        help='Low-pass filter cutoff frequency (Hz).')
    parser.add_argument('--raw_unit', type=str, choices=['uV', 'V'], default='V',
                        help='Unit of the raw data. If "uV", data are converted to volts.')
    
    # Optional flags
    parser.add_argument('--skip_reference', action='store_true',
                        help='Skip re-referencing to average.')
    parser.add_argument('--skip_ica', action='store_true',
                        help='Skip ICA artifact removal.')
    parser.add_argument('--ica_n', type=int, default=None,
                        help='Number of components for ICA. Defaults to None.')
    
    # Segmentation parameters
    parser.add_argument('--seg_len', type=float, default=10,
                        help='Length of each segment in seconds.')
    
    return parser.parse_args()

def get_subject_id(filename):
    """Extract subject ID from TUAB filename (e.g., '00000000_s001_t000.edf' -> '00000000')."""
    return filename.split('_')[0]

def split_files_by_subject(files, ratio=0.8):
    """Split a list of files into train and val based on unique subject IDs."""
    if not files:
        return [], []
    
    # Extract and sort unique subjects
    subjects = list(set([get_subject_id(os.path.basename(f)) for f in files]))
    subjects.sort()
    
    split_idx = int(len(subjects) * ratio)
    train_subs = set(subjects[:split_idx])
    val_subs = set(subjects[split_idx:])
    
    train_files, val_files = [], []
    for f in files:
        sub_id = get_subject_id(os.path.basename(f))
        if sub_id in train_subs:
            train_files.append(f)
        else:
            val_files.append(f)
            
    return train_files, val_files, list(train_subs), list(val_subs)

def build_file_mapping(input_dir):
    """
    Map each target EDF file to its assigned split and label.
    Returns: dict[filepath] = (split, label)
    """
    file_map = {}
    
    base_edf_dir = os.path.join(input_dir, "edf")
    channel_std = "01_tcp_ar"
    
    # --- 1. Process Train/Val Normal ---
    train_normal_dir = os.path.join(base_edf_dir, "train", "normal", channel_std)
    train_normal_files = glob.glob(os.path.join(train_normal_dir, "*.edf"))
    t_n_files, v_n_files, t_n_subs, v_n_subs = split_files_by_subject(train_normal_files, ratio=0.8)
    
    for f in t_n_files: file_map[f] = ("train", "normal")
    for f in v_n_files: file_map[f] = ("val", "normal")
    
    logging.info(f"Normal Subjects -> Train: {len(t_n_subs)} | Val: {len(v_n_subs)}")
    
    # --- 2. Process Train/Val Abnormal ---
    train_abnormal_dir = os.path.join(base_edf_dir, "train", "abnormal", channel_std)
    train_abnormal_files = glob.glob(os.path.join(train_abnormal_dir, "*.edf"))
    t_a_files, v_a_files, t_a_subs, v_a_subs = split_files_by_subject(train_abnormal_files, ratio=0.8)
    
    for f in t_a_files: file_map[f] = ("train", "abnormal")
    for f in v_a_files: file_map[f] = ("val", "abnormal")
    
    logging.info(f"Abnormal Subjects -> Train: {len(t_a_subs)} | Val: {len(v_a_subs)}")
    
    # --- 3. Process Eval Normal ---
    eval_normal_dir = os.path.join(base_edf_dir, "eval", "normal", channel_std)
    for f in glob.glob(os.path.join(eval_normal_dir, "*.edf")):
        file_map[f] = ("eval", "normal")
        
    # --- 4. Process Eval Abnormal ---
    eval_abnormal_dir = os.path.join(base_edf_dir, "eval", "abnormal", channel_std)
    for f in glob.glob(os.path.join(eval_abnormal_dir, "*.edf")):
        file_map[f] = ("eval", "abnormal")
        
    return file_map

def main():
    args = parse_args()

    # Create output directories
    preproc_dir = os.path.join(args.output_dir, 'Preprocess')
    segment_dir = os.path.join(args.output_dir, 'Segment')
    os.makedirs(preproc_dir, exist_ok=True)
    os.makedirs(segment_dir, exist_ok=True)

    # Setup logging
    log_file = os.path.join(args.output_dir, 'logs.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    logging.info("=== TUAB Dataset Preprocessing + Segmentation ===")
    logging.info(f"Input dir      : {args.input_dir}")
    logging.info(f"Output dir     : {args.output_dir}")
    logging.info(f"Raw unit       : {args.raw_unit} -> will be converted to V if needed")
    logging.info(f"Target freq    : {args.target_freq} Hz")
    logging.info(f"Segment length : {args.seg_len} s")
    logging.info(f"Bandpass       : {args.l_freq} - {args.h_freq} Hz")
    logging.info(f"Notch freq     : {args.notch_freq} Hz")
    logging.info(f"Skip ICA       : {args.skip_ica}")
    logging.info(f"Skip reference : {args.skip_reference}")
    logging.info("=" * 50)

    # Build dataset mapping (train/val/eval)
    logging.info("Parsing dataset splits (80/20 subject-wise split for train/val)...")
    file_map = build_file_mapping(args.input_dir)
    
    if not file_map:
        logging.error(f"No valid .edf files found in {args.input_dir}/edf/")
        return
    logging.info(f"Total files found: {len(file_map)}")
    logging.info("=" * 50)

    # ------------------------------------------------------------------
    # Step 1: Preprocess each raw EDF and save to Preprocess/
    # ------------------------------------------------------------------
    logging.info("Starting preprocessing...")
    
    for fpath, (split_name, label_name) in file_map.items():
        fname = os.path.basename(fpath)
        
        # Output directory: Preprocess/split/label/
        save_preproc_dir = os.path.join(preproc_dir, split_name, label_name)
        os.makedirs(save_preproc_dir, exist_ok=True)
        out_edf = os.path.join(save_preproc_dir, fname)

        if os.path.exists(out_edf):
            logging.info(f"Preprocessed file exists, skipping step 1: {fname}")
            continue

        try:
            raw = mne.io.read_raw_edf(fpath, preload=True, verbose='ERROR')

            # Clean channel names (TUH specific: 'EEG FP1-REF' -> 'Fp1')
            rename_dict = {}
            for ch in raw.ch_names:
                new_name = ch.replace('EEG ', '').replace('-REF', '').replace('-LE', '')
                new_name = new_name.replace('FP', 'Fp').replace('Z', 'z')
                rename_dict[ch] = new_name
            raw.rename_channels(rename_dict)

            # Check required channels
            missing = [ch for ch in TARGET_CHANNELS if ch not in raw.ch_names]
            if missing:
                logging.warning(f"Missing channels in {fname}: {missing}. Skipping.")
                continue

            # Pick and reorder
            raw.pick_channels(TARGET_CHANNELS, verbose='ERROR')
            raw.reorder_channels(TARGET_CHANNELS)

            # Filtering
            if args.notch_freq:
                raw.notch_filter(freqs=[args.notch_freq, 2 * args.notch_freq], verbose='ERROR')
            if args.l_freq or args.h_freq:
                raw.filter(l_freq=args.l_freq, h_freq=args.h_freq, phase='zero-double', verbose='ERROR')

            # Resampling
            if args.target_freq and raw.info['sfreq'] != args.target_freq:
                raw.resample(args.target_freq, npad='auto', verbose='ERROR')

            # Rereferencing
            if not args.skip_reference:
                raw.set_eeg_reference(ref_channels='average', projection=False, verbose='ERROR')

            # ICA Artifact Removal
            if not args.skip_ica:
                ica = ICA(n_components=args.ica_n, method='fastica', random_state=97, max_iter='auto')
                ica.fit(raw, verbose='ERROR')
                ic_labels = label_components(raw, ica, method="iclabel")
                labels = ic_labels["labels"]
                exclude_idx = [idx for idx, lab in enumerate(labels) if lab not in ['brain', 'other']]
                ica.exclude = exclude_idx
                if exclude_idx:
                    ex_labels = [labels[i] for i in exclude_idx]
                    idx_str = ", ".join(map(str, exclude_idx))
                    lab_str = ", ".join(ex_labels)
                    logging.info(f"-> ICA excluded {len(exclude_idx)} components: {idx_str} (Labels: {lab_str})")
                raw = ica.apply(raw, verbose='ERROR')

            if args.raw_unit == 'uV':
                raw._data /= 1e6   # µV -> V

            # Export
            meas_date = raw.info.get('meas_date', None)
            if meas_date is None or not (1985 <= meas_date.year <= 2084):
                dummy_date = datetime(2000, 1, 1, tzinfo=timezone.utc)
                raw.set_meas_date(dummy_date)
                logging.info(f"Fixed invalid EDF date for {fname}: {meas_date} -> {dummy_date}")
            export_raw(out_edf, raw, fmt='edf', overwrite=True, verbose='ERROR')

        except Exception as e:
            logging.error(f"Error processing {fname}: {e}")

    # ------------------------------------------------------------------
    # Step 2: Segment all preprocessed EDFs and save to Segment/
    # ------------------------------------------------------------------
    logging.info("=" * 50)
    logging.info("Starting segmentation...")
    
    # Locate all preprocessed EDFs
    preproc_files = sorted(glob.glob(os.path.join(preproc_dir, "**", "*.edf"), recursive=True))
    if not preproc_files:
        logging.error("No preprocessed files found. Segmentation aborted.")
        return

    points_per_seg = int(args.target_freq * args.seg_len)
    total_segments = 0

    for fpath in preproc_files:
        fname = os.path.basename(fpath)
        base = os.path.splitext(fname)[0]
        
        # Extract split and label from the directory structure (Preprocess/split/label/fname)
        parts = fpath.split(os.sep)
        label_name = parts[-2]
        split_name = parts[-3]

        # Output directory: Segment/split/label/
        save_seg_dir = os.path.join(segment_dir, split_name, label_name)
        os.makedirs(save_seg_dir, exist_ok=True)

        raw = mne.io.read_raw_edf(fpath, preload=True, verbose='ERROR')

        data = raw.get_data()  # shape (channels, times)
        
        if data.shape[0] != len(TARGET_CHANNELS):
            logging.warning(f"{fname} has {data.shape[0]} channels, expected {len(TARGET_CHANNELS)}. Skipping.")
            continue

        total_pts = data.shape[1]
        num_segments = total_pts // points_per_seg
        
        if num_segments == 0:
            logging.warning(f"File too short: {fname} ({total_pts} pts < {points_per_seg} pts). Skipping.")
            continue

        seg_count = 0
        for seg_idx in range(num_segments):
            start = seg_idx * points_per_seg
            end = start + points_per_seg
            segment = data[:, start:end]  # (channels, points)

            # Reshape to (channels, seconds, freq)
            segment_reshaped = segment.reshape(
                len(TARGET_CHANNELS),
                int(args.seg_len),
                args.target_freq
            ).astype(np.float32)

            # Name format: originalname_id.npy
            out_name = f"{base}_{seg_idx}.npy"
            out_path = os.path.join(save_seg_dir, out_name)
            np.save(out_path, segment_reshaped)
            seg_count += 1

        total_segments += seg_count
        logging.info(f"{fname} -> {seg_count} segments ({split_name}/{label_name})")

    logging.info("=" * 50)
    logging.info(f"All done! Total segments created: {total_segments}")

if __name__ == "__main__":
    main()