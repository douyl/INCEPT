import os
import argparse
import glob
import logging
import numpy as np
import mne
from mne.export import export_raw
import warnings

warnings.filterwarnings("ignore")

# Channel definitions
FINAL_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
]
ORIG_CHANNELS = [f'EEG {ch}' for ch in FINAL_CHANNELS]   # names in raw EDF

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess and segment MentalArithmetic EEG dataset.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing raw .edf files.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Root directory. Preprocessed EDFs go into Preprocess/, segments into Segment/.')
    parser.add_argument('--raw_unit', type=str, choices=['uV', 'V'], default='V',
                        help='Unit of the raw data. If "uV", data are converted to volts.')
    parser.add_argument('--target_freq', type=int, default=250,
                        help='Target sampling frequency after resampling (Hz).')
    parser.add_argument('--seg_len', type=float, default=5,
                        help='Length of each segment in seconds.')
    return parser.parse_args()

def main():
    args = parse_args()

    # Create output directories
    preproc_dir = os.path.join(args.output_dir, 'Preprocess')
    segment_dir = os.path.join(args.output_dir, 'Segment')
    os.makedirs(preproc_dir, exist_ok=True)
    os.makedirs(segment_dir, exist_ok=True)

    # Setup logging (file + console)
    log_file = os.path.join(args.output_dir, 'logs.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    logging.info("=== MentalArithmetic Preprocessing + Segmentation ===")
    logging.info(f"Input dir      : {args.input_dir}")
    logging.info(f"Output dir     : {args.output_dir}")
    logging.info(f"Raw unit       : {args.raw_unit} -> will be converted to V if needed")
    logging.info(f"Target freq    : {args.target_freq} Hz")
    logging.info(f"Segment length : {args.seg_len} s")
    logging.info("=" * 50)

    # ------------------------------------------------------------------
    # Step 1: Preprocess each raw EDF and save to Preprocess/
    # ------------------------------------------------------------------
    raw_files = sorted(glob.glob(os.path.join(args.input_dir, '*.edf')))
    if not raw_files:
        logging.error(f"No .edf files found in {args.input_dir}")
        return

    logging.info(f"Found {len(raw_files)} raw files. Starting preprocessing...")
    for fpath in raw_files:
        fname = os.path.basename(fpath)
        out_edf = os.path.join(preproc_dir, fname)

        try:
            raw = mne.io.read_raw_edf(fpath, preload=True, verbose='ERROR')

            # Check required channels
            missing = [ch for ch in ORIG_CHANNELS if ch not in raw.ch_names]
            if missing:
                logging.error(f"Missing channels in {fname}: {missing}. Skipping.")
                continue

            # Select, rename, reorder
            raw.pick_channels(ORIG_CHANNELS, verbose='ERROR')
            raw.rename_channels({ch: ch.replace('EEG ', '') for ch in raw.ch_names})
            raw.reorder_channels(FINAL_CHANNELS)

            # Resample if needed
            if raw.info['sfreq'] != args.target_freq:
                raw.resample(args.target_freq, npad='auto', verbose='ERROR')

            # Convert unit from µV to V if requested
            if args.raw_unit == 'uV':
                raw._data /= 1e6   # µV -> V

            # Save as EDF
            export_raw(out_edf, raw, fmt='edf', overwrite=True, verbose='ERROR')
            logging.info(f"Preprocessed: {fname}")

        except Exception as e:
            logging.error(f"Error processing {fname}: {e}")

    # ------------------------------------------------------------------
    # Step 2: Segment all preprocessed EDFs and save to Segment/
    # ------------------------------------------------------------------
    preproc_files = sorted(glob.glob(os.path.join(preproc_dir, '*.edf')))
    if not preproc_files:
        logging.error("No preprocessed files found. Segmentation aborted.")
        return

    points_per_seg = int(args.target_freq * args.seg_len)
    total_segments = 0

    logging.info(f"Starting segmentation of {len(preproc_files)} files...")
    for fpath in preproc_files:
        fname = os.path.basename(fpath)
        base = os.path.splitext(fname)[0]   # e.g. Subject00_1

        # Parse subject and label from filename (expected: SubjectXX_Y)
        try:
            parts = base.split('_')
            if len(parts) != 2:
                logging.error(f"Unexpected filename format: {fname}. Skipping.")
                continue
            subject_id, label_id = parts[0], parts[1]
        except Exception as e:
            logging.error(f"Error parsing filename {fname}: {e}")
            continue

        # Create output directory: Segment/subject/label/
        save_dir = os.path.join(segment_dir, subject_id, label_id)
        os.makedirs(save_dir, exist_ok=True)

        # Read preprocessed EDF
        try:
            raw = mne.io.read_raw_edf(fpath, preload=True, verbose='ERROR')
        except Exception as e:
            logging.error(f"Failed to read {fname}: {e}")
            continue

        data = raw.get_data()   # shape (channels, times), already in V
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
            segment = data[:, start:end]   # (channels, points)

            # Reshape to (channels, seconds, freq)
            segment = segment.reshape(
                len(FINAL_CHANNELS),
                int(args.seg_len),
                args.target_freq
            ).astype(np.float32)

            out_name = f"{base}_{seg_idx}.npy"
            out_path = os.path.join(save_dir, out_name)
            np.save(out_path, segment)
            seg_count += 1

        total_segments += seg_count
        logging.info(f"{fname} -> {seg_count} segments")

    logging.info("=" * 50)
    logging.info(f"All done! Total segments created: {total_segments}")

if __name__ == "__main__":
    main()