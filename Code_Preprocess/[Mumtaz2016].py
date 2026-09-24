import os
import argparse
import glob
import logging
import numpy as np
import mne
from mne.export import export_raw
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# Channel Definitions
# ==============================================================================
FINAL_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
]

# Original names in Mumtaz2016 raw EDF
ORIG_CHANNELS = [
    'EEG Fp1-LE', 'EEG Fp2-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG C3-LE', 'EEG C4-LE', 
    'EEG P3-LE', 'EEG P4-LE', 'EEG O1-LE', 'EEG O2-LE', 'EEG F7-LE', 'EEG F8-LE', 
    'EEG T3-LE', 'EEG T4-LE', 'EEG T5-LE', 'EEG T6-LE', 'EEG Fz-LE', 'EEG Cz-LE', 'EEG Pz-LE'
]

# Create a rename dictionary mapping orig to final
# Ensure mapping relies on matching sub-strings to safely strip 'EEG ' and '-LE'
RENAME_DICT = {
    orig: orig.replace('EEG ', '').replace('-LE', '') for orig in ORIG_CHANNELS
}

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess and segment Mumtaz2016 MDD dataset.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing raw .edf files.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Root directory. Preprocessed EDFs go to Preprocess/, segments to Segment/.')
    parser.add_argument('--raw_unit', type=str, choices=['uV', 'V'], default='V',
                        help='Unit of the raw data. If "uV", data are converted to volts.')
    parser.add_argument('--target_freq', type=int, default=250,
                        help='Target sampling frequency after resampling (Hz).')
    parser.add_argument('--seg_len', type=float, default=5,
                        help='Length of each segment in seconds.')
    
    parser.add_argument('--l_freq', type=float, default=None,
                        help='Lower pass-band edge in Hz. If None, no high-pass filtering.')
    parser.add_argument('--h_freq', type=float, default=None,
                        help='Upper pass-band edge in Hz. If None, no low-pass filtering.')
    parser.add_argument('--notch_freq', type=float, default=None,
                        help='Frequency to notch filter (e.g., 50.0 or 60.0). If None, no notch filter.')

    return parser.parse_args()

def get_file_split_map(input_dir):
    """
    Parses directory, filters out 'TASK' files, separates H and MDD, 
    sorts them, prints them, and maps each filename to (split, label).
    """
    files_H, files_MDD = [], []
    for file in os.listdir(input_dir):
        if not file.endswith('.edf'):
            continue
        if 'TASK' not in file:
            if 'MDD' in file:
                files_MDD.append(file)
            else:
                files_H.append(file)
                
    # Sort files to ensure consistency across different OS/devices
    files_H = sorted(files_H)
    files_MDD = sorted(files_MDD)
    
    # Print as requested to check sorting consistency
    print("=== Sorted Healthy (H) Files ===")
    print(files_H)
    print(f"Total H: {len(files_H)}\n")
    
    print("=== Sorted MDD Files ===")
    print(files_MDD)
    print(f"Total MDD: {len(files_MDD)}\n")
    
    # Map splits: train, val, eval
    file_info_map = {}
    
    # Train
    for f in files_H[:40]:   file_info_map[f] = ('train', 'H')
    for f in files_MDD[:42]: file_info_map[f] = ('train', 'MDD')
    
    # Val
    for f in files_H[40:48]:   file_info_map[f] = ('val', 'H')
    for f in files_MDD[42:52]: file_info_map[f] = ('val', 'MDD')
    
    # Eval
    for f in files_H[48:]:   file_info_map[f] = ('eval', 'H')
    for f in files_MDD[52:]: file_info_map[f] = ('eval', 'MDD')
    
    return file_info_map

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

    logging.info("=== MDD Dataset Preprocessing + Segmentation ===")
    logging.info(f"Input dir      : {args.input_dir}")
    logging.info(f"Output dir     : {args.output_dir}")
    logging.info(f"Raw unit       : {args.raw_unit}")
    logging.info(f"Target freq    : {args.target_freq} Hz")
    logging.info(f"Segment length : {args.seg_len} s")
    logging.info(f"Bandpass       : {args.l_freq} - {args.h_freq} Hz")
    logging.info(f"Notch freq     : {args.notch_freq} Hz")
    logging.info("=" * 50)

    # Determine data splits and print sorted arrays
    logging.info("Parsing dataset splits...")
    file_split_map = get_file_split_map(args.input_dir)
    if not file_split_map:
        logging.error(f"No valid .edf files found in {args.input_dir}")
        return

    # ------------------------------------------------------------------
    # Step 1: Preprocess each raw EDF and save to Preprocess/
    # ------------------------------------------------------------------
    logging.info(f"Found {len(file_split_map)} target files. Starting preprocessing...")
    
    for fname in file_split_map.keys():
        fpath = os.path.join(args.input_dir, fname)
        out_edf = os.path.join(preproc_dir, fname)

        if os.path.exists(out_edf):
            logging.info(f"Preprocessed file exists, skipping step 1: {fname}")
            continue

        try:
            raw = mne.io.read_raw_edf(fpath, preload=True, verbose='ERROR')

            # Check required channels
            missing = [ch for ch in ORIG_CHANNELS if ch not in raw.ch_names]
            if missing:
                logging.error(f"Missing channels in {fname}: {missing}. Skipping.")
                continue

            # Select, rename, reorder
            raw.pick_channels(ORIG_CHANNELS, verbose='ERROR')
            raw.rename_channels(RENAME_DICT)
            raw.reorder_channels(FINAL_CHANNELS)

            if args.l_freq is not None or args.h_freq is not None:
                raw.filter(l_freq=args.l_freq, h_freq=args.h_freq, verbose='ERROR')

            if args.notch_freq is not None:
                raw.notch_filter(freqs=args.notch_freq, verbose='ERROR')

            # Resample
            if raw.info['sfreq'] != args.target_freq:
                raw.resample(args.target_freq, npad='auto', verbose='ERROR')

            # Convert unit from µV to V if requested
            if args.raw_unit == 'uV':
                raw._data /= 1e6   # µV -> V

            # Save as EDF
            export_raw(out_edf, raw, fmt='edf', overwrite=True, verbose='ERROR')

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
        base = os.path.splitext(fname)[0]

        # Retrieve split and label logic
        if fname not in file_split_map:
            logging.warning(f"{fname} not in mapping dict. Skipping.")
            continue
            
        split_name, label_name = file_split_map[fname]

        # Create output directory: Segment/train/MDD/ etc.
        save_dir = os.path.join(segment_dir, split_name, label_name)
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
            segment_reshaped = segment.reshape(
                len(FINAL_CHANNELS),
                int(args.seg_len),
                args.target_freq
            ).astype(np.float32)

            # Keep original base name and append segment index
            out_name = f"{base}_{seg_idx}.npy"
            out_path = os.path.join(save_dir, out_name)
            np.save(out_path, segment_reshaped)
            seg_count += 1

        total_segments += seg_count
        logging.info(f"{fname} -> {seg_count} segments ({split_name}/{label_name})")

    logging.info("=" * 50)
    logging.info(f"All done! Total segments created: {total_segments}")

if __name__ == "__main__":
    main()