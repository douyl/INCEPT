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
ORIG_CHANNELS = [
    "EEG Fp1", "EEG F3", "EEG C3", "EEG P3", "EEG O1", "EEG F7", "EEG T3", "EEG T5", "EEG Fc1", "EEG Fc5",
    "EEG Cp1", "EEG Cp5", "EEG F9", "EEG Fz", "EEG Cz", "EEG Pz", "EEG Fp2", "EEG F4", "EEG C4", "EEG P4",
    "EEG O2", "EEG F8", "EEG T4", "EEG T6", "EEG Fc2", "EEG Fc6", "EEG Cp2", "EEG Cp6", "EEG F10"
]

FINAL_CHANNELS = [
    'Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5', 'Fc1', 'Fc5',
    'Cp1', 'Cp5', 'F9', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4',
    'O2', 'F8', 'T4', 'T6', 'Fc2', 'Fc6', 'Cp2', 'Cp6', 'F10'
]

# ==============================================================================
# Seizure Annotations (Siena Dataset)
# ==============================================================================
SEIZURE_RECORDS = [
    # PN00
    {'file': 'PN00-1.edf', 'reg_start': '19.39.33', 'start_time': '19.58.36', 'end_time': '19.59.46'},
    {'file': 'PN00-2.edf', 'reg_start': '02.18.17', 'start_time': '02.38.37', 'end_time': '02.39.31'},
    {'file': 'PN00-3.edf', 'reg_start': '18.15.44', 'start_time': '18.28.29', 'end_time': '19.29.29'},
    {'file': 'PN00-4.edf', 'reg_start': '20.51.43', 'start_time': '21.08.29', 'end_time': '21.09.43'},
    {'file': 'PN00-5.edf', 'reg_start': '22.22.04', 'start_time': '22.37.08', 'end_time': '22.38.15'},
    # PN01
    {'file': 'PN01-1.edf', 'reg_start': '19.00.44', 'start_time': '21.51.02', 'end_time': '21.51.56'},
    {'file': 'PN01-1.edf', 'reg_start': '19.00.44', 'start_time': '07.53.17', 'end_time': '07.54.31'},
    # PN03
    {'file': 'PN03-1.edf', 'reg_start': '22.44.37', 'start_time': '09.29.10', 'end_time': '09.31.01'},
    {'file': 'PN03-2.edf', 'reg_start': '21.31.04', 'start_time': '07.13.05', 'end_time': '07.15.18'},
    # PN05
    {'file': 'PN05-2.edf', 'reg_start': '06.46.02', 'start_time': '08.45.25', 'end_time': '08.46.00'},
    {'file': 'PN05-3.edf', 'reg_start': '06.01.23', 'start_time': '07.55.19', 'end_time': '07.55.49'},
    {'file': 'PN05-4.edf', 'reg_start': '06.38.35', 'start_time': '07.38.43', 'end_time': '07.39.22'},
    # PN06
    {'file': 'PN06-1.edf', 'reg_start': '04.21.22', 'start_time': '05.54.25', 'end_time': '05.55.29'},
    {'file': 'PN06-2.edf', 'reg_start': '21.11.29', 'start_time': '23.39.09', 'end_time': '23.40.18'},
    {'file': 'PN06-3.edf', 'reg_start': '06.25.51', 'start_time': '08.10.26', 'end_time': '08.11.08'},
    {'file': 'PN06-4.edf', 'reg_start': '11.16.09', 'start_time': '12.55.08', 'end_time': '12.56.11'},
    {'file': 'PN06-5.edf', 'reg_start': '13.24.41', 'start_time': '14.44.24', 'end_time': '14.45.08'},
    # PN07
    {'file': 'PN07-1.edf', 'reg_start': '23.18.10', 'start_time': '05.25.49', 'end_time': '05.26.51'},
    # PN09
    {'file': 'PN09-1.edf', 'reg_start': '14.08.54', 'start_time': '16.09.43', 'end_time': '16.11.03'},
    {'file': 'PN09-2.edf', 'reg_start': '15.02.09', 'start_time': '17.00.56', 'end_time': '17.01.55'},
    {'file': 'PN09-3.edf', 'reg_start': '14.20.23', 'start_time': '16.20.44', 'end_time': '16.21.48'},
    # PN10
    {'file': 'PN10-1.edf', 'reg_start': '05.40.05', 'start_time': '07.45.50', 'end_time': '07.46.59'},
    {'file': 'PN10-2.edf', 'reg_start': '09.30.15', 'start_time': '11.40.13', 'end_time': '11.41.04'},
    {'file': 'PN10-3.edf', 'reg_start': '13.33.18', 'start_time': '15.43.53', 'end_time': '15.45.02'},
    {'file': 'PN10-4.5.6.edf', 'reg_start': '12.11.21', 'start_time': '12.49.50', 'end_time': '12.49.55'},
    {'file': 'PN10-4.5.6.edf', 'reg_start': '12.11.21', 'start_time': '14.00.25', 'end_time': '14.00.44'},
    {'file': 'PN10-4.5.6.edf', 'reg_start': '12.11.21', 'start_time': '15.18.26', 'end_time': '15.19.23'},
    {'file': 'PN10-7.8.9.edf', 'reg_start': '16.49.25', 'start_time': '17.35.13', 'end_time': '17.36.01'},
    {'file': 'PN10-7.8.9.edf', 'reg_start': '16.49.25', 'start_time': '18.20.24', 'end_time': '18.20.42'},
    {'file': 'PN10-7.8.9.edf', 'reg_start': '16.49.25', 'start_time': '20.24.48', 'end_time': '20.25.03'},
    {'file': 'PN10-10.edf', 'reg_start': '08.45.22', 'start_time': '10.58.19', 'end_time': '10.58.33'},
    # PN11
    {'file': 'PN11-1.edf', 'reg_start': '11.31.25', 'start_time': '13.37.19', 'end_time': '13.38.14'},
    # PN12
    {'file': 'PN12-1.2.edf', 'reg_start': '15.51.31', 'start_time': '16.13.23', 'end_time': '16.14.26'},
    {'file': 'PN12-1.2.edf', 'reg_start': '15.51.31', 'start_time': '18.31.01', 'end_time': '18.32.09'},
    {'file': 'PN12-3.edf', 'reg_start': '08.42.35', 'start_time': '08.55.27', 'end_time': '08.57.03'},
    {'file': 'PN12-4.edf', 'reg_start': '15.59.19', 'start_time': '18.42.51', 'end_time': '18.43.54'},
    # PN13
    {'file': 'PN13-1.edf', 'reg_start': '08.24.28', 'start_time': '10.22.10', 'end_time': '10.22.58'},
    {'file': 'PN13-2.edf', 'reg_start': '06.55.02', 'start_time': '08.55.51', 'end_time': '08.56.56'},
    {'file': 'PN13-3.edf', 'reg_start': '12.00.01', 'start_time': '14.05.54', 'end_time': '14.08.25'},
    # PN14
    {'file': 'PN14-1.edf', 'reg_start': '11.44.58', 'start_time': '13.46.00', 'end_time': '13.46.27'},
    {'file': 'PN14-2.edf', 'reg_start': '15.50.13', 'start_time': '17.54.52', 'end_time': '17.55.04'},
    {'file': 'PN14-3.edf', 'reg_start': '16.17.45', 'start_time': '21.10.05', 'end_time': '21.10.46'},
    {'file': 'PN14-4.edf', 'reg_start': '14.18.30', 'start_time': '15.49.33', 'end_time': '15.50.56'},
    # PN16
    {'file': 'PN16-1.edf', 'reg_start': '20.45.21', 'start_time': '22.45.05', 'end_time': '22.47.08'},
    {'file': 'PN16-2.edf', 'reg_start': '00.53.55', 'start_time': '03.16.49', 'end_time': '03.18.36'},
    # PN17
    {'file': 'PN17-1.edf', 'reg_start': '20.14.28', 'start_time': '22.34.48', 'end_time': '22.35.58'},
    {'file': 'PN17-2.edf', 'reg_start': '13.52.18', 'start_time': '16.01.09', 'end_time': '16.02.32'}
]


def time_to_samples(time_str, start_time_str, sampling_rate):
    """Convert HH.MM.SS time string to sample indices."""
    time_str = time_str.replace(':', '.')
    start_time_str = start_time_str.replace(':', '.')

    h, m, s = map(int, time_str.split('.'))
    start_h, start_m, start_s = map(int, start_time_str.split('.'))

    if h < start_h:
        h += 24

    delta = (h - start_h) * 3600 + (m - start_m) * 60 + (s - start_s)
    return int(delta * sampling_rate)


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess and segment Siena Scalp EEG dataset.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing raw Siena .edf files.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Root directory. Preprocessed EDFs go to Preprocess/, segments to Segment/.')
    parser.add_argument('--raw_unit', type=str, choices=['uV', 'V'], default='V',
                        help='Unit of the raw data. If "uV", data are converted to volts.')
    parser.add_argument('--target_freq', type=int, default=250,
                        help='Target sampling frequency after resampling (Hz).')
    parser.add_argument('--seg_len', type=float, default=10,
                        help='Length of each segment in seconds.')
    
    parser.add_argument('--l_freq', type=float, default=0.1,
                        help='Lower pass-band edge in Hz.')
    parser.add_argument('--h_freq', type=float, default=75.0,
                        help='Upper pass-band edge in Hz.')
    parser.add_argument('--notch_freq', type=float, default=50.0,
                        help='Frequency to notch filter.')

    return parser.parse_args()


def main():
    args = parse_args()

    # Base output directories
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

    logging.info("=== Siena Dataset Preprocessing + Segmentation ===")
    logging.info(f"Input dir      : {args.input_dir}")
    logging.info(f"Output dir     : {args.output_dir}")
    logging.info(f"Raw unit       : {args.raw_unit}")
    logging.info(f"Target freq    : {args.target_freq} Hz")
    logging.info(f"Segment length : {args.seg_len} s")
    logging.info(f"Bandpass       : {args.l_freq} - {args.h_freq} Hz")
    logging.info(f"Notch freq     : {args.notch_freq} Hz")
    logging.info("=" * 50)

    # Find all EDF files in input directory (recursively)
    raw_files = sorted(glob.glob(os.path.join(args.input_dir, '**', '*.edf'), recursive=True))
    if not raw_files:
        logging.error(f"No valid .edf files found in {args.input_dir}")
        return

    # ------------------------------------------------------------------
    # Step 1: Preprocess each raw EDF and save to Preprocess/Subject_ID/
    # ------------------------------------------------------------------
    logging.info(f"Found {len(raw_files)} raw files. Starting preprocessing...")
    
    for fpath in raw_files:
        fname = os.path.basename(fpath)
        # Extract subject ID (e.g., 'PN00-1.edf' -> 'PN00')
        subject_id = fname.split('-')[0]
        
        # Create Preprocess/PN00 folder
        subj_preproc_dir = os.path.join(preproc_dir, subject_id)
        os.makedirs(subj_preproc_dir, exist_ok=True)
        
        out_edf = os.path.join(subj_preproc_dir, fname)

        if os.path.exists(out_edf):
            logging.info(f"Preprocessed file exists, skipping step 1: {subject_id}/{fname}")
            continue

        try:
            raw = mne.io.read_raw_edf(fpath, preload=True, verbose='ERROR')

            # 1. Channel Matching (Case Insensitive)
            existing_ch_lower = {ch.lower(): ch for ch in raw.ch_names}
            
            selected_real_names = []
            missing_channels = []
            
            for orig_ch in ORIG_CHANNELS:
                orig_lower = orig_ch.lower()
                if orig_lower in existing_ch_lower:
                    selected_real_names.append(existing_ch_lower[orig_lower])
                else:
                    missing_channels.append(orig_ch)
                    
            if missing_channels:
                logging.error(f"Missing required channels in {fname}: {missing_channels}. Skipping file.")
                continue

            # 2. Pick, Rename, and Reorder
            raw.pick_channels(selected_real_names, verbose='ERROR')
            
            # Create mapping from actual names in file to our final target names
            rename_dict = {real_name: final_name for real_name, final_name in zip(selected_real_names, FINAL_CHANNELS)}
            raw.rename_channels(rename_dict)
            raw.reorder_channels(FINAL_CHANNELS)

            # 3. Filtering
            if args.l_freq is not None or args.h_freq is not None:
                raw.filter(l_freq=args.l_freq, h_freq=args.h_freq, verbose='ERROR')

            if args.notch_freq is not None:
                raw.notch_filter(freqs=args.notch_freq, verbose='ERROR')

            # 4. Resampling
            if raw.info['sfreq'] != args.target_freq:
                raw.resample(args.target_freq, npad='auto', verbose='ERROR')

            # 5. Unit Conversion
            if args.raw_unit == 'uV':
                raw._data /= 1e6   # µV -> V

            # Save as EDF
            export_raw(out_edf, raw, fmt='edf', overwrite=True, verbose='ERROR')
            logging.info(f"Successfully preprocessed: {subject_id}/{fname}")

        except Exception as e:
            logging.error(f"Error processing {fname}: {e}")

    # ------------------------------------------------------------------
    # Step 2: Segment all preprocessed EDFs
    # Save to Segment/Subject_ID/normal/ or Segment/Subject_ID/seizure/
    # ------------------------------------------------------------------
    preproc_files = sorted(glob.glob(os.path.join(preproc_dir, '**', '*.edf'), recursive=True))
    if not preproc_files:
        logging.error("No preprocessed files found. Segmentation aborted.")
        return

    points_per_seg = int(args.target_freq * args.seg_len)
    total_segments_0 = 0
    total_segments_1 = 0

    logging.info(f"Starting segmentation of {len(preproc_files)} preprocessed files...")
    
    for fpath in preproc_files:
        fname = os.path.basename(fpath)
        base = os.path.splitext(fname)[0]
        # Extract subject ID (e.g., 'PN00-1.edf' -> 'PN00')
        subject_id = fname.split('-')[0]

        try:
            raw = mne.io.read_raw_edf(fpath, preload=True, verbose='ERROR')
        except Exception as e:
            logging.error(f"Failed to read {fname}: {e}")
            continue

        data = raw.get_data()   # shape (29, times), already in V
        total_pts = data.shape[1]

        # Get seizure events for current file
        file_seizures = [sz for sz in SEIZURE_RECORDS if sz["file"] == fname]
        seizure_samples = []
        for sz in file_seizures:
            start_idx = time_to_samples(sz["start_time"], sz["reg_start"], args.target_freq)
            end_idx = time_to_samples(sz["end_time"], sz["reg_start"], args.target_freq)
            seizure_samples.append((start_idx, end_idx))

        segments_to_save = [] # List of tuples: (segment_data, label)

        # 2.1 Regular segmentation (Non-overlapping)
        for i in range(0, total_pts, points_per_seg):
            if i + points_per_seg <= total_pts:
                seg = data[:, i:i + points_per_seg]
                label = 0
                for sz_start, sz_end in seizure_samples:
                    # If the 10s segment intersects with a seizure interval
                    if (i < sz_start < i + points_per_seg) or (i < sz_end < i + points_per_seg) or (sz_start <= i and sz_end >= i + points_per_seg):
                        label = 1
                        break
                segments_to_save.append((seg, label))

        # 2.2 Seizure-enhanced segmentation (Oversampling with 5s sliding window)
        step_enhanced = int(5 * args.target_freq) # 5 seconds step
        for sz_start, sz_end in seizure_samples:
            # Pad by 1 second on each side (or target_freq samples)
            start_enh = max(0, sz_start - args.target_freq)
            end_enh = min(total_pts, sz_end + args.target_freq)

            for i in range(start_enh, end_enh, step_enhanced):
                if i + points_per_seg <= total_pts:
                    seg = data[:, i:i + points_per_seg]
                    segments_to_save.append((seg, 1))

        # Save generated segments
        seg_count_0 = 0
        seg_count_1 = 0
        
        for idx, (seg, label) in enumerate(segments_to_save):
            # Reshape to Mumtaz style: (channels, seconds, freq)
            seg_reshaped = seg.reshape(
                len(FINAL_CHANNELS),
                int(args.seg_len),
                args.target_freq
            ).astype(np.float32)

            # Determine label string for folder mapping
            label_str = 'normal' if label == 0 else 'seizure'
            
            # Create Segment/PN00/normal or Segment/PN00/seizure folders dynamically
            out_dir = os.path.join(segment_dir, subject_id, label_str)
            os.makedirs(out_dir, exist_ok=True)
            
            # Use .npy extension for standard 3D array saving
            out_name = f"{base}_{idx}.npy"
            out_path = os.path.join(out_dir, out_name)
            
            np.save(out_path, seg_reshaped)
            
            if label == 0:
                seg_count_0 += 1
            else:
                seg_count_1 += 1

        total_segments_0 += seg_count_0
        total_segments_1 += seg_count_1
        logging.info(f"{subject_id}/{fname} -> {seg_count_0} Normal, {seg_count_1} Seizure segments.")

    logging.info("=" * 50)
    logging.info("All processing completed!")
    logging.info(f"Total Normal segments  : {total_segments_0}")
    logging.info(f"Total Seizure segments : {total_segments_1}")
    logging.info(f"Grand Total            : {total_segments_0 + total_segments_1}")

if __name__ == "__main__":
    main()