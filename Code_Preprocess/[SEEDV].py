import os
import argparse
import glob
import logging
import numpy as np
import mne
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# Channel definitions & Dataset Configurations
# ==============================================================================
FINAL_CHANNELS = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 
    'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 
    'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 
    'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2'
]

LABEL_MAP = {
    0: 'Disgust',
    1: 'Fear',
    2: 'Sad',
    3: 'Neutral',
    4: 'Happy'
}

TRIAL_SPLIT_MAP = {
    'train': [0, 1, 2, 3, 4],
    'val':   [5, 6, 7, 8, 9],
    'eval':  [10, 11, 12, 13, 14]
}

# Trial times (in seconds) for each session
TRIALS_OF_SESSIONS = {
    '1': {'start': [30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204],
          'end': [102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359]},
    '2': {'start': [30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741],
          'end': [267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817]},
    '3': {'start': [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888],
          'end': [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066]},
}

# Label sequences for each session
LABELS_OF_SESSIONS = {
    '1': [4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0],
    '2': [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0],
    '3': [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0],
}

def parse_args():
    parser = argparse.ArgumentParser(description='SEED-V EEG Preprocessing and Segmentation.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing raw .cnt files.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Root directory for processed data.')
    parser.add_argument('--montage_file', type=str, default='/public_bme2/grpdgshen/douyl/EEG_Dataset/SEED/SEED_RawData/SEED-V/channel_62_pos.locs',
                        help='Path to channel_62_pos.locs file.')
    parser.add_argument('--target_freq', type=int, default=250, 
                        help='Target sampling frequency after resampling (Hz).')
    parser.add_argument('--line_freq', type=float, default=50.0, 
                        help='Line frequency for notch filter (Hz).')
    parser.add_argument('--l_freq', type=float, default=0.1, 
                        help='High-pass filter cutoff.')
    parser.add_argument('--h_freq', type=float, default=50.0, 
                        help='Low-pass filter cutoff.')
    parser.add_argument('--skip_reference', action='store_true', 
                        help='Skip re-referencing to average.')
    parser.add_argument('--seg_len', type=int, default=4, 
                        help='Length of each segment in seconds.')
    parser.add_argument('--raw_unit', type=str, choices=['uV', 'V'], default='V',
                        help='Unit of the raw data. If "uV", data are converted to volts.')
    return parser.parse_args()


def extract_subject_and_session(filename):
    """
    Extract Subject ID and Session ID by splitting the filename with '_'.
    Assumes format like '1_1_20180508.cnt'.
    """
    parts = filename.split('_')
    if len(parts) >= 2:
        return parts[0], parts[1]
    return None, None


def main():
    args = parse_args()

    # Setup directories and logging
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'logs.txt'), mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    logging.info("=== SEED-V Preprocessing & Segmentation ===")
    logging.info(f"Input dir      : {args.input_dir}")
    logging.info(f"Output dir     : {args.output_dir}")
    logging.info(f"Segment length : {args.seg_len} s")
    
    # Find raw files strictly in input_dir (no subdirectories)
    search_pattern = os.path.join(args.input_dir, "*.cnt")
    all_files = sorted(glob.glob(search_pattern))
    
    if not all_files:
        logging.error(f"No .cnt files found in {args.input_dir}")
        return

    logging.info(f"Found {len(all_files)} target .cnt files. Starting pipeline...")

    # Global segment counter
    total_segments = 0
    points_per_seg = int(args.seg_len * args.target_freq)

    # Process files
    for fpath in all_files:
        fname = os.path.basename(fpath)
        sub_id, sess_id = extract_subject_and_session(fname)

        if not sub_id or not sess_id or sess_id not in TRIALS_OF_SESSIONS:
            logging.warning(f"Could not parse valid Subject/Session from {fname}. Skipping.")
            continue

        preproc_dir = os.path.join(args.output_dir, "Preprocess")
        segment_dir = os.path.join(args.output_dir, "Segment")
        os.makedirs(preproc_dir, exist_ok=True)
        
        # We save Preprocessed as .fif in MNE
        out_fif_path = os.path.join(preproc_dir, fname.replace('.cnt', '.fif'))

        try:
            # ==========================================
            # Part A: Preprocessing
            # ==========================================
            raw = mne.io.read_raw_cnt(fpath, preload=True, verbose='ERROR')
            
            # Standardize channel names
            rename_dict = {ch: ch.replace('FP', 'Fp').replace('Z', 'z') for ch in raw.ch_names}
            raw.rename_channels(rename_dict)

            # Handle Missing Channels (Zero-pad for Interpolation)
            missing_chs = [c for c in FINAL_CHANNELS if c not in raw.ch_names]
            if missing_chs:
                logging.warning(f"{fname} missing channels: {missing_chs}. Padding with zeros.")
                for missing_ch in missing_chs:
                    info_zeros = mne.create_info([missing_ch], raw.info['sfreq'], 'eeg')
                    raw_zeros = mne.io.RawArray(np.zeros((1, raw.n_times)), info_zeros, verbose='ERROR')
                    raw.add_channels([raw_zeros], force_update_info=True)

            # Pick and strictly reorder channels
            # raw.pick_channels(FINAL_CHANNELS)
            raw.pick(FINAL_CHANNELS)
            raw.reorder_channels(FINAL_CHANNELS)
            raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})

            # Set Montage
            if os.path.exists(args.montage_file):
                montage = mne.channels.read_custom_montage(args.montage_file)
                raw.set_montage(montage, on_missing='ignore')
            else:
                logging.warning(f"Montage file not found at {args.montage_file}!")
                raise RuntimeError(f"Failed to set montage: {e}")

            # Interpolate if padded zeros were added
            if missing_chs:
                raw.info['bads'].extend(missing_chs)
                try:
                    raw.interpolate_bads(reset_bads=True, verbose='ERROR')
                    logging.info(f"-> Interpolating missing channels: {missing_chs}", fname)
                except Exception as e:
                    logging.warning(f"Interpolation failed for {fname}: {e}")

            # Filtering
            if args.line_freq:
                raw.notch_filter(freqs=[args.line_freq, 2 * args.line_freq], verbose='ERROR')
            raw.filter(l_freq=args.l_freq, h_freq=args.h_freq, phase='zero-double', verbose='ERROR')

            # Resampling
            if raw.info['sfreq'] != args.target_freq:
                raw.resample(args.target_freq, npad='auto', verbose='ERROR')

            # Re-referencing
            if not args.skip_reference:
                raw.set_eeg_reference(ref_channels='average', projection=False, verbose='ERROR')

            # Unit Conversion: if 'uV', convert down to Volts
            if args.raw_unit == 'uV':
                raw._data /= 1e6

            # Save intermediate preprocessed file
            raw.save(out_fif_path, overwrite=True, verbose='ERROR')



            # ==========================================
            # Part B: Segmentation (Epoching per SEED protocol)
            # ==========================================
            data_matrix = raw.get_data()  # shape: (n_channels, n_times)
            
            curr_trials = TRIALS_OF_SESSIONS[sess_id]
            curr_labels = LABELS_OF_SESSIONS[sess_id]
            file_segments = 0

            # Iterate over all 15 trials for the current session
            for trial_idx in range(15):
                # Determine mode (Train/Val/Eval)
                if trial_idx in TRIAL_SPLIT_MAP['val']:
                    mode = 'val'
                elif trial_idx in TRIAL_SPLIT_MAP['eval']:
                    mode = 'eval'
                else:
                    mode = 'train'

                emotion_name = LABEL_MAP[curr_labels[trial_idx]]
                
                # Fetch time bounds
                start_sec = curr_trials['start'][trial_idx]
                end_sec = curr_trials['end'][trial_idx]
                
                start_idx = int(start_sec * args.target_freq)
                end_idx = int(end_sec * args.target_freq)

                trial_data = data_matrix[:, start_idx:end_idx]
                total_samples = trial_data.shape[1]
                
                # How many non-overlapping segments?
                n_slices = total_samples // points_per_seg
                
                # Output Directory: Segment / Mode / Subject / Emotion
                sub_folder = f"subject_{sub_id}"
                save_dir = os.path.join(segment_dir, mode, sub_folder, emotion_name)
                os.makedirs(save_dir, exist_ok=True)

                for slice_id in range(n_slices):
                    seg_start = slice_id * points_per_seg
                    seg_end = (slice_id + 1) * points_per_seg
                    
                    segment = trial_data[:, seg_start:seg_end]
                    
                    # Reshape to (62, seg_len, target_freq)
                    segment_reshaped = segment.reshape(
                        len(FINAL_CHANNELS), 
                        int(args.seg_len), 
                        args.target_freq
                    ).astype(np.float32)
                    
                    file_name = f"sub{sub_id}_sess{sess_id}_trial{trial_idx}_time{slice_id}.npy"
                    np.save(os.path.join(save_dir, file_name), segment_reshaped)
                    file_segments += 1
                    
            total_segments += file_segments
            logging.info(f"Processed: {fname} -> {file_segments} segments")

        except Exception as e:
            logging.error(f"Error processing {fname}: {e}")

    logging.info("=" * 50)
    logging.info(f"Pipeline Complete! Total segments created: {total_segments}")

if __name__ == "__main__":
    main()