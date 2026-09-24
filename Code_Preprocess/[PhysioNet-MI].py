import os
import argparse
import glob
import logging
import re
import numpy as np
import mne
from mne.export import export_raw
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# Channel definitions
# ==============================================================================
FINAL_CHANNELS = [
    'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 
    'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 
    'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FT8', 'T7', 'T8', 
    'T9', 'T10', 'TP7', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 
    'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz'
]

# Original names in PhysioNet raw EDF
RAW_CHANNELS = [
    'Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..',
    'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.', 'Af7.', 'Af3.', 'Afz.', 
    'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 
    'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..', 'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 
    'Poz.', 'Po4.', 'Po8.', 'O1..', 'Oz..', 'O2..', 'Iz..'
]

RENAME_DICT = dict(zip(RAW_CHANNELS, FINAL_CHANNELS))
TARGET_TASKS = ['04', '06', '08', '10', '12', '14']

def parse_args():
    parser = argparse.ArgumentParser(description='PhysioNet MI Preprocessing and Segmentation.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing raw .edf files.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Root directory for processed data.')
    parser.add_argument('--raw_unit', type=str, choices=['uV', 'V'], default='V',
                        help='Unit of the raw data. If "uV", data are converted to volts.')
    parser.add_argument('--target_freq', type=int, default=250, 
                        help='Target sampling frequency after resampling (Hz).')
    parser.add_argument('--notch_freq', type=float, default=60.0, 
                        help='Line frequency for notch filter.')
    parser.add_argument('--l_freq', type=float, default=0.1, 
                        help='High-pass filter cutoff.')
    parser.add_argument('--h_freq', type=float, default=None, 
                        help='Low-pass filter cutoff (None for max).')
    parser.add_argument('--skip_reference', action='store_true', 
                        help='Skip re-referencing to average.')
    return parser.parse_args()

def extract_subject_and_task(filename):
    """Extract Subject ID and Task ID from filename (e.g., 'S001R04.edf')."""
    match = re.search(r'(S\d+)R(\d+)', filename)
    if match:
        return match.group(1), match.group(2)
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

    logging.info("=== PhysioNet MI Preprocessing & Segmentation ===")
    
    # Find and filter target files
    all_files = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.edf"), recursive=True))
    target_files = [f for f in all_files if extract_subject_and_task(os.path.basename(f))[1] in TARGET_TASKS]

    if not target_files:
        logging.error(f"No valid MI .edf files found in {args.input_dir}")
        return

    logging.info(f"Found {len(target_files)} target MI files. Starting pipeline...")

    # Global segment counter
    total_segments = 0

    # Process files
    for fpath in target_files:
        fname = os.path.basename(fpath)
        subject_id, task_id = extract_subject_and_task(fname)

        preproc_subj_dir = os.path.join(args.output_dir, "Preprocess", subject_id)
        segment_subj_dir = os.path.join(args.output_dir, "Segment", subject_id)
        os.makedirs(preproc_subj_dir, exist_ok=True)
        os.makedirs(segment_subj_dir, exist_ok=True)
        
        out_edf_path = os.path.join(preproc_subj_dir, fname)

        if os.path.exists(out_edf_path):
            logging.info(f"File exists, skipping: {fname}")
            continue

        try:
            # === Part A: Preprocessing ===
            raw = mne.io.read_raw_edf(fpath, preload=True, verbose='ERROR')
            
            # Select and rename channels
            available_chs = [ch for ch in raw.ch_names if ch in RENAME_DICT]
            raw.pick_channels(available_chs, ordered=False, verbose='ERROR')
            raw.rename_channels({k: v for k, v in RENAME_DICT.items() if k in raw.ch_names})

            # Interpolate bads
            if len(raw.info['bads']) > 0:
                raw.interpolate_bads(verbose='ERROR')

            # Montage & Filtering
            raw.set_montage(mne.channels.make_standard_montage('standard_1020'), on_missing='warn', verbose='ERROR')
            raw.filter(l_freq=args.l_freq, h_freq=args.h_freq, verbose='ERROR')
            raw.notch_filter(freqs=args.notch_freq, verbose='ERROR')

            if raw.info['sfreq'] != args.target_freq:
                raw.resample(args.target_freq, npad='auto', verbose='ERROR')
                
            if not args.skip_reference:
                raw.set_eeg_reference(ref_channels='average', projection=False, verbose='ERROR')

            # Reorder channels exactly as FINAL_CHANNELS
            final_picks = [ch for ch in FINAL_CHANNELS if ch in raw.ch_names]
            raw.reorder_channels(final_picks)

            # Convert unit from µV to V if requested
            if args.raw_unit == 'uV':
                raw._data /= 1e6   # µV -> V

            export_raw(out_edf_path, raw, fmt='edf', overwrite=True, verbose='ERROR')
            
            # === Part B: Segmentation (Epoching) ===
            events_from_annot, event_dict = mne.events_from_annotations(raw, verbose='ERROR')
            
            tmax = 4.0 - 1.0 / raw.info['sfreq']
            epochs = mne.Epochs(raw, events_from_annot, event_dict, 
                                tmin=0, tmax=tmax, baseline=None, 
                                preload=True, verbose='ERROR')
            
            data = epochs.get_data()
            events = epochs.events[:, 2]

            # Slice the last 4 seconds
            n_points = int(4 * args.target_freq)
            if data.shape[2] < n_points:
                logging.warning(f"File {fname} too short ({data.shape[2]} pts). Skipping segmentation.")
                continue
                
            data = data[:, :, -n_points:]
            
            # Reshape
            bz, ch_nums, _ = data.shape
            data = data.reshape(bz, ch_nums, 4, args.target_freq)

            # Save segments
            save_count = 0
            for i, (sample, event) in enumerate(zip(data, events)):
                if event == 1:
                    continue  # Skip Rest (T0)
                
                label = event - 2 if task_id in ['04', '08', '12'] else event
                
                label_dir = os.path.join(segment_subj_dir, str(label))
                os.makedirs(label_dir, exist_ok=True)
                
                save_name = f"{fname.replace('.edf', '')}_trial{i}.npy"
                np.save(os.path.join(label_dir, save_name), sample)
                save_count += 1
            
            # Accumulate global segments count
            total_segments += save_count
            logging.info(f"Processed: {fname} -> {save_count} segments")

        except Exception as e:
            logging.error(f"Error processing {fname}: {e}")

    logging.info("=" * 50)
    logging.info(f"All done! Total segments created: {total_segments}")

if __name__ == "__main__":
    main()