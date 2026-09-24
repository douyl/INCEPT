import os
import argparse
import glob
import logging
import shutil
import numpy as np
import mne
from mne.export import export_raw
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# Channel Definitions (At the top of your script)
# ==============================================================================
FINAL_CHANNELS = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']

# ISRUC Label mapping
LABEL2ID = {'0': 0, '1': 1, '2': 2, '3': 3, '5': 4}

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess and segment ISRUC_S1 dataset.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing raw .rec and .txt files (e.g., .../group1).')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Root directory. Preprocessed EDFs go to Preprocess/, segments to Segment/.')
    parser.add_argument('--raw_unit', type=str, choices=['uV', 'V'], default='V',
                        help='Unit of the raw data. If "uV", data are converted to volts.')
    parser.add_argument('--target_freq', type=int, default=250,
                        help='Target sampling frequency after resampling (Hz).')
    parser.add_argument('--seg_len', type=int, default=30,
                        help='Length of a single sleep stage epoch in seconds.')
    parser.add_argument('--seq_len', type=int, default=20,
                        help='Number of epochs in one sequence.')
    
    parser.add_argument('--l_freq', type=float, default=0.3,
                        help='Lower pass-band edge in Hz. If None, no high-pass filtering.')
    parser.add_argument('--h_freq', type=float, default=35.0,
                        help='Upper pass-band edge in Hz. If None, no low-pass filtering.')
    parser.add_argument('--notch_freq', type=float, default=50.0,
                        help='Frequency to notch filter. If None, no notch filter.')

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

    logging.info("=== ISRUC Dataset Preprocessing + Segmentation ===")
    logging.info(f"Input dir      : {args.input_dir}")
    logging.info(f"Output dir     : {args.output_dir}")
    logging.info(f"Target freq    : {args.target_freq} Hz")
    logging.info(f"Epoch length   : {args.seg_len} s")
    logging.info(f"Sequence length: {args.seq_len} epochs")
    logging.info(f"Bandpass       : {args.l_freq} - {args.h_freq} Hz")
    logging.info(f"Notch freq     : {args.notch_freq} Hz")
    logging.info("=" * 50)

    # ------------------------------------------------------------------
    # Step 1: Preprocess raw .rec files and save to Preprocess/
    # ------------------------------------------------------------------
    # Scan subject folders (1 to 100)
    subject_dirs = sorted([d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))])
    logging.info(f"Found {len(subject_dirs)} subject directories. Starting preprocessing...")

    for subj_id in subject_dirs:
        subj_path = os.path.join(args.input_dir, subj_id)
        rec_file = os.path.join(subj_path, f"{subj_id}.rec")
        out_edf = os.path.join(preproc_dir, f"{subj_id}.edf")

        if not os.path.exists(rec_file):
            logging.warning(f"Raw file {rec_file} not found. Skipping.")
            continue

        if os.path.exists(out_edf):
            logging.info(f"Preprocessed file exists, skipping step 1 for: {subj_id}")
            continue

        try:
            # Copy to temp file to avoid mne read issues with non-standard extensions
            temp_edf = os.path.join(args.output_dir, f"temp_{subj_id}.edf")
            shutil.copyfile(rec_file, temp_edf)

            raw = mne.io.read_raw_edf(temp_edf, preload=True, verbose='ERROR')
            os.remove(temp_edf) # Clean up temp file immediately

            if str(subj_id) == '8':
                logging.warning(f"Subject {subj_id} is missing F3/F4 electrodes. Skipping completely.")
                continue
            if str(subj_id) == '40':
                # Subject 40 的通道顺序混乱，按 0-based 索引手动映射
                # 目标顺序: F3, C3, O1, F4, C4, O2
                # 对应位置: 第10(idx 9), 第7(idx 6), 第8(idx 7), 第9(idx 8), 第5(idx 4), 第6(idx 5)
                target_orig_channels = [
                    raw.ch_names[9], raw.ch_names[6], raw.ch_names[7], 
                    raw.ch_names[8], raw.ch_names[4], raw.ch_names[5]
                ]
            else:
                # 正常的受试者直接取 2:8
                target_orig_channels = raw.ch_names[2:8]
            logging.info(f"Subject {subj_id} extracted channels: {target_orig_channels}")
            raw.pick_channels(target_orig_channels, verbose='ERROR')
            dynamic_rename_dict = {orig: final for orig, final in zip(target_orig_channels, [
                    'F3', 'C3', 'O1', 'F4', 'C4', 'O2'
                ])}
            raw.rename_channels(dynamic_rename_dict)
            raw.reorder_channels(FINAL_CHANNELS)

            # Filtering
            if args.l_freq is not None or args.h_freq is not None:
                raw.filter(l_freq=args.l_freq, h_freq=args.h_freq, fir_design='firwin', verbose='ERROR')

            if args.notch_freq is not None:
                raw.notch_filter(freqs=args.notch_freq, verbose='ERROR')

            # Resample
            if raw.info['sfreq'] != args.target_freq:
                raw.resample(args.target_freq, npad='auto', verbose='ERROR')

            # Convert unit to V if requested
            if args.raw_unit == 'uV':
                raw._data /= 1e6

            # Save as standard EDF
            export_raw(out_edf, raw, fmt='edf', overwrite=True, verbose='ERROR')

        except Exception as e:
            logging.error(f"Error processing subject {subj_id}: {e}")
            if os.path.exists(temp_edf):
                os.remove(temp_edf)



    # ------------------------------------------------------------------
    # Step 2: Segment all preprocessed EDFs and save to Segment/
    # ------------------------------------------------------------------
    preproc_files = sorted(glob.glob(os.path.join(preproc_dir, '*.edf')))
    if not preproc_files:
        logging.error("No preprocessed files found. Segmentation aborted.")
        return

    pts_per_epoch = int(args.target_freq * args.seg_len)
    total_seqs_created = 0

    logging.info(f"Starting segmentation of {len(preproc_files)} preprocessed files...")
    
    for fpath in preproc_files:
        subj_id = os.path.splitext(os.path.basename(fpath))[0]
        label_file = os.path.join(args.input_dir, subj_id, f"{subj_id}_1.txt")

        if not os.path.exists(label_file):
            logging.warning(f"Label file {label_file} not found. Skipping.")
            continue

        try:
            # Read preprocessed EDF
            raw = mne.io.read_raw_edf(fpath, preload=True, verbose='ERROR')
            data = raw.get_data() # Shape: (6, total_points)

            # Read and map labels
            labels = []
            with open(label_file, 'r') as lf:
                for line in lf:
                    val = line.strip()
                    if val in LABEL2ID:
                        labels.append(LABEL2ID[val])
            
            # Align signal length with label length
            max_epochs = data.shape[1] // pts_per_epoch
            num_epochs = min(max_epochs, len(labels))

            if num_epochs == 0:
                logging.warning(f"Valid epochs = 0 for {subj_id}. Skipping.")
                continue

            # Crop data to exact epoch boundaries
            data = data[:, :num_epochs * pts_per_epoch]
            # Reshape into epochs: (6, num_epochs, pts_per_epoch) -> (num_epochs, 6, pts_per_epoch)
            data_epochs = data.reshape(len(FINAL_CHANNELS), num_epochs, pts_per_epoch)
            data_epochs = np.transpose(data_epochs, (1, 0, 2))
            labels_epochs = np.array(labels[:num_epochs])

            # Group into sequences (e.g., sequences of 20 epochs)
            num_seqs = num_epochs // args.seq_len
            if num_seqs == 0:
                logging.warning(f"Not enough epochs to form a sequence for {subj_id}. Skipping.")
                continue

            # Drop remaining epochs at the end that don't form a full sequence
            data_seqs = data_epochs[:num_seqs * args.seq_len]
            labels_seqs = labels_epochs[:num_seqs * args.seq_len]

            # Final reshape: (num_seqs, seq_len=20, channels=6, 30_sec, 250_hz)
            data_seqs = data_seqs.reshape(
                num_seqs, 
                args.seq_len, 
                len(FINAL_CHANNELS), 
                args.seg_len, 
                args.target_freq
            ).astype(np.float32)

            labels_seqs = labels_seqs.reshape(num_seqs, args.seq_len)

            # Define save directories (Sequence to sequence paradigm)
            seq_out_dir = os.path.join(segment_dir, 'sequence', subj_id)
            label_out_dir = os.path.join(segment_dir, 'label', subj_id)
            os.makedirs(seq_out_dir, exist_ok=True)
            os.makedirs(label_out_dir, exist_ok=True)

            # Save each sequence as an independent file
            for i in range(num_seqs):
                np.save(os.path.join(seq_out_dir, f"{subj_id}_{i}.npy"), data_seqs[i])
                np.save(os.path.join(label_out_dir, f"{subj_id}_{i}.npy"), labels_seqs[i])

            total_seqs_created += num_seqs
            logging.info(f"Subject {subj_id} -> {num_seqs} sequences generated.")

        except Exception as e:
            logging.error(f"Failed to segment {subj_id}: {e}")

    total_segments = total_seqs_created * args.seq_len
    logging.info("=" * 50)
    logging.info(f"All done! Total sequences created: {total_seqs_created} (Total segments: {total_segments})")

if __name__ == "__main__":
    main()