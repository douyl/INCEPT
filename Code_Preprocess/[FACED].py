import os
import argparse
import glob
import logging
import pickle
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# Configuration & Constants
# ==============================================================================
FACED_CHANNELS = [
    'Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6',
    'Cz', 'C3', 'C4', 'T3', 'T4', 'CP1', 'CP2', 'CP5', 'CP6', 'Pz', 'P3', 'P4',
    'T5', 'T6', 'PO3', 'PO4', 'Oz', 'O1', 'O2', 'A2', 'A1'
]

SFREQ = 250  
TARGET_CHANNELS = 32
TOTAL_DURATION_SEC = 30
TOTAL_POINTS = SFREQ * TOTAL_DURATION_SEC  # 30s * 250Hz = 7500 points

# Emotion label mapping (Video ID -> Emotion Label)
VIDEO_LABEL_IDS = np.array([
    0, 0, 0,       # Video 0-2
    1, 1, 1,       # Video 3-5
    2, 2, 2,       # Video 6-8
    3, 3, 3,       # Video 9-11
    4, 4, 4, 4,    # Video 12-15
    5, 5, 5,       # Video 16-18
    6, 6, 6,       # Video 19-21
    7, 7, 7,       # Video 22-24
    8, 8, 8        # Video 25-27
])

EMOTION_NAMES = {
    0: 'Anger', 
    1: 'Disgust', 
    2: 'Fear', 
    3: 'Sadness', 
    4: 'Neutral',
    5: 'Amusement', 
    6: 'Inspiration', 
    7: 'Joy', 
    8: 'Tenderness'
}

def parse_args():
    parser = argparse.ArgumentParser(description='Process and segment FACED Dataset from pkl to npy.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing raw .pkl files.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Root directory for saving output segments.')
    parser.add_argument('--seg_len', type=int, choices=[1, 10, 30], default=30,
                        help='Length of each segment in seconds (1, 10, or 30). Default: 30.')
    parser.add_argument('--raw_unit', type=str, choices=['uV', 'V'], default='V',
                        help='Unit of the raw data. If "uV", data are converted to volts.')
    return parser.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    log_file = os.path.join(os.path.dirname(args.output_dir), 'logs.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    logging.info("=== FACED Dataset Segmentation ===")
    logging.info(f"Input dir      : {args.input_dir}")
    logging.info(f"Output dir     : {args.output_dir}")
    logging.info(f"Raw unit       : {args.raw_unit} -> will be converted to V if needed")
    logging.info(f"Segment length : {args.seg_len} s")
    logging.info(f"Target shape   : ({TARGET_CHANNELS}, {args.seg_len}, {SFREQ})")
    logging.info(f"Data type      : np.float32")
    logging.info("=" * 50)

    # ------------------------------------------------------------------
    # Find and process all .pkl files
    # ------------------------------------------------------------------
    raw_files = sorted(glob.glob(os.path.join(args.input_dir, '*.pkl')))
    if not raw_files:
        logging.error(f"No .pkl files found in {args.input_dir}")
        return

    points_per_seg = args.seg_len * SFREQ
    num_segments = TOTAL_DURATION_SEC // args.seg_len
    total_segments_global = 0

    logging.info(f"Found {len(raw_files)} files. Starting segmentation...")

    for fpath in raw_files:
        fname = os.path.basename(fpath)
        subject_id = os.path.splitext(fname)[0]   # e.g. sub010

        try:
            # 1. Load Pickle data
            with open(fpath, 'rb') as f:
                # Expected shape: (Video_Num=28, Channels=32, TimePoints=7500)
                raw_data = pickle.load(f)

            # 2. Validate dimensions
            if raw_data.shape[2] != TOTAL_POINTS:
                logging.warning(f"File {fname} has {raw_data.shape[2]} points, expected {TOTAL_POINTS}. Skipping.")
                continue
            
            if raw_data.shape[1] != TARGET_CHANNELS:
                logging.warning(f"File {fname} has {raw_data.shape[1]} channels, expected {TARGET_CHANNELS}. Skipping.")
                continue

            if args.raw_unit == 'uV':
                raw_data = raw_data / 1e6   # convert microvolts to volts

            saved_count = 0

            # 3. Iterate over each video (Trial)
            for video_idx in range(raw_data.shape[0]):
                video_data = raw_data[video_idx]  # Shape: (32, 7500)
                
                # Map video to emotion label
                label_id = VIDEO_LABEL_IDS[video_idx]
                emotion_folder = EMOTION_NAMES[label_id]
                
                # Create output directory: output_dir/subject_id/emotion_name/
                save_dir = os.path.join(args.output_dir, subject_id, emotion_folder)
                os.makedirs(save_dir, exist_ok=True)
                
                # 4. Segment and save
                for t_idx in range(num_segments):
                    start_pt = t_idx * points_per_seg
                    end_pt = start_pt + points_per_seg
                    
                    # Slice data: (Channels, Segment_Points)
                    segment = video_data[:, start_pt:end_pt]
                    
                    # Reshape to (Channels, Seconds, Freq) and cast to float32
                    segment_reshaped = segment.reshape(
                        TARGET_CHANNELS, 
                        args.seg_len, 
                        SFREQ
                    ).astype(np.float32)
                    
                    # Filename format: sub[ID]_video[VID]_time[TID].npy
                    out_name = f"{subject_id}_video{video_idx:02d}_time{t_idx:02d}.npy"
                    out_path = os.path.join(save_dir, out_name)
                    
                    np.save(out_path, segment_reshaped)
                    saved_count += 1
            
            total_segments_global += saved_count
            logging.info(f"Processed: {fname} -> {saved_count} segments")

        except Exception as e:
            logging.error(f"Error processing {fname}: {e}")

    logging.info("=" * 50)
    logging.info(f"All done! Total segments created: {total_segments_global}")

if __name__ == "__main__":
    main()