import os
import glob
import numpy as np
import mne
import json
import logging
import sys
import shutil  # Added for cleanup
from datetime import timedelta
from tqdm import tqdm

# ================= Configuration =================
# Input root containing subfolders 000, 001, ..., 150
INPUT_ROOT = "/public_bme2/grpdgshen/douyl/EEG_Dataset/TUEG/TUEG_250Hz/Preprocess"
# Output root
OUTPUT_ROOT = "/public_bme2/grpdgshen/douyl/EEG_Dataset/TUEG/TUEG_250Hz/Segment"

# Standard 19 Channels (Order matters)
TARGET_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
]

SFREQ = 250
SEGMENT_DURATION_SEC = 30
POINTS_PER_SEGMENT = SEGMENT_DURATION_SEC * SFREQ  # 7500 points
THRESHOLD_UV = 120.0  # [KEEP] Threshold 120uV

# Crop margin: 1 minute at start and end
CROP_MARGIN_SEC = 60 
# =================================================

# Setup Logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def format_time(seconds):
    """Helper to format seconds into HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))

def process_single_edf(edf_path, output_subdir, subdir_name):
    """
    Process a single EDF file and save segments as individual NPY files.
    Structure: OUTPUT_ROOT/000/edf_filename_dir/segment_0.npy
    """
    file_name = os.path.basename(edf_path)
    base_name = os.path.splitext(file_name)[0]  # e.g., 'aaaaaaaa_s001_t000'
    
    # Define the specific folder for this EDF's segments
    # e.g. .../SegmentNPY/000/aaaaaaaa_s001_t000/
    edf_specific_dir = os.path.join(output_subdir, base_name)
    
    try:
        # 1. Read Raw Data
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='error')
        
        # Check Sampling Rate
        if int(raw.info['sfreq']) != SFREQ:
            logger.warning(f"SKIP {file_name}: Rate {raw.info['sfreq']} != {SFREQ}")
            return None

        total_duration = raw.times[-1]
        
        # (1) Quality Check: Skip if < 5 minutes
        if total_duration < 300:
            logger.warning(f"SKIP {file_name}: Duration {format_time(total_duration)} < 5 min")
            return None

        # (2) Select and Reorder Channels
        try:
            raw.pick(TARGET_CHANNELS)
            raw.reorder_channels(TARGET_CHANNELS)
        except ValueError as e:
            logger.warning(f"SKIP {file_name}: Missing channels or pick error. {e}")
            return None

        # Get data in VOLTS
        data_volts = raw.get_data()

        # (3) Margin Clipping logic
        margin_pts = CROP_MARGIN_SEC * SFREQ 
        
        # Check data length
        if data_volts.shape[1] <= 2 * margin_pts:
            logger.warning(f"SKIP {file_name}: Data too short after margin crop.")
            return None

        # Effective data range
        valid_data = data_volts[:, margin_pts : -margin_pts]
        
        # (4) Segmentation & Artifact Rejection
        num_possible_segments = valid_data.shape[1] // POINTS_PER_SEGMENT
        
        discarded_info = []
        saved_count = 0

        # Create directory only if we are about to save something? 
        # Actually, create it now, delete later if empty.
        if not os.path.exists(edf_specific_dir):
            os.makedirs(edf_specific_dir)

        for i in range(num_possible_segments):
            start_idx = i * POINTS_PER_SEGMENT
            end_idx = start_idx + POINTS_PER_SEGMENT
            
            # Absolute Time (for logging)
            abs_start_time = CROP_MARGIN_SEC + i * SEGMENT_DURATION_SEC
            abs_end_time = abs_start_time + SEGMENT_DURATION_SEC
            
            segment = valid_data[:, start_idx:end_idx]
            
            # Check Artifacts (> THRESHOLD_UV)
            segment_uv = segment * 1e6 
            max_val = np.max(np.abs(segment_uv))
            
            if max_val > THRESHOLD_UV:
                discarded_info.append(f"{format_time(abs_start_time)}-{format_time(abs_end_time)} (Max: {max_val:.1f}uV)")
                continue 
            
            # Reshape for Dataset: (C, N, T) -> (19, 30, 250)
            segment_reshaped = segment.reshape(len(TARGET_CHANNELS), 30, 250).astype(np.float32)
            
            # --- Save Individual Segment ---
            # File name: segment_0.npy, segment_1.npy, etc.
            seg_filename = f"segment_{saved_count}.npy"
            seg_path = os.path.join(edf_specific_dir, seg_filename)
            np.save(seg_path, segment_reshaped)
            
            saved_count += 1

        # Log results
        num_discarded = len(discarded_info)
        
        if saved_count == 0:
            dropped_details = ", ".join(discarded_info)
            logger.info(f"DROP {file_name}: All segments rejected ({num_discarded} artifacts). --> Dropped Time Segments: {dropped_details}")
            
            # Cleanup: Remove the empty directory we created
            if os.path.exists(edf_specific_dir):
                os.rmdir(edf_specific_dir)
            return None
        
        # [KEEP] Single line logging
        log_msg = f"{file_name}: Kept {saved_count}, Dropped {num_discarded} segments."
        if num_discarded > 0:
            log_msg += f"  --> Dropped Time Segments: {', '.join(discarded_info)}"
        
        logger.info(log_msg)
        
        # Return metadata
        # path now points to the folder containing the segments: "000/aaaaaaaa_s001_t000"
        rel_path = os.path.join(subdir_name, base_name)
        
        return {
            "path": rel_path,  # This is a DIRECTORY now
            "count": saved_count
        }

    except Exception as e:
        logger.error(f"ERROR processing {file_name}: {e}")
        # Cleanup if error occurred and directory was created but might be partial
        # (Optional: keep partial data or delete. Here we assume manual check if error.)
        return None

def process_folder_wrapper(subdir_name):
    """
    Process all files in a specific subfolder.
    """
    src_folder = os.path.join(INPUT_ROOT, subdir_name)
    dst_folder = os.path.join(OUTPUT_ROOT, subdir_name)
    
    if not os.path.exists(src_folder):
        logger.error(f"Source folder not found: {src_folder}")
        return

    # Ensure the parent output subfolder exists (e.g., .../SegmentNPY/000/)
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    
    edf_files = sorted(glob.glob(os.path.join(src_folder, "*.edf")))
    
    logger.info(f"========== START Processing Folder: {subdir_name} | Found {len(edf_files)} EDFs ==========")
    
    folder_manifest = []
    
    for edf_path in edf_files:
        meta_entry = process_single_edf(edf_path, dst_folder, subdir_name)
        if meta_entry:
            folder_manifest.append(meta_entry)
            
    # Save Metadata
    if folder_manifest:
        json_name = f"metadata_{subdir_name}.json"
        json_path = os.path.join(OUTPUT_ROOT, json_name)
        
        with open(json_path, 'w') as f:
            json.dump(folder_manifest, f, indent=2)
            
        logger.info(f"========== END Folder {subdir_name}: Saved metadata to {json_name} ==========")
    else:
        logger.warning(f"========== END Folder {subdir_name}: No valid segments found ==========")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    # Automatically find all subfolders
    all_subdirs = sorted([
        d for d in os.listdir(INPUT_ROOT) 
        if os.path.isdir(os.path.join(INPUT_ROOT, d))
    ])
    
    logger.info(f"Starting Preprocessing. Found {len(all_subdirs)} folders.")

    for subdir in tqdm(all_subdirs, desc="Total Progress", unit="folder"):
        process_folder_wrapper(subdir)
        
    logger.info("All Done.")