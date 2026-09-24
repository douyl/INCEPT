import os
import sys
import argparse
import numpy as np
import mne
import warnings
import glob
from mne.preprocessing import ICA
from mne_icalabel import label_components
from functools import partial
from tqdm import tqdm

from pebble import ProcessPool
from concurrent.futures import TimeoutError

warnings.filterwarnings("ignore")

# --- 1. 日志管理系统 ---
class BufferedLogger:
    """子进程日志缓冲，最后统一返回主进程写入"""
    def __init__(self, fname):
        self.fname = fname
        self.logs = []

    def log(self, message, level="INFO"):
        # 错误和警告加前缀，普通信息保持干净
        prefix = f"[{level}] " if level in ["WARNING", "ERROR"] else ""
        # 存入格式：(用于显示的纯文本, 日志级别)
        full_msg = f"[{self.fname}] {prefix}{message}"
        self.logs.append({'msg': full_msg, 'level': level})

class MainLogManager:
    """主进程日志管理，负责写入文件"""
    def __init__(self, log_root):
        self.log_root = log_root
        self.folder_handles = {}

    def _get_handles(self, folder_name):
        if folder_name not in self.folder_handles:
            f_dir = os.path.join(self.log_root, folder_name)
            os.makedirs(f_dir, exist_ok=True)
            
            self.folder_handles[folder_name] = {
                'all': open(os.path.join(f_dir, "process.txt"), "a", encoding='utf-8'),
                'error': open(os.path.join(f_dir, "error.txt"), "a", encoding='utf-8'),
                'warn': open(os.path.join(f_dir, "warning.txt"), "a", encoding='utf-8')
            }
        return self.folder_handles[folder_name]

    def write_buffer(self, folder_name, log_list):
        handles = self._get_handles(folder_name)
        for entry in log_list:
            msg = entry['msg']
            level = entry['level']
            
            # 写入所有日志文件
            handles['all'].write(msg + "\n")
            if level == "ERROR":
                handles['error'].write(msg + "\n")
            elif level == "WARNING":
                handles['warn'].write(msg + "\n")
        
        for f in handles.values(): f.flush()

    def close(self):
        for handles in self.folder_handles.values():
            for f in handles.values(): f.close()

def get_args():
    parser = argparse.ArgumentParser(description="EEG Preprocessing (MNE)")
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_root', type=str, required=True)
    parser.add_argument('--select_folders', type=str, default=None, help="Filter specific folders (e.g., '000-002' or '000,001,002' or '000,001-002')")

    parser.add_argument('--target_freq', type=int, default=250)
    parser.add_argument('--line_freq', type=float, default=60.0, choices=[50.0, 60.0])
    parser.add_argument('--l_freq', type=float, default=0.1)
    parser.add_argument('--h_freq', type=float, default=50.0)
    
    parser.add_argument('--ica_n', default=None)
    parser.add_argument('--skip_ica', action='store_true')

    parser.add_argument('--montage', type=str, default='standard_1020')
    # parser.add_argument('--n_jobs', type=int, default=1, help='Number of CPU cores')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout in seconds (default 300s)')
    
    return parser.parse_args()

# --- 2. 核心处理逻辑 ---
def run_ica_pipeline(raw, logger, n_components=None):
    """ICA 拟合与自动剔除"""
    # ICA 专用高通滤波 (1Hz) 以获得更好的分解效果
    raw_filt = raw.copy().filter(l_freq=1.0, h_freq=None, phase="zero-double", verbose='ERROR')

    ica = ICA(n_components=n_components, method='fastica', random_state=97, max_iter='auto')
    # 禁止输出 Fitting components...
    ica.fit(raw_filt, verbose='ERROR') 

    # ICLabel 自动标注
    try:
        ic_labels = label_components(raw, ica, method="iclabel")
        labels = ic_labels["labels"]
        # 排除非 brain 和 非 other 的成分
        exclude_idx = [idx for idx, lab in enumerate(labels) if lab not in ['brain', 'other']]
        ica.exclude = exclude_idx
        
        # 格式化输出：ICA excluded 3 components: 0, 2, 14 (Labels: eye, muscle, eye)
        if exclude_idx:
            ex_labels = [labels[i] for i in exclude_idx]
            idx_str = ", ".join(map(str, exclude_idx))
            lab_str = ", ".join(ex_labels)
            logger.log(f"-> ICA excluded {len(exclude_idx)} components: {idx_str} (Labels: {lab_str})")
        else:
            logger.log("-> ICA excluded 0 components")
            
    except Exception as e:
        logger.log(f"ICLabel failed, skipping exclusion: {e}", level="WARNING")

    return ica.apply(raw.copy(), verbose='ERROR')

def process_single_file(fpath, args, preprocess_dir):
    """单个文件处理流程"""
    fname = os.path.basename(fpath)
    folder_name = os.path.relpath(fpath, args.input_dir).split(os.sep)[0]
    logger = BufferedLogger(fname)
    
    save_dir = os.path.join(preprocess_dir, folder_name)
    os.makedirs(save_dir, exist_ok=True)
    out_edf = os.path.join(save_dir, fname)
    # out_edf = os.path.join(preprocess_dir, fname)

    # [断点续传] 如果文件已存在，直接跳过
    if os.path.exists(out_edf):
        logger.log("File exists, skipping.") # 可选：如果觉得刷屏可以注释掉
        return folder_name, logger.logs

    logger.log(f"Processing started!")

    try:
        # --- 1. 读取与重命名 ---
        raw = mne.io.read_raw_edf(fpath, preload=True, verbose='ERROR')

        rename_dict = {}
        for ch in raw.ch_names:
            new_name = ch.replace('EEG ', '').replace('-REF', '').replace('-LE', '')
            new_name = new_name.replace('FP', 'Fp').replace('Z', 'z')
            if new_name == 'T1': new_name = 'FT9'    # 检测癫痫的临床电极，与FT9位置相近
            if new_name == 'T2': new_name = 'FT10'
            rename_dict[ch] = new_name
        mne.rename_channels(raw.info, rename_dict)

        # --- 2. 通道检查与筛选 ---
        cur_chs = raw.ch_names
        mandatory_chs = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                         'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
        
        # 核心19通道检查
        missing = [c for c in mandatory_chs if c not in cur_chs]
        if missing:
            raise ValueError(f"Missing mandatory channels: {missing}")

        # FT9/FT10 检查
        ft_pair = ['FT9', 'FT10']
        missing_ft = [c for c in ft_pair if c not in cur_chs]
        keep_ft = False
        if len(missing_ft) == 0:
            keep_ft = True
        elif len(missing_ft) == 2:
            logger.log("No FT9/FT10 found.", level="WARNING")
        else: # 缺1个
            logger.log(f"Missing {missing_ft[0]}, dropping FT pair.", level="WARNING")

        # A1/A2 参考电极检查
        ref_pair = ['A1', 'A2']
        missing_ref = [c for c in ref_pair if c not in cur_chs]
        use_linked_ears = False
        if len(missing_ref) == 0:
            use_linked_ears = True
        elif len(missing_ref) == 2:
            logger.log("No A1/A2 found. Using Average Ref.", level="WARNING")
        else: # 缺1个
            logger.log(f"Missing {missing_ref[0]}, cannot use Linked Ears. Using Average Ref.", level="WARNING")

        # 选定最终通道
        to_pick = list(mandatory_chs)
        if keep_ft: to_pick.extend(ft_pair)
        if use_linked_ears: to_pick.extend(ref_pair)
        
        raw.pick(to_pick, verbose='ERROR')
        raw.set_channel_types({c: 'eeg' for c in raw.ch_names})
        raw.set_montage(args.montage)

        # --- 3. 信号处理 ---
        # 陷波滤波
        if args.line_freq:
            raw.notch_filter(freqs=[args.line_freq, 2*args.line_freq], verbose='ERROR')
        
        # 带通滤波
        logger.log(f"-> Filtering ({args.l_freq}-{args.h_freq} Hz)")
        raw.filter(l_freq=args.l_freq, h_freq=args.h_freq, phase='zero-double', verbose='ERROR')

        # 降采样
        if raw.info['sfreq'] != args.target_freq:
            logger.log(f"-> Resampling to {args.target_freq}Hz")
            raw.resample(args.target_freq, npad='auto', verbose='ERROR')

        # 重参考
        if use_linked_ears:
            logger.log(f"-> Re-referencing to Linked Ears (A1+A2)")
            raw.set_eeg_reference(ref_channels=ref_pair, verbose='ERROR')
            raw.drop_channels(ref_pair)
        else:
            logger.log(f"-> Re-referencing to Average")
            raw.set_eeg_reference(ref_channels='average', verbose='ERROR')

        # 通道排序
        final_order = list(mandatory_chs)
        if keep_ft: final_order.extend(ft_pair)
        raw.reorder_channels(final_order)

        # --- 4. ICA ---
        if not args.skip_ica:
            raw = run_ica_pipeline(raw, logger, n_components=args.ica_n)

        # --- 5. 保存 ---
        mne.export.export_raw(out_edf, raw, fmt='edf', overwrite=True, verbose='ERROR')
        logger.log("Success!")

    except Exception as e:
        logger.log(f"Failed: {str(e)}", level="ERROR")

    return folder_name, logger.logs


# --- 3. 主程序 ---
def main():
    args = get_args()

    # 路径设置
    output_base = args.output_root
    preprocess_dir = os.path.join(output_base, 'Preprocess')
    log_root = os.path.join(output_base, 'Log')
    os.makedirs(preprocess_dir, exist_ok=True)

    main_logger = MainLogManager(log_root)
    
    # 获取文件
    if args.select_folders:
        targets = [f"{i:03d}" for p in args.select_folders.split(',') for i in range(int(p.split('-')[0]), int(p.split('-')[-1]) + 1)]
        files = [f for t in targets for f in sorted(glob.glob(os.path.join(args.input_dir, t, "**", "*.edf"), recursive=True))]
    else:
        files = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.edf"), recursive=True))
    
    if not files:
        print("No .edf files found.")
        return

    print(f"=== Pipeline Started (Pebble Engine) ===")
    print(f"Input: {args.input_dir}")
    print(f"Select Folders: {args.select_folders if args.select_folders else 'All'}")
    print(f"Total Files: {len(files)}")
    print(f"Timeout Limit: {args.timeout} seconds")
    print(f"Output: {output_base}")
    print("-" * 30)

    worker_func = partial(process_single_file, args=args, preprocess_dir=preprocess_dir)

    # --- 使用 Pebble ProcessPool ---
    # max_workers=1 保证每次只跑一个文件，模拟串行，但拥有独立的进程空间和超时控制能力
    with ProcessPool(max_workers=1) as pool:
        
        # 使用 tqdm 显示进度
        for fpath in tqdm(files, total=len(files), unit="file", desc="Preprocessing"):
            
            # 提交任务，设置超时时间
            future = pool.schedule(worker_func, args=(fpath,), timeout=args.timeout)
            
            try:
                # 获取结果（这步是阻塞的，直到任务完成或超时）
                folder_name, log_list = future.result()
                main_logger.write_buffer(folder_name, log_list)
            
            except TimeoutError:
                # === 捕获超时异常 ===
                fname = os.path.basename(fpath)
                folder_name = os.path.relpath(fpath, args.input_dir).split(os.sep)[0]
                
                # 手动构建超时日志
                err_logger = BufferedLogger(fname)
                err_logger.log(f"TIMEOUT ERROR: Processing exceeded {args.timeout}s. Killed.", level="ERROR")
                main_logger.write_buffer(folder_name, err_logger.logs)
                # Pebble 会自动清理被杀死的进程，循环继续处理下一个文件
                
            except Exception as e:
                # === 捕获其他非预期的进程崩溃 (如 Segmentation Fault) ===
                fname = os.path.basename(fpath)
                folder_name = os.path.relpath(fpath, args.input_dir).split(os.sep)[0]
                
                err_logger = BufferedLogger(fname)
                err_logger.log(f"CRITICAL PROCESS ERROR: {str(e)}", level="ERROR")
                main_logger.write_buffer(folder_name, err_logger.logs)

    main_logger.close()
    print("\n[All Done]")


if __name__ == "__main__":
    main()