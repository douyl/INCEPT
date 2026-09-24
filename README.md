# INCEPT / CBraDINO

This repository contains the EEG preprocessing, self-supervised pretraining, and downstream evaluation code used by the project.

## Repository layout

```text
.
├── Code_Preprocess/             # Dataset-specific preprocessing scripts
├── Code_Training/
│   ├── configs/train/           # 200-epoch pretraining configuration
│   ├── evaluations/             # Fine-tuning and linear-probing scripts
│   ├── experiments/             # Runtime outputs (initially empty)
│   └── train/                   # Pretraining entry point
└── Data_Reference/              # Small example of the expected data layout
    └── TUEG/
```

The released training framework is based on `CBraDINO_Base-Aug-Dynamic-TimeFreq-SH-ChAttn`. Generated experiment artifacts and checkpoints are intentionally not included.

## Setup

Create a Python environment with PyTorch/CUDA and the dependencies imported by the scripts (including MNE, NumPy, SciPy, scikit-learn, pandas, PyYAML, tqdm, and tensorboard). Run all commands below from the repository root unless stated otherwise.

Define paths for convenience:

```bash
export REPO_ROOT="$(pwd)"
export DATA_ROOT=/path/to/EEG_Dataset
export EVAL_DIR=/path/to/pretrained_checkpoint
```

For downstream evaluation, `EVAL_DIR` must contain both `config.yaml` and `teacher_checkpoint.pth` from a pretrained run.

## Reference data structure

`Data_Reference/TUEG` contains one subject/recording example showing the expected data layout before and after preprocessing:

```text
Data_Reference/
└── TUEG/
    ├── TUEG_RawData/
    │   └── 000/
    │       └── aaaaaaaa/
    │           └── s001_2015/
    │               └── 01_tcp_ar/
    │                   └── aaaaaaaa_s001_t000.edf
    └── TUEG_250Hz/
        ├── Preprocess/
        │   └── 000/
        │       └── aaaaaaaa_s001_t000.edf
        └── Segment/
            └── 000/
                └── aaaaaaaa_s001_t000/
                    ├── segment_0.npy
                    ├── segment_1.npy
                    ├── ...
                    └── segment_28.npy
```

`[TUEG]-1-Preprocess.py` reads the nested raw EDF recording and writes the standardized 250 Hz EDF under `TUEG_250Hz/Preprocess`. `[TUEG]-2-Segment.py` then converts that preprocessed recording into individual 30-second NumPy segments under `TUEG_250Hz/Segment`.

The other supported datasets follow the same general organization: raw recordings are kept in a dataset-specific raw-data directory, preprocessing standardizes the signals (including the target sampling rate), and segmentation produces model-ready samples in a `Segment` or `Segment_*` directory. Exact directory names and segment lengths are dataset-specific; refer to the commands below.

## 1. EEG preprocessing

```bash
cd "$REPO_ROOT/Code_Preprocess"
```

The following commands reproduce the dataset-specific invocations from `Process_Downstream.job`. Replace the paths under `DATA_ROOT` when your raw data uses a different layout.

### MentalArithmetic

```bash
python "[MentalArithmetic].py" --raw_unit V \
  --input_dir "$DATA_ROOT/MentalArithmetic/MentalArithmetic_RawData" \
  --output_dir "$DATA_ROOT/MentalArithmetic/MentalArithmetic_250Hz"
```

### FACED

```bash
python "[FACED].py" --seg_len 30 --raw_unit uV \
  --input_dir "$DATA_ROOT/FACED/FACED_RawData/Processed_data" \
  --output_dir "$DATA_ROOT/FACED/FACED_250Hz/Segment_30s"

python "[FACED].py" --seg_len 10 --raw_unit uV \
  --input_dir "$DATA_ROOT/FACED/FACED_RawData/Processed_data" \
  --output_dir "$DATA_ROOT/FACED/FACED_250Hz/Segment_10s"
```

### PhysioNet-MI

```bash
python "[PhysioNet-MI].py" --raw_unit V \
  --input_dir "$DATA_ROOT/PhysioNet-MI/PhysioNet-MI_RawData" \
  --output_dir "$DATA_ROOT/PhysioNet-MI/PhysioNet-MI_250Hz"
```

### SEED-V

```bash
python "[SEEDV].py" --seg_len 4 --raw_unit V \
  --input_dir "$DATA_ROOT/SEED/SEED_RawData/SEED-V/EEG_raw" \
  --output_dir "$DATA_ROOT/SEED/SEED_250Hz/SEED-V_4s"

python "[SEEDV].py" --seg_len 1 --raw_unit V \
  --input_dir "$DATA_ROOT/SEED/SEED_RawData/SEED-V/EEG_raw" \
  --output_dir "$DATA_ROOT/SEED/SEED_250Hz/SEED-V_1s"
```

### Mumtaz2016

```bash
python "[Mumtaz2016].py" --l_freq 0.3 --h_freq 75 --notch_freq 50 --raw_unit V \
  --input_dir "$DATA_ROOT/Mumtaz2016/Mumtaz2016_RawData" \
  --output_dir "$DATA_ROOT/Mumtaz2016/Mumtaz2016_250Hz_03-75filter"
```

### ISRUC-S1

```bash
python "[ISRUC].py" --l_freq 0.3 --h_freq 35 --notch_freq 50 --raw_unit V \
  --input_dir "$DATA_ROOT/ISRUC-SLEEP/ISRUC_RawData/ISRUC_S1" \
  --output_dir "$DATA_ROOT/ISRUC-SLEEP/ISRUC_S1_250Hz"
```

### TUAB

```bash
python "[TUAB].py" --skip_reference --skip_ica \
  --l_freq 0.3 --h_freq 75 --notch_freq 60 --raw_unit V \
  --input_dir "$DATA_ROOT/TUAB/TUAB_RawData" \
  --output_dir "$DATA_ROOT/TUAB/TUAB_250Hz"
```

### ADFTD

```bash
python "[ADFTD].py" --seg_len 10 --raw_unit V \
  --input_dir "$DATA_ROOT/ADFTD/ADFTD_RawData" \
  --output_dir "$DATA_ROOT/ADFTD/ADFTD_250Hz"

python "[ADFTD].py" --seg_len 4 --raw_unit V \
  --input_dir "$DATA_ROOT/ADFTD/ADFTD_RawData" \
  --output_dir "$DATA_ROOT/ADFTD/ADFTD_250Hz"

python "[ADFTD].py" --seg_len 30 --raw_unit V \
  --input_dir "$DATA_ROOT/ADFTD/ADFTD_RawData" \
  --output_dir "$DATA_ROOT/ADFTD/ADFTD_250Hz"
```

### Siena

```bash
python "[Siena].py" --l_freq 0.1 --h_freq 75 --notch_freq 50 --raw_unit V \
  --input_dir "$DATA_ROOT/Siena/Siena_RawData" \
  --output_dir "$DATA_ROOT/Siena/Siena_250Hz"
```

### TUAR

```bash
python "[TUAR].py" --seed 0 --skip_reference --skip_ica \
  --l_freq 0.3 --h_freq 75 --notch_freq 60 --raw_unit V \
  --input_dir /path/to/TUAR/edf \
  --output_dir "$DATA_ROOT/TUAR"
```

### TUEG

TUEG preprocessing has two stages. Stage 1 accepts paths on the command line:

```bash
python "[TUEG]-1-Preprocess.py" \
  --input_dir "$DATA_ROOT/TUEG/TUEG_RawData" \
  --output_root "$DATA_ROOT/TUEG/TUEG_250Hz/Preprocess" \
  --target_freq 250 --line_freq 60 --l_freq 0.1 --h_freq 50
```

Optional Stage 1 arguments include `--select_folders`, `--ica_n`, `--skip_ica`, `--montage`, and `--timeout`.

Stage 2 currently defines `INPUT_ROOT` and `OUTPUT_ROOT` near the top of `[TUEG]-2-Segment.py`. Set them to the Stage 1 output and desired segment directory, respectively, then run:

```bash
python "[TUEG]-2-Segment.py"
```

## 2. Self-supervised pretraining (200 epochs)

Before training, edit `Code_Training/configs/train/vitb_eeg_200epoch.yaml` and set `dataset.data_root` to the preprocessed TUEG segment directory.

Run from `Code_Training`:

```bash
cd "$REPO_ROOT/Code_Training"
env -u SLURM_JOB_ID torchrun --nproc_per_node=2 --master_port=29520 train/train.py --output_dir experiments/train_200epoch --config-file configs/train/vitb_eeg_200epoch.yaml
```

This is the two-GPU command from `train_4gpu.job`. Change `--master_port` if that port is already in use.

### One-GPU local batch-size test

The 200-epoch YAML points to the local `Data_Reference` sample. From `Code_Training`, run the following command to test the pipeline on one GPU with batch size 2:

```bash
torchrun --nproc_per_node=1  train/train.py --output_dir experiments/test --config-file configs/train/vitb_eeg_200epoch.yaml train.batch_size_per_gpu=2
```

Here, `train.batch_size_per_gpu=2` is a temporary command-line override used only for this small `Data_Reference` test. For real training, remove this override from the command and edit `train.batch_size_per_gpu` directly in `configs/train/vitb_eeg_200epoch.yaml` to the batch size supported by the available GPU memory.

## 3. Downstream evaluation

Run the following commands from `Code_Training`. They reproduce every dataset command in `eval_finetune.job` and `eval_linearprobe.job`, with an explicit `--eval_dir` so the scripts can locate `config.yaml` and `teacher_checkpoint.pth`.

```bash
cd "$REPO_ROOT/Code_Training"
```

### Fine-tuning

```bash
# MentalArithmetic
python "evaluations/[MentalArithmetic]-finetune.py" --eval_dir "$EVAL_DIR" --use_patch flatten --use_fp16 --epochs 20 --data "$DATA_ROOT/MentalArithmetic/MentalArithmetic_250Hz/Segment"

# FACED
python "evaluations/[FACED]-finetune.py" --eval_dir "$EVAL_DIR" --use_patch flatten --use_lrscheduler --use_fp16 --data "$DATA_ROOT/FACED/FACED_250Hz/Segment_30s"

# PhysioNet-MI
python "evaluations/[PhysioNet-MI]-finetune.py" --eval_dir "$EVAL_DIR" --use_patch flatten --use_lrscheduler --use_fp16 --data "$DATA_ROOT/PhysioNet-MI/PhysioNet-MI_250Hz/Segment"

# SEED-V
python "evaluations/[SEEDV]-finetune.py" --eval_dir "$EVAL_DIR" --use_patch flatten --use_lrscheduler --use_fp16 --data "$DATA_ROOT/SEED/SEED_250Hz/SEED-V_4s/Segment"

# Mumtaz2016
python "evaluations/[Mumtaz2016]-finetune.py" --eval_dir "$EVAL_DIR" --use_patch flatten --use_lrscheduler --use_fp16 --data "$DATA_ROOT/Mumtaz2016/Mumtaz2016_250Hz_03-75filter/Segment"

# ISRUC-S1
python "evaluations/[ISRUC-S1]-finetune.py" --eval_dir "$EVAL_DIR" --use_patch flatten --use_lrscheduler --use_fp16 --epochs 30 --batchsize 60 --data "$DATA_ROOT/ISRUC-SLEEP/ISRUC_S1_250Hz/Segment"

# TUAB
python "evaluations/[TUAB]-finetune.py" --eval_dir "$EVAL_DIR" --use_patch flatten --use_fp16 --epochs 10 --data "$DATA_ROOT/TUAB/TUAB_250Hz/Segment"

# ADFTD
python "evaluations/[ADFTD]-finetune.py" --eval_dir "$EVAL_DIR" --use_patch flatten --use_lrscheduler --use_fp16 --data "$DATA_ROOT/ADFTD/ADFTD_250Hz/Segment_10s"

# Siena
python "evaluations/[Siena]-finetune.py" --eval_dir "$EVAL_DIR" --use_patch flatten --use_lrscheduler --use_fp16 --data "$DATA_ROOT/Siena/Siena_250Hz/Segment"

# TUAR
python "evaluations/[TUAR]-finetune.py" --eval_dir "$EVAL_DIR" --use_patch pooling --use_lrscheduler --use_fp16 --epochs 20 --data "$DATA_ROOT/TUAR/TUAR_250Hz/Segment"
```

### Linear probing

```bash
# MentalArithmetic
python "evaluations/[MentalArithmetic]-linearprobe.py" --eval_dir "$EVAL_DIR" --use_patch flatten --use_fp16 --data "$DATA_ROOT/MentalArithmetic/MentalArithmetic_250Hz/Segment"

# FACED
python "evaluations/[FACED]-linearprobe.py" --eval_dir "$EVAL_DIR" --use_patch flatten --use_lrscheduler --use_fp16 --data "$DATA_ROOT/FACED/FACED_250Hz/Segment_30s"

# PhysioNet-MI
python "evaluations/[PhysioNet-MI]-linearprobe.py" --eval_dir "$EVAL_DIR" --use_patch flatten --use_lrscheduler --use_fp16 --data "$DATA_ROOT/PhysioNet-MI/PhysioNet-MI_250Hz/Segment"

# SEED-V
python "evaluations/[SEEDV]-linearprobe.py" --eval_dir "$EVAL_DIR" --use_patch flatten --use_lrscheduler --use_fp16 --data "$DATA_ROOT/SEED/SEED_250Hz/SEED-V_4s/Segment"

# Mumtaz2016
python "evaluations/[Mumtaz2016]-linearprobe.py" --eval_dir "$EVAL_DIR" --use_patch flatten --use_lrscheduler --use_fp16 --data "$DATA_ROOT/Mumtaz2016/Mumtaz2016_250Hz_03-75filter/Segment"

# ISRUC-S1
python "evaluations/[ISRUC-S1]-linearprobe.py" --eval_dir "$EVAL_DIR" --use_patch flatten --use_lrscheduler --use_fp16 --data "$DATA_ROOT/ISRUC-SLEEP/ISRUC_S1_250Hz/Segment"

# TUAB
python "evaluations/[TUAB]-linearprobe.py" --eval_dir "$EVAL_DIR" --use_patch flatten --use_lrscheduler --use_fp16 --epochs 10 --data "$DATA_ROOT/TUAB/TUAB_250Hz/Segment"

# ADFTD
python "evaluations/[ADFTD]-linearprobe.py" --eval_dir "$EVAL_DIR" --use_patch flatten --use_lrscheduler --use_fp16 --data "$DATA_ROOT/ADFTD/ADFTD_250Hz/Segment_10s"

# Siena
python "evaluations/[Siena]-linearprobe.py" --eval_dir "$EVAL_DIR" --use_patch flatten --use_lrscheduler --use_fp16 --data "$DATA_ROOT/Siena/Siena_250Hz/Segment"

# TUAR
python "evaluations/[TUAR]-linearprobe.py" --eval_dir "$EVAL_DIR" --use_patch flatten --use_lrscheduler --use_fp16 --data "$DATA_ROOT/TUAR/TUAR_250Hz/Segment"
```

## Notes

- Dataset licenses and access conditions are not included; obtain each dataset from its official source.
- `experiments` is intentionally empty and is used for generated training output.
- Evaluation checkpoints are not included. Supply them through `--eval_dir` as described above.
