# Taming Foundation Model with Invariance-Oriented Pre-Training for Broad-Spectrum EEG Analysis Across Signal-Level, Brain-State, and Brain-Health Tasks

Yulong Dou, Han Wu, Guo Chen, Fangmao Ju, Zhiming Cui, and Dinggang Shen

This is the official implementation of **INCEPT**, an invariance-oriented EEG foundation model for transferable analysis across signal-level, brain-state, and brain-health tasks. The manuscript is currently under review.

## Environment Setup

The code was tested with Python 3.9, PyTorch 2.0.0, and CUDA 11.7. Create a clean Conda environment and install only the packages required by this repository:

```bash
git clone https://github.com/douyl/INCEPT.git
cd INCEPT

conda create -n incept python=3.9 -y
conda activate incept

pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

All commands below assume that they are run from the repository root unless stated otherwise. Define the dataset root once for convenience:

```bash
export REPO_ROOT="$(pwd)"
export DATA_ROOT=/path/to/EEG_Dataset
```

## 1. Data Preprocessing

The scripts in `Code_Preprocess` convert raw EEG recordings into standardized 250 Hz recordings and model-ready NumPy segments. The repository supports TUEG for self-supervised pre-training and ten downstream datasets for evaluation.

| Dataset | Preprocessing Code |
|---|---|
| TUEG | `[TUEG]-1-Preprocess.py`, `[TUEG]-2-Segment.py` |
| ADFTD | `[ADFTD].py` |
| FACED | `[FACED].py` |
| ISRUC-S1 | `[ISRUC].py` |
| MentalArithmetic | `[MentalArithmetic].py` |
| Mumtaz2016 | `[Mumtaz2016].py` |
| PhysioNet-MI | `[PhysioNet-MI].py` |
| SEED-V | `[SEEDV].py` |
| Siena | `[Siena].py` |
| TUAB | `[TUAB].py` |
| TUAR | `[TUAR].py` |

### TUEG Preprocessing

TUEG preprocessing consists of two stages. The first stage standardizes the raw EDF recordings, and the second stage creates 30-second NumPy samples.

```bash
cd "$REPO_ROOT/Code_Preprocess"

python "[TUEG]-1-Preprocess.py" \
  --input_dir "$DATA_ROOT/TUEG/TUEG_RawData" \
  --output_root "$DATA_ROOT/TUEG/TUEG_250Hz/Preprocess" \
  --target_freq 250 \
  --line_freq 60 \
  --l_freq 0.1 \
  --h_freq 50
```

Before running the second stage, set `INPUT_ROOT` and `OUTPUT_ROOT` near the top of `[TUEG]-2-Segment.py` to the preprocessed EDF directory and the desired segment directory:

```bash
python "[TUEG]-2-Segment.py"
```

`Data_Reference/TUEG` provides a small example of the expected input and output layout:

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

Other datasets follow the same overall workflow: raw recordings are read from a dataset-specific directory, standardized to the target sampling rate, and saved as model-ready samples under a `Segment` or `Segment_*` directory. Their output directories additionally contain the dataset splits and labels required by the corresponding evaluation scripts.

### Downstream Dataset Preprocessing

Run the following commands from `Code_Preprocess`. Adjust the input paths if the downloaded datasets use a different directory layout.

```bash
# MentalArithmetic
python "[MentalArithmetic].py" --raw_unit V \
  --input_dir "$DATA_ROOT/MentalArithmetic/MentalArithmetic_RawData" \
  --output_dir "$DATA_ROOT/MentalArithmetic/MentalArithmetic_250Hz"

# FACED (30-second segments used by the evaluation commands below)
python "[FACED].py" --seg_len 30 --raw_unit uV \
  --input_dir "$DATA_ROOT/FACED/FACED_RawData/Processed_data" \
  --output_dir "$DATA_ROOT/FACED/FACED_250Hz/Segment_30s"

# Optional FACED 10-second segments
python "[FACED].py" --seg_len 10 --raw_unit uV \
  --input_dir "$DATA_ROOT/FACED/FACED_RawData/Processed_data" \
  --output_dir "$DATA_ROOT/FACED/FACED_250Hz/Segment_10s"

# PhysioNet-MI
python "[PhysioNet-MI].py" --raw_unit V \
  --input_dir "$DATA_ROOT/PhysioNet-MI/PhysioNet-MI_RawData" \
  --output_dir "$DATA_ROOT/PhysioNet-MI/PhysioNet-MI_250Hz"

# SEED-V
python "[SEEDV].py" --seg_len 4 --raw_unit V \
  --montage_file "$DATA_ROOT/SEED/SEED_RawData/SEED-V/channel_62_pos.locs" \
  --input_dir "$DATA_ROOT/SEED/SEED_RawData/SEED-V/EEG_raw" \
  --output_dir "$DATA_ROOT/SEED/SEED_250Hz/SEED-V_4s"

# Optional SEED-V 1-second segments
python "[SEEDV].py" --seg_len 1 --raw_unit V \
  --montage_file "$DATA_ROOT/SEED/SEED_RawData/SEED-V/channel_62_pos.locs" \
  --input_dir "$DATA_ROOT/SEED/SEED_RawData/SEED-V/EEG_raw" \
  --output_dir "$DATA_ROOT/SEED/SEED_250Hz/SEED-V_1s"

# Mumtaz2016
python "[Mumtaz2016].py" --l_freq 0.3 --h_freq 75 --notch_freq 50 --raw_unit V \
  --input_dir "$DATA_ROOT/Mumtaz2016/Mumtaz2016_RawData" \
  --output_dir "$DATA_ROOT/Mumtaz2016/Mumtaz2016_250Hz_03-75filter"

# ISRUC-S1
python "[ISRUC].py" --l_freq 0.3 --h_freq 35 --notch_freq 50 --raw_unit V \
  --input_dir "$DATA_ROOT/ISRUC-SLEEP/ISRUC_RawData/ISRUC_S1" \
  --output_dir "$DATA_ROOT/ISRUC-SLEEP/ISRUC_S1_250Hz"

# TUAB
python "[TUAB].py" --skip_reference --skip_ica \
  --l_freq 0.3 --h_freq 75 --notch_freq 60 --raw_unit V \
  --input_dir "$DATA_ROOT/TUAB/TUAB_RawData" \
  --output_dir "$DATA_ROOT/TUAB/TUAB_250Hz"

# ADFTD (10-second segments used by the evaluation commands below)
python "[ADFTD].py" --seg_len 10 --raw_unit V \
  --input_dir "$DATA_ROOT/ADFTD/ADFTD_RawData" \
  --output_dir "$DATA_ROOT/ADFTD/ADFTD_250Hz"

# Optional ADFTD 4-second segments
python "[ADFTD].py" --seg_len 4 --raw_unit V \
  --input_dir "$DATA_ROOT/ADFTD/ADFTD_RawData" \
  --output_dir "$DATA_ROOT/ADFTD/ADFTD_250Hz"

# Optional ADFTD 30-second segments
python "[ADFTD].py" --seg_len 30 --raw_unit V \
  --input_dir "$DATA_ROOT/ADFTD/ADFTD_RawData" \
  --output_dir "$DATA_ROOT/ADFTD/ADFTD_250Hz"

# Siena
python "[Siena].py" --l_freq 0.1 --h_freq 75 --notch_freq 50 --raw_unit V \
  --input_dir "$DATA_ROOT/Siena/Siena_RawData" \
  --output_dir "$DATA_ROOT/Siena/Siena_250Hz"

# TUAR
python "[TUAR].py" --seed 0 --skip_reference --skip_ica \
  --l_freq 0.3 --h_freq 75 --notch_freq 60 --raw_unit V \
  --input_dir /path/to/TUAR/edf \
  --output_dir "$DATA_ROOT/TUAR"
```

## 2. Self-Supervised Pretraining

Run all pre-training commands from `Code_Training`:

```bash
cd "$REPO_ROOT/Code_Training"
```

### Single-GPU Test

For a quick pipeline test, set `dataset.data_root` in `configs/train/vitb_eeg_200epoch.yaml` to:

```yaml
dataset:
  data_root: ../Data_Reference/TUEG/TUEG_250Hz/Segment
```

Then run the model on one GPU with a temporary per-GPU batch size of 2:

```bash
torchrun --nproc_per_node=1 train/train.py \
  --output_dir experiments/test \
  --config-file configs/train/vitb_eeg.yaml \
  train.batch_size_per_gpu=2
```

The `train.batch_size_per_gpu=2` override is intended only for this small `Data_Reference` test.

### Full Pretraining

For full pre-training, set `dataset.data_root` in the YAML file to the complete TUEG segment directory and remove the test-time batch-size override. With the provided YAML parameters, training requires **2 to 4 NVIDIA A100 80 GB GPUs**.

The following example uses four GPUs:

```bash
env -u SLURM_JOB_ID torchrun \
  --nproc_per_node=4 \
  --master_port=29520 \
  train/train.py \
  --output_dir experiments/pretrain \
  --config-file configs/train/vitb_eeg.yaml
```

Set `--nproc_per_node=2` when using two GPUs. Change `--master_port` if port `29520` is already occupied.

## 3. Downstream Evaluation

### Download the Pretrained Checkpoint

Download `config.yaml` and `teacher_checkpoint.pth` from the [INCEPT checkpoint folder](https://drive.google.com/drive/u/0/folders/1YKJ43MgHDlu2LqGfqi1DMmdN9jytE-ot), then place them as follows:

```text
Code_Training/
└── evaluations/
    ├── checkpoints/
    │   ├── config.yaml
    │   └── teacher_checkpoint.pth
    └── downstream/
```

The evaluation scripts read the pre-trained model from `evaluations/checkpoints` and save downstream checkpoints to `evaluations/downstream/<dataset>`. Both are the default paths; `--eval_dir` and `--output_dir` can be used to override them.

Run the commands below from `Code_Training`:

```bash
cd "$REPO_ROOT/Code_Training"
```

### MentalArithmetic

```bash
# Fine-tuning
python "evaluations/[MentalArithmetic]-finetune.py" \
  --eval_dir evaluations/checkpoints --output_dir evaluations/downstream \
  --use_patch flatten --use_fp16 --epochs 20 \
  --data "$DATA_ROOT/MentalArithmetic/MentalArithmetic_250Hz/Segment"

# Linear probing
python "evaluations/[MentalArithmetic]-linearprobe.py" \
  --eval_dir evaluations/checkpoints --output_dir evaluations/downstream \
  --use_patch flatten --use_fp16 \
  --data "$DATA_ROOT/MentalArithmetic/MentalArithmetic_250Hz/Segment"
```

### FACED

```bash
# Fine-tuning
python "evaluations/[FACED]-finetune.py" \
  --eval_dir evaluations/checkpoints --output_dir evaluations/downstream \
  --use_patch flatten --use_lrscheduler --use_fp16 \
  --data "$DATA_ROOT/FACED/FACED_250Hz/Segment_30s"

# Linear probing
python "evaluations/[FACED]-linearprobe.py" \
  --eval_dir evaluations/checkpoints --output_dir evaluations/downstream \
  --use_patch flatten --use_lrscheduler --use_fp16 \
  --data "$DATA_ROOT/FACED/FACED_250Hz/Segment_30s"
```

### PhysioNet-MI

```bash
# Fine-tuning
python "evaluations/[PhysioNet-MI]-finetune.py" \
  --eval_dir evaluations/checkpoints --output_dir evaluations/downstream \
  --use_patch flatten --use_lrscheduler --use_fp16 \
  --data "$DATA_ROOT/PhysioNet-MI/PhysioNet-MI_250Hz/Segment"

# Linear probing
python "evaluations/[PhysioNet-MI]-linearprobe.py" \
  --eval_dir evaluations/checkpoints --output_dir evaluations/downstream \
  --use_patch flatten --use_lrscheduler --use_fp16 \
  --data "$DATA_ROOT/PhysioNet-MI/PhysioNet-MI_250Hz/Segment"
```

### SEED-V

```bash
# Fine-tuning
python "evaluations/[SEEDV]-finetune.py" \
  --eval_dir evaluations/checkpoints --output_dir evaluations/downstream \
  --use_patch flatten --use_lrscheduler --use_fp16 \
  --data "$DATA_ROOT/SEED/SEED_250Hz/SEED-V_4s/Segment"

# Linear probing
python "evaluations/[SEEDV]-linearprobe.py" \
  --eval_dir evaluations/checkpoints --output_dir evaluations/downstream \
  --use_patch flatten --use_lrscheduler --use_fp16 \
  --data "$DATA_ROOT/SEED/SEED_250Hz/SEED-V_4s/Segment"
```

### Mumtaz2016

```bash
# Fine-tuning
python "evaluations/[Mumtaz2016]-finetune.py" \
  --eval_dir evaluations/checkpoints --output_dir evaluations/downstream \
  --use_patch flatten --use_lrscheduler --use_fp16 \
  --data "$DATA_ROOT/Mumtaz2016/Mumtaz2016_250Hz_03-75filter/Segment"

# Linear probing
python "evaluations/[Mumtaz2016]-linearprobe.py" \
  --eval_dir evaluations/checkpoints --output_dir evaluations/downstream \
  --use_patch flatten --use_lrscheduler --use_fp16 \
  --data "$DATA_ROOT/Mumtaz2016/Mumtaz2016_250Hz_03-75filter/Segment"
```

### ISRUC-S1

```bash
# Fine-tuning
python "evaluations/[ISRUC-S1]-finetune.py" \
  --eval_dir evaluations/checkpoints --output_dir evaluations/downstream \
  --use_patch flatten --use_lrscheduler --use_fp16 \
  --epochs 30 --batchsize 60 \
  --data "$DATA_ROOT/ISRUC-SLEEP/ISRUC_S1_250Hz/Segment"

# Linear probing
python "evaluations/[ISRUC-S1]-linearprobe.py" \
  --eval_dir evaluations/checkpoints --output_dir evaluations/downstream \
  --use_patch flatten --use_lrscheduler --use_fp16 \
  --data "$DATA_ROOT/ISRUC-SLEEP/ISRUC_S1_250Hz/Segment"
```

### TUAB

```bash
# Fine-tuning
python "evaluations/[TUAB]-finetune.py" \
  --eval_dir evaluations/checkpoints --output_dir evaluations/downstream \
  --use_patch flatten --use_fp16 --epochs 10 \
  --data "$DATA_ROOT/TUAB/TUAB_250Hz/Segment"

# Linear probing
python "evaluations/[TUAB]-linearprobe.py" \
  --eval_dir evaluations/checkpoints --output_dir evaluations/downstream \
  --use_patch flatten --use_lrscheduler --use_fp16 --epochs 10 \
  --data "$DATA_ROOT/TUAB/TUAB_250Hz/Segment"
```

### ADFTD

```bash
# Fine-tuning
python "evaluations/[ADFTD]-finetune.py" \
  --eval_dir evaluations/checkpoints --output_dir evaluations/downstream \
  --use_patch flatten --use_lrscheduler --use_fp16 \
  --data "$DATA_ROOT/ADFTD/ADFTD_250Hz/Segment_10s"

# Linear probing
python "evaluations/[ADFTD]-linearprobe.py" \
  --eval_dir evaluations/checkpoints --output_dir evaluations/downstream \
  --use_patch flatten --use_lrscheduler --use_fp16 \
  --data "$DATA_ROOT/ADFTD/ADFTD_250Hz/Segment_10s"
```

### Siena

```bash
# Fine-tuning
python "evaluations/[Siena]-finetune.py" \
  --eval_dir evaluations/checkpoints --output_dir evaluations/downstream \
  --use_patch flatten --use_lrscheduler --use_fp16 \
  --data "$DATA_ROOT/Siena/Siena_250Hz/Segment"

# Linear probing
python "evaluations/[Siena]-linearprobe.py" \
  --eval_dir evaluations/checkpoints --output_dir evaluations/downstream \
  --use_patch flatten --use_lrscheduler --use_fp16 \
  --data "$DATA_ROOT/Siena/Siena_250Hz/Segment"
```

### TUAR

```bash
# Fine-tuning
python "evaluations/[TUAR]-finetune.py" \
  --eval_dir evaluations/checkpoints --output_dir evaluations/downstream \
  --use_patch pooling --use_lrscheduler --use_fp16 --epochs 20 \
  --data "$DATA_ROOT/TUAR/TUAR_250Hz/Segment"

# Linear probing
python "evaluations/[TUAR]-linearprobe.py" \
  --eval_dir evaluations/checkpoints --output_dir evaluations/downstream \
  --use_patch flatten --use_lrscheduler --use_fp16 \
  --data "$DATA_ROOT/TUAR/TUAR_250Hz/Segment"
```
