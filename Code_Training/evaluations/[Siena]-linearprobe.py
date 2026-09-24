import os, sys, yaml, torch, argparse, random, gc, logging, warnings, datetime, mne
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S', stream=sys.stdout)
logger = logging.getLogger(__name__)

SIENA_CHANNELS_29 = [
    'Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5', 'FC1', 'FC5',
    'CP1', 'CP5', 'F9', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4',
    'O2', 'F8', 'T4', 'T6', 'FC2', 'FC6', 'CP2', 'CP6', 'F10'
]

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Bal_Acc": round(balanced_accuracy_score(y_true, y_pred), 5),
        "Acc": round(accuracy_score(y_true, y_pred), 5),
        "AUROC": round(roc_auc_score(y_true, y_prob), 5),
        "AUC-PR": round(average_precision_score(y_true, y_prob), 5),
    }

class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'): value = DotDict(value)
            self[key] = value

# ==========================
# 1. Coordinate Loading
# ==========================
def get_channel_coords():
    # Directly fetch from MNE standard_1020
    montage = mne.channels.make_standard_montage('standard_1020')
    pos_dict = montage.get_positions()['ch_pos']
    coords_np = np.stack([pos_dict[name] for name in SIENA_CHANNELS_29])
    logger.info(f"Loaded 29 Channel Coordinates from MNE standard_1020 for Siena")
    return torch.tensor(coords_np, dtype=torch.float32)

# ==========================
# 1. Dataset
# ==========================
class SienaDataset(Dataset):
    def __init__(self, root, split="train"):
        # normal -> 0, seizure -> 1
        self.classes = ['normal', 'seizure']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = []
        
        self.n_channels = 29
        self.n_segments = 10
        
        if not os.path.exists(root):
            logger.warning(f"Warning: Directory not found {root}")
            return

        eval_subjects = ['PN16', 'PN17']
        eval_pool = []
        train_val_pool = []
        # Traverse: Root -> Subject -> Class(normal/seizure) -> .npy files
        for sub_name in sorted(os.listdir(root)):
            sub_path = os.path.join(root, sub_name)
            if not os.path.isdir(sub_path): continue
            for class_name in self.classes:
                class_path = os.path.join(sub_path, class_name)
                if not os.path.isdir(class_path): continue
                for file in sorted(os.listdir(class_path)):
                    if file.endswith('.npy'):
                        full_path = os.path.join(class_path, file)
                        item = (full_path, self.class_to_idx[class_name])
                        if sub_name in eval_subjects:
                            eval_pool.append(item)
                        else:
                            train_val_pool.append(item)

        if split == "eval" or split == "test":
            self.samples = eval_pool
        else:
            # 1. Sort the pool to ensure absolute deterministic order before shuffling
            train_val_pool.sort(key=lambda x: x[0])
            # 2. Use a local random instance with fixed seed 42
            local_rng = random.Random(42)
            local_rng.shuffle(train_val_pool)
            # 3. Perform 8:2 split
            split_idx = int(len(train_val_pool) * 0.8)
            if split == "train":
                self.samples = train_val_pool[:split_idx]
            elif split == "val":
                self.samples = train_val_pool[split_idx:]
        
        self.ch_idx = torch.arange(self.n_channels)
        self.t_idx = torch.arange(self.n_segments)
        logger.info(f"[{split.upper()}] Samples: {len(self.samples)} | Classes: 2 | Channels: {self.n_channels} | Segments: {self.n_segments}")

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        data = np.load(path) # Expected shape: (29, 10, 250)
        
        # Z-score normalization
        mean = data.mean(axis=(1, 2), keepdims=True)
        std = data.std(axis=(1, 2), keepdims=True)
        data = np.divide(data - mean, std, out=np.zeros_like(data), where=std > 1e-8)
        
        # To Tensor: (C, N, T) -> (T, C, N) for EEGDino Input
        data = torch.from_numpy(data).float().permute(2, 0, 1) 
        
        return data, label, self.ch_idx, self.t_idx

# ==========================
# 2. Model Utilities
# ==========================
def load_backbone(cfg, ckpt_path, n_segments=1):
    import Code_Training.models.vision_transformer as ViT
    ViT.get_eeg_coords = get_channel_coords
    logger.info(f"Monkey-patched {ViT.__name__}.get_eeg_coords for Siena 29ch.")

    args = dict(num_channels=29, num_patches_per_channel=n_segments, patch_time_dim=cfg.dataset.patch_time_dim, 
                init_values=cfg.student.layerscale, ffn_layer=cfg.student.ffn_layer, block_chunks=cfg.student.block_chunks,
                qkv_bias=cfg.student.qkv_bias, proj_bias=cfg.student.proj_bias, ffn_bias=cfg.student.ffn_bias, 
                num_register_tokens=cfg.student.num_register_tokens, drop_path_rate=cfg.student.drop_path_rate, 
                drop_path_uniform=cfg.student.drop_path_uniform, channel_embed_sh_degree=cfg.student.channel_embed_sh_degree)

    model = ViT.__dict__[cfg.student.arch.replace("_memeff", "")](**args)
    state = torch.load(ckpt_path, map_location="cpu")
    clean_state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in (state["teacher"] if "teacher" in state else state).items()}

    # Drop incompatible buffers/weights
    for k in ["channel_embed.channel_coords", "channel_embed.channel_spherical_harmonics"]:
        if k in clean_state: del clean_state[k]

    msg = model.load_state_dict(clean_state, strict=False)
    logger.info(f"Loaded backbone weights. Missing keys: {msg.missing_keys}")
    return model

def get_embedding(features, n_cls_layers, use_patch):
    # 1. Take the last n blocks
    subset = features[-n_cls_layers:]
    # 2. Process CLS Tokens
    cls_feats = [cls.flatten(1) for _, cls in subset]   # (B, C, D) -> (B, C*D)
    out = torch.cat(cls_feats, dim=-1)  # (B, n_cls_layers * C * D)
    # 3. Process Patch Tokens (from the very last block)
    last_patch_tokens = subset[-1][0]  # (B, C*N, D)
    
    if use_patch == "pooling":
        out = torch.cat((out, last_patch_tokens.mean(dim=1)), dim=-1)  # (B, C*N, D) -> (B, D)
    elif use_patch == "flatten":
        out = torch.cat((out, last_patch_tokens.flatten(1)), dim=-1)  # (B, C*N, D) -> (B, C*N*D)
    return out

# ==========================
# 3. Solver (Linear Probe)
# ==========================
class LPModel(nn.Module):
    def __init__(self, backbone, input_dim, num_classes=2, n_cls_layers=4, use_patch=False):
        super().__init__()
        self.backbone = backbone
        self.n_cls_layers = n_cls_layers
        self.use_patch = use_patch
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        
        # Single linear head
        self.head = nn.Linear(input_dim, num_classes)

    def forward(self, x, c, t):
        # Extract features without computing gradients for backbone
        with torch.no_grad():
            feats = self.backbone.get_intermediate_layers(x, self.n_cls_layers, return_class_token=True, ch_idxs=c, time_idxs=t)
            x_embed = F.normalize(get_embedding(feats, self.n_cls_layers, self.use_patch), dim=-1)
        
        # Pass embeddings to the single linear head
        return self.head(x_embed)

def run_inference(loader, model, dev, args):
    model.eval()
    targets, preds, probs = [], [], []
    with torch.no_grad():
        for x, y, c, t in loader:
            x, y, c, t = x.to(dev), y.to(dev), c.to(dev), t.to(dev)
            with autocast(enabled=args.use_fp16): 
                logits = model(x, c, t)
                out_prob = torch.softmax(logits, dim=1)
            preds.extend(logits.argmax(1).cpu().numpy())
            targets.extend(y.cpu().numpy())
            probs.extend(out_prob[:, 1].cpu().numpy())
    return targets, preds, probs

def run_linearprobe_seed(cfg, args, dev, current_seed):
    logger.info("\n" + "="*20 + " Siena Linear Probing (29ch) " + "="*20)
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logger.info(f"Run ID: {run_id} | Seed: {current_seed} | CLSLayers: {args.n_cls_layers} | UsePatch: {args.use_patch} | FP16: {args.use_fp16}")

    ds_tr = SienaDataset(args.data, "train")
    ds_val = SienaDataset(args.data, "val")
    ds_test = SienaDataset(args.data, "eval")
    
    loaders = {
        "train": DataLoader(ds_tr, args.batchsize, True, num_workers=8, pin_memory=True, drop_last=True),
        "val": DataLoader(ds_val, args.batchsize, False, num_workers=8, pin_memory=True),
        "test": DataLoader(ds_test, args.batchsize, False, num_workers=8)
    }

    backbone = load_backbone(cfg, f"{args.eval_dir}/teacher_checkpoint.pth", n_segments=ds_tr.n_segments).to(dev)
    
    with torch.no_grad():
        d = ds_tr[0]
        feats = backbone.get_intermediate_layers(
            d[0].unsqueeze(0).to(dev), 
            args.n_cls_layers, 
            return_class_token=True, 
            ch_idxs=d[2].unsqueeze(0).to(dev), 
            time_idxs=d[3].unsqueeze(0).to(dev)
        )
        input_dim = get_embedding(feats, args.n_cls_layers, args.use_patch).shape[1]

    model = LPModel(backbone, input_dim, num_classes=2, n_cls_layers=args.n_cls_layers, use_patch=args.use_patch).to(dev)
    
    # Train only the linear head
    optimizer = torch.optim.AdamW([{'params': model.head.parameters(), 'lr': args.lr}], weight_decay=args.weightdecay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(loaders["train"]), eta_min=1e-6) if args.use_lrscheduler else None
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = GradScaler(enabled=args.use_fp16)

    save_path = os.path.join(args.output_dir, "Siena", f"temp_lp_ckpt_{run_id}.pth")
    best_val_auroc, best_epoch = -1.0, -1
    for ep in range(args.epochs):
        model.train()
        model.backbone.eval() # Ensure backbone is kept in eval mode
        epoch_loss = 0.0
        for step, (x, y, c, t) in enumerate(loaders["train"]):
            if ep == 0 and step == 0: torch.cuda.reset_peak_memory_stats()
            x, y, c, t = x.to(dev), y.to(dev), c.to(dev), t.to(dev)
            optimizer.zero_grad()
            with autocast(enabled=args.use_fp16): 
                logits = model(x, c, t)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            if hasattr(args, 'clip_value') and args.clip_value > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), args.clip_value)
            scaler.step(optimizer)
            scaler.update()
            scheduler and scheduler.step()
            epoch_loss += loss.item()
            if ep == 0 and step == 0: logger.info(f"Training | Max Mem: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

        targets, preds, probs = run_inference(loaders["val"], model, dev, args)
        curr_metric = compute_metrics(targets, preds, probs)

        is_best = False
        if curr_metric['AUROC'] > best_val_auroc:
            best_val_auroc, best_epoch = curr_metric['AUROC'], ep + 1
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            is_best = True
        
        logger.info(f"Ep {ep+1:03d} | Loss: {epoch_loss/len(loaders['train']):.5f} | Val: {curr_metric} "
                    f"| LR: {optimizer.param_groups[0]['lr']:.2e}" + (" [NEW BEST]" if is_best else ""))
    
    final_metrics = {}
    if best_epoch != -1 and os.path.exists(save_path):
        logger.info(f"Loading Best Checkpoint (Ep {best_epoch}) for Final Test...")
        model.load_state_dict(torch.load(save_path))
        
        final_t, final_p, final_prob = run_inference(loaders["test"], model, dev, args)
        final_metrics = compute_metrics(final_t, final_p, final_prob)
        logger.info(f"FINAL TEST RESULTS (Seed: {current_seed}): {final_metrics}")
        
        global_save_path = os.path.join(args.output_dir, "Siena", f"{run_id}_LP_seed{current_seed}.pth")
        os.replace(save_path, global_save_path)
    else:
        logger.warning(f"No valid checkpoint saved for Seed {current_seed}, test skipped.")
    
    del model, optimizer, scaler, scheduler, backbone; gc.collect(); torch.cuda.empty_cache()
    return final_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_dir', default="evaluations/checkpoints")
    parser.add_argument('--output_dir', default="evaluations/downstream")
    parser.add_argument('--data', required=True, help="Path to Siena datasets")
    parser.add_argument('--seed', type=int, nargs='+', default=[42, 3407, 0, 1024, 1234])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weightdecay', type=float, default=0.1)
    parser.add_argument('--clip_value', type=float, default=0)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--use_lrscheduler', action='store_true')
    parser.add_argument('--use_fp16', action='store_true', help='Use mixed precision (AMP) for training and inference')
    parser.add_argument('--n_cls_layers', type=int, default=1, help='Number of blocks to extract CLS token')
    parser.add_argument('--use_patch', type=str, default='flatten', choices=['flatten', 'pooling', 'false'])
    args = parser.parse_args()
    if args.use_patch.lower() == 'false':
        args.use_patch = False

    logger.info("="*20 + " ARGS " + "="*20)
    for k, v in vars(args).items(): logger.info(f"{k:<25}: {v}")
    logger.info("="*46)

    dev = torch.device('cuda')
    cfg = DotDict(yaml.safe_load(open(f"{args.eval_dir}/config.yaml")))
    os.makedirs(os.path.join(args.output_dir, "Siena"), exist_ok=True)

    all_metrics_results = []
    total_seeds = len(args.seed)
    
    for idx, current_seed in enumerate(args.seed):
        logger.info(f"\n>>> Starting Run {idx + 1}/{total_seeds} [Seed: {current_seed}]")
        setup_seed(current_seed)
        metrics = run_linearprobe_seed(cfg, args, dev, current_seed)
        if metrics: all_metrics_results.append((current_seed, metrics))

    logger.info("\n" + "="*20 + " FINAL RESULTS ACROSS ALL SEEDS " + "="*20)
    if all_metrics_results:
        aggregated_metrics = {k: [] for k in all_metrics_results[0][1].keys()}
        for seed, metrics in all_metrics_results:
            logger.info(f"Seed {seed:<5} | {metrics}")
            for key, value in metrics.items():
                aggregated_metrics[key].append(value)
        logger.info("-" * 62)
        for key, values in aggregated_metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            logger.info(f"{key:<10}: Mean = {mean_val:.5f}, Std = {std_val:.5f}")
        logger.info("="*64)