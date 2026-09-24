# Code_Training/data/collate.py

import torch
import random

def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, mask_generator=None, augmenter=None):
    """
    Args:
        samples_list: Batch list of raw data without augmentation: [{"data": Tensor(T, C, N)}, ...]
        mask_generator: Instance of MaskingGenerator (Dynamic call)
        augmenter: Instance of DataAugmentationDINO
    """
    # -------------------------------------------------------------------------
    # 1. Determine Batch Dimensions (Dynamic per batch)
    # -------------------------------------------------------------------------
    # Get random shapes for this specific batch
    # global_size: (g_C, g_N), local_size: (l_C, l_N)
    global_size, local_size = augmenter.sample_batch_dimensions()
    
    # -------------------------------------------------------------------------
    # 2. Apply Augmentation (Crop & Augment per sample)
    # -------------------------------------------------------------------------
    processed_samples = []
    for s in samples_list:
        raw_data = s["data"]  # (T, 19, 30)
        processed_sample = augmenter(raw_data, global_size, local_size)
        processed_samples.append(processed_sample)
    samples_list = processed_samples
    
    # -------------------------------------------------------------------------
    # 3. Collate Logic (Stacking)
    # -------------------------------------------------------------------------
    n_global_crops = len(samples_list[0]["global_crops"]) 
    n_local_crops = len(samples_list[0]["local_crops"])
    # Stack Global Crops: List[ (250, g_C, g_N) ] -> Tensor (B*2, 250, g_C, g_N)
    collated_global_crops = torch.stack([s["global_crops"][i] for i in range(n_global_crops) for s in samples_list])
    # Stack Local Crops: List[ (250, l_C, l_N) ] -> Tensor (B*8, 250, l_C, l_N)
    collated_local_crops = torch.stack([s["local_crops"][i] for i in range(n_local_crops) for s in samples_list])
    # Stack Indices: Global indices: (B*2, g_C) and (B*2, g_N)
    global_ch_idxs = torch.stack([s["global_indices"][i]["ch"] for i in range(n_global_crops) for s in samples_list])
    global_time_idxs = torch.stack([s["global_indices"][i]["time"] for i in range(n_global_crops) for s in samples_list])
    # Stack Indices: Lobal indices: (B*8, l_C) and (B*8, l_N)
    local_ch_idxs = torch.stack([s["local_indices"][i]["ch"] for i in range(n_local_crops) for s in samples_list])
    local_time_idxs = torch.stack([s["local_indices"][i]["time"] for i in range(n_local_crops) for s in samples_list])

    # -------------------------------------------------------------------------
    # 4. Masking Logic (Using Dynamic Dimensions)
    # -------------------------------------------------------------------------
    B = len(collated_global_crops)  # B*2
    C_global, N_global = global_size[0], global_size[1]  # for example: 19, 30
    num_tokens = C_global * N_global

    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        n_masked = int(num_tokens * random.uniform(prob_min, prob_max))
        mask = mask_generator(shape=(C_global, N_global), num_masking_patches=n_masked)
        masks_list.append(torch.BoolTensor(mask))
        upperbound += int(num_tokens * prob_max)
    for i in range(n_samples_masked, B):
        mask = mask_generator(shape=(C_global, N_global), num_masking_patches=0)
        masks_list.append(torch.BoolTensor(mask))

    random.shuffle(masks_list)
    collated_masks = torch.stack(masks_list).flatten(1)  # (B, C_global*N_global)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()
    mask_sum = collated_masks.sum(-1).clamp(min=1.0)
    masks_weight = (1 / mask_sum).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    return {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
        
        "global_ch_idxs": global_ch_idxs.long(),
        "global_time_idxs": global_time_idxs.long(),
        "local_ch_idxs": local_ch_idxs.long(),
        "local_time_idxs": local_time_idxs.long(),
    }