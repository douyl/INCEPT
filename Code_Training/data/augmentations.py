# Code_Training/data/augmentations.py

import logging
import torch
import random
# import numpy as np

logger = logging.getLogger("incept")

class DataAugmentationDINO(object):
    def __init__(
        self,
        # --- Dimensions ---
        full_channels=19,
        full_time_patches=30,
        patch_time_dim=250,
        local_crops_number=8,
        global_crop_channels_range=(15, 19), 
        global_crop_patches_range=(20, 30),
        local_crop_channels_range=(10, 15),
        local_crop_patches_range=(10, 15),
        
        # --- Augmentation Probabilities ---
        global_aug_probs=(1.0, 0.1), # Probabilities for the two global views
        local_aug_prob=0.5,          # Probability for local views
        # --- PCA & Spectral Augmentation Params ---
        pca_var_range=(0.90, 0.95),    # Range to pick Top-K variance threshold (e.g., keep 90%-95% energy)
        pca_weight_range=(0.7, 1.3),   # Range to scale the remaining "noise" components
        # --- Channel Dropout (Bad Leads) ---
        dropout_ch_prob=0.1,        # Probability to drop channels
        dropout_max_channels=2,     # Max number of channels to drop (set to 0)
    ):
        # Store dimensions
        self.full_C = full_channels
        self.full_N = full_time_patches
        self.T = patch_time_dim
        self.n_local_crops = local_crops_number
        self.global_C_range = global_crop_channels_range
        self.global_N_range = global_crop_patches_range
        self.local_C_range = local_crop_channels_range
        self.local_N_range = local_crop_patches_range
        
        # Probabilities & Params
        self.global_aug_probs = global_aug_probs
        self.local_aug_prob = local_aug_prob
        self.pca_var_range = pca_var_range
        self.pca_weight_range = pca_weight_range
        self.dropout_ch_prob = dropout_ch_prob
        self.dropout_max_channels = dropout_max_channels

        # Define EEG Frequency Bands
        self.bands_dict = {
            'Delta': (0.1, 4), 'Theta': (4, 8), 'Alpha': (8, 12), 
            'Beta': (12, 30), 'Gamma': (30, 124)
        }

        # Logging configuration
        logger.info(
            f"  \nEEG Data Augmentation (Dynamic Batch Shapes)\n"
            f"  [Global Range] C: {self.global_C_range}, N: {self.global_N_range}\n"
            f"  [Local Range]  C: {self.local_C_range}, N: {self.local_N_range}\n"
            f"  [Aug] Method: FFT-based Multi-band PCA + Channel Dropout\n"
            f"  [PCA] PCA Var Range: {self.pca_var_range}, Noise Weight: {self.pca_weight_range}\n"
            f"  [Dropout] Prob: {self.dropout_ch_prob}, Max Ch: {self.dropout_max_channels}"
        )

    def _augment_wrapper(self, data, p_apply):
        """
        Wrapper to handle reshaping and probability application.
        Input: (T, C, N) -> Flatten -> Augment -> Reshape back
        """
        if random.random() >= p_apply:
            return data

        x_cont = data.permute(1, 2, 0)  # (T, C, N) -> (C, N, T)
        C_curr, N_curr, T_curr = x_cont.shape
        x_flat = x_cont.reshape(C_curr, -1)  # (C, N, T) -> (C, N*T)

        # Apply FFT-PCA Augmentation
        x_aug_flat = self._apply_pca_augmentation(x_flat)  # (C, N*T)

        # Apply Channel Dropout
        x_aug_flat = self._apply_channel_dropout(x_aug_flat)  # (C, N*T)

        # 5. Reshape back: 
        x_aug = x_aug_flat.reshape(C_curr, N_curr, T_curr).permute(2, 0, 1)  # (C, N*T) -> (C, N, T) -> (T, C, N)
        
        return x_aug

    def _apply_pca_augmentation(self, x_flat):
        """
        Applies Multi-band PCA Augmentation.
        Decomposes signal into bands, applies PCA per band, keeps Top-K components, and perturbs the remaining 'noise' components.
        Args:
            x_flat: Input Tensor (Float32). Shape: (Channels, Time)
        """
        C, n_samples = x_flat.shape  # (C, N*T) = (19, 30*250) on CPU
        
        # ---------------- 1. Vectorized FFT & Filtering ----------------
        fft_spectrum = torch.fft.rfft(x_flat, dim=1)  # (C, N*T) -> (C, Freq_Bins=T/2+1) = (19, 3751)
        freqs = torch.fft.rfftfreq(n_samples, 1/self.T)  # (Freq_Bins,) = (F,)
        
        # Create Boolean Masks for 5 bands (Delta, Theta, Alpha, Beta, Gamma)
        masks = torch.stack([(freqs >= l) & (freqs < h) for l, h in self.bands_dict.values()])  # (5, F)
        n_bands = masks.shape[0]  # 5

        # Apply masks using broadcasting (Batching over bands)
        filtered_spectra = fft_spectrum.unsqueeze(0) * masks.unsqueeze(1)  # (1, C, F) * (5, 1, F) -> (5, C, F)
        
        # Inverse FFT to get time-domain signals for each band
        band_signals = torch.fft.irfft(filtered_spectra, n=n_samples, dim=2)  # (5, C, F) -> (5, C, N*T)
        
        # Calculate Residual (Original - Sum of all bands)
        residual = x_flat - torch.sum(band_signals, dim=0)  # (C, N*T)

        # ---------------- 2. Batch PCA Calculation ----------------
        # Center the data per band
        means = torch.mean(band_signals, dim=2, keepdim=True)  # (5, C, 1)
        centered = band_signals - means  # (5, C, N*T) - (5, C, 1) -> (5, C, N*T)
        
        # Compute Covariance Matrix (Batch)
        cov_matrices = torch.matmul(centered, centered.transpose(1, 2)) / (n_samples - 1)  # (5, C, T) @ (5, T, C) -> (5, C, C)
        
        # Eigen Decomposition (returns eigenvalues L and eigenvectors V)
        L, V = torch.linalg.eigh(cov_matrices)  # L: (5, C) ascending, V: (5, C, C)
        
        # Sort indices descending (we want largest variance first)
        sorted_idx = torch.argsort(L, dim=1, descending=True)  # (5, C)
        # Reorder Eigenvalues
        L_sorted = torch.gather(L, 1, sorted_idx)  # (5, C)
        # Reorder Eigenvectors (Columns correspond to eigenvalues)
        idx_expanded = sorted_idx.unsqueeze(1).expand(-1, C, -1)  # (5, C) -> (5, 1, C) -> (5, C, C)
        V_sorted = torch.gather(V, 2, idx_expanded)  # (5, C, C)
        
        # Project Data to PCA Space (计算每个频段的C个主成分)
        sources = torch.matmul(V_sorted.transpose(1, 2), centered)  # (5, C, C)^T @ (5, C, N*T) -> (5, C, N*T)

        # ---------------- 3. Augmentation (Perturb Noise) ----------------
        # Pick a random variance threshold (e.g., 0.92)
        var_threshold = random.uniform(*self.pca_var_range)
        # Calculate Cumulative Variance Ratio (主成分的方差，计算每个频段内C个主成分各可以恢复多少信号)
        explained_ratios = torch.cumsum(L_sorted, dim=1) / torch.sum(L_sorted, dim=1, keepdim=True)  # (5, C)
        
        # Identify Top-K components (where cumulative variance > threshold) and noise components 
        noise_mask = explained_ratios > var_threshold  # (5, C) Boolean Mask
        noise_mask[:, 0] = False # Always keep the 1st component (Principal Component)

        # Generate Random Weights for noise components
        noise_weights = torch.empty((n_bands, C, 1)).uniform_(*self.pca_weight_range)  # (5, C, 1)
        
        # Create final scaling factor: 1.0 for signal, random weight for noise
        scalers = torch.where(noise_mask.unsqueeze(-1), noise_weights, torch.tensor(1.0))  # (5, C, 1)
        
        # Apply scaling to sources
        sources_aug = sources * scalers  # (5, C, N*T) * (5, C, 1) -> (5, C, N*T)
        
        # ---------------- 4. Reconstruction ----------------
        # Project back to Signal Space
        recon_bands = torch.matmul(V_sorted, sources_aug) + means   # (5, C, C) @ (5, C, N*T) -> (5, C, N*T)
        
        # Sum all bands and add the unchanged residual
        x_aug = torch.sum(recon_bands, dim=0) + residual  # (5, C, N*T) -> (C, N*T)
        return x_aug

    def _apply_channel_dropout(self, x_flat):
        """Sets random channels to zero to simulate bad leads."""
        if self.dropout_ch_prob > 0 and random.random() < self.dropout_ch_prob:
            C = x_flat.shape[0]
            num_drop = random.randint(1, self.dropout_max_channels)
            # Randomly select channels indices
            drop_indices = torch.randperm(C)[:num_drop]
            x_flat[drop_indices, :] = 0.0  # Set to 0
        return x_flat

    def sample_batch_dimensions(self):
        # 1. Sample Global Dimensions
        g_c = random.randint(*self.global_C_range)
        g_n = random.randint(*self.global_N_range)
        # 2. Sample Local Dimensions
        l_c = random.randint(*self.local_C_range)
        l_n = random.randint(*self.local_N_range)
        # Enforce hard constraint: Local < Global (optional but good for safety)
        l_c = min(l_c, g_c)
        l_n = min(l_n, g_n)
        return (g_c, g_n), (l_c, l_n)

    def _crop(self, data, target_C, target_N):
        """
        Randomly crops the input data to target dimensions.
        Data: (T, C_full, N_full)
        """
        # Random Channel Selection
        crop_ch_idxs = torch.randperm(self.full_C)[:target_C].sort()[0]
        # Random Time Window Selection
        start_t = torch.randint(0, self.full_N - target_N + 1, (1,)).item()
        crop_time_idxs = torch.arange(start_t, start_t + target_N)
        crop_data = data.index_select(1, crop_ch_idxs).index_select(2, crop_time_idxs)  # (T, C, N) -> (T, target_C, target_N)
        return crop_data, crop_ch_idxs, crop_time_idxs

    def __call__(self, data, global_size, local_size):
        """
        Args:
            data: Raw Tensor (T, C_full, N_full)
            global_size: Tuple (C, N) determined by collate_fn
            local_size: Tuple (c, n) determined by collate_fn
        """
        output = {}
        
        # --- 1. Global Crops (Dynamic Size) ---
        global_crops = []
        global_indices = []
        g_C, g_N = global_size
        for i in range(2):
            crop_data, ch_idxs, time_idxs = self._crop(data, g_C, g_N)
            p = self.global_aug_probs[i]
            crop_aug = self._augment_wrapper(crop_data, p_apply=p)
            global_crops.append(crop_aug)
            global_indices.append({"ch": ch_idxs, "time": time_idxs})
        output["global_crops"] = global_crops
        output["global_indices"] = global_indices

        # --- 2. Local Crops (Dynamic Size) ---
        local_crops = []
        local_indices = []
        l_C, l_N = local_size
        for _ in range(self.n_local_crops):
            crop_data, ch_idxs, time_idxs = self._crop(data, l_C, l_N)
            crop_aug = self._augment_wrapper(crop_data, p_apply=self.local_aug_prob)
            local_crops.append(crop_aug)
            local_indices.append({"ch": ch_idxs, "time": time_idxs})
        output["local_crops"] = local_crops
        output["local_indices"] = local_indices

        return output
