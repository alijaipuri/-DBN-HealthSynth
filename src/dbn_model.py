import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RBM(nn.Module):
    """Restricted Boltzmann Machine - building block of DBN"""
    def __init__(self, n_visible, n_hidden, learning_rate=0.01):
        super(RBM, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.training_history = []

    def sample_h(self, v):
        wx = torch.mm(v, self.W.t()) + self.h_bias
        p_h = torch.sigmoid(wx)
        return p_h, torch.bernoulli(p_h)

    def sample_v(self, h):
        wx = torch.mm(h, self.W) + self.v_bias
        p_v = torch.sigmoid(wx)
        return p_v, torch.bernoulli(p_v)

    def contrastive_divergence(self, v0, k=1):
        p_h0, h0 = self.sample_h(v0)
        vk, hk = v0.clone(), h0.clone()
        for _ in range(k):
            p_hk, hk = self.sample_h(vk)
            p_vk, vk = self.sample_v(hk)
        self.W.data += self.lr * (torch.mm(p_h0.t(), v0) - torch.mm(p_hk.t(), vk)) / v0.size(0)
        self.v_bias.data += self.lr * torch.mean(v0 - vk, dim=0)
        self.h_bias.data += self.lr * torch.mean(p_h0 - p_hk, dim=0)
        loss = torch.mean((v0 - vk) ** 2).item()
        return loss

    def forward(self, v):
        p_h, h = self.sample_h(v)
        p_v, v_reconstructed = self.sample_v(h)
        return p_v

    def get_hidden_representation(self, v):
        p_h, _ = self.sample_h(v)
        return p_h


class DBN(nn.Module):
    """Deep Belief Network composed of stacked RBMs"""
    def __init__(self, layer_sizes, learning_rate=0.01):
        super(DBN, self).__init__()
        self.layer_sizes = layer_sizes
        self.rbm_layers = nn.ModuleList()
        self.training_losses = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.fine_tune_losses = []

        for i in range(len(layer_sizes) - 1):
            rbm = RBM(layer_sizes[i], layer_sizes[i+1], learning_rate)
            self.rbm_layers.append(rbm)

        # Fine-tuning network
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        self.fine_tune_net = nn.Sequential(*layers)
        self.decoder = self._build_decoder()

    def _build_decoder(self):
        rev = list(reversed(self.layer_sizes))
        layers = []
        for i in range(len(rev) - 1):
            layers.append(nn.Linear(rev[i], rev[i+1]))
            if i < len(rev) - 2:
                layers.append(nn.BatchNorm1d(rev[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
            else:
                layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def pretrain(self, data_tensor, epochs=50, batch_size=64, progress_callback=None):
        logger.info("Starting DBN pre-training (greedy layer-wise)...")
        input_data = data_tensor.clone()
        for layer_idx, rbm in enumerate(self.rbm_layers):
            logger.info(f"Pre-training RBM layer {layer_idx+1}/{len(self.rbm_layers)}")
            layer_losses = []
            for epoch in range(epochs):
                perm = torch.randperm(input_data.size(0))
                shuffled = input_data[perm]
                epoch_loss = 0
                n_batches = 0
                for i in range(0, shuffled.size(0), batch_size):
                    batch = shuffled[i:i+batch_size]
                    if batch.size(0) < 2:
                        continue
                    loss = rbm.contrastive_divergence(batch, k=1)
                    epoch_loss += loss
                    n_batches += 1
                avg_loss = epoch_loss / max(n_batches, 1)
                layer_losses.append(avg_loss)
                if progress_callback:
                    progress_callback(layer_idx, epoch, epochs, avg_loss)
            self.training_losses[f'rbm_{layer_idx+1}'] = layer_losses
            with torch.no_grad():
                next_input = rbm.get_hidden_representation(input_data)
            input_data = next_input.detach()
        logger.info("Pre-training complete.")

    def fine_tune(self, data_tensor, epochs=30, batch_size=64):
        logger.info("Fine-tuning DBN with backpropagation...")
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            perm = torch.randperm(data_tensor.size(0))
            shuffled = data_tensor[perm]
            epoch_loss = 0
            n_batches = 0
            for i in range(0, shuffled.size(0), batch_size):
                batch = shuffled[i:i+batch_size]
                if batch.size(0) < 2:
                    continue
                optimizer.zero_grad()
                encoded = self.fine_tune_net(batch)
                decoded = self.decoder(encoded)
                loss = criterion(decoded, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg = epoch_loss / max(n_batches, 1)
            self.fine_tune_losses.append(avg)
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                logger.info(f"Fine-tune Epoch {epoch+1}/{epochs}, Loss: {avg:.4f}")
        self.is_fitted = True
        logger.info("Fine-tuning complete.")

    def generate_samples(self, n_samples, temperature=1.0, noise_level=0.1):
        """Generate synthetic patient records with temperature scaling"""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before generating samples.")
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.layer_sizes[-1]) * temperature
            samples = self.decoder(z)
            noise = torch.randn_like(samples) * noise_level
            samples = samples + noise
        return samples.numpy()

    def get_latent_representation(self, data_tensor):
        self.eval()
        with torch.no_grad():
            return self.fine_tune_net(data_tensor).numpy()

    def forward(self, x):
        encoded = self.fine_tune_net(x)
        return self.decoder(encoded)
