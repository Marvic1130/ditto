import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import wandb
import sys
import os
import time
from pathlib import Path
import pandas as pd

# Add external to path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.log_helpers import build_round_log
from external.FedAvg.fed_avg import FedAvg
from external.FedAvg.utils import average_weights

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb * 1024 * 1024 # return bytes

class Ditto(FedAvg):
    def __init__(self, args, cfg=None):
        super().__init__(args, cfg)
        self.lam = getattr(self.args, 'lam', 0.1)
        
    def _train_client(self, root_model, train_loader, client_idx):
        model = copy.deepcopy(root_model)
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.args.lr, momentum=self.args.momentum
        )
        
        global_state = {k: v.detach().clone() for k, v in root_model.named_parameters()}
        
        total_loss = 0.0
        total_batches = 0
        total_correct = 0
        total_samples = 0
        for epoch in range(self.args.n_client_epochs):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                if self.args.model_name == 'cnn' and len(data.shape) == 2:
                     # Try to infer square image size
                     side = int(data.shape[1] ** 0.5)
                     if side * side == data.shape[1]:
                         data = data.view(-1, 1, side, side)
                     else:
                         # Fallback or error? For now assume 3 channels if not square?
                         # Actually LEAF data is usually grayscale.
                         # If it's CIFAR (3072), side would be ~55.4, so it fails.
                         # CIFAR is 3x32x32 = 3072.
                         if data.shape[1] == 3072:
                             data = data.view(-1, 3, 32, 32)
                         elif data.shape[1] == 784:
                             data = data.view(-1, 1, 28, 28)

                optimizer.zero_grad()

                logits = model(data)
                task_loss = F.cross_entropy(logits, target)
                
                # Ditto Regularization
                reg_loss = 0.0
                for name, param in model.named_parameters():
                    if name in global_state:
                        reg_loss += torch.sum((param - global_state[name]) ** 2)
                
                loss = task_loss + (self.lam / 2.0) * reg_loss
                
                loss.backward()
                optimizer.step()

                preds = logits.argmax(dim=1)
                total_correct += (preds == target).sum().item()
                total_samples += data.size(0)
                total_loss += loss.item()
                total_batches += 1

            avg_loss = total_loss / max(total_batches, 1)
            avg_acc = total_correct / max(total_samples, 1)
            return model, avg_loss, avg_acc, total_samples

    def train(self):
        wandb.init(
            project=self.cfg.wandb_project if self.cfg and hasattr(self.cfg, 'wandb_project') else "HyperQLoRA-HFL",
            name=f"ditto_{self.cfg.dataset.name if self.cfg else 'default'}",
            config=vars(self.args),
            reinit=True
        )
        
        train_losses = []
        
        # Calculate model size for traffic estimation
        model_size_bytes = get_model_size(self.root_model)

        for epoch in range(self.args.n_epochs):
            round_start_time = time.time()
            clients_models = []
            clients_losses = []
            client_accs = []
            client_durations = []

            # Determine participants per round
            m = max(int(self.args.frac * self.args.n_clients), 1)
            # Collect per-client sample counts for weighted aggregation
            client_sample_counts = []
            
            self.root_model.train()

            if self.train_loaders:
                client_ids = list(self.train_loaders.keys())
                # Align selection count with actual available clients (like FedAvg)
                m = max(int(self.args.frac * len(client_ids)), 1)
                selected_ids = np.random.choice(client_ids, m, replace=False)
                
                for cid in selected_ids:
                    loader = self.train_loaders[cid]
                    client_start_time = time.time()
                    client_model, client_loss, client_acc, client_samples = self._train_client(
                        root_model=self.root_model,
                        train_loader=loader,
                        client_idx=cid
                    )
                    client_duration = time.time() - client_start_time
                    client_durations.append(client_duration)

                    clients_models.append(client_model.state_dict())
                    clients_losses.append(client_loss)
                    client_accs.append(client_acc)
                    client_sample_counts.append(int(client_samples))
            else:
                # Fallback for non-LEAF
                idx_clients = np.random.choice(range(self.args.n_clients), m, replace=False)
                for client_idx in idx_clients:
                    self.train_loader.sampler.set_client(client_idx)
                    client_start_time = time.time()
                    client_model, client_loss, client_acc, client_samples = self._train_client(
                        root_model=self.root_model,
                        train_loader=self.train_loader,
                        client_idx=client_idx,
                    )
                    client_duration = time.time() - client_start_time
                    client_durations.append(client_duration)

                    clients_models.append(client_model.state_dict())
                    clients_losses.append(client_loss)
                    client_accs.append(client_acc)
                    client_sample_counts.append(int(client_samples))

            server_start_time = time.time()
            updated_weights = average_weights(clients_models, client_sample_counts)
            self.root_model.load_state_dict(updated_weights)
            server_duration = time.time() - server_start_time

            avg_loss = sum(clients_losses) / len(clients_losses)
            train_losses.append(avg_loss)

            if (epoch + 1) % self.args.log_every == 0:
                test_losses = []
                test_accs = []
                
                self.root_model.eval()
                
                loaders = []
                if hasattr(self, 'test_loaders') and self.test_loaders:
                     loaders = list(self.test_loaders.values())
                elif self.test_loader:
                     loaders = [self.test_loader]
                
                with torch.no_grad():
                    total_global_correct = 0
                    total_global_samples = 0
                    for loader in loaders:
                        total_loss = 0.0
                        total_correct = 0.0
                        total_samples = 0
                        
                        for data, target in loader:
                            data, target = data.to(self.device), target.to(self.device)
                            if self.args.model_name == 'cnn' and len(data.shape) == 2:
                                 # Try to infer square image size
                                 side = int(data.shape[1] ** 0.5)
                                 if side * side == data.shape[1]:
                                     data = data.view(-1, 1, side, side)
                                 elif data.shape[1] == 3072:
                                     data = data.view(-1, 3, 32, 32)
                                 elif data.shape[1] == 784:
                                     data = data.view(-1, 1, 28, 28)
                            logits = self.root_model(data)
                            # Use cross-entropy on raw logits for consistency with training
                            loss = F.cross_entropy(logits, target, reduction='sum')
                            total_loss += loss.item()
                            total_correct += (logits.argmax(dim=1) == target).sum().item()
                            total_samples += data.size(0)
                        
                        if total_samples > 0:
                            test_losses.append(total_loss / total_samples)
                            test_accs.append(total_correct / total_samples * 100.0)
                            total_global_correct += total_correct
                            total_global_samples += total_samples
                
                # Traffic estimation (per-client average, for consistency)
                # Downlink: server -> client sends global model weights (model_size)
                # Uplink: client -> server sends updated weights or gradients (model_size)
                # Report per-client average bytes, not total across m clients
                avg_tx_bytes = model_size_bytes
                avg_rx_bytes = model_size_bytes
                avg_tx_mb = avg_tx_bytes / (1024 * 1024)
                avg_rx_mb = avg_rx_bytes / (1024 * 1024)

                test_accs_fraction = [acc / 100.0 for acc in test_accs]
                
                max_client_duration = max(client_durations) if client_durations else 0.0
                mean_client_duration = np.mean(client_durations) if client_durations else 0.0
                simulated_duration = max_client_duration + server_duration

                log_data = build_round_log(
                    round_idx=epoch + 1,
                    meta_loss=None,
                    client_losses=clients_losses,
                    client_accs=client_accs,
                    val_losses=test_losses,
                    val_accs=test_accs_fraction,
                    duration=simulated_duration,
                    tx_bytes=avg_tx_bytes,
                    rx_bytes=avg_rx_bytes,
                )
                log_data['client_duration'] = mean_client_duration
                log_data['max_client_duration'] = max_client_duration
                log_data['server_duration'] = server_duration
                log_data['simulated_duration'] = simulated_duration
                # Wall clock duration for parity across algorithms
                log_data['wall_clock_duration'] = time.time() - round_start_time
                
                wandb.log(log_data)
                
                weighted_test_acc = total_global_correct / total_global_samples * 100.0 if total_global_samples > 0 else 0.0
                print(f"Round {epoch+1} | Train Loss: {avg_loss:.4f} | Test Acc: {weighted_test_acc:.2f}%")

                # Append per-round convergence metrics to CSV
                try:
                    out_dir = Path('save/ditto')
                    out_dir.mkdir(parents=True, exist_ok=True)
                    csv_path = out_dir / 'convergence.csv'
                    train_acc_mean = float(np.mean(client_accs)) if client_accs else None
                    test_loss_mean = float(np.mean(test_losses)) if test_losses else None
                    test_acc_mean = float(np.mean(test_accs_fraction)) if test_accs_fraction else None
                    row = {
                        'round': int(epoch + 1),
                        'train_loss': float(avg_loss),
                        'train_acc': train_acc_mean,
                        'test_loss': test_loss_mean,
                        'test_acc': test_acc_mean,
                        'max_client_duration': float(max_client_duration) if isinstance(max_client_duration, (int, float)) else None,
                    }
                    df = pd.DataFrame([row])
                    header = not csv_path.exists()
                    df.to_csv(csv_path, mode='a', header=header, index=False)
                except Exception as e:
                    print(f"[Ditto] Failed to write convergence.csv: {e}")
        
        self.evaluate_holdout()
        # Export final per-client test metrics
        try:
            self.root_model.eval()
            rows = []
            if hasattr(self, 'test_loaders') and self.test_loaders:
                for cid, loader in self.test_loaders.items():
                    total_loss = 0.0
                    total_correct = 0
                    total_samples = 0
                    with torch.no_grad():
                        for data, target in loader:
                            data, target = data.to(self.device), target.to(self.device)
                            if self.args.model_name == 'cnn' and len(data.shape) == 2:
                                side = int(data.shape[1] ** 0.5)
                                if side * side == data.shape[1]:
                                    data = data.view(-1, 1, side, side)
                                elif data.shape[1] == 3072:
                                    data = data.view(-1, 3, 32, 32)
                                elif data.shape[1] == 784:
                                    data = data.view(-1, 1, 28, 28)
                            logits = self.root_model(data)
                            # Use cross-entropy on raw logits for consistency with training
                            loss = F.cross_entropy(logits, target, reduction='sum')
                            total_loss += loss.item()
                            total_correct += (logits.argmax(dim=1) == target).sum().item()
                            total_samples += data.size(0)
                    if total_samples > 0:
                        rows.append({
                            'client_id': cid,
                            'test_loss': total_loss / total_samples,
                            'test_acc': total_correct / total_samples,
                            'num_samples': total_samples,
                        })
            if rows:
                df = pd.DataFrame(rows)
                out_dir = Path('save/ditto')
                out_dir.mkdir(parents=True, exist_ok=True)
                csv_path = out_dir / "final.csv"
                df.to_csv(csv_path, index=False)
                print(f"Saved Ditto final per-client metrics to {csv_path}")
        except Exception as e:
            print(f"[Ditto] Failed to save final per-client metrics: {e}")
        wandb.finish()

class DittoWrapper:
    def __init__(self, args, cfg=None):
        self.args = args
        self.cfg = cfg
        
    def run(self):
        ditto = Ditto(self.args, self.cfg)
        ditto.train()
