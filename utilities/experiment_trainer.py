# experiment_trainer.py
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# Internal PyTorch Wrapper
class FallDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        # Transpose transforms (SeqLen, Channels) to (Channels, SeqLen) for PyTorch 1D CNNs
        return torch.tensor(self.X[idx].T, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    """
    Focal Loss for Binary Classification.
    Addresses class imbalance and down-weights 'easy' examples.
    """
    def __init__(self, alpha=0.80, gamma=2.0, reduction='mean'):
        super().__init__()
        # Alpha balances the Positive (Fall) vs Negative (Normal) class.
        # Since falls are rare, alpha should be high (e.g., 0.75 - 0.90)
        self.alpha = alpha 
        
        # Gamma is the "Sniper" parameter. 
        # Higher gamma (2.0 to 5.0) aggressively ignores easy examples.
        self.gamma = gamma
        
        self.reduction = reduction

    def forward(self, logits, targets):
        # 1. Get raw probabilities
        probs = torch.sigmoid(logits)
        
        # 2. Calculate p_t (The probability of the TRUE class)
        # If target is 1, p_t = probs. If target is 0, p_t = 1 - probs.
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # 3. Calculate standard BCE Loss (without reduction yet)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # 4. Apply the Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # 5. THE FOCAL EQUATION: Apply the Gamma focusing parameter
        focal_loss = alpha_t * ((1 - p_t) ** self.gamma) * bce_loss
        
        # 6. Reduce the loss back to a single number for the optimizer
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class ExperimentTrainer:
    def __init__(self, exp_name, description, model, criterion, optimizer, device='cuda'):
        """Initializes the training environment and logging footprint."""
        self.exp_name = exp_name
        self.description = description
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Ensure telemetry folders exist
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

    def _prepare_loaders(self, X_train, y_train, X_val, y_val, batch_size):
        """Invisible bridge: NumPy Arrays -> PyTorch DataLoaders"""
        train_loader = DataLoader(FallDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(FallDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

    def train(self, X_train, y_train, X_val, y_val, epochs=30, batch_size=128):
        """The Master Training Loop."""
        print(f"\n=== LAUNCHING EXPERIMENT: {self.exp_name} ===")
        print(f"Device: {self.device} | Epochs: {epochs} | Batch: {batch_size}")
        
        train_loader, val_loader = self._prepare_loaders(X_train, y_train, X_val, y_val, batch_size)
        
        history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_rec': []}
        best_f1 = 0.0
        
        for epoch in range(epochs):
            # --- Training Phase ---
            self.model.train()
            train_loss = 0.0
            for b_X, b_y in train_loader:
                b_X, b_y = b_X.to(self.device), b_y.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(b_X), b_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                
            # --- Validation Phase ---
            self.model.eval()
            val_loss = 0.0
            all_preds, all_targets = [], []
            with torch.no_grad():
                for v_X, v_y in val_loader:
                    v_X, v_y = v_X.to(self.device), v_y.to(self.device)
                    logits = self.model(v_X)
                    val_loss += self.criterion(logits, v_y).item()
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(v_y.cpu().numpy())
                    
            # --- Metrics & Telemetry ---
            t_loss = train_loss / len(train_loader)
            v_loss = val_loss / len(val_loader)
            v_f1 = f1_score(all_targets, all_preds)
            v_rec = recall_score(all_targets, all_preds, zero_division=0)
            
            history['train_loss'].append(t_loss)
            history['val_loss'].append(v_loss)
            history['val_f1'].append(v_f1)
            history['val_rec'].append(v_rec)
            
            print(f"Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Val F1: {v_f1:.4f} | Recall: {v_rec:.4f}")
            
            # Save Checkpoint
            if v_f1 > best_f1:
                best_f1 = v_f1
                torch.save(self.model.state_dict(), f"models/{self.exp_name}_best.pth")
                
        print(f"\n[Training Complete] Best weights saved to models/{self.exp_name}_best.pth")
        
        # Load best weights back into model before returning
        self.model.load_state_dict(torch.load(f"models/{self.exp_name}_best.pth"))
        
        # Generate Telemetry Assets
        self._generate_report(history, best_f1, all_targets, all_preds)
        
        return self.model, history

    def _generate_report(self, history, best_f1, final_targets, final_preds):
        """Internal helper: Draws charts and writes the text file."""
        epochs = len(history['train_loss'])
        
        # 1. Plotting
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs+1), history['train_loss'], label='Train Loss', color='blue')
        plt.plot(range(1, epochs+1), history['val_loss'], label='Val Loss', color='red')
        plt.title('Loss Curve')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs+1), history['val_f1'], label='Val F1', color='purple')
        plt.plot(range(1, epochs+1), history['val_rec'], label='Val Recall', color='green')
        plt.title('Safety Metrics')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"logs/{self.exp_name}_curve.png")
        plt.close()
        
        # 2. Text Report
        cm = confusion_matrix(final_targets, final_preds)
        with open(f"logs/{self.exp_name}_report.txt", "w") as f:
            f.write(f"=== EXPERIMENT: {self.exp_name} ===\n")
            f.write(f"DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"DESCRIPTION: {self.description}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Best Validation F1: {best_f1:.4f}\n")
            f.write(f"Final Validation Recall: {history['val_rec'][-1]:.4f}\n")
            f.write("-" * 40 + "\n")
            f.write("Final Epoch Confusion Matrix:\n")
            f.write(f"TN: {cm[0][0]} | FP: {cm[0][1]}\n")
            f.write(f"FN: {cm[1][0]} | TP: {cm[1][1]}\n")