# experiment_tester.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime

# We need the simple Dataset wrapper again for the dataloader
class FallDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx].T, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

class ExperimentTester:
    def __init__(self, exp_name, model_architecture, device='cuda'):
        """
        Initializes the tester by locating the saved experiment weights 
        and loading them into the provided empty model architecture.
        """
        self.exp_name = exp_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model_architecture.to(self.device)
        
        # 1. Locate and load the physical weight file
        weight_path = f"models/{self.exp_name}_best.pth"
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"CRITICAL ERROR: Cannot find saved weights at {weight_path}")
            
        print(f"Loading locked weights from: {weight_path}")
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        
        # 2. Lock the model into evaluation mode (disables Dropout/BatchNorm updates)
        self.model.eval()

    def run_blind_test(self, X_test, y_test, test_name="Final Blind Test", extraction_method="Unknown", batch_size=128):
        """Executes the test and appends the autopsy report to the text log."""
        print(f"\n=== INITIATING BLIND TEST: {test_name} ===")
        
        test_loader = DataLoader(FallDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
        all_preds, all_targets = [], []
        
        # 3. Pure Inference (No Gradients)
        with torch.no_grad():
            for b_X, b_y in test_loader:
                b_X, b_y = b_X.to(self.device), b_y.to(self.device)
                logits = self.model(b_X)
                preds = (torch.sigmoid(logits) > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(b_y.cpu().numpy())
                
        # 4. Calculate Metrics
        report = classification_report(all_targets, all_preds, target_names=['Normal (0)', 'Fall (1)'])
        cm = confusion_matrix(all_targets, all_preds)
        
        print("\n" + report)
        print("Confusion Matrix:")
        print(f"TN: {cm[0][0]} | FP: {cm[0][1]}")
        print(f"FN: {cm[1][0]} | TP: {cm[1][1]}")
        
        # 5. Append to the Text Ledger
        log_path = f"logs/{self.exp_name}_report.txt"
        
        with open(log_path, "a") as f:
            f.write("\n" + "="*40 + "\n")
            f.write(f"BLIND TEST EXECUTION: {test_name}\n")
            f.write(f"DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"TEST EXTRACTION METHOD: {extraction_method}\n")
            f.write("-" * 40 + "\n")
            f.write(report + "\n")
            f.write("Confusion Matrix:\n")
            f.write(f"TN: {cm[0][0]} | FP: {cm[0][1]}\n")
            f.write(f"FN: {cm[1][0]} | TP: {cm[1][1]}\n")
            f.write("="*40 + "\n")
            
        print(f"\n[Success] Test results securely appended to {log_path}")
        
        return all_preds, all_targets