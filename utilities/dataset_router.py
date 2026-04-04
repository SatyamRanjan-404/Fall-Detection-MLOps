# dataset_router.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class DatasetRouter:
    def __init__(self, dataset_path, test_size=0.2, random_state=42):
        """Initializes the router with strict reproducibility rules."""
        self.dataset_path = Path(dataset_path)
        self.test_size = test_size
        self.random_state = random_state
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Error: Dataset path '{self.dataset_path}' does not exist.")

    def _tag_file(self, file_path):
        """Internal Helper: Ultra-fast scan of a single file to detect falls."""
        try:
            # We ONLY load the FallCheck column to keep RAM usage near zero
            df = pd.read_csv(file_path, usecols=['FallCheck'])
            return 1 if df['FallCheck'].sum() > 0 else 0
        except Exception as e:
            print(f"Warning: Corrupted file skipped -> {file_path}")
            return None

    def create_splits(self):
        """Scans, tags, and performs the leak-proof stratified split."""
        all_csv_files = list(self.dataset_path.rglob("*.csv"))
        print(f"\n--- Initializing File Router on {self.dataset_path} ---")
        print(f"Found {len(all_csv_files)} total CSV files. Tagging...")

        file_paths = []
        file_labels = []

        # Tag every file
        for file in tqdm(all_csv_files, desc="Routing Files"):
            label = self._tag_file(file)
            if label is not None:
                file_paths.append(str(file))
                file_labels.append(label)

        # The Leak-Proof Stratified Split
        train_files, val_files, train_labels, val_labels = train_test_split(
            file_paths, 
            file_labels, 
            test_size=self.test_size, 
            stratify=file_labels, 
            random_state=self.random_state
        )

        print("\n=== ROUTING SUCCESSFUL ===")
        print(f"Train Files: {len(train_files)} (Falls: {sum(train_labels)})")
        print(f"Val Files:   {len(val_files)} (Falls: {sum(val_labels)})")
        print("==========================")

        return train_files, val_files