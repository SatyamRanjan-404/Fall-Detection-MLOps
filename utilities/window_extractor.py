# window_extractor.py
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

class WindowExtractor:
    def __init__(self, window_size=50, fall_threshold=0.4):
        """Initializes the base rules for all extractions."""
        self.window_size = window_size
        self.fall_threshold = fall_threshold
        # The 10 columns we need (SMV will be calculated dynamically)
        self.target_cols = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ', 'EulerX', 'EulerY', 'EulerZ', 'SMV']

    def _process_file(self, file_path):
        """Internal helper: Loads CSV, checks size, and calculates SMV 10th channel."""
        df = pd.read_csv(file_path)
        if len(df) < self.window_size:
            return None, None
            
        df['SMV'] = np.sqrt(df['AccX']**2 + df['AccY']**2 + df['AccZ']**2)
        features = df[self.target_cols].values
        labels = df['FallCheck'].values
        return features, labels

    def _generate_summary(self, method_name, y_array):
        """Internal helper: Standardizes the logging output."""
        zeros, ones = np.sum(y_array == 0), np.sum(y_array == 1)
        ratio = zeros / ones if ones > 0 else 0
        return f"[{datetime.now().strftime('%H:%M:%S')}] Extractor: {method_name} | Total: {len(y_array)} | Normal: {zeros} | Falls: {ones} | Ratio: {ratio:.2f}:1"

    # ==========================================
    # THE EXTRACTION METHODS
    # ==========================================
    
    def extract_standard(self, file_list, stride=25):
        """Simulates real-world blind testing. Constant stride."""
        X_list, y_list = [], []
        for file in tqdm(file_list, desc="Extracting (Standard)"):
            features, labels = self._process_file(file)
            if features is None: continue
            
            start = 0
            while start + self.window_size <= len(features):
                window_y = labels[start : start + self.window_size]
                X_list.append(features[start : start + self.window_size])
                y_list.append(1 if np.mean(window_y) >= self.fall_threshold else 0)
                start += stride
                
        y_arr = np.array(y_list, dtype=np.float32)
        return np.array(X_list, dtype=np.float32), y_arr, self._generate_summary("Standard", y_arr)

    def extract_dynamic(self, file_list, normal_stride=25, fall_stride=5):
        """The Balancer Trick. Heavy overlap on falls, wide strides on walking."""
        X_list, y_list = [], []
        for file in tqdm(file_list, desc="Extracting (Dynamic)"):
            features, labels = self._process_file(file)
            if features is None: continue
            
            start = 0
            while start + self.window_size <= len(features):
                window_y = labels[start : start + self.window_size]
                is_fall = 1 if np.mean(window_y) >= self.fall_threshold else 0
                
                X_list.append(features[start : start + self.window_size])
                y_list.append(is_fall)
                start += fall_stride if is_fall == 1 else normal_stride
                
        y_arr = np.array(y_list, dtype=np.float32)
        return np.array(X_list, dtype=np.float32), y_arr, self._generate_summary("Dynamic", y_arr)

    def extract_transition(self, file_list, stride=5):
        """Dense extraction purely for pre-trimmed transition files."""
        X_list, y_list = [], []
        for file in tqdm(file_list, desc="Extracting (Transition)"):
            features, labels = self._process_file(file)
            if features is None: continue
            
            start = 0
            while start + self.window_size <= len(features):
                window_y = labels[start : start + self.window_size]
                X_list.append(features[start : start + self.window_size])
                y_list.append(1 if np.mean(window_y) >= self.fall_threshold else 0)
                start += stride # Constant dense stride
                
        y_arr = np.array(y_list, dtype=np.float32)
        return np.array(X_list, dtype=np.float32), y_arr, self._generate_summary("Transition", y_arr)
    
    def extract_strict_overlap(self, file_list, overlap_fraction=0.5):
        """
        Creates sliding windows with an exact percentage overlap (e.g., 0.5 = 50%).
        It naturally drops leftover tuples at the end of a file and never 
        crosses file boundaries, preventing anomalous transitions.
        """
        # Calculate the mathematical stride
        # Example: Window 50 * (1 - 0.5) = Stride 25
        stride = int(self.window_size * (1.0 - overlap_fraction))
        
        X_list, y_list = [], []
        
        for file in tqdm(file_list, desc=f"Extracting ({int(overlap_fraction*100)}% Overlap)"):
            # 1. Load the specific isolated file and calculate SMV
            features, labels = self._process_file(file)
            if features is None: continue
            
            start = 0
            # 2. The File Boundary Lock
            # If the file has 927 rows, the last valid 'start' is 875.
            # (875 + 50 = 925). The next start would be 900 (900 + 50 = 950),
            # which fails the loop condition. The remaining 2 rows are safely ignored.
            while start + self.window_size <= len(features):
                
                # Extract exactly 50 rows
                window_y = labels[start : start + self.window_size]
                X_list.append(features[start : start + self.window_size])
                
                # Label it (1 if fall, 0 if normal)
                y_list.append(1 if np.mean(window_y) >= self.fall_threshold else 0)
                
                # Move forward by exactly 50% of the window
                start += stride 
                
        y_arr = np.array(y_list, dtype=np.float32)
        
        # Return the 3D X tensor, 1D y tensor, and the telemetry log
        return np.array(X_list, dtype=np.float32), y_arr, self._generate_summary(f"Overlap {int(overlap_fraction*100)}%", y_arr)