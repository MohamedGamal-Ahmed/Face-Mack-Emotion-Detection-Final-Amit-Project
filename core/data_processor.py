import os
import hashlib
from PIL import Image
from pathlib import Path
import numpy as np

class DataProcessor:
    """
    AMIT Steps 1-4: Data Processing Module
    Responsibilities:
    Step 1: Load and Explore Dataset
    Step 2: Check for and Handle Duplicates (SHA256)
    Step 3: Handle Missing Values / Corrupted Files
    Step 4: Check Outliers (BBox sizes)
    """

    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.images_path = self.dataset_path / 'images'
        self.labels_path = self.dataset_path / 'labels'
        self.status = {"Step 1": "Pending", "Step 2": "Pending", "Step 3": "Pending", "Step 4": "Pending"}

    def scan_dataset(self):
        """Step 1: Load data from sources"""
        data = {'train': [], 'val': [], 'test': []}
        for split in data.keys():
            split_path = self.images_path / split
            if split_path.exists():
                data[split] = list(split_path.glob('*.*'))
        self.status["Step 1"] = "Complete"
        return data

    def check_duplicates(self, files):
        """Step 2: Detect duplicates using SHA256"""
        hashes = {}
        duplicates = []
        for f in files:
            with open(f, 'rb') as img_file:
                file_hash = hashlib.sha256(img_file.read()).hexdigest()
                if file_hash in hashes:
                    duplicates.append((f.name, hashes[file_hash]))
                else:
                    hashes[file_hash] = f.name
        self.status["Step 2"] = "Complete"
        return duplicates

    def handle_corrupted_missing(self, files):
        """Step 3: Detect missing labels or corrupted images"""
        report = {"corrupted": [], "missing_labels": []}
        for f in files:
            # Check Image Integrity
            try:
                with Image.open(f) as img:
                    img.verify()
            except Exception:
                report["corrupted"].append(f.name)
                continue

            # Check for Label
            label_file = self.labels_path / f.parent.name / f.with_suffix('.txt').name
            if not label_file.exists():
                report["missing_labels"].append(f.name)
        
        self.status["Step 3"] = "Complete"
        return report

    def check_outliers(self, labels_files):
        """Step 4: Check for bbox size outliers (too small/large)"""
        sizes = []
        for lf in labels_files:
            if not lf.exists(): continue
            with open(lf, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        w, h = float(parts[3]), float(parts[4])
                        sizes.append(w * h)
        
        if not sizes: return []
        
        q1, q3 = np.percentile(sizes, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = [s for s in sizes if s < lower_bound or s > upper_bound]
        self.status["Step 4"] = "Complete"
        return outliers
