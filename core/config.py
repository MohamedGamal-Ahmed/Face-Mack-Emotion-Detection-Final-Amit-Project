import os

# Base Directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
MODELS_DIR = os.path.join(BASE_DIR, 'model')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')

# User Dataset Paths
DATASET_ROOT = r'd:\M.Gamal\Learn\Projects\final project\images\Mask vs No-Mask'
MASK_XML_DIR = os.path.join(DATASET_ROOT, 'annotations')
EMOTION_ROOT = r'd:\M.Gamal\Learn\Projects\final project\images' # Emotions are subfolders here

# Model Weights
MODEL_PATH = os.path.join(MODELS_DIR, 'best.pt')

# Asset Mapping (Step 8 Results)
RESULTS_CSV = os.path.join(STATIC_DIR, 'results.csv')
CONFUSION_MATRIX = 'confusion_matrix.png'
F1_CURVE = 'F1_curve.png'
PR_CURVE = 'PR_curve.png'
RESULTS_PNG = 'results.png'

# Classes
CLASS_NAMES = ['mask', 'no mask']

# Camera Settings (0 for Laptop, 1 for Iriun/External)
CAMERA_INDEX = 0 
