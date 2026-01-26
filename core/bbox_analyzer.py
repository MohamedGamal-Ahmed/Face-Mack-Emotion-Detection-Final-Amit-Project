import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import xml.etree.ElementTree as ET

def analyze_bbox_outliers(xml_dir, output_path):
    """Analyze BBox sizes and detect outliers (AMIT Step 4)"""
    areas = []
    for xml_file in Path(xml_dir).glob('*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            w = int(bbox.find('xmax').text) - int(bbox.find('xmin').text)
            h = int(bbox.find('ymax').text) - int(bbox.find('ymin').text)
            areas.append(w * h)
            
    if not areas: return
    
    plt.figure(figsize=(10, 6))
    sns.histplot(areas, kde=True, color='#e63946')
    plt.title('BBox Area Distribution (Step 4: Outlier Detection)')
    plt.xlabel('Area (px^2)')
    
    # Calculate Z-score based outliers
    mean = np.mean(areas)
    std = np.std(areas)
    threshold = 3
    outliers = [a for a in areas if abs(a - mean) > threshold * std]
    
    plt.axvline(mean + threshold*std, color='blue', linestyle='--', label='Outlier Threshold')
    plt.legend()
    plt.savefig(output_path)
    plt.close()
    return len(outliers)
