from flask import Flask, render_template, Response, request, jsonify
import os
import cv2
import time
from core.config import *
from core.data_analyzer import DataAnalyzer
from engine.inference import InferenceEngine

app = Flask(__name__)

# Verify Directory Infrastructure
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, 'captures'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'uploads'), exist_ok=True)

# AMIT SCIENTIFIC PIPELINE (Steps 2-6)
print("SYSTEM RECONCILIATION: Running AMIT Scientific Pipeline...")
try:
    # DATASET_ROOT is now correctly pointed in config.py to fix 98% missing labels
    science = DataAnalyzer(
        images_dir=DATASET_ROOT, 
        xml_dir=MASK_XML_DIR, 
        emotion_root=EMOTION_ROOT,
        output_dir=STATIC_DIR
    )
    # Generate Academic Reports
    print("-> Step 3: Integrity Analysis")
    science.check_data_integrity()
    print("-> Step 4: Outlier Detection")
    science.analyze_bbox_outliers()
    print("-> Step 5: Global Class Distribution")
    science.plot_class_distribution()
    print("-> Step 6: Spatial Concentration Heatmap")
    science.generate_heatmap()
    print("-> Step 6: Augmentation Before/After")
    science.plot_augmentation_comparison()
    print("AMIT Pipeline: Scientific Reports Generated Successfully.")
except Exception as e:
    print(f"PIPELINE CRITICAL ERROR: {e}")

# Load Deep Inference Node
engine = InferenceEngine(MODEL_PATH, camera_index=CAMERA_INDEX)

@app.route('/')
def index():
    return render_template('landing.html', 
                         student="Mohamed Gamal", 
                         mentor="Dr. George Samuel")

@app.route('/video_feed')
def video_feed():
    return Response(engine.generate_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    """Captures and archives the current analytical frame"""
    success, frame = engine.get_latest_frame()
    if success:
        filename = engine.capture_snapshot(frame)
        return jsonify({"success": True, "filepath": f"/static/captures/{filename}"})
    return jsonify({"success": False, "error": "Inference node not providing frames"})

@app.route('/upload', methods=['POST'])
def upload():
    """AJAX Scientific Inference for uploaded files"""
    if 'file' not in request.files: 
        return jsonify({"success": False, "error": "No data stream provided"})
    
    file = request.files['file']
    if not file: return jsonify({"success": False, "error": "Empty reference"})
    
    # Secure and Save
    upload_path = os.path.join(BASE_DIR, 'uploads', file.filename)
    file.save(upload_path)
    
    # Process Analysis
    img = cv2.imread(upload_path)
    if img is None: return jsonify({"success": False, "error": "Incompatible data format"})
    
    result = engine.predict_frame(img)
    
    # Archive to static for preview
    res_name = f"analysis_{int(time.time())}.jpg"
    res_path = os.path.join(STATIC_DIR, res_name)
    cv2.imwrite(res_path, result)
    
    return jsonify({"success": True, "result_url": f"/static/{res_name}"})

if __name__ == '__main__':
    # Running on standard 5000 for Iriun/Remote accessibility
    app.run(debug=True, port=5000)
