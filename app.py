from flask import Flask, render_template, request, jsonify, send_file
import os
import cv2
import numpy as np
from depth_estimator import DepthEstimator
from mesh_generator import MeshGenerator

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize components
depth_est = DepthEstimator()
mesh_gen = MeshGenerator()

# Create upload directory
os.makedirs('uploads', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Save uploaded image
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.jpg')
        file.save(img_path)
        
        # Load and preprocess image
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Generate depth map
        depth_map = depth_est.estimate_depth(image_rgb)
        
        # Create 3D mesh
        mesh_path = mesh_gen.create_mesh(image_rgb, depth_map)
        
        return jsonify({
            'success': True,
            'mesh_file': mesh_path
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(f'uploads/{filename}', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)