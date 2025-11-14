from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = '!bandar-bhalu'
model = YOLO("models/yolov8m_best")

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv'}
MAX_FILE_SIZE = 500 * 1024 * 1024  

#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


def process_video(file_path,filename):
    cap = cv2.VideoCapture(file_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_video_path = f"outputs/{filename}_out.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    frame_count=0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1

        try:
            results = model(frame, verbose=False)
            annotated_frame = results[0].plot()
            
            num_detections = len(results[0].boxes)
            detection_count += num_detections
            
            out.write(annotated_frame)
        except Exception as e:
            print(f" Warning: Error processing frame {frame_count}: {e}")
            continue
    cap.release()
    
            
        
    

@app.route('/')
def index():
    """Render the upload page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    try:    
        if 'video' not in request.files:
            return jsonify({'error':'No file provided'}), 400
        
        file = request.files['video']
        
        filename = secure_filename(file.filename)
        time = datetime.now().strftime('%Y%M%D_%H%M%S')
        filename = filename + '_' + time
        filepath = f"outputs\{filename}"
        
        file.save(filepath)
        return jsonify({
            'filename': filename,
            'message': 'Video uploaded Sucessfully',
            'success':True
        }), 200
    except Exception as e:
        return jsonify({'error', f'Upload Failed: {str(e)}'}),500
        
    


@app.route('/process/<filename>', methods=['POST'])
def process_video(filename):
    process_video()
    


@app.route('/results/<filename>')
def show_results(filename):
    return render_template('results.html', filename=filename)


@app.route('/delete/<filename>', methods=['DELETE'])
def delete_video(filename):
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({
                'success': True,
                'message': 'File deleted successfully'
            }), 200
        else:
            return jsonify({'error': 'File not found'}), 404
            
    except Exception as e:
        return jsonify({'error': f'Delete failed: {str(e)}'}), 500


# ========================================
# ERROR HANDLERS
# ========================================

@app.errorhandler(413)
def too_large(e):
    """Handle file too large errors"""
    return jsonify({'error': 'File too large. Maximum size is 500MB'}), 413


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    return jsonify({'error': 'Internal server error occurred'}), 500


# ========================================
# RUN APPLICATION
# ========================================

if __name__ == '__main__':
    print("="*50)
    print("Cheating Detection System Starting...")
    print("="*50)
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}")
    print(f"Max file size: {MAX_FILE_SIZE // (1024*1024)}MB")
    print("="*50)
    print("Server running at: http://localhost:5000")
    print("="*50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)