from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import numpy as np
from ultralytics import YOLO
import json
from ultralytics import YOLO
import cv2
import os
import time
from collections import defaultdict, OrderedDict
import numpy as np
import subprocess

app = Flask(__name__,
            static_folder='static',
            static_url_path='/static')
app.secret_key = '!bandar-bhalu'
model = YOLO("models/yolov8m_best.pt")

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/outputs'
SUMMARY_FOLDER = 'summary'
TEMP_FOLDER = 'temp_output'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv'}
MAX_FILE_SIZE = 500 * 1024 * 1024  

#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


# ===================================================== Model Predict ==========================================

# ============================================================
# IMPROVED TRACKING CLASSES
# ============================================================

class PersonTracker:
    """
    Improved person tracker with centroid tracking and ID persistence
    """
    
    def __init__(self, max_disappeared=60, max_distance=200, min_detection_confidence=0.6):
        """
        Initialize tracker
        
        Args:
            max_disappeared: Maximum frames a person can be missing before removing
            max_distance: Maximum distance (pixels) to consider same person
            min_detection_confidence: Minimum confidence for person detection
        """
        self.next_id = 1
        self.persons = OrderedDict()  # {person_id: bbox}
        self.disappeared = OrderedDict()  # {person_id: frame_count}
        self.centroids = OrderedDict()  # {person_id: (cx, cy)}
        self.first_seen = OrderedDict()  # {person_id: frame_number} - track when first seen
        
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.min_detection_confidence = min_detection_confidence
    
    def register(self, bbox, frame_number):
        """
        Register a new person
        
        Args:
            bbox: [x1, y1, x2, y2]
            frame_number: Current frame number
        
        Returns:
            person_id: Assigned ID
        """
        person_id = self.next_id
        self.persons[person_id] = bbox
        self.disappeared[person_id] = 0
        self.centroids[person_id] = self._get_centroid(bbox)
        self.first_seen[person_id] = frame_number
        self.next_id += 1
        return person_id
    
    def deregister(self, person_id):
        """Remove a person from tracking"""
        if person_id in self.persons:
            del self.persons[person_id]
            del self.disappeared[person_id]
            del self.centroids[person_id]
            del self.first_seen[person_id]
    
    def _get_centroid(self, bbox):
        """
        Calculate centroid of bounding box
        
        Args:
            bbox: [x1, y1, x2, y2]
        
        Returns:
            (cx, cy): Center point
        """
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        return (cx, cy)
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_distance(self, centroid1, centroid2):
        """
        Calculate Euclidean distance between two centroids
        
        Args:
            centroid1: (x1, y1)
            centroid2: (x2, y2)
        
        Returns:
            float: Distance in pixels
        """
        return np.sqrt((centroid1[0] - centroid2[0])**2 + 
                      (centroid1[1] - centroid2[1])**2)
    
    def update(self, input_bboxes, frame_number):
        """
        Update tracker with new detections
        
        Args:
            input_bboxes: List of detected bounding boxes [[x1,y1,x2,y2], ...]
            frame_number: Current frame number
        
        Returns:
            persons: OrderedDict of {person_id: bbox}
        """
        # If no detections, mark all as disappeared
        if len(input_bboxes) == 0:
            for person_id in list(self.disappeared.keys()):
                self.disappeared[person_id] += 1
                
                # Remove if disappeared too long
                if self.disappeared[person_id] > self.max_disappeared:
                    self.deregister(person_id)
            
            return self.persons
        
        # Calculate centroids for input boxes
        input_centroids = [self._get_centroid(bbox) for bbox in input_bboxes]
        
        # If no existing persons, register all as new
        if len(self.persons) == 0:
            for bbox in input_bboxes:
                self.register(bbox, frame_number)
        else:
            # Match existing persons with new detections
            person_ids = list(self.persons.keys())
            existing_centroids = [self.centroids[pid] for pid in person_ids]
            existing_bboxes = [self.persons[pid] for pid in person_ids]
            
            # Calculate cost matrix using both distance and IoU
            D = np.zeros((len(existing_centroids), len(input_centroids)))
            
            for i, (ex_centroid, ex_bbox) in enumerate(zip(existing_centroids, existing_bboxes)):
                for j, (in_centroid, in_bbox) in enumerate(zip(input_centroids, input_bboxes)):
                    # Centroid distance
                    dist = self._calculate_distance(ex_centroid, in_centroid)
                    
                    # IoU similarity
                    iou = self._calculate_iou(ex_bbox, in_bbox)
                    
                    # Combined cost: 
                    # Give MORE weight to IoU to make matching stricter
                    # Weight: 40% distance, 60% IoU
                    cost = (0.4 * dist) + (0.6 * (1 - iou) * 200)
                    
                    D[i, j] = cost
            
            # Find best matches using greedy approach
            used_rows = set()
            used_cols = set()
            matches = []
            
            # Greedy matching: assign each detection to closest existing person
            for _ in range(min(len(person_ids), len(input_bboxes))):
                # Find minimum cost
                min_cost = np.inf
                min_row = -1
                min_col = -1
                
                for i in range(len(person_ids)):
                    if i in used_rows:
                        continue
                    for j in range(len(input_bboxes)):
                        if j in used_cols:
                            continue
                        if D[i, j] < min_cost:
                            min_cost = D[i, j]
                            min_row = i
                            min_col = j
                
                # Stricter threshold - only match if cost is low
                if min_cost < self.max_distance:
                    matches.append((min_row, min_col))
                    used_rows.add(min_row)
                    used_cols.add(min_col)
                else:
                    break
            
            # Update matched persons
            for row, col in matches:
                person_id = person_ids[row]
                self.persons[person_id] = input_bboxes[col]
                self.centroids[person_id] = input_centroids[col]
                self.disappeared[person_id] = 0
            
            # Mark unmatched existing persons as disappeared
            unmatched_persons = set(range(len(person_ids))) - used_rows
            for i in unmatched_persons:
                person_id = person_ids[i]
                self.disappeared[person_id] += 1
                
                if self.disappeared[person_id] > self.max_disappeared:
                    self.deregister(person_id)
            
            # Only register new persons if they're not near existing ones
            # This prevents creating duplicate IDs for the same person
            unmatched_inputs = set(range(len(input_bboxes))) - used_cols
            for j in unmatched_inputs:
                new_bbox = input_bboxes[j]
                new_centroid = input_centroids[j]
                
                # Check if this detection is too close to any existing person
                too_close = False
                for person_id in self.persons.keys():
                    dist = self._calculate_distance(new_centroid, self.centroids[person_id])
                    if dist < self.max_distance * 0.5:  # Half the max distance
                        too_close = True
                        break
                
                # Only register if not too close to existing persons
                if not too_close:
                    self.register(new_bbox, frame_number)
        
        return self.persons


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def convert_video_for_web(input_path, output_path):
    """Convert video to web-compatible format using ffmpeg"""
    try:
        # Check if ffmpeg is available
        subprocess.run(['ffmpeg', '-version'], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL, 
                      check=True)
        
        # Convert video to web-compatible format
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-c:v', 'libx264',           # H.264 video codec
            '-preset', 'medium',          # Encoding speed
            '-crf', '23',                 # Quality (lower = better, 23 is good)
            '-c:a', 'aac',                # AAC audio codec
            '-b:a', '128k',               # Audio bitrate
            '-movflags', '+faststart',    # Enable streaming
            '-y',                         # Overwrite output file
            output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✓ Video converted for web: {output_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg conversion failed: {e}")
        return False
    except FileNotFoundError:
        print("⚠️ FFmpeg not found. Install it for web-compatible videos.")
        return False

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0.0


def associate_parts_with_persons(person_bboxes_dict, part_detections, iou_threshold=0.1):
    """
    Associate body parts with persons
    
    Args:
        person_bboxes_dict: OrderedDict of {person_id: bbox}
        part_detections: List of detection dicts
        iou_threshold: Minimum IoU for association
    
    Returns:
        associations: {person_id: [detections]}
    """
    associations = defaultdict(list)
    
    for part in part_detections:
        part_bbox = part['bbox']
        best_iou = 0
        best_person_id = None
        
        for person_id, person_bbox in person_bboxes_dict.items():
            iou = calculate_iou(part_bbox, person_bbox)
            
            if iou > best_iou and iou > iou_threshold:
                best_iou = iou
                best_person_id = person_id
        
        if best_person_id is not None:
            associations[best_person_id].append(part)
    
    return associations


def make_cheating_decision(detection_history, total_frames):
    """Decide if person is cheating based on detection history"""
    if not detection_history:
        return "UNKNOWN", "No detections", 0.0
    
    detection_counts = defaultdict(int)
    for detection in detection_history:
        detection_counts[detection['class']] += 1
    
    total_detections = len(detection_history)
    
    # RULE 1: Cheat sheets = DEFINITE CHEATING
    if detection_counts['Cheatchits'] > 0:
        confidence = min(detection_counts['Cheatchits'] / total_frames * 100, 100)
        return "CHEATING", "Cheat sheets detected", confidence
    
    # RULE 2: Watching others
    watching_count = (detection_counts['backwatching'] + 
                     detection_counts['sidewatching'] + 
                     detection_counts['frontwatching'])
    watching_percentage = (watching_count / total_frames) * 100
    
    if watching_percentage > 30:
        confidence = min(watching_percentage, 100)
        return "CHEATING", f"Watching others {watching_percentage:.1f}% of time", confidence
    
    # RULE 3: Suspicious movements
    suspicious_count = (detection_counts['SuspicousHandMovement'] + 
                       detection_counts['SuspicousHeaddMovement'])
    suspicious_percentage = (suspicious_count / total_frames) * 100
    
    if suspicious_percentage > 50:
        confidence = min(suspicious_percentage, 100)
        return "CHEATING", f"Suspicious behavior {suspicious_percentage:.1f}% of time", confidence
    
    # RULE 4: Normal behavior
    normal_count = (detection_counts['NormalHand'] + 
                   detection_counts['NormalHead'])
    normal_percentage = (normal_count / total_detections) * 100
    
    if normal_percentage > 70:
        confidence = min(normal_percentage, 100)
        return "NOT CHEATING", "Mostly normal behavior", confidence
    
    # RULE 5: Mixed behavior
    confidence = 50.0
    return "SUSPICIOUS", "Mixed behavior patterns", confidence


def get_bright_color(person_id):
    """
    Get a bright, distinct color for each person ID
    
    Args:
        person_id: Person ID number
    
    Returns:
        (B, G, R) tuple - OpenCV color format
    """
    # Bright, vibrant colors (BGR format)
    colors = [
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
        (0, 165, 255),    # Orange
        (255, 255, 0),    # Yellow
        (255, 0, 127),    # Deep Pink
        (0, 255, 0),      # Lime
        (147, 20, 255),   # Deep Pink
        (255, 191, 0),    # Deep Sky Blue
        (203, 192, 255),  # Pink
        (92, 92, 205),    # Indian Red
    ]
    
    # Cycle through colors if more persons than colors
    return colors[(person_id - 1) % len(colors)]


# ============================================================
# MAIN PROCESSING FUNCTION WITH IMPROVED TRACKING
# ============================================================

def process_video_with_stable_tracking(input_video_path, output_video_path, 
                                       behavior_model_path, person_model_path=None,
                                       person_conf_threshold=0.7):
    """
    Process video with stable person tracking
    
    Args:
        input_video_path: Path to input video
        output_video_path: Path to output video
        behavior_model_path: Path to behavior detection model
        person_model_path: Path to person detection model (optional)
        person_conf_threshold: Confidence threshold for person detection (higher = fewer false positives)
    """
    
    print("="*70)
    print("VIDEO PROCESSING WITH STABLE PERSON TRACKING")
    print("="*70)
    
    # Load models
    print("\n[1/8] Loading models...")
    behavior_model = YOLO(behavior_model_path)
    print(f"✓ Behavior model: {list(behavior_model.names.values())}")
    
    if person_model_path is None:
        person_model = YOLO('yolov8n.pt')
        print("✓ Person detector: YOLOv8n (pretrained)")
    else:
        person_model = YOLO(person_model_path)
        print(f"✓ Person detector: Custom model")
    
    # Open video
    print("\n[2/8] Opening video...")
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print("❌ Error: Could not open video")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✓ Video: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")
    
    # Setup output
    print("\n[3/8] Setting up output...")
    output_dir = os.path.dirname(output_video_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    print(f"✓ Output: {output_video_path}")
    
    # Initialize improved tracker
    print("\n[4/8] Initializing improved tracking system...")
    # Increased max_disappeared and max_distance for more stable tracking
    tracker = PersonTracker(max_disappeared=60, max_distance=200)
    
    person_histories = defaultdict(list)
    person_frame_counts = defaultdict(int)
    cheating_persons = set()  # Track IDs of people who are cheating
    
    print("✓ Tracking system initialized")
    print(f"  - Max disappeared frames: 60 (~2 seconds)")
    print(f"  - Max matching distance: 200 pixels")
    print(f"  - Person detection confidence: {person_conf_threshold}")
    
    # Process video
    print("\n[5/8] Processing video frames...")
    print("-"*70)
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        annotated_frame = frame.copy()
        
        # ===== STEP 1: Detect persons with higher confidence threshold =====
        person_results = person_model(frame, classes=[0], conf=person_conf_threshold, verbose=False)
        
        detected_bboxes = []
        for box in person_results[0].boxes:
            bbox = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            
            # Additional size filter: ignore very small detections (likely false positives)
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            # Ignore detections that are too small (adjust threshold based on your video)
            min_area = (frame_width * frame_height) * 0.01  # At least 1% of frame
            if area > min_area:
                detected_bboxes.append(bbox)
        
        # ===== STEP 2: Update tracker =====
        tracked_persons = tracker.update(detected_bboxes, frame_count)
        
        # Update frame counts for active persons
        for person_id in tracked_persons.keys():
            person_frame_counts[person_id] += 1
        
        # ===== STEP 3: Detect behaviors =====
        behavior_results = behavior_model(frame, verbose=False)
        
        part_detections = []
        for box in behavior_results[0].boxes:
            bbox = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0])
            class_name = behavior_model.names[class_id]
            confidence = float(box.conf[0])
            
            part_detections.append({
                'bbox': bbox,
                'class': class_name,
                'conf': confidence
            })
        
        # ===== STEP 4: Associate parts with persons =====
        associations = associate_parts_with_persons(tracked_persons, part_detections, iou_threshold=0.1)
        
        # ===== STEP 5: Update histories =====
        for person_id in associations:
            person_histories[person_id].extend(associations[person_id])
        
        # ===== STEP 6: Visualize =====
        
        # Draw person boxes with verdicts
        for person_id, person_bbox in tracked_persons.items():
            x1, y1, x2, y2 = map(int, person_bbox)
            
            verdict, reason, confidence = make_cheating_decision(
                person_histories[person_id],
                person_frame_counts[person_id]
            )
            
            # Track cheating persons
            if verdict == "CHEATING":
                cheating_persons.add(person_id)
            
            # Get bright color for this person
            person_color = get_bright_color(person_id)
            
            # Override color based on verdict (optional - comment out if you want consistent person colors)
            if verdict == "CHEATING":
                box_color = (0, 0, 255)  # Red for cheating
            elif verdict == "SUSPICIOUS":
                box_color = (0, 165, 255)  # Orange for suspicious
            else:
                box_color = person_color  # Bright unique color otherwise
            
            # Draw person box (thick and bright)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 4)
            
            # Draw label with verdict
            label = f"Person {person_id}: {verdict}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            
            # Draw label background (semi-transparent effect by drawing filled rectangle)
            cv2.rectangle(annotated_frame, 
                         (x1, y1 - label_size[1] - 20), 
                         (x1 + label_size[0] + 15, y1), 
                         box_color, -1)
            
            # Draw label text in white
            cv2.putText(annotated_frame, label, (x1 + 7, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Draw centroid (for debugging/tracking visualization)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.circle(annotated_frame, (cx, cy), 7, box_color, -1)
            cv2.circle(annotated_frame, (cx, cy), 9, (255, 255, 255), 2)
        
        # Draw behavior detections (smaller boxes inside person boxes)
        for person_id in associations:
            for detection in associations[person_id]:
                x1, y1, x2, y2 = map(int, detection['bbox'])
                class_name = detection['class']
                conf = detection['conf']
                
                # Draw detection box in bright yellow
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                
                # Draw detection label
                det_label = f"{class_name} {conf:.2f}"
                cv2.putText(annotated_frame, det_label, (x1, y1 - 7), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Draw frame info at top
        info_text = f"Frame: {frame_count}/{total_frames} | Active Persons: {len(tracked_persons)} | Total IDs: {tracker.next_id - 1}"
        cv2.rectangle(annotated_frame, (5, 5), (900, 45), (0, 0, 0), -1)
        cv2.putText(annotated_frame, info_text, (10, 32), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Write frame
        out.write(annotated_frame)
        
        # Progress
        if frame_count % 30 == 0 or frame_count == total_frames:
            elapsed = time.time() - start_time
            speed = frame_count / elapsed
            eta = (total_frames - frame_count) / speed if speed > 0 else 0
            
            print(f"Frame: {frame_count:5d}/{total_frames} ({frame_count/total_frames*100:5.1f}%) | "
                  f"Speed: {speed:5.1f} fps | ETA: {eta:6.1f}s | "
                  f"Active: {len(tracked_persons)} | Total IDs: {tracker.next_id - 1} | "
                  f"Cheating: {len(cheating_persons)}")
    
    # Cleanup
    cap.release()
    out.release()
    
    # Final verdicts
    print("\n" + "="*70)
    print("FINAL CHEATING VERDICTS")
    print("="*70)
    
    # Sort by person ID
    for person_id in sorted(person_histories.keys()):
        verdict, reason, confidence = make_cheating_decision(
            person_histories[person_id],
            person_frame_counts[person_id]
        )
        
        # Mark as cheating if detected
        if verdict == "CHEATING":
            cheating_persons.add(person_id)
        
        print(f"\nPerson {person_id}:")
        print(f"  Verdict: {verdict}")
        print(f"  Reason: {reason}")
        print(f"  Confidence: {confidence:.1f}%")
        print(f"  Frames present: {person_frame_counts[person_id]}/{total_frames} ({person_frame_counts[person_id]/total_frames*100:.1f}%)")
        print(f"  Total detections: {len(person_histories[person_id])}")
        
        # Detection breakdown
        detection_counts = defaultdict(int)
        for det in person_histories[person_id]:
            detection_counts[det['class']] += 1
        
        if detection_counts:
            print(f"  Detection breakdown:")
            for class_name, count in sorted(detection_counts.items(), 
                                           key=lambda x: x[1], reverse=True):
                percentage = (count / len(person_histories[person_id])) * 100
                print(f"    {class_name}: {count} ({percentage:.1f}%)")
    
    total_time = time.time() - start_time
    
    # Display cheating persons summary
    print("\n" + "="*70)
    print("CHEATING PERSONS DETECTED")
    print("="*70)
    
    if cheating_persons:
        print(f"\n⚠️  ALERT: {len(cheating_persons)} person(s) caught cheating!")
        print(f"\nCheating Person IDs: {sorted(cheating_persons)}")
        print("\nDetailed Cheating Report:")
        
        for person_id in sorted(cheating_persons):
            verdict, reason, confidence = make_cheating_decision(
                person_histories[person_id],
                person_frame_counts[person_id]
            )
            print(f"\n  Person {person_id}:")
            print(f"    - Reason: {reason}")
            print(f"    - Confidence: {confidence:.1f}%")
            print(f"    - Visible in: {person_frame_counts[person_id]} frames ({person_frame_counts[person_id]/total_frames*100:.1f}% of video)")
    else:
        print("\n✓ No cheating detected in this video!")
    
    print("\n" + "="*70)
    print("PROCESSING COMPLETE!")
    print("="*70)
    print(f"✓ Total frames processed: {frame_count}")
    print(f"✓ Total unique persons detected: {len(person_histories)}")
    print(f"✓ Total person IDs created: {tracker.next_id - 1}")
    print(f"✓ Persons caught cheating: {len(cheating_persons)}")
    print(f"✓ Processing time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"✓ Processing speed: {frame_count/total_time:.2f} fps")
    print(f"✓ Output saved: {output_video_path}")
    
    if os.path.exists(output_video_path):
        file_size = os.path.getsize(output_video_path) / (1024 * 1024)
        print(f"✓ Output file size: {file_size:.2f} MB")
    
    print("="*70)
    
    # Return summary data
    return {
        'total_persons': len(person_histories),
        'cheating_persons': list(sorted(cheating_persons)),
        'total_frames': frame_count,
        'processing_time': total_time
    }

#===================================================== Prediction End ==========================================================================
    
            

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
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        
        filename = secure_filename(file.filename)
        name_without_ext = os.path.splitext(filename)[0]
        time = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename =  filename = f"{name_without_ext}_{time}{os.path.splitext(filename)[1]}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        file.save(filepath)
        return jsonify({
            'filename': filename,
            'message': 'Video uploaded Sucessfully',
            'success':True
        }), 200
    except Exception as e:
        return jsonify({'error': f'Upload Failed: {str(e)}'}),500
        
    


@app.route('/process/<filename>', methods=['POST'])
def process_video(filename):
    
    try:
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        
        # Output filename (same name but in outputs folder)
        base_name = os.path.splitext(filename)[0]
        #output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_out.mp4")
        temp_path = os.path.join(TEMP_FOLDER, f"{base_name}_out.mp4")
        summary_path = os.path.join(SUMMARY_FOLDER, f"{base_name}_summary.json")
        print(f"DEBUG - Paths:")
        print(f"  Input exists: {os.path.exists(input_path)} - {input_path}")
        print(f"  Output dir exists: {os.path.exists(os.path.dirname(temp_path))} - {os.path.dirname(temp_path)}")
        print(f"  Output path: {temp_path}")
            
        summary = process_video_with_stable_tracking(
            input_video_path=input_path,
            output_video_path=temp_path,
            behavior_model_path="models/yolov8m_best.pt",
            person_model_path=None,  # Uses YOLOv8n pretrained
            person_conf_threshold=0.7  # Higher = fewer false positives (try 0.6-0.8)
        )
        with open(summary_path, 'w', encoding='utf-8') as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        
        return jsonify({'success': True}), 200
    except Exception as e:
        
        print("Processing error:", e)
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500
        
    

# @app.route('/results/<filename>')
# def show_results(filename):
    
    
#     base_name = os.path.splitext(filename)[0]
        
#     summary_path = os.path.join(SUMMARY_FOLDER, f"{base_name}_summary.json")
#     # output_video_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_out.mp4")
#     output_video_path = f"outputs\{base_name}_out.mp4"
    
#     summary = {}
#     if os.path.exists(summary_path):
#         with open(summary_path, 'r', encoding='utf-8') as fh:
#             summary = json.load(fh)
#     else:
#     # You can choose to abort(404) here or pass empty summary & message to template
#         summary = {'error': 'summary not available (processing may have failed or not finished yet)'}
    
#     return render_template('results.html', filename=filename, summary=summary,
#                            output_video=output_video_path)
@app.route('/results/<filename>')
def show_results(filename):
    base_name = os.path.splitext(filename)[0]
    
    # Full path for reading the summary file
    summary_path = os.path.join(SUMMARY_FOLDER, f"{base_name}_summary.json")
    
    # Relative path for url_for (use forward slashes, no 'static/' prefix)
    output_video_path = f"outputs/{base_name}_out.mp4"  # ✅ Forward slash!
    
    temp_output = os.path.join(TEMP_FOLDER, f"{base_name}_out.mp4")
    final_output = os.path.join(OUTPUT_FOLDER, f"{base_name}_out.mp4")
    
    if os.path.exists(temp_output) and not os.path.exists(final_output):
        convert_video_for_web(temp_output, final_output)
        os.remove(temp_output) 
    
    # Load summary
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as fh:
            summary = json.load(fh)
        print(f"✓ Summary loaded successfully:")
        print(f"  Total persons: {summary.get('total_persons', 'N/A')}")
        print(f"  Cheating persons: {summary.get('cheating_persons', [])}")
        print(f"  Total frames: {summary.get('total_frames', 'N/A')}")
        print(f"  Processing time: {summary.get('processing_time', 'N/A')}")
    else:
        print(f"⚠️ Summary file not found at: {summary_path}")
        summary = {
            'error': 'Summary not available',
            'total_persons': 0,
            'cheating_persons': [],
            'total_frames': 0,
            'processing_time': 0
        }
    
    # Verify output video exists
    video_full_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_out.mp4")
    if os.path.exists(video_full_path):
        print(f"✓ Output video found at: {video_full_path}")
    else:
        print(f"⚠️ Output video NOT found at: {video_full_path}")
    
    # Convert summary to JSON string for JavaScript
    summary_json = json.dumps(summary)
    
    return render_template('results.html', 
                         filename=filename, 
                         summary=summary_json,  # ✅ Pass as JSON string
                         output_video=output_video_path)

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