from flask import Flask, render_template, request, jsonify
import os, shutil, json, time, subprocess
from werkzeug.utils import secure_filename
from datetime import datetime
from collections import defaultdict, OrderedDict
import cv2, numpy as np
from ultralytics import YOLO
from flask import send_file
from mlflow.tracking import MlflowClient
import mlflow

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = '!bandar-bhalu'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'TEMP_output')
SUMMARY_FOLDER = os.path.join(BASE_DIR, 'summary')


MAX_FILE_SIZE = 500 * 1024 * 1024
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv'}
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# ======================= MLflow Model Loading =======================
client = MlflowClient()
os.makedirs("./models_cache", exist_ok=True)

# Load person detector model
model_name = "person-detector_model"

try:
    # Get the registered model
    model_versions = client.search_model_versions(f"name='{model_name}'")
    if not model_versions:
        raise ValueError(f"No versions found for model: {model_name}")
    
    latest = sorted(model_versions, key=lambda x: int(x.version), reverse=True)[0]
    run_id = latest.run_id
    
    print(f"Loading {model_name} from run: {run_id}")
    
    # Download the artifact 
    model_path = client.download_artifacts(
        run_id, 
        "models/YOLOv8n.pt", 
        dst_path="./models_cache"
    )
    
    person_model = YOLO(model_path)
    print(f"✓ Person detector loaded from: {model_path}")
    
except Exception as e:
    print(f"Error loading person detector: {e}")
    import traceback
    traceback.print_exc()
    raise

# Load cheating detector model (same approach)
model_name = "cheating-detector_model"

try:
    model_versions = client.search_model_versions(f"name='{model_name}'")
    if not model_versions:
        raise ValueError(f"No versions found for model: {model_name}")
    
    latest = sorted(model_versions, key=lambda x: int(x.version), reverse=True)[0]
    run_id = latest.run_id
    
    print(f"Loading {model_name} from run: {run_id}")
    
    model_path = client.download_artifacts(
        run_id, 
        "models/YOLOv8m_best.pt",  
        dst_path="./models_cache"
    )
    
    behavior_model = YOLO(model_path)
    print(f"✓ Cheating detector loaded from: {model_path}")
    
except Exception as e:
    print(f"Error loading cheating detector: {e}")
    import traceback
    traceback.print_exc()
    raise

print("\n" + "="*50)
print("Cheating Detection System Starting...")
print("="*50)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ======================= Tracker =======================
class PersonTracker:
    def __init__(self, max_disappeared=60, max_distance=200):
        self.next_id = 1
        self.persons = OrderedDict()
        self.disappeared = OrderedDict()
        self.centroids = OrderedDict()
        self.first_seen = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, bbox, frame_number):
        pid = self.next_id
        self.persons[pid] = bbox
        self.disappeared[pid] = 0
        self.centroids[pid] = self._get_centroid(bbox)
        self.first_seen[pid] = frame_number
        self.next_id += 1
        return pid

    def deregister(self, pid):
        for d in [self.persons, self.disappeared, self.centroids, self.first_seen]:
            if pid in d: del d[pid]

    def _get_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1+x2)//2, (y1+y2)//2)

    def _calculate_distance(self, c1, c2):
        return np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

    def _calculate_iou(self, box1, box2):
        x1_i = max(box1[0], box2[0])
        y1_i = max(box1[1], box2[1])
        x2_i = min(box1[2], box2[2])
        y2_i = min(box1[3], box2[3])
        if x2_i < x1_i or y2_i < y1_i: return 0.0
        inter = (x2_i-x1_i)*(y2_i-y1_i)
        union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
        return inter/union if union>0 else 0.0

    def update(self, input_bboxes, frame_number):
        if not input_bboxes:
            for pid in list(self.disappeared.keys()):
                self.disappeared[pid] += 1
                if self.disappeared[pid] > self.max_disappeared:
                    self.deregister(pid)
            return self.persons

        input_centroids = [self._get_centroid(b) for b in input_bboxes]
        if not self.persons:
            for bbox in input_bboxes: self.register(bbox, frame_number)
        else:
            pids = list(self.persons.keys())
            ex_centroids = [self.centroids[pid] for pid in pids]
            ex_boxes = [self.persons[pid] for pid in pids]
            D = np.zeros((len(pids), len(input_bboxes)))
            for i, (ec, eb) in enumerate(zip(ex_centroids, ex_boxes)):
                for j, (ic, ib) in enumerate(zip(input_centroids, input_bboxes)):
                    dist = self._calculate_distance(ec, ic)
                    iou = self._calculate_iou(eb, ib)
                    D[i,j] = 0.4*dist + 0.6*(1-iou)*200
            used_rows, used_cols, matches = set(), set(), []
            for _ in range(min(len(pids), len(input_bboxes))):
                min_cost = np.inf
                for i in range(len(pids)):
                    if i in used_rows: continue
                    for j in range(len(input_bboxes)):
                        if j in used_cols: continue
                        if D[i,j]<min_cost:
                            min_cost, min_row, min_col = D[i,j], i, j
                if min_cost<self.max_distance:
                    matches.append((min_row, min_col))
                    used_rows.add(min_row)
                    used_cols.add(min_col)
            for row, col in matches:
                pid = pids[row]
                self.persons[pid] = input_bboxes[col]
                self.centroids[pid] = input_centroids[col]
                self.disappeared[pid] = 0
            for i in set(range(len(pids))) - used_rows:
                pid = pids[i]
                self.disappeared[pid] +=1
                if self.disappeared[pid]>self.max_disappeared: self.deregister(pid)
            for j in set(range(len(input_bboxes))) - used_cols:
                nc = input_centroids[j]
                if all(self._calculate_distance(nc, self.centroids[pid])>=self.max_distance*0.5 for pid in self.persons.keys()):
                    self.register(input_bboxes[j], frame_number)
        return self.persons

# ======================= Helpers =======================
def calculate_iou(box1, box2):
    x1_i = max(box1[0], box2[0])
    y1_i = max(box1[1], box2[1])
    x2_i = min(box1[2], box2[2])
    y2_i = min(box1[3], box2[3])
    if x2_i < x1_i or y2_i < y1_i: return 0.0
    inter = (x2_i-x1_i)*(y2_i-y1_i)
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    return inter/union if union>0 else 0.0

def associate_parts_with_persons(person_bboxes, part_detections, iou_thresh=0.1):
    assoc = defaultdict(list)
    for part in part_detections:
        best_iou, best_pid = 0, None
        for pid, pb in person_bboxes.items():
            iou = calculate_iou(part['bbox'], pb)
            if iou>best_iou and iou>iou_thresh: best_iou, best_pid = iou, pid
        if best_pid: assoc[best_pid].append(part)
    return assoc


def make_cheating_decision(detection_history, total_frames):
    if not detection_history: return "UNKNOWN","No detections",0.0
    counts = defaultdict(int)
    for d in detection_history: counts[d['class']]+=1
    if counts['Cheatchits']>0:
        return "CHEATING","Cheat sheets detected",min(counts['Cheatchits']/total_frames*100,100)
    watching = counts['backwatching']+counts['sidewatching']+counts['frontwatching']
    if (watching/total_frames)*100>30: return "CHEATING",f"Watching others {(watching/total_frames)*100:.1f}%",min((watching/total_frames)*100,100)
    suspicious = counts['SuspicousHandMovement']+counts['SuspicousHeaddMovement']
    if (suspicious/total_frames)*100>50: return "CHEATING",f"Suspicious behavior {(suspicious/total_frames)*100:.1f}%",min((suspicious/total_frames)*100,100)
    normal = counts['NormalHand']+counts['NormalHead']
    if len(detection_history)>0 and (normal/len(detection_history))*100>70: return "NOT CHEATING","Mostly normal behavior",min((normal/len(detection_history))*100,100)
    return "SUSPICIOUS","Mixed behavior patterns",50.0

def get_bright_color(pid):
    colors=[(255,0,255),(0,255,255),(0,165,255),(255,255,0),(255,0,127),(0,255,0),(147,20,255),(255,191,0),(203,192,255),(92,92,205)]
    return colors[(pid-1)%len(colors)]

# ======================= Video Processing =======================
def process_video_with_stable_tracking(input_path, output_path, behavior_model, person_model, person_conf_threshold=0.7):
    print("="*70)
    print("VIDEO PROCESSING WITH STABLE PERSON TRACKING")
    print("="*70)


    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened(): print("❌ Error: Could not open video"); return
    w,h,fps,total_frames = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w,h))

    tracker = PersonTracker(max_disappeared=60,max_distance=200)
    person_histories, person_frame_counts, cheating_persons = defaultdict(list), defaultdict(int), set()

    frame_count = 0
    start_time = time.time()
    

    while True:
        
        ret, frame = cap.read()
        if not ret: break
        frame_count +=1

        person_results = person_model(frame, classes=[0], conf=person_conf_threshold, verbose=False)
        detected_bboxes=[]
        for box in person_results[0].boxes:
            x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
            area = (x2-x1)*(y2-y1)
            if area>(w*h*0.01): detected_bboxes.append([x1,y1,x2,y2])
        tracked_persons = tracker.update(detected_bboxes, frame_count)
        for pid in tracked_persons.keys(): person_frame_counts[pid]+=1

        behavior_results = behavior_model(frame, verbose=False)
        part_detections=[]
        for box in behavior_results[0].boxes:
            bbox = box.xyxy[0].cpu().numpy()
            part_detections.append({'bbox':bbox,'class':behavior_model.names[int(box.cls[0])],'conf':float(box.conf[0])})
        associations = associate_parts_with_persons(tracked_persons, part_detections)
        for pid in associations: person_histories[pid].extend(associations[pid])

        # Visualization
        annotated_frame = frame.copy()
        for pid, bbox in tracked_persons.items():
            x1,y1,x2,y2 = map(int,bbox)
            verdict,_,_ = make_cheating_decision(person_histories[pid],person_frame_counts[pid])
            if verdict=="CHEATING": cheating_persons.add(pid)
            color = (0,0,255) if verdict=="CHEATING" else get_bright_color(pid)
            cv2.rectangle(annotated_frame,(x1,y1),(x2,y2),color,4)
            label = f"Person {pid}: {verdict}"
            cv2.putText(annotated_frame,label,(x1+7,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
        out.write(annotated_frame)

        if frame_count%30==0 or frame_count==total_frames:
            elapsed=time.time()-start_time
            speed=frame_count/elapsed
            eta=(total_frames-frame_count)/speed if speed>0 else 0
            print(f"Frame: {frame_count}/{total_frames} | Speed: {speed:.1f} fps | ETA: {eta:.1f}s | Active: {len(tracked_persons)} | Total IDs: {tracker.next_id-1} | Cheating: {len(cheating_persons)}")

    cap.release()
    out.release()

    # Final summary
    print("\n"+"="*70)
    print("FINAL CHEATING VERDICTS")
    for pid in sorted(person_histories.keys()):
        verdict, reason, confidence = make_cheating_decision(person_histories[pid], person_frame_counts[pid])
        print(f"Person {pid}: Verdict={verdict}, Reason={reason}, Confidence={confidence:.1f}%, Frames={person_frame_counts[pid]}/{total_frames}")

    print(f"\nCheating persons: {sorted(cheating_persons)}")
    print("Processing complete!")
    return {'total_persons':len(person_histories),'cheating_persons':list(sorted(cheating_persons)),'total_frames':frame_count,'processing_time':time.time()-start_time}
    

    
    
# ======================= Flask Routes =======================
@app.route('/')
def index(): return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    file = request.files.get('video')
    if not file or file.filename=='' or not allowed_file(file.filename):
        return jsonify({'error':'Invalid file'}),400
    filename = secure_filename(file.filename)
    name,time_ext=os.path.splitext(filename)[0],os.path.splitext(filename)[1]
    timestamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    filename=f"{name}_{timestamp}{time_ext}"
    filepath=os.path.join(UPLOAD_FOLDER,filename)
    file.save(filepath)
    return jsonify({'filename':filename,'success':True,'message':'Video uploaded successfully'}),200

@app.route('/process/<filename>', methods=['POST'])
def process_video(filename):
    input_path=os.path.join(UPLOAD_FOLDER,filename)
    temp_path=os.path.join(OUTPUT_FOLDER,f"{os.path.splitext(filename)[0]}_out.mp4")
    summary_path=os.path.join(SUMMARY_FOLDER,f"{os.path.splitext(filename)[0]}_summary.json")

    summary=process_video_with_stable_tracking(input_path,temp_path,behavior_model,person_model)
    h264_temp = temp_path.replace('.mp4', '_h264_temp.mp4')
    subprocess.run(['ffmpeg', '-i', temp_path, '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-y', h264_temp])
    os.rename(h264_temp, temp_path)

    
    shutil.copy(temp_path, os.path.join(app.static_folder, "outputs", os.path.basename(temp_path)))
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return jsonify({'success':True}),200

@app.route('/results/<filename>')
def show_results(filename):
    summary_path = os.path.join(SUMMARY_FOLDER, f"{os.path.splitext(filename)[0]}_summary.json")
    output_video_path = f"outputs/{os.path.splitext(filename)[0]}_out.mp4"

    download_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}_out.mp4")
    
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f: 
            summary = json.load(f)
    else: 
        summary = {'error': 'Summary not available', 'total_persons': 0, 'cheating_persons': [], 'total_frames': 0, 'processing_time': 0}
    
    return render_template('results.html',
                           filename=filename,
                           summary=json.dumps(summary),
                           output_video=output_video_path,
                           download_path=download_path)

@app.route('/download_result/<filename>')
def download_result(filename):
    download_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}_out.mp4")
    return send_file(download_path, as_attachment=True)

@app.route('/delete/<filename>',methods=['DELETE'])
def delete_video(filename):

    name_without_ext, ext = os.path.splitext(filename)
    
    print(f"Deleting files for: {filename}")
    print(f"Name without extension: {name_without_ext}")
    print(f"Extension: {ext}")
    
    filepaths = [
        os.path.join(UPLOAD_FOLDER, filename),
        os.path.join(OUTPUT_FOLDER, name_without_ext + '_out' + ext),
        os.path.join(SUMMARY_FOLDER, name_without_ext + '_summary.json'),
        os.path.join("static", "outputs", name_without_ext + '_out' + ext),
    ]

    deleted = False
    deleted_files = []

    print("Searching for files:")
    for path in filepaths:
        print(f"  Checking: {path}")
        if os.path.exists(path):
            os.remove(path)
            deleted = True
            deleted_files.append(os.path.basename(path))
            print(f"  ✓ Deleted: {path}")

    if deleted:
        return jsonify({
            'success': True, 
            'message': f'{len(deleted_files)} file(s) deleted',
            'deleted_files': deleted_files
        }), 200
    else:
        return jsonify({
            'error': 'No files found to delete',
            'debug': {
                'filename': filename,
                'searched_paths': [os.path.basename(p) for p in filepaths]
            }
        }), 404

@app.errorhandler(413)
def too_large(e): return jsonify({'error':'File too large. Max 500MB'}),413
@app.errorhandler(500)
def internal_error(e): return jsonify({'error':'Internal server error'}),500

if __name__=='__main__':
    os.makedirs(UPLOAD_FOLDER,exist_ok=True)
    os.makedirs(OUTPUT_FOLDER,exist_ok=True)
    os.makedirs(SUMMARY_FOLDER,exist_ok=True)
    print("="*50)
    print("Cheating Detection System Starting...")
    print(f"Upload folder: {UPLOAD_FOLDER} | Allowed formats: {', '.join(ALLOWED_EXTENSIONS)} | Max size: {MAX_FILE_SIZE//(1024*1024)}MB")
    print("="*50)
    app.run(debug=True, host='0.0.0.0', port=5001)
    
    
