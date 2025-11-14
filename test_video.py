# from ultralytics import YOLO
# import cv2
# import os
# from pathlib import Path
# import time

# def process_video(input_video_path, output_video_path, model_path):
    
#     print("="*60)
#     print("VIDEO PROCESSING STARTED")
#     print("="*60)
    
#     # Check 1: Verify input video exists
#     print("\n[1/7] Checking input video...")
#     if not os.path.exists(input_video_path):
#         print(f"❌ Error: Input video not found at '{input_video_path}'")
#         exit()
#     print(f"✓ Input video found: {input_video_path}")
    
#     # Check 2: Verify model exists
#     print("\n[2/7] Checking model...")
#     if not os.path.exists(model_path):
#         print(f"❌ Error: Model not found at '{model_path}'")
#         exit()
#     print(f"✓ Model found: {model_path}")
    
#     # Check 3: Load YOLO model
#     print("\n[3/7] Loading YOLO model...")
#     try:
#         model = YOLO(model_path)
#         print(f"✓ Model loaded successfully")
#         print(f"  Model classes: {list(model.names.values())}")
#     except Exception as e:
#         print(f"❌ Error loading model: {e}")
#         exit()
    
#     # Check 4: Open video capture
#     print("\n[4/7] Opening video file...")
#     cap = cv2.VideoCapture(input_video_path)
    
#     if not cap.isOpened():
#         print("❌ Error: Could not open video file")
#         print("Possible reasons:")
#         print("  - File is corrupted")
#         print("  - Codec not supported")
#         print("  - File path is incorrect")
#         exit()
#     print("✓ Video opened successfully")
    
#     # Check 5: Get and verify video properties
#     print("\n[5/7] Reading video properties...")
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     # Verify properties are valid
#     if frame_width <= 0 or frame_height <= 0 or fps <= 0:
#         print("❌ Error: Invalid video properties")
#         print(f"  Width: {frame_width}, Height: {frame_height}, FPS: {fps}")
#         cap.release()
#         exit()
    
#     print("✓ Video properties:")
#     print(f"  Resolution: {frame_width} x {frame_height}")
#     print(f"  FPS: {fps}")
#     print(f"  Total Frames: {total_frames}")
#     print(f"  Duration: {total_frames/fps:.2f} seconds")
    
#     # Check 6: Create output directory and video writer
#     print("\n[6/7] Setting up output...")
#     output_dir = os.path.dirname(output_video_path)
    
#     # Create output directory if needed
#     if output_dir and not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#         print(f"✓ Created output directory: {output_dir}")
#     else:
#         print(f"✓ Output directory exists: {output_dir if output_dir else 'current directory'}")
    
#     # Create video writer
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
#     # Verify video writer is opened
#     if not out.isOpened():
#         print("❌ Error: Could not create output video writer")
#         print("Possible reasons:")
#         print("  - Invalid output path")
#         print("  - Codec not available")
#         print("  - Insufficient permissions")
#         cap.release()
#         exit()
#     print(f"✓ Output video writer created: {output_video_path}")
    
#     # Check 7: Process video frames
#     print("\n[7/7] Processing video frames...")
#     print("-"*60)
    
#     frame_count = 0
#     detection_count = 0
#     start_time = time.time()
    
#     # Test first frame to ensure everything works
#     ret, test_frame = cap.read()
#     if not ret:
#         print("❌ Error: Could not read first frame")
#         cap.release()
#         out.release()
#         exit()
    
#     print("✓ First frame read successfully")
#     print("Running detection on first frame...")
    
#     try:
#         test_results = model(test_frame, verbose=False)
#         test_annotated = test_results[0].plot()
#         out.write(test_annotated)
#         num_detections = len(test_results[0].boxes)
#         print(f"✓ First frame processed: {num_detections} objects detected")
#         frame_count = 1
#         detection_count += num_detections
#     except Exception as e:
#         print(f"❌ Error processing first frame: {e}")
#         cap.release()
#         out.release()
#         exit()
    
#     print("\nProcessing remaining frames...")
#     print("-"*60)

#     while True:
#         ret, frame = cap.read()
        
#         if not ret:
#             break
        
#         frame_count += 1

#         try:
#             results = model(frame, verbose=False)
#             annotated_frame = results[0].plot()
            
#             # Count detections
#             num_detections = len(results[0].boxes)
#             detection_count += num_detections
            
#             out.write(annotated_frame)
            
#         except Exception as e:
#             print(f"⚠️  Warning: Error processing frame {frame_count}: {e}")
#             continue
        
#         # Progress indicator every 30 frames
#         if frame_count % 30 == 0:
#             elapsed_time = time.time() - start_time
#             fps_processing = frame_count / elapsed_time
#             progress = (frame_count / total_frames) * 100
#             eta = (total_frames - frame_count) / fps_processing if fps_processing > 0 else 0
#             avg_detections = detection_count / frame_count
            
#             print(f"Frame: {frame_count:5d}/{total_frames} ({progress:5.1f}%) | "
#                   f"Speed: {fps_processing:5.1f} fps | "
#                   f"ETA: {eta:6.1f}s | "
#                   f"Avg detections: {avg_detections:.1f}")
    
#     # Cleanup
#     cap.release()
#     out.release()
    
#     # Final statistics
#     total_time = time.time() - start_time
    
#     print("-"*60)
#     print("\n" + "="*60)
#     print("PROCESSING COMPLETE!")
#     print("="*60)
#     print(f"✓ Total frames processed: {frame_count}")
#     print(f"✓ Total detections: {detection_count}")
#     print(f"✓ Average detections per frame: {detection_count/frame_count:.2f}")
#     print(f"✓ Processing time: {total_time:.2f} seconds")
#     print(f"✓ Processing speed: {frame_count/total_time:.2f} fps")
#     print(f"✓ Output video saved to: {output_video_path}")
    
#     # Verify output file exists and has size
#     if os.path.exists(output_video_path):
#         file_size = os.path.getsize(output_video_path) / (1024 * 1024)  # MB
#         print(f"✓ Output file size: {file_size:.2f} MB")
#     else:
#         print("⚠️  Warning: Output file not found after processing")
    
#     print("="*60)

# # Usage        
# input_video_path = "test_video/test_video1.mp4"
# output_video_path = "outputs/output_video1_m.mp4"
# model_path = "models/yolov8m_best.pt"

# process_video(input_video_path, output_video_path, model_path)

# from ultralytics import YOLO
# import cv2
# import os
# import time
# from collections import defaultdict
# import numpy as np

# # ============================================================
# # HELPER FUNCTIONS (Same as before)
# # ============================================================

# def calculate_iou(box1, box2):
#     """Calculate Intersection over Union between two boxes"""
#     x1_1, y1_1, x2_1, y2_1 = box1
#     x1_2, y1_2, x2_2, y2_2 = box2
    
#     x1_i = max(x1_1, x1_2)
#     y1_i = max(y1_1, y1_2)
#     x2_i = min(x2_1, x2_2)
#     y2_i = min(y2_1, y2_2)
    
#     if x2_i < x1_i or y2_i < y1_i:
#         return 0.0
    
#     intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
#     box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
#     box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
#     union_area = box1_area + box2_area - intersection_area
    
#     iou = intersection_area / union_area if union_area > 0 else 0.0
#     return iou


# def match_person_across_frames(current_persons, previous_persons, iou_threshold=0.5):
#     """Match persons between frames using IoU"""
#     matches = {}
#     used_ids = set()
    
#     for curr_idx, curr_bbox in enumerate(current_persons):
#         best_iou = 0
#         best_id = None
        
#         for person_id, prev_bbox in previous_persons.items():
#             if person_id in used_ids:
#                 continue
            
#             iou = calculate_iou(curr_bbox, prev_bbox)
            
#             if iou > best_iou and iou > iou_threshold:
#                 best_iou = iou
#                 best_id = person_id
        
#         if best_id is not None:
#             matches[curr_idx] = best_id
#             used_ids.add(best_id)
    
#     return matches


# def associate_parts_with_persons(person_bboxes, part_detections, iou_threshold=0.3):
#     """Associate body parts with persons"""
#     associations = defaultdict(list)
    
#     for part in part_detections:
#         part_bbox = part['bbox']
#         best_iou = 0
#         best_person_idx = None
        
#         for person_idx, person_bbox in enumerate(person_bboxes):
#             iou = calculate_iou(part_bbox, person_bbox)
            
#             if iou > best_iou and iou > iou_threshold:
#                 best_iou = iou
#                 best_person_idx = person_idx
        
#         if best_person_idx is not None:
#             associations[best_person_idx].append(part)
    
#     return associations


# def make_cheating_decision(detection_history, total_frames):
#     """Decide if person is cheating"""
#     if not detection_history:
#         return "UNKNOWN", "No detections", 0.0
    
#     detection_counts = defaultdict(int)
#     for detection in detection_history:
#         detection_counts[detection['class']] += 1
    
#     total_detections = len(detection_history)
    
#     # RULE 1: Cheat sheets
#     if detection_counts['Cheatchits'] > 0:
#         confidence = min(detection_counts['Cheatchits'] / total_frames * 100, 100)
#         return "CHEATING", "Cheat sheets detected", confidence
    
#     # RULE 2: Watching others
#     watching_count = (detection_counts['backwatching'] + 
#                      detection_counts['sidewatching'] + 
#                      detection_counts['frontwatching'])
#     watching_percentage = (watching_count / total_frames) * 100
    
#     if watching_percentage > 30:
#         confidence = min(watching_percentage, 100)
#         return "CHEATING", f"Watching others {watching_percentage:.1f}% of time", confidence
    
#     # RULE 3: Suspicious movements
#     suspicious_count = (detection_counts['SuspicousHandMovement'] + 
#                        detection_counts['SuspicousHeaddMovement'])
#     suspicious_percentage = (suspicious_count / total_frames) * 100
    
#     if suspicious_percentage > 50:
#         confidence = min(suspicious_percentage, 100)
#         return "CHEATING", f"Suspicious behavior {suspicious_percentage:.1f}% of time", confidence
    
#     # RULE 4: Normal behavior
#     normal_count = (detection_counts['NormalHand'] + 
#                    detection_counts['NormalHead'])
#     normal_percentage = (normal_count / total_detections) * 100
    
#     if normal_percentage > 70:
#         confidence = min(normal_percentage, 100)
#         return "NOT CHEATING", "Mostly normal behavior", confidence
    
#     # RULE 5: Mixed behavior
#     confidence = 50.0
#     return "SUSPICIOUS", "Mixed behavior patterns", confidence


# # ============================================================
# # OPENCV PERSON DETECTION FUNCTIONS
# # ============================================================

# def detect_persons_hog(frame):
#     """
#     Detect persons using OpenCV HOG descriptor
    
#     Args:
#         frame: Input frame
    
#     Returns:
#         person_bboxes: List of [[x1, y1, x2, y2], ...]
#         weights: Confidence scores for each detection
#     """
#     # Initialize HOG descriptor
#     hog = cv2.HOGDescriptor()
#     hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
#     # Detect people
#     # winStride: smaller = more accurate but slower (try (4,4) or (8,8))
#     # padding: border padding
#     # scale: image pyramid scale factor
#     (bboxes, weights) = hog.detectMultiScale(
#         frame,
#         winStride=(8, 8),
#         padding=(8, 8),
#         scale=1.05,
#         useMeanshiftGrouping=False
#     )
    
#     # Convert (x, y, w, h) to (x1, y1, x2, y2)
#     person_bboxes = []
#     for (x, y, w, h) in bboxes:
#         person_bboxes.append([x, y, x + w, y + h])
    
#     return person_bboxes, weights


# def detect_persons_haar(frame):
#     """
#     Detect persons using Haar Cascade
    
#     Args:
#         frame: Input frame
    
#     Returns:
#         person_bboxes: List of [[x1, y1, x2, y2], ...]
#     """
#     # Load Haar Cascade
#     cascade_path = cv2.data.haarcascades + 'haarcascade_fullbody.xml'
#     person_cascade = cv2.CascadeClassifier(cascade_path)
    
#     # Convert to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # Detect persons
#     persons = person_cascade.detectMultiScale(
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=3,
#         minSize=(30, 30)
#     )
    
#     # Convert (x, y, w, h) to (x1, y1, x2, y2)
#     person_bboxes = []
#     for (x, y, w, h) in persons:
#         person_bboxes.append([x, y, x + w, y + h])
    
#     return person_bboxes


# def non_max_suppression(boxes, overlapThresh=0.3):
#     """
#     Apply Non-Maximum Suppression to remove overlapping boxes
    
#     Args:
#         boxes: List of bounding boxes [[x1, y1, x2, y2], ...]
#         overlapThresh: IoU threshold for considering boxes as duplicates
    
#     Returns:
#         Filtered list of boxes
#     """
#     if len(boxes) == 0:
#         return []
    
#     # Convert to numpy array
#     boxes = np.array(boxes).astype("float")
    
#     # Initialize list of picked indexes
#     pick = []
    
#     # Get coordinates
#     x1 = boxes[:, 0]
#     y1 = boxes[:, 1]
#     x2 = boxes[:, 2]
#     y2 = boxes[:, 3]
    
#     # Calculate area
#     area = (x2 - x1 + 1) * (y2 - y1 + 1)
    
#     # Sort by bottom-right y-coordinate
#     idxs = np.argsort(y2)
    
#     while len(idxs) > 0:
#         # Get last index and add to picked
#         last = len(idxs) - 1
#         i = idxs[last]
#         pick.append(i)
        
#         # Find overlaps
#         xx1 = np.maximum(x1[i], x1[idxs[:last]])
#         yy1 = np.maximum(y1[i], y1[idxs[:last]])
#         xx2 = np.minimum(x2[i], x2[idxs[:last]])
#         yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
#         # Calculate overlap
#         w = np.maximum(0, xx2 - xx1 + 1)
#         h = np.maximum(0, yy2 - yy1 + 1)
#         overlap = (w * h) / area[idxs[:last]]
        
#         # Delete overlapping boxes
#         idxs = np.delete(idxs, np.concatenate(([last],
#             np.where(overlap > overlapThresh)[0])))
    
#     return boxes[pick].astype("int").tolist()


# # ============================================================
# # MAIN PROCESSING FUNCTION WITH OPENCV PERSON DETECTION
# # ============================================================

# def process_video_with_opencv_person_detection(
#     input_video_path, 
#     output_video_path, 
#     behavior_model_path,
#     person_detector='hog'  # 'hog' or 'haar'
# ):
#     """
#     Process video with OpenCV person detection
    
#     Args:
#         input_video_path: Path to input video
#         output_video_path: Path to output video
#         behavior_model_path: Path to behavior detection model
#         person_detector: 'hog' or 'haar'
#     """
    
#     print("="*70)
#     print("VIDEO PROCESSING WITH OPENCV PERSON DETECTION")
#     print("="*70)
    
#     # Load behavior model
#     print(f"\n[1/7] Loading behavior detection model...")
#     behavior_model = YOLO(behavior_model_path)
#     print(f"✓ Model loaded: {list(behavior_model.names.values())}")
    
#     # Initialize person detector
#     print(f"\n[2/7] Initializing {person_detector.upper()} person detector...")
#     if person_detector == 'haar':
#         cascade_path = cv2.data.haarcascades + 'haarcascade_fullbody.xml'
#         if not os.path.exists(cascade_path):
#             print("❌ Error: Haar cascade file not found")
#             return
#         print("✓ Haar cascade person detector initialized")
#     else:
#         print("✓ HOG person detector initialized")
    
#     # Open video
#     print("\n[3/7] Opening video...")
#     cap = cv2.VideoCapture(input_video_path)
    
#     if not cap.isOpened():
#         print("❌ Error: Could not open video")
#         return
    
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     print(f"✓ Video: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")
    
#     # Setup output
#     print("\n[4/7] Setting up output...")
#     output_dir = os.path.dirname(output_video_path)
#     if output_dir and not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
#     print(f"✓ Output: {output_video_path}")
    
#     # Initialize tracking
#     print("\n[5/7] Initializing tracking system...")
#     person_tracker = {}
#     next_person_id = 1
#     person_histories = defaultdict(list)
#     person_frame_counts = defaultdict(int)
#     print("✓ Tracking initialized")
    
#     # Process video
#     print("\n[6/7] Processing frames...")
#     print("-"*70)
    
#     frame_count = 0
#     start_time = time.time()
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         frame_count += 1
#         annotated_frame = frame.copy()
        
#         # ===== STEP 1: Detect persons using OpenCV =====
#         if person_detector == 'hog':
#             current_persons, weights = detect_persons_hog(frame)
#         else:  # haar
#             current_persons = detect_persons_haar(frame)
        
#         # Apply Non-Maximum Suppression to remove duplicate detections
#         if len(current_persons) > 0:
#             current_persons = non_max_suppression(current_persons, overlapThresh=0.3)
        
#         # ===== STEP 2: Match persons across frames =====
#         matches = match_person_across_frames(current_persons, person_tracker)
        
#         new_person_tracker = {}
#         person_ids_this_frame = []
        
#         for person_idx, person_bbox in enumerate(current_persons):
#             if person_idx in matches:
#                 person_id = matches[person_idx]
#             else:
#                 person_id = next_person_id
#                 next_person_id += 1
            
#             new_person_tracker[person_id] = person_bbox
#             person_ids_this_frame.append(person_id)
#             person_frame_counts[person_id] += 1
        
#         person_tracker = new_person_tracker
        
#         # ===== STEP 3: Detect behaviors =====
#         behavior_results = behavior_model(frame, verbose=False)
        
#         part_detections = []
#         for box in behavior_results[0].boxes:
#             bbox = box.xyxy[0].cpu().numpy()
#             class_id = int(box.cls[0])
#             class_name = behavior_model.names[class_id]
#             confidence = float(box.conf[0])
            
#             part_detections.append({
#                 'bbox': bbox,
#                 'class': class_name,
#                 'conf': confidence
#             })
        
#         # ===== STEP 4: Associate parts with persons =====
#         associations = associate_parts_with_persons(current_persons, part_detections)
        
#         # ===== STEP 5: Update histories =====
#         for person_idx, person_id in enumerate(person_ids_this_frame):
#             if person_idx in associations:
#                 person_histories[person_id].extend(associations[person_idx])
        
#         # ===== STEP 6: Visualize =====
        
#         # Draw person boxes with verdicts
#         for person_id, person_bbox in person_tracker.items():
#             x1, y1, x2, y2 = map(int, person_bbox)
            
#             verdict, reason, confidence = make_cheating_decision(
#                 person_histories[person_id],
#                 person_frame_counts[person_id]
#             )
            
#             # Color based on verdict
#             if verdict == "CHEATING":
#                 color = (0, 0, 255)  # Red
#             elif verdict == "SUSPICIOUS":
#                 color = (0, 165, 255)  # Orange
#             elif verdict == "NOT CHEATING":
#                 color = (0, 255, 0)  # Green
#             else:
#                 color = (128, 128, 128)  # Gray
            
#             # Draw person box
#             cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            
#             # Draw label
#             label = f"Person {person_id}: {verdict}"
#             label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
#             cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
#                          (x1 + label_size[0], y1), color, -1)
#             cv2.putText(annotated_frame, label, (x1, y1 - 5), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
#         # Draw behavior detections
#         for person_idx in associations:
#             for detection in associations[person_idx]:
#                 x1, y1, x2, y2 = map(int, detection['bbox'])
#                 class_name = detection['class']
#                 conf = detection['conf']
                
#                 cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
#                 det_label = f"{class_name} {conf:.2f}"
#                 cv2.putText(annotated_frame, det_label, (x1, y1 - 5), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
#         # Write frame
#         out.write(annotated_frame)
        
#         # Progress
#         if frame_count % 30 == 0:
#             elapsed = time.time() - start_time
#             speed = frame_count / elapsed
#             eta = (total_frames - frame_count) / speed if speed > 0 else 0
#             print(f"Frame: {frame_count:5d}/{total_frames} | "
#                   f"Speed: {speed:5.1f} fps | ETA: {eta:6.1f}s | "
#                   f"Persons: {len(person_tracker)}")
    
#     cap.release()
#     out.release()
    
#     # Final verdicts
#     print("\n" + "="*70)
#     print("FINAL CHEATING VERDICTS")
#     print("="*70)
    
#     for person_id in sorted(person_histories.keys()):
#         verdict, reason, confidence = make_cheating_decision(
#             person_histories[person_id],
#             person_frame_counts[person_id]
#         )
        
#         print(f"\nPerson {person_id}:")
#         print(f"  Verdict: {verdict}")
#         print(f"  Reason: {reason}")
#         print(f"  Confidence: {confidence:.1f}%")
#         print(f"  Frames present: {person_frame_counts[person_id]}")
#         print(f"  Total detections: {len(person_histories[person_id])}")
        
#         detection_counts = defaultdict(int)
#         for det in person_histories[person_id]:
#             detection_counts[det['class']] += 1
        
#         print(f"  Detection breakdown:")
#         for class_name, count in sorted(detection_counts.items(), 
#                                        key=lambda x: x[1], reverse=True):
#             percentage = (count / len(person_histories[person_id])) * 100
#             print(f"    {class_name}: {count} ({percentage:.1f}%)")
    
#     total_time = time.time() - start_time
    
#     print("\n" + "="*70)
#     print("PROCESSING COMPLETE!")
#     print("="*70)
#     print(f"✓ Frames: {frame_count}")
#     print(f"✓ Persons tracked: {len(person_histories)}")
#     print(f"✓ Time: {total_time:.2f}s")
#     print(f"✓ Speed: {frame_count/total_time:.2f} fps")
#     print(f"✓ Output: {output_video_path}")
#     print("="*70)


# # ============================================================
# # USAGE
# # ============================================================

# if __name__ == "__main__":
#     process_video_with_opencv_person_detection(
#         input_video_path="test_video/test_video1.mp4",
#         output_video_path="outputs/output_opencv_tracking.mp4",
#         behavior_model_path="models/yolov8m_best.pt",
#         person_detector='hog'  # Use 'hog' or 'haar'
#     )





from ultralytics import YOLO
import cv2
import os
import time
from collections import defaultdict, OrderedDict
import numpy as np

# ============================================================
# IMPROVED TRACKING CLASSES
# ============================================================

class PersonTracker:
    """
    Improved person tracker with centroid tracking and ID persistence
    """
    
    def __init__(self, max_disappeared=30, max_distance=100):
        """
        Initialize tracker
        
        Args:
            max_disappeared: Maximum frames a person can be missing before removing
            max_distance: Maximum distance (pixels) to consider same person
        """
        self.next_id = 1
        self.persons = OrderedDict()  # {person_id: bbox}
        self.disappeared = OrderedDict()  # {person_id: frame_count}
        self.centroids = OrderedDict()  # {person_id: (cx, cy)}
        
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
    
    def register(self, bbox):
        """
        Register a new person
        
        Args:
            bbox: [x1, y1, x2, y2]
        
        Returns:
            person_id: Assigned ID
        """
        person_id = self.next_id
        self.persons[person_id] = bbox
        self.disappeared[person_id] = 0
        self.centroids[person_id] = self._get_centroid(bbox)
        self.next_id += 1
        return person_id
    
    def deregister(self, person_id):
        """Remove a person from tracking"""
        del self.persons[person_id]
        del self.disappeared[person_id]
        del self.centroids[person_id]
    
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
    
    def update(self, input_bboxes):
        """
        Update tracker with new detections
        
        Args:
            input_bboxes: List of detected bounding boxes [[x1,y1,x2,y2], ...]
        
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
                self.register(bbox)
        else:
            # Match existing persons with new detections
            person_ids = list(self.persons.keys())
            existing_centroids = [self.centroids[pid] for pid in person_ids]
            existing_bboxes = [self.persons[pid] for pid in person_ids]
            
            # Calculate cost matrix using both distance and IoU
            # Cost = weighted combination of centroid distance and (1 - IoU)
            D = np.zeros((len(existing_centroids), len(input_centroids)))
            
            for i, (ex_centroid, ex_bbox) in enumerate(zip(existing_centroids, existing_bboxes)):
                for j, (in_centroid, in_bbox) in enumerate(zip(input_centroids, input_bboxes)):
                    # Centroid distance (normalized)
                    dist = self._calculate_distance(ex_centroid, in_centroid)
                    
                    # IoU similarity
                    iou = self._calculate_iou(ex_bbox, in_bbox)
                    
                    # Combined cost: 
                    # - Lower distance = lower cost (good)
                    # - Higher IoU = lower cost (good)
                    # Weight: 70% distance, 30% IoU
                    cost = (0.7 * dist) + (0.3 * (1 - iou) * 100)
                    
                    D[i, j] = cost
            
            # Find best matches using Hungarian algorithm (simplified greedy approach)
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
                
                # If cost is reasonable, it's a match
                # Threshold: max_distance for centroid difference
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
            
            # Register unmatched new detections as new persons
            unmatched_inputs = set(range(len(input_bboxes))) - used_cols
            for j in unmatched_inputs:
                self.register(input_bboxes[j])
        
        return self.persons


# ============================================================
# HELPER FUNCTIONS
# ============================================================

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


def associate_parts_with_persons(person_bboxes_dict, part_detections, iou_threshold=0.2):
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


# ============================================================
# MAIN PROCESSING FUNCTION WITH IMPROVED TRACKING
# ============================================================

def process_video_with_stable_tracking(input_video_path, output_video_path, 
                                       behavior_model_path, person_model_path=None):
    """
    Process video with stable person tracking
    
    Args:
        input_video_path: Path to input video
        output_video_path: Path to output video
        behavior_model_path: Path to behavior detection model
        person_model_path: Path to person detection model (optional)
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
    # max_disappeared: how many frames before removing person (30 frames = 1 sec at 30fps)
    # max_distance: maximum pixel distance to consider same person (adjust based on video resolution)
    tracker = PersonTracker(max_disappeared=30, max_distance=150)
    
    person_histories = defaultdict(list)
    person_frame_counts = defaultdict(int)
    
    print("✓ Tracking system initialized")
    print(f"  - Max disappeared frames: 30 (~1 second)")
    print(f"  - Max matching distance: 150 pixels")
    
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
        
        # ===== STEP 1: Detect persons =====
        person_results = person_model(frame, classes=[0], verbose=False)
        
        detected_bboxes = []
        for box in person_results[0].boxes:
            bbox = box.xyxy[0].cpu().numpy()
            detected_bboxes.append(bbox)
        
        # ===== STEP 2: Update tracker (this handles matching automatically) =====
        tracked_persons = tracker.update(detected_bboxes)
        
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
        associations = associate_parts_with_persons(tracked_persons, part_detections)
        
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
            
            # Color based on verdict
            if verdict == "CHEATING":
                color = (0, 0, 255)  # Red
            elif verdict == "SUSPICIOUS":
                color = (0, 165, 255)  # Orange
            elif verdict == "NOT CHEATING":
                color = (0, 255, 0)  # Green
            else:
                color = (128, 128, 128)  # Gray
            
            # Draw person box (thicker for emphasis)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw label with verdict
            label = f"Person {person_id}: {verdict}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Draw label background
            cv2.rectangle(annotated_frame, 
                         (x1, y1 - label_size[1] - 15), 
                         (x1 + label_size[0] + 10, y1), 
                         color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, (x1 + 5, y1 - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw centroid (for debugging)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.circle(annotated_frame, (cx, cy), 5, color, -1)
        
        # Draw behavior detections
        for person_id in associations:
            for detection in associations[person_id]:
                x1, y1, x2, y2 = map(int, detection['bbox'])
                class_name = detection['class']
                conf = detection['conf']
                
                # Draw detection box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                
                # Draw detection label
                det_label = f"{class_name} {conf:.2f}"
                cv2.putText(annotated_frame, det_label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Draw frame info
        info_text = f"Frame: {frame_count} | Active Persons: {len(tracked_persons)}"
        cv2.putText(annotated_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        out.write(annotated_frame)
        
        # Progress
        if frame_count % 30 == 0 or frame_count == total_frames:
            elapsed = time.time() - start_time
            speed = frame_count / elapsed
            eta = (total_frames - frame_count) / speed if speed > 0 else 0
            
            print(f"Frame: {frame_count:5d}/{total_frames} ({frame_count/total_frames*100:5.1f}%) | "
                  f"Speed: {speed:5.1f} fps | ETA: {eta:6.1f}s | "
                  f"Active: {len(tracked_persons)} | Total IDs: {tracker.next_id - 1}")
    
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
    
    print("\n" + "="*70)
    print("PROCESSING COMPLETE!")
    print("="*70)
    print(f"✓ Total frames processed: {frame_count}")
    print(f"✓ Total unique persons: {len(person_histories)}")
    print(f"✓ Total IDs created: {tracker.next_id - 1}")
    print(f"✓ Processing time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"✓ Processing speed: {frame_count/total_time:.2f} fps")
    print(f"✓ Output saved: {output_video_path}")
    
    if os.path.exists(output_video_path):
        file_size = os.path.getsize(output_video_path) / (1024 * 1024)
        print(f"✓ Output file size: {file_size:.2f} MB")
    
    print("="*70)


# ============================================================
# USAGE
# ============================================================

if __name__ == "__main__":
    process_video_with_stable_tracking(
        input_video_path="test_video/test_video1.mp4",
        output_video_path="outputs/output_stable_tracking.mp4",
        behavior_model_path="models/yolov8m_best.pt",
        person_model_path="models/yolov8_human_best.pt"  # Uses YOLOv8n pretrained
    )