from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import time

def process_video(input_video_path, output_video_path, model_path):
    
    print("="*60)
    print("VIDEO PROCESSING STARTED")
    print("="*60)
    
    # Check 1: Verify input video exists
    print("\n[1/7] Checking input video...")
    if not os.path.exists(input_video_path):
        print(f"❌ Error: Input video not found at '{input_video_path}'")
        exit()
    print(f"✓ Input video found: {input_video_path}")
    
    # Check 2: Verify model exists
    print("\n[2/7] Checking model...")
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at '{model_path}'")
        exit()
    print(f"✓ Model found: {model_path}")
    
    # Check 3: Load YOLO model
    print("\n[3/7] Loading YOLO model...")
    try:
        model = YOLO(model_path)
        print(f"✓ Model loaded successfully")
        print(f"  Model classes: {list(model.names.values())}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        exit()
    
    # Check 4: Open video capture
    print("\n[4/7] Opening video file...")
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print("❌ Error: Could not open video file")
        print("Possible reasons:")
        print("  - File is corrupted")
        print("  - Codec not supported")
        print("  - File path is incorrect")
        exit()
    print("✓ Video opened successfully")
    
    # Check 5: Get and verify video properties
    print("\n[5/7] Reading video properties...")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Verify properties are valid
    if frame_width <= 0 or frame_height <= 0 or fps <= 0:
        print("❌ Error: Invalid video properties")
        print(f"  Width: {frame_width}, Height: {frame_height}, FPS: {fps}")
        cap.release()
        exit()
    
    print("✓ Video properties:")
    print(f"  Resolution: {frame_width} x {frame_height}")
    print(f"  FPS: {fps}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")
    
    # Check 6: Create output directory and video writer
    print("\n[6/7] Setting up output...")
    output_dir = os.path.dirname(output_video_path)
    
    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ Created output directory: {output_dir}")
    else:
        print(f"✓ Output directory exists: {output_dir if output_dir else 'current directory'}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Verify video writer is opened
    if not out.isOpened():
        print("❌ Error: Could not create output video writer")
        print("Possible reasons:")
        print("  - Invalid output path")
        print("  - Codec not available")
        print("  - Insufficient permissions")
        cap.release()
        exit()
    print(f"✓ Output video writer created: {output_video_path}")
    
    # Check 7: Process video frames
    print("\n[7/7] Processing video frames...")
    print("-"*60)
    
    frame_count = 0
    detection_count = 0
    start_time = time.time()
    
    # Test first frame to ensure everything works
    ret, test_frame = cap.read()
    if not ret:
        print("❌ Error: Could not read first frame")
        cap.release()
        out.release()
        exit()
    
    print("✓ First frame read successfully")
    print("Running detection on first frame...")
    
    try:
        test_results = model(test_frame, verbose=False)
        test_annotated = test_results[0].plot()
        out.write(test_annotated)
        num_detections = len(test_results[0].boxes)
        print(f"✓ First frame processed: {num_detections} objects detected")
        frame_count = 1
        detection_count += num_detections
    except Exception as e:
        print(f"❌ Error processing first frame: {e}")
        cap.release()
        out.release()
        exit()
    
    print("\nProcessing remaining frames...")
    print("-"*60)

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1

        try:
            results = model(frame, verbose=False)
            annotated_frame = results[0].plot()
            
            # Count detections
            num_detections = len(results[0].boxes)
            detection_count += num_detections
            
            out.write(annotated_frame)
            
        except Exception as e:
            print(f"⚠️  Warning: Error processing frame {frame_count}: {e}")
            continue
        
        # Progress indicator every 30 frames
        if frame_count % 30 == 0:
            elapsed_time = time.time() - start_time
            fps_processing = frame_count / elapsed_time
            progress = (frame_count / total_frames) * 100
            eta = (total_frames - frame_count) / fps_processing if fps_processing > 0 else 0
            avg_detections = detection_count / frame_count
            
            print(f"Frame: {frame_count:5d}/{total_frames} ({progress:5.1f}%) | "
                  f"Speed: {fps_processing:5.1f} fps | "
                  f"ETA: {eta:6.1f}s | "
                  f"Avg detections: {avg_detections:.1f}")
    
    # Cleanup
    cap.release()
    out.release()
    
    # Final statistics
    total_time = time.time() - start_time
    
    print("-"*60)
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print(f"✓ Total frames processed: {frame_count}")
    print(f"✓ Total detections: {detection_count}")
    print(f"✓ Average detections per frame: {detection_count/frame_count:.2f}")
    print(f"✓ Processing time: {total_time:.2f} seconds")
    print(f"✓ Processing speed: {frame_count/total_time:.2f} fps")
    print(f"✓ Output video saved to: {output_video_path}")
    
    # Verify output file exists and has size
    if os.path.exists(output_video_path):
        file_size = os.path.getsize(output_video_path) / (1024 * 1024)  # MB
        print(f"✓ Output file size: {file_size:.2f} MB")
    else:
        print("⚠️  Warning: Output file not found after processing")
    
    print("="*60)

# Usage        
input_video_path = "test_video/test_video1.mp4"
output_video_path = "outputs/output_video.mp4"
model_path = "models/best.pt"

process_video(input_video_path, output_video_path, model_path)