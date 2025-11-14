from ultralytics import YOLO
import cv2

model = YOLO('models/yolov8m_best.pt')

results = model('test_image/test2.png')

for result in results:
    # Show image with detections
    result.show()
    
    # Print detection details
    print(f"Boxes: {result.boxes}")
    print(f"Class names: {result.names}")
    
    # Save result
    result.save('test_image/output2.jpg')
