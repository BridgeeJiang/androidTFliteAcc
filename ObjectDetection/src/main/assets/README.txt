Object Detection Model Requirements
===================================

This application requires a YOLOv8 TensorFlow Lite model file named "yolov8n.tflite".

To obtain the model:

1. Download YOLOv8n model from Ultralytics:
   - Visit: https://github.com/ultralytics/ultralytics
   - Or use: pip install ultralytics && yolo export model=yolov8n.pt format=tflite

2. Convert to TensorFlow Lite format if needed:
   - The model should have input shape [1, 640, 640, 3] (float32)
   - The model should have output shape [1, 8400, 84] (float32)

3. Place the "yolov8n.tflite" file in this assets directory

4. The coco_labels.txt file contains the 80 COCO class labels that correspond to the model output

Alternative models:
- You can use other YOLO models (yolov8s, yolov8m, yolov8l, yolov8x) by updating the gradle.properties file
- Make sure the model follows the same input/output format as YOLOv8

For testing purposes, you can also use any compatible object detection model that:
- Takes 640x640x3 RGB input
- Outputs detection results in YOLO format