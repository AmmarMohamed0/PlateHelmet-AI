# PlateHelmet-AI

## Introduction
PlateHelmet-AI is an AI-powered system designed to detect helmet violations and recognize vehicle license plates in real-time from video feeds. This project leverages deep learning and computer vision techniques to enhance road safety and enforce traffic rules efficiently. The application detects individuals without helmets and identifies their vehicle license plates for further action, streamlining the process of monitoring traffic violations.

## Features
- Real-time helmet violation detection.
- License plate recognition using OCR (Optical Character Recognition).
- Integration with YOLOv11 for object detection and tracking.
- Automatic data logging of violations with date, time, and license plate information.
- Exportable violation records in an Excel file.
- Easy visualization with bounding boxes and alerts in the video feed.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AmmarMohamed0/PlateHelmet-AI.git
   cd PlateHelmet-AI
   ```
3. Download the pre-trained YOLOv11 model and place it in the root directory
4. Ensure you have PaddleOCR installed and properly configured.
5. Install Excel automation library `xlwings`:
   ```bash
   pip install xlwings
   ```

## Usage
1. Place your input video file (e.g., `sample.mp4`) in the project directory.
2. Run the main script:
   ```bash
   python main.py
   ```
3. During execution:
   - Detected violations will trigger alerts on the video feed.
   - Cropped images of license plates will be saved in a date-specific folder.
   - Violation records will be logged in an Excel file within the same folder.
4. Press `q` to stop the application.

### Code Snippet Example
```python
# Perform OCR on a cropped license plate image
text = perform_ocr(cropped_image)
print(f"Detected Number Plate: {text}")
```

## Technologies
- **Python**: Programming language.
- **OpenCV**: For video processing and image manipulation.
- **YOLOv11**: Deep learning model for object detection and tracking.
- **PaddleOCR**: For text recognition in license plates.
- **cvzone**: For advanced visualization and annotations.
- **xlwings**: For automating Excel file handling.
