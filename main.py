import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
from paddleocr import PaddleOCR
import os
from datetime import datetime
import xlwings as xw

# Initialize PaddleOCR for text recognition
ocr = PaddleOCR()

# Function to perform OCR on an image array
def perform_ocr(image_array):
    if image_array is None:
        raise ValueError("Image is None")
    
    # Perform OCR on the image array with text recognition enabled
    results = ocr.ocr(image_array, rec=True) # rec = recognition
    detected_text = []  # List to store detected text

    # Process OCR results if they are not None
    if results[0] is not None:
        for result in results[0]:
            text = result[1][0]  # Extract the detected text
            detected_text.append(text)  # Add the text to the list

    # Join all detected texts into a single string and return it
    return ''.join(detected_text)

# Load the YOLOv11 model from the specified file
model = YOLO("sample.pt")
names = model.names  # Get the class names from the model

# Define a polygon area for region of interest (ROI)
area = [(1, 173), (62, 468), (608, 431), (364, 155)]

# Create a directory for the current date to save results
current_date = datetime.now().strftime('%Y-%m-%d')
if not os.path.exists(current_date):
    os.makedirs(current_date)  # Create the directory if it doesn't exist

# Initialize the Excel file path in the current date folder
excel_file_path = os.path.join(current_date, f"{current_date}.xlsx")

# Open the Excel file using xlwings, or create a new one if it doesn't exist
wb = xw.Book(excel_file_path) if os.path.exists(excel_file_path) else xw.Book()
ws = wb.sheets[0]  # Get the first sheet in the workbook

# Write headers in the Excel file if they are not already present
if ws.range("A1").value is None:
    ws.range("A1").value = ["Number Plate", "Date", "Time"]

# Track processed track IDs to avoid duplicate processing
processed_track_ids = set()

# Open the video file or webcam for processing
cap = cv2.VideoCapture('final.mp4')

# Define colors for bounding boxes (BGR format)
colors = {
    'no-helmet': (0, 0, 255),  # Red for no-helmet
    'numberplate': (0, 255, 0),  # Green for numberplate
}

# Main loop to process each frame of the video
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:  # Break the loop if no frame is read
        break
    
    # Resize the frame to a fixed size (1020x500)
    frame = cv2.resize(frame, (1020, 500))
    
    # Run YOLOv8 tracking on the frame
    results = model.track(frame, persist=True)
    
    # Initialize flags and variables for detection
    no_helmet_detected = False  # Flag for no-helmet detection
    numberplate_box = None  # Bounding box for number plate
    numberplate_track_id = None  # Track ID for number plate
    
    # Check if there are any detected boxes in the results
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Get the bounding boxes, class IDs, track IDs, and confidence scores
        boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
        class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
        track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
        confidences = results[0].boxes.conf.cpu().tolist()  # Confidence scores
        
        # Loop through each detected object
        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = names[class_id]  # Get the class name (e.g., 'no-helmet', 'numberplate')
            x1, y1, x2, y2 = box  # Get the bounding box coordinates
            cx = (x1 + x2) // 2  # Calculate the center x-coordinate
            cy = (y1 + y2) // 2  # Calculate the center y-coordinate
            
            # Check if the center of the bounding box is inside the polygon area
            result = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
            if result >= 0:  # If inside the polygon
                if c == 'no-helmet':
                    no_helmet_detected = True  # Mark that no-helmet is detected
                elif c == 'numberplate':
                    numberplate_box = box  # Store the numberplate bounding box
                    numberplate_track_id = track_id  # Store the track ID for the numberplate
            
            # Draw bounding box and label for each detected object
            color = colors.get(c, (255, 0, 255))  # Default to white if class color is not defined
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Draw bounding box
            label = f"{c} {conf:.2f}"  # Create label with class name and confidence score
            cvzone.putTextRect(frame, label, (x1, y1 - 10), scale=1, thickness=2, colorR=color)  # Display label
        
        # If both no-helmet and numberplate are detected and the track ID is not already processed
        if no_helmet_detected and numberplate_box is not None and numberplate_track_id not in processed_track_ids:
            x1, y1, x2, y2 = numberplate_box  # Get the bounding box coordinates
            crop = frame[y1:y2, x1:x2]  # Crop the numberplate region from the frame
            crop = cv2.resize(crop, (120, 85))  # Resize the cropped image
            
            # Perform OCR on the cropped image to detect the number plate text
            text = perform_ocr(crop)
            print(f"Detected Number Plate: {text}")
            
            # Save the cropped image with the current time as the filename
            current_time = datetime.now().strftime('%H-%M-%S')
            crop_image_path = os.path.join(current_date, f"{text}_{current_time}.jpg")
            cv2.imwrite(crop_image_path, crop)
            
            # Save the detected number plate, date, and time to the Excel file
            last_row = ws.range("A" + str(ws.cells.last_cell.row)).end('up').row
            ws.range(f"A{last_row+1}").value = [text, current_date, current_time]
            
            # Add the track ID to the processed set to avoid duplicate processing
            processed_track_ids.add(numberplate_track_id)
    
    # Draw the polygon area on the frame
    cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 255), 2)
    
    # Display an alarm message on the frame if a violation is detected
    if no_helmet_detected:
        cvzone.putTextRect(frame, "VIOLATION: NO HELMET DETECTED!", (50, 50), scale=2, thickness=3, colorR=(0, 0, 255))
    
    # Display the frame in the 'RGB' window
    cv2.imshow("RGB", frame)
    
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Save and close the Excel workbook
wb.save(excel_file_path)
wb.close()