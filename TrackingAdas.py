from flask import Flask, render_template, Response
import jetson.inference
import jetson.utils
import numpy as np
import cv2
import math
import argparse
import sys

# Initialize Flask app
app = Flask(__name__)

# Argument parser for object detection parameters
parser = argparse.ArgumentParser(description="Object and lane detection with tracking using Jetson inference and OpenCV.",
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--network", type=str, default="trafficcamnet", help="pre-trained model to load (e.g., trafficcamnet)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")      

detected_objects = []
try:
    args = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

# Load the TrafficCamNet model for object detection
net = jetson.inference.detectNet(args.network, sys.argv, args.threshold)

# Enable Tracking
net.SetTrackingEnabled(True)
net.SetTrackingParams(minFrames=3, dropFrames=15, overlapThreshold=0.5)

# Constants for distance calculation (object detection)
KNOWN_WIDTH = 78  # Known width of the object in inches
FOCAL_LENGTH = 720
PROXIMITY_ALERT_DISTANCE = 20 * 12  # 20 feet in inches

# Global variables for video feed
current_camera_path = "v4l2:///dev/video0"

@app.route('/')
def index():
    return render_template('index8.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def calculate_distance(known_width, focal_length, perceived_width):
    """Calculate distance from camera to object."""
    return (known_width * focal_length) / perceived_width

# Region of Interest Function
def make_coordinates(image, line_parameters, side):
    try:
        slope, intercept = line_parameters
        if abs(slope) > 1e5 or abs(intercept) > 1e5:
            raise ValueError("Slope or intercept is too large")
        if abs(slope) < 1e-5:
            raise ValueError("Slope is too small")
    except (TypeError, ValueError) as e:
        print(f"Unexpected line_parameters: {line_parameters}, error: {str(e)}")
        slope = 0.5 if side == 'left' else -0.5
        intercept = image.shape[0]
    y1 = image.shape[0]
    y2 = int(y1 * (14 / 20))  # Adjust this line to change the length of the line
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2], dtype=int)

def averaged_slope_intercept(image, lines):
    if lines is None:
        return None
    
    left_fit = []
    right_fit = []
    
    MIN_SLOPE = 0.5  # Minimum slope threshold
    MAX_SLOPE = 3.0  # Maximum slope threshold
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            
            if slope < -MIN_SLOPE and slope > -MAX_SLOPE:  # Left lane (negative slope)
                left_fit.append((slope, intercept))
            elif slope > MIN_SLOPE and slope < MAX_SLOPE:  # Right lane (positive slope)
                right_fit.append((slope, intercept))
    
    if left_fit:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average, 'left')
    else:
        left_line = None
    
    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_coordinates(image, right_fit_average, 'right')
    else:
        right_line = None
    
    return np.array([left_line, right_line], dtype=object)

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            if line is not None:
                x1, y1, x2, y2 = line
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 15)
    return line_image

def region_of_interest(image):
    height, width = image.shape[:2]
    triangle = np.array([
       [(360, 540), (850, 530), (440, 205)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def gen_frames():
    """Generate frames from the video feed, process object and lane detection."""
    camera = jetson.utils.videoSource(current_camera_path)

    while True:
        img = camera.Capture()
        detections = net.Detect(img)

        jetson.utils.cudaDeviceSynchronize()

        # Convert frame to a format that OpenCV can process
        frame = jetson.utils.cudaToNumpy(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
       
        # Copy the full frame for later use
        full_frame = frame.copy()

        # Crop the frame to only process the bottom 2/3
        height, width = frame.shape[:2]
        bottom_part_frame = frame[height // 3:, :]  # Crop from the top third down

        # Object Detection and Tracking on the bottom 2/3 of the frame
        for detection in detections:
            ID = detection.ClassID
            class_desc = net.GetClassDesc(ID)
            left, top, right, bottom = int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)

            # Only apply detection if it occurs in the bottom part of the frame
            if top >= height // 3:
                perceived_width = right - left
                distance = calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, perceived_width) / 12  # Convert to feet

                # Draw detection box
                cv2.rectangle(full_frame, (left, top), (right, bottom), (255, 0, 0), 2)
                cv2.putText(full_frame, f"{class_desc}: {distance:.2f} ft", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                # Proximity alert if object is within 20 feet
                if distance <= 20:
                    # Draw red proximity alert box
                    cv2.rectangle(full_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    # Draw warning text
                    cv2.putText(full_frame, "Proximity Alert!", (left, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                # Track the object
                if detection.TrackStatus >= 0:  # actively tracking
                    cv2.putText(full_frame, f"TrackID: {detection.TrackID}", (left, bottom + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.putText(full_frame, f"Tracked for {detection.TrackFrames} frames", (left, bottom + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:  # if tracking was lost
                    cv2.putText(full_frame, f"Tracking lost for object {detection.TrackID}", (left, bottom + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Lane Detection on the bottom 2/3 of the frame
        canny_image = canny(bottom_part_frame)
        cropped_canny = region_of_interest(canny_image)  # Use only the bottom part for lane detection
        lines = cv2.HoughLinesP(cropped_canny, 2, np.pi / 180, 100, np.array([]), minLineLength=5, maxLineGap=10)
        averaged_lines = averaged_slope_intercept(bottom_part_frame, lines)
        line_image = display_lines(frame, averaged_lines)

        # Add lane detection results onto the full frame
        combo_image = cv2.addWeighted(full_frame, 0.8, line_image, 1, 1)

        # Encode the final frame for streaming
        ret, buffer = cv2.imencode('.jpg', combo_image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)

