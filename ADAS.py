from flask import Flask, render_template, Response, jsonify, request
import jetson_inference
import jetson_utils
import numpy as np
import cv2
import math
import sys
import argparse
import os
import threading
import time
from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, Log

app = Flask(__name__)

parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.",
                                 formatter_class=argparse.RawTextHelpFormatter,
                                 epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="trafficcamnet", help="pre-trained model to load (e.g., trafficcamnet)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")

detected_objects = []
try:
    args = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

# Create video sources and outputs
input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv)

# Load the TrafficCamNet model
net = detectNet(args.network, sys.argv, args.threshold)

net.SetTrackingEnabled(True)
net.SetTrackingParams(minFrames=3, dropFrames=15, overlapThreshold=0.5)

# Constants for distance calculation
REAL_WIDTH = 78  # Known width of the object in inches (6.5 feet = 78 inches)
KNOWN_WIDTH = 78
FOCAL_LENGTH = 720
PROXIMITY_ALERT_DISTANCE = 7 * 12  # 15 feet in inches

# Global variables for the current camera input path and camera switch flag
current_camera_path = "v4l2:///dev/video0"
switch_camera_flag = False

@app.route('/detected_objects')
def get_detected_objects():
    return jsonify(detected_objects)

def calculate_distance(known_width, focal_length, perceived_width):
    """Calculate distance from camera to object."""
    return (known_width * focal_length) / perceived_width

@app.route('/')
def index():
    return render_template('index8.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/switch_camera', methods=['POST'])
def switch_camera():
    global current_camera_path, switch_camera_flag
    new_camera_path = request.json.get('camera_path')
    if new_camera_path:
        current_camera_path = new_camera_path
        switch_camera_flag = True
        return jsonify({"message": "Camera switched successfully"}), 200
    return jsonify({"message": "Invalid camera path"}), 400
def update_line_positions(frame_width, frame_height, frame):
    global horizontal_line_position, moving_right
    line_length_h = frame_width // 4
    if moving_right:
        horizontal_line_position += speed
        if horizontal_line_position + line_length_h >= frame_width:
            moving_right = False
    else:
        horizontal_line_position -= speed
        if horizontal_line_position - line_length_h <= 0:
            moving_right = True

    start_point_h = (horizontal_line_position - line_length_h, frame_height // 2)
    end_point_h = (horizontal_line_position + line_length_h, frame_height // 2)
    cv2.line(frame, start_point_h, end_point_h, (0, 255, 0), 2)

    line_length_v = frame_height // 5
    start_point_v = (frame_width // 2, frame_height // 2 - line_length_v // 2)
    end_point_v = (frame_width // 2, frame_height // 2 + line_length_v // 2)
    cv2.line(frame, start_point_v, end_point_v, (0, 255, 0), 2)

def draw_proximity_alert(proximity_alert, frame, frame_width, frame_height):
    if proximity_alert:
        alert_text = "PROXIMITY ALERT!"
        font_scale = 0.75
        font_thickness = 2
        text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x = (frame_width - text_size[0]) // 2
        text_y = (frame_height + text_size[1]) // 2
        cv2.putText(frame, alert_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)

        rect_start = (text_x - 10, text_y - text_size[1] - 10)
        rect_end = (text_x + text_size[0] + 10, text_y + 10)
        cv2.rectangle(frame, rect_start, rect_end, (0, 0, 255), 2)
horizontal_line_position = 0
moving_right = True
speed = 30  # Adjust this value to control the speed of the horizontal line movement

def gen_frames():
    global horizontal_line_position, moving_right, current_camera_path, switch_camera_flag
    camera = jetson.utils.videoSource(current_camera_path)
    while True:
        if switch_camera_flag:
            camera = jetson.utils.videoSource(current_camera_path)
            switch_camera_flag = False

        img = camera.Capture()
        detections = net.Detect(img)

        jetson.utils.cudaDeviceSynchronize()

        frame = jetson.utils.cudaToNumpy(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        scale_percent = 50  # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        proximity_alert = False
        distances = []

        for detection in detections:
            ID = detection.ClassID
            class_desc = net.GetClassDesc(ID)
            top = int(detection.Top * scale_percent / 100)
            left = int(detection.Left * scale_percent / 100)
            bottom = int(detection.Bottom * scale_percent / 100)
            right = int(detection.Right * scale_percent / 100)

            perceived_width = right - left
            distance = calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, perceived_width) / 12  # Convert inches to feet
            distances.append((distance, left, top, ID, class_desc))
            detected_objects.append({'id': ID, 'class': class_desc, 'distance': distance, 'position': (left, top)})
            if len(detected_objects) > 4:
                detected_objects.pop(0)

            if distance * 12 <= PROXIMITY_ALERT_DISTANCE:
                proximity_alert = True

        distances.sort(key=lambda x: x[0])
        closest_distances = distances[:3]

        # Sort distances by their y-coordinate (top position) to avoid overlapping text
        closest_distances.sort(key=lambda x: x[2])

        # Display the distances ensuring no overlap
        last_y_position = 0
        text_spacing = 20  # Minimum space between lines of text

        for (distance, left, top, ID, class_desc) in closest_distances:
            y_position = max(top - 10, last_y_position + text_spacing)
            cv2.putText(resized_frame, f"{class_desc}: {distance:.2f} ft", (left, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1) 
            last_y_position = y_position

        frame_height, frame_width, _ = resized_frame.shape
        center_x = frame_width // 2
        center_y = frame_height // 2
        update_line_positions(frame_width, frame_height, resized_frame)

        draw_proximity_alert(proximity_alert, resized_frame, frame_width, frame_height)

        _, buffer = cv2.imencode('.jpg', resized_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3300, threaded=True)
