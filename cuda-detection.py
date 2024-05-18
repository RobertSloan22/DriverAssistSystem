from flask import Flask, render_template, Response, jsonify
import jetson.inference
import jetson.utils
import numpy as np
import cv2
import math
from jetson_inference import detectNet
import sys
import argparse
from jetson_utils import videoSource, videoOutput, Log

app = Flask(__name__)

# Parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.",
                                 formatter_class=argparse.RawTextHelpFormatter,
                                 epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
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

# Load the object detection network
net = detectNet(args.network, sys.argv, args.threshold)

# Constants for distance calculation
REAL_WIDTH = 78  # Known width of the object in inches (6.5 feet = 78 inches)
KNOWN_WIDTH = 78
FOCAL_LENGTH = 720
PROXIMITY_ALERT_DISTANCE = 15 * 12  # 15 feet in inches

@app.route('/detected_objects')
def get_detected_objects():
    return jsonify(detected_objects)

def calculate_distance(known_width, focal_length, perceived_width):
    """Calculate distance from camera to object."""
    return (known_width * focal_length) / perceived_width

# Flask route to serve the main page
@app.route('/')
def index():
    # Ensure 'index5.html' is in the templates directory
    return render_template('index7.html')

# Flask route to serve video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Initialize horizontal line position and movement direction
# Initialize horizontal line position and movement direction
horizontal_line_position = 0
moving_right = True
speed = 45  # Adjust this value to control the speed of the horizontal line movement

def gen_frames():
    global horizontal_line_position, moving_right

    # Create a video source object for the USB camera (adjust the device number if needed)
    camera = jetson.utils.videoSource("v4l2:///dev/video0")  # Update device path if different

    while True:
        img = camera.Capture()
        detections = net.Detect(img)
        # Convert the image to an array format that can be used with OpenCV
        frame = jetson.utils.cudaToNumpy(img)

        proximity_alert = False
        distances = []  # Store distances for later drawing
        annotations = []  # Store annotations for later drawing

        # Collect distances and annotations
        for detection in detections:
            ID = detection.ClassID
            top = int(detection.Top)
            left = int(detection.Left)
            bottom = int(detection.Bottom)
            right = int(detection.Right)
            jetson.utils.cudaDrawRect(img, (left, top, right, bottom), (255, 0, 0, 150))

            # Calculate the perceived width of the detected object
            perceived_width = right - left
            # Calculate the distance to the detected object in feet
            distance = calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, perceived_width) / 12  # Convert inches to feet
            distances.append((distance, left, top, ID))  # Store distance, position, and class ID
            detected_objects.append({'id': ID, 'distance': distance, 'position': (left, top)})
            if len(detected_objects) > 4:
                detected_objects.pop(0)

            # Check for proximity alert
            if distance * 12 <= PROXIMITY_ALERT_DISTANCE:  # Convert feet back to inches for the alert check
                proximity_alert = True

        # Flip the frame vertically and horizontally
        frame = cv2.flip(frame, -1)  # -1 means flipping around both axes

        # Draw distances and annotations on the flipped frame
        for (distance, left, top, ID) in distances:
            cv2.putText(frame, f"Distance: {distance:.2f} ft", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        for (text, left, bottom) in annotations:
            cv2.putText(frame, text, (left, bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw the horizon marker
        frame_height, frame_width, _ = frame.shape
        center_x = frame_width // 2
        center_y = frame_height // 2
        line_length_h = frame_width // 4  # 25% in each direction, total 50%

        # Update the horizontal line position
        if moving_right:
            horizontal_line_position += speed
            if horizontal_line_position + line_length_h >= frame_width:
                moving_right = False
        else:
            horizontal_line_position -= speed
            if horizontal_line_position - line_length_h <= 0:
                moving_right = True

        start_point_h = (horizontal_line_position - line_length_h, center_y)
        end_point_h = (horizontal_line_position + line_length_h, center_y)
        cv2.line(frame, start_point_h, end_point_h, (0, 255, 0), 4)  # Green horizontal line

        # Draw the vertical line
        line_length_v = frame_height // 5  # 40% of the height, evenly centered
        start_point_v = (center_x, center_y - line_length_v // 2)
        end_point_v = (center_x, center_y + line_length_v // 2)
        cv2.line(frame, start_point_v, end_point_v, (0, 255, 0), 4)  # Green vertical line

        # Draw proximity alert if needed
        if proximity_alert:
            alert_text = "PROXIMITY ALERT!"
            font_scale = 1.5
            font_thickness = 4
            text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] + text_size[1]) // 2
            cv2.putText(frame, alert_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)

            # Draw the red rectangle around the text
            rect_start = (text_x - 10, text_y - text_size[1] - 10)
            rect_end = (text_x + text_size[0] + 10, text_y + 10)
            cv2.rectangle(frame, rect_start, rect_end, (0, 0, 255), 2)

        # Encode the frame in JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        # Yield the frame data
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, threaded=True)
