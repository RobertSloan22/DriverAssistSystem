from flask import Flask, render_template, Response, jsonify, request
import jetson.inference
import jetson.utils
import numpy as np
import cv2
import sys
import argparse
from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, Log

app = Flask(__name__)

# Argument parsing for Jetson Inference
parser = argparse.ArgumentParser(description="Combined Lane and Object Detection",
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="trafficcamnet", help="pre-trained model to load")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")

try:
    args = parser.parse_known_args()[0]
except:
    parser.print_help()
    sys.exit(0)

input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv)

# Load the object detection network
net = jetson.inference.detectNet(args.network, sys.argv, args.threshold)
net.SetTrackingEnabled(True)
net.SetTrackingParams(minFrames=3, dropFrames=15, overlapThreshold=0.5)

# Constants for distance calculation
KNOWN_WIDTH = 78  # in inches
FOCAL_LENGTH = 720  # Adjust based on camera calibration
PROXIMITY_ALERT_DISTANCE = 20 * 12  # 7 feet in inches

# Global variables
detected_objects = []
horizontal_line_position = 0
moving_right = True
speed = 45  # Adjust this value to control the speed of the horizontal line movement
switch_camera_flag = False
current_camera_path = "v4l2:///dev/video0"

# Lane detection class (same as before but modified to accept frames)
class Lane:
    def __init__(self, orig_frame):
        self.orig_frame = orig_frame

        # This will hold an image with the lane lines
        self.lane_line_markings = None
        self.warped_frame = None
        self.transformation_matrix = None
        self.inv_transformation_matrix = None
        self.orig_image_size = self.orig_frame.shape[::-1][1:]

        width = self.orig_image_size[0]
        height = self.orig_image_size[1]

        self.padding = int(0.25 * width)  # padding from side of the image in pixels
        self.roi_points = np.float32([
            [width * 0.4, height * 0.6],  # Adjusted ROI points
            [width * 0.1, height],
            [width * 0.9, height],
            [width * 0.6, height * 0.6]
        ])
        self.desired_roi_points = np.float32([
            [self.padding, 0],  # Top-left corner
            [self.padding, self.orig_image_size[1]],  # Bottom-left corner
            [self.orig_image_size[0] - self.padding, self.orig_image_size[1]],  # Bottom-right corner
            [self.orig_image_size[0] - self.padding, 0]  # Top-right corner
        ])

    def get_line_markings(self):
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(self.orig_frame)

        gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)

        gray_frame = gpu_gray.download()
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        gpu_blurred = cv2.cuda_GpuMat()
        gpu_blurred.upload(blurred_frame)

        canny_detector = cv2.cuda.createCannyEdgeDetector(50, 150)
        gpu_edges = canny_detector.detect(gpu_blurred)

        self.lane_line_markings = gpu_edges.download()

        return self.lane_line_markings

    def perspective_transform(self):
        self.transformation_matrix = cv2.getPerspectiveTransform(self.roi_points, self.desired_roi_points)
        self.inv_transformation_matrix = cv2.getPerspectiveTransform(self.desired_roi_points, self.roi_points)

        self.warped_frame = cv2.warpPerspective(self.lane_line_markings, self.transformation_matrix, self.orig_image_size, flags=cv2.INTER_LINEAR)

        return self.warped_frame

    def detect_lane_lines(self):
        histogram = np.sum(self.warped_frame[self.warped_frame.shape[0] // 2:, :], axis=0)
        midpoint = np.int32(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 9
        window_height = np.int32(self.warped_frame.shape[0] // nwindows)
        nonzero = self.warped_frame.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        minpix = 50

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = self.warped_frame.shape[0] - (window + 1) * window_height
            win_y_high = self.warped_frame.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        if len(leftx) == 0 or len(lefty) == 0 or len(rightx) == 0 or len(righty) == 0:
            return None, None, None  # No lane lines detected

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, self.warped_frame.shape[0] - 1, self.warped_frame.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        return left_fitx, right_fitx, ploty

    def overlay_lane_lines(self, left_fitx, right_fitx, ploty):
        if left_fitx is None or right_fitx is None:
            return self.orig_frame  # Return original frame if no lane lines detected

        warp_zero = np.zeros_like(self.warped_frame).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))]).astype(np.int32)
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        newwarp = cv2.warpPerspective(color_warp, self.inv_transformation_matrix, (self.orig_frame.shape[1], self.orig_frame.shape[0]))

        result = cv2.addWeighted(self.orig_frame, 1, newwarp, 0.6, 0)

        return result

def draw_cuda_line(gpu_frame, start_point, end_point, color, thickness=2):
    """Draw a line using CUDA on a GpuMat frame."""
    frame_cpu = gpu_frame.download()  # Transfer to CPU
    cv2.line(frame_cpu, start_point, end_point, color, thickness)  # Draw on CPU
    gpu_frame.upload(frame_cpu)  # Transfer back to GPU

def adjust_gamma(image, gamma=0.5):
    """Apply gamma correction to adjust brightness."""
    inv_gamma = 0.25 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def update_line_positions(frame_width, frame_height, gpu_frame):
    global horizontal_line_position, moving_right, speed
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
    draw_cuda_line(gpu_frame, start_point_h, end_point_h, (0, 0, 255), 2)

    line_length_v = frame_height // 5
    start_point_v = (frame_width // 2, frame_height // 2 - line_length_v // 2)
    end_point_v = (frame_width // 2, frame_height // 2 + line_length_v // 2)
    draw_cuda_line(gpu_frame, start_point_v, end_point_v, (0, 0, 255), 2)
    
    
def calculate_distance(known_width, focal_length, perceived_width):
    """Calculate distance from camera to object."""
    if perceived_width == 0:
        return float('inf')
    return (known_width * focal_length) / perceived_width
def draw_proximity_alert(proximity_alert, frame, frame_width, frame_height):
    if proximity_alert:
        alert_text = "PROXIMITY ALERT!"
        font_scale = 1.25
        font_thickness = 2
        text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x = (frame_width - text_size[0]) // 2
        text_y = (frame_height + text_size[1]) // 2
        cv2.putText(frame, alert_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)

        rect_start = (text_x - 10, text_y - text_size[1] - 10)
        rect_end = (text_x + text_size[0] + 10, text_y + 10)
        cv2.rectangle(frame, rect_start, rect_end, (0, 0, 255), 2)
@app.route('/')
def index():
    return render_template('index8.html')  # Create a combined index page

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

        # Convert RGBA to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        # Resize the frame for performance
        scale_percent = 50
        frame_width = int(frame.shape[1] * scale_percent / 100)  # Define frame_width
        frame_height = int(frame.shape[0] * scale_percent / 100)  # Define frame_height
        dim = (frame_width, frame_height)
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        lane_frame = frame.copy()
        cuda_mem = jetson.utils.cudaFromNumpy(frame)

        detections = net.Detect(cuda_mem, frame_width, frame_height)
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

            if len(detected_objects) > 10:
                detected_objects.pop(0)

            if distance * 12 <= PROXIMITY_ALERT_DISTANCE:
                proximity_alert = True

            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 3)
            cv2.putText(frame, f"{class_desc}: {distance:.2f} ft", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # CUDA-based GpuMat for further operations
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)

        # Draw moving lines using CUDA
        update_line_positions(frame_width, frame_height, gpu_frame)

        # Download the frame with drawn lines
        frame_with_lines = gpu_frame.download()

        # Continue with lane detection and object detection logic...
        lane_obj = Lane(orig_frame=lane_frame)
        lane_line_markings = lane_obj.get_line_markings()
        lane_obj.perspective_transform()
        left_fitx, right_fitx, ploty = lane_obj.detect_lane_lines()
        frame_with_lane_lines = lane_obj.overlay_lane_lines(left_fitx, right_fitx, ploty)

        # Overlay the lane lines on the frame with object detection
        combined_frame = cv2.addWeighted(frame_with_lane_lines, .5, frame_with_lines, .5, 1)
        draw_proximity_alert(proximity_alert, combined_frame, frame_width, frame_height)  # Pass frame dimensions

        _, buffer = cv2.imencode('.jpg', combined_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3300, debug=False, threaded=True)
