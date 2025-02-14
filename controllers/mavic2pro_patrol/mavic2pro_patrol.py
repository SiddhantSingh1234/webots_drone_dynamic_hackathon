from controller import Robot, Camera, Compass, GPS, Gyro, InertialUnit, Keyboard, LED, Motor
from ultralytics import YOLO
import cv2
import numpy as np

# Constants
K_VERTICAL_THRUST = 68.5
K_VERTICAL_OFFSET = 0.6
K_VERTICAL_P = 3.0
K_ROLL_P = 50.0
K_PITCH_P = 30.0
TARGET_ALTITUDE = 1.0

def clamp(value, low, high):
    return max(low, min(value, high))

# YOLOv8

model = YOLO("yolov8n.pt")  # Load YOLOv8 model (smallest version)

def detect_objects(image):
    if image.shape[2] == 4:  # If image has 4 channels (RGBA), convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    results = model(image)  # Run YOLO object detection
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw box
            label = f"{model.names[int(box.cls[0])]}: {box.conf[0]:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# def detect_objects(image):
#     """
#     Detects objects in the given camera image and draws green bounding boxes around them.
    
#     Parameters:
#         image (numpy.ndarray): The image captured by the drone's camera.
    
#     Returns:
#         numpy.ndarray: The processed image with object detection overlays.
#     """
#     # Convert the image from BGRA to BGR
#     image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

#     # Convert image to grayscale and apply thresholding
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

#     # Find contours in the thresholded image
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Draw bounding boxes around detected objects
#     for contour in contours:
#         if cv2.contourArea(contour) > 500:  # Ignore small objects
#             x, y, w, h = cv2.boundingRect(contour)
#             cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

#     return image  # Return the image with detected objects

# # Load YOLO model
# net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
# layer_names = net.getLayerNames()
# output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
# classes = open("coco.names").read().strip().split("\n")  # Load class names

# def detect_objects(image):
#     image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
#     height, width = image.shape[:2]

#     # Convert image to blob and set as input
#     blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
#     net.setInput(blob)

#     # Perform forward pass to get detections
#     detections = net.forward(output_layers)

#     for output in detections:
#         for detection in output:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5:
#                 center_x, center_y, w, h = (detection[:4] * np.array([width, height, width, height])).astype("int")
#                 x, y = int(center_x - w / 2), int(center_y - h / 2)
#                 cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box
#                 cv2.putText(image, f"{classes[class_id]}: {confidence:.2f}", (x, y - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#     return image

# # Edge Detection and Morphology

# def detect_objects(image):
#     image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150)

#     # Use morphological transformations to close gaps in edges
#     kernel = np.ones((3,3), np.uint8)
#     edges = cv2.dilate(edges, kernel, iterations=2)

#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for contour in contours:
#         if cv2.contourArea(contour) > 500:
#             x, y, w, h = cv2.boundingRect(contour)
#             cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     return image

# bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# def detect_objects(image):
#     image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
#     fg_mask = bg_subtractor.apply(image)  # Remove background
#     contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     for contour in contours:
#         if cv2.contourArea(contour) > 500:  # Ignore small objects
#             x, y, w, h = cv2.boundingRect(contour)
#             cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle
#     return image

def main():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    
    # Initialize devices
    camera = robot.getDevice("camera")
    camera.enable(timestep)
    imu = robot.getDevice("inertial unit")
    imu.enable(timestep)
    gps = robot.getDevice("gps")
    gps.enable(timestep)
    compass = robot.getDevice("compass")
    compass.enable(timestep)
    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)
    keyboard = Keyboard()
    keyboard.enable(timestep)
    
    front_left_motor = robot.getDevice("front left propeller")
    front_right_motor = robot.getDevice("front right propeller")
    rear_left_motor = robot.getDevice("rear left propeller")
    rear_right_motor = robot.getDevice("rear right propeller")
    motors = [front_left_motor, front_right_motor, rear_left_motor, rear_right_motor]
    
    for motor in motors:
        motor.setPosition(float('inf'))
        motor.setVelocity(1.0)
    
    print("Drone started. Use keyboard to control.")
    global TARGET_ALTITUDE
    DETECTION_INTERVAL = 100
    counter = 0

    while robot.step(timestep) != -1:
        counter = counter + 1    
        roll, pitch, _ = imu.getRollPitchYaw()
        altitude = gps.getValues()[2]
        roll_velocity, pitch_velocity, _ = gyro.getValues()
        
        key = keyboard.getKey()
        roll_disturbance = pitch_disturbance = yaw_disturbance = 0.0
        
        # while key > 0:
        #     if key == Keyboard.UP:
        #         pitch_disturbance = -3.0  # Increased from -2.0
        #     elif key == Keyboard.DOWN:
        #         pitch_disturbance = 3.0   # Increased from 2.0
        #     elif key == Keyboard.RIGHT:
        #         yaw_disturbance = -2.0    # Increased from -1.3
        #     elif key == Keyboard.LEFT:
        #         yaw_disturbance = 2.0     # Increased from 1.3
        #     elif key == Keyboard.SHIFT + Keyboard.UP:
        #         TARGET_ALTITUDE += 0.1    # Increased from 0.05
        #         print(f"Target Altitude: {TARGET_ALTITUDE}")
        #     elif key == Keyboard.SHIFT + Keyboard.DOWN:
        #         TARGET_ALTITUDE -= 0.1    # Increased from 0.05
        #         print(f"Target Altitude: {TARGET_ALTITUDE}")
        #     key = keyboard.getKey()

        while (key > 0):
            # Reset disturbances before processing new key presses
            roll_disturbance = 0.0
            pitch_disturbance = 0.0
            yaw_disturbance = 0.0

            # Forward and backward movement with smooth pitch control
            if key == Keyboard.UP:
                pitch_disturbance = -2.5  # Reduced for smoother movement
            elif key == Keyboard.DOWN:
                pitch_disturbance = 2.5   # Reduced for smoother movement

            # Left and right yaw control with smoother rotation
            elif key == Keyboard.RIGHT:
                yaw_disturbance = -1.7  # Slightly reduced for stability
            elif key == Keyboard.LEFT:
                yaw_disturbance = 1.7   # Slightly reduced for stability

            # Adjust altitude with SHIFT + UP/DOWN
            elif key == Keyboard.SHIFT + Keyboard.UP:
                TARGET_ALTITUDE += 0.1  # Keep altitude change controlled
                print(f"Target Altitude: {TARGET_ALTITUDE}")
            elif key == Keyboard.SHIFT + Keyboard.DOWN:
                TARGET_ALTITUDE -= 0.1
                print(f"Target Altitude: {TARGET_ALTITUDE}")
            key = keyboard.getKey()

        roll_input = K_ROLL_P * clamp(roll, -1.0, 1.0) + roll_velocity + roll_disturbance
        pitch_input = K_PITCH_P * clamp(pitch, -1.0, 1.0) + pitch_velocity + pitch_disturbance
        yaw_input = yaw_disturbance
        vertical_input = K_VERTICAL_P * pow(clamp(TARGET_ALTITUDE - altitude + K_VERTICAL_OFFSET, -1.0, 1.0), 3)

        front_left_motor_input = K_VERTICAL_THRUST + vertical_input - roll_input + pitch_input - yaw_input
        front_right_motor_input = K_VERTICAL_THRUST + vertical_input + roll_input + pitch_input + yaw_input
        rear_left_motor_input = K_VERTICAL_THRUST + vertical_input - roll_input - pitch_input + yaw_input
        rear_right_motor_input = K_VERTICAL_THRUST + vertical_input + roll_input - pitch_input - yaw_input

        front_left_motor.setVelocity(front_left_motor_input)
        front_right_motor.setVelocity(-front_right_motor_input)
        rear_left_motor.setVelocity(-rear_left_motor_input)
        rear_right_motor.setVelocity(rear_right_motor_input)
        
        # Object detection
        if counter % DETECTION_INTERVAL == 0:
            image = np.frombuffer(camera.getImage(), dtype=np.uint8)
            image = image.reshape((camera.getHeight(), camera.getWidth(), 4))
            processed_image = detect_objects(image)
            cv2.imshow("Object Detection", processed_image)
            cv2.waitKey(1)
    
    robot.cleanup()

if __name__ == "__main__":
    main()