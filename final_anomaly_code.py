import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np
from functools import partial
import os
import time

class AnomalyDetectorNode(Node):
    """
    A ROS2 node that detects anomalies from multiple camera feeds.
    When an anomaly is detected, it creates a circular suppression zone around the
    rover's current position, pausing detection from that specific camera until
    the rover has moved outside the zone.
    """
    def __init__(self):
        super().__init__('anomaly_detector_node')

        # --- Parameters ---
        self.declare_parameter('suppression_radius', 5.0)  # The 'x' meters for the zone
        self.suppression_radius = self.get_parameter('suppression_radius').get_parameter_value().double_value
        self.get_logger().info(f"Suppression radius set to: {self.suppression_radius} meters")

        # --- CV and ROS Tools ---
        self.bridge = CvBridge()

        # --- State Management ---
        # IMPORTANT: Update these topic names to match your rover's camera topics
        self.camera_topics = {
            'front_cam': '/front_cam/zed_node/rgb/image_rect_color',
            'rear_cam': '/rear_cam/zed_node/rgb/image_rect_color',
            'left_cam': '/left_cam/zed_node/rgb/image_rect_color',
            'right_cam': '/right_cam/zed_node/rgb/image_rect_color'
        }
        self.detection_suppressed = {cam: False for cam in self.camera_topics}
        self.last_detection_pos = {cam: None for cam in self.camera_topics}
        self.current_position = None

        # --- Subscriptions ---
        # Odometry subscription to track rover position
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/odom',  # Standard odometry topic
            self.odometry_callback,
            10)

        # Create a subscription for each camera
        for camera_name, topic in self.camera_topics.items():
            self.create_subscription(
                Image,
                topic,
                partial(self.image_callback, camera_name=camera_name),
                10)
            self.get_logger().info(f"Subscribed to {camera_name} on topic {topic}")

        # --- Image Saving ---
        self.output_dir = "anomaly_images"
        os.makedirs(self.output_dir, exist_ok=True)
        self.get_logger().info(f"Saving detected anomaly images to: {os.path.abspath(self.output_dir)}")

    def odometry_callback(self, msg):
        """Stores the rover's current position from the /odom topic."""
        self.current_position = msg.pose.pose.position

    def image_callback(self, msg, camera_name):
        """Processes images from a specific camera and handles detection logic."""
        # 1. Check if detection for this camera is currently suppressed
        if self.detection_suppressed[camera_name]:
            if self.current_position and self.last_detection_pos[camera_name]:
                last_pos = self.last_detection_pos[camera_name]
                dist = np.sqrt((self.current_position.x - last_pos.x)**2 + (self.current_position.y - last_pos.y)**2)

                if dist > self.suppression_radius:
                    self.get_logger().info(f"Rover has left suppression zone for {camera_name}. Resuming detection.")
                    self.detection_suppressed[camera_name] = False
                    self.last_detection_pos[camera_name] = None
                else:
                    # Still inside the zone, so skip processing this frame
                    return
            else:
                # Can't check distance without position data, so we wait
                return

        # 2. Perform Anomaly Detection
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # NOTE: Calibrate these HSV values for the specific Martian terrain color
        # This example targets a reddish-brown range.
        lower_mars_hsv = np.array([0, 70, 50])
        upper_mars_hsv = np.array([20, 255, 255])

        # Create a mask for the "normal" Mars terrain color
        mars_mask = cv2.inRange(hsv, lower_mars_hsv, upper_mars_hsv)
        
        # Invert the mask to find everything that is NOT the terrain color
        anomaly_mask = cv2.bitwise_not(mars_mask)

        # Clean up the mask using morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        processed_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, kernel)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        anomaly_found_in_frame = False
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:  # Contour area threshold
                anomaly_found_in_frame = True
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, 'ANOMALY', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 3. If an anomaly was found, trigger the suppression logic
        if anomaly_found_in_frame:
            self.get_logger().info(f"ANOMALY DETECTED by {camera_name}!")
            
            if self.current_position:
                self.get_logger().info(f"Pausing detection for {camera_name}. Suppression zone of {self.suppression_radius}m activated.")
                self.detection_suppressed[camera_name] = True
                self.last_detection_pos[camera_name] = self.current_position

                # Save the image of the detected anomaly
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = os.path.join(self.output_dir, f"{camera_name}_anomaly_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                self.get_logger().info(f"Saved anomaly image to {filename}")
            else:
                self.get_logger().warn("Anomaly detected, but no odometry data available to set suppression zone.")
        
        # Display the processed frame for debugging
        cv2.imshow(f"Output - {camera_name}", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = AnomalyDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

