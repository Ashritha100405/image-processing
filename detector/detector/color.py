import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ColorDetector(Node):
    def __init__(self):
        super().__init__('color_detector')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/front_cam/zed_node/rgb/image_rect_color',
            self.listener_callback,
            10)
        self.get_logger().info("Color detector node started")
        cv2.namedWindow("Tracking")
        cv2.createTrackbar("Lower Hue", "Tracking", 0, 255, lambda x: None)
        cv2.createTrackbar("Lower Sat", "Tracking", 0, 255, lambda x: None)
        cv2.createTrackbar("Lower Val", "Tracking", 0, 255, lambda x: None)
        cv2.createTrackbar("Upper Hue", "Tracking", 255, 255, lambda x: None)
        cv2.createTrackbar("Upper Sat", "Tracking", 255, 255, lambda x: None)
        cv2.createTrackbar("Upper Val", "Tracking", 255, 255, lambda x: None)
        
    def listener_callback(self, msg):
        self.get_logger().info("Received image frame")
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        l_h = cv2.getTrackbarPos("Lower Hue", "Tracking")
        l_s = cv2.getTrackbarPos("Lower Sat", "Tracking")
        l_v = cv2.getTrackbarPos("Lower Val", "Tracking")
        u_h = cv2.getTrackbarPos("Upper Hue", "Tracking")
        u_s = cv2.getTrackbarPos("Upper Sat", "Tracking")
        u_v = cv2.getTrackbarPos("Upper Val", "Tracking")
        
        lower_bound = np.array([l_h, l_s, l_v])
        upper_bound = np.array([u_h, u_s, u_v])
        
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        mask = cv2.bitwise_not(mask)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.get_logger().info(f"Contours detected: {len(contours)}")
        for cnt in contours:
            area = cv2.contourArea(cnt)
            self.get_logger().info(f"Contour area: {area}")
            if area > 300:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, 'Non-Mars Object', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow("Input Image", frame)
        cv2.imshow("Mask", mask)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ColorDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()