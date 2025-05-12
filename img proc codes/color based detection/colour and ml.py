import cv2
import numpy as np

class AdvancedMarsAnomalyDetector:
    def __init__(self):
        # Mars terrain color parameters (brownish/reddish tones)
        self.color_lower_terrain = np.array([0, 20, 50])
        self.color_upper_terrain = np.array([30, 150, 200])
        
        # Edge detection parameters
        self.canny_low = 50
        self.canny_high = 150
        
        # Size parameters
        self.min_object_area = 500
        self.max_object_area = 30000
        
        # Shape and texture parameters
        self.min_edge_density = 0.1
        self.max_circularity = 0.9
        
        # Rock detection parameters (for filtering)
        self.rock_color_lower = np.array([0, 10, 40])  # Broader range for Mars rocks
        self.rock_color_upper = np.array([40, 170, 220])
        self.rock_texture_threshold = 15  # Rocks have natural texture variation
        self.rock_edge_ratio_max = 0.3  # Natural rocks have smoother edges
        
        # Non-Mars color ranges (for detecting anomalies by color)
        # These ranges specifically target colors not commonly found on Mars
        self.non_mars_colors = [
            # Blue ranges
            {'lower': np.array([90, 50, 50]), 'upper': np.array([130, 255, 255])},
            # Green ranges
            {'lower': np.array([35, 50, 50]), 'upper': np.array([85, 255, 255])},
            # Purple/Pink ranges
            {'lower': np.array([140, 50, 50]), 'upper': np.array([170, 255, 255])},
            # Bright white (high value, low saturation)
            {'lower': np.array([0, 0, 200]), 'upper': np.array([180, 30, 255])},
            # Vivid yellow
            {'lower': np.array([20, 100, 100]), 'upper': np.array([35, 255, 255])},
            # Cyan
            {'lower': np.array([85, 50, 50]), 'upper': np.array([95, 255, 255])}
        ]
    
    def detect_rocks(self, image, hsv):
        """
        Detect natural Mars rocks to exclude them from anomalies
        """
        # Create rock mask based on typical Mars rock colors
        rock_mask = cv2.inRange(hsv, self.rock_color_lower, self.rock_color_upper)
        
        # Find rock contours
        rock_contours, _ = cv2.findContours(rock_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by texture and shape to identify rocks
        rock_regions = []
        for cnt in rock_contours:
            area = cv2.contourArea(cnt)
            if area < 200:  # Skip very small regions
                continue
                
            # Get bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Ensure ROI is within image bounds
            if x < 0 or y < 0 or x+w > image.shape[1] or y+h > image.shape[0]:
                continue
                
            roi = image[y:y+h, x:x+w]
            if roi.size == 0:
                continue
                
            # Calculate texture using grayscale variance (rocks have natural texture)
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            texture_variance = np.var(gray_roi)
            
            # Calculate shape irregularity (rocks are irregular)
            perimeter = cv2.arcLength(cnt, True)
            if area > 0:  # Prevent division by zero
                shape_irregularity = perimeter * perimeter / (4 * np.pi * area)
            else:
                shape_irregularity = 0
                
            # Rocks typically have moderate texture and irregular shapes
            if 5 < texture_variance < 200 and shape_irregularity > 1.2:
                rock_regions.append(cnt)
                
        return rock_regions
    
    def detect_anomalies(self, image):
        """
        Detect anomalies using multiple techniques while avoiding rock detection
        """
        # Step 1: Color-based detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create a combined mask for non-Mars colors
        non_mars_color_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for color_range in self.non_mars_colors:
            color_mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
            non_mars_color_mask = cv2.bitwise_or(non_mars_color_mask, color_mask)
        
        # Traditional terrain color mask (detects what is regular Mars terrain)
        terrain_mask = cv2.inRange(hsv, self.color_lower_terrain, self.color_upper_terrain)
        terrain_anomaly_mask = cv2.bitwise_not(terrain_mask)
        
        # Combine non-Mars colors with terrain anomalies
        combined_anomaly_mask = cv2.bitwise_or(non_mars_color_mask, terrain_anomaly_mask)
        
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        anomaly_mask = cv2.morphologyEx(combined_anomaly_mask, cv2.MORPH_OPEN, kernel)
        anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_CLOSE, kernel)
        
        # Step 2: Edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        
        # Step 3: Detect rocks to exclude them
        rock_regions = self.detect_rocks(image, hsv)
        
        # Create rock mask
        rock_mask = np.zeros_like(gray)
        cv2.drawContours(rock_mask, rock_regions, -1, 255, -1)
        
        # Remove rocks from anomaly mask
        anomaly_mask = cv2.bitwise_and(anomaly_mask, cv2.bitwise_not(rock_mask))
        
        # Find contours in anomaly mask
        contours, _ = cv2.findContours(anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Final verification of anomalies
        verified_anomalies = []
        for cnt in contours:
            # Basic area filtering
            area = cv2.contourArea(cnt)
            if area < self.min_object_area or area > self.max_object_area:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Ensure ROI is within image bounds
            if x < 0 or y < 0 or x+w > image.shape[1] or y+h > image.shape[0]:
                continue
            
            roi = image[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            
            # Create a mask for this specific contour
            contour_mask = np.zeros_like(gray)
            cv2.drawContours(contour_mask, [cnt], 0, 255, -1)
            contour_mask = contour_mask[y:y+h, x:x+w]
            
            # Check edge density in ROI
            edge_roi = edges[y:y+h, x:x+w]
            if contour_mask.sum() > 0:  # Prevent division by zero
                edge_density = np.sum(edge_roi & contour_mask) / np.sum(contour_mask)
            else:
                edge_density = 0
            
            # Calculate shape features
            perimeter = cv2.arcLength(cnt, True)
            if area > 0:  # Prevent division by zero
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
                
            # Calculate variance in color (high for artificial objects)
            if roi.size > 0:
                color_variance = np.mean(np.var(roi, axis=(0, 1)))
            else:
                color_variance = 0
                
            # Calculate texture complexity using standard deviation of gray levels
            if roi.size > 0:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                texture_complexity = np.std(gray_roi)
            else:
                texture_complexity = 0
                
            # Calculate edge smoothness (artificial objects have smoother edges)
            edge_pixels = cv2.Canny(gray_roi, 50, 150)
            edge_count = np.sum(edge_pixels > 0)
            edge_ratio = edge_count / area if area > 0 else 0
            
            # Check if region contains non-Mars colors
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            has_non_mars_color = False
            
            # Calculate percent of non-Mars colors in the ROI
            non_mars_pixel_count = 0
            total_pixels = roi.shape[0] * roi.shape[1]
            
            for color_range in self.non_mars_colors:
                color_mask_roi = cv2.inRange(hsv_roi, color_range['lower'], color_range['upper'])
                non_mars_pixel_count += np.sum(color_mask_roi > 0)
            
            non_mars_color_percentage = non_mars_pixel_count / total_pixels if total_pixels > 0 else 0
            
            # If significant portion has non-Mars colors, mark as an anomaly immediately
            if non_mars_color_percentage > 0.15:  # If more than 15% has non-Mars colors
                has_non_mars_color = True
                
            # ---- ROCK DETECTION FILTERING ----
            # Check if this might be a rock based on texture and color properties
            avg_hue = np.mean(hsv_roi[:,:,0])
            avg_saturation = np.mean(hsv_roi[:,:,1])
            
            # Features that indicate a rock:
            is_rock = False
            
            # 1. Natural texture range typical for Mars rocks
            if 5 < texture_complexity < self.rock_texture_threshold:
                is_rock = True
                
            # 2. Color in range of Mars rocks (reddish-brown)
            if 5 < avg_hue < 30 and 20 < avg_saturation < 150:
                is_rock = True
                
            # 3. Irregular shape typical for rocks
            shape_irregularity = perimeter * perimeter / (4 * np.pi * area) if area > 0 else 0
            if shape_irregularity > 1.2:
                is_rock = True
                
            # If it has non-Mars colors, override the rock classification
            if has_non_mars_color:
                is_rock = False
                
            # Skip this object if it seems to be a rock
            if is_rock and not has_non_mars_color:
                continue
            
            # Multi-criteria decision for non-rock anomalies
            is_anomaly = False
            
            # Method 1: Non-Mars colors detected
            if has_non_mars_color:
                is_anomaly = True
                
            # Method 2: Sharp edges + regular shape (unlike natural rocks)
            if edge_density > self.min_edge_density and edge_ratio > self.rock_edge_ratio_max:
                is_anomaly = True
                
            # Method 3: High color variance + non-rock texture
            if color_variance > 500 and texture_complexity > 25:
                is_anomaly = True
                
            # Method 4: Regular geometric shape (most natural objects aren't perfectly geometric)
            approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
            vertex_count = len(approx)
            if 4 <= vertex_count <= 8 and circularity < 0.8:  # Regular polygon shape
                is_anomaly = True
                
            # Method 5: Artificial texture patterns
            if texture_complexity < 5 or texture_complexity > 40:  # Either too smooth or too complex
                is_anomaly = True
            
            if is_anomaly:
                verified_anomalies.append({
                    'contour': cnt,
                    'bbox': (x, y, w, h),
                    'area': area,
                    'edge_density': edge_density,
                    'circularity': circularity,
                    'color_variance': color_variance,
                    'has_non_mars_color': has_non_mars_color
                })
        
        return verified_anomalies
    
    def visualize_anomalies(self, image, anomalies):
        """
        Visualize detected anomalies on the image
        """
        output = image.copy()
        for anomaly in anomalies:
            x, y, w, h = anomaly['bbox']
            
            # Use different colors based on detection type
            if 'has_non_mars_color' in anomaly and anomaly['has_non_mars_color']:
                # Non-Mars color detections in blue
                color = (255, 0, 0)  # Blue (BGR)
                label = f'Non-Mars Color'
            else:
                # Other anomalies in green
                color = (0, 255, 0)  # Green (BGR)
                label = f'Anomaly'
            
            cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)
            cv2.putText(output, 
                        label, 
                        (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return output

def main():
    # Read the image
    image = cv2.imread('mars.png')
    
    if image is None:
        print("Error: Could not read the image file 'mars.png'")
        return
    
    # Create detector
    detector = AdvancedMarsAnomalyDetector()
    
    # Detect anomalies
    anomalies = detector.detect_anomalies(image)
    
    # Visualize results
    result_image = detector.visualize_anomalies(image, anomalies)
    
    # Display results
    cv2.imshow('Original Image', image)
    cv2.imshow('Mars Terrain Anomaly Detection', result_image)
    
    # Create a mask to show detected objects
    mask = np.zeros_like(image)
    for anomaly in anomalies:
        x, y, w, h = anomaly['bbox']
        # Use different colors based on detection type
        if 'has_non_mars_color' in anomaly and anomaly['has_non_mars_color']:
            color = (255, 0, 0)  # Blue (BGR)
        else:
            color = (0, 255, 0)  # Green (BGR)
        cv2.rectangle(mask, (x, y), (x+w, y+h), color, -1)
    
    # Create a visual overlay
    overlay = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
    cv2.imshow('Anomaly Overlay', overlay)
    
    # Create a visualization of non-Mars colors
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    non_mars_color_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for color_range in detector.non_mars_colors:
        color_mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
        non_mars_color_mask = cv2.bitwise_or(non_mars_color_mask, color_mask)
    
    color_vis = np.zeros_like(image)
    color_vis[non_mars_color_mask > 0] = [0, 0, 255]  # Red for non-Mars colors
    color_overlay = cv2.addWeighted(image, 0.7, color_vis, 0.3, 0)
    cv2.imshow('Non-Mars Colors', color_overlay)
    
    # Count anomalies by type
    color_anomalies = sum(1 for a in anomalies if 'has_non_mars_color' in a and a['has_non_mars_color'])
    other_anomalies = len(anomalies) - color_anomalies
    
    print(f"Detected {len(anomalies)} anomalies:")
    print(f" - {color_anomalies} based on non-Mars colors")
    print(f" - {other_anomalies} based on other anomaly criteria")
    
    # Wait for key press
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()