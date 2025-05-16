import cv2
import numpy as np

class EnhancedMarsAnomalyDetector:
    def __init__(self):
        # Updated Mars terrain color parameters for light sandy/beige environment
        self.color_lower_terrain = np.array([0, 0, 160])  # Lighter beige/tan colors
        self.color_upper_terrain = np.array([40, 60, 220]) 
        
        # Edge detection parameters - lowered thresholds for subtle features
        self.canny_low = 30
        self.canny_high = 120
        
        # Size parameters - reduced min size to catch smaller objects
        self.min_object_area = 50   # Lowered from 500
        self.max_object_area = 50000
        
        # Shape and texture parameters
        self.min_edge_density = 0.05  # Lowered to detect subtle edges
        self.max_circularity = 0.95
        
        # Updated rock detection parameters
        self.rock_color_lower = np.array([0, 0, 150])  # Lighter Mars surface
        self.rock_color_upper = np.array([40, 80, 220])
        self.rock_texture_threshold = 20
        self.rock_edge_ratio_max = 0.3
        
        # Enhanced non-Mars color ranges with increased sensitivity
        self.non_mars_colors = [
            # Blue ranges (expanded)
            {'lower': np.array([90, 40, 40]), 'upper': np.array([130, 255, 255])},
            # Green ranges (expanded)
            {'lower': np.array([35, 40, 40]), 'upper': np.array([85, 255, 255])},
            # Purple/Pink ranges
            {'lower': np.array([140, 40, 40]), 'upper': np.array([170, 255, 255])},
            # Bright white (adjusted for light environment)
            {'lower': np.array([0, 0, 210]), 'upper': np.array([180, 30, 255])},
            # Vivid yellow (expanded)
            {'lower': np.array([20, 80, 80]), 'upper': np.array([35, 255, 255])},
            # Cyan
            {'lower': np.array([85, 40, 40]), 'upper': np.array([95, 255, 255])},
            # Red (both lower and upper hue ranges)
            {'lower': np.array([0, 50, 50]), 'upper': np.array([10, 255, 255])},
            {'lower': np.array([170, 50, 50]), 'upper': np.array([180, 255, 255])},
            # Orange
            {'lower': np.array([10, 50, 50]), 'upper': np.array([25, 255, 255])}
        ]
        
        # Additional color contrast parameters
        self.color_contrast_threshold = 30
        
    def enhance_image(self, image):
        """Pre-process image to enhance features"""
        # Convert to lab color space for better color differentiation
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        # Split channels, apply CLAHE to L-channel, then merge
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Sharpen the image
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    
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
            if area < 100:  # Skip very small regions
                continue
                
            # Get bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Ensure ROI is within image bounds
            if x < 0 or y < 0 or x+w > image.shape[1] or y+h > image.shape[0]:
                continue
                
            roi = image[y:y+h, x:x+w]
            if roi.size == 0:
                continue
                
            # Calculate texture using grayscale variance
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            texture_variance = np.var(gray_roi)
            
            # Calculate shape irregularity (rocks are irregular)
            perimeter = cv2.arcLength(cnt, True)
            if area > 0:
                shape_irregularity = perimeter * perimeter / (4 * np.pi * area)
            else:
                shape_irregularity = 0
                
            # Rocks typically have moderate texture and irregular shapes
            if 1.5 < texture_variance < 200 and shape_irregularity > 1.5:
                rock_regions.append(cnt)
                
        return rock_regions
    
    def calculate_color_contrast(self, roi, surrounding):
        """Calculate color contrast between an ROI and its surrounding area"""
        if roi.size == 0 or surrounding.size == 0:
            return 0
            
        roi_mean = np.mean(roi, axis=(0,1))
        surr_mean = np.mean(surrounding, axis=(0,1))
        
        # Calculate Euclidean distance between color means
        contrast = np.sqrt(np.sum((roi_mean - surr_mean)**2))
        return contrast
    
    def detect_anomalies(self, image):
        """
        Enhanced method to detect anomalies using multiple techniques
        """
        # Step 0: Pre-enhance the image
        enhanced_image = self.enhance_image(image)
        
        # Step 1: Color-based detection
        hsv = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2HSV)
        
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
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        anomaly_mask = cv2.morphologyEx(combined_anomaly_mask, cv2.MORPH_OPEN, kernel)
        anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_CLOSE, kernel)
        
        # Step 2: Edge detection
        gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # Dilate edges to connect nearby edges
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Step 3: Detect rocks to exclude them
        rock_regions = self.detect_rocks(enhanced_image, hsv)
        
        # Create rock mask
        rock_mask = np.zeros_like(gray)
        cv2.drawContours(rock_mask, rock_regions, -1, 255, -1)
        
        # Remove rocks from anomaly mask
        anomaly_mask = cv2.bitwise_and(anomaly_mask, cv2.bitwise_not(rock_mask))
        
        # Combine edge detection with color anomalies
        combined_mask = cv2.bitwise_or(anomaly_mask, edges)
        
        # Find contours in combined mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Final verification of anomalies
        verified_anomalies = []
        for cnt in contours:
            # Basic area filtering
            area = cv2.contourArea(cnt)
            if area < self.min_object_area or area > self.max_object_area:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Extend bounding box slightly for surrounding context
            ext_x = max(0, x - 10)
            ext_y = max(0, y - 10)
            ext_w = min(image.shape[1] - ext_x, w + 20)
            ext_h = min(image.shape[0] - ext_y, h + 20)
            
            # Ensure ROI is within image bounds
            if x < 0 or y < 0 or x+w > image.shape[1] or y+h > image.shape[0]:
                continue
            
            roi = enhanced_image[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            
            # Get surrounding region (excluding the ROI)
            surrounding_mask = np.ones((ext_h, ext_w), dtype=np.uint8) * 255
            roi_mask = np.zeros((ext_h, ext_w), dtype=np.uint8)
            roi_mask[y-ext_y:y-ext_y+h, x-ext_x:x-ext_x+w] = 255
            surrounding_mask = cv2.bitwise_xor(surrounding_mask, roi_mask)
            
            surrounding_region = enhanced_image[ext_y:ext_y+ext_h, ext_x:ext_x+ext_w]
            
            # Create a mask for this specific contour
            contour_mask = np.zeros_like(gray)
            cv2.drawContours(contour_mask, [cnt], 0, 255, -1)
            contour_mask = contour_mask[y:y+h, x:x+w]
            
            # Check edge density in ROI
            edge_roi = edges[y:y+h, x:x+w]
            if contour_mask.sum() > 0:
                edge_density = np.sum(edge_roi & contour_mask) / np.sum(contour_mask)
            else:
                edge_density = 0
            
            # Calculate shape features
            perimeter = cv2.arcLength(cnt, True)
            if area > 0:
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
                
            # Calculate edge smoothness
            edge_pixels = cv2.Canny(gray_roi, 50, 150)
            edge_count = np.sum(edge_pixels > 0)
            edge_ratio = edge_count / area if area > 0 else 0
            
            # Calculate color contrast with surrounding
            color_contrast = self.calculate_color_contrast(roi, surrounding_region)
            
            # Check if region contains non-Mars colors
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            non_mars_pixel_count = 0
            total_pixels = roi.shape[0] * roi.shape[1]
            
            for color_range in self.non_mars_colors:
                color_mask_roi = cv2.inRange(hsv_roi, color_range['lower'], color_range['upper'])
                non_mars_pixel_count += np.sum(color_mask_roi > 0)
            
            non_mars_color_percentage = non_mars_pixel_count / total_pixels if total_pixels > 0 else 0
            
            # Flag for non-Mars colors
            has_non_mars_color = non_mars_color_percentage > 0.0005  # More sensitive threshold
                
            # ---- ROCK DETECTION FILTERING ----
            avg_hue = np.mean(hsv_roi[:,:,0])
            avg_saturation = np.mean(hsv_roi[:,:,1])
            
            # Features that indicate a rock:
            is_rock = False
            
            # 1. Natural texture range typical for Mars rocks
            if 3 < texture_complexity < self.rock_texture_threshold:
                is_rock = True
                
            # 2. Color in range of Mars rocks (reddish-brown to beige)
            if 5 < avg_hue < 30 and 10 < avg_saturation < 120:
                is_rock = True
                
            # 3. Irregular shape typical for rocks
            shape_irregularity = perimeter * perimeter / (4 * np.pi * area) if area > 0 else 0
            if shape_irregularity > 1.2:
                is_rock = True
                
            # If it has non-Mars colors, override the rock classification
            if has_non_mars_color:
                is_rock = False
                
            # Skip this object if it seems to be a rock
            if is_rock and not has_non_mars_color and color_contrast < 50:
                continue
            
            # Multi-criteria decision for anomalies
            is_anomaly = False
            anomaly_score = 0  # Use a scoring system for better detection
            
            # Method 1: Non-Mars colors detected
            if has_non_mars_color:
                is_anomaly = True
                anomaly_score += 5
                
            # Method 2: High contrast with surroundings
            if color_contrast > self.color_contrast_threshold:
                is_anomaly = True
                anomaly_score += color_contrast / 10
                
            # Method 3: Edge detection
            if edge_density > self.min_edge_density or edge_ratio > self.rock_edge_ratio_max:
                is_anomaly = True
                anomaly_score += 3
                
            # Method 4: Color variance (uniform color is often artificial)
            if color_variance > 100 or color_variance < 5:
                is_anomaly = True
                anomaly_score += 2
                
            # Method 5: Regular geometric shape
            approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
            vertex_count = len(approx)
            if 2 <= vertex_count <= 8 and circularity > 0.5:
                is_anomaly = True
                anomaly_score += 3
                
            # Method 6: Unusual texture
            if texture_complexity < 2 or texture_complexity > 40:
                is_anomaly = True
                anomaly_score += 2
            
            if is_anomaly or anomaly_score > 5:  # Use score threshold as backup
                verified_anomalies.append({
                    'contour': cnt,
                    'bbox': (x, y, w, h),
                    'area': area,
                    'edge_density': edge_density,
                    'circularity': circularity,
                    'color_variance': color_variance,
                    'has_non_mars_color': has_non_mars_color,
                    'color_contrast': color_contrast,
                    'anomaly_score': anomaly_score
                })
        
        return verified_anomalies, enhanced_image
    
    def visualize_anomalies(self, image, anomalies, enhanced_image=None):
        """
        Visualize detected anomalies on the image
        """
        if enhanced_image is not None:
            output = enhanced_image.copy()
        else:
            output = image.copy()
            
        for anomaly in anomalies:
            x, y, w, h = anomaly['bbox']
            
            # Color coding based on detection confidence
            score = anomaly.get('anomaly_score', 0)
            
            if 'has_non_mars_color' in anomaly and anomaly['has_non_mars_color']:
                # Non-Mars color detections in blue
                color = (255, 0, 0)  # Blue (BGR)
                thickness = 2
                label = f'Color Anomaly ({score:.1f})'
            elif score > 8:
                # High confidence in red
                color = (0, 0, 255)  # Red (BGR)
                thickness = 3
                label = f'Strong Anomaly ({score:.1f})'
            else:
                # Other anomalies in green
                color = (0, 255, 0)  # Green (BGR)
                thickness = 2
                label = f'Anomaly ({score:.1f})'
            
            cv2.rectangle(output, (x, y), (x+w, y+h), color, thickness)
            cv2.putText(output, 
                        label, 
                        (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
            # Draw a highlight dot at center of anomaly
            center_x = x + w//2
            center_y = y + h//2
            cv2.circle(output, (center_x, center_y), 5, color, -1)
            
        return output


def main():
    # Read the image
    image = cv2.imread('real_check.png')
    
    if image is None:
        print("Error: Could not read the image file")
        return
    
    # Create detector
    detector = EnhancedMarsAnomalyDetector()
    
    # Detect anomalies
    anomalies, enhanced_image = detector.detect_anomalies(image)
    
    # Visualize results
    result_image = detector.visualize_anomalies(image, anomalies, enhanced_image)
    
    # Display results
    cv2.imshow('Original Image', image)
    cv2.imshow('Enhanced Image', enhanced_image)
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
    hsv = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2HSV)
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