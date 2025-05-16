""" This code detects the objects based on color"""

# Import the necessary libraries
import cv2
import numpy as np

# Create a dummy function. It is required for cv2.createTrackbar()
def do_nothing(x):
    pass

# Create a named window to create trackbars
cv2.namedWindow("Tracking")

# Create trackbars to adjust the cut-off color range
cv2.createTrackbar("Lower Hue", "Tracking", 0, 255, do_nothing)
cv2.createTrackbar("Lower Sat", "Tracking", 0, 255, do_nothing)
cv2.createTrackbar("Lower Val", "Tracking", 0, 255, do_nothing)
cv2.createTrackbar("Upper Hue", "Tracking", 255, 255, do_nothing)
cv2.createTrackbar("Upper Sat", "Tracking", 255, 255, do_nothing)
cv2.createTrackbar("Upper Val", "Tracking", 255, 255, do_nothing)

#cap = cv2.VideoCapture(0)

while True:
    # Read the image
    #isTrue,img = cap.read()
    img = cv2.imread('real_check.png')
    # Resize the image
    #img = cv2.resize(img, None, fx=0.24,fy=0.24)
    
    # Convert BGR image to HSV color
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Get the track bars to get lower & upper 
    # HSV (actually BGR) color values.
    l_h = cv2.getTrackbarPos("Lower Hue", "Tracking")
    l_s = cv2.getTrackbarPos("Lower Sat", "Tracking")
    l_v = cv2.getTrackbarPos("Lower Val", "Tracking")
    u_h = cv2.getTrackbarPos("Upper Hue", "Tracking")
    u_s = cv2.getTrackbarPos("Upper Sat", "Tracking")
    u_v = cv2.getTrackbarPos("Upper Val", "Tracking")
    
    # Convert the HSV (BGR) color values to array
    lower_bound = np.array([10, 50, 100])
    upper_bound = np.array([25, 255, 255])
    
    # Create the mask 
    # The function gives the values within the
    # lower_bound & upper_bound range
    brown_mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask = cv2.bitwise_not(brown_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)


    
    # Perform the bitwise AND of image with itself
    # & using the mask of cut-off colors
    #result = cv2.bitwise_and(img, img, mask=mask)
    # Find contours in non-background regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 300:  # Filter small noise
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, 'Non-Mars Object', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    
    # Display all the images
    cv2.imshow("Input Image", img)
    cv2.imshow("Mask", mask)
    #cv2.imshow("Result", result)

    # Wait for 1 millisecond
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
    
    # When "Escape" key is pressed, save the images 
    # & close all windows
     
    
# Close all windows
cv2.destroyAllWindows()