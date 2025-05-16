import cv2
import cv2.aruco as aruco
import numpy as np

cap = cv2.VideoCapture(0)

#predefined dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
parameters = aruco.DetectorParameters()

#detector
detector = aruco.ArucoDetector(aruco_dict, parameters)

while True:
    ret, frame = cap.read()
    if not ret:
        print("error no return image")
        break

    # Detect
    corners, ids, rejected = detector.detectMarkers(frame)

    #detected
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        for i, corner in zip(ids, corners):
            print(f"Marker ID {i[0]} at position {corner}")

    #result
    cv2.imshow('ArUco Marker Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

