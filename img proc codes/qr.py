import cv2

camera_id = 0
delay = 1
window_name = 'QR Code'

qcd = cv2.QRCodeDetector()
cap = cv2.VideoCapture(camera_id)

while True:
    ret, frame = cap.read()

    if ret:
        ret_qr, decoded_info, points, _ = qcd.detectAndDecodeMulti(frame)
        #retval, decoded_info, points, straight_qrcode
        if ret_qr:
            for s, p in zip(decoded_info, points):
            #decoded_info->tuple of strings in the qr(if cannot be decoded ->empty)
            #points->cordinates of the 4 corners
                if s:
                    print(s)
                    color = (0, 255, 0)
                    #outline the qr green->succcessfully decoded
                else:
                    color = (0, 0, 255)
                    #outline the qr red
                frame = cv2.polylines(frame, [p.astype(int)], True, color, 8)
        cv2.imshow(window_name, frame)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cv2.destroyWindow(window_name)

