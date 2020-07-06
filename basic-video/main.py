import cv2

cap = cv2.VideoCapture(0)

while True:
	_, frame = cap.read()
	frame = cv2.flip(frame, 1)

	cv2.imshow("frame", frame)
	
	key = cv2.waitKey(1)
	if key == 27:
		break

cap.release()
cv2.destroyAllWindows()