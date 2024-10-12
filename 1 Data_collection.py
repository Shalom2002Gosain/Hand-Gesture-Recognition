import cv2
import os

# Directory to save gesture images
DATA_DIR = 'C:\\Users\\shalo\\Downloads\\archive (2)\\train'
GESTURE_CLASSES = ['thumbs_up', 'thumbs_down', 'fist', 'palm']
IMG_SIZE = 64

# Create directories if they don't exist
for gesture in GESTURE_CLASSES:
    os.makedirs(os.path.join(DATA_DIR, gesture), exist_ok=True)

def capture_gesture(gesture_name):
    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # Mirror the frame
        cv2.putText(frame, f'Gesture: {gesture_name}, Images: {count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Gesture Capture', frame)
        
        # Capture region of interest
        roi = frame[100:400, 100:400]
        cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

        key = cv2.waitKey(1)
        if key == ord('c'):  # Capture image when 'c' is pressed
            img_path = os.path.join(DATA_DIR, gesture_name, f'{count}.jpg')
            roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(img_path, roi_resized)
            count += 1

        elif key == ord('q'):  # Quit on 'q'
            break

    cap.release()
    cv2.destroyAllWindows()

# Capture data for a specific gesture
capture_gesture('thumbs_up')
