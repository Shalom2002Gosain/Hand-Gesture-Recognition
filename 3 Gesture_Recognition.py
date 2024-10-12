import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('hand_gesture_model.h5')
GESTURE_CLASSES = ['thumbs_up', 'thumbs_down', 'fist', 'palm']
IMG_SIZE = 64

# Real-time hand gesture recognition
def recognize_gesture():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        roi = frame[100:400, 100:400]
        cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

        # Preprocess the frame
        img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
        img = np.expand_dims(img, axis=0)
        img = img.astype('float32') / 255

        # Predict gesture
        prediction = model.predict(img)
        class_idx = np.argmax(prediction)
        class_label = GESTURE_CLASSES[class_idx]

        # Display the gesture on the screen
        cv2.putText(frame, class_label, (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start gesture recognition
recognize_gesture()
