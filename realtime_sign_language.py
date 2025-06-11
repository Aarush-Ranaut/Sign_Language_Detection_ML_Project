import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

clf = joblib.load('hand_sign_classifier.pkl')
print("Loaded trained classifier.")

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(2)
sentence = ""

current_letter = None
letter_start_time = None
letter_added = False

no_hand_start_time = None
space_added = False

print("Realtime detection with zoom started. Press 'b' for backspace, 'c' to clear sentence, 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    display_frame = frame.copy()

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    detection_result = detector.detect(image)

    current_time = time.time()
    predicted_letter = ""

    if detection_result.hand_landmarks:
        hand_landmarks = detection_result.hand_landmarks[0]

        xs = [lm.x * image.width for lm in hand_landmarks]
        ys = [lm.y * image.height for lm in hand_landmarks]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x
        height = max_y - min_y
        margin = 0.2

        new_min_x = max(0, min_x - margin * width)
        new_max_x = min(frame.shape[1] - 1, max_x + margin * width)
        new_min_y = max(0, min_y - margin * height)
        new_max_y = min(frame.shape[0] - 1, max_y + margin * height)

        expanded_width = new_max_x - new_min_x
        expanded_height = new_max_y - new_min_y
        frame_ar = frame.shape[1] / frame.shape[0]

        if expanded_width / expanded_height > frame_ar:
            crop_height = expanded_width / frame_ar
            center_y = (new_min_y + new_max_y) / 2
            crop_min_y = max(0, center_y - crop_height / 2)
            crop_max_y = min(frame.shape[0], center_y + crop_height / 2)
            crop_min_x = new_min_x
            crop_max_x = new_max_x
        else:
            crop_width = expanded_height * frame_ar
            center_x = (new_min_x + new_max_x) / 2
            crop_min_x = max(0, center_x - crop_width / 2)
            crop_max_x = min(frame.shape[1], center_x + crop_width / 2)
            crop_min_y = new_min_y
            crop_max_y = new_max_y

        if crop_max_x > crop_min_x and crop_max_y > crop_min_y:
            cropped_rgb = frame_rgb[int(crop_min_y):int(crop_max_y), int(crop_min_x):int(crop_max_x)]
            resized_rgb = cv2.resize(cropped_rgb, (frame.shape[1], frame.shape[0]))
            zoomed_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=resized_rgb)
            zoomed_detection = detector.detect(zoomed_image)

            if zoomed_detection.hand_landmarks:
                hand_landmarks_for_prediction = zoomed_detection.hand_landmarks[0]
                display_frame = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)
                for lm in hand_landmarks_for_prediction:
                    x = int(lm.x * display_frame.shape[1])
                    y = int(lm.y * display_frame.shape[0])
                    cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
            else:
                hand_landmarks_for_prediction = hand_landmarks
                for lm in hand_landmarks:
                    x = int(lm.x * frame.shape[1])
                    y = int(lm.y * frame.shape[0])
                    cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
        else:
            hand_landmarks_for_prediction = hand_landmarks
            for lm in hand_landmarks:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)

        landmarks_flat = []
        for lm in hand_landmarks_for_prediction:
            landmarks_flat.extend([lm.x, lm.y, lm.z])
        landmarks_flat = np.array(landmarks_flat).reshape(1, -1)

        predicted_letter = clf.predict(landmarks_flat)[0]

        if current_letter != predicted_letter:
            current_letter = predicted_letter
            letter_start_time = current_time
            letter_added = False
        else:
            if letter_start_time and (current_time - letter_start_time) >= 2 and not letter_added:
                sentence += current_letter
                letter_added = True

        no_hand_start_time = None
        space_added = False
    else:
        if no_hand_start_time is None:
            no_hand_start_time = current_time
        else:
            if (current_time - no_hand_start_time) >= 2 and not space_added:
                sentence += ' '
                space_added = True

        current_letter = None
        letter_start_time = None
        letter_added = False

    cv2.putText(display_frame, f"Sentence: {sentence}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Predicted: {predicted_letter}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Realtime Hand Sign Detection with Zoom", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('b'):
        if sentence:  # Check if sentence is not empty before removing
            sentence = sentence[:-1]
    elif key == ord('c'):
        sentence = ""  # Clear the entire sentence
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()