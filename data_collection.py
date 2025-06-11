import cv2
import mediapipe as mp
import pandas as pd
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = 'C:\\Users\\Aarush\\PycharmProjects\\ML Project\\hand_landmarker.task'
try:
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)
    print("✅ Hand landmarker model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    exit()

data_rows = []

cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("❌ Error: Could not open webcam")
    exit()

print("Data collection started. Press a letter key (a-z) to start capturing 35 samples, Enter to quit")

capture_count = 0
current_label = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Couldn't read frame")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    display_frame = frame.copy()
    current_detection = None
    hand_landmarks_for_data = None

    frame_height, frame_width = frame.shape[:2]

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    detection_result = detector.detect(image)

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
        new_max_x = min(frame_width - 1, max_x + margin * width)
        new_min_y = max(0, min_y - margin * height)
        new_max_y = min(frame_height - 1, max_y + margin * height)

        expanded_width = new_max_x - new_min_x
        expanded_height = new_max_y - new_min_y
        frame_ar = frame_width / frame_height


        if expanded_width / expanded_height > frame_ar:

            crop_height = expanded_width / frame_ar
            center_y = (new_min_y + new_max_y) / 2
            crop_min_y = max(0, center_y - crop_height / 2)
            crop_max_y = min(frame_height, center_y + crop_height / 2)
            crop_min_x = new_min_x
            crop_max_x = new_max_x
        else:

            crop_width = expanded_height * frame_ar
            center_x = (new_min_x + new_max_x) / 2
            crop_min_x = max(0, center_x - crop_width / 2)
            crop_max_x = min(frame_width, center_x + crop_width / 2)
            crop_min_y = new_min_y
            crop_max_y = new_max_y


        if crop_max_x > crop_min_x and crop_max_y > crop_min_y:
            cropped_rgb = frame_rgb[int(crop_min_y):int(crop_max_y), int(crop_min_x):int(crop_max_x)]
            resized_rgb = cv2.resize(cropped_rgb, (frame_width, frame_height))
            zoomed_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=resized_rgb)
            zoomed_detection = detector.detect(zoomed_image)

            if zoomed_detection.hand_landmarks:
                current_detection = zoomed_detection
                hand_landmarks_for_data = zoomed_detection.hand_landmarks[0]
                display_frame = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)
                for lm in hand_landmarks_for_data:
                    x = int(lm.x * display_frame.shape[1])
                    y = int(lm.y * display_frame.shape[0])
                    cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
            else:
                current_detection = detection_result
                hand_landmarks_for_data = hand_landmarks
                for lm in hand_landmarks:
                    x = int(lm.x * frame.shape[1])
                    y = int(lm.y * frame.shape[0])
                    cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
        else:

            current_detection = detection_result
            hand_landmarks_for_data = hand_landmarks
            for lm in hand_landmarks:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
    else:

        pass


    if current_label is not None and capture_count > 0 and current_detection and current_detection.hand_landmarks:
        landmarks_flat = []
        for lm in hand_landmarks_for_data:
            landmarks_flat.extend([lm.x, lm.y, lm.z])
        data_rows.append([current_label] + landmarks_flat)
        capture_count -= 1
        print(f"Saved sample {35 - capture_count} for label '{current_label}'")
        if capture_count == 0:
            print(f"Finished capturing 35 samples for label '{current_label}'")
            current_label = None


    cv2.putText(display_frame, "Press [a-z] to capture 35 samples, Enter to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Data Collection", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key in range(97, 123):  # a-z
        current_label = chr(key)
        capture_count = 35
        print(f"Starting capture of 35 samples for label '{current_label}'")
    elif key == 13:  # Enter key
        break

cap.release()
cv2.destroyAllWindows()


if data_rows:
    columns = ['label']
    for i in range(21):
        for coord in ['x', 'y', 'z']:
            columns.append(f'lm{i}_{coord}')
    df = pd.DataFrame(data_rows, columns=columns)

    file_exists = os.path.isfile('hand_landmarks_data2.csv')
    df.to_csv(
        'hand_landmarks_data2.csv',
        mode='a' if file_exists else 'w',
        index=False,
        header=not file_exists
    )
    print(f"Appended {len(df)} samples to 'hand_landmarks_data2.csv'")
else:
    print("No data collected.")