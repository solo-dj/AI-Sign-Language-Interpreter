import cv2
import mediapipe as mp
import pandas as pd
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Folder to store data
DATA_PATH = "data"
os.makedirs(DATA_PATH, exist_ok=True)

label = input("Enter the label for this session (e.g. A, B, C): ")

cap = cv2.VideoCapture(0)
data = []

print("[INFO] Collecting data for label:", label)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            data.append(landmarks)

    cv2.putText(image, f"Label: {label}  Samples: {len(data)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Collecting", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save to CSV
df = pd.DataFrame(data)
df['label'] = label
df.to_csv(f"{DATA_PATH}/{label}.csv", index=False)
print(f"[INFO] Saved {len(data)} samples to {label}.csv")
