import cv2
import mediapipe as mp
from pythonosc import udp_client

client = udp_client.SimpleUDPClient("127.0.0.1", 8000)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            for i, lm in enumerate(hand.landmark):
                if i in [8, 12, 16]:  # índice, medio, anular
                    client.send_message(f"/finger{i}", [lm.x, lm.y])

    if cv2.waitKey(1) == 27:
        break

cap.release()