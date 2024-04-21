import cv2
import numpy as np
import time
from src.em import Emotional

emotional = Emotional()

cap = cv2.VideoCapture(0)

emotion_colors = {
    'Гнев': (0, 255, 255),       
    'Презрение': (0, 255, 0),    
    'Отвращение': (255, 140, 0), 
    'Страх': (255, 20, 147),     
    'Счастье': (255, 0, 255),    
    'Нейтральность': (0, 191, 255), 
    'Грусть': (255, 255, 0),     
    'Удивление': (255, 0, 0)     
}


prev_frame_time = 0
new_frame_time = 0

while True:
    _, frame = cap.read()

    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps_text = f'FPS: {fps}'

    emotion = emotional(frame)
    
    color = emotion_colors[emotion]
    
    cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, color, 4, cv2.LINE_AA)
    
    cv2.putText(frame, fps_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 4, cv2.LINE_AA)

    cv2.imshow('Обнаружение эмоций', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
