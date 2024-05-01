from ultralytics import YOLO
import time
import cv2

time_s = time.time()


model_path = 'C:/Users/ASUS/Desktop/calling_texting/yolov8_classification_human_action/weights/best.pt'

model_cls = YOLO(model_path)
result = model_cls.predict(source = "0", save_txt = True, save = True, show = True)


'''
cnt = 0
for r in result:
    cnt += 1
    if cnt == 10:
        break
        
'''

