
from ultralytics import YOLO
import cv2
import time
s_time = time.time()


model_path = 'C:/Users/ASUS/Desktop/calling_texting/yolov8_classification_human_action/weights/best.pt'

model = YOLO(model_path)
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
   
size = (frame_width, frame_height) 

result = cv2.VideoWriter('aa.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 10, size) 


while cap.isOpened():
    success, frame = cap.read()


    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame, save_txt = True, save = True, project = 'C:/Users/ASUS/Desktop/try_1')
        

        if results[0].probs.top1 ==0:
            print('-----------calling-----------')
        print('********************************************')
        #result.write(frame) 
        
        # Visualize the results on the frame
        #annotated_frame = results[0].plot()

        # Display the annotated frame
        #cv2.imshow("YOLOv8 Inference", annotated_frame)


        if time.time() > s_time + 10:  # Press 'q' to exit
            
            break
    else:
        break
cap.release()
result.release()
cv2.destroyAllWindows()
