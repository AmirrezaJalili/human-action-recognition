from ultralytics import YOLO
import cv2
import time
s_time = time.time()



model = YOLO('best.pt')

cap = cv2.VideoCapture(0)

frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
size = (frame_width, frame_height) 

result = cv2.VideoWriter('HAR.avi',  
                         cv2.VideoWriter_fourcc(*'mp4v'), 10, size) 

names_dic = {0: 'calling',
 1: 'drinking',
 2: 'eating',
 3: 'laughing',
 4: 'sitting',
 5: 'sleeping',
 6: 'standing',
 7: 'texting',
 8: 'using_laptop'}

while cap.isOpened():
    success, frame = cap.read()


    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame, save_txt = True, save = True)
        annotated_frame = results[0].plot()
        
        
        idx = results[0].probs.top1
        print(names_dic[idx])
        
        
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

    

        result.write(frame)
        if time.time() > s_time + 60: 
            
            break
    else:
        break
cap.release()
result.release()
cv2.destroyAllWindows()
