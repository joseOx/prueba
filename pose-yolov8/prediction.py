from ultralytics import YOLO
import cv2
import numpy as np

#load model
model = YOLO('yolov8n-pose.pt') #Carga un modelo preentrenado

# Open video file
video_path = 'video/video.mp4'
#video_path= 0
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    succes, frame = cap.read()

    if succes:
        #run yolov8 inference on the frame
        results = model(frame, save=True)
        print(results)

        #Visualize the results on the frame
        annotated_frame = results[0].plot()

        #display the annotated frame
        cv2.imshow("YOLOV8 inference", annotated_frame)

        #Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        #Break the loop if the end of the video is reached
        break
#release the viddeo capture object and close the display window
cap.release()
cv2.destroyAllWindows()