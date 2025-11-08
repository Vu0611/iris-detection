import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
   while cap.isOpened():
       ret, frame = cap.read()
       faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
       for (x, y, w, h) in faces:
           cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


       # Recolor image to RGB
       image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       image.flags.writeable = False


       # Make detection
       results = pose.process(image)


       # Recolor back to BGR
       image.flags.writeable = True
       image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


       # Extract landmarks
       try:
           landmarks = results.pose_landmarks.landmark
           print(landmarks)
       except:
           pass


       # Render detections
       mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                 )


       cv2.imshow('Mediapipe Feed', image)


       if cv2.waitKey(10) & 0xFF == ord('q'):
           break


   cap.release()
   cv2.destroyAllWindows()

