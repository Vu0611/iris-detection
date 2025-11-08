#https://developers.google.com/mediapipe/solutions/vision/face_landmarker
#https://github.com/googlesamples/mediapipe/blob/main/examples/face_landmarker/python/[MediaPipe_Python_Tasks]_Face_Landmarker.ipynb




#@markdown We implemented some functions to visualize the face landmark detection results. <br/> Run the following cell to activate the functions.
import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import math


def distance(idx1, idx2, face_landmarks, rgb_image):
   x1 = int(face_landmarks[idx1].x * rgb_image.shape[1])
   y1 = int(face_landmarks[idx1].y * rgb_image.shape[0])
   x2 = int(face_landmarks[idx2].x * rgb_image.shape[1])
   y2 = int(face_landmarks[idx2].y * rgb_image.shape[0])
   d = math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))
   return d


def draw_landmarks_on_image(rgb_image, detection_result):
   face_landmarks_list = detection_result.face_landmarks
   annotated_image = np.copy(rgb_image)


   # Indices for the iris and eye landmarks
   #left_eye_indices = [33, 133, 160, 158, 159, 145, 153, 154, 155, 133]
   #right_eye_indices = [263, 362, 387, 385, 386, 374, 380, 381, 382, 362]
   #left_iris_indices = [474, 475, 476, 477, 468]
   #right_iris_indices = [469, 470, 471, 472, 473, 474]
   left_eye_indices = [263, 362, 386, 374] #right, left, up, down
   right_eye_indices = [33, 133, 159, 145, 168,5, 164]
   left_iris_indices = [468]
   right_iris_indices = [473]
   nose_indices = [168, 5, 164] #up, middle, down








   # Combine all indices into one list
   eye_and_iris_and_nose_indices = left_eye_indices + right_eye_indices + left_iris_indices + right_iris_indices


   # Loop through the detected faces to visualize.
   for idx in range(len(face_landmarks_list)):
       face_landmarks = face_landmarks_list[idx]


       #left & right eye horizontal and vertical distance  from iris:
       left_eye_h_d = (distance(263, 473, face_landmarks, rgb_image) - distance(362, 473, face_landmarks, rgb_image)) / distance(263, 362, face_landmarks, rgb_image)
       right_eye_h_d = (distance(33, 468, face_landmarks, rgb_image) - distance(133, 468, face_landmarks, rgb_image)) / distance(33, 133, face_landmarks, rgb_image)
       left_eye_v_d = (distance(386, 473, face_landmarks, rgb_image) - distance(374, 473, face_landmarks, rgb_image)) / distance(386, 374,face_landmarks,rgb_image)
       right_eye_v_d = (distance(159, 468, face_landmarks, rgb_image) - distance(145, 468, face_landmarks,rgb_image)) / distance(159, 145, face_landmarks, rgb_image)




       value = left_eye_h_d
       #if abs(left_eye_h_d) >= abs(right_eye_h_d):
       #    value = left_eye_h_d
       #else:
       #    value = right_eye_h_d


       looking = "Looking Left"
       if value > 0.2:
           looking = "Looking Right"
       elif value >= -0.2:
           looking = "Looking Forward"


       cv2.putText(annotated_image, looking, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 1)






       #if right_eye_h_d > 0.15:
       #    print("RY: Looking Right: " + str(right_eye_h_d))
       #elif right_eye_h_d < -0.15:
       #    #print("RY: Looking Left "+ str(right_eye_h_d))
       #else:
       #    #print ("RY: Looking forward "+ str(right_eye_h_d))


       def getAngle(idx1, idx2, idx3, face_landmarks, rgb_image):
           x1 = int(face_landmarks[idx1].x * rgb_image.shape[1])
           y1 = int(face_landmarks[idx1].y * rgb_image.shape[0])
           x2 = int(face_landmarks[idx2].x * rgb_image.shape[1])
           y2 = int(face_landmarks[idx2].y * rgb_image.shape[0])
           x3 = int(face_landmarks[idx3].x * rgb_image.shape[1])
           y3 = int(face_landmarks[idx3].y * rgb_image.shape[0])


           # Vectors
           vector1 = np.array([x1 - x2, y1 - y2])
           vector2 = np.array([x3 - x2, y3 - y2])


           # Dot product and magnitudes
           dot_product = np.dot(vector1, vector2)
           magnitude1 = np.linalg.norm(vector1)
           magnitude2 = np.linalg.norm(vector2)


           # Calculate the cosine of the angle
           cosine_angle = dot_product / (magnitude1 * magnitude2)


           # Ensure the value is within the valid range for arccos
           cosine_angle = min(max(cosine_angle, -1.0), 1.0)


           # Calculate the angle in radians
           angle_rad = np.arccos(cosine_angle)


           # Convert radians to degrees
           angle_deg = np.degrees(angle_rad)


           return angle_deg


       if getAngle(168, 5, 164, face_landmarks, rgb_image) <= 160:
           print("Not facing the camera")
       else:
           print ("Facing the camera")
       for i in eye_and_iris_and_nose_indices:
           #print(i)
           landmark = face_landmarks[i]
           x = int(landmark.x * rgb_image.shape[1])
           y = int(landmark.y * rgb_image.shape[0])
           cv2.circle(annotated_image, (x, y), 2, (0, 255, 0), -1)  # Green dot
           cv2.putText(annotated_image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)  # Blue text


   return annotated_image




def draw_landmarks_on_image2(rgb_image, detection_result):
   face_landmarks_list = detection_result.face_landmarks
   annotated_image = np.copy(rgb_image)


   # Loop through the detected faces to visualize.
   for idx in range(len(face_landmarks_list)):
       face_landmarks = face_landmarks_list[idx]


       # Draw the face landmarks and their indices.
       for i, landmark in enumerate(face_landmarks):
           x = int(landmark.x * rgb_image.shape[1])
           y = int(landmark.y * rgb_image.shape[0])
           cv2.circle(annotated_image, (x, y), 2, (0, 255, 0), -1)
           cv2.putText(annotated_image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)


   return annotated_image






# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# STEP 2: Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                    output_face_blendshapes=True,
                                    output_facial_transformation_matrixes=True,
                                    num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)




cap = cv2.VideoCapture(0)
while cap.isOpened():
   # Read feed
   ret, frame = cap.read()
   if not ret:
       break


   rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


   # To improve performance, optionally mark the image as not writeable to pass by reference.


   cv2.imwrite('test.png', rgb_frame)
   image = mp.Image.create_from_file("test.png")


   #image = mp.Image.create_from_array(rgb_frame)


   detection_result = detector.detect(image)


   # STEP 5: Process the detection result. In this case, visualize it.
   annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)


   #cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))


   # Show to screen
   cv2.imshow('OpenCV Feed', annotated_image)


   if cv2.waitKey(10) & 0xFF == ord('q'):
       break
cap.release()
cv2.destroyAllWindows()

