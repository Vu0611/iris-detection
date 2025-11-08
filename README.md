# Iris Detection: Real-Time Gaze and Head Orientation Tracking

## Overview
This project detects a person’s **iris** and **head orientation** in real time using **MediaPipe face landmarks**.  
It determines whether the user is **looking left, right, or forward**, and whether they are **facing the screen**, enabling applications such as a *virtual receptionist* that only responds when a person is attentive.

---

## Features
- Real-time **iris tracking** using `face_landmarker_v2_with_blendshapes.task`  
- Computes **iris-to-eye corner ratios** to infer gaze direction  
- Calculates **nose-angle geometry** to determine facing direction  
- Visual output overlay on webcam feed  
- Fully implemented in **Python + OpenCV**, no external ML model training required  

---

## Project Structure
```
├── iris_detection.py            # Main executable for real-time detection
├── face_pose_detection.py       # Combined face + pose detection code
├── requirements.txt             # Dependencies
└── README.md                    # Documentation (this file)
```

---

## 🧮 Methodology
1. **Landmark Selection:**  
   Identifies key points (iris center, eye corners, nose tip/root/base).  
2. **Gaze Detection:**  
   Calculates normalized distance ratio between the iris and eye corners;  
   uses a ±0.2 threshold to classify gaze as left/forward/right.  
3. **Facing Detection:**  
   Computes internal angle of nose landmarks; ≤160° ⇒ not facing screen.  
4. **Visualization:**  
   Draws landmarks and direction text on the camera frame using OpenCV.

---

## Dependencies
```bash
pip install opencv-python==4.9.0.80 mediapipe==0.10.5 numpy==1.24.3 mpmath==1.3.0
```

---

## How to Run
```bash
python iris_detection.py
```
Ensure your webcam is enabled. Press **Q** to quit the live window.

---

## Code & Resources
Full implementation and documentation are available in the Google Drive folder:  
🔗 [Iris Detection Project (Drive)](https://drive.google.com/drive/folders/1MtNwn8LJ_aC8fhYFVHX_ksdCs3FN4lmt)

---

## Future Improvements
- Add detection for **up/down gaze**  
- Integrate **person tracking** to maintain unique IDs  
- Combine with **Virtual Receptionist AI** for complete interaction flow  

---