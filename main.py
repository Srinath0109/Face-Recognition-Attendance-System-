import cv2
import pandas as pd
from datetime import datetime
from face_recognition import recognize_face

def mark_attendance(name):
    df = pd.read_csv("attendance.csv")
    
    if name not in df["Name"].values:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df = df.append({"Name": name, "Timestamp": now}, ignore_index=True)
        df.to_csv("attendance.csv", index=False)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame, name = recognize_face(frame)

    if name != "Unknown":
        mark_attendance(name)

    cv2.imshow("Face Recognition Attendance", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
