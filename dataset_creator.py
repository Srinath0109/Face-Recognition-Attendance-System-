import cv2
import os

def capture_images(user_id, name):
    folder = f"dataset/{user_id}_{name}"
    os.makedirs(folder, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    count = 0

    while count < 50:  # Capture 50 images per user
        ret, frame = cap.read()
        if not ret:
            break
        
        file_path = os.path.join(folder, f"{count}.jpg")
        cv2.imwrite(file_path, frame)
        cv2.imshow("Capturing Face", frame)
        
        count += 1
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Images saved for {name}")

if __name__ == "__main__":
    user_id = input("Enter User ID: ")
    name = input("Enter Name: ")
    capture_images(user_id, name)
