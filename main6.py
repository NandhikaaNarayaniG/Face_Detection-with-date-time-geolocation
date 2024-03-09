import cv2
import datetime
import geocoder # For geolocation
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)  

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
# Directory to save the detected face images
output_dir = 'detected_faces' #directory to which your image has to be saved
os.makedirs(output_dir, exist_ok=True)

while True:    
    ret, frame = cap.read()    
    if not ret:
        print("Error: Could not read frame.")
        break
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Get current timestamp and date
    current_time = datetime.datetime.now().strftime("%H%M%S")
    current_date = datetime.datetime.now().strftime("%Y%m%d")

    # Get user's geolocation
    g = geocoder.ip('me')
    location = g.latlng  
    
    for i, (x, y, w, h) in enumerate(faces):        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {current_time}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Date: {current_date}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Location: {location}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        
        key = cv2.waitKey(1) & 0xFF
        if key == 13: 
            face_image_name = f"detected_face_{current_date}_{current_time}_face_{i+1}.jpg"
            face_image_path = os.path.join(output_dir, face_image_name)
            face_roi = frame[y:y+h, x:x+w]  
            cv2.imwrite(face_image_path, face_roi)
            print(f"Face image saved: {face_image_path}")

    # Display the frame
    cv2.imshow("Live Video with Face Detection", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


