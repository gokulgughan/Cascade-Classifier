import cv2
frontal_face= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
profile_face = cv2.CascadeClassifier('haarcascade_profileface.xml')
webcam = cv2.VideoCapture(0)
while True:
    _, img = webcam.read()
    img = cv2.flip(img,1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    frontal_faces = frontal_face.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(20, 20))
    profile_faces = profile_face.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(20,20))

    for (x, y, w, h) in frontal_faces:
        face = img[y:y+h, x:x+w]
        blur = cv2.GaussianBlur(face, (99,99), 30)
        img[y:y+h, x:x+w] = blur
    
    for(x, y, w, h) in profile_faces:
        face=img[y:y+h, x:x+w]
        blur = cv2.GaussianBlur(face, (99,99), 30)
        img[y:y+h, x:x+w]
    cv2.imshow("Face Recon", img)
    if cv2.waitKey(10) == 27:
        break
webcam.release()
cv2.destroyAllWindows()