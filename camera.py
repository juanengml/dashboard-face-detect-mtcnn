from os import system
from mtcnn import MTCNN 
from bounding_box import bounding_box as bb
import cv2

haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')

detector = MTCNN()

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
         self.video = cv2.VideoCapture(0)
         self.frameWidth = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
         self.frameHeight = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        cv2.imwrite("frame.jpg", frame)
        image = cv2.imread("frame.jpg")
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(img)
        for face in faces:
           x ,y , width, heigh = face['box']
           bb.add(image,x, y, x+width, y+heigh, "Face","fuchsia")

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


