from imutils import build_montages
from datetime import datetime
import numpy as np
from imagezmq.imagezmq import ImageHub
import argparse
import imutils
import cv2
from recognize_faces import face_encode_frame

#We put all files path in arguments
ap = argparse.ArgumentParser()  
ap.add_argument("-p", "--prototxt", type=str, default='MobileNetSSD_deploy.prototxt',
                help="add path of Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", type=str, default='MobileNetSSD_deploy.caffemodel',
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
                help="we choose minimum confidence level is 0.2")
ap.add_argument("-mW", "--FrameW", type=int, default=1,
                help="Set frame width")
ap.add_argument("-mH", "--FrameH", type=int, default=1,
                help="Set frame height")
ap.add_argument("-fd", "--face-detect",  type=int, default=1,
                help="Always set to 1, for face detection")
ap.add_argument("-dm", "--detection-method", type=str, default='haarcascade_frontalface_default.xml',
                help="face detection model to use: either 'hog' or 'cnn' ")
ap.add_argument("-ef", "--encoding-file", type=str, default='encodings.pickle',
                help="face detection model to use: either 'hog' or 'cnn' ")


args = vars(ap.parse_args())

#Add encoding file path into 
EncodingMethod = args['encoding_file']
if not EncodingMethod:
    EncodingMethod = None

detection_method = args['detection_method']

#Set face detection is true
isface_detect = 0
if args['face_detect'] != 0:
    isface_detect = 1
	

imageHub = ImageHub()

#Those classes was already traained in MobileNet SSD, which can detected all those objects

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Load our pretrained model into our process
print("We are loading pre-trained model...")
PreModel = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# We are considering to detect persons, dogs, chairs, bottles from trained model
CONSIDER = set(["person", "dog", "chair", "bottle"])
objCount = {obj: 0 for obj in CONSIDER}
frameDictionary = {}

ActivedTime = {}
lastActiveCheck = datetime.now()

# We can use as many raspberry pi we want
# we are using 1 raspberry pi
NUM_PIS = 1
CHECK_FOR_ACTIVATION = 40
ACTIVE_TIME = NUM_PIS * CHECK_FOR_ACTIVATION

fW = args["FrameW"]
fH = args["FrameH"]

print("Detecting Objects from our pre-trained model: {}...".format(", ".join(obj for obj in
                                                 CONSIDER)))

while True:
    (NameOfRaspi, RecvFrame) = imageHub.recv_image()
    imageHub.send_reply(b'OK')

    if NameOfRaspi not in ActivedTime.keys():
        print("Receiving data from {} security camera...".format(NameOfRaspi))

    ActivedTime[NameOfRaspi] = datetime.now()

    RecvFrame = imutils.resize(RecvFrame, width=400)
    (h, w) = RecvFrame.shape[:2]
	
    blobImg = cv2.dnn.blobFromImage(cv2.resize(RecvFrame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    PreModel.setInput(blobImg)
    detections = PreModel.forward()

    objCount = {obj: 0 for obj in CONSIDER}

    for i in np.arange(0, detections.shape[2]):
	
        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:

            idx = int(detections[0, 0, i, 1])

            if CLASSES[idx] in CONSIDER:
                objCount[CLASSES[idx]] += 1

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                cv2.rectangle(RecvFrame, (startX, startY), (endX, endY),
                              (255, 0, 0), 2)

                if CLASSES[idx] == 'person':
                    if isface_detect:
                        RecvFrame, names = face_encode_frame(RecvFrame, detection_method, EncodingMethod)
             
    cv2.putText(RecvFrame, NameOfRaspi, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	
    Obj_label = ", ".join("{}: {}".format(obj, count) for (obj, count) in
                      objCount.items())
    cv2.putText(RecvFrame, Obj_label, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    frameDictionary[NameOfRaspi] = RecvFrame

    montages = build_montages(frameDictionary.values(), (w, h), (fW, fH))

    for (i, montage) in enumerate(montages):
        cv2.imshow("Home security camera ({})".format(i),
                   montage)

    Quitkey = cv2.waitKey(1) & 0xFF

    if (datetime.now() - lastActiveCheck).seconds > ACTIVE_TIME:
        for (NameOfRaspi, ts) in list(ActivedTime.items()):
            if (datetime.now() - ts).seconds > ACTIVE_TIME:
                print("Lost connection to {}".format(NameOfRaspi))
                ActivedTime.pop(NameOfRaspi)
                frameDictionary.pop(NameOfRaspi)
        lastActiveCheck = datetime.now()

    if Quitkey == ord("q"):
        break

cv2.destroyAllWindows()
