import face_recognition
import pickle
import cv2
import imutils
import platform

data = None


def load_encodings(encodings_file):
    global data
    if data is None:
        # load the known faces and embeddings
        print("[INFO] loading encodings...")
        data = pickle.loads(open(encodings_file, "rb").read())
        print("[INFO] loading encodings... DONE")


def face_encode_frame(frame, detection_method, encodings_file, face_detector=None):
   
    if encodings_file is None:
        return
    load_encodings(encodings_file)

    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_image = imutils.resize(rgb_image, width=750)
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    r = frame.shape[1] / float(rgb_image.shape[1])

    if face_detector:
        scale_factor = 1.1
        if platform.system() == 'Linux':
            scale_factor = 1.05

        rects = face_detector.detectMultiScale(gray, scaleFactor=scale_factor,
                                               minNeighbors=5, minSize=(30, 30),
                                               flags=cv2.CASCADE_SCALE_IMAGE)

        boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects if w > 70 and h > 70]
    else:
        boxes = face_recognition.face_locations(rgb_image, model=detection_method)

    if len(boxes) == 0:
        return frame, []

    encodings = face_recognition.face_encodings(rgb_image, boxes)

    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data['encodings'], encoding)
        name = "Unknown"

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = data['names'][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)
        names.append(name)

    for ((top, right, bottom, left), name) in zip(boxes, names):
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

        cv2.rectangle(frame, (left, top), (right, bottom),
                      (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)

    return frame, names
