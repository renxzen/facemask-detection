from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2


def pixelate_face(face, blocks=9):
    (h, w) = face.shape[:2]
    x_steps = np.linspace(0, w, blocks + 1, dtype="int")
    y_steps = np.linspace(0, h, blocks + 1, dtype="int")

    for i in range(1, len(y_steps)):
        for j in range(1, len(x_steps)):
            x_start = x_steps[j - 1]
            y_start = y_steps[i - 1]
            x_end = x_steps[j]
            y_end = y_steps[i]

            roi = face[y_start:y_end, x_start:x_end]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(face, (x_start, y_start), (x_end, y_end), (B, G, R), -1)

    return face


def detect_and_predict(frame, face_net, mask_net):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()
    # print(detections.shape)

    faces = []
    locs = []
    preds = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x_start, y_start, x_end, y_end) = box.astype("int")

            (x_start, y_start) = (max(0, x_start), max(0, y_start))
            (x_end, y_end) = (min(w - 1, x_end), min(h - 1, y_end))

            face = frame[y_start:y_end, x_start:x_end]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((x_start, y_start, x_end, y_end))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = mask_net.predict(faces, batch_size=32)

    return (locs, preds)


if __name__ == "__main__":
    proto_path = r"models/faces.prototxt"
    weights_path = r"models/faces.model"
    face_net = cv2.dnn.readNet(proto_path, weights_path)

    mask_net = load_model("models/facemasks.model")

    print("[INFO] Starting video stream...")
    vs = VideoStream(src=0).start()

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        (locs, preds) = detect_and_predict(frame, face_net, mask_net)

        for (box, pred) in zip(locs, preds):

            (x_start, y_start, x_end, y_end) = box
            (mask, without_mask) = pred

            label = "Mask" if mask > without_mask else "NoMask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            face = frame[y_start:y_end, x_start:x_end]
            frame[y_start:y_end, x_start:x_end] = (
                pixelate_face(face, 20) if label == "NoMask" else face
            )

            label = f"{round(max(mask, without_mask) * 100,2)} {label}"

            cv2.putText(
                frame,
                label,
                (x_start, y_start - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2,
            )
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), color, 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()
