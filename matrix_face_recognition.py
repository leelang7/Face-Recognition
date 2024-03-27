# you can get.py through this cli : jupyter nbconvert --to script notebook.ipynb
# .ipynb -> .py

import sys
sys.path.append('.')
import dlib, cv2
import numpy as np

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('./dlib_face_recognition_resnet_model_v1.dat')

descs = np.load('img/descs.npy', allow_pickle=True)
if descs.ndim == 0:
    descs = descs.item()  # 딕셔너리로 변환

def encode_face(img):
    dets = detector(img, 1)

    if len(dets) == 0:
        return np.empty(0)

    for k, d in enumerate(dets):
        shape = sp(img, d)
        face_descriptor = facerec.compute_face_descriptor(img, shape)

        return np.array(face_descriptor)

video_path = 'img/matrix_reloaded_1080p.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    exit()

_, img_bgr = cap.read() # (800, 1920, 3)
padding_size = 0
resized_width = 1920
video_size = (resized_width, int(img_bgr.shape[0] * resized_width // img_bgr.shape[1]))
output_size = (resized_width, int(img_bgr.shape[0] * resized_width // img_bgr.shape[1] + padding_size * 2))

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
writer = cv2.VideoWriter('%s_output.mp4' % (video_path.split('.')[0]), fourcc, cap.get(cv2.CAP_PROP_FPS), output_size)

while True:
    ret, img_bgr = cap.read()
    if not ret:
        break

    img_bgr = cv2.resize(img_bgr, video_size)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    dets = detector(img_bgr, 1)

    for k, d in enumerate(dets):
        shape = sp(img_rgb, d)
        face_descriptor = facerec.compute_face_descriptor(img_rgb, shape)

        min_dist = 999
        min_name = 'unknown'

        for name, saved_desc in descs.items():
            dist = np.linalg.norm(face_descriptor - saved_desc)

            if dist < min_dist:
                min_dist = dist
                min_name = name

        color = (255, 255, 255) if min_name != 'unknown' else (0, 0, 255)
        cv2.rectangle(img_bgr, pt1=(d.left(), d.top()), pt2=(d.right(), d.bottom()), color=color, thickness=2)
        cv2.putText(img_bgr, min_name, org=(d.left(), d.top()), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color, thickness=2)

    writer.write(img_bgr)

    cv2.imshow('img', img_bgr)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
writer.release()
