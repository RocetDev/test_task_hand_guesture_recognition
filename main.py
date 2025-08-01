import cv2
from ultralytics import YOLO

PATH_TO_MODEL = 'yolo_model_v2.pt'

yolo_model = YOLO(PATH_TO_MODEL)


window_name = 'Hand gesture detection'
cap = cv2.VideoCapture(0)
border_size = 160


print('START PROGRAM...')
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not ret:
        break

    model_frame = cv2.copyMakeBorder(frame, 0, border_size, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    results = yolo_model(frame, verbose=False)

    for result in results:
        boxes = result.boxes.cpu().numpy()

        print(boxes)

        for i, box in enumerate(boxes):
            r = box.xyxy[0].astype(int)

            confidence = box.conf[0]
            class_id = box.cls[0].astype(int)

            label = f'{yolo_model.names[class_id],} {confidence:.2f}'
            cv2.rectangle(frame, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
            cv2.putText(frame, label, (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow(window_name, frame)
    keyCode = cv2.waitKey(100)

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) <1:
        break

cap.release()
cv2.destroyAllWindows()
print('END PROGRAM')