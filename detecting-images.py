import cv2
import torch
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

def unsafe_load(file):
    return torch.load(file, map_location='cpu', weights_only=False), file

def detect_objects_in_photo(image_path):
    image_orig = cv2.imread(image_path)
    
    tasks.torch_safe_load = unsafe_load
    yolo_model = YOLO('runs/detect/train/weights/best.pt')

    results = yolo_model(image_orig)

    for result in results:
        classes = result.names
        cls = result.boxes.cls
        conf = result.boxes.conf
        detections = result.boxes.xyxy

        for pos, detection in enumerate(detections):
            if conf[pos] >= 0.5:
                xmin, ymin, xmax, ymax = detection
                label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}" 
                
                if (label.split()[0]):
                    point = label.split()[1]
                    label = 'weapon ' + point
                    
                color = (0, int(cls[pos]), 255)
                cv2.rectangle(image_orig, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                cv2.putText(image_orig, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    result_path = "./imgs/Test/teste.jpg"
    cv2.imwrite(result_path, image_orig)
    return result_path

def detect_objects_in_video(video_path):
    tasks.torch_safe_load = unsafe_load
    yolo_model = YOLO('runs/detect/train/weights/best.pt')

    video_capture = cv2.VideoCapture(video_path)
    width = int(video_capture.get(3))
    height = int(video_capture.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    result_video_path = "./results/detected_objects_video2.avi"
    out = cv2.VideoWriter(result_video_path, fourcc, 20.0, (width, height))

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        results = yolo_model(frame)

        for result in results:
            classes = result.names
            cls = result.boxes.cls
            conf = result.boxes.conf
            detections = result.boxes.xyxy

            for pos, detection in enumerate(detections):
                if conf[pos] >= 0.5:
                    xmin, ymin, xmax, ymax = detection
                    label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}" 
                    
                    if (label.split()[0]):
                        point = label.split()[1]
                        label = 'weapon ' + point
                    
                    color = (0, int(cls[pos]), 255)
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                    cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        cv2.imshow('Weapon Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        out.write(frame)
    video_capture.release()
    out.release()

    return result_video_path

def detect_objects_and_plot(path_orig):
    image_orig = cv2.imread(path_orig)
    
    tasks.torch_safe_load = unsafe_load
    yolo_model = YOLO('./runs/detect/Normal_Compressed/weights/best.pt')
    results = yolo_model(image_orig)

    for result in results:
        classes = result.names
        cls = result.boxes.cls
        conf = result.boxes.conf
        detections = result.boxes.xyxy

        for pos, detection in enumerate(detections):
            if conf[pos] >= 0.5:
                xmin, ymin, xmax, ymax = detection
                label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}" 
                
                if (label.split()[0]):
                    point = label.split()[1]
                    label = 'weapon ' + point
                    
                color = (0, int(cls[pos]), 255)
                cv2.rectangle(image_orig, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                cv2.putText(image_orig, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    cv2.imshow("Test", image_orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def detect_objects_in_realtime():
    tasks.torch_safe_load = unsafe_load
    yolo_model = YOLO('./runs/detect/train/weights/best.pt')

    cap = cv2.VideoCapture(0)  # 0 -> the default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame)

        for result in results:
            classes = result.names
            cls = result.boxes.cls
            conf = result.boxes.conf
            detections = result.boxes.xyxy

            for pos, detection in enumerate(detections):
                xmin, ymin, xmax, ymax = detection
                label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}"
                
                # if (label.split()[0]):
                #     point = label.split()[1]
                #     label = 'weapon ' + point

                color = (0, int(cls[pos]), 255)
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        cv2.imshow('Real-Time Weapon Detection', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    video_path = "videos/test_video.mp4"
    result_video_path = detect_objects_in_video(video_path)

