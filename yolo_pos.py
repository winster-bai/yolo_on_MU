import cv2
import time
# import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 加载YOLOv8n模型
model = YOLO('yolov8n-pose.onnx')

# 打开本地视频文件
cap = cv2.VideoCapture("people.mp4")


# 获取视频帧的宽度和高度
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 计时器和FPS初始化
prev_time = 0
fps = 0

while True:
    # 读取帧
    ret, frame = cap.read()
    
    if not ret:
        print("无法读取帧")
        break

    # 将帧传递给模型进行预测，并明确指定使用CPU
    results = model(frame, device='cpu')

    # 获取预测结果并绘制在帧上
    for result in results:

        # 先将预测结果转换为PIL图像，再转换为NumPy数组
        pil_image = Image.fromarray(results[0].plot())
        frame = np.array(pil_image)

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = map(int, box[:4])
            confidence = confidences[i]
            class_id = class_ids[i]
            label = result.names[class_id]
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), 
                        # cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # 计算FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    frame = cv2.resize(frame, (640, 480))  # 将帧的大小调整为640x480

    # 将FPS绘制在帧的左上角
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    

    # 显示帧
    cv2.imshow('YOLOv8n Real-time Object Detection', frame)
    
    # 按下'Q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
