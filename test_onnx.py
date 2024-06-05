import cv2
import numpy as np
import onnxruntime as ort
import yaml
import time 


def preprocess(frame, input_size):
    image = cv2.resize(frame, input_size,interpolation=cv2.INTER_NEAREST)
    # 转换图片到数组
    image_data = np.array(image).transpose(2, 0, 1)  # 转换成CHW
    image_data = image_data.astype(np.float32)
    image_data /= 255.0  # 归一化
    image_data = np.expand_dims(image_data, axis=0)  # 增加batch维度
    return image_data

def postprocess(output, image, input_size, show_size, classes):
    for detection in output:
        x1, y1, x2, y2, conf , class_id = detection
        if conf > 0.4:
            x1 = int(x1 / input_size[0] * show_size[0])
            x2 = int(x2 / input_size[0] * show_size[0])
            y1 = int(y1 / input_size[1] * show_size[1])
            y2 = int(y2 / input_size[1] * show_size[1])
            class_id = int(class_id)  
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 画框
            class_name = classes[class_id]
            cv2.putText(image, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return image

def main():
    input_size = (128, 128)

    with open('coco.yaml', 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    classes = data['names']


    window_name = 'FullScreen Image'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # 加载模型
    session = ort.InferenceSession('yolov10n.onnx')
    input_name = session.get_inputs()[0].name

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    prev_time = 0
    while True:
        ret, frame = cap.read()
        show_size = [320,240]

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        current_time = time.time()
        # 预处理图像
        input_tensor = preprocess(frame, input_size)

        # 进行推理
        outputs = session.run(None, {input_name: input_tensor})
        output = outputs[0][0]

        # 后处理
        show_image = postprocess(output, frame, input_size, show_size, classes)

        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time  # 更新前一帧的时间
        cv2.putText(show_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 显示结果
        cv2.imshow(window_name, show_image)
        if cv2.waitKey(1) == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()