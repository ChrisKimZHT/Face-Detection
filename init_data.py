import datetime

from AdaFace.net import Backbone
from yolov5face.sort import Sort
from tqdm import tqdm
from yolov5face.utils.torch_utils import select_device, load_classifier, time_synchronized
from yolov5face.utils.general import check_img_size, check_requirements, non_max_suppression_face, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov5face.utils.datasets import letterbox
from yolov5face.models.experimental import attempt_load
import argparse
import os
import cv2
import torch
import numpy as np
import sys


sys.path.append("./yolov5face")


sys.path.append("./AdaFace")


def load_AdaFace_model(weight_path):
    model = Backbone((112, 112), 18, 'ir')
    statedict = torch.load(weight_path)['state_dict']
    model_statedict = {key[6:]: val for key,
                       val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model


def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:, :, ::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2, 0, 1)]).float()
    return tensor


def dynamic_resize(shape, stride=64):
    max_size = max(shape[0], shape[1])
    if max_size % stride != 0:
        max_size = (int(max_size / stride) + 1) * stride
    return max_size


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / \
            2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    # clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords


def show_results(img, xywh, conf, landmarks, class_num):
    h, w, c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0),
                  thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
             (255, 255, 0), (0, 255, 255)]

    for i in range(5):
        point_x = int(landmarks[2 * i] * w)
        point_y = int(landmarks[2 * i + 1] * h)
        cv2.circle(img, (point_x, point_y), tl + 1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(int(class_num)) + ': ' + str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3,
                [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def detect(model, img0):
    stride = int(model.stride.max())  # model stride
    imgsz = opt.img_size
    if imgsz <= 0:  # original size
        imgsz = dynamic_resize(img0.shape)
    imgsz = check_img_size(imgsz, s=64)  # check img_size
    img = letterbox(img0, imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=opt.augment)[0]
    # Apply NMS
    pred = non_max_suppression_face(pred, opt.conf_thres, opt.iou_thres)[0]
    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]].to(
        device)  # normalization gain whwh
    gn_lks = torch.tensor(img0.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(
        device)  # normalization gain landmarks
    boxes = []
    h, w, c = img0.shape
    if pred is not None:
        pred[:, :4] = scale_coords(
            img.shape[2:], pred[:, :4], img0.shape).round()
        pred[:, 5:15] = scale_coords_landmarks(
            img.shape[2:], pred[:, 5:15], img0.shape).round()
        for j in range(pred.size()[0]):
            xywh = (xyxy2xywh(pred[j, :4].view(1, 4)) / gn).view(-1)
            xywh = xywh.data.cpu().numpy()
            conf = pred[j, 4].cpu().numpy()
            landmarks = (pred[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
            class_num = pred[j, 15].cpu().numpy()
            x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
            y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
            x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
            y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
            boxes.append([x1, y1, x2 - x1, y2 - y1, conf])
    return boxes


if __name__ == '__main__':
    default_output_path = f'init_outputs/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='./yolov5face/yolov5s-face.pt',
                        help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.02, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')

    parser.add_argument('--input_video', default="video/my_test.mp4")
    parser.add_argument('--output_path', type=str, default=default_output_path)
    parser.add_argument('--log_file', type=str, default=None)

    opt = parser.parse_args()

    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)

    # 重定向输出
    if opt.log_file is not None:
        log_file = open(f"{opt.output_path}/{opt.log_file}",
                        'w', encoding='utf-8')
        sys.stdout = log_file
        sys.stderr = log_file

    print(opt)

    if not os.path.exists(os.path.join(opt.output_path, 'face')):
        os.makedirs(os.path.join(opt.output_path, 'face'))

    device = select_device(opt.device)
    yolo5face = attempt_load(
        opt.weights, map_location=device)  # load FP32 model

    radio = 0.20
    index = 0
    face_img = []

    file_info = {}

    f = open(os.path.join(opt.output_path, "face_id.txt"), "w")

    with torch.no_grad():
        cap = cv2.VideoCapture(opt.input_video)
        tracker = Sort()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for num_frame in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            box_xywh = []
            confidences = []
            boxes = detect(yolo5face, frame)
            detections = []
            for box in boxes:
                x, y, width, height, confidence = box
                if (float(confidence) >= radio):
                    detections.append(
                        [x, y, x + width, y + height, float(confidence)])
            # detections = np.array([box for box in boxes if float(box[4]) >radio])
            detections = np.array(detections)

            tracks = tracker.update(detections)
            for i, track in enumerate(tracks):
                x, y, x1, y1, track_id = track
                x, y, x1, y1 = int(max(0, x)), int(max(0, y)), int(
                    min(frame.shape[1], x1)), int(min(frame.shape[0], y1))
                img_crop = frame[int(y):int(y1), int(x):int(x1)].copy()
                img_crop = cv2.resize(img_crop, (112, 112))
                face_img.append(img_crop)
                cv2.rectangle(frame, (int(x), int(y)),
                              (int(x1), int(y1)), (255, 0, 0), 2)
                cv2.putText(frame, str(int(track_id)), (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 0, 255), 5)
                cv2.imwrite(os.path.join(
                    opt.output_path, 'face', f"{index}.jpg"), img_crop)

                f.write(
                    f"{index},{int(track_id)},{num_frame},{int(x)},{int(y)},{int(x1)},{int(y1)}\n")
                index += 1
            # if(index>100):
            #     break

        f.close()

        AdaFace = load_AdaFace_model(
            "./AdaFace/adaface_ir18_vgg2.ckpt").to(device)
        AdaFace.eval()
        features = []
        # dirlist = os.listdir(opt.data_face)
        with torch.no_grad():  # 禁用梯度计算
            for img in tqdm(face_img):
                bgr_tensor_input = to_input(img)
                bgr_tensor_input = bgr_tensor_input.to(device)
                feature, _ = AdaFace(bgr_tensor_input)
                features.append(feature)
        features_tensor = torch.cat(features)
        similarity_scores = features_tensor @ features_tensor.T
        torch.save(features_tensor, os.path.join(
            opt.output_path, "face_tensor.pth"))
