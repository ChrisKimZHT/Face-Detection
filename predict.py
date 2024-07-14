from AdaFace.net import Backbone
from yolov5face.utils.torch_utils import select_device
from yolov5face.models.experimental import attempt_load
import argparse

import cv2
import datetime
import torch
import os
import numpy as np
from pygments.lexer import default
from tqdm import tqdm
from PIL import Image
import sys

sys.path.append("./yolov5face")
sys.path.append("./AdaFace")


def get_faceinfo(data_path):
    face_list = []
    with open(data_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            face_list.append([int(x) for x in line.strip().split(',')])
    face_list = np.array(face_list)
    return face_list


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


if __name__ == '__main__':
    default_output_path = f'predict_outputs/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--tensor_path', default="./face_tensor.pth")
    parser.add_argument('--yolo5face_weights', nargs='+', type=str,
                        default='./yolov5face/yolov5s-face.pt',
                        help='model.pt path(s)')
    parser.add_argument('--video_path', default="video/my_test.mp4")

    parser.add_argument('--predict_img_path')
    parser.add_argument('--input_path')
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

    face_info_path = os.path.join(opt.input_path, "face_id.txt")
    device = select_device(opt.device)

    face_list = get_faceinfo(face_info_path)
    yolo5face = attempt_load(opt.yolo5face_weights, map_location=device)
    yolo5face.eval()
    AdaFace = load_AdaFace_model("./AdaFace/adaface_ir18_vgg2.ckpt").to(device)
    AdaFace.eval()

    features = torch.load(os.path.join(opt.input_path, "face_tensor.pth"))
    with torch.no_grad():
        with Image.open(opt.predict_img_path).convert('RGB') as aligned_rgb_img:
            aligned_rgb_img = aligned_rgb_img.resize((112, 112))
            bgr_tensor_input = to_input(aligned_rgb_img)
            bgr_tensor_input = bgr_tensor_input.to(device)
            feature, _ = AdaFace(bgr_tensor_input)
            features_tensor = torch.cat((features, feature))
    similarity_scores = features_tensor @ features_tensor.T

    feature_score = similarity_scores[-1]
    sorted_indices = torch.argsort(feature_score, descending=True)
    test_array = np.array(feature_score.cpu())
    if (feature_score[sorted_indices[1]] < 0.7):
        print("查无此人")
    else:
        id_face = face_list[sorted_indices[1]][1]
        id_face_seq = np.array([x for x in face_list if (x[1] == id_face)])
        sorted_seq = id_face_seq[id_face_seq[:, 2].argsort()]
        cap = cv2.VideoCapture(opt.video_path)
        for face_info in sorted_seq[-5:]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, face_info[2])
            ret, frame = cap.read()
            cv2.rectangle(frame, (face_info[3], face_info[4]), (
                face_info[5], face_info[6]), (0, 255, 0), 5, cv2.LINE_AA)
            cv2.imwrite(os.path.join(opt.output_path, f"{face_info[1]}_{face_info[2]}.jpg"), frame)

    if opt.log_file is not None:
        log_file.close()
