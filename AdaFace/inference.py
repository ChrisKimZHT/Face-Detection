import net
import torch
import os
from face_alignment import align
import numpy as np
from tqdm import tqdm
from PIL import Image

adaface_models = {
    'ir_18':"/root/PycharmProjects/pythonProject/AdaFace/adaface_ir18_vgg2.ckpt",
}

def load_pretrained_model(architecture='ir_18'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
    return tensor

if __name__ == '__main__':

    model = load_pretrained_model('ir_18')
    model.eval()  # 确保模型处于评估模式
    test_image_path = '../data_face'
    dirlist = os.listdir(test_image_path)

    features = []
    with torch.no_grad():  # 禁用梯度计算
        for fname in tqdm(dirlist):
            path = os.path.join(test_image_path, fname)
            with Image.open(path).convert('RGB') as aligned_rgb_img:
                bgr_tensor_input = to_input(aligned_rgb_img)
                feature, _ = model(bgr_tensor_input)
                features.append(feature)

    # 将特征拼接在一起，计算相似性得分
    features_tensor = torch.cat(features)
    similarity_scores = features_tensor @ features_tensor.T
    print(similarity_scores)

