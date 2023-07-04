import logging
import yaml
import cv2
import numpy as np
import sys
import torch

from face_recognition.core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from face_recognition.core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from face_recognition.core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from face_recognition.core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from face_recognition.core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper
from face_recognition.core.model_loader.face_recognition.FaceRecModelLoader import FaceRecModelLoader
from face_recognition.core.model_handler.face_recognition.FaceRecModelHandler import FaceRecModelHandler

def evaluation_similarity(gt_image, gen_image):
    # the input resolution should be 224*224
    if torch.cuda.is_available():
            device = "cuda:0"
    else:
            device = "cpu"
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
    logging.config.fileConfig("face_recognition/config/logging.conf")
    logger = logging.getLogger('api')

    with open('face_recognition/config/model_conf.yaml') as f:
        model_conf = yaml.load(f)

    model_path = 'face_recognition/models'

    # face detection model setting.
    scene = 'non-mask'
    model_category = 'face_detection'
    model_name =  model_conf[scene][model_category]
    try:
        faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
        model, cfg = faceDetModelLoader.load_model()
        faceDetModelHandler = FaceDetModelHandler(model, device, cfg)
    except Exception as e:
        #logger.error('Falied to load face detection Model.')
        #logger.error(e)
        sys.exit(-1)

    # face landmark model setting.
    model_category = 'face_alignment'
    model_name =  model_conf[scene][model_category]
    try:
        faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)
        model, cfg = faceAlignModelLoader.load_model()
        faceAlignModelHandler = FaceAlignModelHandler(model, device, cfg)
    except Exception as e:
        #logger.error('Failed to load face landmark model.')
        #logger.error(e)
        sys.exit(-1)

    # face recognition model setting.
    model_category = 'face_recognition'
    model_name =  model_conf[scene][model_category]    
    try:
        faceRecModelLoader = FaceRecModelLoader(model_path, model_category, model_name)
        model, cfg = faceRecModelLoader.load_model()
        faceRecModelHandler = FaceRecModelHandler(model, device, cfg)
    except Exception as e:
        #logger.error('Failed to load face recognition model.')
        #logger.error(e)
        sys.exit(-1)
    # get face features.
    face_cropper = FaceRecImageCropper()
    dets_gt = faceDetModelHandler.inference_on_image(gt_image)
    dets_gen = faceDetModelHandler.inference_on_image(gen_image)
    dets = [dets_gt,dets_gen]
    image = [gt_image, gen_image]
    feature_list = []
    for i in range(2):
        landmarks = faceAlignModelHandler.inference_on_image(image[i], dets[i][0])
        landmarks_list = []
        for (x, y) in landmarks.astype(np.int32):
            landmarks_list.extend((x, y))
        cropped_image = face_cropper.crop_image_by_mat(image[i], landmarks_list)
        feature = faceRecModelHandler.inference_on_image(cropped_image)
        feature_list.append(feature)
    score = np.dot(feature_list[0], feature_list[1])
    

    return score


if __name__ == '__main__':
    gt_image = cv2.imread('/home/nfs/xshi2/resized_119/cam00.JPG')
    gen_image = cv2.imread('/home/nfs/xshi2/resized/cam02.JPG')

    similarity = evaluation_similarity(gt_image, gen_image)
    print(similarity)





