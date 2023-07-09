import json
import sys
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
sys.path.append("../detr")
from models import build_model
from src.settings import TSR_MODEL, TSR_MODEL_CONFIG, TD_MODEL, TD_MODEL_CONFIG


def VietOCR_model():
    # Load vietOCR model
    config = Cfg.load_config_from_name('vgg_transformer')
    config['cnn']['pretrained']=False
    config['device'] = 'cpu'

    viet_ocr = Predictor(config)
    return viet_ocr

def PaddleOCR_model():
    paddle_ocr = PaddleOCR(show_log = False)
    return paddle_ocr

def Table_structure_model():
    str_model_path = TSR_MODEL
    str_config_path = TSR_MODEL_CONFIG
    with open(str_config_path, 'r') as f:
        str_config = json.load(f)
    str_args = type('Args', (object,), str_config)
    str_args.device = 'cpu'
    str_model, _, _ = build_model(str_args)
    print("Table Structure model initialized.")

    str_model.load_state_dict(torch.load(str_model_path,
                                        map_location=torch.device('cpu')), strict=False)
    str_model.to('cpu')
    str_model.eval()
    return str_model

def Table_detection_model():
    det_model_path = TD_MODEL
    det_config_path = TD_MODEL_CONFIG
    with open(det_config_path, 'r') as f:
        det_config = json.load(f)
    det_args = type('Args', (object,), det_config)
    det_args.device = 'cpu'
    det_model, _, _ = build_model(det_args)
    print("Table Detection model initialized.")
    det_model.load_state_dict(torch.load(det_model_path,
                                        map_location=torch.device('cpu')), strict=False)
    det_model.to('cpu')
    det_model.eval()
    return det_model