"""
Created on October 27, 2025.
Prediction_tta_lnconvnext.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from config.serde import read_config


class PredictionTTA:
    """
    Prediction helper for HF DINOv3 ConvNeXt models.
    Assumes forward returns .pooler_output and head is attached as model.head.
    """
    def __init__(self, cfg_path, label_names):
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.label_names = label_names
        self.setup_cuda()

    def setup_cuda(self, cuda_device_id=0):
        if torch.cuda.is_available():
            torch.backends.cudnn.fastest = True
            torch.cuda.set_device(cuda_device_id)
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def setup_model_from_path(self, model, weights_path: str):
        self.model = model.to(self.device)
        state = torch.load(weights_path, map_location="cpu")
        self.model.load_state_dict(state, strict=True)

    def predict_only(self, test_loader):
        self.model.eval()
        preds_cache = torch.Tensor([]).to(self.device)
        labels_cache = torch.Tensor([]).to(self.device)

        for _, (image, label) in enumerate(tqdm(test_loader)):
            image = image.to(self.device)
            label = label.to(self.device).float()

            with torch.no_grad():
                out = self.model(image)
                logits = self.model.head(out.pooler_output)  # HF ConvNeXt (DINOv3) pattern
                probs = torch.sigmoid(logits)

                preds_cache = torch.cat((preds_cache, probs))
                labels_cache = torch.cat((labels_cache, label))

        return preds_cache, labels_cache
