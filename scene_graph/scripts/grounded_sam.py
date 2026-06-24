import torch
import threading
from PIL import Image
import numpy as np
from typing import Any, Dict

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import (
    sam_model_registry,
    SamPredictor
)


MODEL_CONFIG_PATH = "/home/jjiang/jing/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
MODEL_CHECKPOINT_PATH = "/home/jjiang/jing/Grounded-Segment-Anything/groundingdino_swint_ogc.pth"
SAM_VERSION = "vit_b"
SAM_CHECKPOINT_PATH = "/home/jjiang/jing/Grounded-Segment-Anything/sam_vit_b_01ec64.pth"

class GroundedSAM:
    def __init__(self, config_path=MODEL_CONFIG_PATH, checkpoint_path=MODEL_CHECKPOINT_PATH, device="cuda:0"):
        self.device = device
        self.model = self.load_model(config_path, checkpoint_path, device)
        self.sam = sam_model_registry[SAM_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        self.sam.to(device).eval()
        self.predictor = SamPredictor(self.sam)
        self._segment_lock = threading.Lock()


    def load_model(self, 
                   model_config_path,
                   model_checkpoint_path, 
                   device,
                   ):
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        return model.to(device).eval()
    
    def load_image(self, image):
        image_pil = Image.fromarray(image)
        transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image
    
    
    def get_grounding_output(self, model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        image = image.to(device)
        with torch.inference_mode():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases

    def segment(self, model, image: np.ndarray, prompt: str, box_threshold, text_threshold, device, with_logits=True):
        with self._segment_lock:
            return self._segment_unlocked(
                model,
                image,
                prompt,
                box_threshold,
                text_threshold,
                device,
                with_logits=with_logits,
            )

    def _segment_unlocked(self, model, image: np.ndarray, prompt: str, box_threshold, text_threshold, device, with_logits=True):
        image_pil, image_tensor = self.load_image(image)
        boxes, phrases = self.get_grounding_output(
            model,
            image_tensor,
            prompt,
            box_threshold,
            text_threshold,
            with_logits=with_logits,
            device=device,
        )
        size = image_pil.size
        H, W = size[1], size[0]
        if boxes.size(0) == 0:
            return np.empty((0, 1, H, W), dtype=bool), phrases

        self.predictor.set_image(image)
        for i in range(boxes.size(0)):
            boxes[i] = boxes[i] * torch.Tensor([W, H, W, H])
            boxes[i][:2] -= boxes[i][2:] / 2
            boxes[i][2:] = boxes[i][:2] + boxes[i][2:]

        boxes = boxes.cpu()
        transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes, image.shape[:2]).to(device)

        try:
            with torch.inference_mode():
                masks, _, _ = self.predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
            masks = masks.detach().cpu().numpy().astype(bool)
        finally:
            self._reset_predictor_image()

        return masks, phrases

    def _reset_predictor_image(self):
        if hasattr(self.predictor, "reset_image"):
            self.predictor.reset_image()
            return

        self.predictor.features = None
        self.predictor.is_image_set = False
        self.predictor.orig_h = None
        self.predictor.orig_w = None
        self.predictor.input_h = None
        self.predictor.input_w = None

    def post_process(self, outputs) -> Dict[str, Any]:
        # Implement post-processing logic to convert model outputs into desired format
        pass
