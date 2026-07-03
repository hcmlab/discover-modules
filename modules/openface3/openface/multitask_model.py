import torch
import numpy as np
import cv2
from torchvision import transforms
from typing import Tuple


from openface.model.MTL import MTL


class MultitaskPredictor:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = MTL().to(self.device)
        self._load_model(model_path)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),  
            transforms.Resize((224, 224)),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])

    def _load_model(self, model_path: str):
        print(f"Loading multitask model from {model_path}...")
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def preprocess(self, face: np.ndarray) -> torch.Tensor:
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_tensor = self.transform(face_rgb).unsqueeze(0).to(self.device)
        return face_tensor

    def predict(self, face: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Preprocess the input face
        face_tensor = self.preprocess(face)

        # Perform multitasking predictions
        with torch.no_grad():
            emotion_output, gaze_output, au_output = self.model(face_tensor)
        return emotion_output, gaze_output, au_output
