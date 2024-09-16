# shoplifting_detection/inference.py

import torch
import cv2
import numpy as np
import yaml
from easydict import EasyDict
from models.video_focalnet import VideoFocalNet
from models.text_generator import TextGenerator
from models.text_analyzer import TextAnalyzer

def preprocess_video(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)

    cap.release()
    return np.array(frames)

def inference(config, video_path):
    device = torch.device(config.DEVICE)

    # Load models
    video_model = VideoFocalNet(config.DATA.NUM_CLASSES, config.DATA.NUM_FRAMES).to(device)
    video_model.load_state_dict(torch.load(f"{config.OUTPUT_DIR}/best_model.pth"))
    text_generator = TextGenerator(config.MODEL.TEXT_GENERATOR.NAME).to(device)
    text_analyzer = TextAnalyzer(config.MODEL.TEXT_ANALYZER.NAME).to(device)

    video_model.eval()
    text_generator.eval()
    text_analyzer.eval()

    # Preprocess video
    frames = preprocess_video(video_path, config.DATA.NUM_FRAMES)
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float().unsqueeze(0).to(device)

    with torch.no_grad():
        video_outputs = video_model(frames)
        text_description = text_generator.generate(frames[:, 0])
        text_prediction = text_analyzer.analyze(text_description)

        combined_output = (video_outputs + text_prediction.unsqueeze(1).float()) / 2
        pred = torch.argmax(combined_output, dim=1)

    return pred.item(), text_description[0]

if __name__ == "__main__":
    with open("config/config.yaml", "r") as f:
        config = EasyDict(yaml.safe_load(f))

    video_path = "data/Shoplifting/test_video.mp4"
    prediction, description = inference(config, video_path)

    print(f"Video description: {description}")
    print(f"Prediction: {'Shoplifting' if prediction == 1 else 'Normal behavior'}")