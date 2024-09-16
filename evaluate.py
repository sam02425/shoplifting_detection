# This is the evaluate.py file in the shoplifting_detection\ directory
import torch
import yaml
from easydict import EasyDict
from tqdm import tqdm

from data.dataloader import create_dataloaders
from models.video_focalnet import VideoFocalNet
from models.text_generator import TextGenerator
from models.text_analyzer import TextAnalyzer
from utils.logger import setup_logger
from utils.metrics import accuracy, precision_recall_fscore

def evaluate(config):
    logger = setup_logger(config.OUTPUT_DIR, filename="eval.log")

    _, val_loader = create_dataloaders(config)

    video_model = VideoFocalNet(config.DATA.NUM_CLASSES, config.DATA.NUM_FRAMES).to(config.DEVICE)
    video_model.load_state_dict(torch.load(f"{config.OUTPUT_DIR}/best_model.pth"))
    text_generator = TextGenerator(config.MODEL.TEXT_GENERATOR.NAME).to(config.DEVICE)
    text_analyzer = TextAnalyzer(config.MODEL.TEXT_ANALYZER.NAME).to(config.DEVICE)

    video_model.eval()
    text_generator.eval()
    text_analyzer.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for videos, labels in tqdm(val_loader, desc="Evaluating"):
            videos = videos.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            video_outputs = video_model(videos)
            text_descriptions = text_generator.generate(videos[:, 0])
            text_predictions = text_analyzer.analyze(text_descriptions)

            combined_outputs = (video_outputs + text_predictions.unsqueeze(1).float()) / 2
            preds = torch.argmax(combined_outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy(torch.tensor(all_preds), torch.tensor(all_labels))
    precision, recall, f1 = precision_recall_fscore(torch.tensor(all_preds), torch.tensor(all_labels))

    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-score: {f1:.4f}")

if __name__ == "__main__":
    with open("config/config.yaml", "r") as f:
        config = EasyDict(yaml.safe_load(f))
    evaluate(config)