# shoplifting_detection/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import yaml
from easydict import EasyDict
from tqdm import tqdm

from data.dataloader import create_dataloaders
from models.video_focalnet import VideoFocalNet
from models.text_generator import TextGenerator
from models.text_analyzer import TextAnalyzer
from utils.logger import setup_logger
from utils.metrics import accuracy, precision_recall_fscore

def train(config):
    logger = setup_logger(config.OUTPUT_DIR)
    writer = SummaryWriter(config.OUTPUT_DIR)

    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"{key}: {value}")

    train_loader, val_loader = create_dataloaders(config)

    video_model = VideoFocalNet(config.DATA.NUM_CLASSES, config.DATA.NUM_FRAMES).to(config.DEVICE)
    text_generator = TextGenerator(config.MODEL.TEXT_GENERATOR.NAME).to(config.DEVICE)
    text_analyzer = TextAnalyzer(config.MODEL.TEXT_ANALYZER.NAME).to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(video_model.parameters(), lr=config.TRAIN.LEARNING_RATE, weight_decay=config.TRAIN.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.TRAIN.NUM_EPOCHS)

    best_acc = 0
    scaler = GradScaler()

    for epoch in range(config.TRAIN.NUM_EPOCHS):
        video_model.train()
        text_generator.eval()
        text_analyzer.eval()

        train_loss = 0
        train_acc = 0
        optimizer.zero_grad()
        for i, (videos, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.TRAIN.NUM_EPOCHS}")):
            videos = videos.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            with autocast():
                # Video model prediction
                video_outputs = video_model(videos)

                # Generate text descriptions (use only the first frame for simplicity)
                with torch.no_grad():
                    text_descriptions = text_generator.generate(videos[:, 0])

                # Analyze text with prompt-based model
                with torch.no_grad():
                    text_predictions = text_analyzer.analyze(text_descriptions)

                # Combine video and text predictions
                combined_outputs = (video_outputs + text_predictions) / 2

                loss = criterion(combined_outputs, labels)
                loss = loss / config.TRAIN.ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * config.TRAIN.ACCUMULATION_STEPS
            train_acc += accuracy(combined_outputs, labels)

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        val_loss, val_acc = validate(config, video_model, text_generator, text_analyzer, val_loader, criterion)

        logger.info(f"Epoch {epoch+1}/{config.TRAIN.NUM_EPOCHS}, "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(video_model.state_dict(), f"{config.OUTPUT_DIR}/best_model.pth")

    writer.close()

def validate(config, video_model, text_generator, text_analyzer, val_loader, criterion):
    video_model.eval()
    text_generator.eval()
    text_analyzer.eval()

    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for videos, labels in val_loader:
            videos = videos.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            # Video model prediction
            video_outputs = video_model(videos)

            # Generate text descriptions (use only the first frame for simplicity)
            text_descriptions = text_generator.generate(videos[:, 0])

            # Analyze text with prompt-based model
            text_predictions = text_analyzer.analyze(text_descriptions)

            # Ensure text_predictions has the same batch size as video_outputs
            if text_predictions.size(0) == 1:
                text_predictions = text_predictions.expand(video_outputs.size(0), -1)

            # Combine video and text predictions
            combined_outputs = (video_outputs + text_predictions) / 2

            loss = criterion(combined_outputs, labels)
            val_loss += loss.item()
            val_acc += accuracy(combined_outputs, labels)

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    return val_loss, val_acc

if __name__ == "__main__":
    with open("config/config.yaml", "r") as f:
        config = EasyDict(yaml.safe_load(f))
    train(config)