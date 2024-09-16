# shoplifting_detection/data/dataloader.py

from torch.utils.data import DataLoader
from .dataset import ShopliftingDataset
import torchvision.transforms as transforms

def create_dataloaders(config):
    # Default image size if not specified in config
    img_size = getattr(config.DATA, 'IMG_SIZE', 224)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ShopliftingDataset(
        root_dir=config.DATA.ROOT,
        annotation_file=config.DATA.TRAIN_FILE,
        transform=transform,
        num_frames=config.DATA.NUM_FRAMES
    )

    val_dataset = ShopliftingDataset(
        root_dir=config.DATA.ROOT,
        annotation_file=config.DATA.VAL_FILE,
        transform=transform,
        num_frames=config.DATA.NUM_FRAMES
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, val_loader