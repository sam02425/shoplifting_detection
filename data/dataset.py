import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

class ShopliftingDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None, num_frames=16):
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames

        with open(annotation_file, 'r') as f:
            self.annotations = f.readlines()

        self.label_dict = {'0': 0, '1': 1}  # 0 for Shoplifting, 1 for Normal

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        line = self.annotations[idx].strip().split(',')
        if len(line) != 2:
            raise ValueError(f"Invalid annotation format at line {idx}: {self.annotations[idx]}")

        video_path, label = line
        video_path = os.path.join(self.root_dir, video_path)

        if label not in self.label_dict:
            raise KeyError(f"Unknown label '{label}' at line {idx}: {self.annotations[idx]}")

        label = self.label_dict[label]

        frames = self._load_video(video_path)

        if self.transform:
            frames = [self.transform(frame) for frame in frames]
            frames = torch.stack(frames)

        return frames, label

    def _load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)

        cap.release()
        return frames  # Return as a list of frames