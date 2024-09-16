# shoplifting_detection/data/data_preparation.py

import os
import cv2
import random
from tqdm import tqdm

def extract_frames(video_path, output_dir, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_extract = min(num_frames, total_frames)

    frame_indices = sorted(random.sample(range(total_frames), frames_to_extract))

    extracted_frames = []
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            extracted_frames.append(frame)

    cap.release()
    return extracted_frames

def process_videos(raw_dir, processed_dir, annotation_file):
    os.makedirs(processed_dir, exist_ok=True)

    with open(annotation_file, 'w') as f:
        for video_file in tqdm(os.listdir(raw_dir)):
            if video_file.endswith('.mp4'):
                video_path = os.path.join(raw_dir, video_file)
                frames = extract_frames(video_path)

                output_path = os.path.join(processed_dir, f"{os.path.splitext(video_file)[0]}.npy")
                np.save(output_path, np.array(frames))

                # Randomly assign label (0: normal, 1: shoplifting)
                label = random.choice([0, 1])
                f.write(f"{output_path},{label}\n")

if __name__ == "__main__":
    raw_dir = "shoplifting_detection/data/raw_videos"
    processed_dir = "shoplifting_detection/data/processed_videos"
    annotation_file = "shoplifting_detection/data/annotations/train_annotations.txt"

    process_videos(raw_dir, processed_dir, annotation_file)