import numpy as np
import ast
import pandas as pd
import albumentations as A
import cv2
import random
import torch
import os

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoProcessor


class ImageCaptioningDataset(Dataset):
    """Custom dataset with image, caption pairs"""

    def __init__(self, annotations_file, processor, augmentation, margin=10, text_input=None):
        annotations_temp = pd.read_csv(annotations_file)
        self.processor = processor
        self.augmentation = augmentation
        self.margin = margin
        self.text_input = text_input

        # Precompute all the paths to check
        image_paths = annotations_temp['filename'].apply(lambda x: x.replace(
            "/projects/simca/extracted_dataset/postprocessed_dataset", "/media/tapicella/Data/data"))

        # Efficiently identify rows with invalid paths
        valid_paths_mask = image_paths.apply(os.path.exists)

        # Filter out rows with invalid paths
        self.annotations = annotations_temp[valid_paths_mask].copy()
        self.episodes_id = annotations_temp['episode_id'][valid_paths_mask].copy()
        self.object_id = annotations_temp['object_id'][valid_paths_mask].copy()

        print("Dataset built!")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation_example = self.annotations.iloc[idx]
        img_path = annotation_example['filename'].replace("/projects/simca/extracted_dataset/postprocessed_dataset", "/media/tapicella/Data/data")
        print(img_path)
        caption = annotation_example['caption']
        bb = ast.literal_eval(annotation_example['bounding_box'])

        # img_array_original = np.load(img_path)[:, :, :3]
        img_array_original = np.load(img_path, allow_pickle=True)['arr_0'].item()['image']
        bbox_exp_original = [bb[0] - self.margin if (bb[0] - self.margin) >= 0 else 0,
                    bb[1] - self.margin if (bb[1] - self.margin) >= 0 else 0,
                    bb[2] + self.margin if (bb[2] + self.margin) < img_array_original.shape[0] else (img_array_original.shape[0] - 1),
                    bb[3] + self.margin if (bb[3] + self.margin) < img_array_original.shape[1] else (img_array_original.shape[1] - 1)]

        img_array = img_array_original.copy()
        bbox_exp = bbox_exp_original.copy()
        if self.augmentation:
            augmented = self.augmentation(image=img_array_original, bboxes=[bbox_exp_original])
            if len(augmented['bboxes']) != 0:
                img_array = augmented['image']
                bbox_exp = augmented['bboxes'][0]

        img = Image.fromarray(img_array).convert('RGB')

        if self.text_input:
            prompt = self.text_input + "<loc_{}><loc_{}><loc_{}><loc_{}>".format(int(bbox_exp[0]), int(bbox_exp[1]),
                                                                                 int(bbox_exp[2]), int(
                    bbox_exp[3])) if self.text_input == '<REGION_TO_DESCRIPTION>' else self.text_input
            try:
                img_final = img if self.text_input == '<REGION_TO_DESCRIPTION>' else img.crop(bbox_exp)
                encoding = self.processor(text=prompt, images=img_final, return_tensors="pt")
            except:
                img = Image.fromarray(img_array_original).convert('RGB')
                img_final = img if self.text_input == '<REGION_TO_DESCRIPTION>' else img.crop(bbox_exp_original)
                encoding = self.processor(text=prompt, images=img_final, return_tensors="pt")
        else:
            try:
                img_final = img.crop(bbox_exp)
                encoding = self.processor(images=img_final, return_tensors="pt")
            except:
                img = Image.fromarray(img_array_original).convert('RGB')
                img_final = img.crop(bbox_exp_original)
                encoding = self.processor(images=img_final, return_tensors="pt")

        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = caption
        return encoding # , img_array_original, img_array, bbox_exp_original, bbox_exp


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    csv_path = "/media/tapicella/Data/data/SImCa_test/fine_tuning/train_coca_ens_clip_gibson.csv"
    augmentation = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.GaussNoise(std_range=(0.0, 0.05), mean_range=(0.0, 0.0), p=0.5),
        A.Affine(rotate=(-10.0, 10.0), shear=(-10.0, 10.0), scale=1, p=0.5)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b", model_max_length=512)

    dataset = ImageCaptioningDataset(annotations_file=csv_path, processor=processor, augmentation=augmentation)
    dataset_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Visualise some samples
    for i, sample_batch in enumerate(dataset_loader):
        # Load data
        enc, img_array_original, img_array, bbox_exp_original, bbox_exp = sample_batch

        img_array_original = img_array_original.detach().numpy()[0]
        img_array = img_array.cpu().detach().numpy()[0]

        # Visualise
        cv2.imshow("RGB original", cv2.cvtColor(img_array_original, cv2.COLOR_RGB2BGR))
        cv2.imshow("RGB", cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        img_rect_original = cv2.rectangle(cv2.cvtColor(img_array_original, cv2.COLOR_RGB2BGR),(int(bbox_exp_original[0]),int(bbox_exp_original[1])),(int(bbox_exp_original[2]),int(bbox_exp_original[3])),(0,255,0),3)
        img_rect = cv2.rectangle(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR),(int(bbox_exp[0]),int(bbox_exp[1])),(int(bbox_exp[2]),int(bbox_exp[3])),(0,255,0),3)
        cv2.imshow("Bbox original", img_rect_original)
        cv2.imshow("Bbox", img_rect)
        cv2.waitKey(10)
