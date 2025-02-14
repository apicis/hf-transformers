import os
import numpy as np
import ast
import pandas as pd
import albumentations as A
import cv2
import random
import torch

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoProcessor


class ImageCaptioningDatasetTriplet(Dataset):
    """Custom dataset with image, caption pairs"""

    def __init__(self, annotations_file, processor, augmentation, margin=10, text_input=None):
        annotations_temp = pd.read_csv(annotations_file)
        self.processor = processor
        self.augmentation = augmentation
        self.margin = margin
        self.text_input = text_input

        # Precompute all the paths to check
        image_paths = annotations_temp['filename']#.apply(lambda x: x.replace("/projects/simca/extracted_dataset/postprocessed_dataset", "/media/tapicella/Data/data"))

        # Efficiently identify rows with invalid paths
        valid_paths_mask = image_paths.apply(os.path.exists)

        # Filter out rows with invalid paths
        self.annotations = annotations_temp[valid_paths_mask].copy()
        self.episodes_id = annotations_temp['episode_id'][valid_paths_mask].copy()
        self.object_id = annotations_temp['object_id'][valid_paths_mask].copy()

    def __len__(self):
        return len(self.annotations)

    def select_examples(self, annotations, bounding_box, episode_id, object_id):
        """
        Select a positive and a negative example based on the criteria.

        Args:
            annotations (DataFrame): The full annotation dataset.
            episode_id (int): The episode_id of the current anchor image.
            object_id (int): The object_id of the current anchor image.

        Returns:
            positive_image, positive_caption, negative_image, negative_caption
        """

        # Select positive example (same episode_id and same object_id)
        positive_annotations = annotations[(annotations['bounding_box'] != f"{bounding_box}") &
                                           (annotations['episode_id'] == episode_id) &
                                           (annotations['object_id'] == object_id)]
        positive_example = positive_annotations.sample(1).iloc[0]
        positive_image = positive_example['filename']#.replace("/projects/simca/extracted_dataset/postprocessed_dataset", "/media/tapicella/Data/data")
        positive_caption = positive_example['caption']
        positive_bb = ast.literal_eval(positive_example['bounding_box'])
        positive_episode_id = positive_example['episode_id']
        positive_object_id = positive_example['object_id']

        # Select negative example (different episode_id and different object_id)
        negative_annotations = annotations[(annotations['episode_id'] != episode_id) &
                                           (annotations['object_id'] != object_id)]
        negative_example = negative_annotations.sample(1).iloc[0]
        negative_image = negative_example['filename']#.replace("/projects/simca/extracted_dataset/postprocessed_dataset", "/media/tapicella/Data/data")
        negative_caption = negative_example['caption']
        negative_bb = ast.literal_eval(negative_example['bounding_box'])
        negative_episode_id = negative_example['episode_id']
        negative_object_id = negative_example['object_id']

        return positive_image, positive_caption, positive_bb, positive_episode_id, positive_object_id, negative_image, negative_caption, negative_bb, negative_episode_id, negative_object_id

    def __getitem__(self, idx):

        # Get the current anchor image and caption
        anchor_example = self.annotations.iloc[idx]
        anchor_image_path = anchor_example['filename']#.replace("/projects/simca/extracted_dataset/postprocessed_dataset", "/media/tapicella/Data/data")
        anchor_caption = anchor_example['caption']
        anchor_bb = ast.literal_eval(anchor_example['bounding_box'])
        anchor_episode_id = anchor_example['episode_id']
        anchor_object_id = anchor_example['object_id']

        # Select positive and negative examples
        positive_image_path, positive_caption, positive_bb, positive_episode_id, positive_object_id, negative_image_path, negative_caption, negative_bb, negative_episode_id, negative_object_id = self.select_examples(
            self.annotations, anchor_bb, anchor_episode_id, anchor_object_id)

        # Load RGB
        anchor_array_original = np.load(anchor_image_path, allow_pickle=True)['arr_0'].item()['image']
        positive_array_original = np.load(positive_image_path, allow_pickle=True)['arr_0'].item()['image']
        negative_array_original = np.load(negative_image_path, allow_pickle=True)['arr_0'].item()['image']

        # Expand bounding box
        anchor_bb_original = [anchor_bb[0] - self.margin if (anchor_bb[0] - self.margin) >= 0 else 0,
                              anchor_bb[1] - self.margin if (anchor_bb[1] - self.margin) >= 0 else 0,
                              anchor_bb[2] + self.margin if (anchor_bb[2] + self.margin) < anchor_array_original.shape[
                                  0] else (anchor_array_original.shape[0] - 1),
                              anchor_bb[3] + self.margin if (anchor_bb[3] + self.margin) < anchor_array_original.shape[
                                  1] else (anchor_array_original.shape[1] - 1)]
        positive_bb_original = [positive_bb[0] - self.margin if (positive_bb[0] - self.margin) >= 0 else 0,
                                positive_bb[1] - self.margin if (positive_bb[1] - self.margin) >= 0 else 0,
                                positive_bb[2] + self.margin if (positive_bb[2] + self.margin) <
                                                                positive_array_original.shape[0] else (
                                            positive_array_original.shape[0] - 1),
                                positive_bb[3] + self.margin if (positive_bb[3] + self.margin) <
                                                                positive_array_original.shape[1] else (
                                            positive_array_original.shape[1] - 1)]
        negative_bb_original = [negative_bb[0] - self.margin if (negative_bb[0] - self.margin) >= 0 else 0,
                                negative_bb[1] - self.margin if (negative_bb[1] - self.margin) >= 0 else 0,
                                negative_bb[2] + self.margin if (negative_bb[2] + self.margin) <
                                                                negative_array_original.shape[0] else (
                                            negative_array_original.shape[0] - 1),
                                negative_bb[3] + self.margin if (negative_bb[3] + self.margin) <
                                                                negative_array_original.shape[1] else (
                                            negative_array_original.shape[1] - 1)]

        anchor_array = anchor_array_original.copy()
        positive_array = positive_array_original.copy()
        negative_array = negative_array_original.copy()
        anchor_bbox_exp = anchor_bb_original.copy()
        positive_bbox_exp = positive_bb_original.copy()
        negative_bbox_exp = negative_bb_original.copy()
        if self.augmentation:
            augmented = self.augmentation(image=anchor_array_original, bboxes=[anchor_bb_original])
            if len(augmented['bboxes']) != 0:
                anchor_array = augmented['image']
                anchor_bbox_exp = augmented['bboxes'][0]
            augmented = self.augmentation(image=positive_array_original, bboxes=[positive_bb_original])
            if len(augmented['bboxes']) != 0:
                positive_array = augmented['image']
                positive_bbox_exp = augmented['bboxes'][0]
            augmented = self.augmentation(image=negative_array_original, bboxes=[negative_bb_original])
            if len(augmented['bboxes']) != 0:
                anchor_array = augmented['image']
                anchor_bbox_exp = augmented['bboxes'][0]
            del augmented

        # Load the anchor, positive, and negative images
        anchor_image = Image.fromarray(anchor_array).convert('RGB')
        positive_image = Image.fromarray(positive_array).convert('RGB')
        negative_image = Image.fromarray(negative_array).convert('RGB')

        try:
            img_final = anchor_image.crop(anchor_bbox_exp)
            anchor_encoding = self.processor(images=img_final, return_tensors="pt")
        except:
            anchor_image = Image.fromarray(anchor_array_original).convert('RGB')
            img_final = anchor_image.crop(anchor_bb_original)
            anchor_encoding = self.processor(images=img_final, return_tensors="pt")

        try:
            img_final = positive_image.crop(positive_bbox_exp)
            positive_encoding = self.processor(images=img_final, return_tensors="pt")
        except:
            positive_image = Image.fromarray(positive_array_original).convert('RGB')
            img_final = positive_image.crop(positive_bb_original)
            positive_encoding = self.processor(images=img_final, return_tensors="pt")

        try:
            img_final = negative_image.crop(negative_bbox_exp)
            negative_encoding = self.processor(images=img_final, return_tensors="pt")
        except:
            negative_image = Image.fromarray(negative_array_original).convert('RGB')
            img_final = negative_image.crop(negative_bb_original)
            negative_encoding = self.processor(images=img_final, return_tensors="pt")

        # remove batch dimension
        anchor_encoding = {k: v.squeeze() for k, v in anchor_encoding.items()}
        anchor_encoding["text"] = anchor_caption
        anchor_encoding["episode_id"] = anchor_episode_id
        anchor_encoding["object_id"] = anchor_object_id
        anchor_encoding["img_array_original"] = anchor_array_original
        anchor_encoding["img_array"] = anchor_array
        anchor_encoding["bbox_exp_original"] = anchor_bb_original
        anchor_encoding["bbox_exp"] = anchor_bbox_exp

        positive_encoding = {k: v.squeeze() for k, v in positive_encoding.items()}
        positive_encoding["text"] = positive_caption
        positive_encoding["episode_id"] = positive_episode_id
        positive_encoding["object_id"] = positive_object_id
        positive_encoding["img_array_original"] = positive_array_original
        positive_encoding["img_array"] = positive_array
        positive_encoding["bbox_exp_original"] = positive_bb_original
        positive_encoding["bbox_exp"] = positive_bbox_exp

        negative_encoding = {k: v.squeeze() for k, v in negative_encoding.items()}
        negative_encoding["text"] = negative_caption
        negative_encoding["episode_id"] = negative_episode_id
        negative_encoding["object_id"] = negative_object_id
        negative_encoding["img_array_original"] = negative_array_original
        negative_encoding["img_array"] = negative_array
        negative_encoding["bbox_exp_original"] = negative_bb_original
        negative_encoding["bbox_exp"] = negative_bbox_exp

        return anchor_encoding, positive_encoding, negative_encoding


if __name__ == "__main__":
    random.seed(40)
    np.random.seed(40)
    torch.manual_seed(40)
    torch.cuda.manual_seed(40)

    csv_path = "/media/tapicella/Data/data/gibson_randomGoal_coca_mask2former_train.csv"
    augmentation = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.GaussNoise(std_range=(0.0, 0.05), mean_range=(0.0, 0.0), p=0.5),
        A.Affine(rotate=(-10.0, 10.0), shear=(-10.0, 10.0), scale=1, p=0.5)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b", model_max_length=512)

    dataset = ImageCaptioningDatasetTriplet(annotations_file=csv_path, processor=processor, augmentation=augmentation)
    dataset_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Visualise some samples
    for i, sample_batch in enumerate(tqdm(dataset_loader)):
        # Load data
        anchor_encoding, positive_encoding, negative_encoding = sample_batch

        anchor_array_original = anchor_encoding["img_array_original"]
        anchor_bbox_original = anchor_encoding["bbox_exp_original"]
        anchor_episode_id = anchor_encoding["episode_id"].item()
        anchor_object_id = anchor_encoding["object_id"].item()

        positive_array_original = positive_encoding["img_array_original"]
        positive_bbox_original = positive_encoding["bbox_exp_original"]
        positive_episode_id = positive_encoding["episode_id"].item()
        positive_object_id = positive_encoding["object_id"].item()

        negative_array_original = negative_encoding["img_array_original"]
        negative_bbox_original = negative_encoding["bbox_exp_original"]
        negative_episode_id = negative_encoding["episode_id"].item()
        negative_object_id = negative_encoding["object_id"].item()

        anchor_array_original = anchor_array_original.detach().numpy()[0]
        positive_array_original = positive_array_original.detach().numpy()[0]
        negative_array_original = negative_array_original.detach().numpy()[0]

        # Visualise
        img_vis = cv2.rectangle(cv2.cvtColor(anchor_array_original, cv2.COLOR_RGB2BGR),
                                          (int(anchor_bbox_original[0]), int(anchor_bbox_original[1])),
                                          (int(anchor_bbox_original[2]), int(anchor_bbox_original[3])), (0, 255, 0), 3)
        cv2.imshow("Anchor", cv2.resize(img_vis, (img_vis.shape[1] // 4, img_vis.shape[0] // 4)))

        img_vis = cv2.rectangle(cv2.cvtColor(positive_array_original, cv2.COLOR_RGB2BGR),
                                          (int(positive_bbox_original[0]), int(positive_bbox_original[1])),
                                          (int(positive_bbox_original[2]), int(positive_bbox_original[3])), (0, 255, 0), 3)
        cv2.imshow("Positive", cv2.resize(img_vis, (img_vis.shape[1] // 4, img_vis.shape[0] // 4)))

        img_vis = cv2.rectangle(cv2.cvtColor(negative_array_original, cv2.COLOR_RGB2BGR),
                                          (int(negative_bbox_original[0]), int(negative_bbox_original[1])),
                                          (int(negative_bbox_original[2]), int(negative_bbox_original[3])), (0, 255, 0), 3)
        cv2.imshow("Negative",
                   cv2.resize(img_vis, (img_vis.shape[1] // 4, img_vis.shape[0] // 4)))
        print(f"{anchor_episode_id=}")
        print(f"{anchor_object_id=}")
        print(f"{positive_episode_id=}")
        print(f"{positive_object_id=}")
        print(f"{negative_episode_id=}")
        print(f"{negative_object_id=}")
        cv2.waitKey(0)
