import os 
import torch
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import ast
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import pickle
import argparse

class ImageCaptioningDataset(Dataset):
    """Costum dataset with image, caption pairs"""
    
    def __init__(self, annotations_file, processor):
        self.annotations = pd.read_csv(annotations_file)
        self.processor = processor
        for ind, image in enumerate(self.annotations['filename']):
            if "/media/tapicella/Win11_OS/Users/tapicella/Downloads/gibson_dataset" in image:
                self.annotations['filename'][ind] = image.replace("/media/tapicella/Win11_OS/Users/tapicella/Downloads/gibson_dataset", "/media/tapicella/Win11_OS/Users/tapicella/Downloads/gibson_dataset")
            elif "/work/tgalliena/SImCa/data/sampled_images" in image:
                self.annotations['filename'][ind] = image.replace("/work/tgalliena/SImCa/data/sampled_images", "/media/tapicella/Win11_OS/Users/tapicella/Downloads/gibson_dataset")
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_path = self.annotations['filename'][idx]
        caption = self.annotations['caption'][idx]
        
        img_array = np.load(img_path)[:, :, :3]
        img = Image.fromarray(img_array).convert('RGB')
        
        bb = ast.literal_eval(self.annotations['bounding_box'][idx])
        bbox_1 = [bb[0] - 10 if (bb[0] - 10) >= 0 else 0,
            bb[1] - 10 if (bb[1] - 10) >= 0 else bb[1],
            bb[2] + 10 if (bb[2] + 10) <= img.size[0] else bb[2],
            bb[3] + 10 if (bb[3] + 10) <= img.size[1] else bb[3]]

        img_cropped = img.crop(bbox_1)
        
        encoding = self.processor(images=img_cropped, return_tensors="pt")
        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        encoding["text"] = caption
        return encoding

def collate_fn(batch):
    # pad the input_ids and attention_mask
    processed_batch = {}
    for key in batch[0].keys():
        if key != "text":
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            text_inputs = processor.tokenizer(
                [example["text"] for example in batch], padding=True,  return_tensors="pt"
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
    return processed_batch
           
def parse_arguments():
    parser = argparse.ArgumentParser(description="Finetune BLIP-2")
    parser.add_argument('--train-csv',
                        type=str,
                        default="/media/tapicella/Data/data/SImCa_test/fine_tuning/train_coca_ens_clip_gibson.csv",
                        help='Path to the CSV file with training sample')
    parser.add_argument('--val-csv',
                        type=str,
                        default="/media/tapicella/Data/data/SImCa_test/fine_tuning/val_coca_ens_clip_gibson.csv",
                        help='Path to the CSV file with training sample')
    parser.add_argument('--early-stopping',
                        action=argparse.BooleanOptionalAction,
                        help='Use early stopping',
                        default=True)
    parser.add_argument('--patience',
                        type=int,
                        help='Patience for early stopping',
                        default=10)
    parser.add_argument('--num-epochs',
                        type=int,
                        help='Patience for early stopping',
                        default=10)
    parser.add_argument('--output-path',
                        type=str,
                        help='Path to save the finetuned model')
    parser.add_argument('--save-interval',
                        type=int,
                        help='Interval of epochs between to savings',
                        default=1)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b", model_max_length=512)
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, device_map='auto')

    # Define the LoraConfig
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj"]
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    
    train_dataset = ImageCaptioningDataset(args.train_csv, processor)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64, collate_fn=collate_fn)
    
    val_dataset = ImageCaptioningDataset(args.val_csv, processor)
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=16, collate_fn=collate_fn)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_epochs = args.num_epochs
    patience = args.patience
    min_eval_loss = float("inf")
    early_stopping_hook = 0
    tracking_information = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        cap_loss = 0
        neg_loss = 0
        model.train()
        epoch = epoch + 1
        print("Epoch:", epoch)
        for idx, batch in zip(tqdm(range(len(train_dataloader)), desc='Training batch: ...'), train_dataloader):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device, torch.float16)
            attention_mask = batch.pop('attention_mask').to(device)
            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            attention_mask=attention_mask,
                            labels=input_ids,
                            return_dict=True)
            
            loss = outputs["loss"]
            epoch_loss += loss.item()
            cap_loss += outputs["loss_cap"].item()
            neg_loss += outputs["loss_neg"].item()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        print("Epoch: {} - Training loss: {} - Caption loss: {} - Negative loss: {}".format(epoch, epoch_loss/len(train_dataloader), cap_loss/len(train_dataloader), neg_loss/len(train_dataloader)))

        model.eval()
        eval_loss = 0
        for idx, batch in zip(tqdm(range(len(val_dataloader)), desc='Validating batch: ...'), val_dataloader):
            input_ids = batch.pop('input_ids').to(device)
            pixel_values = batch.pop('pixel_values').to(device)

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            labels=input_ids)
            
            loss = outputs.loss
            eval_loss += loss.item()

        tracking_information.append((epoch_loss/len(train_dataloader), eval_loss/len(val_dataloader), optimizer.param_groups[0]["lr"]))
        print("Epoch: {} - Training loss: {} - Eval Loss: {} - LR: {}".format(epoch, epoch_loss/len(train_dataloader), eval_loss/len(val_dataloader), optimizer.param_groups[0]["lr"]))
        
        if epoch % args.save_interval == 0:
            save_dir = f"{args.output_path}/checkpoint_epoch_{epoch}/"
            model.save_pretrained(save_dir, from_pt=True)
        if eval_loss >= min_eval_loss and args.early_stopping:
            early_stopping_hook += 1
            if early_stopping_hook > patience:
                break
    pickle.dump(tracking_information, open(f"{args.output_path}/tracking_information.pkl", "wb"))
    print("The finetuning process has done!")
