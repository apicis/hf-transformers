import torch
import numpy as np
import random
import os
import hydra
import wandb
import torch.distributed as dist
import albumentations as A

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoProcessor, Blip2ForConditionalGeneration, Florence2ForConditionalGeneration
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.distributed import init_process_group, destroy_process_group
from transformers.image_captioning_dataset import ImageCaptioningDataset
from transformers.image_captioning_dataset_triplet import ImageCaptioningDatasetTriplet


def ddp_setup():
    """
        This function sets up the distributed data parallel environment.
        It is used in the main function to set the rank, local rank and world size
    device
        Returns: rank, local_rank, world_size
    """
    if os.environ.get("OMPI_COMMAND"):
        # from mpirun
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    else:
        # from slurm
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NPROCS"])

    return rank, local_rank, world_size


def collate_fn(batch):
    # pad the input_ids and attention_mask
    processed_batch = {}
    for key in batch[0].keys():
        if key != "text":
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            text_inputs = processor.tokenizer(
                [example["text"] for example in batch], padding=True, return_tensors="pt"
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
    return processed_batch


def collate_fn_triplet(batch):
    # pad the input_ids and attention_mask
    keys = ["pixel_values", "input_ids", "attention_mask"]
    processed_batch = {key: [] for key in keys}
    temp_value = []
    for b in batch:
        for key in b[0].keys():
            anchor_value = b[0][key]
            positive_value = b[1][key]
            negative_value = b[2][key]
            if key == "pixel_values":
                processed_batch[key].append(anchor_value)
                processed_batch[key].append(positive_value)
                processed_batch[key].append(negative_value)
            elif key == "text":
                temp_value.append(anchor_value)
                temp_value.append(positive_value)
                temp_value.append(negative_value)
    text_inputs = processor.tokenizer(temp_value, padding=True, return_tensors="pt")
    processed_batch["input_ids"] = text_inputs["input_ids"]
    processed_batch["attention_mask"] = text_inputs["attention_mask"]
    processed_batch["pixel_values"] = torch.stack(processed_batch["pixel_values"],dim=0)
    return processed_batch


class Trainer:
    def __init__(self, model,
                 num_epochs,
                 train_dataloader,
                 val_dataloader,
                 optimizer,
                 device,
                 use_wandb,
                 use_negative,
                 use_triplet,
                 save_interval,
                 run,
                 early_stopping,
                 patience,
                 multigpu):
        self.model = model
        self.num_epochs = num_epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.device = device
        self.use_wandb = use_wandb
        self.use_negative = use_negative
        self.use_triplet = use_triplet
        self.save_interval = save_interval
        self.run = run
        self.early_stopping = early_stopping
        self.patience = patience
        self.multigpu = multigpu

        if self.device == 0 or self.device == "cuda":
            if self.use_wandb:
                self.dest_dir = os.path.join(os.getcwd(), "checkpoints", run.project, run.id)
            else:
                self.dest_dir = os.path.join(os.getcwd(), "checkpoints", "blip2")
            os.makedirs(self.dest_dir, exist_ok=True)

    def training_loop(self):
        min_eval_loss = float('inf')
        early_stopping_hook = 0
        for epoch in range(1, self.num_epochs + 1):

            # One epoch training loop
            self.model.train()
            if self.multigpu:
                self.train_dataloader.sampler.set_epoch(epoch)
            self.train_one_epoch(self.model, self.train_dataloader, epoch, self.optimizer, self.device, self.use_wandb,
                                 self.use_negative, self.use_triplet, self.multigpu)
            torch.cuda.empty_cache()

            # One epoch validation loop
            self.model.eval()
            if self.multigpu:
                self.val_dataloader.sampler.set_epoch(epoch)
            eval_loss = self.validate_one_epoch(self.model, self.val_dataloader, epoch, self.device, self.use_wandb,
                                                self.multigpu)
            torch.cuda.empty_cache()

            save_logs = True
            if self.early_stopping:
                if eval_loss >= min_eval_loss:
                    save_logs = False
                    early_stopping_hook += 1
                    if early_stopping_hook >= self.patience:
                        print("Early stopping!")
                        break
                else:
                    min_eval_loss = eval_loss
                    early_stopping_hook = 0
                    save_logs = True

            if save_logs:
                # Save checkpoint
                if epoch % self.save_interval == 0 and (self.device == 0 or self.device == "cuda"):
                    save_path = os.path.join(self.dest_dir, f"checkpoint_{epoch}.pt")
                    if self.multigpu:
                        self.model.module.save_pretrained(save_path, from_pt=True)
                    else:
                        self.model.save_pretrained(save_path, from_pt=True)
                    print("Saving model at {}".format(save_path))

    def train_one_epoch(self, model, train_dataloader, epoch, optimizer, device, use_wandb, use_negative, use_triplet, multigpu):
        epoch_loss = 0
        if use_negative or use_triplet:
            cap_loss = 0
            if use_negative:
                neg_loss = 0
            if use_triplet:
                triplet_loss = 0
        train_dataloader_len = len(train_dataloader)
        for _, batch in enumerate(tqdm(train_dataloader)):
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device, torch.float16)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            attention_mask=attention_mask,
                            labels=input_ids,
                            return_dict=True)

            optimizer.zero_grad()
            loss = outputs["loss"]
            if use_negative or use_triplet:
                cap_loss += outputs["loss_cap"]
                if use_negative:
                    neg_loss += outputs["loss_neg"]
                if use_triplet:
                    triplet_loss += outputs["loss_trip"]
            epoch_loss += loss
            loss.backward()
            optimizer.step()

            del outputs, input_ids, pixel_values, attention_mask, loss, batch
        train_dataloader_len = torch.tensor(train_dataloader_len, dtype=torch.int, device=device)
        if multigpu:
            dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
            if use_negative or use_triplet:
                dist.all_reduce(cap_loss, op=dist.ReduceOp.SUM)
                if use_negative:
                    dist.all_reduce(neg_loss, op=dist.ReduceOp.SUM)
                if use_triplet:
                    dist.all_reduce(triplet_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_dataloader_len, op=dist.ReduceOp.SUM)

        if device == 0 or device == "cuda":
            epoch_total_loss = (epoch_loss / train_dataloader_len).item()
            msg = "Epoch: {} - Training loss: {}".format(epoch, epoch_total_loss)
            wandb_dict = {"train/epoch_loss": epoch_total_loss, "train/epoch": epoch}
            if use_negative or use_triplet:
                epoch_cap_loss = (cap_loss / train_dataloader_len).item()
                msg += "- Caption loss: {} ".format(epoch_cap_loss)
                wandb_dict["train/epoch_caption_loss"] = epoch_cap_loss
                if use_negative:
                    epoch_neg_loss = (neg_loss / train_dataloader_len).item()
                    msg += "- Negative loss: {}".format(epoch_neg_loss)
                    wandb_dict["train/epoch_negative_loss"] = epoch_neg_loss
                if use_triplet:
                    epoch_triplet_loss = (triplet_loss / train_dataloader_len).item()
                    msg += "- Triplet loss: {}".format(epoch_triplet_loss)
                    wandb_dict["train/epoch_triplet_loss"] = epoch_triplet_loss
            print(msg)
            if use_wandb:
                wandb.log(wandb_dict)
            del epoch_total_loss
        del train_dataloader_len, epoch_loss
        torch.cuda.empty_cache()

    @torch.no_grad()
    def validate_one_epoch(self, model, val_dataloader, epoch, device, use_wandb, multigpu):
        eval_loss = 0
        val_dataloader_len = len(val_dataloader)
        for _, batch in enumerate(tqdm(val_dataloader)):
            input_ids = batch['input_ids'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_ids=input_ids,
                                pixel_values=pixel_values,
                                labels=input_ids,
                                return_dict=True)

            eval_loss += outputs["loss"]
            del outputs, input_ids, pixel_values, batch
        val_dataloader_len = torch.tensor(val_dataloader_len, dtype=torch.int, device=device)
        if multigpu:
            dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
        if device == 0 or device == "cuda":
            epoch_total_loss = (eval_loss / val_dataloader_len).item()
            print("Epoch: {} - Validation loss: {}".format(epoch, epoch_total_loss))
            if use_wandb:
                wandb.log({"val/epoch_loss": epoch_total_loss, "val/epoch": epoch})
            del epoch_total_loss
        del val_dataloader_len
        torch.cuda.empty_cache()
        return eval_loss


@hydra.main(config_path="configs", config_name="train_configs")
def main(config: DictConfig):
    OmegaConf.resolve(config)

    # Load configs into variables
    use_wandb = config["wandb"]["use_wandb"]
    model_name = config["model"]["model_name"]
    text_input = config["model"]["text_input"]
    model_max_length = config["model"]["model_max_length"]
    train_csv = config["dataset"]["train_csv"]
    val_csv = config["dataset"]["val_csv"]
    use_augmentation = config["dataset"]["use_augmentation"]
    seed = config["training_setup"]["seed"]
    batch_size = config["training_setup"]["batch_size"]
    num_workers = config["training_setup"]["num_workers"]
    num_epochs = config["training_setup"]["epochs"]
    save_interval = config["training_setup"]["save_interval"]
    patience = config["training_setup"]["patience"]
    learning_rate = config["training_setup"]["learning_rate"]
    early_stopping = config["training_setup"]["early_stopping"]
    multigpu = config["training_setup"]["multigpu"]
    use_triplet = config["training_setup"]["use_triplet"]
    use_negative = config["training_setup"]["use_negative"]
    triplet_loss_weight = config["training_setup"]["triplet_loss_weight"]
    negative_loss_weight = config["training_setup"]["negative_loss_weight"]

    # Initialize seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Initialize device
    if multigpu:
        rank, local_rank, world_size = ddp_setup()
        device = local_rank
        # init the distributed process group
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    run = None
    if use_wandb and (device == 0 or device == "cuda"):
        print(OmegaConf.to_yaml(config))
        wandb_proj = config["wandb"]["project"]
        wandb_entity = config["wandb"]["entity"]
        wandb_name = config["wandb"]["display_name"]
        RUN_CONFIG = OmegaConf.to_container(config)
        run = wandb.init(
            project=wandb_proj,
            entity=wandb_entity,
            name=wandb_name,
            config=RUN_CONFIG,
        )
    del config

    # Select model and processor
    global processor
    if "blip2" in model_name:
        processor = AutoProcessor.from_pretrained(model_name, model_max_length=model_max_length)
        model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16,
                                                              device_map=device,
                                                              use_triplet=use_triplet,
                                                              use_negative=use_negative,
                                                              caption_loss_weight=1.0,
                                                              triplet_loss_weight=triplet_loss_weight,
                                                              negative_loss_weight=negative_loss_weight)  # device_map='auto' allocates resources for the model automatically
    elif "Florence-2" in model_name:
        processor = AutoProcessor.from_pretrained(model_name, model_max_length=model_max_length, trust_remote_code=True)
        model = Florence2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16,
                                                                  use_negative=use_negative).to(
            device)
    # Define the LoraConfig
    config_lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj"]
    )

    model = get_peft_model(model, config_lora)
    model.print_trainable_parameters()

    if multigpu:
        model = DDP(model, device_ids=[device])

    # Load dataset
    augmentation = None
    if use_augmentation:
        augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.GaussNoise(std_range=(0.0, 0.05), mean_range=(0.0, 0.0), p=0.5),
            A.Affine(rotate=(-10.0, 10.0), shear=(-10.0, 10.0), scale=1, p=0.5)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))

    if use_triplet:
        train_dataset = ImageCaptioningDatasetTriplet(train_csv, processor, augmentation, text_input=text_input)
    else:
        train_dataset = ImageCaptioningDataset(train_csv, processor, augmentation, text_input=text_input)

    val_dataset = ImageCaptioningDataset(val_csv, processor, augmentation=None, text_input=text_input)

    if use_triplet:
        collate_function = collate_fn_triplet
    else:
        collate_function = collate_fn

    if multigpu:
        train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers,
                                      sampler=DistributedSampler(train_dataset), collate_fn=collate_function)
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers,
                                    sampler=DistributedSampler(val_dataset), collate_fn=collate_function)
    else:
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                      collate_fn=collate_function)
        val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                    collate_fn=collate_function)
    print("Dataset built!")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    trainer = Trainer(model=model,
                      num_epochs=num_epochs,
                      train_dataloader=train_dataloader,
                      val_dataloader=val_dataloader,
                      optimizer=optimizer,
                      device=device,
                      use_wandb=use_wandb,
                      use_negative=use_negative,
                      use_triplet=use_triplet,
                      save_interval=save_interval,
                      run=run,
                      early_stopping=early_stopping,
                      patience=patience,
                      multigpu=multigpu)
    trainer.training_loop()
    if multigpu:
        destroy_process_group()
    print("The finetuning process has done!")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
