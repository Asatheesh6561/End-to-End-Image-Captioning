import torch
from torch.utils.data import *
from imageCaptioningWithAttention.components.utils import *
from tqdm import tqdm
import os
from imageCaptioningWithAttention.components.models import *
class Trainer():
    def __init__(self, train_dataset, val_dataset, learning_rate, lr_backbone, weight_decay, lr_drop, batch_size, num_workers, num_epochs, checkpoint, clip_max_norm):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.lr = learning_rate
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.lr_drop = lr_drop
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.checkpoint = checkpoint
        self.clip_max_norm = clip_max_norm
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 

    def train_model(self, model, criterion, model_save_dir):
        model = model.to(self.device)
        print(f'Device: {self.device}')
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Number of parameters: {n_parameters}')
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": self.lr_backbone}
        ]
        optimizer = torch.optim.AdamW(param_dicts, self.lr, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.lr_drop)
        sampler_train = RandomSampler(self.train_dataset)
        sampler_val = SequentialSampler(self.val_dataset)
        batch_sampler_train = BatchSampler(sampler_train, self.batch_size, drop_last=True)
        data_loader_train = DataLoader(self.train_dataset, batch_sampler=batch_sampler_train, num_workers=self.num_workers)
        data_loader_val = DataLoader(self.val_dataset, self.batch_size, sampler=sampler_val, drop_last=False, num_workers=self.num_workers)
        start_epoch = 0
        if os.path.exists(self.checkpoint):
            print('Loading Checkpoint')
            chkpt = torch.load(self.checkpoint, map_location='cpu')
            model.load_state_dict(chkpt['model'])
            optimizer.load_state_dict(chkpt['optimizer'])
            lr_scheduler.load_state_dict(chkpt['lr_scheduler'])
            start_epoch = chkpt['epoch'] + 1
        print('Start Training')
        for epoch in range(start_epoch, self.num_epochs):
            print(f'Epoch: {epoch}')
            epoch_loss = self.train_one_epoch(model, criterion, data_loader_train, optimizer, self.device, self.clip_max_norm)
            lr_scheduler.step()
            print(f'Training Loss: {epoch_loss}')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch
            }, os.path.join(model_save_dir, f'model_{epoch}.pth'))
            if (epoch+1) % 10 == 0:
                validation_loss = self.evaluate(model, criterion, data_loader_val, self.device)
                print(f"Validation Loss: {validation_loss}")

    def train_one_epoch(self, model, criterion, data_loader, optimizer, device, max_norm):
        model.train()
        criterion.train()
        epoch_loss = 0.0
        total = len(data_loader)
        with tqdm(total=total) as bar:
            for images, masks, captions, caption_masks in data_loader:
                samples = NestedTensor(images, masks).to(device)
                captions, caption_masks = captions.to(device), caption_masks.to(device)
                outputs = model(samples, captions[:, :-1], caption_masks[:, :-1])
                loss = criterion(outputs.permute(0, 2, 1), captions[:, 1:].long())
                loss_value = loss.item()
                epoch_loss += loss_value

                optimizer.zero_grad()
                loss.backward()
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                bar.update(1)
        return epoch_loss / total        

    def evaluate(self, model, criterion, data_loader, device):
        model.eval()
        criterion.eval()

        validation_loss = 0.0
        total = len(data_loader)

        with tqdm(total=total) as pbar:
            for images, masks, caps, cap_masks in data_loader:
                samples = NestedTensor(images, masks).to(device)
                caps = caps.to(device)
                cap_masks = cap_masks.to(device)

                outputs = model(samples, caps[:, :-1], cap_masks[:, :-1])
                loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:].long())

                validation_loss += loss.item()
                pbar.update(1)
            
        return validation_loss / total
    
    def get_caption_and_mask(self, start_token, max_length):
        caption_template = torch.zeros((1, max_length), dtype=torch.long)
        mask_template = torch.ones((1, max_length), dtype=torch.bool)

        caption_template[:, 0] = start_token
        mask_template[:, 0] = False

        return caption_template, mask_template
        
    def evaluate_caption(self, model, image, caption, caption_mask, max_length):
        model.eval()
        for i in range(max_length - 1):
            predictions = model(image, caption, caption_mask)
            predictions = predictions[:, i, :]
            predicted_id = torch.argmax(predictions, axis=-1)

            if predicted_id[0] == 102:
                return caption

            caption[:, i+1] = predicted_id[0]
            caption_mask[:, i+1] = False

        return caption
    
    