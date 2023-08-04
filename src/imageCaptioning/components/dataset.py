import os
from torch.utils.data import Dataset
from transformers import BertTokenizer
from PIL import Image
from imageCaptioning.components.utils import *
import numpy as np
class ImageCaptionDataset(Dataset):
    def __init__(self, data_path, data, max_length, limit, transform, mode):
        super(ImageCaptionDataset, self).__init__()
        self.data_path = data_path
        self.data = [(data_path + image_path, caption) for _, image_path, captions in data for caption in captions.split('@')]
        if mode == 'training':
            self.data = self.data[:limit]
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower=True)
        self.max_length = max_length + 1
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, caption = self.data[index]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0))

        encoded_caption = self.tokenizer.encode_plus(caption, max_length=self.max_length, pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=False, truncation=True)
        caption = np.array(encoded_caption['input_ids'])
        caption_mask = (1-np.array(encoded_caption['attention_mask'])).astype(bool)
        return image.tensors.squeeze(0), image.mask, caption, caption_mask