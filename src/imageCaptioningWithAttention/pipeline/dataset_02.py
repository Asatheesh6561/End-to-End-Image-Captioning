from imageCaptioningWithAttention.config.configuration import ConfigurationManager
from imageCaptioningWithAttention.components.dataset import ImageCaptionDataset
from imageCaptioningWithAttention.constants import *
from torchvision import transforms
import numpy as np
class DatasetPipeline():
    def __init__(self, all_data):
        self.config = ConfigurationManager(CONFIG_FILE_PATH, PARAMS_FILE_PATH)
        self.dataset_config = self.config.get_dataset_config()
        self.all_data = all_data
        self.train_transform = transforms.Compose([transforms.Lambda(self.under_max), transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[
                              0.8, 1.5], saturation=[0.2, 1.5]), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.val_transform = transforms.Compose([transforms.Lambda(self.under_max), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def under_max(self, image):
        if image.mode != 'RGB':
            image = image.convert("RGB")

        shape = np.array(image.size, dtype=np.float32)
        long_dim = max(shape)
        scale = 299 / long_dim

        new_shape = (shape * scale).astype(int)
        image = image.resize(new_shape)

        return image

    def get_dataset(self, mode):
        if mode == 'training':
            return ImageCaptionDataset(self.dataset_config.data_path, self.all_data, self.dataset_config.max_length, self.dataset_config.limit, self.train_transform, mode)
        if mode == 'validation':
            return ImageCaptionDataset(self.dataset_config.data_path, self.all_data, self.dataset_config.max_length, self.dataset_config.limit, self.val_transform, mode)
        