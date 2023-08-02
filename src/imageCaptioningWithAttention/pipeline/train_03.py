from imageCaptioningWithAttention.config.configuration import ConfigurationManager
from imageCaptioningWithAttention.components.trainer import Trainer
from imageCaptioningWithAttention.components.models import *
from imageCaptioningWithAttention.constants import *
from PIL import Image
import os
from transformers import BertTokenizer

class TrainPipeline():
    def __init__(self, train_dataset, val_dataset):
        self.config = ConfigurationManager(CONFIG_FILE_PATH, PARAMS_FILE_PATH)
        self.train_config = self.config.get_train_config()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model, self.criterion = build_model(self.train_config.vocab_size, self.train_config.hidden_dim, self.train_config.pad_token_id, self.train_config.max_length, self.train_config.nheads, self.train_config.enc_layers, self.train_config.dec_layers, self.train_config.dim_feed_forward, self.train_config.dropout, self.train_config.lr_backbone, self.train_config.backbone, self.train_config.dilation, self.train_config.pre_norm)
        self.trainer = Trainer(train_dataset, val_dataset, self.train_config.lr, self.train_config.lr_backbone, self.train_config.weight_decay, self.train_config.lr_drop, self.train_config.batch_size, self.train_config.num_workers, self.train_config.num_epochs, self.train_config.checkpoint, self.train_config.clip_max_norm)
    
    def train_model(self):
        self.trainer.train_model(self.model, self.criterion, self.train_config.model_save_dir)

    def predict(self, image_path, model_name):
        self.model.eval()
        self.model = self.load_model(model_name)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        image = Image.open(image_path)
        image = self.val_dataset.transform(image)
        image = image.unsqueeze(0)
        start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
        caption, caption_mask = self.trainer.get_caption_and_mask(start_token, self.train_config.max_length)
        output = self.trainer.evaluate(self.model, image, caption, caption_mask, self.train_config.max_length)
        result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
        return result

    def load_model(self, model_name):
        checkpoint = torch.load(os.path.join(self.train_config.model_save_dir, model_name))
        self.model.load_state_dict(checkpoint['model'])
        return self.model