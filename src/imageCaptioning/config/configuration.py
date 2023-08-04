from imageCaptioning.constants import *
from imageCaptioning.utils.common import read_yaml, create_directories
from imageCaptioning.entity.config_entity import DataProcessingConfig, DatasetConfig, TrainConfig
class ConfigurationManager():
    def __init__(self, config_file_path, params_file_path):
        super(ConfigurationManager, self).__init__()
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)
        create_directories([self.config.artifacts_root], verbose=True)
    
    def get_data_processing_config(self):
        config = self.config.data_processing
        create_directories([config.root_dir], verbose=True)
        data_processing_config = DataProcessingConfig(config.root_dir, config.dataset_name, config.local_data_file, config.unzip_dir)
        return data_processing_config
    
    def get_dataset_config(self):
        config = self.config.dataset
        params = self.params
        dataset_config = DatasetConfig(config.data_path, config.limit, params.max_length)
        return dataset_config
    
    def get_train_config(self):
        config = self.config.train
        params = self.params
        create_directories([config.model_save], verbose=True)
        train_config = TrainConfig(config.model_save, params.lr_backbone, params.lr, params.num_epochs, params.lr_drop, params.start_epoch, params.weight_decay, params.backbone, params.position_embedding, params.dilation, params.batch_size, params.num_workers, params.checkpoint, params.clip_max_norm, params.hidden_dim, params.pad_token_id, params.max_length, params.dropout, params.vocab_size, params.enc_layers, params.dec_layers, params.dim_feed_forward, params.nheads, params.pre_norm)
        return train_config