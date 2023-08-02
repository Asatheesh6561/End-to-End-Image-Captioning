from pathlib import Path

class DataProcessingConfig():
    def __init__(self, root_dir, dataset_name, local_data_file, unzip_dir):
        super(DataProcessingConfig, self).__init__()
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.local_data_file = local_data_file
        self.unzip_dir = unzip_dir

class DatasetConfig():
    def __init__(self, data_path, limit, max_length):
        self.data_path = data_path
        self.limit = limit
        self.max_length = max_length

class TrainConfig():
    def __init__(self, model_save_dir, lr_backbone, 
                    lr, 
                    num_epochs, 
                    lr_drop, 
                    start_epoch, 
                    weight_decay, 
                    backbone, 
                    position_embedding, 
                    dilation, 
                    batch_size, 
                    num_workers, 
                    checkpoint, 
                    clip_max_norm, 
                    hidden_dim, 
                    pad_token_id, 
                    max_length, 
                    dropout, 
                    vocab_size, 
                    enc_layers, 
                    dec_layers, 
                    dim_feed_forward, 
                    nheads, 
                    pre_norm):
        self.model_save_dir = model_save_dir
        self.lr_backbone = lr_backbone
        self.lr = lr
        self.num_epochs =  num_epochs
        self.lr_drop = lr_drop
        self.start_epoch = start_epoch
        self.weight_decay = weight_decay
        self.backbone = backbone 
        self.position_embedding = position_embedding 
        self.dilation = dilation 
        self.batch_size = batch_size 
        self.num_workers = num_workers 
        self.checkpoint = checkpoint 
        self.clip_max_norm = clip_max_norm 
        self.hidden_dim = hidden_dim 
        self.pad_token_id = pad_token_id 
        self.max_length = max_length 
        self.dropout = dropout 
        self.vocab_size = vocab_size 
        self.enc_layers = enc_layers 
        self.dec_layers = dec_layers 
        self.dim_feed_forward = dim_feed_forward 
        self.nheads = nheads 
        self.pre_norm = pre_norm