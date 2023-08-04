from src.imageCaptioning import logger
from src.imageCaptioning.pipeline.data_processing_01 import DataProcessingPipeline
from src.imageCaptioning.pipeline.dataset_02 import DatasetPipeline
from src.imageCaptioning.pipeline.train_03 import TrainPipeline
import os

def main():
    try:
        STAGE_NAME = 'Data Processing'
        logger.info(f'Stage {STAGE_NAME} started')
        data_processing_pipeline = DataProcessingPipeline()
        data_processing_pipeline.get_data()
        data_processing_pipeline.load_to_database('table_image_caption')
        logger.info(f'Stage {STAGE_NAME} completed.')

        STAGE_NAME = 'Dataset'

        logger.info(f'Stage {STAGE_NAME} started')
        all_data = data_processing_pipeline.server.read_query(data_processing_pipeline.db_connection, 'SELECT * FROM table_image_caption')
        dataset_pipeline = DatasetPipeline(all_data)
        train_dataset = dataset_pipeline.get_dataset('training')
        val_dataset = dataset_pipeline.get_dataset('validation')
        logger.info(f'Stage {STAGE_NAME} completed.')

        STAGE_NAME = 'Train'

        logger.info(f'Stage {STAGE_NAME} started')
        train_pipeline = TrainPipeline(train_dataset, val_dataset)
        output = train_pipeline.predict(os.path.join(data_processing_pipeline.data_processing_config.root_dir, 'Images/12830823_87d2654e31.jpg'), 'model_20.pth')
        print(output)
    except Exception as e:
        logger.exception(e)
        raise(e)
if __name__ == '__main__':
    main()