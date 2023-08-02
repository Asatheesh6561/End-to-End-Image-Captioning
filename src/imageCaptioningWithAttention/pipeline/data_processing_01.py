from imageCaptioningWithAttention.config.configuration import ConfigurationManager
from imageCaptioningWithAttention.components.data_processing import DataProcessing, MySQLServer
from imageCaptioningWithAttention.constants import *
from imageCaptioningWithAttention import logger
import mysql.connector
from mysql.connector import Error

class DataProcessingPipeline():
    def __init__(self):
        self.config = ConfigurationManager(CONFIG_FILE_PATH, PARAMS_FILE_PATH)
        self.data_processing_config = self.config.get_data_processing_config()
        self.data_processing = DataProcessing(self.data_processing_config)
        self.server = MySQLServer('localhost', 'root', 'Yesminister22!', 'image_captioning')
        self.db_connection = self.server.db_connection()
        
    def get_data(self):
        try:
            self.data_processing.get_dataset_from_kaggle()
        except Exception as e:
            raise e

    def load_to_database(self, table_name):
        number_of_items = self.server.read_query(self.db_connection, f'SELECT COUNT(*) FROM {table_name}')
        if number_of_items[0][0] == 0:
            create_image_caption_table = f"""
                CREATE TABLE {table_name} (
                    image_id INT PRIMARY KEY,
                    image_path VARCHAR(255),
                    captions VARCHAR(1000)
                );
            """
            self.server.execute_query(self.db_connection, create_image_caption_table)
            load_query = self.server.get_load_query(self.data_processing_config, table_name)
            self.server.execute_query(self.db_connection, load_query)
        else:
            logger.info(f'Table {table_name} already created with {number_of_items} items.')
