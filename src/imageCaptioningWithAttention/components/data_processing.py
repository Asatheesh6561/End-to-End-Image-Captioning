import kaggle
from imageCaptioningWithAttention import logger
from imageCaptioningWithAttention.utils.common import get_size
import mysql.connector
from mysql.connector import Error
from pathlib import Path
import os

class DataProcessing():
    def __init__(self, data_processing_config):
        super(DataProcessing, self).__init__()
        self.data_processing_config = data_processing_config

    def get_dataset_from_kaggle(self):
        if not os.path.exists(self.data_processing_config.local_data_file):
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(self.data_processing_config.dataset_name, path=self.data_processing_config.unzip_dir, unzip=True)
            logger.info(f'{self.data_processing_config.local_data_file} downloaded.')
        else:
            logger.info(f'File already exists with size: {get_size(Path(self.data_processing_config.local_data_file))}')

class MySQLServer():
    def __init__(self, host_name, user_name, password, db_name):
        super(MySQLServer, self).__init__()
        self.host_name = host_name
        self.user_name = user_name
        self.password = password
        self.db_name = db_name

    def server_connection(self):
        connection = None
        try:
            connection = mysql.connector.connect(
            host=self.host_name,
            user=self.user_name,
            password=self.password,
        )
            logger.info(f'Server connection successful with host name {self.host_name} and username {self.user_name}.')
        except Error as err:
            logger.error(f'Cannot connect with error {err}')
        return connection
    
    def create_database(self, connection, query):
        cursor = connection.cursor()
        try:
            cursor.execute(query)
            logger.info(f'Successfully created database.')
        except Error as err:
            logger.error(f'Cannot create database: {err}')

    def db_connection(self):
        connection = None
        try:
            connection = mysql.connector.connect(
            host=self.host_name,
            user=self.user_name,
            password=self.password,
            database=self.db_name
        )
            print(connection)
            logger.info(f'Database connection successful with host name {self.host_name} and username {self.user_name}.')
        except Error as err:
            logger.error(f'Cannot connect with error {err}')
        return connection
    
    def execute_query(self, connection, query):
        cursor = connection.cursor()
        try:
            cursor.execute(query)
            connection.commit()
            logger.info(f'Query {query} successful.')
        except Error as err:
            logger.error(f'Cannot execute query with error: {err}')

    def get_load_query(self, data_processing_config, table_name):
        pop_image_caption = 'INSERT INTO %s VALUES\n' % (table_name,)
        with open(data_processing_config.local_data_file) as f:
            lines = [line.strip() for line in f]
        for i in range(1, len(lines), 5):
            image_filename = lines[i].split(',', 1)[0]
            image_path = f'\'/Images/{image_filename}\''
            captions = '\''+"@".join([lines[i+j].split(',', 1)[1].replace('\'', '\\\'') for j in range(5)]) + '\''
            if i == 1: print(captions)
            image_str = '(%s, %s, %s)' % (str((i//5+1)), image_path, captions)
            pop_image_caption += image_str + ',\n'
        pop_image_caption = pop_image_caption[:-2] + ';'
        return pop_image_caption
    
    def read_query(self, connection, query):
        cursor = connection.cursor()
        result = None
        try:
            cursor.execute(query)
            result = cursor.fetchall()
            return result
        except Error as err:
            logger.error(f'Could not read query due to {err}')