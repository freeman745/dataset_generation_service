from configparser import ConfigParser
import os

class ConfigWrapper:
    def __init__(self, config_path):
        self.config = ConfigParser()
        self.config.read(config_path)

        self.RPS_NAME_SPACE = os.getenv('RPS_NAME_SPACE')
        
        # Model Config
        self.binary_thresh = float(self.config.get('Model_config', 'binary_thresh'))
        self.depth_open_thresh = int(self.config.get('Model_config', 'depth_open_thresh'))
        self.rgb_open_thresh = int(self.config.get('Model_config', 'rgb_open_thresh'))

        # ENV
        self.ENV = dict(self.config.items('ENV'))[self.RPS_NAME_SPACE]

        # SAVE
        self.save = dict(self.config.items('SAVE'))[self.ENV]
        
        # Saving Config
        self.parent_folder = self.config.get('Saving_config', 'parent_folder')
        self.image_path = self.config.get('Saving_config', 'image_path')
        self.anns_path = self.config.get('Saving_config', 'anns_path')
        self.labels = self.config.get('Saving_config', 'labels')

        os.makedirs(self.image_path, exist_ok=True)
        os.makedirs(self.anns_path, exist_ok=True)
        
        self.padding = float(self.config.get('Saving_config', 'padding'))
        self.bg = self.config.get('Saving_config', 'bg')
        self.db_size = int(self.config.get('Saving_config', 'db_size'))
        
        try:
            self.jan_max = int(float(os.getenv('DGS_JAN_MAX_IMAGE')))
        except:
            self.jan_max = int(float(self.config.get('Saving_config', 'jan_max')))

        # OL_KPI Config
        try:
            self.iou_thresh = float(os.getenv('DGS_IOU'))
        except:
            self.iou_thresh = float(self.config.get('OL_KPI', 'iou_thresh'))
        
        # Server Config
        self.ip = self.config.get('Server', 'ip')
        self.port = int(self.config.get('Server', 'port'))

        # Dataset
        try:
            self.batch_size = int(os.getenv('DGS_DATASET_GENERATION_INTERVAL'))
        except:
            self.batch_size = int(self.config.get('Dataset', 'batch_size'))
        self.iteration = 1
        self.source = self.config.get('Dataset', 'source')
        self.mongodb = self.config.get('Dataset', 'mongodb')

        # Slack
        self.bot_token = self.config.get('Slack', 'bot_token')
        self.channel = self.config.get('Slack', 'channel')
        self.mongodb_url = self.config.get('Slack', 'mongodb_url').replace('ENV', self.ENV)

        # Third party
        self.template_match = int(self.config.get('Third_party', 'template_match'))
        self.tm = self.config.get('Third_party', 'tm')
        self.ol = self.config.get('Third_party', 'ol')

    def print_config(self):
        """Prints all instance variables of the ConfigWrapper object."""
        print("Current ConfigWrapper Instance Variables:")
        for key, value in vars(self).items():
            print(f"{key} = {value}")
