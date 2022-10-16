import pathlib

import yaml
from loguru import logger

APP_DIR = pathlib.Path(__file__).parent
DATA_DIR = pathlib.Path(__file__).parent / 'data'
CONFIG_DIR = DATA_DIR / 'configs'
LOGS_DIR = DATA_DIR / 'logs'
AI_MODELS_DIR = DATA_DIR / 'ai_models' / 'train_models'

app_config = yaml.safe_load(
    (CONFIG_DIR / 'app.yaml').open('rb')
)
DB_TITLE = DATA_DIR / 'dbs' / app_config['DB']['FILENAME']
MODEL_NAME = AI_MODELS_DIR / app_config['AI']['MODEL_NAME']


# log files
logger.add(LOGS_DIR / 'trace.log', level=0)
logger.add(LOGS_DIR / 'info.log', level=20)
logger.add(LOGS_DIR / 'error.log', level=30)
logger.add(LOGS_DIR / 'debug.log',
           filter=lambda record: record['name'] in ['DEBUG', 'TRACE'])
