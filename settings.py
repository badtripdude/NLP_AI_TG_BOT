import pathlib
import sys

import yaml
from loguru import logger

APP_DIR = pathlib.Path(__file__).parent
DATA_DIR = pathlib.Path(__file__).parent / 'data'
CONFIG_DIR = DATA_DIR / 'configs'
LOGS_DIR = DATA_DIR / 'logs'
AI_MODELS = DATA_DIR / 'ai_models'

DB_TITLE = DATA_DIR / 'db.sqlite'

app_config = yaml.safe_load(
    (CONFIG_DIR / 'app.yaml').open('rb')
)

# log console
logger.add(sys.stderr,
           level=0)
# log files
logger.add(LOGS_DIR / 'trace.log', level=0)
logger.add(LOGS_DIR / 'info.log', level=20)
logger.add(LOGS_DIR / 'error.log', level=30)
logger.add(LOGS_DIR / 'debug.log',
           filter=lambda record: record['name'] in ['DEBUG', 'TRACE'])
