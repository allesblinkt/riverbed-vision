import logging
import logging.config

import os

MAIN_LOG_LEVEL = 'DEBUG'
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')

LOGGING = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'verbose': {
            'format': '%(levelname)s %(asctime)s %(name)s -- %(message)s'
        },
        'simple': {
            'format': '%(levelname)s %(name)s -- %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose'
        },

        'file_handler': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(LOG_DIR, 'brain.log'),
            'maxBytes': 5 * 1024 * 1024,  # 5 MB
            'backupCount': 5,
            'formatter': 'verbose',
        }
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file_handler'],
            'propagate': True,
            'level': MAIN_LOG_LEVEL,
        }

    }
}

# logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.DEBUG)
logging.config.dictConfig(LOGGING)

log = logging.getLogger(__name__)


def makelog(name):
    return logging.getLogger(name)
