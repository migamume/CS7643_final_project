import logging.config
import os

class LogProvider:
    # Help from https://medium.com/nerd-for-tech/python-logging-simplification-e24824a748f9
    DEFAULT_LEVEL = logging.INFO
    _logger = None

    @classmethod
    def _log_setup(cls, log_cfg_path):
        import yaml

        if os.path.exists(log_cfg_path):
            with open(log_cfg_path, 'rt') as cfg_file:
                try:
                    config = yaml.safe_load(cfg_file.read())
                    logging.config.dictConfig(config)
                except Exception as e:
                    print(e)
                    print('Error with file, using Default logging')
                    logging.basicConfig(level=cls.DEFAULT_LEVEL)
        else:
            logging.basicConfig(level=cls.DEFAULT_LEVEL)
            print('Config file not found, using Default logging')

    @classmethod
    def get_logger(cls) -> logging.Logger:
        if cls._logger is None:
            try:
                log_config_path = os.environ['LOG_CFG_PATH']
                cls._log_setup(log_config_path)
                cls._logger = logging.getLogger('dev')
            except:
                cls._logger = logging
                cls._logger.warning('Failed to setup logger using env var LOG_CFG_PATH')
        return cls._logger
