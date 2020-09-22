import logging
import os

class Logger:
    LOG_FILE_NAME = 'output.log'
    METRICS_FILE_NAME = 'metrics.log'

    def __init__(self,log_dir):
        self.level = 'info'
        self.log_dir = log_dir

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.message_logger = self._init_message_logger()

        self.metrics_writer = open(os.path.join(
            self.log_dir, self.METRICS_FILE_NAME), 'at')

    def _init_message_logger(self):
        message_logger = logging.getLogger('messages')
        message_logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '[%(levelname)s] [%(asctime)s] %(message)s')
        std_handler = logging.StreamHandler()
        std_handler.setLevel(message_logger.level)
        std_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(
            os.path.join(self.log_dir, self.LOG_FILE_NAME))
        file_handler.setLevel(message_logger.level)
        file_handler.setFormatter(formatter)

        message_logger.addHandler(std_handler)
        message_logger.addHandler(file_handler)
        return message_logger

    def metrics(self, metrics_message):
        self.message_logger.info(metrics_message)
        self.metrics_writer.write(metrics_message+'\n')
        self.metrics_writer.flush()

    def info(self, message):
        self.message_logger.info(message)