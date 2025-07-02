from config import VICUNA_PATH, LLAMA_PATH

import logging
import os
import time
import sys

def get_model_path_and_template(model_name):
    full_model_dict={
        "gpt-4":{
            "path":"gpt-4",
            "template":"gpt-4"
        },
        "gpt-3.5-turbo": {
            "path":"gpt-3.5-turbo",
            "template":"gpt-3.5-turbo"
        },
        "vicuna":{
            "path":VICUNA_PATH,
            "template":"vicuna_v1.1"
        },
        "llama-2":{
            "path":LLAMA_PATH,
            "template":"llama-2"
        },
        "claude-instant-1":{
            "path":"claude-instant-1",
            "template":"claude-instant-1"
        },
        "claude-2":{
            "path":"claude-2",
            "template":"claude-2"
        },
        "palm-2":{
            "path":"palm-2",
            "template":"palm-2"
        }
    }
    path, template = full_model_dict[model_name]["path"], full_model_dict[model_name]["template"]
    return path, template



class PrintLogger:
    def __init__(self, logger):
        self.logger = logger

    def write(self, message):
        # if message.strip() != "":
        #     self.logger.info(message.strip())
        if message.strip() != "" or message == "\n":
            self.logger.info(message)

    def flush(self):
        pass  # for compatibility with sys

def setup_logger(log_dir: str, attck_model: str, target_model: str, judge_model: str,) -> str:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H:%M:%S")
    log_path = os.path.join(log_dir, f"attack_{attck_model}_target_{target_model}_judge_{judge_model}_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        # format="%(asctime)s [%(levelname)s] %(message)s",
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_path, mode='a'),
            logging.StreamHandler(sys.__stdout__)  # 显式写入原始 stdout
        ]
    )

    # 让 print(...) 也能被记录
    sys.stdout = PrintLogger(logging.getLogger())
    sys.stderr = PrintLogger(logging.getLogger())

    logging.info(f"Logger initialized. Log file: {log_path}")
    return log_path