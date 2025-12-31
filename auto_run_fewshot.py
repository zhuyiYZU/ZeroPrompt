# -*- coding: utf-8 -*-
import logging
import subprocess
import time
from itertools import product
import pandas as pd



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)  # 配置日志记录器

    real_datasets = ['twitter', 'MR', 'SST-2']  # 'newstitle', 'twitter', 'MR', 'SST-2', 'snippets','agnewstitle'
    batch_sizes = {32}
    learning_rates = {'2e-5','3e-5','4e-5'}
    # learning_rates = {'5e-5'}
    shots = {20}
    # seeds = range(124,144)
    seeds = {123}  # agnewstitle 150, snippets  152   newstitle 131
    template_id = {0}
    verbalizer = {'kpt'}  # 'manual',
    for n in real_datasets:
        for t, j, i, k, m, v in product(template_id, seeds, batch_sizes, learning_rates, shots, verbalizer):

            cmd = (
                f"python fewshot.py --result_file ./result/GPT4_generation.txt "
                f"--dataset {n} --template_id {t} --seed {j} "
                f"--batch_size {i} --shot {m} --learning_rate {k} --verbalizer {v}"
            )

            logging.info(f"Executing command: {cmd}")
            print(cmd)
            try:

                subprocess.run(cmd, shell=True, check=True)
                logging.info(f"Command executed successfully: {cmd}")
            except subprocess.CalledProcessError as e:
                logging.error(f"Command failed: {cmd}. Error: {e.stderr.decode().strip()}")

            time.sleep(2)

