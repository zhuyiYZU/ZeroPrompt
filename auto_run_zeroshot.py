# -*- coding: utf-8 -*-
import logging
import subprocess
import time
from itertools import product
import pandas as pd
from sklearn.metrics import accuracy_score

# from datasets.TextClassification.process import datasets


def process_csv(input_file, output_file, label_columns, min_occurrence, dataset):
    """
    Filter rows from a CSV file where the label appears more than the specified number of times,
    and save them to a new CSV file.

    Parameters:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output CSV file.
        label_columns (int): Number of label columns (to select from the first few columns).
        min_occurrence (int): Minimum number of times a label must appear to be selected (greater than this value).
    """
    # Read the input CSV file
    df = pd.read_csv(input_file)

    # List to store the selected rows
    collected_data = []

    # Iterate through each row
    for _, row in df.iterrows():
        labels = pd.Series(row[:label_columns])  # Extract the first 'label_columns' columns as labels
        text = row.iloc[label_columns]  # The text column follows the label columns

        # Count the occurrences of each label
        label_counts = labels.value_counts()

        # Find labels that appear more than 'min_occurrence' times
        true_labels = label_counts[label_counts > min_occurrence].index
        if not true_labels.empty:
            # If multiple labels meet the condition, pick the first one
            true_label = true_labels[0]
            collected_data.append({'Label': true_label, 'Text': text})

    # Convert the collected data to a DataFrame
    new_data = pd.DataFrame(collected_data)
    if dataset in ["agnewstitle", "snippets", "newstitle"]:
        new_data.iloc[:, 0] += 1  # Adjust the label for these specific datasets
    # Save the data to the output CSV file
    new_data.to_csv(output_file, index=False, header=None, quoting=1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)  # Configure the logging system

    topic_data = ['snippets', 'agnewstitle']  # Datasets to process
    # emo_data = ['twitter']
    batch_sizes = {32}
    # learning_rates = {'2e-5','3e-5','4e-5','5e-5'}
    learning_rates = {'3e-5'}
    shots = {20}
    # seeds = range(90,93)
    seeds = {123}  # For agnewstitle 150, snippets 152, newstitle 131
    template_nums = {15}
    template_names = {'yi_template', 'gpt35_15_template', 'gpt4_15_0_template', 'gpt2_template', 'gpt4_15_1_template'}  #'gpt2_template','yi_template''gpt35_15_template','gpt4_15_0_template','gpt4_15_1_template'
    # topic_template_names = {'topic_gpt4_15_0_template'}
    # emo_template_names = {'emo_gpt4_15_0_template'}
    template_id = {0}
    verbalizer = {'kpt'}  # 'manual',
    for n, c, b in product(topic_data, template_nums, template_names):
        if c == 5:
            template_id = {0, 1, 2, 3, 4}
        elif c == 15:
            template_id = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
        elif c == 10:
            template_id = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
        # if n in ['twitter', 'MR', 'SST-2']:
        #     b = 'emo_gpt4_15_0_template'
        # Predict pseudo-labels
        for t, j, i, k, m, v in product(template_id, seeds, batch_sizes, learning_rates, shots, verbalizer):
            cmd = (
                f"python zeroshot.py --result_file ./result/compare-gen-result.txt "
                f"--dataset {n} --template_id {t} --seed {j} "
                f"--batch_size {i} --shot {m} --learning_rate {k} --verbalizer {v} --template_num {c}  --template_name {b}"
            )

            logging.info(f"Executing command: {cmd}")
            # print(cmd)
            try:
                subprocess.run(cmd, shell=True, check=True)
                logging.info(f"Command executed successfully: {cmd}")
            except subprocess.CalledProcessError as e:
                logging.error(f"Command failed: {cmd}. Error: {e.stderr.decode().strip()}")

            time.sleep(2)
        # Vote to select the true labels
        input_file = f"./datasets/generation_data/{b}_{n}_m_label.csv"

        # Set different voting thresholds (5, 10, 15), and save as three different training files
        vote_value = [4, 9, 14]
        for v in vote_value:
            output_file = f"./datasets/TextClassification/{n}/{b}_{v}_train.csv"
            process_csv(input_file, output_file, c, v, n)

    # Train the model and predict on the real dataset

    # real_datasets = ['newstitle', 'twitter', 'MR', 'SST-2', 'snippets',
    #                  'agnewstitle']  # 'newstitle', 'twitter', 'MR', 'SST-2', 'snippets','agnewstitle'
    # batch_sizes = {32}
    # learning_rates = {'4e-5'}
    # # learning_rates = {'3e-5'}
    # shots = {20}
    # seeds = range(150, 160)
    # # seeds = {123}  # agnewstitle 150, snippets  152   newstitle 131
    # template_id = {0}
    # verbalizer = {'kpt'}  # 'manual',
    # for n in real_datasets:
    #     for t, j, i, k, m, v in product(template_id, seeds, batch_sizes, learning_rates, shots, verbalizer):
    #
    #         cmd = (
    #             f"python fewshot.py --result_file ./result/GPT4_generation.txt "
    #             f"--dataset {n} --template_id {t} --seed {j} "
    #             f"--batch_size {i} --shot {m} --learning_rate {k} --verbalizer {v}"
    #         )
    #
    #         logging.info(f"Executing command: {cmd}")
    #         print(cmd)
    #         try:
    #
    #             subprocess.run(cmd, shell=True, check=True)
    #             logging.info(f"Command executed successfully: {cmd}")
    #         except subprocess.CalledProcessError as e:
    #             logging.error(f"Command failed: {cmd}. Error: {e.stderr.decode().strip()}")
    #
    #         time.sleep(2)
