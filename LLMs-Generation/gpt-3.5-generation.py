# This Python file uses the following encoding: utf-8
import csv
import openai
import codecs
import pandas as pd
import time

#输入 api_key
chat_gpt_key1 = ''


def read_csv_row_range(csv_file, start_row, end_row):
    with open(csv_file, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for index, row in enumerate(csv_reader):
            if start_row <= index + 1 <= end_row:
                yield row

str = "Task: As a paraphrasing tool, you will generate five sentences with the same meaning as the following sentence, but with different structures and vocabulary, based on the text I am about to provide:\n" \
        "Input: This is a news about {'mask'}."


messages = []
content = str

print("user:" + content)

openai.api_key = chat_gpt_key1

messages.append({"role": "user", "content": content})

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    # model="text-davinci-003",
    messages=messages
)
time.sleep(20)
chat_response = completion
answer = chat_response.choices[0].message.content
print(f'ChatGPT: {answer}')






