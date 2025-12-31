import csv
import openai
import codecs
import pandas as pd
import time
list_label = list()
i = 0  # i记录list_label中的位置
label1 = 0
label2 = 0
label3 = 0
label4 = 0
count_num = 0



# def read_csv_row_range(csv_file, start_row, end_row):
#     with open(csv_file, 'r', encoding='utf-8') as file:
#         csv_reader = csv.reader(file)
#         for index, row in enumerate(csv_reader):
#             if start_row <= index + 1 <= end_row:
#                 yield row


input_file = 'test_gpt35/SST-2_test.csv'
out_file = 'result/output_SST_gpt35.csv'
start_row = 0  # 起始行号
end_row = 2  # 结束行号
row_step = 3  # 行步长
#


openai.api_key = key
openai.base_url = ''
# messages = []
with open (input_file,mode='r',newline='',encoding='utf-8') as infile:

    reader = csv.reader(infile)

    for row in reader:
        messages = []
        # classification_prompt = f"Task: You will work as a short text classification tool to classify texts into " \
        #                         f"corresponding categories based on the texts and categories I provide. \n "
        # classification_prompt += f"Question: Based on the following input, determine which category it belongs to " \
        #                          f"and classify it as:\n"
        # classification_prompt += f"[0] negative\n "
        # classification_prompt += f"[1] positive\n"
        #
        # classification_prompt += f"input content: {row[0]}\n"
        #
        # classification_prompt += "Requirements: In the output, you do not need to output the content of the " \
        #                          "parse, do not need to explain, only need to output the category number, for example, judged as " \
        #                          "negative, output: [0], judged as positive, output: [1] \n"
        # classification_prompt += "Your output is:"

        # snippets
        # classification_prompt = f"Task: You will work as a short text classification tool to classify texts into " \
        #                         f"corresponding categories based on the texts and categories I provide. \n "
        # classification_prompt += f"Question: Based on the following input, determine which category it belongs to " \
        #                          f"and classify it as:\n"
        # classification_prompt += f"[1] business\n"
        # classification_prompt += f"[2] computers\n"
        # classification_prompt += f"[3] culture-arts-entertainment\n"
        # classification_prompt += f"[4] education-science\n"
        # classification_prompt += f"[5] engineering\n"
        # classification_prompt += f"[6] health\n"
        # classification_prompt += f"[7] politics-society\n"
        # classification_prompt += f"[8] sports\n"
        # classification_prompt += f"input content: {row[1]}\n"
        #
        # classification_prompt += "Requirements: n the output, you do not need to output the content of the " \
        #                          "parse, do not need to explain, only need to output the category number, for example, judged as " \
        #                          "business, output: [1], judged as computers, output: [2], judged as culture-arts-entertainment, " \
        #                          "output: [3], judged as education-science, output: [4],judged as engineering, output: [5],judged as health, "\
        #                           "output: [6],judged as politics-society, output: [7],judged as sports, output: [8], "
        # classification_prompt += "Your output is:："

        # classification_prompt = f"Task: You will work as a short text classification tool to classify texts into " \
        #                         f"corresponding categories based on the texts and categories I provide. \n "
        # classification_prompt += f"Question: Based on the following input, determine which category it belongs to " \
        #                          f"and classify it as:\n"
        # classification_prompt += f"[1] business\n"
        # classification_prompt += f"[2] entertainment\n"
        # classification_prompt += f"[3] health\n"
        # classification_prompt += f"[4] technology\n"
        # classification_prompt += f"[5] sport\n"
        # classification_prompt += f"[6] us\n"
        # classification_prompt += f"[7] world\n"
        # classification_prompt += f"input content: {row[1]}\n"
        #
        # classification_prompt += "Requirements: n the output, you do not need to output the content of the " \
        #                          "parse, do not need to explain, only need to output the category number, for example, judged as " \
        #                          "business, output: [1], judged as entertainment, output: [2], judged as health, " \
        #                          "output: [3], judged as technology, output: [4],judged as sport, output: [5],judged as us, "\
        #                           "output: [6],judged as world, output: [7], "
        # classification_prompt += "Your output is:："
        # classification_prompt = f"Task: You will work as a short text classification tool to classify texts into " \
        #                         f"corresponding categories based on the texts and categories I provide. \n "
        classification_prompt = f"Question: Based on the following input, determine which category it belongs to " \
                                 f"and classify it as:\n"
        classification_prompt += f"[0] negative\n "
        classification_prompt += f"[1] positive\n"

        classification_prompt += f"input content: {row[1]}\n"

        classification_prompt += "Requirements: In the output, you do not need to output the content of the " \
                                 "parse, do not need to explain, only need to output the category number, for example, judged as " \
                                 "negative, output: [0], judged as positive, output: [1] \n"
        classification_prompt += "Your output is:"
        content= classification_prompt

        print("user:" + content)
        messages.append({"role": "user", "content": content})

        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        # time.sleep(20)
        chat_response = completion
        answer = chat_response.choices[0].message.content
        print(f'LLM: {answer}')
        messages.append({"role": "assistant", "content": answer})
        text_answer = answer
        if '1' in answer:
            answer = '1'
        # elif '2' in answer:
        #     answer = '2'
        # elif '3' in answer:
        #     answer = '3'
        # elif '4' in answer:
        #     answer = '4'
        # elif '5' in answer:
        #     answer = '5'
        elif '0' in answer:
            answer = '0'

        else:
            answer = '9'


        new_row = row + [answer] +[text_answer]
        with open(out_file, mode='a', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(new_row)






    #     if ('1' in answer or 'politics' in answer) and '1' in row[0]:
    #         label1 += 1
    #         count_num += 1
    #         acc = (label1+label2+label3+label4)/count_num
    #         # print('消极分类正确：', label0, '积极分类正确：', label1, 'Acc:', acc)
    #         print('politics分类正确：', label1, 'sports分类正确：', label2,'business分类正确：',label3,'technology分类正确',label4, 'Acc:', acc)
    #         with open('result\output_agnews_gpt35.txt', 'a', encoding='utf-8') as file:
    #             writer = csv.writer(file)
    #             writer.writerow([label1, label2, label3, label4, count_num, acc])
    #     elif ('2' in answer or 'sports' in answer) and '2' in row[0]:
    #         label2 += 1
    #         count_num += 1
    #         acc = (label1+label2+label3+label4)/count_num
    #         # print('消极分类正确：', label0, '积极分类正确：', label1, 'Acc:', acc)
    #         print('politics分类正确：', label1, 'sports分类正确：', label2,'business分类正确：',label3,'technology分类正确',label4, 'Acc:', acc)
    #         with open('result\output_agnews_gpt35.txt', 'a', encoding='utf-8') as file:
    #             writer = csv.writer(file)
    #             writer.writerow([label0, label1, count_num, acc])
    #     else:
    #         count_num += 1
    #         print("分类错误")
    #         with open('output_n2_l2.txt', 'a', encoding='utf-8') as file:
    #             writer = csv.writer(file)
    #             writer.writerow([label0, label1, count_num])
    # # 增加起始行号和结束行号
    # start_row += row_step
    # end_row += row_step
    #
    # # 判断是否达到文件末尾
    # with open(csv_file, 'r', encoding='utf-8') as file:
    #     csv_reader = csv.reader(file)
    #     line_count = sum(1 for _ in csv_reader)
    #     if end_row >= line_count:
    #         break









# def query_gpt4(question):
#     openai.api_key = "sk-ndVKGvLGotuLCUcRE97bBc4699A442B799395dDb3b9f44Bc"
#     #openai.base_url = url
#     openai.base_url = 'https://4.0.wokaai.com/v1/'
#
#
#     try:
#         response = openai.chat.completions.create(
#             model="claude-3-opus-20240229",  # 确认使用 GPT-4 模型
#             messages=[
#                 {"role": "user", "content": question}
#             ]
#         )
#         chat_response = response
#         answer = chat_response.choices[0].message.content
#         print(f'LLM: {answer}')
#         # print(response)
#         # return response['choices'][0].message['content']
#
#     except Exception as e:
#         return str(e)
#
# # 问题
# question = "请介绍一下你自己"
#
# # 获取并打印回答
# answer = query_gpt4(question)

