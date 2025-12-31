import torch
from modelscope import snapshot_download
from modelscope import AutoModelForCausalLM
from transformers import AutoTokenizer
import csv
import transformers



# label0 = 0
# label1 = 0
# count_num = 0


# def read_csv_row_range(csv_file, start_row, end_row):
#     with open(csv_file, 'r', encoding='gbk',errors='ignore') as file:
#         csv_reader = csv.reader(file)
#         for index, row in enumerate(csv_reader):
#             if start_row <= index + 1 <= end_row:
#                 yield row
#
#
# csv_file = 'CT_MLM.csv'
# start_row = 0  # 起始行号
# end_row = 2  # 结束行号
# row_step = 3  # 行步长


def get_examples(path):
    examples = []
    with open(path, encoding='utf8') as f:
        reader = csv.reader(f, delimiter=',')
        for idx, row in enumerate(reader):
            label, headline = row
            text_a = headline.replace('\\', ' ')

            examples.append(text_a)
    return examples

model_path = '/home/star/Szq/LLMs/wokaai/models/Llama3-8B'
def load_model(model_path):

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        model_kwargs={"torch_dtype": torch.float16},
        device="cuda",
    )
    return pipeline

pipline= load_model(model_path)
output=pipline('<s>what would you like to do today?\n<s><s>Assistant:')
print(output)




def predict(path):
    messages = []
    test_result = []

    # '/home/mjy/wangye/zero_shot/datasets/TextClassification/snippets/train.csv'
    examples = get_examples(path)

    for example in examples:
        classification_prompt = f"Task: You will work as a short text classification tool to classify texts into " \
                                f"corresponding categories based on the texts and categories I provide. \n "
        classification_prompt += f"Question: Based on the following input, determine which category it belongs to " \
                                 f"and classify it as:"
        classification_prompt += f"[1] Politics: News and information related to government, political events, " \
                                         f"policies, and political figures. "
        classification_prompt += f"[2] Sports: Updates, scores, and news about sporting events, athletes, teams, " \
                                 f"and competitions. "
        classification_prompt += f"[3] Business: Information and updates on commerce, finance, markets, " \
                                 f"companies, and economic activities. "
        classification_prompt += f"[4] Technology: News and content about technological innovations, " \
                                 f"advancements, gadgets, software, and tech companies. "

        classification_prompt += f"If you think the text contains multiple information, select categories based " \
                                 f"on the main features. Pay attention to the overall characteristics of the " \
                                 f"text, not just the individual words or phrases.\n "
        classification_prompt += f"input content: {example}\n"

        classification_prompt += "Requirements: In the output, you do not need to output the content of the " \
                                 "parse, only need to output the category number, for example, judged as " \
                                 "Politics, output: [1], judged as Sports, output: [2], judged as Business, " \
                                 "output: [3], judged as Technology, output: [4], "
        classification_prompt += "Your output is:："
        text_with_prompt = classification_prompt
        messages.append({"role": "user", "content": text_with_prompt})
        # prompt = pipeline.tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=True
        # )
        #
        # terminators = [
        #     pipeline.tokenizer.eos_token_id,
        #     pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        # ]
        outputs = pipeline(
            messages,
            max_new_tokens=512,
            # eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )

        response_LLM = outputs
        print(response_LLM)
    return test_result



# predict('/home/star/文档/wy/zero-shot/LLAMA/test_data/agnews_title_test.csv')

# while True:
#     messages = []
#     for row in read_csv_row_range(csv_file, start_row, end_row):
#         content = str + row[1]
#         print("user:" + content)
#         messages.append({"role": "user", "content": content})
#         prompt = pipeline.tokenizer.apply_chat_template(
#                 messages,
#                 tokenize=False,
#                 add_generation_prompt=True
#             )
#
#         terminators = [
#                 pipeline.tokenizer.eos_token_id,
#                 pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
#             ]
#         outputs = pipeline(
#             prompt,
#             max_new_tokens=512,
#             eos_token_id=terminators,
#             do_sample=True,
#             temperature=0.6,
#             top_p=0.9
#         )
#         response_LLM = outputs[0]["generated_text"][len(prompt):]
#         print('LLM:')
#         print(response_LLM)
#         if ('n8' in response_LLM or 'negative' in response_LLM) and '1' in row[0]:
#             label0 += 1
#             count_num += 1
#             acc = (label0 + label1) / count_num
#             print('类别1分类正确：', label0, '类别2分类正确：', label1, 'Acc:', acc)
#             with open('output_CT_MLM.txt', 'a', encoding='utf-8') as file:
#                 writer = csv.writer(file)
#                 writer.writerow([label0, label1, count_num, acc])
#         elif ('p8' in response_LLM or 'positive' in response_LLM) and '2' in row[0]:
#             label1 += 1
#             count_num += 1
#             acc = (label0 + label1) / count_num
#             print('类别1分类正确：', label0, '类别2分类正确：', label1, 'Acc:', acc)
#             with open('output_CT_MLM.txt', 'a', encoding='utf-8') as file:
#                 writer = csv.writer(file)
#                 writer.writerow([label0, label1, count_num, acc])
#         else:
#             count_num += 1
#             print("分类错误")
#             with open('output_CT_MLM.txt', 'a', encoding='utf-8') as file:
#                 writer = csv.writer(file)
#                 writer.writerow([label0, label1, count_num])
#
#     start_row += row_step
#     end_row += row_step
#
#         # 判断是否达到文件末尾
#     with open(csv_file, 'r', encoding='gbk', errors='ignore') as file:
#         csv_reader = csv.reader(file)
#         line_count = sum(1 for _ in csv_reader)
#         if end_row >= line_count:
#             break

# messages = [{"role": "system", "content": ""}]
#
# messages.append(
#                 {"role": "user", "content": str}
#             )
#
# prompt = pipeline.tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )
#
# terminators = [
#         pipeline.tokenizer.eos_token_id,
#         pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
#     ]
# outputs = pipeline(
#     prompt,
#     max_new_tokens=512,
#     eos_token_id=terminators,
#     do_sample=True,
#     temperature=0.6,
#     top_p=0.9
# )
#
# content = outputs[0]["generated_text"][len(prompt):]
#
# print(content)