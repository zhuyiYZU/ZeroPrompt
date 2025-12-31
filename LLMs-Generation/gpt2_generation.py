from transformers import pipeline, set_seed
import re
import csv
import pandas as pd
def get_examples( path):
    examples = []
    with open(path, encoding='utf8') as f:
        reader = csv.reader(f, delimiter=',')
        for idx, row in enumerate(reader):
            label, headline = row
            text_a = headline.replace('\\', ' ')

            examples.append(text_a)
    return examples

def gpt_generation(example, prompt):
    '''
    Args:
        example:
        prompt: prompt = "Please generate five sentences with the same semantics but different expressions based on the sentence I am about to provide:"

    Returns:

    '''
    generator = pipeline('text-generation', model='/home/star/文档/wy/zero-shot/model/gpt-2')
    set_seed(42)
    prompt += example
    sentences_dict = generator(prompt, max_length=30, num_return_sequences=5, truncation=True)
    print(f'{prompt}')
    #
    semantics_list = [re.search(
        f'Please generate five similar texts based on the text I provided, with consistent semantics：(.*)',
        d['generated_text']).group(1) for d in sentences_dict]
    return semantics_list
example = 'This is a news about [mask]'
prompt = 'Please generate ten similar texts based on the text I provided, with consistent semantics：'
result = gpt_generation(example, prompt)

print(result)

# datasets = ['twitter']#'agnewstitle','newstitle'
# for dataset in datasets:
#
#     examples = get_examples(f'/home/mjy/wangye/zero_shot/datasets/TextClassification/{dataset}/train.csv')
#     # examples = ['pentairpool welcome pentair pool products inc manufacturer manufacturer filters sanitizers cleaners heaters pumps california']
#     result_list = []
#     for example in examples:
#         generator = pipeline('text-generation', model='/home/mjy/wangye/zero_shot/model/gpt-2')
#         set_seed(42)
#         prompt = "Please generate five sentences with the same semantics but different expressions based on the sentence I am about to provide:"
#         prompt += example
#         sentences_dict = generator(prompt, max_length=30, num_return_sequences=5,truncation=True)
#         # 提取每个值中的字符串部分
#         semantics_list = [re.search(r'Please generate five sentences with the same semantics but different expressions based on the sentence I am about to provide:(.*)', d['generated_text']).group(1) for d in sentences_dict]
#         result_list.append(semantics_list)

        # print(result_list)
    # 打印结果

    #
    # data_gen = pd.DataFrame(result_list[1:],columns=result_list[0])
    # # df.to_csv(f'/home/mjy/wangye/zero_shot/datasets/TextClassification/{dataset}/gen_train.csv',index=False, header=False,quoting=1)
    #
    # #插入
    # data1 = pd.read_csv(f'/home/mjy/wangye/zero_shot/datasets/TextClassification/{dataset}/train.csv', header=None, usecols=[0])
    # # data_gen = pd.read_csv(f'/home/mjy/wangye/zero_shot/datasets/TextClassification/{dataset}/gen_train.csv', header=None)
    # data_gen.insert(loc=0,column='label', value=data1)
    # data_gen.to_csv(f'/home/mjy/wangye/zero_shot/datasets/TextClassification/{dataset}/re_train.csv', index=False, header=False,quoting=1)

    # # 读取原始CSV文件和新的CSV文件
    # with open(f'/home/mjy/wangye/zero_shot/datasets/TextClassification/{dataset}/re_train.csv', 'r', encoding='utf-8') as original_file, \
    #      open(f'/home/mjy/wangye/zero_shot/datasets/TextClassification/{dataset}/expend.csv', 'w', newline='', encoding='utf-8') as new_file:
    #
    #     # 创建CSV读写对象
    #     csv_reader = csv.reader(original_file)
    #     csv_writer = csv.writer(new_file)
    #
    #     # 逐行读取原始文件并写入新文件
    #     for row in csv_reader:
    #         label = row[0]  # 提取标签
    #         texts = row[1:]  # 提取文本
    #
    #         # 将每个文本与标签写入新文件的一行
    #         for text in texts:
    #             # print([label,text])
    #             csv_writer.writerow([label, text])
