import torch
from modelscope import snapshot_download
from modelscope import AutoModelForCausalLM
from transformers import AutoTokenizer
import csv
import transformers
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score


model_dir = snapshot_download('FlagAlpha/Llama3-Chinese-8B-Instruct')
pipeline = transformers.pipeline(
    "text-generation",
    model=model_dir,
    model_kwargs={"torch_dtype": torch.float16},
    device="cuda",
)



output_path = 'result/output_twitter_llama3.csv'
test_path = 'test_data/twitter_test.csv'
data = pd.read_csv(test_path,header=None)
test_list = data.iloc[:,0].tolist()
label_list = data.iloc[:,1].tolist()




test_result = []
with open(test_path,mode='r',encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        messages = []
        #snippets
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

        #TMN
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
        #                          "business, output: [1], judged as computers, output: [2], judged as culture-arts-entertainment, " \
        #                          "output: [3], judged as education-science, output: [4],judged as engineering, output: [5],judged as health, "\
        #                           "output: [6],judged as politics-society, output: [7], "
        # classification_prompt += "Your output is:："

        classification_prompt = f"Task: You will work as a short text classification tool to classify texts into " \
                                f"corresponding categories based on the texts and categories I provide. \n "
        classification_prompt += f"Question: Based on the following input, determine which category it belongs to " \
                                 f"and classify it as:\n"
        classification_prompt += f"[0] negative\n "
        classification_prompt += f"[1] positive\n"

        classification_prompt += f"input content: {row[0]}\n"

        classification_prompt += "Requirements: In the output, you do not need to output the content of the " \
                                 "parse, do not need to explain, only need to output the category number, for example, judged as " \
                                 "negative, output: [0], judged as positive, output: [1] \n"
        classification_prompt += "Your output is:"

        text_with_prompt = classification_prompt

        print("user:" + text_with_prompt)
        messages.append({"role": "user", "content": text_with_prompt})
        prompt = pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = pipeline(
            prompt,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )
        response_LLM = outputs[0]["generated_text"][len(prompt):]
        print('LLM:')
        print(response_LLM)


        if '0' in response_LLM:
            response_LLM = '0'
        # elif '2' in response_LLM:
        #     response_LLM = '2'
        # elif '3' in response_LLM:
        #     response_LLM = '3'
        # elif '4' in response_LLM:
        #     response_LLM = '4'
        # elif '5' in response_LLM:
        #     response_LLM = '5'
        # elif '6' in response_LLM:
        #     response_LLM = '6'
        # elif '7' in response_LLM:
        #     response_LLM = '7'
        else:
            response_LLM = '1'
        test_result.append(response_LLM)

with open(output_path,mode='w',newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['true_label','pred_label','test_content'])
    for a, b, c in zip(label_list,test_result,test_list):
        writer.writerow([a, b, c])

df = pd.read_csv(output_path)
true_labels = df['true_label']
pred_labels = df['pred_label']

true_labels = true_labels.astype(int)
pred_labels = pred_labels.astype(int)

acc = accuracy_score(true_labels,pred_labels)
pre = precision_score(true_labels,pred_labels,average='micro')
rec = recall_score(true_labels,pred_labels,average='micro')
f1 = f1_score(true_labels,pred_labels,average='micro')

print(f'acc:{acc:.4f}')
print(f'pre:{pre:.4f}')
print(f'rec:{rec:.4f}')
print(f'f1:{f1:.4f}')
