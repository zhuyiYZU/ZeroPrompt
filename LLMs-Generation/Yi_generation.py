from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import csv
import pandas as pd

class TextClassifier:
    def __init__(self, model_path):
        self.model_path = model_path

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype='auto'
        ).eval()
        self.model.to(self.device)

    def get_examples(self, path):
        examples = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, headline = row
                text_a = headline.replace('\\', ' ')

                examples.append(text_a)
        return examples
    def generation_prompt(self, input_contents):
        test_result = {}
        #prompt

        classification_prompt = f"Task: You will act as a tool for syntactic paraphrasing, completing the task of rephrasing sentences for me. \n"
        classification_prompt += f"Question: Please generate ten sentences with the same meaning, but with different wording and sentence structure based on the following input content: \n"
        classification_prompt += f"Input content: {input_contents}\n"

        classification_prompt += "Requirement: The sentence I input contains a {'mask'} symbol, and these symbols should remain in the sentences you provide."
        classification_prompt += "Your output is:"

        text_with_prompt = classification_prompt
        messages = [
            {"role": "user", "content": text_with_prompt}
        ]

        input_ids = self.tokenizer.apply_chat_template(conversation=messages, tokenize=True,
                                                       add_generation_prompt=True, return_tensors='pt')
        output_ids = self.model.generate(input_ids.to(self.device))
        response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        return response

    def generation_example(self, path):
        test_result = []
        # prompt
        #'/home/mjy/wangye/zero_shot/datasets/TextClassification/snippets/train.csv'
        examples = Yi.get_examples(path)

        for example in examples:
            classification_prompt = f"Task: You will act as a tool for text paraphrasing, completing the task of rephrasing text for me. \n"
            classification_prompt += f"Question: Please generate ten sentences with the same meaning but different expressions based on the following input content: \n"
            classification_prompt += f"Input content: {example}\n"

            classification_prompt += "Requirement: These sentences should convey the same meaning, but the sentence structure must be different."
            classification_prompt += "Your output is:"

            text_with_prompt = classification_prompt
            messages = [
                {"role": "user", "content": text_with_prompt}
            ]

            input_ids = self.tokenizer.apply_chat_template(conversation=messages, tokenize=True,
                                                           add_generation_prompt=True, return_tensors='pt')
            output_ids = self.model.generate(input_ids.to(self.device))
            response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
            test_result.append(response)
            print(test_result)
        return response
    def predict(self, path):
        test_result = []

        #'/home/mjy/wangye/zero_shot/datasets/TextClassification/snippets/train.csv'
        examples = Yi.get_examples(path)

        for example in examples:
            classification_prompt = f"Task: You will work as a short text classification tool to classify texts into " \
                                    f"corresponding categories based on the texts and categories I provide. \n "
            classification_prompt += f"Question: Based on the following input, determine which category it belongs to " \
                                     f"and classify it as:" \

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
            messages = [
                {"role": "user", "content": text_with_prompt}
            ]

            input_ids = self.tokenizer.apply_chat_template(conversation=messages, tokenize=True,
                                                           add_generation_prompt=True, return_tensors='pt')
            output_ids = self.model.generate(input_ids.to(self.device))
            response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
            test_result.append(response)
            print(response)
        return test_result
model_path = '/opt/heimao/app/Yi-34B-Chat-4bits'
Yi = TextClassifier(model_path=model_path)
# response = Yi.generation_prompt('This is a sentence that expresses the emotion {"mask"}')
pre_examples = Yi.predict('/home//zero_shot/datasets/TextClassification/agnewstitle/test.csv')

test_data1 = pd.read_csv('/home//zero_shot/result/Yi_agnewstitle.csv')
# test_data2 = test_data1.drop(test_data1.columns[0], axis=1)

test_data1.insert(loc=0,column='predict_label', value=pre_examples)


test_data1.to_csv('/home//zero_shot/result/Yi_agnewstitle.csv', index=False, header=False,quoting=1)
