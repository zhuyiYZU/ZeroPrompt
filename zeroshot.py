
# from openprompt.utils.custom_tqdm import tqdm
from tqdm import tqdm

# from fewshot import accelerator
from openprompt.data_utils.text_classification_dataset import *
import torch
from openprompt.data_utils.utils import InputExample
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import *
from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer, KnowledgeableVerbalizer
from openprompt.prompts import ManualTemplate, PtuningTemplate
from accelerate import Accelerator

parser = argparse.ArgumentParser("")
parser.add_argument("--shot", type=int, default=0)
parser.add_argument("--seed", type=int, default=144)

parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--model", type=str, default='xlmroberta')
parser.add_argument("--model_name_or_path", default='/home/w790-ace/Models/xlm-roberta-large')
parser.add_argument("--verbalizer", type=str)
parser.add_argument("--calibration", action="store_true")
parser.add_argument("--nocut", action="store_true")
parser.add_argument("--filter", default="none", type=str)
parser.add_argument("--template_id", type=int)
parser.add_argument("--max_token_split", default=-1, type=int)
parser.add_argument("--dataset",type=str)
parser.add_argument("--result_file", type=str, default="../sfs_scripts/results_zeroshot.txt")
parser.add_argument("--write_filter_record", action="store_true")
parser.add_argument("--learning_rate", default=5e-5, type=float)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--template_name", default='gpt35_15_0_template', type=str)
parser.add_argument("--template_num", default=15, type=int)
parser.add_argument("--gpt_path", default='gpt35-5-data', type=str)

args = parser.parse_args()

from openprompt.utils.reproduciblity import set_seed
set_seed(args.seed)

from openprompt.plms import load_plm
plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

dataset = {}

# if args.dataset == "twitter":
#     dataset['train'] = TwtProcessor().get_train_examples(f"./datasets/TextClassification/twitter/gpt35-{args.template_num}-data")
#     dataset['test'] = TwtProcessor().get_test_examples(f"./datasets/TextClassification/twitter/gpt35-{args.template_num}-data")
#     class_labels =TwtProcessor().get_labels()
#     scriptsbase = "TextClassification/twitter"
#     scriptformat = "txt"
#     cutoff=0.5 if (not args.nocut) else 0.0
#     max_seq_l = 128
#     batch_s = 30
# elif args.dataset == "newstitle":
#     dataset['train'] = NewstitleProcessor().get_train_examples(f"./datasets/TextClassification/newstitle/gpt35-{args.template_num}-data")
#     dataset['test'] = NewstitleProcessor().get_test_examples(f"./datasets/TextClassification/newstitle/gpt35-{args.template_num}-data")
#     class_labels = NewstitleProcessor().get_labels()
#     scriptsbase = "TextClassification/newstitle"
#     scriptformat = "txt"
#     cutoff = 0.5
#     max_seq_l = 128
#     batch_s = args.batch_size
# elif args.dataset == "agnewstitle":
#     dataset['train'] = AgnewsTitleProcessor().get_train_examples(f'./datasets/TextClassification/agnewstitle/gpt35-{args.template_num}-data')
#     dataset['test'] = AgnewsTitleProcessor().get_test_examples(f'./datasets/TextClassification/agnewstitle/gpt35-{args.template_num}-data')
#     class_labels = AgnewsTitleProcessor().get_labels()
#     scriptsbase = "TextClassification/agnewstitle"
#     scriptformat = "txt"
#     cutoff = 0.5
#     max_seq_l = 128
#     batch_s = args.batch_size
# elif args.dataset == "MR":
#     dataset['train'] = TwtProcessor().get_train_examples(f"./datasets/TextClassification/MR/gpt35-{args.template_num}-data/")
#     dataset['test'] = TwtProcessor().get_test_examples(f"./datasets/TextClassification/MR/gpt35-{args.template_num}-data/")
#     class_labels = TwtProcessor().get_labels()
#     scriptsbase = "TextClassification/MR"
#     scriptformat = "txt"
#     cutoff = 0.5
#     max_seq_l = 128
#     batch_s = args.batch_size
# elif args.dataset == "SST-2":
#     dataset['train'] = TwtProcessor().get_train_examples(f"./datasets/TextClassification/SST-2/gpt35-{args.template_num}-data/")
#     dataset['test'] = TwtProcessor().get_test_examples(f"./datasets/TextClassification/SST-2/gpt35-{args.template_num}-data/")
#     class_labels = TwtProcessor().get_labels()
#     scriptsbase = "TextClassification/SST-2"
#     scriptformat = "txt"
#     cutoff = 0.5
#     max_seq_l = 128
#     batch_s = args.batch_size
# elif args.dataset == "snippets":
#     dataset['train'] = SnippetsProcessor().get_train_examples(f'./datasets/TextClassification/snippets/gpt35-{args.template_num}-data')
#     dataset['test'] = SnippetsProcessor().get_test_examples(f'./datasets/TextClassification/snippets/gpt35-{args.template_num}-data')
#     class_labels = SnippetsProcessor().get_labels()
#     scriptsbase = "TextClassification/snippets"
#     scriptformat = "txt"
#     cutoff = 0.5
#     max_seq_l = 128
#     batch_s = args.batch_size
if args.dataset == "agnewstitle":
    dataset['train'] = UnlabeledDataProcessor().get_train_examples(f'./datasets/generation_data')
    dataset['test'] = UnlabeledDataProcessor().get_test_examples(f'./datasets/generation_data')
    class_labels = UnlabeledDataProcessor().get_labels()
    scriptsbase = "TextClassification/agnewstitle"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 128
    batch_s = args.batch_size
elif args.dataset == "snippets":
    dataset['train'] = SnippetsUnlabeledDataProcessor().get_train_examples(f'./datasets/generation_data')
    dataset['test'] = SnippetsUnlabeledDataProcessor().get_test_examples(f'./datasets/generation_data')
    class_labels = SnippetsUnlabeledDataProcessor().get_labels()
    scriptsbase = "TextClassification/snippets"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 128
    batch_s = args.batch_size
elif args.dataset == "newstitle":
    dataset['train'] = NewstitleUnlabeledDataProcessor().get_train_examples(f'./datasets/generation_data')
    dataset['test'] = NewstitleUnlabeledDataProcessor().get_test_examples(f'./datasets/generation_data')
    class_labels = NewstitleUnlabeledDataProcessor().get_labels()
    scriptsbase = "TextClassification/newstitle"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 128
    batch_s = args.batch_size
elif args.dataset == "SST-2":
    dataset['train'] = EmotionUnlabeledDataProcessor().get_train_examples(f'./datasets/generation_data')
    dataset['test'] = EmotionUnlabeledDataProcessor().get_test_examples(f'./datasets/generation_data')
    class_labels = EmotionUnlabeledDataProcessor().get_labels()
    scriptsbase = "TextClassification/SST-2"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 128
    batch_s = args.batch_size
elif args.dataset == "twitter":
    dataset['train'] = EmotionUnlabeledDataProcessor().get_train_examples(f'./datasets/generation_data')
    dataset['test'] = EmotionUnlabeledDataProcessor().get_test_examples(f'./datasets/generation_data')
    class_labels = EmotionUnlabeledDataProcessor().get_labels()
    scriptsbase = "TextClassification/twitter"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 128
    batch_s = args.batch_size
elif args.dataset == "MR":
    dataset['train'] = EmotionUnlabeledDataProcessor().get_train_examples(f'./datasets/generation_data')
    dataset['test'] = EmotionUnlabeledDataProcessor().get_test_examples(f'./datasets/generation_data')
    class_labels = EmotionUnlabeledDataProcessor().get_labels()
    scriptsbase = "TextClassification/MR"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 128
    batch_s = args.batch_size
else:
    raise NotImplementedError

#

mytemplate = ManualTemplate(tokenizer=tokenizer).from_file(f"scripts/{scriptsbase}/{args.template_name}.txt", choice=args.template_id)
# mytemplate = PtuningTemplate(model=plm, tokenizer=tokenizer).from_file(f"./scripts/{scriptsbase}/ptuning_template.txt", choice=args.template_id)

if args.verbalizer == "kpt":
    myverbalizer = KnowledgeableVerbalizer(tokenizer, classes=class_labels, candidate_frac=cutoff, max_token_split=args.max_token_split).from_file(f"scripts/{scriptsbase}/knowledgeable_verbalizer.{scriptformat}")
elif args.verbalizer == "manual":
    myverbalizer = ManualVerbalizer(tokenizer, classes=class_labels).from_file(f"scripts/{scriptsbase}/manual_verbalizer.{scriptformat}")
elif args.verbalizer == "soft":
    raise NotImplementedError
elif args.verbalizer == "auto":
    raise NotImplementedError

# (contextual) calibration
if args.calibration:
    from openprompt.data_utils.data_sampler import FewShotSampler
    support_sampler = FewShotSampler(num_examples_total=200, also_sample_dev=False)
    dataset['support'] = support_sampler(dataset['train'], seed=args.seed)

    for example in dataset['support']:
        example.label = -1 # remove the labels of support set for clarification
    support_dataloader = PromptDataLoader(dataset=dataset["support"], template=mytemplate, tokenizer=tokenizer, 
        tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3, 
        batch_size=batch_s,shuffle=False, teacher_forcing=False, predict_eos_token=False,
        truncate_method="tail")


from openprompt import PromptForClassification

accelerator = Accelerator()
device = accelerator.device
use_cuda = True
prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False, plm_eval_mode=args.plm_eval_mode)
# if use_cuda:
#     prompt_model=  prompt_model.cuda()

prompt_model = accelerator.prepare(prompt_model)
myrecord = ""
# HP
if args.calibration:
    org_label_words_num = [len(prompt_model.verbalizer.label_words[i]) for i in range(len(class_labels))]
    from openprompt.utils.calibrate import calibrate
    # calculate the calibration logits
    cc_logits = calibrate(prompt_model, support_dataloader)
    print("the calibration logits is", cc_logits)
    myrecord += "Phase 1 {}\n".format(org_label_words_num)

    myverbalizer.register_calibrate_logits(cc_logits)
    new_label_words_num = [len(myverbalizer.label_words[i]) for i in range(len(class_labels))]
    myrecord += "Phase 2 {}\n".format(new_label_words_num)


    # from filter_method import *
    # if args.filter == "tfidf_filter":
    #     record = tfidf_filter(myverbalizer, cc_logits, class_labels)
    #     myrecord += record
    # elif args.filter == "none":
    #     pass
    # else:
    #     raise NotImplementedError

    
    # register the logits to the verbalizer so that the verbalizer will divide the calibration probability in producing label logits
    # currently, only ManualVerbalizer and KnowledgeableVerbalizer support calibration.
    
#
if args.write_filter_record:
    record_prefix = "="*20+"\n"
    record_prefix += f"dataset {args.dataset}\t"
    record_prefix += f"temp {args.template_id}\t"
    record_prefix += f"seed {args.seed}\t"
    record_prefix += f"cali {args.calibration}\t"
    record_prefix += f"filt {args.filter}\t"
    record_prefix += "\n"
    myrecord = record_prefix +myrecord
    with open("../sfs_scripts/filter_record_file.txt",'a')  as fout_rec:
        fout_rec.write(myrecord)
    exit()


# zero-shot test
test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3, 
    batch_size=batch_s,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")

test_dataloader = accelerator.prepare(test_dataloader)

allpreds = []
alllabels = []
pbar = tqdm(test_dataloader)
with torch.no_grad():
    for step, inputs in enumerate(pbar):
        # if use_cuda:
        #     inputs = inputs.cuda()
        # inputs = {key: value.to(accelerator.device) for key,value in inputs.items()}
        inputs = inputs.to(accelerator.device)
        logits = prompt_model(inputs)
        labels = inputs['label']
        # alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())



# 定义目标路径和备用路径

target_path = f'./datasets/generation_data/{args.template_name}_{args.dataset}_m_label.csv'
# target_path = f'./datasets/generation_data/{args.dataset}_mutil_label.csv'
backup_path = f'./datasets/generation_data/labeled_test.csv'

# 检查目标路径是否存在
if not os.path.exists(target_path):
    # 如果目标路径不存在，从备用路径读取数据并保存到目标路径
    if os.path.exists(backup_path):
        data = pd.read_csv(backup_path)
        data.to_csv(target_path, quoting=1, index=False)
        print(f"目标路径不存在，已从备用路径读取数据并保存到 {target_path}")
    else:
        raise FileNotFoundError(f"备用路径 {backup_path} 不存在，无法读取数据！")
else:
    # 如果目标路径存在，直接读取数据
    data = pd.read_csv(target_path)

# 插入新列
data.insert(0, 'label_1', allpreds, allow_duplicates=True)

# 保存数据
data.to_csv(target_path, quoting=1, index=False)
print(f"数据已更新并保存到 {target_path}")



  # roughly ~0.853 when using template 0



content_write = "="*20+"\n"
content_write += f"dataset {args.dataset}\t"
content_write += f"temp {args.template_id}\t"
content_write += f"seed {args.seed}\t"
content_write += f"verb {args.verbalizer}\t"
content_write += f"cali {args.calibration}\t"
content_write += f"filt {args.filter}\t"
content_write += f"nocut {args.nocut}\t"
content_write += f"maxsplit {args.max_token_split}\t"

content_write += "\n"
# content_write += f"Acc: {acc}\t"
# content_write += f"Pre: {pre}\t"
# content_write += f"Rec: {recall}\t"
# content_write += f"F1s: {F1score}\t"
content_write += "\n\n"

print(content_write)

with open(f"{args.result_file}", "a") as fout:
    fout.write(content_write)
