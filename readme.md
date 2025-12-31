# ZeroPrompt: Soft Prompt-tuning for Zero-shot Text Classification

## Installation

First, install all dependencies by running:

```sh
pip install -r requirement.txt
```
Then, download the corresponding model weights in the model folder and modify the file paths accordingly.

## Running the Scripts
You can generate short text data for the corresponding categories using the generate.py file, Place the unlabeled data into the corresponding dataset folder, and be sure to modify the path, and then obtain the true labels of the data by running the auto_run_zero.py file.
```sh
python generate.py
```
You can run the script directly using:
```sh
python auto_run_zero.py
```

The prompts constructed by the large model will label the corresponding data, and all data will be placed in the datasets/TextClassification/ folder. Then run:
```sh
python auto_run_fewshot.py
```
to train the soft prompt-tuning model with pseudo-labeled data and make predictions.

## Example Shell Scripts
You can also run the corresponding files separately. Example shell scripts:
```sh
python fewshot.py --result_file ./output_fewshot.txt --dataset agnews --template_id 0 --seed 144 --shot 1 --verbalizer kpt
python zeroshot.py --result_file ./output_zeroshot.txt --dataset agnews --template_id 0 --seed 123 --verbalizer kpt
```