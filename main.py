import os
import re
import json
import torch
from peft import PeftModel
from peft import LoraConfig, get_peft_model
import os
import evaluate
from datasets import Dataset, load_from_disk
from datasets import load_dataset
import pyarrow as pa
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, \
    Seq2SeqTrainer, BitsAndBytesConfig, AutoModelForCausalLM
import numpy as np
import wandb
wandb.login(key = '239be5b07ed02206e0e9e1c0afc955ee13a98900')
os.environ["WANDB_PROJECT"]="T5-finetune"
metric = evaluate.load("sacrebleu")



def read_file(f_name):
    data = []
    with open(f_name, 'r', encoding='utf-8') as f:
        sent_tuples = []
        txt = False
        for l in f:
            l = l.strip()

            if len(l) == 0:
                if txt:
                    data.append(sent_tuples)
                sent_tuples = []
                txt = False
            elif l.startswith('{'):
                try:
                    json.loads(l)
                except:
                    print(l)
                    print(f)
                sent_tuples.append(json.loads(l))
            else:
                # text line
                txt = True

        if txt:
            data.append(sent_tuples)

    return data


def convert_quintuple(q):
    quintuple = json.loads(q)
    subject_value = " ".join([s.split("&&")[1] for s in quintuple["subject"]])
    object_value = " ".join([o.split("&&")[1] for o in quintuple["object"]])
    aspect_value = " ".join([a.split("&&")[1] for a in quintuple["aspect"]])
    predicate_value = " ".join([p.split("&&")[1] for p in quintuple["predicate"]])
    label_value = quintuple["label"]
    if len(quintuple["subject"]) == 0:
        subject_value = 'unknown'
    if len(quintuple["object"]) == 0:
        object_value = 'unknown'
    if len(quintuple["aspect"]) == 0:
        aspect_value = 'unknown'
    formatted_output = f"{subject_value}, {object_value}, {aspect_value}, {predicate_value}, {label_value}"
    res = '(' + formatted_output + ')'
    return res


def convert_dataset(path):
    files = os.listdir(path)
    data_dict = {}
    for f in files:
        file_path = os.path.join(path, f)
        # a = read_file(file_path)
        # print(a)
        senandtuple = []

        with open(file_path, 'r', encoding='utf-8') as file:
            line = file.read().split('\n\n')
            for l in line:
                if l == '':
                    continue
                sentence = l.split('\t')[1]
                if 'alt' in l or 'des' in l or 'title' in l or 'Title' in l:
                    continue
                else:
                    senandtuple.append(sentence)
        quintuple = []
        current_sentence = None
        for item in senandtuple:
            a = item.split('\n')
            if len(a) == 1:
                data_dict[a[0]] = '(unknown, unknown, unknown, unknown, unknown)'
            elif len(a) == 2:
                data_dict[a[0]] = convert_quintuple(a[1])
            elif len(a) > 2:
                list_quintuple = ''
                for i in range(1, len(a)):
                    if i != len(a) - 1:
                        list_quintuple = list_quintuple + convert_quintuple(a[i]) + ';'
                    else:
                        list_quintuple = list_quintuple + convert_quintuple(a[i])
                data_dict[a[0]] = list_quintuple
    # dct = {k: [v] for k, v in data_dict.items()}
    df = pd.DataFrame(list(data_dict.items()), columns=['key', 'value'])
    # print(df.head())
    hg_ds = Dataset.hg_dataset = Dataset(pa.Table.from_pandas(df))
    return hg_ds


if __name__ == '__main__':
    train_ds = convert_dataset('/kaggle/working/peft/VLSP2023_ComOM_training_v2')
    # train_ds.save_to_disk('train_dataset')
    dev_ds = convert_dataset('/kaggle/working/peft/VLSP2023_ComOM_dev_v2')
    # dev_ds.save_to_disk('dev_dataset')
    #test_ds = convert_dataset('/kaggle/working/T5_fine-tune/VLSP2023_ComOM_testing_v2')
    # test_ds.save_to_disk('test_dataset')
    # train_ds = load_from_disk('train_dataset')
    # # dev_ds  = load_from_disk('dev_dataset')
    # # test_ds = load_from_disk('test_dataset')
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    # model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-large",device_map={"":0},
    # trust_remote_code=True,
    # quantization_config=bnb_config)
    model = AutoModelForCausalLM.from_pretrained("1TuanPham/bkai-vietnamese-llama2-7b-sharded",device_map={"":0},
    trust_remote_code=True,
    quantization_config=bnb_config)
    prefix = 'Please extract five elements including subject, object, aspect, predicate, and comparison type in the sentence'
    max_input_length = 156
    max_target_length = 156
    model.config.use_cache = False
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    def preprocess_function(examples):
        inputs = [prefix + ex + '###>' for ex in examples['key']]
        targets = [ex for ex in examples['value']]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    tokenized_ds_train = train_ds.map(preprocess_function, batched=True)
    #tokenized_ds_test = test_ds.map(preprocess_function, batched=True)
    tokenized_ds_dev = dev_ds.map(preprocess_function, batched=True)
    args = Seq2SeqTrainingArguments(
        "T5_fine_tune",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=25,
        predict_with_generate=True,
        report_to='wandb',

    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels


    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        #meteor_result = meteor.compute(predictions=decoded_preds, references=decoded_labels)
        #prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        # f1_metric = evaluate.load('f1')
        # macro = f1_metric.compute(predictions = decoded_preds, references = decoded_labels, average = 'macro')
        return result


    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_ds_train,
        eval_dataset=tokenized_ds_dev,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model()
