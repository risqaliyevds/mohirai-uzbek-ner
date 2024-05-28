import json
import os
import re
from copy import deepcopy

import numpy as np
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import Trainer, TrainingArguments

from core.sentence import wav2vec2_normalize_sentence

ners_tags = [
    "LOC",
    "ORG",
    "PERSON",
    "DATE",
    "MONEY",
    "PERCENT",
    "QUANTITY",
    "TIME",
    "PRODUCT",
    "EVENT",
    "WORK_OF_ART",
    "LANGUAGE",
    "CARDINAL",
    "ORDINAL",
    "NORP",
    "FACILITY",
    "LAW",
    "GPE",
]


def align_labels_with_input_ids(word_ids, old_labels):
    new_labels = []
    prev_word_id = None

    for word_id in word_ids:
        if word_id is None:
            # if the word_id is None, i.e. the token is [CLS] or [SEP]
            new_labels.append(-100)
        else:
            label = old_labels[word_id]
            if prev_word_id == word_id and label % 2 == 1:
                # label is intermediate i.e. I-XXX
                label += 1
            new_labels.append(label)
        prev_word_id = word_id
    return new_labels


def tokenize_and_align(examples, tokenizer):
    # tokenize examples
    model_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )

    # align labels
    model_inputs["labels"] = []
    # iterate over each example
    for i in range(len(model_inputs["input_ids"])):
        # get word_ids
        word_ids = model_inputs.word_ids(i)
        # get labels
        ner_tags = examples["ner_tags"][i]
        # compute new labels
        new_labels = align_labels_with_input_ids(word_ids, ner_tags)
        # store new labels
        model_inputs["labels"].append(new_labels)

    return model_inputs


def prepare_dataset(json_file, tokenizer, label_2_id):
    with open(json_file, "r") as f:
        dataset = json.load(f)

    print(f"Number of samples: {len(dataset)}")

    id_2_label = {v: k for k, v in label_2_id.items()}

    dataset_for_ner = []
    for id, chunk in tqdm(enumerate(dataset), total=len(dataset), desc="Processing"):
        labels = chunk["labels"]
        labels = [label_2_id[label] for label in labels]

        words = chunk["text"].split(" ")

        dataset_for_ner.append({"id": id + 1, "tokens": words, "ner_tags": labels})

    dataset = Dataset.from_list(dataset_for_ner)
    dataset = dataset.shuffle(seed=73)
    dataset = dataset.train_test_split(test_size=1000)

    print(f"Train {dataset['train']},\nTest: {dataset['test']}")

    dataset = dataset.map(
        lambda examples: tokenize_and_align(examples, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    print(f"Train {dataset['train']},\nTest: {dataset['test']}")

    return dataset


def compute_metrics(pred, tokenizer):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # Filter out -100 labels
    valid_indices = labels != -100
    true_labels = labels[valid_indices]
    pred_labels = preds[valid_indices]

    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="weighted"
    )
    accuracy = np.sum(true_labels == pred_labels) / len(true_labels)

    print("\n\n\n ---------------------------------")
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
    print(" ---------------------------------\n\n\n")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def main():
    print("Start training")

    label_2_id = {
        "O": 0,
        "B-LOC": 1,
        "I-LOC": 2,
        "B-ORG": 3,
        "I-ORG": 4,
        "B-PERSON": 5,
        "I-PERSON": 6,
        "B-DATE": 7,
        "I-DATE": 8,
        "B-MONEY": 9,
        "I-MONEY": 10,
        "B-PERCENT": 11,
        "I-PERCENT": 12,
        "B-QUANTITY": 13,
        "I-QUANTITY": 14,
        "B-TIME": 15,
        "I-TIME": 16,
        "B-PRODUCT": 17,
        "I-PRODUCT": 18,
        "B-EVENT": 19,
        "I-EVENT": 20,
        "B-WORK_OF_ART": 21,
        "I-WORK_OF_ART": 22,
        "B-LANGUAGE": 23,
        "I-LANGUAGE": 24,
        "B-CARDINAL": 25,
        "I-CARDINAL": 26,
        "B-ORDINAL": 27,
        "I-ORDINAL": 28,
        "B-NORP": 29,
        "I-NORP": 30,
        "B-FACILITY": 31,
        "I-FACILITY": 32,
        "B-LAW": 33,
        "I-LAW": 34,
        "B-GPE": 35,
        "I-GPE": 36,
    }

    id_2_label = {v: k for k, v in label_2_id.items()}

    model = AutoModelForTokenClassification.from_pretrained(
        os.getenv("MODEL_PATH"),
        id2label=id_2_label,
        label2id=label_2_id,
        ignore_mismatched_sizes=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(os.getenv("MODEL_PATH"))

    dataset = prepare_dataset(os.getenv("JSON_FILE"), tokenizer, label_2_id)
    print("Dataset loaded")

    print(f"Number of classes: {model.config.num_labels}")
    print("Sample of dataset")
    print(dataset["train"][0])

    data_collator = DataCollatorForTokenClassification(tokenizer)

    batch_size = int(os.environ.get("BATCH_SIZE", 400))
    logging_steps = 1
    num_train_epochs = 3
    lr_initial = 1e-5
    # weight_decay = 1e-3
    output_dir = os.environ.get("OUTPUT_DIR", "output")
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_steps=1,
        num_train_epochs=num_train_epochs,
        learning_rate=lr_initial,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        # weight_decay=weight_decay,
        evaluation_strategy="steps",
        push_to_hub=False,
        # max_steps=int(os.environ.get("MAX_STEPS", 5000)),
        eval_steps=int(os.environ.get("EVAL_SAVE_STEPS", 1000)),
        save_steps=int(os.environ.get("EVAL_SAVE_STEPS", 1000)),
        log_level="error",
        report_to="tensorboard",
        # save_best_model=True,
        save_total_limit=int(os.environ.get("SAVE_TOTAL_LIMIT", 3)),
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=lambda pred: compute_metrics(pred, tokenizer),
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()


if __name__ == "__main__":
    main()

# cd /mnt/sdb/stt
# export PYTHONPATH=../stt
# source venv-3.10/bin/activate
#
# export CUDA_VISIBLE_DEVICES=2
#
# export SAVE_TOTAL_LIMIT=3
# export BATCH_SIZE=16
# export EVAL_SAVE_STEPS=1000
# export MAX_STEPS=500
#
# export JSON_FILE=../for_ner_model_huggingface/uzbek_ner.json
# export MODEL_PATH=FacebookAI/xlm-roberta-large-finetuned-conll03-english
# export OUTPUT_DIR=../ner_models/train/xlm-roberta-large-finetuned-conll03-english-hug-28-05-2024
# python ../ner_roberta/train.py
