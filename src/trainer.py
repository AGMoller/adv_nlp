import os
import string

import datasets
import nltk
import numpy as np
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

import wandb
from config import DATA_DIR, PROCESSED_DATA_PATH, ROOT_DIR, SRC_DIR
from utils import read_jsonl

PREFIX = "summarize: "
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 64


def clean_text(text):
    sentences = nltk.sent_tokenize(text.strip())
    sentences_cleaned = [s for sent in sentences for s in sent.split("\n")]
    sentences_cleaned_no_titles = [
        sent
        for sent in sentences_cleaned
        if len(sent) > 0 and sent[-1] in string.punctuation
    ]
    text_cleaned = "\n".join(sentences_cleaned_no_titles)
    return text_cleaned


def preprocess_data(examples):

    model_checkpoint = "strombergnlp/dant5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    texts_cleaned = [clean_text(text) for text in examples["text"]]
    inputs = [PREFIX + text for text in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["summary"], max_length=MAX_TARGET_LENGTH, truncation=True
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = [
        "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
    ]
    decoded_labels = [
        "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
    ]

    # Compute ROUGE scores
    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )

    # Extract ROUGE f1 scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # Add mean generated length to metrics
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


def model_init():
    model_checkpoint = "strombergnlp/dant5-small"
    return AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)


if __name__ == "__main__":

    torch.cuda.set_device(1)

    wandb.init(project="danish-t5", entity="hrmussa")

    data = datasets.load_dataset("json", data_files="data/danewsroom.jsonl")

    # test = data["train"].train_test_split(
    #     test_size=int(data.num_rows["train"] * 0.2), seed=42, shuffle=True
    # )

    # val = test["train"].train_test_split(
    #     test_size=int(test.num_rows["train"] * 0.5), seed=42, shuffle=True
    # )

    train = data["train"].train_test_split(test_size=100000, seed=42, shuffle=True)
    val = data["train"].train_test_split(test_size=10000, seed=10, shuffle=True)
    test = data["train"].train_test_split(test_size=10000, seed=2, shuffle=True)

    data["train"] = train["test"]
    data["validation"] = val["test"]
    data["test"] = test["test"]

    tokenized_datasets = data.map(preprocess_data, batched=True)

    batch_size = 8
    model_name = "t5-da-test"
    model_dir = SRC_DIR / model_name

    args = Seq2SeqTrainingArguments(
        model_dir,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=200,
        learning_rate=4e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=2,
        predict_with_generate=True,
        fp16=False,
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        report_to="wandb",
        run_name="t5-da-test",
    )

    model_checkpoint = "strombergnlp/dant5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    data_collator = DataCollatorForSeq2Seq(tokenizer)
    metric = datasets.load_metric("rouge")

    trainer = Seq2SeqTrainer(
        model_init=model_init,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # get test split
    # test_tokenized_dataset = tokenized_datasets["test"]

    # # pad texts to the same length
    # def preprocess_test(examples):
    #     inputs = [PREFIX + text for text in examples["text"]]
    #     model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True,
    #                             padding="max_length")
    #     return model_inputs

    # test_tokenized_dataset = test_tokenized_dataset.map(preprocess_test, batched=True)

    # # prepare dataloader
    # test_tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    # dataloader = torch.utils.data.DataLoader(test_tokenized_dataset, batch_size=32)

    # # generate text for each batch
    # all_predictions = []
    # for i,batch in enumerate(dataloader):
    #     predictions = model.generate(**batch)
    #     all_predictions.append(predictions)

    # # flatten predictions
    # all_predictions_flattened = [pred for preds in all_predictions for pred in preds]

    # # tokenize and pad titles
    # all_titles = tokenizer(test_tokenized_dataset["title"], max_length=max_target_length,
    #                     truncation=True, padding="max_length")["input_ids"]

    # # compute metrics
    # predictions_labels = [all_predictions_flattened, all_titles]
    # compute_metrics(predictions_labels)
    # # {'gen_len': 13.0259,
    # # 'rouge1': 37.9268,
    # # 'rouge2': 24.4912,
    # # 'rougeL': 35.9087,
    # # 'rougeLsum': 35.9278}

    wandb.finish()

    # a = 1
