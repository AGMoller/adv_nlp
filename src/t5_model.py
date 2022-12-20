from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

import wandb
from config import DATA_DIR, PROCESSED_DATA_PATH, ROOT_DIR, SRC_DIR
from utils import read_jsonl

PREFIX = "summarize: "
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 64


def get_device() -> Union[str, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


class T5_Model:
    def __init__(self, model_name: Union[str, Path], device):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.SUMMARIZE_PREFIX = "summarize: "
        self.MAX_INPUT_LENGTH = 512
        self.MAX_TARGET_LENGTH_SUMMARIZE = 64
        self.device = device

    def summarize(self, text: Union[str, List[str]]) -> str:

        if isinstance(text, str):
            text = [text]

        inputs = [self.SUMMARIZE_PREFIX + text for text in text]

        model_inputs = self.tokenizer(
            inputs,
            max_length=self.MAX_INPUT_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        model_inputs = model_inputs.to(get_device())

        summary_ids = self.model.generate(
            model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            num_beams=4,
            max_length=self.MAX_TARGET_LENGTH_SUMMARIZE,
            early_stopping=True,
        )

        summaries = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)

        return summaries

    def train(
        self,
        texts: List[str],
        summaries: List[str],
        num_training_epochs=1,
        batch_size=8,
    ):

        # Define adamW optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=4e-3, weight_decay=0.01
        )

        # cast model to device
        model = self.model.to(self.device)
        model.train()

        encoding = self.tokenizer(
            [self.SUMMARIZE_PREFIX + sequence for sequence in texts],
            padding="longest",
            max_length=self.MAX_INPUT_LENGTH,
            truncation=True,
            return_tensors="pt",
        )

        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

        # encode the targets
        target_encoding = self.tokenizer(
            summaries,
            padding="longest",
            max_length=self.MAX_TARGET_LENGTH_SUMMARIZE,
            truncation=True,
            return_tensors="pt",
        )
        labels = target_encoding.input_ids

        # replace padding token id's of the labels by -100 so it's ignored by the loss
        labels[labels == self.tokenizer.pad_token_id] = -100
        # the forward function automatically creates the correct decoder_input_ids
        loss = model(input_ids=input_ids, labels=labels).loss
        loss.item()

        loss.backward()

        optimizer.step()

        a = 1


if __name__ == "__main__":

    model = T5_Model("strombergnlp/dant5-small", get_device())

    texts = [
        "DR Nyheders medarbejder strejker, fordi den l\u00f8nnede frokost er blevet afskaffet. Tillidsmand frygter fyringer.\n\nDR's ledelse har valgt at afskaffe den l\u00f8nnede frokostpause. I protest over udmeldingen nedl\u00e6gger DR Nyheders medarbejdere arbejdet, skriver DR Nyheder.\n\nMedarbejderne besluttede at nedl\u00e6gge arbejdet fredag formiddag p\u00e5 et fagligt m\u00f8de. Arbejdsnedl\u00e6ggelsen er planlagt til l\u00f8rdag morgen klokken ni.\n\nStrejken betyder at nyhedsprogrammer p\u00e5 tv bliver erstattet af andre programmer. Radioaviserne og nyhedsd\u00e6kningen p\u00e5 dr.dk forts\u00e6tter dog i begr\u00e6nset omfang.\n\nI en udtalelse fra journalisternes faglige m\u00f8de begrundes den overenskomststridige arbejdsnedl\u00e6ggelse med utilfredshed med, at DR har varslet, at alle medarbejdere i DR fra sommeren 2017 selv skal betale deres spisepauser, s\u00e5ledes at den effektive arbejdstid forl\u00e6nges med en halv time i gennemsnit om dagen, skriver DR Nyheder.\n\nF\u00e6llestillidsmand i DR Henrik Friis har tidligere sagt til fagbladet Journalisten, at han forventer, at der kommer fyringer.\n\n- Medarbejderne g\u00f8r alt, hvad de kan for at v\u00e6re fleksible, de l\u00f8ber st\u00e6rkt, men f\u00f8ler ikke, at ledelsen anerkender indsatsen. Nu vil de s\u00e5 tage den betalte frokostpause. Det vil f\u00f8re til et k\u00e6mpe motivationstab og ramme arbejdsmilj\u00f8et, sagde f\u00e6llestillidsmand Henrik Friis Vilmar til Journalisten tirsdag.\n\n- Den eneste m\u00e5de, DR kan finde pengene p\u00e5, er p\u00e5 l\u00f8nninger. Hvis spisepausen bliver selvbetalt, s\u00e5 skal vi jo vagts\u00e6ttes 2,5 time mere om ugen. Det betyder, at man kan fyre et tilsvarende antal mennesker, for hvor skulle besparelsen ellers komme fra? l\u00f8d det fra Henrik Friis Vilmar.\n\nOm onsdagen fik Journalisten svar fra Nikolas Lyhne-Knudsen, direkt\u00f8r for \u00f8konomi, teknologi og medieproduktion i DR.\n\n- Det er forventningen, at der med den lange indfasning frem til juni 2017 vil kunne ske en gradvis tilpasning af organisationen, s\u00e5 antallet af afskedigelser kan reduceres s\u00e5 meget som muligt, skrev Nikolas Lyhne-Knudsen til Journalisten.\n\nLyhne-Knudsen skrev yderligere, at man vurderer, at afskaffelse af den l\u00f8nnede spisepause vil kunne frigive 45 millioner kroner \u00e5rligt.",
        "DR Nyheders medarbejder strejker, fordi den l\u00f8nnede frokost er blevet afskaffet. Tillidsmand frygter fyringer.\n\nDR's ledelse har valgt at afskaffe den l\u00f8nnede frokostpause. I protest over udmeldingen nedl\u00e6gger DR Nyheders medarbejdere arbejdet, skriver DR Nyheder.\n\nMedarbejderne besluttede at nedl\u00e6gge arbejdet fredag formiddag p\u00e5 et fagligt m\u00f8de. Arbejdsnedl\u00e6ggelsen er planlagt til l\u00f8rdag morgen klokken ni.\n\nStrejken betyder at nyhedsprogrammer p\u00e5 tv bliver erstattet af andre programmer. Radioaviserne og nyhedsd\u00e6kningen p\u00e5 dr.dk forts\u00e6tter dog i begr\u00e6nset omfang.\n\nI en udtalelse fra journalisternes faglige m\u00f8de begrundes den overenskomststridige arbejdsnedl\u00e6ggelse med utilfredshed med, at DR har varslet, at alle medarbejdere i DR fra sommeren 2017 selv skal betale deres spisepauser, s\u00e5ledes at den effektive arbejdstid forl\u00e6nges med en halv time i gennemsnit om dagen, skriver DR Nyheder.\n\nF\u00e6llestillidsmand i DR Henrik Friis har tidligere sagt til fagbladet Journalisten, at han forventer, at der kommer fyringer.\n\n- Medarbejderne g\u00f8r alt, hvad de kan for at v\u00e6re fleksible, de l\u00f8ber st\u00e6rkt, men f\u00f8ler ikke, at ledelsen anerkender indsatsen. Nu vil de s\u00e5 tage den betalte frokostpause. Det vil f\u00f8re til et k\u00e6mpe motivationstab og ramme arbejdsmilj\u00f8et, sagde f\u00e6llestillidsmand Henrik Friis Vilmar til Journalisten tirsdag.\n\n- Den eneste m\u00e5de, DR kan finde pengene p\u00e5, er p\u00e5 l\u00f8nninger. Hvis spisepausen bliver selvbetalt, s\u00e5 skal vi jo vagts\u00e6ttes 2,5 time mere om ugen. Det betyder, at man kan fyre et tilsvarende antal mennesker, for hvor skulle besparelsen ellers komme fra? l\u00f8d det fra Henrik Friis Vilmar.\n\nOm onsdagen fik Journalisten svar fra Nikolas Lyhne-Knudsen, direkt\u00f8r for \u00f8konomi, teknologi og medieproduktion i DR.\n\n- Det er forventningen, at der med den lange indfasning frem til juni 2017 vil kunne ske en gradvis tilpasning af organisationen, s\u00e5 antallet af afskedigelser kan reduceres s\u00e5 meget som muligt, skrev Nikolas Lyhne-Knudsen til Journalisten.\n\nLyhne-Knudsen skrev yderligere, at man vurderer, at afskaffelse af den l\u00f8nnede spisepause vil kunne frigive 45 millioner kroner \u00e5rligt.",
        "DR Nyheders medarbejder strejker, fordi den l\u00f8nnede frokost er blevet afskaffet. Tillidsmand frygter fyringer.\n\nDR's ledelse har valgt at afskaffe den l\u00f8nnede frokostpause. I protest over udmeldingen nedl\u00e6gger DR Nyheders medarbejdere arbejdet, skriver DR Nyheder.\n\nMedarbejderne besluttede at nedl\u00e6gge arbejdet fredag formiddag p\u00e5 et fagligt m\u00f8de. Arbejdsnedl\u00e6ggelsen er planlagt til l\u00f8rdag morgen klokken ni.\n\nStrejken betyder at nyhedsprogrammer p\u00e5 tv bliver erstattet af andre programmer. Radioaviserne og nyhedsd\u00e6kningen p\u00e5 dr.dk forts\u00e6tter dog i begr\u00e6nset omfang.\n\nI en udtalelse fra journalisternes faglige m\u00f8de begrundes den overenskomststridige arbejdsnedl\u00e6ggelse med utilfredshed med, at DR har varslet, at alle medarbejdere i DR fra sommeren 2017 selv skal betale deres spisepauser, s\u00e5ledes at den effektive arbejdstid forl\u00e6nges med en halv time i gennemsnit om dagen, skriver DR Nyheder.\n\nF\u00e6llestillidsmand i DR Henrik Friis har tidligere sagt til fagbladet Journalisten, at han forventer, at der kommer fyringer.\n\n- Medarbejderne g\u00f8r alt, hvad de kan for at v\u00e6re fleksible, de l\u00f8ber st\u00e6rkt, men f\u00f8ler ikke, at ledelsen anerkender indsatsen. Nu vil de s\u00e5 tage den betalte frokostpause. Det vil f\u00f8re til et k\u00e6mpe motivationstab og ramme arbejdsmilj\u00f8et, sagde f\u00e6llestillidsmand Henrik Friis Vilmar til Journalisten tirsdag.\n\n- Den eneste m\u00e5de, DR kan finde pengene p\u00e5, er p\u00e5 l\u00f8nninger. Hvis spisepausen bliver selvbetalt, s\u00e5 skal vi jo vagts\u00e6ttes 2,5 time mere om ugen. Det betyder, at man kan fyre et tilsvarende antal mennesker, for hvor skulle besparelsen ellers komme fra? l\u00f8d det fra Henrik Friis Vilmar.\n\nOm onsdagen fik Journalisten svar fra Nikolas Lyhne-Knudsen, direkt\u00f8r for \u00f8konomi, teknologi og medieproduktion i DR.\n\n- Det er forventningen, at der med den lange indfasning frem til juni 2017 vil kunne ske en gradvis tilpasning af organisationen, s\u00e5 antallet af afskedigelser kan reduceres s\u00e5 meget som muligt, skrev Nikolas Lyhne-Knudsen til Journalisten.\n\nLyhne-Knudsen skrev yderligere, at man vurderer, at afskaffelse af den l\u00f8nnede spisepause vil kunne frigive 45 millioner kroner \u00e5rligt.",
        "DR Nyheders medarbejder strejker, fordi den l\u00f8nnede frokost er blevet afskaffet. Tillidsmand frygter fyringer.\n\nDR's ledelse har valgt at afskaffe den l\u00f8nnede frokostpause. I protest over udmeldingen nedl\u00e6gger DR Nyheders medarbejdere arbejdet, skriver DR Nyheder.\n\nMedarbejderne besluttede at nedl\u00e6gge arbejdet fredag formiddag p\u00e5 et fagligt m\u00f8de. Arbejdsnedl\u00e6ggelsen er planlagt til l\u00f8rdag morgen klokken ni.\n\nStrejken betyder at nyhedsprogrammer p\u00e5 tv bliver erstattet af andre programmer. Radioaviserne og nyhedsd\u00e6kningen p\u00e5 dr.dk forts\u00e6tter dog i begr\u00e6nset omfang.\n\nI en udtalelse fra journalisternes faglige m\u00f8de begrundes den overenskomststridige arbejdsnedl\u00e6ggelse med utilfredshed med, at DR har varslet, at alle medarbejdere i DR fra sommeren 2017 selv skal betale deres spisepauser, s\u00e5ledes at den effektive arbejdstid forl\u00e6nges med en halv time i gennemsnit om dagen, skriver DR Nyheder.\n\nF\u00e6llestillidsmand i DR Henrik Friis har tidligere sagt til fagbladet Journalisten, at han forventer, at der kommer fyringer.\n\n- Medarbejderne g\u00f8r alt, hvad de kan for at v\u00e6re fleksible, de l\u00f8ber st\u00e6rkt, men f\u00f8ler ikke, at ledelsen anerkender indsatsen. Nu vil de s\u00e5 tage den betalte frokostpause. Det vil f\u00f8re til et k\u00e6mpe motivationstab og ramme arbejdsmilj\u00f8et, sagde f\u00e6llestillidsmand Henrik Friis Vilmar til Journalisten tirsdag.\n\n- Den eneste m\u00e5de, DR kan finde pengene p\u00e5, er p\u00e5 l\u00f8nninger. Hvis spisepausen bliver selvbetalt, s\u00e5 skal vi jo vagts\u00e6ttes 2,5 time mere om ugen. Det betyder, at man kan fyre et tilsvarende antal mennesker, for hvor skulle besparelsen ellers komme fra? l\u00f8d det fra Henrik Friis Vilmar.\n\nOm onsdagen fik Journalisten svar fra Nikolas Lyhne-Knudsen, direkt\u00f8r for \u00f8konomi, teknologi og medieproduktion i DR.\n\n- Det er forventningen, at der med den lange indfasning frem til juni 2017 vil kunne ske en gradvis tilpasning af organisationen, s\u00e5 antallet af afskedigelser kan reduceres s\u00e5 meget som muligt, skrev Nikolas Lyhne-Knudsen til Journalisten.\n\nLyhne-Knudsen skrev yderligere, at man vurderer, at afskaffelse af den l\u00f8nnede spisepause vil kunne frigive 45 millioner kroner \u00e5rligt.",
    ]

    summary_true = [
        "DR Nyheders medarbejder strejker, fordi den l\u00f8nnede frokost er blevet afskaffet. Tillidsmand frygter fyringer.",
        "DR Nyheders medarbejder strejker, fordi den l\u00f8nnede frokost er blevet afskaffet. Tillidsmand frygter fyringer.",
        "DR Nyheders medarbejder strejker, fordi den l\u00f8nnede frokost er blevet afskaffet. Tillidsmand frygter fyringer.",
        "DR Nyheders medarbejder strejker, fordi den l\u00f8nnede frokost er blevet afskaffet. Tillidsmand frygter fyringer.",
    ]

    model.train(texts, summary_true)
    a = 1
