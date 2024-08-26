from multiprocessing import freeze_support
import datasets
import evaluate
import fire
import numpy as np
import os
import pandas as pd
import torch
import transformers
import wandb
import random
from compute_metrics import compute_metrics
from eval import strip_gloss_punctuation
"""Defines models and functions for loading, manipulating, and writing task data"""
from typing import Optional, List
import re
from transformers import AutoModel, T5ForConditionalGeneration

class IGTLine:
    """A single line of IGT"""
    def __init__(self, transcription: str, segmentation: Optional[str], glosses: Optional[str], translation: Optional[str]):
        self.transcription = transcription
        self.segmentation = segmentation
        self.glosses = glosses
        self.translation = translation
        self.should_segment = True

    def __repr__(self):
        return f"Trnsc:\t{self.transcription}\nSegm:\t{self.segmentation}\nGloss:\t{self.glosses}\nTrnsl:\t{self.translation}\n\n"

    def gloss_list(self, segmented=False) -> Optional[List[str]]:
        """Returns the gloss line of the IGT as a list.
        :param segmented: Icf True, will return each morpheme gloss as a separate item.
        """
        if self.glosses is None:
            return []
        if not segmented:
            return self.glosses.split()
        else:
            return re.split("\s|-", self.glosses)

    def __dict__(self):
        d = {'transcription': self.transcription, 'translation': self.translation}
        if self.glosses is not None:
            d['glosses'] = self.gloss_list(segmented=self.should_segment)
        if self.segmentation is not None:
            d['segmentation'] = self.segmentation
        return d

    
def load_data_file(path: str):
    """Loads a file containing IGT data into a list of entries."""
    all_data = []

    # If we have a directory, recursively load all files and concat together
    if os.path.isdir(path):
        for file in os.listdir(path):
            if file.endswith(".txt"):
                print(file)
                all_data.extend(load_data_file(os.path.join(path, file)))
        return all_data

    # If we have one file, read in line by line
    with open(path, 'r') as file:
        current_entry = [None, None, None, None]  # transc, segm, gloss, transl

        skipped_lines = []
        
        for line in file:
            # Determine the type of line
            # If we see a type that has already been filled for the current entry, something is wrong
            line_prefix = line[:2]
            if line_prefix == '\\t' and current_entry[0] == None:
                current_entry[0] = line[3:].strip()
            elif line_prefix == '\\m' and current_entry[1] == None:
                current_entry[1] = line[3:].strip()
            elif line_prefix == '\\g' and current_entry[2] == None:
                if len(line[3:].strip()) > 0:
                    current_entry[2] = line[3:].strip()
            elif line_prefix == '\\l' and current_entry[3] == None:
                current_entry[3] = line[3:].strip()
                # Once we have the translation, we've reached the end and can save this entry
                all_data.append(IGTLine(transcription=current_entry[0],
                                        segmentation=current_entry[1],
                                        glosses=current_entry[2],
                                        translation=current_entry[3]))
                current_entry = [None, None, None, None]
            elif line_prefix == "\\p":
                # Skip POS lines
                continue
            elif line.strip() != "":
                # Something went wrong
                skipped_lines.append(line)
                continue
            else:
                if not current_entry == [None, None, None, None]:
                    all_data.append(IGTLine(transcription=current_entry[0],
                                            segmentation=current_entry[1],
                                            glosses=current_entry[2],
                                            translation=None))
                    current_entry = [None, None, None, None]
        # Might have one extra line at the end
        if not current_entry == [None, None, None, None]:
            all_data.append({"transcr"})
            all_data.append(IGTLine(transcription=current_entry[0],
                                    segmentation=current_entry[1],
                                    glosses=current_entry[2],
                                    translation=None))
        if len(skipped_lines) == 0:
            print("Looks good")
        else:
            print(f"Skipped {len(skipped_lines)} lines")
            print(skipped_lines)
    return all_data
        
        
def create_hf_dataset(filename, glottocode, metalang, row_id='st'):
    print(f"Loading {filename}")
    raw_data = load_data_file(filename)
    data = []
    for i, line in enumerate(raw_data):
        new_row = {'glottocode': glottocode, 'metalang_glottocode': metalang, "is_segmented": "yes", "source": "sigmorphon_st", "type": "canonical"}
        new_row['ID'] = f"{row_id}_{glottocode}_{i}"
        new_row['transcription'] = line.segmentation
        new_row['glosses'] = line.glosses
        new_row['translation'] = line.translation
        data.append(new_row)

        new_row_unsegmented = {'glottocode': glottocode, 'metalang_glottocode': metalang, "is_segmented": "no", "source": "sigmorphon_st", "type": "canonical"}
        new_row_unsegmented['ID'] = f"{row_id}_{glottocode}_{i}"
        new_row_unsegmented['transcription'] = line.transcription
        new_row_unsegmented['glosses'] = line.glosses
        new_row_unsegmented['translation'] = line.translation
        data.append(new_row_unsegmented)

    return datasets.Dataset.from_list(data)

def create_prompt(
    row,
    use_translation: bool = True,
):
    """Processing function for rows in the dataset, creates an input prompt from the fields in the row."""
    transcription = ' '.join((row['transcription']).split())
    glosses = ' '.join((row['glosses']).split())
    lang = 'an unknown language' if row['glottocode'] == '' else row['glottocode']
    is_segmented = 'unknown' if row['is_segmented'] == '' else row['is_segmented']
    prompt = f"""Provide the glosses for the following transcription in {lang}.

Transcription in {lang}: {transcription}
Transcription segmented: {is_segmented}
"""
    if row['translation'] is not None and use_translation:
        if len(row['translation'].strip()) > 0:
            translation = ' '.join((row['translation']).split())
            prompt += f"Translation in {row['metalang']}: {translation}\n"

    prompt += 'Glosses: '

    row['prompt'] = prompt
    row['glosses'] = glosses
    return row


def tokenize(tokenizer: transformers.ByT5Tokenizer, max_length: int):
    def _tokenize(batch):
        nonlocal tokenizer, max_length

        if "glosses" in batch:
            targets = batch["glosses"]
        else:
            targets = None

        model_inputs = tokenizer(
            batch["prompt"],
            text_target=targets,
            truncation=True,
            padding=False,
            max_length=max_length,
        )
        return model_inputs

    return _tokenize

def main(exp_name: str, data_file: str, glottocode: str, metalang:str, use_translation: bool = False):
    MODEL_INPUT_LENGTH = 1024
    dataset = create_hf_dataset(data_file, glottocode, metalang)
    tokenizer = transformers.ByT5Tokenizer.from_pretrained(
        "google/byt5-base", use_fast=False
    )
    dataset = dataset.filter(lambda x: x["transcription"] is not None and x["glosses"] is not None)
    if not use_translation:
        print("excluding translations")
    dataset = dataset.map(create_prompt, fn_kwargs={"use_translation": use_translation})
    dataset = dataset.map(
        tokenize(tokenizer, max_length=MODEL_INPUT_LENGTH), batched=True
    )
    print("Creating predictions...")
    if not os.path.exists(f"../preds/{exp_name}"):
        os.makedirs(f"../preds/{exp_name}")
        pred_path = f"../preds/{exp_name}/preds.csv"
    model = T5ForConditionalGeneration.from_pretrained("lecslab/glosslm")
    training_args =  transformers.Seq2SeqTrainingArguments(
        output_dir="./tmp",
        evaluation_strategy="epoch",
        per_device_eval_batch_size=1,
        eval_accumulation_steps=10,
        gradient_accumulation_steps=64,
        # gradient_checkpointing=True,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=3,
        predict_with_generate=True,
        logging_steps=100,
        generation_max_length=1024,
        generation_num_beams=3,
        report_to="wandb",
        metric_for_best_model="chrf++",
        fp16=True,
        dataloader_num_workers=4,
        # use_cpu=True
    )
    trainer = transformers.Seq2SeqTrainer(
        model=model,  # The instantiated 🤗 Transformers model to be trained
        args=training_args,  # Training arguments, defined above
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id
        ) 
    )

# Step 4: Batched prediction
    preds = trainer.predict(dataset)
    labels = np.where(preds.predictions != -100, preds.predictions, tokenizer.pad_token_id)
    preds = tokenizer.batch_decode(labels, skip_special_tokens=True)
    preds = [strip_gloss_punctuation(pred) for pred in preds]

    preds_df = pd.DataFrame({
        "id": dataset["id"],
        "glottocode": dataset["glottocode"],
        "is_segmented": dataset["is_segmented"],
        "pred": preds,
    })

    preds_df.to_csv(pred_path, index=False)

    def clean_preds(preds):
        corrected_preds = preds.replace('\.$', '', regex=True)
        corrected_preds = corrected_preds.replace('\,', '', regex=True)
        corrected_preds = corrected_preds.replace('»', '', regex=True)
        corrected_preds = corrected_preds.replace('«', '', regex=True)
        corrected_preds = corrected_preds.replace('\"', '', regex=True)
        corrected_preds = corrected_preds.replace('\. ', ' ', regex=True)
        corrected_preds = corrected_preds.replace('\.\.+', '', regex=True)
        corrected_preds = corrected_preds.replace('\ +', ' ', regex=True)
        return corrected_preds

    preds_df['pred'] = clean_preds(preds_df['pred'])
    preds_df.to_csv(pred_path[:-4] + '.postprocessed.csv', index=False)

    print(f"Predictions for data saved to {pred_path}")

# main(exp_name="test", data_file="/Users/CitronVert/Desktop/glosslm/data/Guarani Corpus/Story1.txt", glottocode="para1311", metalang="English")

if __name__ == '__main__':
    # freeze_support()
    main(exp_name="test", data_file="/Users/CitronVert/Desktop/glosslm/data/Guarani Corpus/Story1.txt", glottocode="para1311", metalang="English")
