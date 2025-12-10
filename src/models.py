from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, CLIPModel, CLIPImageProcessor, VisualBertModel, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
import pandas as pd
from pathlib import Path
import os
from util import get_root_dir, get_labels
import traceback
import numpy as np
from sklearn.model_selection import train_test_split

RANDOM_SEED_FUCKING_THING = 69
BATCH_SIZE = 32

# ImageFile.LOAD_TRUNCATED_IMAGES = True #for some reason, every image in the memotion dataset is corrupted so i need to do this otherwise it won't work
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") # for text tokenizing
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6).to(DEVICE) # for text sentiment anal

processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32") # for image processing
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE) # for imbeddings

vb_model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre").to(DEVICE)

# for img2text
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)

NUM_PATCHES = None
TEXT_SEQUENCE_LENGTH = None
IMAGES_LIST = None
# these "constants" ^ are set on a run of get_embeddings

SENTIMENT_KEY = {
    "sadness":0,
    "joy":1,
    "love":2,
    "anger":3,
    "fear":4,
    "surprise":5
}

class VBDataset(torch.utils.data.Dataset):
    def __init__(self, img_embeds, text_inputs):
        self.img_embeds = img_embeds
        self.text_inputs = text_inputs

    def __len__(self):
        return self.img_embeds.shape[0]

    def __getitem__(self, idx):
        return {
            "img": self.img_embeds[idx],
            "input_ids": self.text_inputs["input_ids"][idx],
            "attention_mask": self.text_inputs["attention_mask"][idx],
        }
    
def train_bert(emotions_path:Path, overwrite=False):
    print(DEVICE)
    model_path = get_root_dir() / 'modelaudaonadum'
    if overwrite == False: 
        if (model_path / 'the_trained_bert_weights.pt').exists():
            bert_model.load_state_dict(torch.load(model_path / 'the_trained_bert_weights.pt'))
            bert_model.to(DEVICE)
            return
    df = pd.read_csv(emotions_path)
    train_df, eval_df = train_test_split(df, test_size=0.1, shuffle=True, random_state=RANDOM_SEED_FUCKING_THING)

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
    
    training_args = TrainingArguments(
        output_dir=str(model_path/'results'),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=str(model_path/'logs'),
        logging_steps=10,
        eval_strategy="epoch",
        seed=RANDOM_SEED_FUCKING_THING
    )
    trainer = Trainer(
        model=bert_model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval
    )
    print("starting training")
    trainer.train()

    print("saving weights")
    torch.save(bert_model.state_dict(), model_path/'the_trained_bert_weights.pt')
    return

def get_bert_sentiment(generated=False, overwrite=False, test_cap=0):
    ''' if generated=False, will only sentiment analysis on captions.
    if generated=True, will sentiment analysis on generated text descriptions from BLIP as well
    don't run with generated=True unless BLIP text descriptions are in big_data.csv or i'll be very very mad >:(
    this also assumes bert was trained (i.e. train_bert was run before)
    if you run this without training bert i'm actually going to kill you (i'm obv not going to actually kill you i'm just a comment in code)
    '''
    data_path = get_root_dir() / 'data'
    big_data = pd.read_csv(data_path/'big_data.csv', keep_default_na=False)
    if overwrite == False:
        if "caption_sentiment" in big_data:
            return big_data["caption_sentiment"].values.tolist()
    captions = list(big_data['caption'])
    if test_cap > 0:
        total_len = len(captions)
        captions = captions[:test_cap]
    
    predicted_labels_all = []

    bert_model.eval()
    for i in range(0, len(captions), BATCH_SIZE):
        captions_batch = captions[i : i + BATCH_SIZE]
        inputs = tokenizer(captions_batch, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = bert_model(**inputs).logits
        
        # get the label ID (0-5)
        predicted_labels = torch.argmax(logits, dim=1).cpu().tolist()
        predicted_labels = [list(SENTIMENT_KEY.keys())[i] for i in predicted_labels] # so we can read it easier
        predicted_labels_all += predicted_labels
    if test_cap > 0:
        predicted_labels_all += [np.nan for i in range(total_len - test_cap)]
    big_data['caption_sentiment'] = predicted_labels_all

    if generated:
        predicted_description_labels_all = []
        descriptions = list(big_data['description'])
        if test_cap > 0:
            descriptions = descriptions[:test_cap]
        
        for i in range(0, len(descriptions), BATCH_SIZE):
            descriptions_batch = descriptions[i: i+ BATCH_SIZE]
            inputs = tokenizer(descriptions_batch, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                logits = bert_model(**inputs).logits
            predicted_description_labels = torch.argmax(logits, dim=1).tolist()
            predicted_description_labels = [list(SENTIMENT_KEY.keys())[i] for i in predicted_description_labels] # so we can read it easier
            predicted_description_labels_all += predicted_description_labels
        if test_cap > 0:
            predicted_description_labels_all += [np.nan for i in range(total_len - test_cap)]
        big_data['description_sentiment'] = predicted_description_labels_all

    big_data.to_csv(data_path/"big_data.csv", index=False)
    return predicted_labels_all


def get_embeddings(img_path:Path, labels_path:Path, img_only=False, overwrite=False, test_cap=0):
    ''' Saves embeddings as image_embeddings.pt and text input as text_inputs.pt
    will retrieve an already generated + saved .pt (if it exists) if overwrite = False, otherwise will generate a new .pt
    if test_cap > 0 , will only iterate thru that many files maximum (for testing purposes).
    img_only is for uhhhhh just generating the IMAGES_LIST constant mainly.
    Returns the image embeds and the text inputs
    '''
    global NUM_PATCHES, TEXT_SEQUENCE_LENGTH, IMAGES_LIST
    data_path = get_root_dir() / 'data'
    if overwrite == False: # if not forcing an overwrite, grab existing files
        if (data_path / 'image_embeddings.pt').exists() and (data_path / 'text_inputs.pt').exists(): # ...if they exist
            img_embeds = torch.load(data_path / 'image_embeddings.pt')
            text_in = torch.load(data_path / 'text_inputs.pt')
            # set the constants we need for later
            _, NUM_PATCHES, _ = img_embeds.shape
            _, TEXT_SEQUENCE_LENGTH = text_in['input_ids'].shape
            return img_embeds, text_in
    big_data = {"img_name":[], "caption":[]} # we need the image names in order so we know which embeds are for which image
    imgs = []
    labels = pd.read_csv(labels_path)
    i = 0 # for test_cap, for testing purposes
    total_images = len([file for file in os.listdir(img_path)]) if test_cap <= 0 else test_cap
    progress_check = 0.05*total_images # i want a progress check every 5% so i know it's not frozen
    last_prog = 0
    for file in img_path.iterdir(): # iterate over all files in image directory
        if test_cap > 0 and i >= test_cap:
            break # for testing purposes
        i += 1
        # get a file, add it to imgs, get its text and add it to text
        try:
            img = Image.open(file).convert('RGB')
            img.load()
            imgs.append(img)
            #print("image appended")
            txt = labels.loc[labels['image_name'] == file.name]['text_corrected'].iloc[0]
            if type(txt) != str:
                print(f"Found non-string \"{str(txt)}\" when searching for {file.name}, converting to string")
                if np.isnan(txt):
                    txt = "" 
                else:
                    txt = str(txt)
            big_data["caption"].append(txt)
            big_data["img_name"].append(file.name)
        except Exception as e:
            print(f"had an error boyo, with {file}, {e}")
            print(traceback.format_exc())
        
        if i//progress_check > last_prog: # for my cheeky little progress bar
            print(f'{i*100.0/total_images:.2f}% ({i}/{total_images})')
            last_prog = i//progress_check
        
    pd.DataFrame(big_data).to_csv(data_path/"big_data.csv", index=False)
    IMAGES_LIST = imgs
    print("DONE")
    print(len(imgs))
    print(len(big_data["caption"]))

    if not img_only:
        img_in = processor(images=imgs, return_tensors='pt').to(DEVICE)
        text_in = tokenizer(text=big_data["caption"], padding="max_length", max_length=512, truncation=True, return_tensors='pt').to(DEVICE)
        text_in = {k: v for k, v in text_in.items()}
        img_embeds = None
        with torch.no_grad(): # save computation power and memory
            # get em
            img_embeds = clip_model.vision_model(**img_in).last_hidden_state
        if img_embeds == None: raise Exception('embeds not correctly generated')
        # set constants to use for visualbert
        _, NUM_PATCHES, _ = img_embeds.shape
        _, TEXT_SEQUENCE_LENGTH = text_in['input_ids'].shape
        # save em
        torch.save(img_embeds, data_path/"image_embeddings.pt")
        torch.save(text_in, data_path/"text_inputs.pt")
        # return em
        return img_embeds, text_in
    else:
        return

def feed_VisualBERT(img_embeds, text_inputs, p, overwrite = False):
    proj = torch.nn.Linear(768, 2048).to(DEVICE)
    projected_img_embeds = proj(img_embeds)
    data_path = get_root_dir() / 'data'
    if overwrite == False:
        if (data_path / f'visualbert_output_{p}.pt').exists(): 
            return torch.load(data_path / f'visualbert_output_{p}.pt')

    dataset = VBDataset(projected_img_embeds, text_inputs)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    vb_model.to(DEVICE)
    vb_model.eval()

    i = 0
    total_batches = len(loader)

    all_outputs = []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["img"].to(DEVICE)
            B = imgs.shape[0]

            token_type_ids = torch.zeros((B, TEXT_SEQUENCE_LENGTH), dtype=torch.long).to(DEVICE)
            visual_token_type_ids = torch.ones((B, NUM_PATCHES), dtype=torch.long).to(DEVICE)
            visual_attention_mask = torch.ones((B, NUM_PATCHES), dtype=torch.long).to(DEVICE)

            outputs = vb_model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                token_type_ids=token_type_ids,

                visual_embeds=imgs,
                visual_attention_mask=visual_attention_mask,
                visual_token_type_ids=visual_token_type_ids
            )
            cls_embeddings = outputs.last_hidden_state[:, 0]  # [B, hidden]
            all_outputs.append(cls_embeddings.cpu())

            print(f'{i*100.0/total_batches:.2f}% ({i}/{total_batches})')
            i += 1

    final_output = torch.cat(all_outputs, dim=0)  # [6991, hidden]
    torch.save(final_output, data_path / 'visualbert_output.pt')

    return final_output

def feed_BLIP(img_path, overwrite=False, test_cap=0):
    global IMAGES_LIST
    '''
    Docstring for feed_BLIP
    
    :param img_path: the path with all the images
    :type img_path: Path
    :param overwrite: forces new files to be made even. if false, will retrieve already existing files (if they exist)
    :param test_cap: will process all images in path unless specified to end early (mainly for testing purposes)
    '''
    print("starting BLIP")
    data_path = get_root_dir() / 'data'
    if overwrite == False: # if not forcing an overwrite, grab existing files
        if (data_path / 'big_data.csv').exists():
            text_descriptions = pd.read_csv(data_path / 'big_data.csv')
            return text_descriptions
        
    if IMAGES_LIST == None:
        get_embeddings(img_path, get_labels(), img_only=True, overwrite=True)
    print("hello")
    big_data = pd.read_csv(data_path/'big_data.csv')
    text_descriptions = []
    prompt_text = "A photo of"
    forbidden_phrases = [
        "with text that says",
        "with text that reads",
        "with the capt that reads",
        "with the capt that says",
        "and the words",
        "with the words"
    ]
    bad_words_ids = [blip_processor.tokenizer.encode(p, add_special_tokens=False) for p in forbidden_phrases]
    total = len(IMAGES_LIST)

    for i in range(0, total, BATCH_SIZE):
        if test_cap > 0 and i >= test_cap:
            break # for testing purposes
        batch_images = IMAGES_LIST[i : i + BATCH_SIZE]
        batch_prompts = [prompt_text] * len(batch_images)
        blip_in = blip_processor(images=batch_images, text=batch_prompts, return_tensors="pt", padding=True).to(DEVICE)
        blip_out = blip_model.generate(**blip_in, max_new_tokens=50,num_beams=5,no_repeat_ngram_size=2,repetition_penalty=1.5,early_stopping=True, bad_words_ids=bad_words_ids)
        captions = [blip_processor.decode(out, skip_special_tokens=True) for out in blip_out]
        text_descriptions += captions
        print(f'{i*100.0/total:.2f}% ({i}/{total})')
    big_data['description'] = text_descriptions
    pd.DataFrame(big_data).to_csv(data_path/"big_data.csv", index=False)
    
    return text_descriptions
