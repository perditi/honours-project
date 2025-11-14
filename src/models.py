from transformers import BertTokenizer, BertModel, CLIPModel, CLIPImageProcessor, VisualBertModel
from PIL import Image, ImageFile
import torch
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
import os
from util import get_root_dir
import traceback
import numpy as np

# ImageFile.LOAD_TRUNCATED_IMAGES = True #for some reason, every image in the memotion dataset is corrupted so i need to do this otherwise it won't work
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") # for text tokenizing
bert_model = BertModel.from_pretrained("bert-base-uncased").to(DEVICE) # for embeds

processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32") # for image processing
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE) # for imbeddings

vb_model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")

NUM_PATCHES = None
TEXT_SEQUENCE_LENGTH = None
# these "constants" ^ are set on a run of get_embeddings

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


def get_embeddings(img_path:Path, labels_path:Path, overwrite=False, test_cap=0):
    global NUM_PATCHES, TEXT_SEQUENCE_LENGTH
    ''' Saves embeddings as image_embeddings.pt and text input as text_inputs.pt
    will retrieve an already generated + saved .pt (if it exists) if overwrite = False, otherwise will generate a new .pt
    if test_cap > 0 , will only iterate thru that many files maximum (for testing purposes)
    Returns the image embeds and the text inputs
    '''
    data_path = get_root_dir() / 'data'
    if overwrite == False: # if not forcing an overwrite, grab existing files
        if (data_path / 'image_embeddings.pt').exists() and (data_path / 'text_inputs.pt').exists(): # ...if they exist
            img_embeds = torch.load(data_path / 'image_embeddings.pt')
            text_in = torch.load(data_path / 'text_inputs.pt')
            # set the constants we need for later
            _, NUM_PATCHES, _ = img_embeds.shape
            _, TEXT_SEQUENCE_LENGTH = text_in['input_ids'].shape
            return img_embeds, text_in
        
    imgs = []
    texts = []
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
            img = Image.open(file)
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
            texts.append(txt)
        except Exception as e:
            print(f"had an error boyo, with {file}, {e}")
            print(traceback.format_exc())
        
        if i//progress_check > last_prog: # for my cheeky little progress bar
            print(f'{i*100.0/total_images:.2f}% ({i}/{total_images})')
            last_prog = i//progress_check
        
    print("DONE")
    print(len(imgs))
    print(len(texts))

    img_in = processor(images=imgs, return_tensors='pt').to(DEVICE)
    text_in = tokenizer(text=texts, padding="max_length", max_length=512, truncation=True, return_tensors='pt').to(DEVICE)
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

def feed_VisualBERT(img_embeds, text_inputs, overwrite = False):
    proj = torch.nn.Linear(768, 2048).to(DEVICE)
    projected_img_embeds = proj(img_embeds)
    data_path = get_root_dir() / 'data'
    if overwrite == False:
        if (data_path / 'visualbert_output.pt').exists(): 
            return torch.load(data_path / 'visualbert_output.pt').last_hidden_state[:, 0]

    dataset = VBDataset(projected_img_embeds, text_inputs)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

   

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