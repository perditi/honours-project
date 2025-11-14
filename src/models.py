from transformers import  CLIPModel, CLIPImageProcessor, CLIPTokenizerFast, VisualBertModel
from PIL import Image, ImageFile
import torch
import pandas as pd
from pathlib import Path
import os
from util import get_root_dir
import traceback
import numpy as np

# ImageFile.LOAD_TRUNCATED_IMAGES = True #for some reason, every image in the memotion dataset is corrupted so i need to do this otherwise it won't work

tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32") # for text tokenizing
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32") # for image processing
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32") # for imbeddings

vb_model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")

def get_embeddings(img_path:Path, labels_path:Path, overwrite=False, test_cap=0):
    ''' Saves embeddings as image_embeddings.pt and text embeddings as text_embeddings.pt
    will retrieve an already generated + saved .pt (if it exists) if overwrite = False, otherwise will generate a new .pt
    if test_cap > 0 , will only iterate thru that many files maximum (for testing purposes)
    Returns the embeddings
    '''
    data_path = get_root_dir() / 'data'
    if overwrite == False: # if not forcing an overwrite, grab existing files
        if (data_path / 'image_embeddings.pt').exists() and (data_path / 'text_embeddings.pt').exists(): # ...if they exist
            return torch.load(data_path / 'image_embeddings.pt'), torch.load(data_path / 'text_embeddings.pt')
        
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
            #print("image opened")
            img.load()
            #print("image loaded")
            # 
            # img.show()
            # if not img.verify(): # why is every image corrupted
            #     print(f"Skipping corrupted image: {file}")
            #     continue
            # print("image verified")
            
            
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

    img_in = processor(images=imgs, return_tensors='pt')
    text_in = tokenizer(text=texts, padding=True, truncation=True, return_tensors='pt')

    img_embeds, text_embeds = None, None
    with torch.no_grad(): # save computation power and memory
        # get em
        img_embeds = clip_model.vision_model(**img_in).last_hidden_state
        text_embeds = clip_model.get_text_features(**text_in)
    if img_embeds == None or text_embeds == None: raise Exception('embeds not correctly generated')
    # save em
    torch.save(img_embeds, data_path/"image_embeddings.pt")
    torch.save(text_embeds, data_path/"text_embeddings.pt")
    # return em
    return img_embeds, text_embeds

def feed_VisualBERT(visual_embeds, text_embeds):
    visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.long
                                       )
    visual_token_type_ids = torch.ones_like(visual_attention_mask)
    outputs = vb_model(
        input_ids=text_embeds["input_ids"],
        attention_mask=text_embeds["attention_mask"],
        token_type_ids=text_embeds.get("token_type_ids"),
        visual_embeds=visual_embeds,
        visual_attention_mask=visual_attention_mask,
        visual_token_type_ids=visual_token_type_ids
    )

    return outputs.pooler_output 