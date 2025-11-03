from transformers import  CLIPModel, CLIPProcessor, CLIPTokenizerFast, VisualBertModel
from PIL import Image
import torch
import pandas as pd
from pathlib import Path
from util import get_root_dir

tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32") # using clip for text tokenizing
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32") # using clip for image embeddings
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

vb_model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")

def get_embeddings(img_path:Path, labels_path:Path, overwrite=False, test_cap=0):
    ''' Saves embeddings as image_embeddings.pt and text embeddings as text_embeddings.pt
    will retrieve an already generated + saved .pt (if it exists) if overwrite = False, otherwise will generate a new .pt
    if test_cap > 0 , will only iterate thru that many files maximum (for testing purposes)
    Returns the embeddings as a dictionary
    '''
    data_path = get_root_dir() / 'data'
    if overwrite == False:
        if (data_path / 'image_embeddings.pt').exists() and (data_path / 'text_embeddings.pt').exists():
            return torch.load(data_path / 'image_embeddings.pt'), torch.load(data_path / 'text_embeddings.pt')
        
    imgs = []
    text = []
    labels = pd.read_csv(labels_path)
    i = 0 # for test_cap, for testing purposes
    for file in img_path.iterdir(): # iterate over all files in image directory
        if test_cap > 0 and i >= test_cap:
            break # for testing purposes

        # get a file, add it to imgs, get its text and add it to text
        try:
            img = Image.open(file)
            imgs.append(img)
            text.append(labels.loc[labels['image_name'] == file.name]['text_corrected'].iloc[0])
        except:
            print("had an error boyo")


        i += 1
    print(imgs)
    print(text)
    # out of loop, actually  do the processing with the img and text separate
    # torch.save(img_embs, data_path/"image_embeddings.pt")


