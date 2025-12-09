import kagglehub as kh
import os
from pathlib import Path
import pandas as pd
import math

ROOT_DIR, LABELS_PATH, IMAGES_FOLDER_PATH = None, None, None

def get_root_dir():
    global ROOT_DIR
    if ROOT_DIR == None:
        if os.path.basename(Path.cwd()) != 'src':
            ROOT_DIR = Path.cwd()
        elif os.path.basename(Path.cwd()) == 'src':
            ROOT_DIR = Path.cwd().parent
    return ROOT_DIR

def get_images(force=False):
    if IMAGES_FOLDER_PATH == None:
        import_images_labels(force)
    return IMAGES_FOLDER_PATH

def get_labels(force=False):
    if LABELS_PATH == None:
        import_images_labels(force)
    return LABELS_PATH

def import_images_labels(force):
    global LABELS_PATH, IMAGES_FOLDER_PATH

    p = Path(kh.dataset_download(handle='williamscott701/memotion-dataset-7k', force_download=force))/'memotion_dataset_7k'
    LABELS_PATH = p/'labels.csv'
    IMAGES_FOLDER_PATH = p/'images'
    return LABELS_PATH, IMAGES_FOLDER_PATH

def cosine_similarity(d, q):
    '''Input: two array-likes of the same length
    Output: the cosine similarity between the two (1 is identical, 0 is orthogonal, -1 is opposites)'''
    if len(d) != len(q):
        raise Exception(f"Both inputs for cosine similarity must be the same length, found length {len(d)} and length {len(q)}")
    return dot_product(d,q)/(magnitude(d)*magnitude(q))
    
def dot_product(a, b):
    result = 0
    for i in range(len(a)):
        result += (a[i]*b[i])
    return result

def magnitude(a):
    result = 0
    for v in a:
        result += v**2
    return math.sqrt(result)

def calc_cosine_sims(img, all_embeds):
    index = None
    print('hiiii')
    big_data = pd.read_csv(ROOT_DIR/"data"/"big_data.csv")
    if type(img) == int:
        index = img
        img = all_embeds[img]
    else:
        index = big_data.loc[big_data["img_name"]==img].index.values[0]
        img = all_embeds[index]
    print(big_data["img_name"][index])
    sim_values = []
    total = len(all_embeds)
    progress_check = 0.05*total
    last_prog = 0
    for i in range(total):
        sim_values.append(cosine_similarity(img, all_embeds[i]))
        if i//progress_check > last_prog: # for my cheeky little progress bar
            print(f'{i*100.0/total:.2f}% ({i}/{total})')
            last_prog = i//progress_check
    big_data['cosine_sim'] = sim_values
    big_data = big_data.sort_values('cosine_sim', ascending=False)
    for index, row in big_data.head(11).iterrows():
        print(f'{row['img_name']} ({row['cosine_sim']})')