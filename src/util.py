import kagglehub as kh
import os
from pathlib import Path
import pandas as pd
import numpy as np
import math
import torch
import shutil
import torch.nn.functional as F


ROOT_DIR, LABELS_PATH, IMAGES_FOLDER_PATH, EMOTIONS_PATH = None, None, None, None

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

def get_emotions(force=False):
    if EMOTIONS_PATH == None:
        import_emotions(force)
    return EMOTIONS_PATH

def import_images_labels(force):
    global LABELS_PATH, IMAGES_FOLDER_PATH

    p = Path(kh.dataset_download(handle='williamscott701/memotion-dataset-7k', force_download=force))/'memotion_dataset_7k'
    LABELS_PATH = p/'labels.csv'
    IMAGES_FOLDER_PATH = p/'images'
    return LABELS_PATH, IMAGES_FOLDER_PATH

def import_emotions(force):
    global EMOTIONS_PATH
    p = Path(kh.dataset_download(handle='nelgiriyewithana/emotions', force_download=force))
    EMOTIONS_PATH = p/'text.csv'
    return EMOTIONS_PATH

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
    print('starting cosine sim')
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
        print(row['description'])
    for val in (0.995, 0.99, 0.98, 0.97, 0.96, 0.95, 0.9):
        what_are_we_doing = big_data[big_data['cosine_sim'] >= val]
        print(f'There are {what_are_we_doing.shape[0]} images with cosine_sim >= {val}')
        print(f'{what_are_we_doing["img_name"].iloc[-1]} with a score of {what_are_we_doing["cosine_sim"].iloc[-1]}')


# this is from gemini
def cluster_and_copy(seed_img, all_embeds, threshold=0.95):
    """
    Finds a 'chain' of similar images starting from seed_img_name.
    """
    # 1. Setup paths and data
    print(f"--- Starting Cluster Search for {seed_img} (Threshold: {threshold}) ---")
    
    # Load the mapping (Index -> Filename)
    # Assuming big_data was generated in the same order as all_embeds
    big_data = pd.read_csv(get_root_dir()/"data"/"big_data.csv")
    
    # Locate the seed index
    if seed_img not in big_data["img_name"].values:
        raise ValueError(f"Image {seed_img} not found in big_data.csv")
        
    start_index = big_data.loc[big_data["img_name"] == seed_img].index[0]
    
    # Ensure directory exists
    source_dir = get_images() # From your util.py
    dest_path = get_root_dir() / 'data' / f'cluster_{seed_img.split('.')[0]}_{threshold}'
    os.makedirs(dest_path, exist_ok=True)

    # 2. Prepare Embeddings (Normalize for faster cosine sim)
    # Cosine Sim is just Dot Product if vectors are normalized
    # We move to GPU for speed if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_embeds = all_embeds.to(device)
    
    # Normalize all embeddings once (Magnitude = 1)
    # This makes cosine_similarity(a, b) == a @ b.T
    all_embeds_norm = F.normalize(all_embeds, p=2, dim=1)

    # 3. BFS Algorithm (The "Chain" Search)
    queue = [start_index]  # Indices we need to process
    visited = {start_index} # Indices we have already added to the cluster
    
    print("Building cluster chain...")
    
    while len(queue) > 0:
        current_idx = queue.pop(0)
        
        # Get the embedding of the current image
        current_vec = all_embeds_norm[current_idx].unsqueeze(0) # Shape [1, 768]
        
        # VECTORIZED CALCULATION (Instant vs. Loop)
        # Calculate similarity against ALL other images at once
        # Result shape: [Total_Images]
        sims = torch.mm(current_vec, all_embeds_norm.T).squeeze(0)
        
        # Find indices where sim > threshold
        # (This returns a tensor of indices)
        found_indices = torch.nonzero(sims > threshold).flatten().tolist()
        
        # Process new finds
        for idx in found_indices:
            if idx not in visited:
                visited.add(idx)
                queue.append(idx) # Add to queue to search ITS neighbors next
                
                # Optional: Print progress
                # img_name = big_data.iloc[idx]['img_name']
                # print(f"-> Added {img_name} (linked by similarity {sims[idx]:.4f})")

    # 4. Copy Files
    print(f"\nFound {len(visited)} images in this cluster. Copying files...")
    
    count = 0
    for idx in visited:
        # Look up name
        file_name = big_data.iloc[idx]['img_name']
        
        # Construct paths
        src_file = source_dir / file_name
        dst_file = dest_path / file_name
        
        try:
            shutil.copy2(src_file, dst_file)
            count += 1
        except FileNotFoundError:
            print(f"Warning: Could not find source file {file_name}")

    print(f"Successfully copied {count} images to {dest_path}")
    return list(visited)