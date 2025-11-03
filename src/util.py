import kagglehub as kh
import os
from pathlib import Path

ROOT_DIR, LABELS_PATH, IMAGES_FOLDER_PATH = None, None, None

def get_root_dir():
    global ROOT_DIR
    if os.path.basename(Path.cwd()) != 'src':
        ROOT_DIR = Path.cwd()
    elif os.path.basename(Path.cwd()) == 'src':
        ROOT_DIR = Path.cwd().parent
    return ROOT_DIR

def import_dataset():
    global LABELS_PATH, IMAGES_FOLDER_PATH
    LABELS_PATH = kh.dataset_download(handle='williamscott701/memotion-dataset-7k', path='memotion_dataset_7k/labels.csv')
    IMAGES_FOLDER_PATH = kh.dataset_download(handle='williamscott701/memotion-dataset-7k', path='memotion_dataset_7k/images')
    return LABELS_PATH, IMAGES_FOLDER_PATH