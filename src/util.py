import kagglehub as kh
import os
from pathlib import Path

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