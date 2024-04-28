import os
import glob
from pathlib import Path
from random import choice, randint

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy import signal
from scipy.io import wavfile
from PIL import Image, ImageDraw, ImageFont
from transliterate import translit
from tqdm import tqdm


class Synth90kDataset(Dataset):
    CHARS = '0123456789абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, 
                 root_dir=None, 
                 mode=None, 
                 paths=None,  
                 img_height=32, 
                 img_width=100,
                 images_count_gen=10_000,
                 font_path='misc/fonts/Times New Roman.ttf',
                 rewrite=False):
        if root_dir and self._is_dataset_empty(Path(root_dir)) or rewrite:
            self.font_path = font_path
            self._gen_images(images_count_gen, root_path=Path(root_dir), rewrite=rewrite)
        
        if root_dir and mode and not paths:
            paths, texts = self._load_from_raw_files(root_dir, mode)
        elif not root_dir and not mode and paths:
            texts = None

        self.paths = paths
        self.texts = texts
        self.img_height = img_height
        self.img_width = img_width

    def _load_from_raw_files(self, root_dir, mode):
        mapping = {}
        with open(os.path.join(root_dir, 'lexicon.txt'), 'r') as fr:
            for i, line in enumerate(fr.readlines()):
                mapping[i] = line.strip()

        paths_file = None
        if mode == 'train':
            paths_file = 'annotation_train.txt'
        elif mode == 'dev':
            paths_file = 'annotation_val.txt'
        elif mode == 'test':
            paths_file = 'annotation_test.txt'

        paths = []
        texts = []
        with open(os.path.join(root_dir, paths_file), 'r') as fr:
            for line in fr.readlines():
                path, index_str = line.strip().split(' ')
                path = os.path.join(root_dir, path)
                index = int(index_str)
                text = mapping[index]
                paths.append(path)
                texts.append(text)
        return paths, texts

    @staticmethod
    def _is_dataset_empty(dataset_path: Path):
        return not dataset_path.exists() or not any(dataset_path.iterdir())
    
    @staticmethod
    def _clear_dataset_folder(dataset_path: Path):
        for folder in dataset_path.iterdir():
            if not folder.is_dir():
                folder.unlink()
                continue
            for image in folder.iterdir():
                image.unlink()
            folder.rmdir()
    
    def _gen_images(self,
                    cnt: int, 
                    root_path: Path, 
                    font_size: int = 28, 
                    text_min_len: int = 4,
                    text_max_len: int = 16,
                    text_color: str = 'black',
                    background_color: str = 'white',
                    rewrite=True):
        if rewrite:
            self._clear_dataset_folder(root_path)
        
        lexicon = root_path.joinpath('lexicon.txt').open(mode='w')
        annotation_train = root_path.joinpath('annotation_train.txt').open('w')
        annotation_val = root_path.joinpath('annotation_val.txt').open('w')
        annotation_test = root_path.joinpath('annotation_test.txt').open('w')
        subfolder = {"path": None, "No": 0}
        cnt_subfolder_images = 500 if cnt >= 10_000 else 250
        for i in tqdm(range(cnt), desc='create images', unit='image'):
            if i % cnt_subfolder_images == 0:
                subfolder["No"] += 1
                subfolder["path"] = root_path.joinpath(str(subfolder["No"]))
                subfolder["path"].mkdir(exist_ok=True, parents=True)
        
            # sample text and font
            text = ''.join(choice(self.CHARS) for _ in range(randint(text_min_len, text_max_len)))
            translit_text = translit(text, language_code='ru', reversed=True)
            font = ImageFont.truetype(self.font_path, font_size, encoding="unic")

            # get the line size
            _, _, text_width, text_height = font.getbbox(text)

            # create a blank canvas with extra space between lines
            canvas = Image.new('RGB', (text_width + 10, text_height + 10), background_color)

            # draw the text onto the text canvas, and use blue as the text color
            draw = ImageDraw.Draw(canvas)
            draw.text((5,5), text, text_color, font)

            # save the blank canvas to a file
            image_subNo = i % cnt_subfolder_images + 1
            image_name = f"{image_subNo}_{translit_text}_{i}.png"
            image_save_path = subfolder["path"].joinpath(image_name)
            canvas.save(image_save_path, "PNG")
            lexicon.write(text + '\n')
            annotation_text = f'{subfolder["No"]}/{image_name} {i}\n'
            annotation_train.write(annotation_text)
            if i > int(cnt * 0.75):
                annotation_val.write(annotation_text)
                annotation_test.write(annotation_text)
        lexicon.close()
        annotation_train.close()
        annotation_val.close()
        annotation_test.close()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        try:
            image = Image.open(path).convert('L')  # grey-scale
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
        image = np.array(image)
        image = image.reshape((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0

        image = torch.FloatTensor(image)
        if self.texts:
            text = self.texts[index]
            target = [self.CHAR2LABEL[c] for c in text]
            target_length = [len(target)]

            target = torch.LongTensor(target)
            target_length = torch.LongTensor(target_length)
            return image, target, target_length
        else:
            return image


def synth90k_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths
