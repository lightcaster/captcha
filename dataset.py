import os
import string
import numpy as np

#from skimage import io
from PIL import Image
from torch.utils.data import Dataset

class CaptchaDataset(Dataset):
    """Captcha Breaking dataset."""

    def __init__(self, root_dir, label_length=5, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.alphabet = list(string.digits + string.ascii_lowercase)
        self.alphabet_dict = dict((c,i) for i, c in enumerate(self.alphabet))

        self.label_length = label_length

        self.image_names, self.labels = self.get_files()

    def get_files(self):
        filenames, labels = [], []
        for f in os.listdir(self.root_dir):
            filenames.append(f)
            labels.append(f[:-4])

        return filenames, labels

    def encode_label(self, label):
        return np.array([self.alphabet_dict[c] for c in label], dtype=np.long)

    def decode_label(self, label):
        return [self.alphabet[int(c)] for c in label]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.image_names[idx])
        #image = io.imread(img_name)
        image = Image.open(img_name)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, self.encode_label(label)

