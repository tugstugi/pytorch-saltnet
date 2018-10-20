"""Dataset class for the Kaggle Salt Identification Challenge."""

__author__ = 'Erdene-Ochir Tuguldur, Yuan Xu'

import os
import copy
import pandas as pd
import skimage.io
from tqdm import tqdm

from torch.utils.data import Dataset


def get_test_image_ids(name):
    script_dir_path = os.path.dirname(os.path.realpath(__file__))
    dataset_dir_path = os.path.join(script_dir_path, name)
    all_files = os.listdir(os.path.join(dataset_dir_path, 'images'))
    image_ids = []
    for file_name in all_files:
        if file_name.endswith(".png"):
            image_ids.append(file_name[:-4])
    return image_ids


def get_train_image_ids(name):
    script_dir_path = os.path.dirname(os.path.realpath(__file__))
    csv_dir_path = script_dir_path if name == 'train' else os.path.join(script_dir_path, 'folds')
    df = pd.read_csv(os.path.join(csv_dir_path, '%s.csv' % name))
    image_ids = []
    for row in df.itertuples():
        image_ids.append(row[1])
    return image_ids


def load_data(name, mode, preload, mask_threshold):
    script_dir_path = os.path.dirname(os.path.realpath(__file__))
    dataset_dir_path = os.path.join(script_dir_path, mode)

    all_data = []
    image_ids = get_train_image_ids(name) if mode == 'train' else get_test_image_ids(name)
    pbar = tqdm(image_ids, desc="Load dataset " + name, total=len(image_ids), unit="images")
    for image_id in pbar:
        data = {}
        data['image_id'] = image_id
        data['dataset_dir'] = dataset_dir_path
        if preload:
            data = load_images_and_masks(data, mode, mask_threshold)

        if data is not None:
            all_data.append(data)
    return all_data


def load_images_and_masks(data, mode, mask_threshold):
    image_id = data['image_id']
    dataset_dir = data['dataset_dir']
    image = skimage.io.imread('%s/images/%s.png' % (dataset_dir, image_id))

    assert image.ndim == 3
    data['input'] = image

    if mode == 'train':
        mask = skimage.io.imread('%s/masks/%s.png' % (dataset_dir, image_id), as_gray=True)
        assert mask.ndim == 2
        mask[mask > 0] = 1
        data['mask'] = mask
        pixel_count = mask.sum()
        if 0 < pixel_count <= mask_threshold:
            return None

    return data


class SaltIdentification(Dataset):

    def __init__(self, mode='train', transform=None, preload=False, name=None, data=None, mask_threshold=0):
        Dataset.__init__(self)
        self.mode = mode
        self.transform = transform
        if name is not None:
            self.name = name
        else:
            self.name = mode

        self.mask_threshold = mask_threshold
        if data is None:
            self.data = load_data(self.name, self.mode, preload, self.mask_threshold)
        else:
            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        if isinstance(index, slice):
            return SaltIdentification(mode=self.mode, transform=self.transform, name=self.name, data=data,
                                      mask_threshold=self.mask_threshold)
        return self.__pull_item__(data)

    def __pull_item__(self, data):
        if 'input' not in data:
            data = load_images_and_masks(data, self.mode, self.mask_threshold)
        data = copy.copy(data)
        if self.transform:
            data = self.transform(data)
        return data


if __name__ == '__main__':
    train_dataset = SaltIdentification(mode='train', name='list_valid0_400')
    print(len(train_dataset))
    assert len(train_dataset) == 399  # there is a bug, we ignore the first element from the csv file because of pandas

    train_dataset = SaltIdentification(mode='train', preload=True)
    valid_dataset = SaltIdentification(mode='train', name='valid', data=train_dataset.data)
    train_dataset = train_dataset[:-100]
    valid_dataset = valid_dataset[-100:]

    assert len(train_dataset) == 4000 - 100
    assert len(valid_dataset) == 100

    test_dataset = SaltIdentification(mode='test')
    assert len(test_dataset) == 18000
    test_dataset = SaltIdentification(mode='test', preload=True)
    assert len(test_dataset) == 18000
