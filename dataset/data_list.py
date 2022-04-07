import numpy as np
from PIL import Image
import copy
from .augmentations import RandAugment

def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def default_loader(path):
    return pil_loader(path)


class ImageList(object):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, image_list, labels=None, transform=None, target_transform=None,
                 loader=default_loader, rand_aug=False):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise Exception

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.labels = [label for (_, label) in imgs]
        self.rand_aug = rand_aug
        if self.rand_aug:
            self.rand_aug_transform = copy.deepcopy(self.transform)
            self.rand_aug_transform.transforms.insert(0, RandAugment(1, 2.0))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img_ = self.loader(path)
        if self.transform is not None:
            img = self.transform(img_)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.rand_aug:
            rand_img = self.rand_aug_transform(img_)
            return img, target, index, rand_img
        else:
            return img, target, index

    def __len__(self):
        return len(self.imgs)
