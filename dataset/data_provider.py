from .data_list import ImageList
import torch.utils.data as util_data
from torchvision import transforms as T

def get_dataloader_from_image_filepath(images_file_path, batch_size=32, resize_size=256, is_train=True, crop_size=224,
                                       center_crop=True, rand_aug=False, random_resized_crop=False, num_workers=4):
    if images_file_path is None:
        return None

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if is_train is not True:
        transformer = T.Compose([
            T.Resize([resize_size, resize_size]),
            T.CenterCrop(crop_size),
            T.ToTensor(),
            normalize])
        images = ImageList(open(images_file_path).readlines(), transform=transformer)
        images_loader = util_data.DataLoader(images, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        if center_crop:
            transformer = T.Compose([T.Resize([resize_size, resize_size]),
                                              T.RandomHorizontalFlip(),
                                              T.CenterCrop(crop_size),
                                              T.ToTensor(),
                                              normalize])
        elif random_resized_crop:
            transformer = T.Compose([T.Resize([resize_size, resize_size]),
                                              T.RandomCrop(crop_size),
                                              T.RandomHorizontalFlip(),
                                              T.ToTensor(),
                                              normalize])
        else:
            transformer = T.Compose([T.Resize([resize_size, resize_size]),
                                     T.RandomResizedCrop(crop_size),
                                     T.RandomHorizontalFlip(),
                                     T.ToTensor(),
                                     normalize])

        images = ImageList(open(images_file_path).readlines(), transform=transformer, rand_aug=rand_aug)
        images_loader = util_data.DataLoader(images, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    return images_loader


def get_dataloaders(args):
    dataloaders = {}
    source_train_loader = get_dataloader_from_image_filepath(args.source_path, batch_size=args.batch_size,
                                           center_crop=args.center_crop, num_workers=args.num_workers,
                                           random_resized_crop=args.random_resized_crop)
    target_train_loader = get_dataloader_from_image_filepath(args.target_path, batch_size=args.batch_size,
                                           center_crop=args.center_crop, num_workers=args.num_workers,
                                           rand_aug=args.rand_aug, random_resized_crop=args.random_resized_crop)
    source_val_loader = get_dataloader_from_image_filepath(args.source_path, batch_size=args.batch_size, is_train=False,
                                                           num_workers=args.num_workers)
    target_val_loader = get_dataloader_from_image_filepath(args.target_path, batch_size=args.batch_size, is_train=False,
                                                           num_workers=args.num_workers)

    if type(args.test_path) is list:
        test_loader = []
        for tst_addr in args.test_path:
            test_loader.append(get_dataloader_from_image_filepath(tst_addr, batch_size=args.batch_size, is_train=False,
                                                         num_workers=args.num_workers))
    else:
        test_loader = get_dataloader_from_image_filepath(args.test_path, batch_size=args.batch_size, is_train=False,
                                                         num_workers=args.num_workers)
    dataloaders["source_tr"] = source_train_loader
    dataloaders["target_tr"] = target_train_loader
    dataloaders["source_val"] = source_val_loader
    dataloaders["target_val"] = target_val_loader
    dataloaders["test"] = test_loader

    return dataloaders


class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader or [])

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)