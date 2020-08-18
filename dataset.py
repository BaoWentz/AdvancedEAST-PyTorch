import os
import numpy as np
import lmdb
import sys
import six
import time

import cfg
from PIL import Image
from natsort import natsorted
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class RawDataset(Dataset):

    def __init__(self, is_val=False):
        self.img_h, self.img_w = cfg.max_train_img_size, cfg.max_train_img_size
        if is_val:
            with open(os.path.join(cfg.data_dir, cfg.val_fname), 'r') as f_val:
                f_list = f_val.readlines()
        else:
            with open(os.path.join(cfg.data_dir, cfg.train_fname), 'r') as f_train:
                f_list = f_train.readlines()

        self.image_path_list = []
        self.labels_path_dic = {}
        for f_line in f_list:
            img_filename = str(f_line).strip().split(',')[0]
            img_path = os.path.join(cfg.data_dir, cfg.train_image_dir_name, img_filename)
            self.image_path_list.append(img_path)
            gt_file = os.path.join(cfg.data_dir, cfg.train_label_dir_name, img_filename[:-4] + '_gt.npy')
            self.labels_path_dic[img_path] = gt_file
        self.image_path_list = natsorted(self.image_path_list)
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        img_path = self.image_path_list[index]
        label = np.load(self.labels_path_dic[img_path])
        try:
            img = Image.open(img_path).convert('RGB')  # for color image

        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            img = Image.new('RGB', (self.img_w, self.img_h))
        img_tensor = transforms.ToTensor()(img)
        label = np.transpose(label, (2, 0, 1))

        return (img_tensor, label)


def data_collate(batch):
    imgs = []
    labels = []
    gt_xy_list = []  # 长度为N的列表，每个值为该图片中所有矩形框的坐标
    # 例如：[(31, 4, 2), (10, 4, 2), (47, 4, 2), (28, 4, 2)]
    for info in batch:
        imgs.append(info[0])
        labels.append(info[1])
        gt_xy_list.append(info[2])
    return torch.stack(imgs, 0), torch.tensor(np.array(labels)), gt_xy_list


class LmdbDataset(Dataset):

    def __init__(self, root):

        self.root = root
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples
            self.filtered_index_list = [index + 1 for index in range(self.nSamples)]

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key)
            label = np.fromstring(label, dtype=np.float64)
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)
            gt_xy_list_Key = 'gt_xy_list-%09d'.encode() % index
            gt_xy_list = txn.get(gt_xy_list_Key)
            gt_xy_list = np.fromstring(gt_xy_list, dtype=np.float64)
            gt_xy_list = np.reshape(gt_xy_list.astype(float), (-1, 4, 2))
            width_height = int(txn.get('width-height'.encode()))
            width, height = width_height, width_height

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('RGB')  # for color image

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                label = np.zeros((height // cfg.pixel_size, width // cfg.pixel_size, 7))
        img_tensor = transforms.ToTensor()(img)
        label = label.reshape((height // cfg.pixel_size, width // cfg.pixel_size, 7))
        label = np.transpose(label, (2, 0, 1))
        # label_tensor = transforms.ToTensor()(label)

        return (img_tensor, label, gt_xy_list)


if __name__ == '__main__':
    tick = time.time()
    train_dataset = RawDataset(is_val=False)
    data_loader_A = torch.utils.data.DataLoader(
                train_dataset, batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=int(cfg.workers),
                pin_memory=True)
    for i, (image_tensors, labels) in enumerate(data_loader_A):
        print(image_tensors.shape, labels.shape)
    tock = time.time()
    print(tock-tick)

    tick = time.time()
    train_dataset_lmdb = LmdbDataset(cfg.lmdb_trainset_dir_name)
    #  val_dataset_lmdb = LmdbDataset(cfg.lmdb_valset_dir_name)
    data_loader_B = torch.utils.data.DataLoader(
                train_dataset_lmdb, batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=int(cfg.workers),
                pin_memory=True)
    for i, (image_tensors, labels) in enumerate(data_loader_B):
        print(image_tensors.shape, labels.shape)
    tock = time.time()
    print(tock-tick)
    count = 0
    for i, (image_tensors1, labels1) in enumerate(data_loader_A):
        for img1, label1 in zip(image_tensors1, labels1):
            for j, (image_tensors2, labels2) in enumerate(data_loader_B):
                for img2, label2 in zip(image_tensors2, labels2):
                    if img1.equal(img2):
                        print(count, '--', label1.equal(label2))
                        count += 1
                #  print(image_tensors.shape, labels.shape)
