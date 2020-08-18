""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """

#  import fire
import os
import lmdb
import cv2
#  import imageio
import cfg
from PIL import Image, ImageDraw
from tqdm import tqdm

import numpy as np
from preprocess import preprocess
from label import shrink, point_inside_of_quad, point_inside_of_nth_quad


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    #  img = imageio.imread(imageBuf)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(gtFile, outputPath, checkValid=True, map_size=8589934592):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=map_size)  # 85899345920/8Gb
    cache = {}
    cnt = 1
    gtFile = os.path.join(cfg.data_dir, gtFile)

    with open(gtFile, 'r', encoding='gbk') as data:
        datalist = data.readlines()

    nSamples = len(datalist)
    width_height = datalist[0].strip('\n').split(',')[-1]  # 图片尺寸
    for i in range(nSamples):
        print(datalist[i])
        imagePath_name = datalist[i].strip('\n').split(',')[0]
        imagePath = os.path.join(cfg.data_dir, cfg.train_image_dir_name, imagePath_name)
        labelPath = os.path.join(cfg.data_dir, cfg.train_label_dir_name, imagePath_name[:-4]+'_gt.npy')
        label = np.load(labelPath)

        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except(Exception):
                print('error occured', i)
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('%s-th image data occured error\n' % str(i))
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'.encode()] = str(nSamples).encode()
    cache['width-height'.encode()] = str(width_height).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


def directCreateDataset(gtFile, outputPath, checkValid=True, map_size=8589934592, data_dir=cfg.data_dir):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=map_size)  # 85899345920/8Gb
    cache = {}
    cnt = 1
    gtFile = os.path.join(data_dir, gtFile)

    with open(gtFile, 'r') as data:
        f_list = data.readlines()

    nSamples = len(f_list)
    for line, _ in zip(f_list, tqdm(range(nSamples))):
        print('第{}张图片：{}'.format(cnt, f_list[cnt - 1]))
        line_cols = str(line).strip().split(',')
        img_name, width, height = \
            line_cols[0].strip(), int(line_cols[1].strip()), \
            int(line_cols[2].strip())
        gt = np.zeros((height // cfg.pixel_size, width // cfg.pixel_size, 7))
        train_label_dir = os.path.join(data_dir, cfg.train_label_dir_name)  # 'labels_%s/' % train_task_id
        xy_list_array = np.load(os.path.join(train_label_dir, img_name[:-4] + '.npy'))  # (N, 4, 2)
        train_image_dir = os.path.join(data_dir, cfg.train_image_dir_name)
        if not os.path.exists(os.path.join(train_image_dir, img_name)):
            print('%s does not exist' % os.path.join(train_image_dir, img_name))
            continue
# ---------------------------------生成标签---------------------------------
        with Image.open(os.path.join(train_image_dir, img_name)) as im:
            draw = ImageDraw.Draw(im)
            for xy_list in xy_list_array:
                _, shrink_xy_list, _ = shrink(xy_list, cfg.shrink_ratio)
                shrink_1, _, long_edge = shrink(xy_list, cfg.shrink_side_ratio)
                p_min = np.amin(shrink_xy_list, axis=0)
                p_max = np.amax(shrink_xy_list, axis=0)
                # floor of the float
                ji_min = (p_min / cfg.pixel_size - 0.5).astype(int) - 1
                # +1 for ceil of the float and +1 for include the end
                ji_max = (p_max / cfg.pixel_size - 0.5).astype(int) + 3
                imin = np.maximum(0, ji_min[1])
                imax = np.minimum(height // cfg.pixel_size, ji_max[1])
                jmin = np.maximum(0, ji_min[0])
                jmax = np.minimum(width // cfg.pixel_size, ji_max[0])
                for i in range(imin, imax):
                    for j in range(jmin, jmax):
                        px = (j + 0.5) * cfg.pixel_size
                        py = (i + 0.5) * cfg.pixel_size
                        if point_inside_of_quad(px, py, shrink_xy_list, p_min, p_max):
                            gt[i, j, 0] = 1
                            line_width, line_color = 1, 'red'
                            ith = point_inside_of_nth_quad(px, py,
                                                           xy_list,
                                                           shrink_1,
                                                           long_edge)
                            vs = [[[3, 0], [1, 2]], [[0, 1], [2, 3]]]
                            if ith in range(2):
                                gt[i, j, 1] = 1
                                if ith == 0:
                                    line_width, line_color = 2, 'yellow'
                                else:
                                    line_width, line_color = 2, 'green'
                                gt[i, j, 2:3] = ith
                                gt[i, j, 3:5] = \
                                    xy_list[vs[long_edge][ith][0]] - [px, py]
                                gt[i, j, 5:] = \
                                    xy_list[vs[long_edge][ith][1]] - [px, py]
                            draw.line([(px - 0.5 * cfg.pixel_size,
                                        py - 0.5 * cfg.pixel_size),
                                       (px + 0.5 * cfg.pixel_size,
                                        py - 0.5 * cfg.pixel_size),
                                       (px + 0.5 * cfg.pixel_size,
                                        py + 0.5 * cfg.pixel_size),
                                       (px - 0.5 * cfg.pixel_size,
                                        py + 0.5 * cfg.pixel_size),
                                       (px - 0.5 * cfg.pixel_size,
                                        py - 0.5 * cfg.pixel_size)],
                                      width=line_width, fill=line_color)
            act_image_dir = os.path.join(cfg.data_dir, cfg.show_act_image_dir_name)
            if cfg.draw_act_quad:
                im.save(os.path.join(act_image_dir, img_name))
        # train_label_dir = os.path.join(data_dir, cfg.train_label_dir_name)  # 'labels_%s/' % train_task_id
        # np.save(os.path.join(train_label_dir, img_name[:-4] + '_gt.npy'), gt)
        imagePath = os.path.join(cfg.data_dir, cfg.train_image_dir_name, img_name)
        label = gt
# ---------------------------写入LMDB---------------------------
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except(Exception):
                print('error occured', i)
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('%s-th image data occured error\n' % str(i))
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        gt_xy_list_Key = 'gt_xy_list-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        cache[gt_xy_list_Key] = xy_list_array

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'.encode()] = str(nSamples).encode()
    cache['width-height'.encode()] = str(width).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


def genData():
    if not os.path.exists(os.path.join(cfg.data_dir, cfg.val_fname)):
        preprocess()
    mapsize_256 = 2.6e8
    train_mapsize = (int(cfg.train_task_id[-3:]) / 256)**2 * mapsize_256 * 1.3
    val_mapsize = train_mapsize // 10
    directCreateDataset(cfg.train_fname, cfg.lmdb_trainset_dir_name, checkValid=True, map_size=train_mapsize)
    directCreateDataset(cfg.val_fname, cfg.lmdb_valset_dir_name, checkValid=True, map_size=val_mapsize)


if __name__ == "__main__":
    genData()
