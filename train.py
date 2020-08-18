import sys
import time
import os

import torch
import torch.utils.data
import torch.optim as optim

from model_VGG import advancedEAST
from losses import quad_loss
from dataset import LmdbDataset, data_collate
from utils import Averager, eval_pre_rec_f1
from imgs2LMDB import genData
import cfg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train():
    """ dataset preparation """
    train_dataset_lmdb = LmdbDataset(cfg.lmdb_trainset_dir_name)
    val_dataset_lmdb = LmdbDataset(cfg.lmdb_valset_dir_name)

    train_loader = torch.utils.data.DataLoader(
        train_dataset_lmdb, batch_size=cfg.batch_size,
        collate_fn=data_collate,
        shuffle=True,
        num_workers=int(cfg.workers),
        pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(
        val_dataset_lmdb, batch_size=cfg.batch_size,
        collate_fn=data_collate,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(cfg.workers),
        pin_memory=True)

    # --------------------训练过程---------------------------------
    model = advancedEAST()
    if int(cfg.train_task_id[-3:]) != 256:
        id_num = cfg.train_task_id[-3:]
        idx_dic = {'384': 256, '512': 384, '640': 512, '736': 640}
        model.load_state_dict(torch.load('./saved_model/3T{}_best_loss.pth'.format(idx_dic[id_num])))
    elif os.path.exists('./saved_model/3T{}_best_loss.pth'.format(cfg.train_task_id)):
        model.load_state_dict(torch.load('./saved_model/3T{}_best_loss.pth'.format(cfg.train_task_id)))

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.decay)
    loss_func = quad_loss

    train_Loss_list = []
    val_Loss_list = []

    '''start training'''
    start_iter = 0
    if cfg.saved_model != '':
        try:
            start_iter = int(cfg.saved_model.split('_')[-1].split('.')[0])
            print('continue to train, start_iter: {}'.format(start_iter))
        except Exception as e:
            print(e)
            pass

    start_time = time.time()
    best_mF1_score = 0
    i = start_iter
    step_num = 0
    start_time = time.time()
    loss_avg = Averager()
    val_loss_avg = Averager()
    eval_p_r_f = eval_pre_rec_f1()

    while(True):
        model.train()
        # train part
        # training-----------------------------
        for image_tensors, labels, gt_xy_list in train_loader:
            step_num += 1
            batch_x = image_tensors.to(device).float()
            batch_y = labels.to(device).float()  # float64转float32

            out = model(batch_x)
            loss = loss_func(batch_y, out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_avg.add(loss)
            train_Loss_list.append(loss_avg.val())
            if i == 5 or (i + 1) % 10 == 0:
                eval_p_r_f.add(out, gt_xy_list)  # 非常耗时！！！

        # save model per 100 epochs.
        if (i + 1) % 1e+2 == 0:
            torch.save(model.state_dict(), './saved_models/{}/{}_iter_{}.pth'.format(cfg.train_task_id, cfg.train_task_id, step_num+1))

        print('Epoch:[{}/{}] Training Loss: {:.3f}'.format(i + 1, cfg.epoch_num, train_Loss_list[-1].item()))
        loss_avg.reset()

        if i == 5 or (i + 1) % 10 == 0:
            mPre, mRec, mF1_score = eval_p_r_f.val()
            print('Training meanPrecision:{:.2f}% meanRecall:{:.2f}% meanF1-score:{:.2f}%'.format(mPre, mRec, mF1_score))
            eval_p_r_f.reset()

        # evaluation--------------------------------
        if (i + 1) % cfg.valInterval == 0:
            elapsed_time = time.time() - start_time
            print('Elapsed time:{}s'.format(round(elapsed_time)))
            model.eval()
            for image_tensors, labels, gt_xy_list in valid_loader:
                batch_x = image_tensors.to(device)
                batch_y = labels.to(device).float()  # float64转float32

                out = model(batch_x)
                loss = loss_func(batch_y, out)

                val_loss_avg.add(loss)
                val_Loss_list.append(val_loss_avg.val())
                eval_p_r_f.add(out, gt_xy_list)

            mPre, mRec, mF1_score = eval_p_r_f.val()
            print('validation meanPrecision:{:.2f}% meanRecall:{:.2f}% meanF1-score:{:.2f}%'.format(mPre, mRec, mF1_score))
            eval_p_r_f.reset()

            if mF1_score > best_mF1_score:  # 记录最佳模型
                best_mF1_score = mF1_score
                torch.save(model.state_dict(), './saved_models/{}/{}_best_mF1_score_{:.3f}.pth'.format(cfg.train_task_id, cfg.train_task_id, mF1_score))
                torch.save(model.state_dict(), './saved_model/{}_best_mF1_score.pth'.format(cfg.train_task_id))

            print('Validation loss:{:.3f}'.format(val_loss_avg.val().item()))
            val_loss_avg.reset()

        if i == cfg.epoch_num:
            torch.save(model.state_dict(), './saved_models/{}/{}_iter_{}.pth'.format(cfg.train_task_id, cfg.train_task_id, i+1))
            print('End the training')
            break
        i += 1

    sys.exit()


if __name__ == '__main__':
    os.makedirs('./saved_models/{}'.format(cfg.train_task_id), exist_ok=True)
    if not os.path.exists(cfg.lmdb_valset_dir_name):  # 生成数据
        genData()
    train()
