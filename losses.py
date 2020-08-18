# coding=utf-8
import torch
import torch.nn as nn

import cfg
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def quad_loss(y_true, y_pred):
    if y_true.size(1) == 7:
        y_true = y_true.permute(0, 2, 3, 1)  # NCHW->NHWC
        y_pred = y_pred.permute(0, 2, 3, 1)  # NCHW->NHWC
    # loss for inside_score
    logits = y_pred[:, :, :, :1]
    labels = y_true[:, :, :, :1]  # NHW1
    # balance positive and negative samples in an image
    beta = 1 - torch.mean(labels)
    # first apply sigmoid activation
    predicts = nn.Sigmoid().to(device)(logits)
    # log +epsilon for stable cal
    inside_score_loss = torch.mean(
        -1 * (beta * labels * torch.log(predicts + cfg.epsilon) +
              (1 - beta) * (1 - labels) * torch.log(1 - predicts + cfg.epsilon)))
    inside_score_loss = inside_score_loss * cfg.lambda_inside_score_loss

    # loss for side_vertex_code
    vertex_logits = y_pred[:, :, :, 1:3]
    vertex_labels = y_true[:, :, :, 1:3]
    vertex_beta = 1 - (torch.mean(y_true[:, :, :, 1:2])
                       / (torch.mean(labels) + cfg.epsilon))
    vertex_predicts = nn.Sigmoid().to(device)(vertex_logits)
    pos = -1 * vertex_beta * vertex_labels * torch.log(vertex_predicts + cfg.epsilon)
    neg = -1 * (1 - vertex_beta) * (1 - vertex_labels) * torch.log(
        1 - vertex_predicts + cfg.epsilon)
    positive_weights = torch.eq(y_true[:, :, :, 0], 1).float()
    side_vertex_code_loss = \
        torch.sum(torch.sum(pos + neg, dim=-1) * positive_weights) / (
                torch.sum(positive_weights) + cfg.epsilon)
    side_vertex_code_loss = side_vertex_code_loss * cfg.lambda_side_vertex_code_loss

    # loss for side_vertex_coord delta
    g_hat = y_pred[:, :, :, 3:]  # N*W*H*8
    g_true = y_true[:, :, :, 3:]
    vertex_weights = torch.eq(y_true[:, :, :, 1], 1).float()
    pixel_wise_smooth_l1norm = smooth_l1_loss(g_hat, g_true, vertex_weights)  # N*W*H
    side_vertex_coord_loss = torch.sum(pixel_wise_smooth_l1norm) / (
            torch.sum(vertex_weights) + cfg.epsilon)
    side_vertex_coord_loss = side_vertex_coord_loss * cfg.lambda_side_vertex_coord_loss
    return inside_score_loss + side_vertex_code_loss + side_vertex_coord_loss


def smooth_l1_loss(prediction_tensor, target_tensor, weights):
    n_q = torch.reshape(quad_norm(target_tensor), weights.size())
    pixel_wise_smooth_l1norm = torch.nn.SmoothL1Loss(reduction='none')(prediction_tensor, target_tensor)
    pixel_wise_smooth_l1norm = torch.sum(pixel_wise_smooth_l1norm, dim=-1) / n_q * weights  # N*W*H
    return pixel_wise_smooth_l1norm


def quad_norm(g_true):  # 尾部短边长度*4
    g_shape = g_true.size()
    delta_xy_matrix = torch.reshape(g_true, [-1, 2, 2])
    diff = delta_xy_matrix[:, 0:1, :] - delta_xy_matrix[:, 1:2, :]
    square = diff**2
    distance = torch.sqrt(torch.sum(square, dim=-1))
    distance = distance * 4.0
    distance = distance + cfg.epsilon
    distance = torch.reshape(distance, g_shape[:-1])
    return distance


if __name__ == '__main__':
    gt_1 = np.load('check/1_gt.npy')
    gt_2 = np.load('check/2_gt.npy')
    gt_1 = gt_1[np.newaxis]
    gt_2 = gt_2[np.newaxis]
    tensor_1 = torch.from_numpy(gt_1).to(device)
    tensor_2 = torch.from_numpy(gt_2).to(device)
    print(tensor_1.shape)
    print(quad_loss(tensor_1, tensor_2))  # GT=1.0282
