import os
import sys
import json
import numpy as np
import logging
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr , kendalltau
from opts import parse_opts
from model.network_attention import C3DVQANet
# from dataset.dataset_snet import VideoDataset
from dataset.dataset_s import VideoDataset  # 1
from tool.draw import mos_scatter
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = 'cuda'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
writer = SummaryWriter()

def train_model(model, device, criterion, optimizer, scheduler, dataloaders, save_checkpoint, epoch_resume=1, num_epochs=25):

    for epoch in tqdm(range(epoch_resume, num_epochs+epoch_resume), unit='epoch', initial=epoch_resume, total=num_epochs+epoch_resume):
        logging.info("epoch:{epoch}".format(epoch=epoch))
        for phase in ['train', 'test']:
            epoch_labels = []
            epoch_preds = []
            epoch_loss = 0.0
            epoch_size = 0

            if phase == 'train':
                model.train()
            else:
                model.eval()

            for ref, dis, labels in dataloaders[phase]:
                ref = ref.to(device)

                dis = dis.to(device)
                labels = labels.to(device).float()


                ref = ref.reshape(-1, ref.shape[2], ref.shape[3], ref.shape[4], ref.shape[5])
                dis = dis.reshape(-1, dis.shape[2], dis.shape[3], dis.shape[4], dis.shape[5])

                optimizer.zero_grad()
                # logging.info("0")
                with torch.set_grad_enabled(phase == 'train'):
                    # logging.info("1")
                    preds = model(ref, dis)
                    # logging.info("2")
                    preds = torch.mean(preds, 0, keepdim=True)
                    # logging.info("3")
                    loss = criterion(preds, labels)
                    # logging.info("4")

                    if torch.cuda.device_count() > 1 and MULTI_GPU_MODE == True:
                        loss = torch.mean(loss)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # logging.info("5")
                epoch_loss += loss.item() * labels.size(0)
                epoch_size += labels.size(0)
                epoch_labels.append(labels.flatten())
                epoch_preds.append(preds.flatten())

            epoch_loss = epoch_loss / epoch_size

            if phase == 'train':
                scheduler.step(epoch_loss)

            epoch_labels = torch.cat(epoch_labels).flatten().data.cpu().numpy()
            epoch_preds = torch.cat(epoch_preds).flatten().data.cpu().numpy()

            logging.info('epoch_labels: {}'.format(epoch_labels))
            logging.info('epoch_preds: {}'.format(epoch_preds))

            epoch_plcc = pearsonr(epoch_labels, epoch_preds)[0]
            epoch_srocc = spearmanr(epoch_labels, epoch_preds)[0]
            epoch_krocc = kendalltau(epoch_labels, epoch_preds)[0]
            epoch_rmse = np.sqrt(np.mean((epoch_labels - epoch_preds)**2))

            logging.info("{phase}-Loss: {loss:.4f}\t RMSE: {rmse:.4f}\t PLCC: {plcc:.4f}\t SROCC: {srocc:.4f}\t KROCC: {krocc:.4f}".format(phase=phase, loss=epoch_loss, rmse=epoch_rmse, plcc=epoch_plcc, srocc=epoch_srocc, krocc=epoch_krocc))

            # if phase == 'train':
            #     writer.add_scalar('RMSE/train', epoch_rmse, epoch)
            #     writer.add_scalar('PLCC/train', epoch_plcc, epoch)
            #     writer.add_scalar('SROCC/train', epoch_srocc, epoch)
            # else:
            #     writer.add_scalar('Loss/test', epoch_loss, epoch)
            #     writer.add_scalar('RMSE/test', epoch_rmse, epoch)
            #     writer.add_scalar('PLCC/test', epoch_plcc, epoch)
            #     writer.add_scalar('SROCC/test', epoch_srocc, epoch)
            #     writer.add_figure('Pred vs. MOS', mos_scatter(epoch_labels, epoch_preds), epoch)
 
            if phase == 'test' and save_checkpoint:
                _checkpoint = '{pt}_{epoch}'.format(pt=save_checkpoint, epoch=epoch)
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, _checkpoint)


if __name__=='__main__':

    opt = parse_opts()

    video_path = opt.video_dir
    subj_dataset = opt.score_file_path
    save_checkpoint = opt.save_model
    load_checkpoint = opt.load_model
    log_file_name = opt.log_file_name
    LEARNING_RATE = opt.learning_rate
    L2_REGULARIZATION = opt.weight_decay
    NUM_EPOCHS = opt.epochs
    MULTI_GPU_MODE = opt.multi_gpu
    channel = opt.channel
    size_x = opt.size_x
    size_y = opt.size_y
    stride_x = opt.stride_x
    stride_y = opt.stride_y
    stride_u = opt.stride_u
    stride_v = opt.stride_v

    logging.basicConfig(filename=log_file_name, filemode='a', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    logging.info('OK parse options')

    video_dataset = {x: VideoDataset(subj_dataset, video_path, x, channel, size_x, size_y, stride_x, stride_y) for x in ['train', 'test']}
    # video_dataset = {x: VideoDataset(subj_dataset, video_path, x, channel, size_x, size_y, stride_u, stride_v) for x in ['train', 'test']}
    logging.info('OK video_dataset')
    dataloaders = {x: torch.utils.data.DataLoader(video_dataset[x], batch_size=1, shuffle=True, num_workers=0, drop_last=True) for x in ['train', 'test']}
    logging.info('OK dataloaders')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1 and MULTI_GPU_MODE == True:
        device_ids = range(0, torch.cuda.device_count())
        model = torch.nn.DataParallel(C3DVQANet().to(device), device_ids=device_ids)
        logging.info("muti-gpu mode enabled, use {0:d} gpus".format(torch.cuda.device_count()))
    else:
        model = C3DVQANet().to(device)
        logging.info('use {0}'.format('cuda' if torch.cuda.is_available() else 'cpu'))

    criterion = nn.MSELoss()  # 使平方误差更小
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REGULARIZATION) # Adam 优化器帮助你更新模型参数，使得预测结果越来越接近真实值。
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5) #如果模型在一段时间内 无法降低损失（误差），它会自动降低学习率，帮助模型更稳定地收敛。
    epoch_resume = 1

    if os.path.exists(load_checkpoint):
        checkpoint = torch.load(load_checkpoint)
        logging.info("loading checkpoint")

        if torch.cuda.device_count() > 1 and MULTI_GPU_MODE == True:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            # model.load_state_dict(checkpoint['model_state_dict'], strict=False) # 加载参数时严格模式设为 False，忽略缺失或多余的参数
            model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        epoch_resume = checkpoint['epoch']

    train_model(model, device, criterion, optimizer, scheduler, dataloaders, save_checkpoint, epoch_resume, num_epochs=NUM_EPOCHS)
    logging.info('OK into train_model')
