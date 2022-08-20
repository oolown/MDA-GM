from __future__ import print_function

import time
from os.path import join

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
import data_loader as data_loader
import model as models
from tensorboardX import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Training settings
batch_size = 10
iteration = 15000
lr = 0.01
momentum = 0.9
cuda = torch.cuda.is_available()
seed = 8
log_interval = 10
l2_decay = 8e-4
class_num = 65
dataset_type = "officehome"
source2_name = 'Product'
source3_name = "Clipart"
source1_name = "Real_World"
target_name = "Art"

s_1_list_path = 'D:/cd/dataset/' + dataset_type + '_list/' + source1_name + '.txt'
s_2_list_path = 'D:/cd/dataset/' + dataset_type + '_list/' + source2_name + '.txt'
s_3_list_path = 'D:/cd/dataset/' + dataset_type + '_list/' + source3_name + '.txt'
t_list_path = 'D:/cd/dataset/' + dataset_type + '_list/' + target_name + '.txt'

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# ------------------------A-----------------
# source1_loader = data_loader.load_training(root_path, source1_name, batch_size, kwargs)
# source2_loader = data_loader.load_training(root_path, source2_name, batch_size, kwargs)
# source3_loader = data_loader.load_training(root_path, source3_name, batch_size, kwargs)
# target_train_loader = data_loader.load_training(root_path, target_name, batch_size, kwargs)
# target_test_loader = data_loader.load_testing(root_path, target_name, batch_size, kwargs)
# -----------------------B------------------
source1_loader = torch.utils.data.DataLoader(data_loader.Office(s_1_list_path),
                                             batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
source2_loader = torch.utils.data.DataLoader(data_loader.Office(s_2_list_path),
                                             batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
source3_loader = torch.\
    utils.data.DataLoader(data_loader.Office(s_3_list_path),
                                             batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
target_train_loader = torch.utils.data.DataLoader(data_loader.Office(t_list_path),
                                                  batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
target_test_loader = torch.utils.data.DataLoader(data_loader.Office(t_list_path, training=False),
                                                 batch_size=batch_size, num_workers=0)


# -----------------------C------------------
# source1_loader, source2_loader, source3_loader, target_train_loader, target_test_loader \
#     = data_loader.generate_dataloader(root_path, source1_name, source2_name,
#                                       source3_name, target_name, batch_size, 0)


def train(model):
    source1_iter = iter(source1_loader)
    source2_iter = iter(source2_loader)
    source3_iter = iter(source3_loader)
    target_iter = iter(target_train_loader)
    correct = 0

    for i in range(1, iteration + 1):
        model.train()
        LEARNING_RATE = lr / math.pow((1 + 10 * (i - 1) / iteration), 0.75)
        if (i - 1) % 100 == 0:
            print("learning rate：", LEARNING_RATE)
        optimizer = torch.optim.SGD([
            {'params': model.sharedNet.parameters()},
            {'params': model.gcns_on1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.gcns_on2.parameters(), 'lr': LEARNING_RATE},
            {'params': model.gcns_on3.parameters(), 'lr': LEARNING_RATE},
            {'params': model.structure_analyzer_1.parameters()},
            {'params': model.structure_analyzer_2.parameters()},
            {'params': model.structure_analyzer_3.parameters()},
            {'params': model.cls_fc_son1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc_son2.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc_son3.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnet1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnet2.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnet3.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)
        # optimizer = torch.optim.SGD([
        #     {'params': model.sharedNet.parameters()},
        #     {'params': model.cls_fc_son1.parameters(), 'lr': LEARNING_RATE},
        #     {'params': model.cls_fc_son2.parameters(), 'lr': LEARNING_RATE},
        #     {'params': model.cls_fc_son3.parameters(), 'lr': LEARNING_RATE},
        #     {'params': model.sonnet1.parameters(), 'lr': LEARNING_RATE},
        #     {'params': model.sonnet2.parameters(), 'lr': LEARNING_RATE},
        #     {'params': model.sonnet3.parameters(), 'lr': LEARNING_RATE},
        # ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)

        try:
            source_data, source_label = source1_iter.next()
        except Exception as err:
            source1_iter = iter(source1_loader)
            source_data, source_label = source1_iter.next()
        try:
            target_data, __ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_train_loader)
            target_data, __ = target_iter.next()
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)
        optimizer.zero_grad()

        cls_loss, mmd_loss, l1_loss, class_loss = model(source_data, target_data, source_label, mark=1)
        gamma = 2 / (1 + math.exp(-10 * i / iteration)) - 1
        loss = cls_loss + gamma * (mmd_loss + l1_loss + class_loss)
        loss.backward()
        optimizer.step()

        if i % (log_interval) == 0:
            print(
                'Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tcls_loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}\tclass_loss: {:.6f}'.format(
                    i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item(),
                    class_loss.item()))
            logger.add_scalar('loss_on1', loss.item(), i)
            logger.add_scalar('cls_loss_on1', cls_loss.item(), i)
            logger.add_scalar('mmd_loss_on1', mmd_loss.item(), i)
            logger.add_scalar('l1_loss_on1', l1_loss.item(), i)
            logger.add_scalar('class_loss_on1', class_loss.item(), i)

        try:
            source_data, source_label = source2_iter.next()
        except Exception as err:
            source2_iter = iter(source2_loader)
            source_data, source_label = source2_iter.next()
        try:
            target_data, __ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_train_loader)
            target_data, __ = target_iter.next()
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)
        optimizer.zero_grad()

        cls_loss, mmd_loss, l1_loss, class_loss = model(source_data, target_data, source_label, mark=2)
        gamma = 2 / (1 + math.exp(-10 * i / iteration)) - 1
        loss = cls_loss + gamma * (mmd_loss + l1_loss + class_loss)
        loss.backward()
        optimizer.step()

        if i % (log_interval) == 0:
            print(
                'Train source2 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tcls_loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}\tclass_loss: {:.6f}'.format(
                    i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item(),
                    class_loss.item()))
            logger.add_scalar('loss_on2', loss.item(), i)
            logger.add_scalar('cls_loss_on2', cls_loss.item(), i)
            logger.add_scalar('mmd_loss_on2', mmd_loss.item(), i)
            logger.add_scalar('l1_loss_on2', l1_loss.item(), i)
            logger.add_scalar('class_loss_on2', class_loss.item(), i)

        try:
            source_data, source_label = source3_iter.next()
        except Exception as err:
            source3_iter = iter(source3_loader)
            source_data, source_label = source3_iter.next()
        try:
            target_data, __ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_train_loader)
            target_data, __ = target_iter.next()
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)
        optimizer.zero_grad()

        cls_loss, mmd_loss, l1_loss, class_loss = model(source_data, target_data, source_label, mark=3)
        gamma = 2 / (1 + math.exp(-10 * i / iteration)) - 1
        loss = cls_loss + gamma * (mmd_loss + l1_loss + class_loss)
        loss.backward()
        optimizer.step()

        if i % (log_interval) == 0:
            print(
                'Train source3 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tcls_loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}\tclass_loss: {:.6f}'.format(
                    i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item(),
                    class_loss.item()))
            logger.add_scalar('loss_on3', loss.item(), i)
            logger.add_scalar('cls_loss_on3', cls_loss.item(), i)
            logger.add_scalar('mmd_loss_on3', mmd_loss.item(), i)
            logger.add_scalar('l1_loss_on3', l1_loss.item(), i)
            logger.add_scalar('class_loss_on3', class_loss.item(), i)

        if i % (log_interval * 20) == 0:
            t_correct, correct_1, correct_2, correct_3 = test(model)
            logger.add_scalar('target_acc', 1. * t_correct / (len(target_test_loader.dataset)), i)
            logger.add_scalar('target_acc_on1', 1. * correct_1 / (len(target_test_loader.dataset)), i)
            logger.add_scalar('target_acc_on2', 1. * correct_2 / (len(target_test_loader.dataset)), i)
            logger.add_scalar('target_acc_on2', 1. * correct_3 / (len(target_test_loader.dataset)), i)

            if t_correct > correct:
                correct = t_correct
                data = {
                    "model": model.state_dict(),
                    "best_acc": 1. * correct / (len(target_test_loader.dataset))
                }
                with open(join(log_dir, 'best.pkl'), 'wb') as f:
                    torch.save(data, f)

            print(source1_name, source2_name, source3_name, "to", target_name, "%s max correct:" % target_name,
                  correct.item(), "\n")


def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    with torch.no_grad():
        # for i, all_test in enumerate(target_test_loader):
        #     data, target, _ = all_test
        for data, target in target_test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            pred1, pred2, pred3 = model(data)

            pred1 = torch.nn.functional.softmax(pred1, dim=1)
            pred2 = torch.nn.functional.softmax(pred2, dim=1)
            pred3 = torch.nn.functional.softmax(pred3, dim=1)

            pred = (pred1 + pred2 + pred3) / 3
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()  # sum up batch loss
            pred = pred.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred1.data.max(1)[1]  # get the index of the max log-probability
            correct1 += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred2.data.max(1)[1]  # get the index of the max log-probability
            correct2 += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred3.data.max(1)[1]  # get the index of the max log-probability
            correct3 += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(target_test_loader.dataset)

        print(target_name, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
            test_loss, correct, len(target_test_loader.dataset),
            100. * correct / len(target_test_loader.dataset)))
        print('\nsource1 accnum {}, source2 accnum {}，source3 accnum {}'.format(correct1, correct2, correct3))
        data = {
            "model": model.state_dict(),
            "last_acc": 1. * correct / (len(target_test_loader.dataset))
        }
        with open(join(log_dir, 'last.pkl'), 'wb') as f:
            torch.save(data, f)

    return correct, correct1, correct2, correct3


if __name__ == '__main__':
    tag = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
    log_dir = 'loggcnclass' + '/' + target_name + '/' + source1_name + '-' + source2_name + '-' + source3_name + '-' + str(lr)+tag
    # time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    logger = SummaryWriter(log_dir)

    model = models.MDAGM(num_classes=class_num)
    # print(model)

    if cuda:
        model.cuda()
    train(model)
