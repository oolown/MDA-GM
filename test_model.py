from __future__ import print_function

from os.path import join
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
import data_loader as data_loader
import model_gcn as models
from tensorboardX import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Training settings
batch_size = 40
iteration = 10000
lr = 0.01
momentum = 0.9

cuda = torch.cuda.is_available()
seed = 8
log_interval = 10
l2_decay = 5e-4

root_path = "D:/dataset/"
dataset_type = "office"
source1_name = 'dslr'
source2_name = "webcam"
target_name = "amazon"
# target_name_2 = "dslr"
# s_1_list_path = './data_list/' + source1_name + '_list.txt'
# s_2_list_path = './data_list/' + source2_name + '_list.txt'
t_list_path = './data_list/' + target_name + '_list.txt'
# log_dir = 'log' + '/' + source1_name + '-' + source2_name + '-gcn0.01-to-' + target_name
# logger = SummaryWriter(log_dir)
# checkpoints_dir = "checkpoint/"
# log_name = os.path.join(log_dir, 'log.txt')
# with open(log_name, "a") as log_file:
#     now = time.strftime("%c")
#     log_file.write('================ Training Loss (%s) ================\n' % now)
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

# source1_loader = data_loader.load_training(root_path, source1_name, batch_size, kwargs)
# source2_loader = data_loader.load_training(root_path, source2_name, batch_size, kwargs)
# target_train_loader = data_loader.load_training(root_path, target_name, batch_size, kwargs)
# target_test_loader = data_loader.load_testing(root_path, target_name_2, batch_size, kwargs)


# source1_loader = torch.utils.data.DataLoader(data_loader.Office(s_1_list_path),
#                                              batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
# source2_loader = torch.utils.data.DataLoader(data_loader.Office(s_2_list_path),
#                                              batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
# target_train_loader = torch.utils.data.DataLoader(data_loader.Office(t_list_path),
#                                                   batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
target_test_loader = torch.utils.data.DataLoader(data_loader.Office(t_list_path, training=False),
                                                 batch_size=batch_size, num_workers=0)


# source1_loader, source2_loader, target_train_loader, target_test_loader \
#     = data_loader.generate_dataloader(root_path, dataset_type, source1_name, source2_name, target_name, batch_size, 0)


def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    correct1 = 0
    correct2 = 0
    pretrain = torch.load('C:/Users/oolown/Desktop/paper/dslr-webcamtoamazon--office_amazon0.008-onelayergcn/best.pkl',
                          map_location=lambda storage, loc: storage)
    model.load_state_dict(pretrain['model'])
    with torch.no_grad():
        # for i, all_test in enumerate(target_test_loader):
        #     data, target, _ = all_test
        for data, target in target_test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            pred1, pred2 = model(data, mark=0)

            pred1 = torch.nn.functional.softmax(pred1, dim=1)
            pred2 = torch.nn.functional.softmax(pred2, dim=1)

            pred = (pred1 + pred2) / 2
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()

            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred1.data.max(1)[1]
            correct1 += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred2.data.max(1)[1]
            correct2 += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(target_test_loader.dataset)
        print(target_name, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
            test_loss, correct, len(target_test_loader.dataset),
            100. * correct / len(target_test_loader.dataset)))
        print('\nsource1 accnum {}, source2 accnum {}'.format(correct1, correct2))
        # logger.add_scalar(source1_name+'accnum', (correct)/(len(target_test_loader.dataset)), i)
        # logger.add_scalar(source1_name+'accnum', correct1, i)
        # logger.add_scalar(source2_name+'accnum', correct2, i)

        # with open(log_name, "a") as log_file: log_file.write('Accuracy:Loss: {:%.6f}'%(100. * correct / len(
        # target_test_loader.dataset))) log_file.write( source1_name + source2_name + "to" + target_name + "%s max
        # correct:" % target_name + correct.item() + "\n")
    return correct


if __name__ == '__main__':

    model = models.MFSAN(num_classes=31)
    print(model)
    if cuda:
        model.cuda()
    test(model)
