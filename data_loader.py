from torchvision import datasets, transforms
import torch
from folder_new import ImageFolder_new
import torch
import cv2
from torch.utils import data
import numpy as np
import torchvision.transforms as tv


# def load_training(root_path, dir, batch_size, kwargs):
#     transform = transforms.Compose(
#         [transforms.Resize([256, 256]),
#          transforms.RandomCrop(224),
#          transforms.RandomHorizontalFlip(),
#          transforms.ToTensor()])
#     data = datasets.ImageFolder(root=root_path + dir, transform=transform)
#     train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
#     return train_loader
#
#
# def load_testing(root_path, dir, batch_size, kwargs):
#     transform = transforms.Compose(
#         [transforms.Resize([224, 224]),
#          transforms.ToTensor()])
#     data = datasets.ImageFolder(root=root_path + dir, transform=transform)
#     test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
#     return test_loader
class Office(data.Dataset):

    def __init__(self, list, training=True):
        self.images = []
        self.labels = []
        self.multi_scale = [256, 257]
        self.output_size = [224, 224]
        self.training = training
        self.mean_color = [104.006, 116.668, 122.678]

        list_file = open(list)
        lines = list_file.readlines()
        for line in lines:
            fields = line.split()
            self.images.append(fields[0])
            self.labels.append(int(fields[1]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label = self.labels[index]
        img = cv2.imread(image_path)
        if type(img) == None:
            print('Error: Image at {} not found.'.format(image_path))

        if self.training and np.random.random() < 0.5:
            img = cv2.flip(img, 1)
        new_size = np.random.randint(self.multi_scale[0], self.multi_scale[1], 1)[0]

        img = cv2.resize(img, (new_size, new_size))
        img = img.astype(np.float32)

        # cropping
        if self.training:
            diff = new_size - self.output_size[0]
            offset_x = np.random.randint(0, diff, 1)[0]
            offset_y = np.random.randint(0, diff, 1)[0]
        else:
            offset_x = img.shape[0] // 2 - self.output_size[0] // 2
            offset_y = img.shape[1] // 2 - self.output_size[1] // 2

        img = img[offset_x:(offset_x + self.output_size[0]),
              offset_y:(offset_y + self.output_size[1])]

        # substract mean
        img -= np.array(self.mean_color)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ToTensor transform cv2 HWC->CHW, only byteTensor will be div by 255.
        tensor = tv.ToTensor()
        img = tensor(img)
        # img = np.transpose(img, (2, 0, 1))

        return img, label


def generate_dataloader(root_path, source1_name, source2_name, source3_name, target_name, batch_size, num_workers):
    # Data loading code
    """
    traindir_source = os.path.join(args.data_path_source, args.src)
    traindir_target = os.path.join(args.data_path_source_t, args.src_t)
    valdir = os.path.join(args.data_path_target, args.tar)
    """

    traindir_source_1 = root_path + source1_name
    traindir_source_2 = root_path + source2_name
    traindir_source_3 = root_path + source3_name
    traindir_target = root_path + target_name
    valdir_target = root_path + target_name

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    source1_train_dataset = datasets.ImageFolder(
        traindir_source_1,
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    source1_loader = torch.utils.data.DataLoader(
        source1_train_dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers, pin_memory=True, sampler=None
    )

    source2_train_dataset = datasets.ImageFolder(
        traindir_source_2,
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    source2_loader = torch.utils.data.DataLoader(
        source2_train_dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers, pin_memory=True, sampler=None
    )

    source3_train_dataset = datasets.ImageFolder(
        traindir_source_3,
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    source3_loader = torch.utils.data.DataLoader(
        source3_train_dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers, pin_memory=True, sampler=None
    )

    target_train_dataset = datasets.ImageFolder(
        traindir_target,
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )

    target_train_loader = torch.utils.data.DataLoader(
        target_train_dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers, pin_memory=True, sampler=None
    )

    target_test_loader = torch.utils.data.DataLoader(
        ImageFolder_new(valdir_target, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return source1_loader, source2_loader, source3_loader, target_train_loader, target_test_loader
