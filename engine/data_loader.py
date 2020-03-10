import torch
import torchvision
import torchvision.transforms as transforms

__all__ = ['DataLoader', 'DataLoader.get_cifar10', 'DataLoader.cifar10_mean', 'DataLoader.cifar10_std', 'DataLoader.cifar10_classes']

class DataLoader():
    cifar10_mean = (0.491, 0.482, 0.447)
    cifar10_std = (0.247, 0.243, 0.262)
    cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    @staticmethod
    def get_cifar10(batch_size:int=64, augmentations:list=None, seed:int=None):
        '''
        Load CIFAR10 data

        :params: 
            batch_size : int
            augmentation transforms : List
            seed for dataloaders: int
        :returns: 
            train dataloader
            test dataloader
        '''
        mean = (0.485, 0.456, 0.406) 
        std = (0.229, 0.224, 0.225)

        transforms_list = [ 
            transforms.ToTensor(), 
            transforms.Normalize(mean, std)
            ]

        if augmentations:
            transforms_list.extend(augmentations)

        train_transforms = transforms.Compose(transforms_list)
        test_transforms = transforms.Compose(transforms_list)

        #Get the Train and Test Set
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)

        # CUDA?
        cuda = torch.cuda.is_available()
        # print("CUDA Available?", cuda)

        if seed:
            torch.manual_seed(seed)
            if cuda:
                torch.cuda.manual_seed(seed)

        # dataloader arguments - something you'll fetch these from cmdprmt
        dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=batch_size)

        train_loader = torch.utils.data.DataLoader(trainset, **dataloader_args)
        test_loader = torch.utils.data.DataLoader(testset, **dataloader_args)

        return train_loader, test_loader