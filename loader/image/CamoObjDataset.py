import torchvision.transforms as transforms
import os
from PIL import Image
import torch.utils.data as data
from loader.custom_transforms import random_flip,random_crop,random_rotation,image_suffix,color_enhance,suppress_foreground_contrast,suppress_contrast_with_overlay
import cv2

class ISODTrainDataset(data.Dataset):
    def __init__(self, dataset_root, texture_type,trainsize,hard_aug=False):

        image_root = dataset_root + '/RGB/'
        
        bound_root = dataset_root + '/bound/'

        gt_root = dataset_root + '/GT/'

        self.trainsize = trainsize
        self.images = sorted([image_root + f for f in os.listdir(image_root) if image_suffix(f)])
        self.gts = sorted([gt_root + f for f in os.listdir(gt_root) if image_suffix(f)])
        #self.depths = sorted([depth_root + f for f in os.listdir(depth_root) if image_suffix(f)])
        #self.texs = sorted([texture_root+f for f in os.listdir(texture_root) if image_suffix(f)])
        self.bounds = sorted([bound_root+f for f in os.listdir(bound_root) if image_suffix(f)])
        
        #assert len(self.images) == len(self.depths) and len(self.gts) == len(self.images)
        self.size = len(self.images)
        print(f'load {self.size} train data from {dataset_root}')
        self.rgb_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.binary_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize),interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])
        self.logistic_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.hard_aug = hard_aug

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        #depth = self.rgb_loader(self.depths[index])
        
        gt = self.binary_loader(self.gts[index])
        bound = self.binary_loader(self.bounds[index])

        image, gt, bound = random_flip(image, gt, bound)
        image, gt, bound = random_crop(image, gt, bound)
        image, gt, bound = random_rotation(image, gt, bound)
        image = color_enhance(image)
        if self.hard_aug:
            image = suppress_foreground_contrast(image,gt)
            image = suppress_contrast_with_overlay(image,gt)

        image = self.rgb_transform(image)
        #depth = self.logistic_transform(depth)
        gt = self.binary_transform(gt)
        bound = self.binary_transform(bound)
        
        return image, gt, bound


    def rgb_loader(self, path):
        return Image.open(path).convert('RGB')

    def binary_loader(self, path):
        return Image.open(path).convert('L')


    def __len__(self):
        return self.size

class ISODValDataset(data.Dataset):
    def __init__(self, dataset_root, trainsize):

        image_root = dataset_root + '/RGB/'
        #depth_root = dataset_root + '/depth/'
        gt_root = dataset_root + '/GT/'
        
        self.trainsize = trainsize
        self.images = sorted([image_root + f for f in os.listdir(image_root) if image_suffix(f)])
        self.gts = sorted([gt_root + f for f in os.listdir(gt_root) if image_suffix(f)])
        #self.depths = sorted([depth_root + f for f in os.listdir(depth_root) if image_suffix(f)])
       
        #assert len(self.images) == len(self.depths) and len(self.gts) == len(self.images)
        self.size = len(self.images)
        print(f'load {self.size} val data from {dataset_root}')
        self.rgb_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.depth_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.binary_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize),interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])


    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        #depth = self.rgb_loader(self.depths[index])
        gt = self.binary_loader(self.gts[index])

        image = self.rgb_transform(image)
        #depth = self.depth_transform(depth)
        gt = self.binary_transform(gt)

        return image, gt

    def rgb_loader(self, path):
        return Image.open(path).convert('RGB')

    def binary_loader(self, path):
        return Image.open(path).convert('L')

    def resize(self, img, gt, depth):
        assert img.size == gt.size and gt.size == depth.size
        h = self.trainsize
        w = self.trainsize
        return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), depth.resize((w, h),Image.NEAREST)

    def __len__(self):
        return self.size

# test dataset and loader
class ISODTestDataset(data.Dataset):
    def __init__(self, dataset_root, trainsize):

        image_root = dataset_root + '/RGB/'
        #depth_root = dataset_root + '/depth/'
        gt_root = dataset_root + '/GT/'

        self.trainsize = trainsize
        self.images = sorted([image_root + f for f in os.listdir(image_root) if image_suffix(f)])
        self.gts = sorted([gt_root + f for f in os.listdir(gt_root) if image_suffix(f)])
        #self.depths = sorted([depth_root + f for f in os.listdir(depth_root) if image_suffix(f)])
        
        
        #assert len(self.images) == len(self.depths) and len(self.gts) == len(self.images)
        self.size = len(self.images)
        print(f'load {self.size} test data from {dataset_root}')
        self.rgb_transform = transforms.Compose([
           
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.binary_transform = transforms.Compose([
            transforms.ToTensor()])


    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = cv2.imread(self.gts[index],cv2.IMREAD_GRAYSCALE)
        sz = gt.shape
        #depth = self.rgb_loader(self.depths[index])
        h,w = sz

        resize = transforms.Resize([384,384])

        image = resize(self.rgb_transform(image))
        #depth = resize(self.binary_transform(depth))

        name = self.images[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        
        return  image, gt,  sz, name


    def rgb_loader(self, path):
        return Image.open(path).convert('RGB')

    def binary_loader(self, path):
        return Image.open(path).convert('L')

    def resize(self, img, gt, depth):
        assert img.size == gt.size and gt.size == depth.size
        h = self.trainsize
        w = self.trainsize
        return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), depth.resize((w, h),Image.NEAREST)

    def __len__(self):
        return self.size


def get_camo_loader(data_cfg, batchsize,  dist = False, ds_type='pretrain', texture_type = None):
    if ds_type == 'pretrain' or ds_type == 'finetune':
        train_datasets = []
        for dataset_name in data_cfg.TRAINSET:
            train_datasets.append(ISODTrainDataset(os.path.join(data_cfg.DATA_ROOT, 'train', dataset_name),texture_type,data_cfg.PRETRAIN_SIZE if ds_type == 'pretrain' else data_cfg.FINETUNE_SIZE))
        if hasattr(data_cfg,'ADDITIONSET') and ds_type == 'pretrain':
            for dataset_name in data_cfg.ADDITIONSET:
                train_datasets.append(ISODTrainDataset(os.path.join(data_cfg.ADDITION_ROOT, 'train', dataset_name),texture_type,data_cfg.PRETRAIN_SIZE, hard_aug = True))
        train_datasets = data.ConcatDataset(train_datasets)
        
        data_loader = data.DataLoader(dataset=train_datasets,
                                      batch_size=batchsize,
                                      pin_memory=True,
                                      drop_last=True,
                                      num_workers=12,
                                      sampler= data.distributed.DistributedSampler(train_datasets) if dist else None)
        
    elif ds_type == 'val':
        val_datasets = []
        for dataset_name in data_cfg.VALSET:
            val_datasets.append(ISODValDataset(os.path.join(data_cfg.DATA_ROOT, ds_type, dataset_name),data_cfg.FINETUNE_SIZE))
        val_datasets = data.ConcatDataset(val_datasets)
        data_loader = data.DataLoader(dataset=val_datasets,
                                      batch_size=batchsize//2,
                                      pin_memory=True,
                                      drop_last=False,
                                      num_workers=12,
                                      sampler=data.distributed.DistributedSampler(val_datasets) if dist else None)
    elif ds_type == 'test':
        test_datasets = []
        for dataset_name in data_cfg.TESTSET:
            test_datasets.append(ISODTestDataset(os.path.join(data_cfg.DATA_ROOT, ds_type,dataset_name),data_cfg.FINETUNE_SIZE))
        
        data_loader = test_datasets
    else:
        raise NotImplementedError("no such dataset")
    return data_loader
