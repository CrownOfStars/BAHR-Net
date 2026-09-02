import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import torch
from torch.utils.data import Sampler
import random
from collections import defaultdict
from loader.custom_transforms import random_flip,random_crop,random_rotation,color_enhance
import cv2

"""
 TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=1,2,3 python -m torch.distributed.launch --nproc_per_node=3 distributed.py --backbone segswin-base --texture /namlab40/ --mfusion HAIM --lr 3e-4 --train_batch 48
"""
# several data augumentation strategies



class SalVideoPairTrainDataset(data.Dataset):
    def __init__(self, dataset_root, dataset_list, trainsize):
        self.trainsize = trainsize
        self.samples = []

        # 遍历所有视频目录
        for dataset in dataset_list:
            dataset_path = os.path.join(dataset_root, dataset)

            for video in sorted(os.listdir(dataset_path)):
                video_path = os.path.join(dataset_path, video)
                if not os.path.isdir(video_path): continue

                rgb_files = self.image_files_sorted(os.path.join(video_path, 'RGB'))
                gt_files = self.image_files_sorted(os.path.join(video_path, 'GT'))
                bound_files = self.image_files_sorted(os.path.join(video_path, 'bound'))
                flow_files = self.image_files_sorted(os.path.join(video_path, 'FLOW-RAFT'))

                # RGB/GT/Bound 要用第i帧和i+1帧，Flow 对应第i帧
                for i in range(len(flow_files)):
                    self.samples.append({
                        'rgb1': rgb_files[i],
                        'rgb2': rgb_files[i + 1],
                        'gt1': gt_files[i],
                        'gt2': gt_files[i + 1],
                        'bound1': bound_files[i],
                        'bound2': bound_files[i + 1],
                        'flow': flow_files[i]
                    })

            print(f'Loaded {len(self.samples)} paired-frame samples from {dataset_path}')

        self.rgb_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.binary_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        self.flow_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        rgb1 = self.rgb_loader(sample['rgb1'])
        rgb2 = self.rgb_loader(sample['rgb2'])
        gt1 = self.binary_loader(sample['gt1'])
        gt2 = self.binary_loader(sample['gt2'])
        bound1 = self.binary_loader(sample['bound1'])
        bound2 = self.binary_loader(sample['bound2'])
        flow = self.rgb_loader(sample['flow'])  # flow可视图，假定为3通道图像

        # 可添加 joint augment: flip, crop, rotate, color
        rgb1, gt1, bound1, rgb2, gt2, bound2 = random_flip(rgb1, gt1, bound1, rgb2, gt2, bound2)
        rgb1, gt1, bound1, rgb2, gt2, bound2 = random_crop(rgb1, gt1, bound1, rgb2, gt2, bound2)
        rgb1 = color_enhance(rgb1)
        rgb2 = color_enhance(rgb2)

        # 转换为 Tensor
        rgb1 = self.rgb_transform(rgb1)
        rgb2 = self.rgb_transform(rgb2)
        gt1 = self.binary_transform(gt1)
        gt2 = self.binary_transform(gt2)
        bound1 = self.binary_transform(bound1)
        bound2 = self.binary_transform(bound2)
        flow = self.flow_transform(flow)

        return {
            'rgb1': rgb1, 'rgb2': rgb2,
            'gt1': gt1, 'gt2': gt2,
            'bound1': bound1, 'bound2': bound2,
            'flow': flow
        }

    def rgb_loader(self, path):
        return Image.open(path).convert('RGB')

    def binary_loader(self, path):
        return Image.open(path).convert('L')

    def image_files_sorted(self,dir_path):
        return sorted([
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

class SalVideoISOTrainDataset(data.Dataset):
    def __init__(self, dataset_root, datasets, trainsize):
        self.trainsize = trainsize
        self.samples = []

        # 遍历所有视频目录
        for dataset in datasets:
            dataset_path = os.path.join(dataset_root, dataset)

            for video in sorted(os.listdir(dataset_path)):
                video_path = os.path.join(dataset_path, video)
                if not os.path.isdir(video_path): continue

                rgb_files = self.image_files_sorted(os.path.join(video_path, 'RGB'))
                gt_files = self.image_files_sorted(os.path.join(video_path, 'GT'))
                bound_files = self.image_files_sorted(os.path.join(video_path, 'bound'))

                # RGB/GT/Bound 要用第i帧和i+1帧，Flow 对应第i帧
                for i in range(len(rgb_files)):
                    self.samples.append((rgb_files[i], gt_files[i], bound_files[i]))

            print(f'Loaded {len(self.samples)} iso-frame samples from {dataset_path}')

        self.rgb_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.binary_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        self.flow_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        rgb1 = self.rgb_loader(sample[0])
        gt1 = self.binary_loader(sample[1])
        bound1 = self.binary_loader(sample[2])

        # 可添加 joint augment: flip, crop, rotate, color
        rgb1, gt1, bound1 = random_flip(rgb1, gt1, bound1)
        rgb1, gt1, bound1 = random_crop(rgb1, gt1, bound1)
        rgb1 = color_enhance(rgb1)

        # 转换为 Tensor
        rgb1 = self.rgb_transform(rgb1)
        
        gt1 = self.binary_transform(gt1)

        bound1 = self.binary_transform(bound1)

        return rgb1, gt1, bound1

    def rgb_loader(self, path):
        return Image.open(path).convert('RGB')

    def binary_loader(self, path):
        return Image.open(path).convert('L')

    def image_files_sorted(self,dir_path):
        return sorted([
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])


class SalVideoPairValDataset(data.Dataset):
    def __init__(self, dataset_root, dataset_list, trainsize):
        self.trainsize = trainsize
        self.samples = []

        # 遍历所有视频目录
        for dataset in dataset_list:
            dataset_path = os.path.join(dataset_root, dataset)

            for video in sorted(os.listdir(dataset_path)):
                video_path = os.path.join(dataset_path, video)
                if not os.path.isdir(video_path): continue

                rgb_files = self.image_files_sorted(os.path.join(video_path, 'RGB'))
                gt_files = self.image_files_sorted(os.path.join(video_path, 'GT'))
                flow_files = self.image_files_sorted(os.path.join(video_path, 'FLOW-RAFT'))

                # RGB/GT/Bound 要用第i帧和i+1帧，Flow 对应第i帧
                for i in range(len(flow_files)):
                    self.samples.append({
                        'rgb1': rgb_files[i],
                        'rgb2': rgb_files[i + 1],
                        'gt1': gt_files[i],
                        'gt2': gt_files[i + 1],
                        'flow': flow_files[i]
                    })

            print(f'Loaded {len(self.samples)} paired-frame samples from {dataset_path}')

        self.rgb_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.binary_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        self.flow_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        rgb1 = self.rgb_loader(sample['rgb1'])
        rgb2 = self.rgb_loader(sample['rgb2'])
        gt1 = self.binary_loader(sample['gt1'])
        gt2 = self.binary_loader(sample['gt2'])
        flow = self.rgb_loader(sample['flow'])  # flow可视图，假定为3通道图像

        # 可添加 joint augment: flip, crop, rotate, color

        # 转换为 Tensor
        rgb1 = self.rgb_transform(rgb1)
        rgb2 = self.rgb_transform(rgb2)
        gt1 = self.binary_transform(gt1)
        gt2 = self.binary_transform(gt2)
        bound1 = self.binary_transform(bound1)
        bound2 = self.binary_transform(bound2)
        flow = self.flow_transform(flow)

        return {
            'rgb1': rgb1, 'rgb2': rgb2,
            'gt1': gt1, 'gt2': gt2,
            'flow': flow
        }

    def rgb_loader(self, path):
        return Image.open(path).convert('RGB')

    def binary_loader(self, path):
        return Image.open(path).convert('L')

    def image_files_sorted(self,dir_path):
        return sorted([
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])


class SalVideoISOValDataset(data.Dataset):
    def __init__(self, dataset_root, datasets, trainsize):
        self.trainsize = trainsize
        self.samples = []

        for dataset in datasets:
        # 遍历所有视频目录
            dataset_path = os.path.join(dataset_root, dataset)

            for video in sorted(os.listdir(dataset_path)):
                video_path = os.path.join(dataset_path, video)
                if not os.path.isdir(video_path): continue

                rgb_files = self.image_files_sorted(os.path.join(video_path, 'RGB'))
                gt_files = self.image_files_sorted(os.path.join(video_path, 'GT'))
        
                # RGB/GT/Bound 要用第i帧和i+1帧，Flow 对应第i帧
                for i in range(len(rgb_files)):
                    self.samples.append({
                        'rgb1': rgb_files[i],
                        'gt1': gt_files[i]
                    })

            print(f'Loaded {len(self.samples)} iso-frame samples from {dataset_path}')

        self.rgb_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.binary_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        self.flow_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        rgb1 = self.rgb_loader(sample['rgb1'])
        gt1 = self.binary_loader(sample['gt1'])

        # 可添加 joint augment: flip, crop, rotate, color

        # 转换为 Tensor
        rgb1 = self.rgb_transform(rgb1)
        gt1 = self.binary_transform(gt1)

        return rgb1,gt1

    def rgb_loader(self, path):
        return Image.open(path).convert('RGB')

    def binary_loader(self, path):
        return Image.open(path).convert('L')

    def image_files_sorted(self,dir_path):
        return sorted([
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

class DistributedVideoSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, sample_ratio=0.4):
        super().__init__(dataset)
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.sample_ratio = sample_ratio  # 比例控制采样子集
        self.epoch = 0  # 默认初始 epoch 为 0

        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            rank = torch.distributed.get_rank()

        self.video_to_indices = defaultdict(list)

        # 当前 dataset 是 ConcatDataset
        if isinstance(dataset, torch.utils.data.ConcatDataset):
            dataset_offsets = []
            offset = 0
            for sub_dataset in dataset.datasets:
                dataset_offsets.append(offset)
                offset += len(sub_dataset)

            for d_idx, sub_dataset in enumerate(dataset.datasets):
                offset = dataset_offsets[d_idx]
                for idx, sample in enumerate(sub_dataset.samples):
                    video = sample[0].split('/')[-3]
                    self.video_to_indices[video].append(offset + idx)
        else:
            for idx, sample in enumerate(dataset.samples):
                video = sample[0].split('/')[-3]
                self.video_to_indices[video].append(idx)


        # 2. 均匀将视频分配给不同进程
        self.video_list = sorted(self.video_to_indices.keys())
        self.videos_per_replica = self.video_list[rank::num_replicas]

        # 3. 收集当前 rank 负责的视频中所有帧对索引
        self.all_local_indices = []
        for v in self.videos_per_replica:
            self.all_local_indices.extend(self.video_to_indices[v])

        self.full_num_samples = len(self.all_local_indices)  # 全量样本数量

    def __iter__(self):
        # 每轮重新打乱（局部的）
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = self.all_local_indices.copy()
            random.shuffle(indices)
        else:
            indices = self.all_local_indices

        # 仅返回一部分样本
        num_to_sample = int(len(indices) * self.sample_ratio)
        sampled_indices = indices[:num_to_sample]

        return iter(sampled_indices)

    def __len__(self):
        # 注意：这里返回的是采样后的长度
        return int(self.full_num_samples * self.sample_ratio)

    def set_epoch(self, epoch):
        self.epoch = epoch



import torch
import math
import random
from torch.utils.data import Sampler

class BalancedDistributedVideoSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, sample_ratio=0.4):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.sample_ratio = sample_ratio
        self.epoch = 0

        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            rank = torch.distributed.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank

        self.total_size = len(dataset)
        self.num_sampled = int(self.total_size * self.sample_ratio)

        # 确保每个rank采样数量一致，舍弃不能整除部分
        self.num_samples_per_rank = self.num_sampled // self.num_replicas
        self.total_used_samples = self.num_samples_per_rank * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # 采样指定比例的全体 index
        indices = list(range(self.total_size))
        if self.shuffle:
            indices = torch.randperm(self.total_size, generator=g).tolist()

        # 取采样比例的前 K 项
        indices = indices[:self.total_used_samples]
        # 分配给每个rank
        start = self.rank * self.num_samples_per_rank
        end = start + self.num_samples_per_rank
        return iter(indices[start:end])

    def __len__(self):
        return self.num_samples_per_rank

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_iso_loader(dataset_root, batchsize, trainsize,dist = False, texture_type = None, ds_type='train'):
    if ds_type == 'train':   
        train_dataset_list = ["DAVIS2016",'DAVSOD','FBMS']
        train_dataset = SalVideoISOTrainDataset(dataset_root,train_dataset_list,trainsize)
        data_loader = data.DataLoader(dataset=train_dataset,
                                      batch_size=batchsize,
                                      pin_memory=True,
                                      drop_last=True,
                                      num_workers=4,sampler= BalancedDistributedVideoSampler(train_dataset,shuffle=False) if dist else None)
        return data_loader
    
    elif ds_type == 'val':   
        val_dataset_list = ["DAVIS",  "DAVSOD", "SegTrack-V2"]
        val_dataset = SalVideoISOValDataset(dataset_root,val_dataset_list,trainsize)
        data_loader = data.DataLoader(dataset=val_dataset,
                                      batch_size=batchsize,
                                      pin_memory=True,
                                      num_workers=4,sampler= data.distributed.DistributedSampler(val_dataset, shuffle=False) if dist else None)
        return data_loader
    elif ds_type == 'test':

        raise NotImplementedError()
        #dataset = SalObjTestDataset(dataset_root,trainsize) 
        # data_loader = data.DataLoader(dataset=dataset,
        #                               batch_size=1,
        #                               num_workers=4)
        data_loader = None
        return data_loader
    else:
        raise NotImplementedError("no such dataset")


def get_v_loader(dataset_root, batchsize, trainsize,dist = False, texture_type = None, ds_type='train'):
    if ds_type == 'train': 
        train_dataset_list = ["DAVIS2016",'DAVSOD','FBMS']  
        train_dataset = SalVideoPairTrainDataset(dataset_root,train_dataset_list,trainsize)
        data_loader = data.DataLoader(dataset=train_dataset,
                                      batch_size=batchsize,
                                      pin_memory=True,
                                      drop_last=True,
                                      num_workers=4,sampler= data.distributed.DistributedSampler(train_dataset) if dist else None)
        
        return data_loader
    
    elif ds_type == 'val':
        val_dataset_list = ["DAVIS",  "DAVSOD", "SegTrack-V2"]
        val_dataset =SalVideoPairValDataset(dataset_root,val_dataset_list,trainsize)
        data_loader = data.DataLoader(dataset=val_dataset,
                                      batch_size=batchsize,
                                      pin_memory=True,
                                      drop_last=True,
                                      num_workers=4,sampler= data.distributed.DistributedSampler(val_dataset) if dist else None)
        return data_loader
    elif ds_type == 'test':

        raise NotImplementedError()
        #dataset = SalObjTestDataset(dataset_root,trainsize) 
        # data_loader = data.DataLoader(dataset=dataset,
        #                               batch_size=1,
        #                               num_workers=4)
        data_loader = None
        return data_loader
    else:
        raise NotImplementedError("no such dataset")


if __name__ == "__main__":
    pass
    