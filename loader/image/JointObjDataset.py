import os
import math
import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import WeightedRandomSampler, ConcatDataset
from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data.sampler import Sampler

# 假设你保留了原有的自定义增强库
from loader.custom_transforms import random_flip, random_crop, random_rotation, image_suffix, color_enhance

class DistributedWeightedRandomSampler(Sampler):
    """
    针对 DDP (分布式训练) 的加权随机采样器 (保持不变)
    """
    def __init__(self, dataset, weights, num_replicas=None, rank=None, 
                 replacement=True, seed=0, drop_last=False):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.seed = seed
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.replacement = replacement

        total_size_requested = len(self.dataset)
        
        if self.drop_last and total_size_requested % self.num_replicas != 0:
            self.num_samples = math.ceil((total_size_requested - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(total_size_requested / self.num_replicas)
            
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights, self.total_size, self.replacement, generator=g
        ).tolist()
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class SingleTaskDataset(data.Dataset):
    def __init__(self, dataset_root, trainsize, task_id, hard_aug=False):
        self.image_root = os.path.join(dataset_root, 'RGB/')
        self.gt_root = os.path.join(dataset_root, 'GT/')
        
        self.trainsize = trainsize
        self.task_id = task_id  # 0: SOD, 1: COD, 2: USOD
        self.hard_aug = hard_aug

        self.images = sorted([self.image_root + f for f in os.listdir(self.image_root) if image_suffix(f)])
        self.gts = sorted([self.gt_root + f for f in os.listdir(self.gt_root) if image_suffix(f)])
        self.size = len(self.images)

        self.rgb_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        
        # ================= 核心优化 1：动态读取模式 =================
        if self.task_id == 2:
            # USOD 数据集：包含红绿颜色的 RGB 图
            gt = Image.open(self.gts[index]).convert('RGB')
        else:
            # SOD / COD 数据集：单通道灰度图
            gt = Image.open(self.gts[index]).convert('L')

        # 1. 空间增强 (PIL 库的增强方法通常兼容 'L' 和 'RGB' 模式)
        image, gt, _ = random_flip(image, gt, gt) 
        image, gt, _ = random_crop(image, gt, gt)
        image, gt, _ = random_rotation(image, gt, gt)
        
        # 2. 颜色增强 (仅对 image)
        image = color_enhance(image)
        
        # 3. 转换为 Tensor
        image = self.rgb_transform(image)
        gt_tensor = self.gt_transform(gt) 
        # 注意：
        # - 如果是灰度图 (task_id 0或1)，gt_tensor 的 shape 是 (1, H, W)
        # - 如果是RGB图 (task_id 2)，gt_tensor 的 shape 是 (3, H, W)

        # ================= 核心优化 2：构建统一双通道掩码 =================
        # 使用 > 0.5 是为了防止 JPEG 压缩伪影或 Resize 插值导致的浮点数不精确
        
        if self.task_id == 0:
            # SOD 任务：提取灰度图为 SOD mask，COD mask 全零
            sod_mask = (gt_tensor > 0.5).float()              # Shape: (1, H, W)
            cod_mask = torch.zeros_like(sod_mask)             # Shape: (1, H, W)
            
        elif self.task_id == 1:
            # COD 任务：提取灰度图为 COD mask，SOD mask 全零
            sod_mask = torch.zeros_like(gt_tensor)            # Shape: (1, H, W)
            cod_mask = (gt_tensor > 0.5).float()              # Shape: (1, H, W)
            
        elif self.task_id == 2:
            # USOD 任务：红色通道(0)提取为 SOD，绿色通道(1)提取为 COD
            # 使用切片 [0:1, :, :] 可以保持维度为 (1, H, W) 而不是变成 (H, W)
            sod_mask = (gt_tensor[0:1, :, :] > 0.5).float()   # Shape: (1, H, W)
            cod_mask = (gt_tensor[1:2, :, :] > 0.5).float()   # Shape: (1, H, W)
            
        else:
            raise ValueError(f"Unknown task_id: {self.task_id}")

        # 沿通道维度拼接
        gt_combined = torch.cat([sod_mask, cod_mask], dim=0)  # Shape: (2, H, W)

        return image, gt_combined, self.task_id

    def __len__(self):
        return self.size

def get_joint_train_loader(data_cfg, batchsize, dist_mode=False, num_workers=8):
    """
    完美适配多文件夹结构的 DataLoader 生成器
    """
    data_root = data_cfg.DATA_ROOT # 例如: './dataset'
    
    # 定义你的任务映射字典 (可以根据需要增减)
    task_mapping = {
        'ISOD_dataset': 0,
        'COD_dataset': 1,
        'USOD_dataset': 2
    }
    
    datasets_list = []
    sample_types = [] # 记录所有样本的 task_id，用于计算采样权重
    
    # 1. 遍历并构建数据集列表
    for task_folder, task_id in task_mapping.items():
        task_train_dir = os.path.join(data_root, task_folder, 'train')
        
        if not os.path.exists(task_train_dir):
            continue
        
        if task_id == 0:
            ds = SingleTaskDataset(
                dataset_root=os.path.join(task_train_dir, 'DUTS-TR'), 
                trainsize=data_cfg.FINETUNE_SIZE,
                task_id=task_id
            )
            datasets_list.append(ds)
            sample_types.extend([task_id] * len(ds))
        else:
        # 遍历具体的 dataset_name (例如 DUTS-TR, CAMO 等)
            for ds_name in os.listdir(task_train_dir):
                ds_path = os.path.join(task_train_dir, ds_name)
                
                # 必须是文件夹
                if os.path.isdir(ds_path):
                    # 实例化单个数据集
                    ds = SingleTaskDataset(
                        dataset_root=ds_path, 
                        trainsize=data_cfg.FINETUNE_SIZE,
                        task_id=task_id
                    )
                    datasets_list.append(ds)
                    
                    # 为该数据集的每一张图片记录类别，用于稍后计算权重
                    sample_types.extend([task_id] * len(ds))

    # 使用 PyTorch 原生的 ConcatDataset 拼接所有数据集
    joint_dataset = ConcatDataset(datasets_list)
    print(f"\nTotal combined training samples: {len(joint_dataset)}")

    # 2. 统一计算采样权重 (修复了原代码中 dist 分支缺少 weights 的 Bug)
    class_counts = {}
    for t in sample_types:
        class_counts[t] = class_counts.get(t, 0) + 1
        
    print(f"Sample distribution per Task ID: {class_counts}")
    
    # 计算每种类别的权重 (数量越少，权重越大)
    class_weights = {}
    for task_id, count in class_counts.items():
        class_weights[task_id] = 1.0 / (count + 1e-6)
        
    # 展开为每个样本的权重序列
    sample_weights = [class_weights[t] for t in sample_types]

    # 3. 构建 Sampler 并生成 Loader
    if dist_mode:
        sampler = DistributedWeightedRandomSampler(
            dataset=joint_dataset,
            weights=sample_weights,
            replacement=True
        )
    else:
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights), # 每个 epoch 抽取的总图片数
            replacement=True
        )

    # 注意：使用了 sampler，必须 shuffle=False
    data_loader = data.DataLoader(
        dataset=joint_dataset,
        batch_size=batchsize,
        pin_memory=True,
        drop_last=True,
        num_workers=num_workers,
        sampler=sampler 
    )
    
    return data_loader


class ValSingleTaskDataset(data.Dataset):
    """
    验证集专属 Dataset：剥离所有数据增强，仅做验证必要的预处理。
    """
    def __init__(self, dataset_root, valsize, task_id):
        self.image_root = os.path.join(dataset_root, 'RGB/')
        self.gt_root = os.path.join(dataset_root, 'GT/')
        
        self.valsize = valsize
        self.task_id = task_id  # 0: SOD, 1: COD, 2: USOD

        self.images = sorted([self.image_root + f for f in os.listdir(self.image_root) if image_suffix(f)])
        self.gts = sorted([self.gt_root + f for f in os.listdir(self.gt_root) if image_suffix(f)])
        self.size = len(self.images)

        # 仅保留基础 Resize 和标准化
        self.rgb_transform = transforms.Compose([
            transforms.Resize((self.valsize, self.valsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.valsize, self.valsize), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        
        # 根据 task_id 动态读取模式
        if self.task_id == 2:
            gt = Image.open(self.gts[index]).convert('RGB')
        else:
            gt = Image.open(self.gts[index]).convert('L')
        
        # 1. 基础转换
        image = self.rgb_transform(image)
        gt_tensor = self.gt_transform(gt) 
        
        if gt_tensor.max() > 1.0:
            gt_tensor = gt_tensor / 255.0

        # 2. 构建统一双通道掩码 (与 Train 保持绝对一致)
        if self.task_id == 0:
            sod_mask = (gt_tensor > 0.5).float()
            cod_mask = torch.zeros_like(sod_mask)
        elif self.task_id == 1:
            sod_mask = torch.zeros_like(gt_tensor)
            cod_mask = (gt_tensor > 0.5).float()
        elif self.task_id == 2:
            sod_mask = (gt_tensor[0:1, :, :] > 0.5).float()
            cod_mask = (gt_tensor[1:2, :, :] > 0.5).float()
            
        gt_combined = torch.cat([sod_mask, cod_mask], dim=0) # Shape: (2, H, W)

        return image, gt_combined, self.task_id

    def __len__(self):
        return self.size

def get_joint_val_loaders(data_cfg, batchsize, dist_mode=False, num_workers=4):
    """
    针对联合训练构建的验证集加载器。
    返回一个字典，包含独立任务的 DataLoader。
    """
    data_root = data_cfg.DATA_ROOT 
    
    # 定义你的任务映射和对应的任务名称
    task_mapping = {
        'ISOD_dataset': {'id': 0, 'name': 'SOD'},
        'COD_dataset': {'id': 1, 'name': 'COD'},
        'USOD_dataset': {'id': 2, 'name': 'USOD'}
    }
    
    # 用于存储最终返回的多个 Dataloader
    val_loaders = {}
    
    for task_folder, task_info in task_mapping.items():
        task_id = task_info['id']
        task_name = task_info['name']
        
        # 注意：这里去 val 文件夹下找数据 (如果你的目录叫 test，请改成 'test')
        task_val_dir = os.path.join(data_root, task_folder, 'val')
        
        if not os.path.exists(task_val_dir):
            print(f"Warning: Validation directory not found -> {task_val_dir}")
            continue
            
        datasets_list = []
        
        # 遍历该任务下的具体子数据集 (例如 val/DUTS-TE, val/CAMO-Test 等)
        for ds_name in os.listdir(task_val_dir):
            ds_path = os.path.join(task_val_dir, ds_name)
            
            if os.path.isdir(ds_path):
                ds = ValSingleTaskDataset(
                    dataset_root=ds_path, 
                    valsize=data_cfg.FINETUNE_SIZE, # 验证集尺寸通常与训练一致，或设为 384 等测试尺寸
                    task_id=task_id
                )
                datasets_list.append(ds)

        if len(datasets_list) == 0:
            continue
            
        # 拼接当前任务下的所有验证子集
        task_joint_dataset = ConcatDataset(datasets_list)
        print(f"Loaded {len(task_joint_dataset)} validation samples for {task_name}")

        # 配置 Sampler
        if dist_mode:
            # 验证集使用标准的 DistributedSampler，不打乱顺序
            sampler = DistributedSampler(task_joint_dataset, shuffle=False)
        else:
            sampler = None

        # 构建 DataLoader
        loader = data.DataLoader(
            dataset=task_joint_dataset,
            batch_size=batchsize,
            shuffle=False, # 验证集坚决不 shuffle
            pin_memory=True,
            drop_last=False, # 验证集坚决不丢弃尾部数据
            num_workers=num_workers,
            sampler=sampler
        )
        
        # 存入字典
        val_loaders[task_name] = loader
        
    return val_loaders


class TestSingleTaskDataset(data.Dataset):
    def __init__(self, dataset_root, testsize, task_id):
        self.image_root = os.path.join(dataset_root, 'RGB/')
        self.gt_root = os.path.join(dataset_root, 'GT/')
        
        self.testsize = testsize
        self.task_id = task_id
        self.name = dataset_root.split('/')[-1]
        
        self.images = sorted([self.image_root + f for f in os.listdir(self.image_root) if image_suffix(f)])
        self.gts = sorted([self.gt_root + f for f in os.listdir(self.gt_root) if image_suffix(f)])
        self.size = len(self.images)
        
    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        gt = Image.open(self.gts[index]).convert('L')
        sz = gt.size
        name = self.images[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        image = self.rgb_transform(image)
        gt = self.gt_transform(gt)
        
        return image, gt, sz, name, self.task_id
        
    def __len__(self):
        return self.size
    
    def rgb_transform(self, image):
        return transforms.Compose([transforms.Resize((self.testsize, self.testsize)), transforms.ToTensor()])(image)
    
    def gt_transform(self, gt):
        return transforms.Compose([transforms.Resize((self.testsize, self.testsize), interpolation=transforms.InterpolationMode.NEAREST), transforms.ToTensor()])(gt)

def get_joint_test_loaders(data_cfg, batchsize, dist_mode=False, num_workers=4):
    data_root = data_cfg.DATA_ROOT 
    task_mapping = {
        #'ISOD_dataset': {'id': 0, 'name': 'SOD'},
        #'COD_dataset': {'id': 1, 'name': 'COD'},
        'USOD_dataset': {'id': 2, 'name': 'USOD'}
    }
    test_loaders = {}
    for task_folder, task_info in task_mapping.items():
        task_id = task_info['id']
        task_name = task_info['name']
        task_test_dir = os.path.join(data_root, task_folder, 'test')
        if not os.path.exists(task_test_dir):
            print(f"Warning: Test directory not found -> {task_test_dir}")
            continue
        datasets_list = []
        for ds_name in os.listdir(task_test_dir):
            ds_path = os.path.join(task_test_dir, ds_name)
            if os.path.isdir(ds_path):
                ds = TestSingleTaskDataset(
                    dataset_root=ds_path, 
                    testsize=data_cfg.FINETUNE_SIZE,
                    task_id=task_id
                )
                if dist_mode:
                    sampler = DistributedSampler(dl, shuffle=False)
                else:
                    sampler = None
                dl = data.DataLoader(ds, batch_size=batchsize,\
                    shuffle=False, pin_memory=True, drop_last=False,\
                    num_workers=num_workers, sampler=sampler)

                if test_loaders.get(task_name) is None:
                    test_loaders[task_name] = []
                test_loaders[task_name].append(dl)
    return test_loaders


if __name__ == "__main__":
    from omegaconf import OmegaConf
    data_cfg = OmegaConf.create({
        'DATA_ROOT': './dataset',
        'FINETUNE_SIZE': 384
    })
    data_loader = get_joint_train_loader(data_cfg, batchsize=8, dist_mode=False, num_workers=0)
    for images, gt_combined, task_ids in data_loader:
        print(images.shape, gt_combined.shape, task_ids)
        break
    
    val_loaders = get_joint_val_loaders(data_cfg, batchsize=8, dist_mode=False, num_workers=0)
    for task_name, loader in val_loaders.items():
        for images, gt_combined, task_ids in loader:
            print(images.shape, gt_combined.shape, task_ids)
            break
        
    test_loaders = get_joint_test_loaders(data_cfg, batchsize=1, dist_mode=False, num_workers=0)
    for task_name, loader in test_loaders.items():
        print(task_name)
        for dl in loader:
            print(len(dl))
            for images, gt, sz, name, task_ids in dl:
                print(images.shape, gt.shape, sz, name, task_ids)
                break
            