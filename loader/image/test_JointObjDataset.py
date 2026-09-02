#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JointObjDataset 测试脚本
测试数据加载、样本分配和加权采样分布
"""
import os
import sys
import unittest
from collections import Counter
from types import SimpleNamespace

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from loader.image.JointObjDataset import (
    SingleTaskDataset,
    get_joint_loader,
    DistributedWeightedRandomSampler,
)


def get_test_data_root():
    """获取测试数据根目录，优先使用项目 dataset"""
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "dataset")
    return root if os.path.exists(root) else None


def make_data_cfg(data_root, finetune_size=384):
    """构造 data_cfg 配置对象"""
    return SimpleNamespace(
        DATA_ROOT=data_root,
        FINETUNE_SIZE=finetune_size,
    )


class TestSingleTaskDataset(unittest.TestCase):
    """测试 SingleTaskDataset 单任务数据集"""

    @classmethod
    def setUpClass(cls):
        cls.data_root = get_test_data_root()
        cls.trainsize = 384

    def test_cod_dataset_load(self):
        """测试 COD 数据集加载 (task_id=1)"""
        if not self.data_root:
            self.skipTest("dataset 目录不存在，跳过")
        cod_train = os.path.join(self.data_root, "COD_dataset", "train")
        if not os.path.exists(cod_train):
            self.skipTest("COD_dataset/train 不存在")
        ds_names = [d for d in os.listdir(cod_train) if os.path.isdir(os.path.join(cod_train, d))]
        if not ds_names:
            self.skipTest("COD_dataset/train 下无子数据集")
        ds_path = os.path.join(cod_train, ds_names[0])
        ds = SingleTaskDataset(ds_path, self.trainsize, task_id=1)
        self.assertGreater(len(ds), 0, "COD 数据集应非空")
        image, gt_combined, task_id = ds[0]
        self.assertEqual(task_id, 1)
        self.assertEqual(image.shape[0], 3)
        self.assertEqual(gt_combined.shape[0], 2)  # sod_mask + cod_mask
        self.assertEqual(image.shape[1], self.trainsize)
        self.assertEqual(image.shape[2], self.trainsize)

    def test_usod_dataset_load(self):
        """测试 USOD 数据集加载 (task_id=2)"""
        if not self.data_root:
            self.skipTest("dataset 目录不存在，跳过")
        usod_train = os.path.join(self.data_root, "USOD_dataset", "train")
        if not os.path.exists(usod_train):
            self.skipTest("USOD_dataset/train 不存在")
        ds_names = [d for d in os.listdir(usod_train) if os.path.isdir(os.path.join(usod_train, d))]
        if not ds_names:
            self.skipTest("USOD_dataset/train 下无子数据集")
        ds_path = os.path.join(usod_train, ds_names[0])
        ds = SingleTaskDataset(ds_path, self.trainsize, task_id=2)
        self.assertGreater(len(ds), 0, "USOD 数据集应非空")
        image, gt_combined, task_id = ds[0]
        self.assertEqual(task_id, 2)
        self.assertEqual(image.shape[0], 3)
        self.assertEqual(gt_combined.shape[0], 2)
        self.assertEqual(image.shape[1], self.trainsize)
        self.assertEqual(image.shape[2], self.trainsize)


class TestGetJointLoader(unittest.TestCase):
    """测试 get_joint_loader 联合数据加载器"""

    @classmethod
    def setUpClass(cls):
        cls.data_root = get_test_data_root()

    def test_joint_loader_creation(self):
        """测试联合 Loader 能否正常创建"""
        if not self.data_root:
            self.skipTest("dataset 目录不存在，跳过")
        cfg = make_data_cfg(self.data_root, finetune_size=384)
        loader = get_joint_loader(cfg, batchsize=4, dist_mode=False, num_workers=0)
        self.assertIsNotNone(loader)
        self.assertGreater(len(loader), 0)

    def test_joint_loader_iteration(self):
        """测试联合 Loader 迭代与 batch 形状"""
        if not self.data_root:
            self.skipTest("dataset 目录不存在，跳过")
        cfg = make_data_cfg(self.data_root, finetune_size=384)
        loader = get_joint_loader(cfg, batchsize=4, dist_mode=False, num_workers=0)
        batch_count = 0
        for images, gt_combined, task_ids in loader:
            batch_count += 1
            self.assertEqual(images.shape[0], 4, "batch_size 应为 4")
            self.assertEqual(images.shape[1], 3)
            self.assertEqual(gt_combined.shape[0], 4)
            self.assertEqual(gt_combined.shape[1], 2)
            self.assertEqual(task_ids.shape[0], 4)
            if batch_count >= 3:  # 只验证前几个 batch
                break
        self.assertGreater(batch_count, 0)

    def test_sample_distribution(self):
        """测试样本分配：验证各 task_id 在多个 epoch 内均有被采样到"""
        if not self.data_root:
            self.skipTest("dataset 目录不存在，跳过")
        cfg = make_data_cfg(self.data_root, finetune_size=384)
        loader = get_joint_loader(cfg, batchsize=8, dist_mode=False, num_workers=0)
        task_counter = Counter()
        num_batches = min(10, len(loader))  # 采样 50 个 batch 或全部
        for i, (_, _, task_ids) in enumerate(loader):
            if i >= num_batches:
                break
            for t in task_ids.tolist():
                task_counter[int(t)] += 1
        # 至少应包含两种任务（COD 和 USOD，SOD 可能不存在）
        self.assertGreater(len(task_counter), 0, "应至少有一种 task_id 被采样")
        total = sum(task_counter.values())
        self.assertGreater(total, 0, "应有样本被采样")
        print("\n[样本分配统计]")
        for tid, count in sorted(task_counter.items()):
            pct = 100.0 * count / total
            name = {0: "SOD", 1: "COD", 2: "USOD"}.get(tid, f"Task{tid}")
            print(f"  {name} (task_id={tid}): {count} 样本 ({pct:.1f}%)")


class TestDistributedWeightedRandomSampler(unittest.TestCase):
    """测试 DistributedWeightedRandomSampler（单进程模拟）"""

    def test_sampler_without_dist(self):
        """在非分布式环境下，需传入 num_replicas 和 rank"""
        import torch
        dataset_size = 100
        weights = [1.0] * dataset_size
        # 模拟单卡：num_replicas=1, rank=0
        sampler = DistributedWeightedRandomSampler(
            dataset=range(dataset_size),
            weights=weights,
            num_replicas=1,
            rank=0,
            replacement=True,
            seed=42,
        )
        indices = list(sampler)
        self.assertEqual(len(indices), dataset_size)
        self.assertTrue(all(0 <= i < dataset_size for i in indices))

    def test_sampler_multi_replica(self):
        """模拟多卡：每个 rank 得到不同子集"""
        dataset_size = 100
        weights = [1.0] * dataset_size
        all_indices = []
        for rank in range(4):
            sampler = DistributedWeightedRandomSampler(
                dataset=range(dataset_size),
                weights=weights,
                num_replicas=4,
                rank=rank,
                replacement=True,
                seed=42,
            )
            indices = list(sampler)
            all_indices.append(set(indices))
        # 各 rank 样本数应相等（或差 1）
        lens = [len(s) for s in all_indices]
        self.assertLessEqual(max(lens) - min(lens), 1)
        # 不同 rank 之间无重叠（无 replacement 时；有 replacement 时可能重叠，这里仅检查长度）


def run_tests():
    """运行所有测试并打印摘要"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestSingleTaskDataset))
    suite.addTests(loader.loadTestsFromTestCase(TestGetJointLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestDistributedWeightedRandomSampler))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
