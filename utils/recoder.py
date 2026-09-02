import os
import csv
import json
import subprocess
import logging
from datetime import datetime
import torch
from omegaconf import OmegaConf
class Recorder:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.rank = getattr(args, 'local_rank', 0)
        self.is_main = (self.rank == 0)

        self.best_mae = 1.0
        self.best_epoch = -1
        self.cur_epoch = -1
        #self.log_freq = config.PRINT_FREQ
        #self.save_freq = config.SAVE_FREQ

        if self.is_main:
            self.log_name = self._make_log_name(args.backbone)
            self.log_path = os.path.join(args.log_path, args.tag if args.tag else self.log_name)

            self.ckpt_path = os.path.join(self.log_path, 'ckpt')
            self.record_path = os.path.join(self.log_path, 'record')
            self._record_rows = []  # 累积所有 record 行，支持后续 epoch 新增 key
            

            self._init_dirs()
            self._init_logger()
            
            self._save_args_config()
            
            
        else:
            self.log = lambda *args, **kwargs: None
            self.save_ckpt = lambda *args, **kwargs: None
            self.update_metrics = lambda *args, **kwargs: None
            self.get_log_path = lambda *args, **kwargs: None
            self.record_metrics = lambda *args, **kwargs: None
            self._remind_commit_proj = lambda *args, **kwargs: None

    def _make_log_name(self, backbone):
        prefix = datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + '-GSformer-'
        return prefix + backbone

    def _init_dirs(self):
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.ckpt_path, exist_ok=True)
        os.makedirs(self.record_path, exist_ok=True)

    def get_log_path(self):
        return self.log_path

    def _init_logger(self):
        log_file = os.path.join(self.log_path, 'log.txt')
        logging.basicConfig(
            filename=log_file,
            format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
            level=logging.INFO,
            filemode='a',
            datefmt='%Y-%m-%d %I:%M:%S %p'
        )

    def log(self, msg):
        logging.info(str(msg))

    def save_ckpt(self, model, name):
        torch.save(model.state_dict(), os.path.join(self.ckpt_path, name))

    def set_epoch(self, epoch):
        self.cur_epoch = epoch

    def update_metrics(self, data):
        self._remind_commit_proj()
        self.record_metrics(data)
        if not data or 'mae' not in data:
            return False
        if self.best_mae > data['mae']:
            self.best_mae = data['mae']
            self.best_epoch = data.get('epoch', self.cur_epoch)
            self.log('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(data['epoch'], data['mae'], self.best_epoch, self.best_mae))
            return True
        self.log('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(data['epoch'], data['mae'], self.best_epoch, self.best_mae))
        return False

    def record_metrics(self, metrics_dict, epoch=None, filename='record.csv'):
        """记录 epoch 迭代时的指标变化，实时保存到 CSV，便于后续查看和绘制可视化图表。

        同一 epoch 多次调用时（如 train 后记录 loss、val 后记录 mae/miou）会合并到同一行。
        """
        if not self.is_main:
            return
        epoch_val = epoch if epoch is not None else metrics_dict.get('epoch', self.cur_epoch)
        row = {'epoch': epoch_val, **{k: v for k, v in metrics_dict.items() if k != 'epoch'}}
        existing = next((r for r in self._record_rows if r.get('epoch') == epoch_val), None)
        if existing:
            existing.update(row)
        else:
            self._record_rows.append(row)
        self._record_rows.sort(key=lambda r: r.get('epoch', -1))
        all_keys = ['epoch'] + sorted({k for r in self._record_rows for k in r if k != 'epoch'})
        filepath = os.path.join(self.record_path, filename)
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
            w.writeheader()
            for r in self._record_rows:
                w.writerow({k: r.get(k, '') for k in all_keys})

    def _save_args_config(self):
        with open(os.path.join(self.log_path, "args.json"), mode="w") as f:
            json.dump(self.args.__dict__, f, indent=4)

        OmegaConf.save(self.config, os.path.join(self.log_path, "config.yaml"))

    def _remind_commit_proj(self):
        """
        check git status, if there are modified files, remind the user to save the project files to the log directory.
        """
        try:
            result = subprocess.run(["git", "status"], capture_output=True, text=True, timeout=5)
            status_output = result.stdout or ""
            status_code = result.returncode
        except Exception:
            status_output = ""
            status_code = -1
        if status_code != 0:
            print("git status failed, please check the git status manually.")
            return
        if "modified" in status_output:
            print("find modified files, please commit the changes to the remote repository.")
            return
        print("no modified files, continue.")