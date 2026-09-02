import argparse
import yaml
import os
from omegaconf import OmegaConf



def update_config_by_args(config, args):
    
    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False

    # merge from specific arguments
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args('data_path'):
        config.DATA.DATA_PATH = args.data_path
    if _check_args('pretrained'):
        config.MODEL.PRETRAINED = args.pretrained
    if _check_args('resume'):
        config.MODEL.RESUME = args.resume
    if _check_args('amp_opt_level'):
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")
        if args.amp_opt_level == 'O0':
            config.AMP_ENABLE = False
    if _check_args('disable_amp'):
        config.AMP_ENABLE = False
    if _check_args('tag'):
        config.TAG = args.tag
    
    if _check_args('mfusion'):
        config.MODEL.MFUSION = args.mfusion
    # [SimMIM]
    if _check_args('enable_amp'):
        config.ENABLE_AMP = args.enable_amp

    # set local rank for distributed training
    config.LOCAL_RANK = args.local_rank



def parse_train_option():
    parser = argparse.ArgumentParser('training script', add_help=False)
    parser.add_argument('--backbone', type=str, help='path to config file')
    parser.add_argument('--mfusion', type=str, default= 'HAIM' )    
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--warmup_epoch', type=int, default=25)
    parser.add_argument('--max_epoch', type=int, default=100, help='max epoch number')
    parser.add_argument('--pretrain_batch', type=int, default=16, help='training batch size')
    parser.add_argument('--finetune_batch', type=int, default=16, help='training batch size')
    parser.add_argument('--optim_preset', default='step_adam',
                        choices=['step_adam', 'step_adam_simple', 'cosine_adamw', 'plateau_adam_simple'],
                        help='optimizer+scheduler: step_adam (分层lr) / step_adam_simple (统一lr) / cosine_adamw')
    parser.add_argument('--gpu_id', type=str, default='1', help='select gpu id')
    parser.add_argument('--train_root', type=str, default='./dataset/ISOD_dataset/train/', help='the train images root')
    parser.add_argument('--val_root', type=str, default='./dataset/ISOD_dataset/val/', help='the val images root')


    parser.add_argument('--log_path', type=str, default='./log/', help='the path to save models and logs')

    parser.add_argument('--pretrain_size', type=int,default=0)
    parser.add_argument('--finetune_size', type=int,default=0)

    parser.add_argument('--texture',choices=[None,'/namlab20/','/namlab25/','/namlab30/','/namlab40/','/namlab50/','/namlab60/','/bound/','/teed/','/cats/','/bound/'])
    parser.add_argument('--test_model',type=str,required=False)
    parser.add_argument('--save_path', type=str, default='./save/', help='the path to save images')
    
    parser.add_argument('--test_now', action='store_true')
    parser.add_argument('--test_batch',type=int,default=16)
    parser.add_argument('--test_path',type=str,default='./dataset/ISOD_dataset/test/',help='test dataset path')
    parser.add_argument('--save_result',type=bool,default=True)
    
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')

    parser.add_argument('--task', choices=['ISOD', 'COD'], default='ISOD', help='task of experiment')
    parser.add_argument('--tag', help='tag of experiment')

    # loss 消融
    parser.add_argument('--loss', default='ablation2',
                       choices=['ablation1', 'ablation2', 'full', 'gt_only', 'edge_only'],
                       help='loss preset for ablation')
    parser.add_argument('--bounds_fn', default='raw', choices=['raw', 'blur3', 'blur5'],
                       help='bounds preprocessing: raw / blur3 / blur5')
    parser.add_argument('--edge_weight', type=float, default=1.0, help='weight for edge loss')
    parser.add_argument('--use_edge2', action='store_true', help='add BlurSupervision on edge2')
    parser.add_argument('--alpha1', type=float, default=2.0,
                        help='StructureLoss mask weight coef (full preset)')
    parser.add_argument('--alpha2', type=float, default=0.3,
                        help='StructureLoss pred weight coef (full preset)')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='StructureLoss baseline weight (full preset)')
    parser.add_argument('--ema_decay', type=float, default=0, help='EMA decay rate, 0 to disable')

    parser.add_argument('--moe_temp_start', type=float, default=0.25,
                        help='SoftMoE 路由初始温度（较小=路由更尖）')
    parser.add_argument('--moe_temp_end', type=float, default=1.0,
                        help='SoftMoE 路由结束温度（训练末期线性升到该值，softmax 更平滑）')

    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1,
                       help='local rank for DistributedDataParallel')

    parser.add_argument('--seed',
                    default=None,
                    type=int,
                    help='seed for initializing training. ')


    args = parser.parse_args()

    config = OmegaConf.load(os.path.join("./config","default","config.yaml"))
    if args.backbone == 'segswin-base':
        cfg2 = OmegaConf.load("./pretrained/configs/segswin/swin_base_patch4_window12_384_22kto1k_finetune.yaml")
    elif args.backbone == 'convnextv2-base':
        cfg2 = OmegaConf.load("./pretrained/configs/convnextv2/base.yaml")
    elif args.backbone == 'hiera-base':
        cfg2 = OmegaConf.load("./pretrained/configs/hiera/base.yaml")
    elif args.backbone == 'pvtv2_b4':
        cfg2 = OmegaConf.load("./pretrained/configs/pvtv2/b4.yaml")
    elif args.backbone == 'pvtv2_b5':
        cfg2 = OmegaConf.load("./pretrained/configs/pvtv2/b5.yaml")
    else:
        raise ValueError(f"Unsupported backbone: {args.backbone}")
    if args.task == 'ISOD' or args.task == 'COD':
        cfg3 = OmegaConf.load(os.path.join("./config","default",f"{args.task}.yaml"))
    else:
        cfg3 = OmegaConf.create({})
    config.BACKBONE = cfg2.MODEL
    config.DATA.PRETRAIN_SIZE = cfg2.DATA.PRETRAIN_SIZE
    if args.pretrain_size > 0:
        config.DATA.PRETRAIN_SIZE = args.pretrain_size
    if args.finetune_size > 0:
        config.DATA.FINETUNE_SIZE = args.finetune_size
    config = OmegaConf.merge(config,cfg3)
    
    update_config_by_args(config,args)
    

    return args, config

def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    new_cfg = yaml_cfg
    

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    
    print('=> merge config from {}'.format(cfg_file))
    
    #config.merge_from_other_cfg(cfg_other)
    
    #config.merge_from_file(cfg_file)
    config.freeze()

def parse_test_option():
    parser = argparse.ArgumentParser('testing script', add_help=False)
    parser.add_argument('--test_model',type=str,required=True)
    parser.add_argument('--gpu_id',type=str,default="0")
    parser.add_argument('--metrics_only', action='store_true', help='仅计算 FPS/Params/FLOPS，不进行完整评估')
    parser.add_argument('--no_metrics', action='store_true', help='跳过 FPS/Params/FLOPS 计算')
    parser.add_argument('--prune', action='store_true', help='使用剪枝模型（移除边缘检测和深监督，仅保留主输出）')
    parser.add_argument('--img_size',type=int,default=384)
    parser.add_argument('--datasets', nargs='+', default=None, help='只测指定数据集，默认用 config 的 TESTSET')
    args = parser.parse_args()
    
    config = OmegaConf.load(os.path.join(args.test_model,'config.yaml'))
    if args.datasets:
        config.DATA.TESTSET = list(args.datasets)
    
    return args, config


def save_eval_result_csv(test_model, eval_result, datasets):
    """写入 eval_result.csv；多进程按数据集拆开测时用文件锁合并行。"""
    import fcntl
    import pandas as pd

    csv_path = os.path.join(test_model, "eval_result.csv")
    lock_path = csv_path + ".lock"
    df_new = pd.DataFrame(eval_result, index=list(datasets))
    preferred = None
    cfg_path = os.path.join(test_model, "config.yaml")
    if os.path.isfile(cfg_path):
        preferred = list(OmegaConf.load(cfg_path).DATA.TESTSET)

    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        if os.path.isfile(csv_path):
            df_old = pd.read_csv(csv_path, index_col=0)
            keep = [i for i in df_old.index if i not in df_new.index]
            df_old = df_old.loc[keep] if keep else df_old.iloc[0:0]
            df = pd.concat([df_old, df_new])
        else:
            df = df_new
        if preferred:
            ordered = [i for i in preferred if i in df.index]
            rest = [i for i in df.index if i not in ordered]
            df = df.loc[ordered + rest]
        df.to_csv(csv_path)
