import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.distributed as dist



def adjust_lr(optimizer1,  init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer1.param_groups:
        param_group['lr'] = decay*init_lr
        lr=param_group['lr']
    return lr

def list_files(path):
    files = []
    for file in os.listdir(path):
        if '.' in file and file[0] != '.':
            files.append(os.path.join(path,file))
    return files

def list_project_files(project_dirs):
    files = {}
    for dir in project_dirs:
        files[dir] = list_files(dir)
    return files

def save_project(dst_dir,project_dirs):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    files = list_project_files(project_dirs)
    for dirname in files.keys():
        dirpath = os.path.join(dst_dir,dirname)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        for file in files[dirname]:
            shutil.copy(file,dirpath)

def search_text(fname,keyword):
    text = open(fname,'r').read()
    return text.find(keyword) != -1    


def remove_temp_log_file():
    log_dir = './log/'
    for f in os.listdir(log_dir):
        log_path = log_dir+f
        
        if log_path.endswith("384"):
            shutil.rmtree(log_path)
        elif not os.path.exists(log_path+'/log.txt') or search_text(log_path+'/log.txt','nan'):
            shutil.rmtree(log_path)
        # elif not search_text(log_path+'/log.txt','Epoch:40'):
        #     shutil.rmtree(log_path)

# set loss function
def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    delta = fin - init
    annealed = init + delta * step / annealing_steps
    return annealed

def apply_pcolor_on_image(img: np.ndarray,mask: np.ndarray,colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    img: cv2.imread得到的np.ndarray
    mask: [0-1]浮点数组成的矩阵,
    colormap:伪彩色色条选择
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap) #将cam的结果转成伪彩色图片
    cam = np.float32(heatmap)/255 + img/255
    cam = cam / np.max(cam)
    return np.uint8(255*cam)

def remove_temp_ckpt_file():
    LOG_PATH = './log/'

    for file in os.listdir(LOG_PATH):
        ckpt_path = LOG_PATH+file+'/ckpt/'
        for pth_file in os.listdir(ckpt_path):
            if not pth_file.endswith("Best_mae_test.pth"):
                os.remove(ckpt_path+pth_file)

def ls_csv_file():
    x= []
    for log in os.listdir('./log/'):
        if os.path.exists('./log/'+log+"/save"):
            x.append(log)
    return sorted(x)

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

if __name__ == "__main__":
    x = ls_csv_file()
    print(x)
    #remove_temp_log_file()
