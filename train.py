import sys,os

from torch.nn.init import xavier_uniform_

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from torch.nn.parameter import Parameter
import torch
from torch.utils.data import DataLoader,Dataset
import torchvision
from torchvision import transforms
import cv2 as cv
from visdom import Visdom
# from scipy.ndimage import morphology
from scipy.ndimage import distance_transform_edt
import numpy as np
import logging
# from src.models.modnet import MODNet
from src.models.modnet_auto2 import MODNet_auto
from  src.trainerv import supervised_training_iter
import random
from PIL import Image
import json, time
from datetime import datetime
torch.set_float32_matmul_precision('high')
viz = Visdom()
import traceback
# logging.basicConfig(filename='./my_log/my_train.log', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
# logging.info('------------------------------------')
#     # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),

# # 自添加
def log(exc_type, exc_value, exc_traceback):
    error_log_path = './error_log.txt'
    with open(error_log_path, "a")as error_file:
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=error_file)
        error_file.write("\n" + "-" *80 + "\n")

sys.excepthook = log
# # 至此

# 
def get_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# 鐠嬪啰鏁ら崙鑺ユ殶閿涘矁顔曠純顕€娈㈤張铏诡潚鐎涙劒锟�
get_random_seed(73)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
    
])

label_transform = transforms.Compose([
    transforms.ToTensor()
])

class my_dataset(Dataset):
    def __init__(self,input_path,label_path,img_transform=None,label_transform=None,w=512,h=512,num=-1):
        self.input_path = input_path
        self.label_path = label_path
        self.img_path = sorted([os.path.join(self.input_path,img) for img in os.listdir(self.input_path)])
        self.alpha_path = sorted([os.path.join(self.label_path,label) for label in os.listdir(self.label_path)])
        print("original_image:",len(self.img_path))
        
        m = np.random.choice(len(self.img_path),num,replace=False)
        self.img_path = np.array(self.img_path)[m].tolist()
        self.alpha_path = np.array(self.alpha_path)[m].tolist()
        # self.img_path = self.img_path[:num]
        # self.alpha_path = self.alpha_path[:num]

        self.w = w
        self.h = h
        self.img_transform = img_transform
        self.label_transform = label_transform
        print("used_image:", len(self.img_path))
        assert len(self.img_path) == len(self.alpha_path)
    def __len__(self):
        return len(self.img_path)
    def __getitem__(self, index):
        img = cv.imread(self.img_path[index])
        label = cv.imread(self.alpha_path[index])
        if img is None:
            raise ValueError(f"Image at index {index} could not be loaded. Path: {self.img_path[index]}")
        img = cv.resize(img, (self.w, self.h))
        if label is None:
            raise ValueError(f"label at index {index} could not be loaded. Path: {self.label_path[index]}")
        label = cv.resize(label, (self.w, self.h))


        gai = random.random()
        if gai >0.5:
            img = cv.flip(img,1)
            label = cv.flip(label,1)

        gai1 = random.random()
        if gai1 > 0.5:
            img = cv.GaussianBlur(img, (5, 5), 1.5)

        trimap = self.getTrimap(label)

        # img = Image.fromarray(cv.cvtColor(img,cv.COLOR_BGR2RGB)).convert('RGB')

        if self.img_transform:
            img = self.img_transform(img)
        if self.label_transform:
            label = self.label_transform(label[:,:,0])
        return img,trimap,label
    def getTrimap(self,alpha):
        fg = np.array(np.equal(alpha, 255).astype(np.float32))
        unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))  # unknown = alpha > 0
        unknown = unknown - fg
        # unknown = morphology.distance_transform_edt(unknown == 0) <= np.random.randint(10, 20)
        unknown = distance_transform_edt(unknown == 0) <= np.random.randint(10, 20)

        trimap = fg
        trimap[unknown] = 0.5
        return trimap[:, :, :1]

# original
# cfg = None
# cfg1 = None

# prune_info_path = "./result/prune/modify_mppm_prune_twice/r_0.5_t0.4/modnet_p_new_trimap_0027_lr0.0001_ratio_0.5_thresh_0.3.json"
prune_info_path = "./result/prune/modify_mppm_prune_once/modnet_p_new_trimap_0027_lr0.0001_ratio_0.5_thresh_0.4.json"

def train(img_path,label_path,resume = False,std=0):
    model_save = "./model_save/1/"
    start_time = datetime.now()
    logging.info(f"Training started at {start_time}")
    print(f"Training started at {start_time}")
    logging.info("Training started")
    if not os.path.exists(model_save):
        os.makedirs(model_save)
    if resume:
        model_name = sorted(os.listdir(model_save))[-1]
        pretrained_model = os.path.join(model_save,model_name)
    else:
        # pretrained_model = "./pretrained/modify_modnet_photographic_portrait_matting"

        pretrained_model = "./model_save/pruned_once_rep_lr_all5x5layers.pth"
    logging.info(f'model load {pretrained_model}')

    # modnet = torch.nn.DataParallel(MODNet(backbone_pretrained=True,cfg=cfg,cfg1=cfg1))
    prune_info = json.load(open(prune_info_path))
    ratio = prune_info['ratio']
    threshold = prune_info['threshold']
    my_cfg = prune_info['new_cfg']
    my_expansion_cfg = prune_info['new_expansion_cfg']
    my_hr_channels = prune_info['new_hr_channels']

    my_lr_channels = prune_info['new_lr_channels']
    my_f_channels = prune_info['new_f_channels']

    # Create results directory if it doesn't exist
    results_dir = './training_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create a timestamp for the results file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(results_dir, f'{timestamp}.txt')
    plot_file = os.path.join(results_dir, f'loss_plot_{timestamp}.png')

    # Initialize results file with header
    with open(results_file, 'w') as f:
        f.write("Training Results - Started at: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n")
        f.write("-" * 80 + "\n")
        f.write("|{:^10s}|{:^22s}|{:^22s}|{:^22s}|\n".format(
            "Epoch", "Semantic Loss", "Detail Loss", "Matte Loss"))
        f.write("-" * 80 + "\n")

    modnet = MODNet_auto(cfg=my_cfg, expansion=my_expansion_cfg, lr_channel=my_lr_channels,
                         hr_channel=my_hr_channels,
                         f_channel=my_f_channels,
                         hr_channels=int(32 * (1 - ratio)),
                         backbone_pretrained=False, deploy=False)

    state_dict = torch.load(pretrained_model)
    modnet.load_state_dict(state_dict, strict=True)
    modnet = torch.nn.DataParallel(modnet)
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # modnet = modnet.to(device)
    # modnet = torch.nn.DataParallel(MODNet(backbone_pretrained=False,cfg=cfg,cfg1=cfg1))

    GPU = True if torch.cuda.device_count() > 0 else False
    if GPU:
        print("Use GPU...")
        # modnet.cuda().half()
        modnet.cuda()
        # modnet.load_state_dict(torch.load(pretrained_model))
    else:
        print('Use CPU...')
        # modnet.load_state_dict(torch.load(pretrained_model, map_location=torch.device('cpu'))) 

    # print(modnet.state_dict())
    bs = 32
    lr = 0.01
    epochs = 100
    num1 = 30000
    logging.info(f'batch_size: {bs},lr :{lr}, epochs: {epochs}')
    optimizer = torch.optim.SGD(modnet.parameters(), lr=lr, momentum=0.9)
    # optimizer=torch.optim.Adam(modnet.parameters(),lr=lr,betas=(0.9,0.99))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.25 * epochs), gamma=0.1)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    dataset_train=my_dataset(img_path,label_path,img_transform,label_transform,num=num1)
    trainloader=DataLoader(dataset_train,batch_size=bs,num_workers=8,pin_memory=True,shuffle=True,worker_init_fn=seed_worker)
    
    viz.line([0.], [0], win='semantic_loss', opts=dict(title='semantic_loss',legend=['semantic_loss']))
    viz.line([0.], [0], win='detail_loss', opts=dict(title='detail_loss',legend=['detail_loss']))
    viz.line([0.], [0], win='matte_loss', opts=dict(title='matte_loss',legend=['matte_loss']))
    
    for epoch in range(std, epochs):
        semantic_loss1 = []
        detail_loss1=[]
        matte_loss1 = []
        for idx, (image, trimap, gt_matte) in enumerate(trainloader):
            trimap = np.transpose(trimap, (0, 3, 1, 2)).float().cuda()
            semantic_loss, detail_loss, matte_loss = \
                            supervised_training_iter(modnet, optimizer, image.cuda(), trimap, gt_matte.cuda(),idx)
            # semantic_loss, detail_loss, matte_loss = \
            #     supervised_training_iter(modnet, optimizer, image.cuda().half(), trimap.half(), gt_matte.cuda().half(),idx)
                # supervised_training_iter(modnet, optimizer, image.cuda(), trimap, gt_matte.cuda(),idx)                
                # supervised_training_iter(modnet, optimizer, image.half(), trimap.half(), gt_matte.half())
            
            semantic_loss1.append(float(semantic_loss))
            detail_loss1.append(float(detail_loss))
            matte_loss1.append(float(matte_loss))
        # viz.line([semantic_loss.cpu().detach().numpy()], [epoch], win='semantic_loss', update='append')
        # viz.line([detail_loss.cpu().detach().numpy()], [epoch], win='detail_loss', update='append')
        # viz.line([matte_loss.cpu().detach().numpy()], [epoch], win='matte_loss', update='append')
        avg_semantic=float(np.mean(semantic_loss1))
        avg_detail = float(np.mean(detail_loss1))
        avg_matte = float(np.mean(matte_loss1))
        info = f"epoch: {epoch+1}/{epochs} semantic_loss: {avg_semantic}, detail_loss: {avg_detail}, matte_loss: {avg_matte}"
        print(idx,info,optimizer.param_groups[0]['lr'])
        logging.info(f"epoch: {epoch+1}/{epochs}, matte_loss: {avg_semantic}")
        logging.info(f"epoch: {epoch+1}/{epochs}, matte_loss: {avg_detail}")
        logging.info(f"epoch: {epoch+1}/{epochs}, matte_loss: {avg_matte}")
        viz.line([avg_semantic], [epoch], win='semantic_loss', update='append')
        viz.line([avg_detail], [epoch], win='detail_loss', update='append')
        viz.line([avg_matte], [epoch], win='matte_loss', update='append')
        lr_scheduler.step()
        torch.save(modnet.state_dict(), os.path.join(model_save, 'new_trimap_{:0>4d}_lr{}.pth'.format(epoch+1,optimizer.param_groups[0]['lr'])))
        print(f'----------{epoch+1}--------------save model over-----------------------------------')
        logging.info(f'------save model------{epoch+1}  {epoch+1}.pth')
    
        with open(results_file, 'a') as f:
            f.write("|{:^10d}|{:^22.6f}|{:^22.6f}|{:^22.6f}|\n".format(
                epoch + 1, avg_semantic, avg_detail, avg_matte))
            if (epoch + 1) % 5 == 0:  # Add a separator line every 5 epochs
                f.write("-" * 80 + "\n")

    with open(results_file, 'a') as f:
        f.write("\nTraining Completed - Ended at: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n") 

    end_time = datetime.now()
    total_time = end_time - start_time
    logging.info(f"Training ended at {end_time}")
    logging.info(f"Total training time: {total_time}")
    print(f"Training ended at {end_time}")
    print(f"Total training time: {total_time}")

img_root = "./src/datasets/new_data/train/image"
label_root = "./src/datasets/new_data/train/alpha/"
train(img_path=img_root,label_path=label_root)