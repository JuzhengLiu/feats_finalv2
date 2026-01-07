import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms.functional as TF
import random
import torch 
import matplotlib.pyplot as plt
import json
import os, re
import os.path as osp
import cv2
from utils.transform import build_transform 
from utils.utils import DATASET
# from torch_geometric.data import Data
from collections import defaultdict


HOME = osp.expanduser('~')

        
def get_data(path):
    data = {}
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            data[name] = {"path": os.path.join(root, name)}
            for _, _, files in os.walk(data[name]["path"], topdown=False):
                data[name]["files"] = files
                
    return data
    

    

@DATASET.register 
class U1652_Image_D2S(data.Dataset):
    def __init__(self, data_path, transform, mode='dro', **opt):
        super().__init__()
        self.transforms = build_transform(transform)
        # _, self.transforms = clip.load("ViT-B/32", device='cuda')

        if 'train' in data_path: # training data
            img_dir = 'drone' if mode == 'dro' else 'satellite' # dro=query, sat=gallery
        else: # test data 
            img_dir = 'query_drone' if mode == 'dro' else 'gallery_satellite' # dro=query, sat=gallery
        
        self.mode = mode
        self.img_dir = osp.join(HOME, data_path, img_dir)
        self.file_list = []
        for scene_name in os.listdir(self.img_dir):
            for file in os.listdir(osp.join(self.img_dir, scene_name)):
                file_path = osp.join(scene_name, file)
                self.file_list.append(file_path)
            
    def __getitem__(self, index):
        filename = self.file_list[index]
        file_path = osp.join(self.img_dir, filename)
        scene_name = int(filename.split('/')[0])

        img = Image.open(file_path)
        img_trans = self.transforms(img) #.unsqueeze(0)
        data = dict(x=img_trans, y=scene_name, name=filename, mode=self.mode)

        return data 

    def __len__(self):
        return len(self.file_list)


@DATASET.register 
class U1652_Image_S2D(data.Dataset):
    def __init__(self, data_path, transform, mode='dro', **opt):
        super().__init__()
        self.transforms = build_transform(transform)
        # _, self.transforms = clip.load("ViT-B/32", device='cuda')

        # img_dir = 'query_drone' if mode == 'dro' else 'gallery_satellite' # dro=query, sat=gallery
        img_dir = 'gallery_drone' if mode == 'dro' else 'query_satellite' # dro=query, sat=gallery
        self.mode = mode
        self.img_dir = osp.join(HOME, data_path, img_dir)
        self.file_list = []
        for scene_name in os.listdir(self.img_dir):
            for file in os.listdir(osp.join(self.img_dir, scene_name)):
                file_path = osp.join(scene_name, file)
                self.file_list.append(file_path)
            
    def __getitem__(self, index):
        filename = self.file_list[index]
        file_path = osp.join(self.img_dir, filename)
        scene_name = int(filename.split('/')[0])

        img = Image.open(file_path)
        img_trans = self.transforms(img).unsqueeze(0)
        data = dict(x=img_trans, y=scene_name, name=filename, mode=self.mode)

        return data 

    def __len__(self):
        return len(self.file_list)
    



def load_feat(data_dir, view):
    feat = torch.load(osp.join(data_dir, f'{view}_feat')).to("cpu")
    gid = torch.load(osp.join(data_dir, f'{view}_id')).to("cpu")
    name = torch.load(osp.join(data_dir, f'{view}_name'))
    return feat, gid, name


@DATASET.register 
class Feat_Single(data.Dataset):
    def __init__(self, data_path, mode='sat', **opt):
        super().__init__()

        feat = opt['feat']
        self.feat_dir = osp.join(HOME, data_path, feat)
        
        if mode == 'sat_160k': # extend satellite gallery
            img_dir = 'gallery_satellite'
            self.img_dir = osp.join(HOME, data_path, img_dir)
            feat_160k, ids_160k, names_160k = load_feat(self.feat_dir, mode)
            feat_sat, ids_sat, names_sat = load_feat(self.feat_dir, 'sat')

            feat = torch.cat([feat_160k, feat_sat], dim=0)
            ids = torch.cat([ids_160k, ids_sat], dim=0)
            names = names_160k + names_sat
            self.mode = 'sat'

        elif mode == 'dro_split': # split drone test data 
            img_dir = 'query_drone'
            with open('trainOnTest.txt', 'r') as f:
                train_list = [int(x.strip()) for x in f.readlines()]
            self.img_dir = osp.join(HOME, data_path, img_dir)
            feat, ids, names = load_feat(self.feat_dir, 'dro')
            self.mode = mode

        else: # regular dro and sat
            img_dir = 'query_drone' if mode == 'dro' else 'gallery_satellite'
            self.img_dir = osp.join(HOME, data_path, img_dir)
            feat, ids, names = load_feat(self.feat_dir, mode)
            self.mode = mode


        self.file_list = []
        for fea, id, name  in zip(feat, ids, names):
            id = id.item() if opt['remove_junk'] else int(name.split('.')[0].split('/')[0])
            if 'dro_split' in self.mode and id in train_list:
                continue
            self.file_list.append({'feat':fea, 'id': id, 'name':name})
            
    def __getitem__(self, index):
        file_dict = self.file_list[index]
        feat, id, name = file_dict['feat'], file_dict['id'], file_dict['name']
        if name == None:
            name = '-1'
        data = dict(x=feat, y=id, name=name, mode=self.mode)
        return data 

    def __len__(self):
        return len(self.file_list)

 

@DATASET.register 
class U1652_Random_drone(data.Dataset):
    def __init__(self, data_path, **opt):
        super().__init__()
        # self.transforms = build_transform(transform)
        # _, self.transforms = clip.load("ViT-B/32", device='cuda')
        # img_dir = 'query_drone' if mode == 'dro' else 'gallery_satellite' # dro=query, sat=gallery
        if data_path.split('/')[-1] == 'train':
            dro_path, sat_path = 'drone', 'satellite'
        else:
            dro_path, sat_path = 'query_drone', 'gallery_satellite'

        feat = opt['feat']

        
        self.sat_dir = osp.join(HOME, data_path, sat_path)
        self.dro_dir = osp.join(HOME, data_path, dro_path)
        self.feat_dir = osp.join(HOME, data_path, feat)
        self.dro_num = opt['dro_num']

        gall_feat, gid, gall_name = load_feat(self.feat_dir, 'sat')
        que_feat, qid, que_name = load_feat(self.feat_dir, 'dro')

        self.sat_list = []
        self.dro_dict = defaultdict(list)
        self.sat_id_list = []


        # satellite
        for fea, id, name  in zip(gall_feat, gid, gall_name):
            id = id.item() if opt['remove_junk'] else int(name.split('/')[0])
            name = id if name is None else name
            self.sat_list.append({'feat':fea, 'id': id, 'name':name})
            self.sat_id_list.append(id)
            
        
        for fea, id, name  in zip(que_feat, qid, que_name):
            id = id.item() if opt['remove_junk'] else int(name.split('/')[0])
            name = id if name is None else name
            self.dro_dict[id].append({'feat':fea, 'name':name})

        self.dro_id_list = list(self.dro_dict.keys())

    #从对应的类别中抽一张出来
    def sample_from_cls(self, id, num=1):
        file_list = self.dro_dict[id]
        feat_choice= np.random.choice(file_list, num)
        feat_list, name_list = [], []
        for f_dict in feat_choice:
            feat_list.append(f_dict['feat'])
            name_list.append(f_dict['name'])
        feat = torch.stack(feat_list, dim=0)
        return feat, name_list
            
    def __getitem__(self, index):
        sat_dict = self.sat_list[index]
        sat_feat, id, sat_name = sat_dict['feat'], sat_dict['id'], sat_dict['name']
        dro_id = np.random.choice(self.dro_id_list, 1)[0]

        dro_feat, dro_name = self.sample_from_cls(dro_id, self.dro_num)

        data = dict(x_s=sat_feat, x_t=dro_feat, 
                      name_s=sat_name, name_t=dro_name,
                      y_s=id, y_t=dro_id)

        return data 

    def __len__(self):
        return len(self.sat_list)


# =========================================================================
# New Dataset Classes for SUES-200 with Multi-Height Support
# =========================================================================

@DATASET.register 
class SUES_Image_D2S(data.Dataset):
    def __init__(self, data_path, transform, mode='dro', **opt):
        super().__init__()
        self.transforms = build_transform(transform)
        self.mode = mode
        
        # Determine sub-folders based on data_path (Training vs Testing) and mode
        # data_path ex: '.../SUES-200-Standard/Training' or '.../SUES-200-Standard/Testing'
        
        is_training = 'Training' in data_path or 'train' in data_path.lower()
        
        if is_training:
            # Training D2S: Drone (Query-like) <-> Satellite (Gallery-like)
            folder_name = 'drone' if mode == 'dro' else 'satellite'
        else:
            # Testing D2S: Query Drone <-> Gallery Satellite
            folder_name = 'query_drone' if mode == 'dro' else 'gallery_satellite'
            
        # Support multiple heights
        self.heights = ['150', '200', '250', '300']
        
        self.file_list = []
        self.root_dir = osp.join(HOME, data_path)
        
        # Traverse all heights
        for h in self.heights:
            # Path: root/150/drone
            target_dir = osp.join(self.root_dir, h, folder_name)
            
            if not osp.exists(target_dir):
                print(f"[Warning] Directory not found: {target_dir}")
                continue
                
            # Traverse scene ids: 0001, 0002...
            for scene_id in os.listdir(target_dir):
                scene_path = osp.join(target_dir, scene_id)
                if not osp.isdir(scene_path):
                    continue
                    
                for file_name in os.listdir(scene_path):
                    # We store the relative path from root to keep height info
                    # e.g., "150/drone/0001/img.jpg" -> Simplification: Store full path or structured info
                    # To align with old code's "filename", we store: "height/scene_id/file_name"
                    # But wait, original code stores "scene_name/file", and joins with img_dir.
                    # Here img_dir changes per height. 
                    # So we store absolute path or path relative to data_path.
                    
                    # Store: {'path': absolute_path, 'name': "150_sceneId_fileName", 'y': sceneId}
                    
                    full_path = osp.join(scene_path, file_name)
                    
                    # Name for evaluation: 150/0001/0001.jpg -> stored as string to be parsed later
                    # We will store relative path to data_path: "150/drone/0001/img.jpg"
                    rel_path = os.path.relpath(full_path, self.root_dir)
                    
                    # Scene ID (class label)
                    try:
                        label = int(scene_id)
                    except ValueError:
                        label = -1
                        
                    self.file_list.append({
                        'path': full_path,
                        'name': rel_path, # Contains height info: "150/..."
                        'y': label
                    })

    def __getitem__(self, index):
        item = self.file_list[index]
        file_path = item['path']
        scene_name = item['y']
        name_str = item['name'] # "150/drone/0001/img.jpg"

        img = Image.open(file_path).convert('RGB')
        img_trans = self.transforms(img)
        
        # Return data consistent with framework
        data = dict(x=img_trans, y=scene_name, name=name_str, mode=self.mode)
        return data 

    def __len__(self):
        return len(self.file_list)


@DATASET.register 
class SUES_Image_S2D(data.Dataset):
    def __init__(self, data_path, transform, mode='dro', **opt):
        super().__init__()
        self.transforms = build_transform(transform)
        self.mode = mode
        
        is_training = 'Training' in data_path or 'train' in data_path.lower()
        
        if is_training:
            # Training S2D: Usually same as D2S but swapped roles conceptually? 
            # In extraction, we usually extract all.
            # But if specific logic needed:
            folder_name = 'satellite' if mode == 'dro' else 'drone' 
            # Wait, usually mode='dro' means Query, mode='sat' means Gallery in this codebase context
            # For S2D: Query=Satellite, Gallery=Drone
            # So if mode='dro' (Query) -> we want Satellite?
            # extract_and_save.py: 
            # if S2D: gallset=mode('dro'), queryset=mode('sat')
            # So gallset (Gallery) calls mode='dro' -> needs Drone images.
            # queryset (Query) calls mode='sat' -> needs Satellite images.
            # Correct.
            pass
        else:
            # Testing S2D: Query Satellite <-> Gallery Drone
            # mode='dro' (mapped to Query in some logic, but in extract_and_save S2D logic:
            # query_view='sat', gall_view='dro'.
            # queryset = DATASET(mode='sat') -> Query Satellite
            # gallset = DATASET(mode='dro') -> Gallery Drone
            
            if mode == 'dro':
                folder_name = 'gallery_drone'
            else: # mode == 'sat'
                folder_name = 'query_satellite'
            
        self.heights = ['150', '200', '250', '300']
        self.file_list = []
        self.root_dir = osp.join(HOME, data_path)
        
        for h in self.heights:
            target_dir = osp.join(self.root_dir, h, folder_name)
            if not osp.exists(target_dir):
                continue
            
            for scene_id in os.listdir(target_dir):
                scene_path = osp.join(target_dir, scene_id)
                if not osp.isdir(scene_path):
                    continue
                for file_name in os.listdir(scene_path):
                    full_path = osp.join(scene_path, file_name)
                    rel_path = os.path.relpath(full_path, self.root_dir)
                    try:
                        label = int(scene_id)
                    except ValueError:
                        label = -1
                    
                    self.file_list.append({
                        'path': full_path,
                        'name': rel_path,
                        'y': label
                    })

    def __getitem__(self, index):
        item = self.file_list[index]
        img = Image.open(item['path']).convert('RGB')
        img_trans = self.transforms(img)
        data = dict(x=img_trans, y=item['y'], name=item['name'], mode=self.mode)
        return data 

    def __len__(self):
        return len(self.file_list)