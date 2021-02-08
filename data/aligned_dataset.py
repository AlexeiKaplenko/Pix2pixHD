import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image

import s3fs

fs = s3fs.S3FileSystem()

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)

        self.A_paths = []
    
        for image_path in sorted(fs.ls(self.dir_A))[1:]:
            self.A_paths.append('s3://'+image_path)
        
        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            #self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  #!!!
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  

            self.B_paths = []
            
            for image_path in sorted(fs.ls(self.dir_B))[1:]:
                self.B_paths.append('s3://'+image_path)

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]              
       
        with fs.open(A_path) as f:
            A = Image.open(f)
            
            params = get_params(self.opt, A.size)
            if self.opt.label_nc == 0:
                transform_A = get_transform(self.opt, params)
                A_tensor = transform_A(A.convert('RGB'))
            else:
                transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
                A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]   
            
            with fs.open(B_path) as f:
                B = Image.open(f)
            
                transform_B = get_transform(self.opt, params)      
                B_tensor = transform_B(B.convert('RGB'))

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))                            

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'