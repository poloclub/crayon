import csv 
import os
import json
from tqdm import tqdm
from utils.utils import NumpyEncoder

import numpy as np
import torch 
import torch.nn as nn 
import torch.optim as optim 
from dataset.loader import getUnshuffledLoader



class SolverWaterbirds:
    def __init__(self, model, loaders, args):
        self.args = args
        self.train_loader = loaders[0]
        self.val_loader = loaders[1]
        self.test_loader = loaders[2]
        self.device = torch.device(f"cuda:{self.args.cuda[0]}")
        self.model = model 
        self.model = self.model.to(self.device)
        self.model_save_dir = os.path.join(args.model_dir, args.model_save_name)
        self.performance_save_dir = os.path.join(args.performance_save_dir, args.performance_save_name)
        if args.performance_save_on and (os.path.exists(self.performance_save_dir) or (not os.path.exists(args.performance_save_dir))): raise NameError
        self.setTrainVars()
        self.setHumanFBVars()
    
    def setTrainVars(self):
        self.vanilla_epoch = self.args.vanilla_epoch
        self.vanilla_optimizer = optim.Adam(self.model.parameters(), self.args.vanilla_lr, weight_decay=self.args.vanilla_decay)

    def setHumanFBVars(self):
        self.method = self.args.method
        self.concept_fbs = None
        self.concept_fbs_tensor = None

        self.fb_epoch = self.args.fb_epoch 
        self.num_fb = self.args.num_fb
        if self.num_fb == -1 and self.method == "crayon-attention": self.num_fb = len(self.train_loader.dataset)
        elif self.num_fb == -1 and self.method == "crayon-pruning": self.num_fb = 2048
        elif self.num_fb == -1 and self.method == "crayon-attention+pruning": self.num_fb = len(self.train_loader.dataset)

        self.w_yes = self.args.w_yes
        self.w_no = self.args.w_no
        self.w_cls = self.args.w_cls 

        if self.method == "crayon-pruning":
            self.fb_optimizer = optim.SGD(self.model.model_ft.fc.parameters(), self.args.fb_lr, weight_decay=self.args.weight_decay)
            self.fb_scheduler = optim.lr_scheduler.MultiStepLR(self.fb_optimizer, milestones=[4,], gamma=0.2)
        else:
            self.fb_optimizer = optim.Adam(self.model.parameters(), self.args.fb_lr, weight_decay=self.args.weight_decay)

        self.fb_loader = None 
        self.performance_array = []
        self.performance_array.append(["WGA", "MGA"])

    def load_model(self, filename=None):
        if filename is None: filename=self.model_save_dir
        
        if not os.path.exists(filename):
            print("There is no pretrained model to load")
            self.model = self.model.to(self.device)
            return False
        
        self.model.load_state_dict(torch.load(filename, map_location=f"cuda:{self.args.cuda[0]}"))
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Pretrained model loading done!")
        
        return True

    def train(self, method="vanilla"):
        if method == "vanilla":
            one_epoch_func = self.train_vanilla_one_epoch
            epoch = self.vanilla_epoch 
        elif method =="crayon-attention":
            one_epoch_func = self.train_crayon_attention_one_epoch
            epoch = self.fb_epoch
        elif method =="crayon-pruning":
            one_epoch_func = self.train_crayon_pruning_one_epoch
            epoch = self.fb_epoch
        elif method == "crayon-attention+pruning":
            one_epoch_func = self.train_crayon_one_epoch
            epoch = self.fb_epoch

        for e in range(epoch):
            print(f"*** Train {method} *** Epoch {e+1:02d}/{epoch:02d} Start!")
            
            self.model.train()
            one_epoch_func()
            
            self.model.eval()
            self.test()
        
    def train_vanilla_one_epoch(self, **kwargs):
        criterion = nn.CrossEntropyLoss()
        loss_tot, correct, num_data = 0, 0, 0
        for data in tqdm(self.train_loader):
            img = data["img"].to(torch.float32).to(self.device)
            label = data["label"].to(self.device)

            logit = self.model(img)
            pred = torch.max(logit, dim=-1)[1]
            loss = criterion(logit, label)

            self.vanilla_optimizer.zero_grad()
            loss.backward()
            self.vanilla_optimizer.step()

            loss_tot += loss.item()
            correct += (pred==label).sum().item()
            num_data += len(img)

        loss = loss_tot/num_data
        acc = correct/num_data
        print(f"Train done --- Loss: {loss:.05f}, Acc: {acc:.05f}")
        
        return loss, acc

    def train_crayon_one_epoch(self, **kwargs):
        criterion = nn.CrossEntropyLoss(reduction="sum")
        correct, num_data = 0, 0
        loss_tot, loss_expl_tot, loss_cls_tot = 0,0,0
        expl_func = self.explain_gradcam
        fbs = torch.Tensor(self.concept_fbs).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(self.device)

        for idx, data in enumerate(self.fb_loader):
            img = data["img"].to(torch.float32).to(self.device)
            label = data["label"].to(self.device)
            prev_expl = data["expl"].detach().to(self.device)
            human_fb = data["fb"].to(self.device).reshape(-1,1)  

            logit = self.model(img, neuron_mask=fbs)
            pred = torch.max(logit, dim=-1)[1]
            loss_cls = criterion(logit, label)

            expl = expl_func(img, label) 
            loss_no, loss_yes = self.get_loss_expl(expl, prev_expl, human_fb)
            loss_expl = loss_no * self.w_no + loss_yes * self.w_yes 
            loss = loss_cls * self.w_cls + loss_expl

            self.fb_optimizer.zero_grad()
            loss.backward()
            self.fb_optimizer.step()

            loss_tot += loss.item()
            correct += (pred==label).sum().item()
            num_data += len(img)

            loss_cls_tot += loss_cls.item()
            loss_expl_tot += loss_expl.item()

        loss = loss_tot/num_data
        acc = correct/num_data

        print(f"Train done --- Loss: {loss:.05f}, Expl: {loss_expl_tot/num_data:.05f}, Cls: {loss_cls_tot/num_data:.05f}, Acc: {acc:.05f}")
        
        return loss, acc

    def train_crayon_attention_one_epoch(self, **kwargs):
        criterion = nn.CrossEntropyLoss(reduction="sum")
        correct, num_data = 0, 0
        loss_tot, loss_expl_tot, loss_cls_tot = 0,0,0
        expl_func = self.explain_gradcam

        for idx, data in enumerate(self.fb_loader):
            img = data["img"].to(torch.float32).to(self.device)
            label = data["label"].to(self.device)
            prev_expl = data["expl"].detach().to(self.device)
            human_fb = data["fb"].to(self.device).reshape(-1,1)  

            logit = self.model(img)
            pred = torch.max(logit, dim=-1)[1]
            loss_cls = criterion(logit, label)

            expl = expl_func(img, label) 
            loss_no, loss_yes = self.get_loss_expl(expl, prev_expl, human_fb)
            loss_expl = loss_no * self.w_no + loss_yes * self.w_yes 
            loss = loss_cls * self.w_cls + loss_no * self.w_no + loss_yes * self.w_yes 

            self.fb_optimizer.zero_grad()
            loss.backward()
            self.fb_optimizer.step()

            loss_tot += loss.item()
            correct += (pred==label).sum().item()
            num_data += len(img)

            loss_cls_tot += loss_cls.item()
            loss_expl_tot += loss_expl.item()

        loss = loss_tot/num_data
        acc = correct/num_data

        print(f"Train done --- Loss: {loss:.05f}, Expl: {loss_expl_tot/num_data:.05f}, Cls: {loss_cls_tot/num_data:.05f}, Acc: {acc:.05f}")
        
        return loss, acc
    
    def train_crayon_pruning_one_epoch(self, **kwargs):
        criterion = nn.CrossEntropyLoss(reduction="sum")
        correct, num_data = 0, 0
        loss_tot = 0
        fbs = torch.Tensor(self.concept_fbs).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(self.device)

        for idx, data in enumerate(self.train_loader):
            img = data["img"].to(torch.float32).to(self.device)
            label = data["label"].to(self.device)

            logit = self.model(img, neuron_mask=fbs, detach_h=True)
            pred = torch.max(logit, dim=-1)[1]
            loss = criterion(logit, label)

            self.fb_optimizer.zero_grad()
            loss.backward()
            self.fb_optimizer.step()

            loss_tot += loss.item()
            correct += (pred==label).sum().item()
            num_data += len(img)

        loss = loss_tot/num_data
        acc = correct/num_data
        self.fb_scheduler.step()

        print(f"Train done --- Loss: {loss:.05f}, Acc: {acc:.05f}")
        
        return loss, acc

    def test(self):
        w_wb_num, w_lb_num, l_wb_num, l_lb_num = 0, 0, 0, 0
        w_wb_cor, w_lb_cor, l_wb_cor, l_lb_cor = 0, 0, 0, 0
        loss_tot, correct, num_data = 0, 0, 0
        criterion = nn.CrossEntropyLoss(reduction="sum")

        if self.concept_fbs is not None: fbs = torch.Tensor(self.concept_fbs).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(self.device)
        else: fbs = None

        self.model.eval()

        for data in self.test_loader:
            img = data["img"].to(torch.float32).to(self.device)
            label = data["label"].to(self.device)
            place = data["place"].to(self.device)

            logit = self.model(img, neuron_mask=fbs)
            pred = torch.max(logit, dim=-1)[1]
            loss = criterion(logit, label)
            corr = (pred==label)

            w_wb_num += (place * label).sum().item()
            w_lb_num += (place * (1-label)).sum().item()
            l_wb_num += ((1-place) * label).sum().item()
            l_lb_num += ((1-place) * (1-label)).sum().item()
            w_wb_cor += (place * label * corr).sum().item()
            w_lb_cor += (place * (1-label) * corr).sum().item()
            l_wb_cor += ((1-place) * label * corr).sum().item()
            l_lb_cor += ((1-place) * (1-label) * corr).sum().item()

            loss_tot += loss.item()
            correct += corr.sum().item()
            num_data += len(img)
            
        loss = loss_tot/num_data
        acc = correct/num_data

        mga = sum([w_wb_cor/w_wb_num,l_wb_cor/l_wb_num,w_lb_cor/w_lb_num,l_lb_cor/l_lb_num]) / 4
        wga = min(w_wb_cor/w_wb_num,l_wb_cor/l_wb_num,w_lb_cor/w_lb_num,l_lb_cor/l_lb_num)
        print(f"Test done ---- Loss: {loss:.05f}, Acc: {acc:.05f}, MA: {mga:.05f}, WGA: {wga:.05f}")
        self.performance_array.append([wga*100, mga*100])

        return loss, acc 

    def get_loss_expl(self, expl, prev_expl, human_fb):
        prev_expl = (1e-10+prev_expl) / torch.max(torch.max(1e-10+prev_expl, dim=-1)[0], dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
        expl = (1e-10+expl) / torch.max(torch.max(1e-10+expl, dim=-1)[0], dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
        human_fb = human_fb.unsqueeze(-1)
        
        assert torch.min(expl) >= 0 and torch.min(prev_expl) >= 0 and torch.max(expl) <= 1
        human_fb = human_fb.squeeze()
        loss_no = torch.sum((human_fb==1) * (torch.sum(torch.sum(expl*prev_expl, dim=-1), dim=-1) / torch.sum(torch.sum(prev_expl, dim=-1), dim=-1)))
        loss_yes = torch.sum((human_fb==0) * (torch.sum(torch.sum(expl*(1-prev_expl), dim=-1), dim=-1) / torch.sum(torch.sum((1-prev_expl), dim=-1), dim=-1)))
        return loss_no, loss_yes

    def explain_gradcam(self, x, y):
        x = x.to(self.device)
        logit, h = self.model(x, return_h=True)  # logit: 64x2, h: 64x2048x9x7

        h.retain_grad()
        logit_c = logit[:,0]*(y==0) + logit[:,1]*(y==1)
        logit_c_sum = torch.sum(logit_c)  # scalar tensor
        grad = torch.autograd.grad(logit_c_sum, h, create_graph=True)[0]  # 64x2048x9x7
        alpha = torch.mean(torch.mean(grad, -1), -1)
        if self.concept_fbs_tensor is not None: alpha = alpha * self.concept_fbs_tensor.unsqueeze(0)  # 64x2048 
        weighted_h = alpha.reshape(alpha.shape[0],alpha.shape[1],1,1) * h
        gradcam = torch.sum(weighted_h,1)
        gradcam = torch.maximum(gradcam, torch.zeros_like(gradcam))
        return gradcam

    def explain_neuron_concept_patch(self):
        print(f"Start --- Get neuron concept patch")
        self.model.eval()

        # if saved return the saved info
        img_patch_names_json_file = f"{self.args.expl_dir}/concept_filename_{self.args.model_name}_{self.fb_loader_type}_{self.args.seed}.json"
        img_patch_coord_json_file = f"{self.args.expl_dir}/concept_coord_{self.args.model_name}_{self.fb_loader_type}_{self.args.seed}.json"
        if os.path.exists(img_patch_names_json_file) and os.path.exists(img_patch_coord_json_file):
            with open(img_patch_names_json_file, 'r') as f: 
                img_patch_names = np.asarray(json.load(f))
            with open(img_patch_coord_json_file, 'r') as f: 
                img_patch_coord = np.asarray(json.load(f))
        else:
            # 1. Make an array with size of num_neuron x num_image
            num_neuron = self.model.model_ft.fc.in_features
            num_image = len(self.train_loader.dataset)
            max_act_mat = torch.zeros([num_neuron, num_image])
            batch_size = self.args.batch_size
            unshuffled_train_loader = getUnshuffledLoader(self.train_loader.dataset, batch_size)

            # 2. Go through the train dataset
            for i, data in enumerate(unshuffled_train_loader):
                imgs = data["img"].to(self.device)
                _, acts = self.model(imgs, return_h=True)  # acts: batch_size x neuron_num x h_act x w_act
                max_acts = torch.max(torch.max(acts, dim=-1)[0], dim=-1)[0].T  # batch_size x neuron_num
                max_act_mat[:,i*batch_size:(i+1)*batch_size] = max_acts.cpu() 

            # 3. top 20 image indices for each neuron, neuron_importance based on the activation
            num_candidate_image = 20
            candidate_img_mat = torch.argsort(-max_act_mat)[:,:num_candidate_image]  # num_neuron x num_candidate

            # 4. for each neuron, examine the candidate images and get patches
            img_patch_names = [[""]*num_candidate_image]*num_neuron
            img_patch_coord = -np.ones([num_neuron,num_candidate_image,2])
            for n, i_idx in enumerate(candidate_img_mat):
                imgs = torch.stack([self.train_loader.dataset[i]["img"] for i in i_idx]).to(self.device)  # num_candidate_image x 3 x 274 x 224
                img_patch_names[n] = [self.train_loader.dataset[i]["img_file"] for i in i_idx]

                _, acts = self.model(imgs, return_h=True)  # acts: num_candidate_image x neuron_num x h_act x w_act
                n_acts = acts[:,n,:,:]  # num_candidate_image x h x w
                h, w = acts.shape[2], acts.shape[3]
                n_acts = n_acts.reshape(num_candidate_image, h*w)
                
                max_act_idx = torch.argmax(n_acts, dim=1)
                h_idx = max_act_idx // w
                w_idx = max_act_idx % w
                
                img_patch_coord[n,:,0] = h_idx.cpu().numpy()
                img_patch_coord[n,:,1] = w_idx.cpu().numpy()

            img_patch_coord_json = json.dumps(img_patch_coord, cls=NumpyEncoder)
            img_patch_names_json = json.dumps(img_patch_names)
            
            with open(img_patch_coord_json_file, "w") as outfile:
                outfile.write(img_patch_coord_json)
            with open(img_patch_names_json_file, "w") as outfile:
                outfile.write(img_patch_names_json)

        self.img_patch_coord = img_patch_coord.astype(np.uint8)
        self.img_patch_names = img_patch_names
        print(f"Done --- Set image patch coord with shape {self.img_patch_coord.shape} and names ({len(self.img_patch_names)},{len(self.img_patch_names[0])})")

    def set_expl(self, expl_file=None, save=True, expl_mode=None):
        # For segmentation map, should know the correspondence between gradcam map <-> seg map
        if expl_mode == "concept" or (expl_mode is None and self.method == "crayon-pruning"):
            with torch.no_grad():
                self.explain_neuron_concept_patch()
        elif expl_mode == "gradcam" or (expl_mode is None and self.method == "crayon-attention"):
            self.set_gradcam(expl_file=expl_file, save=save)

    def set_gradcam(self, expl_file=None, save=True):
        # For segmentation map, should know the correspondence between gradcam map <-> seg map
        if expl_file is None: expl_file = os.path.join(self.args.expl_dir, self.args.expl_save_name)

        if not os.path.exists(expl_file):
            print(f"No expl file {expl_file} -- should newly save")
            expls = self.save_gradcam(expl_file, save)
        else:
            with open(expl_file, 'r') as f: 
                expls = np.asarray(json.load(f))
            print(f"Done --- Loaded GradCAMs for training data (Shape: {expls.shape})")

        self.train_loader.dataset.E = expls

    def get_hfb(self, hfb_file=None, expl_mode=None, num_fb=None):
        print(f"Start --- Get human feedback for the data in training data")
        if hfb_file is None: hfb_file = os.path.join(self.args.hfb_dir, self.args.hfb_save_name)
        if expl_mode is None:
            if self.method=="crayon-attention": expl_mode="gradcam"
            if self.method=="crayon-pruning": expl_mode="concept"
            if self.method=="crayon-attention+pruning": raise ValueError("Should explicitly enter expl_mode")
        if num_fb is None: num_fb = self.num_fb
        
        if num_fb==-1 and expl_mode == "concept": num_fb = 2048 
        elif num_fb==-1 and expl_mode == "gradcam": num_fb = 4795

        assert os.path.exists(hfb_file)
        with open(hfb_file) as f: 
            hfbs = np.asarray(json.load(f))
            print(f"Loaded feedback {hfbs.shape}")

        num_train = len(self.train_loader.dataset)
        if expl_mode == "gradcam" and num_fb < num_train:
            print(f"Leave only {num_fb} gradcam feedback")
            idx = np.random.choice(num_train, num_train-num_fb, replace=False)  
            hfbs[idx] = "maybe"
        elif expl_mode == "concept" and num_fb < 2048: 
            print(f"Leave only {num_fb} concept feedback")
            idx = np.random.choice(2048, 2048-num_fb, replace=False)
            hfbs[idx] = 1

        num_yes, num_maybe, num_no = 0, 0, 0
        if expl_mode == "concept":
            print(f"{np.sum(hfbs)}/2048 neurons are relevant")
            self.concept_fbs = hfbs
            self.concept_fbs_tensor = torch.Tensor(self.concept_fbs).to(self.device)
        elif expl_mode == "gradcam":
            num_yes = np.sum(hfbs=='yes')
            num_maybe = np.sum(hfbs=='maybe')
            num_no = np.sum(hfbs=='no')
            print(f"Relevant: {num_yes}/{len(hfbs)}")
            print(f"Maybe: {num_maybe}/{len(hfbs)}")
            print(f"Irrelevant: {num_no}/{len(hfbs)}")

            self.w_yes = self.w_yes/num_yes if num_yes>0 else self.w_yes 
            self.w_no = self.w_no/num_no if num_no>0 else self.w_no 

            print(f"Set w_yes to {self.w_yes}, w_no to {self.w_no}")
        
        print(f"Done --- Got human feedback")
        return hfbs

    def save_gradcam(self, expl_file="", save=True):
        print(f"Start --- Save Grad-CAMs for training data")
        self.model.eval()

        dataset = self.train_loader.dataset
        loader = getUnshuffledLoader(dataset, self.args.batch_size)
        expls = []

        for i, data in tqdm(enumerate(loader)):
            img = data["img"].to(self.device)
            label = data["label"].to(self.device)
            expl = self.explain_gradcam(img, label)
            expl = (1e-10+expl) / torch.max(torch.max(1e-10+expl, dim=-1)[0], dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
            expls.append(expl.cpu().detach().numpy())
        expls = np.vstack(expls)
        print(f"Done --- Compute Grad-CAMs for training data --- Shape {expls.shape}")

        # Save expls and corrects at expl_dir as a json file
        if save:
            dumped = json.dumps(expls, cls=NumpyEncoder)
            with open(expl_file, "w") as f:
                f.write(dumped)
            print(f"Done --- Save GradCAMs for training data")
        
        return expls

    def save_performance(self, aupr=0):
        # save self.performance_array to self.performance_save_dir
        with open(self.performance_save_dir, 'w') as f:
            writer = csv.writer(f)
            for row in self.performance_array:
                writer.writerow(row)

    

