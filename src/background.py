import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import numpy as np
import torch 

from utils.args import getArgs
from solver.solver_background import SolverBackground
from model.cnn_background import ResNet50Background
from dataset.loader import getLoader, getLoaderWithHumanFB 

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    datasets, loaders = getLoader(args.data_name, args.data_dir, args.batch_size, args.num_workers)
    print(f"Original Train: {len(datasets[0])}, Original Test: {len(datasets[1])}, Mixed_Same Test: {len(datasets[2])}, Mixed_Rand Test: {len(datasets[3])}, Only_fg Test: {len(datasets[4])}")
    model = ResNet50Background()
    solver = SolverBackground(model, loaders, args)
    if solver.load_model():
        with torch.no_grad():
            _, orig_acc = solver.test(test_loader_type="orig")
            _, mixed_same_acc = solver.test(test_loader_type="mixed_same")
            _, mixed_rand_acc = solver.test(test_loader_type="mixed_rand")
            _, only_fg_acc = solver.test(test_loader_type="only_fg")
        bg_gap = mixed_same_acc - mixed_rand_acc
        solver.performance_array.append([orig_acc, mixed_same_acc, mixed_rand_acc, only_fg_acc, bg_gap])
    else:
        raise Exception("Background needs to have a pretrained model. \
                        Please refer to https://github.com/MadryLab/backgrounds_challenge/tree/master")


    if args.method in ["crayon-attention+pruning", "crayon-pruning"]:
        solver.get_hfb(expl_mode="concept", hfb_file=os.path.join(args.hfb_dir, "background_neuron.json"), num_fb=2048)

    if args.method in ["crayon-attention+pruning", "crayon-attention"]:
        solver.set_expl(expl_mode="gradcam", expl_file=os.path.join(args.expl_dir, "background_gradcam.json"))
        gradcam_fbs = solver.get_hfb(expl_mode="gradcam", hfb_file=os.path.join(args.hfb_dir, "background_gradcam.json"), num_fb=args.num_fb)
        expl_dataset = solver.train_loader.dataset
        Xs, ys = zip(*expl_dataset.samples)
        img_transform = expl_dataset.transform
        solver.fb_loader = getLoaderWithHumanFB(Xs, ys, expl_dataset.E, gradcam_fbs, args.batch_size, args.data_name, img_transform)

    solver.train(method=args.method)
    if args.performance_save_on: solver.save_performance()




if __name__=="__main__":
    args = getArgs()
    print(args)
    main(args)