import os 
import numpy as np 
import torch 

from utils.args import getArgs
from solver.solver_waterbirds import SolverWaterbirds
from model.cnn_waterbirds import ResNet50Waterbirds
from dataset.loader import getLoader, getLoaderWithHumanFB


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    _, loaders = getLoader(args.data_name, args.data_dir, args.batch_size, args.num_workers, model_type=args.model_name)

    model =  ResNet50Waterbirds()
    solver = SolverWaterbirds(model, loaders, args)
    solver.load_model()
    solver.test()

    if args.method in ["crayon-attention+pruning", "crayon-pruning"]:
        solver.get_hfb(expl_mode="concept", hfb_file=os.path.join(args.hfb_dir, "waterbirds_neuron.json"), num_fb=2048)

    if args.method in ["crayon-attention+pruning", "crayon-attention"]:
        solver.set_expl(expl_mode="gradcam", expl_file=os.path.join(args.expl_dir, "waterbirds_gradcam.json"))
        gradcam_fbs = solver.get_hfb(expl_mode="gradcam", hfb_file=os.path.join(args.hfb_dir, "waterbirds_gradcam.json"), num_fb=args.num_fb)
        expl_dataset = solver.train_loader.dataset
        Xs = expl_dataset.img_files
        ys = expl_dataset.labels
        img_transform = expl_dataset.transform
        solver.fb_loader = getLoaderWithHumanFB(Xs, ys, expl_dataset.E, gradcam_fbs, args.batch_size, args.data_name, img_transform)

    solver.train(method=args.method)
    if args.performance_save_on: solver.save_performance()


if __name__=="__main__":
    args = getArgs()
    print(args)
    main(args)