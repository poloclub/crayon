import argparse 

def getArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_name", default="waterbirds")
    parser.add_argument("--model_name", default="resnet50")
    
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--vanilla_epoch", type=int, default=10)
    parser.add_argument("--vanilla_decay", type=float, default=0.0001)
    parser.add_argument("--vanilla_lr", type=float, default=0.001)
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--method", default="crayon-attention+pruning", choices=["crayon-attention+pruning", "crayon-attention", "crayon-pruning"])
    parser.add_argument("--w_yes", type=float, default=1000)
    parser.add_argument("--w_no", type=float, default=1000)
    parser.add_argument("--w_cls", type=float, default=1)
    parser.add_argument("--fb_epoch", type=int, default=50)
    parser.add_argument("--fb_lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--num_fb", type=int, default=-1)
    
    parser.add_argument("--cuda", nargs="+", default=[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--performance_save_on", action="store_true")

    parser.add_argument("--model_dir", default=".")
    parser.add_argument("--model_save_name", default=".")
    parser.add_argument("--data_dir", default=".")
    parser.add_argument("--expl_dir", default=".")
    parser.add_argument("--hfb_dir", default=".")
    parser.add_argument("--expl_save_name", default=".")
    parser.add_argument("--hfb_save_name", default=".")
    parser.add_argument("--performance_save_dir", default="")
    parser.add_argument("--performance_save_name", default="")

    args = parser.parse_args()
    return args