import argparse

def set_configs():
    parser = argparse.ArgumentParser()
    # Federated training settings
    parser.add_argument("-N", "--clients", help="The number of participants", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--lr_decay", type=float, default=-1, help="learning rate decay for scheduler"
    )
    parser.add_argument(
        "--early", action="store_true", help="early stop w/o improvement over 20 epochs"
    )
    parser.add_argument("--batch", type=int, default=128, help="batch size")
    parser.add_argument("--rounds", type=int, default=200, help="iterations for communication")
    parser.add_argument("--local_epochs", type=int, default=1, help="local training epochs")
    parser.add_argument("--mode", type=str, default="fedavg", help="different FL algorithms")
    parser.add_argument(
        "--pretrain", action="store_true", help="Use Alexnet/ResNet pretrained on Imagenet"
    )
    # Experiment settings
    parser.add_argument("--exp", type=str, default=None, help="exp name")
    parser.add_argument(
        "--save_path", type=str, default="../checkpoint/", help="path to save the checkpoint"
    )
    parser.add_argument(
        "--resume", action="store_true", help="resume training from the save path checkpoint"
    )
    parser.add_argument("--gpu", type=str, default="0", help='gpu device number e.g., "0,1,2"')
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--subset", type=float, default=1.0, help="subsample training sets")
    
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--test", action="store_true", help="Running test mode.")
    parser.add_argument("--ckpt", type=str, default="None", help="Path for the testing ckpt")
    # Data settings
    parser.add_argument(
        "--data", type=str, default="digits5", help="Different dataset: cifar10, cifar10c"
    )
    parser.add_argument("--merge", action="store_true", help="Use a global val/test set")
    parser.add_argument("--balance", action="store_true", help="Do not balance training data")
    parser.add_argument("--percent", type=float, default=1.0, help="subsample ratio")
    parser.add_argument(
        "--pure_avg",
        action="store_true",
        help="Use pure average, default is weighted avg, i.e., n_local_samples/n_all_samples",
    )

    args = parser.parse_args()
    return args


def parse_exp_name(args, exp_folder):
    if args.pure_avg:
        exp_folder = exp_folder+f'_pureAvg'
    if args.debug:
        exp_folder = exp_folder+'_debug'
    if args.subset < 1.0:
        exp_folder = exp_folder+ f'_subset{args.subset}'
    if args.pretrain:
        exp_folder = exp_folder+'_pretrained'
    if args.early:
        exp_folder = exp_folder+'_early_stop'
    if args.lr_decay>0:
        exp_folder = exp_folder+f'_LRdecay{args.lr_decay}'
    if args.balance:
        exp_folder = exp_folder+'_balance_train'
    
    return exp_folder