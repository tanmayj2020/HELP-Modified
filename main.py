# Importing argument parser
from parse import get_parser
from utils import set_seed , set_gpu , set_path
from help import HELP


def main(args):
    # Setting the seed for result reproducibility 
    set_seed(args)
    # Setting the visible GPUs
    args = set_gpu(args)
    args = set_path(args)
    print(f'==> mode is [{args.mode}] ...')
    # Creating the model class
    model = HELP(args)

    if args.mode == "meta-train":
        model.meta_train()
    elif args.mode == "meta-test":
        model.test_predictor()

if __name__ == "__main__":
    main(get_parser())