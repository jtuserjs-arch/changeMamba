import sys
import os
import random
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import argparse

from changedetection.script.script_utils import populate_name_lists
from changedetection.tasks import get_trainer


def set_seed(seed):
    """固定所有随机种子，保证可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Training on SYSU/LEVIR-CD+/WHU-CD dataset")
    parser.add_argument("--cfg", type=str, default=None)
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs.", default=None, nargs="+")
    parser.add_argument("--pretrained_weight_path", type=str)
    parser.add_argument("--dataset", type=str, default="SYSU")
    parser.add_argument("--type", type=str, default="train")
    parser.add_argument("--train_dataset_path", type=str)
    parser.add_argument("--train_data_list_path", type=str)
    parser.add_argument("--test_dataset_path", type=str)
    parser.add_argument("--test_data_list_path", type=str)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--train_data_name_list", type=list)
    parser.add_argument("--test_data_name_list", type=list)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--max_iters", type=int, default=20000)
    parser.add_argument("--model_type", type=str, default="ChangeMamba-BCD")
    parser.add_argument("--model_param_path", type=str, default="../saved_models")
    parser.add_argument("--resume", type=str)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument('--gate_mode', type=str, default='pixel',
                        choices=['none', 'image', 'pixel'],
                        help='Dynamic gate mode (none, image, pixel)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--use_uncertainty', action='store_true',
                        help='Enable uncertainty estimation head')
    
    parser.add_argument("--eval_interval", type=int, default=500, help="Evaluate every N iterations")
    args = parser.parse_args()

    # 固定随机种子
    if args.seed is not None:
        set_seed(args.seed)

    populate_name_lists(
        args,
        {
            "train_data_list_path": "train_data_name_list",
            "test_data_list_path": "test_data_name_list",
        },
    )
    get_trainer("bcd")(args).training()


if __name__ == "__main__":
    main()