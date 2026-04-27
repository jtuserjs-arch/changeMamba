import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import argparse

from changedetection.script.script_utils import populate_name_lists
from changedetection.tasks import get_inferer


def main():
    parser = argparse.ArgumentParser(description="Inference on SYSU/LEVIR-CD+/WHU-CD dataset")
    parser.add_argument("--cfg", type=str, default=None)
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs.", default=None, nargs="+")
    parser.add_argument("--pretrained_weight_path", type=str)
    parser.add_argument("--dataset", type=str, default="LEVIR-CD+")
    parser.add_argument("--test_dataset_path", type=str)
    parser.add_argument("--test_data_list_path", type=str)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--test_data_name_list", type=list)
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--model_type", type=str, default="MambaBCD_Tiny")
    parser.add_argument("--result_saved_path", type=str, default="../results")
    parser.add_argument("--resume", type=str)
    parser.add_argument('--mc_samples', type=int, default=0,
                        help='MC Dropout 采样次数，0 表示不使用（常规推理）')
    parser.add_argument('--save_uncertainty', action='store_true',
                        help='保存不确定性图（需要 mc_samples > 0）')
    parser.add_argument('--save_gate_weights', action='store_true',
                        help='保存门控权重图（需要 gate_mode != none）')
    # ========== 新增参数 ==========
    parser.add_argument('--gate_mode', type=str, default='pixel', choices=['none', 'image', 'pixel'],
                        help='Dynamic gate mode (none, image, pixel)')
    parser.add_argument('--use_uncertainty', action='store_true',
                        help='Enable uncertainty estimation head')
    # ==============================
    args = parser.parse_args()
    populate_name_lists(args, {"test_data_list_path": "test_data_name_list"})
    get_inferer("bcd")(args).infer()


if __name__ == "__main__":
    main()