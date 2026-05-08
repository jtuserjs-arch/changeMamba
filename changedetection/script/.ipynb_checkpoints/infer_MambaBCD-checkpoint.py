import sys
import os
import argparse

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
    ),
)

from changedetection.script.script_utils import populate_name_lists
from changedetection.tasks import get_inferer


def main():
    parser = argparse.ArgumentParser(
        description="Inference on SYSU/LEVIR-CD+/WHU-CD dataset"
    )

    parser.add_argument("--cfg", type=str, default=None)
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs.",
        default=None,
        nargs="+",
    )

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

    # Dynamic gate
    parser.add_argument(
        "--gate_mode",
        type=str,
        default="pixel",
        choices=["none", "image", "pixel"],
        help="Dynamic gate mode. Use 'none' for original ChangeMamba decoder.",
    )

    # Uncertainty
    parser.add_argument(
        "--use_uncertainty",
        action="store_true",
        help="Enable evidential uncertainty head.",
    )

    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.0,
        help="Dropout rate before final classifier head.",
    )

    parser.add_argument(
        "--mc_samples",
        type=int,
        default=0,
        help="MC Dropout sampling times. 0 means normal inference.",
    )

    parser.add_argument(
        "--save_uncertainty",
        action="store_true",
        help="Save uncertainty maps.",
    )

    parser.add_argument(
        "--save_gate_weights",
        action="store_true",
        help="Save dynamic gate weight maps.",
    )
    parser.add_argument(
        "--save_prob",
        action="store_true",
        help="Save changed-class probability maps.",
    )
    parser.add_argument(
        "--change_threshold",
        type=float,
        default=0.5,
        help="Threshold for changed-class probability.",
    )

    parser.add_argument(
        "--perturb_type",
        type=str,
        default="none",
        choices=["none", "noise", "blur", "shift"],
        help="Perturbation type for robustness evaluation.",
    )

    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.03,
        help="Gaussian noise standard deviation.",
    )

    parser.add_argument(
        "--blur_kernel",
        type=int,
        default=5,
        help="Gaussian blur kernel size.",
    )

    parser.add_argument(
        "--shift_pixels",
        type=int,
        default=4,
        help="Shift pixels for misregistration robustness test.",
    )

    args = parser.parse_args()

    populate_name_lists(
        args,
        {
            "test_data_list_path": "test_data_name_list",
        },
    )

    get_inferer("bcd")(args).infer()


if __name__ == "__main__":
    main()