import os
import json
import time
import datetime

import imageio.v2 as imageio
import torch
import torch.nn.functional as F
from tqdm import tqdm

import changedetection.utils_func.lovasz_loss as L
from changedetection.datasets import build_train_loader
from changedetection.engine import BaseInferer, BaseTrainer
from changedetection.evaluation import BinaryChangeEvaluator
from changedetection.logging_utils import format_log_block
from changedetection.models.ChangeMambaBCD import ChangeMambaBCD
from changedetection.script.script_utils import get_vssm_kwargs


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------
def _parse_interaction_stages(value, default=(2, 3)):
    """
    Accept:
        None
        int
        list / tuple, e.g. [2, 3]
        string, e.g. "2 3" or "2,3"
    Return:
        tuple of int
    """
    if value is None:
        return tuple(default)

    if isinstance(value, int):
        return (value,)

    if isinstance(value, str):
        value = value.replace(",", " ").split()
        if len(value) == 0:
            return tuple(default)
        return tuple(int(v) for v in value)

    return tuple(int(v) for v in value)

    
def _squeeze_label(labels):
    """
    Support labels in [B, H, W] or [B, 1, H, W].
    """
    if labels.dim() == 4 and labels.shape[1] == 1:
        labels = labels[:, 0]
    return labels


def _safe_float(value):
    """
    Convert torch / numpy / python scalar to json-safe float.
    """
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _metrics_to_jsonable(metrics):
    """
    Convert metric dict to pure python floats.
    """
    return {
        "recall": _safe_float(metrics["recall"]),
        "precision": _safe_float(metrics["precision"]),
        "oa": _safe_float(metrics["oa"]),
        "f1": _safe_float(metrics["f1"]),
        "iou": _safe_float(metrics["iou"]),
        "kappa": _safe_float(metrics["kappa"]),
    }


# ------------------------------------------------------------
# Evidential Uncertainty Loss
# ------------------------------------------------------------
def dirichlet_kl(alpha, num_classes=2):
    """
    KL(Dir(alpha) || Dir(1))

    alpha: [N, C]
    """
    beta = torch.ones_like(alpha)

    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    sum_beta = torch.sum(beta, dim=1, keepdim=True)

    lnB_alpha = torch.lgamma(sum_alpha) - torch.sum(
        torch.lgamma(alpha),
        dim=1,
        keepdim=True,
    )
    lnB_beta = torch.sum(
        torch.lgamma(beta),
        dim=1,
        keepdim=True,
    ) - torch.lgamma(sum_beta)

    digamma_alpha = torch.digamma(alpha)
    digamma_sum_alpha = torch.digamma(sum_alpha)

    kl = torch.sum(
        (alpha - beta) * (digamma_alpha - digamma_sum_alpha),
        dim=1,
        keepdim=True,
    ) + lnB_alpha + lnB_beta

    return kl.squeeze(1)


def edl_digamma_loss(alpha, target, num_classes=2, ignore_index=255):
    """
    Evidential deep learning loss.

    alpha:
        [B, C, H, W]
    target:
        [B, H, W]
    """
    target = _squeeze_label(target)
    valid_mask = target != ignore_index

    if valid_mask.sum() == 0:
        return alpha.sum() * 0.0

    alpha = alpha.permute(0, 2, 3, 1)[valid_mask]  # [N, C]
    target = target[valid_mask]  # [N]

    y = F.one_hot(target, num_classes=num_classes).float()

    evidence_sum = torch.sum(alpha, dim=1, keepdim=True)
    data_fit = torch.sum(
        y * (torch.digamma(evidence_sum) - torch.digamma(alpha)),
        dim=1,
    )

    # Only regularize evidence of incorrect classes.
    alpha_tilde = y + (1.0 - y) * alpha
    kl = dirichlet_kl(alpha_tilde, num_classes=num_classes)

    return data_fit.mean() + 0.01 * kl.mean()


# ------------------------------------------------------------
# Trainer
# ------------------------------------------------------------
class BCDTrainer(BaseTrainer):
    task_name = "bcd"

    def __init__(self, args):
        super().__init__(args)

        self.eval_interval = getattr(args, "eval_interval", 500)
        self.metrics_history = []

        os.makedirs(self.args.model_param_path, exist_ok=True)

    def build_model(self, config):
        return ChangeMambaBCD(
            pretrained=self.args.pretrained_weight_path,
            gate_mode=getattr(self.args, "gate_mode", "pixel"),
            use_uncertainty=getattr(self.args, "use_uncertainty", False),
            dropout_rate=getattr(self.args, "dropout_rate", 0.2),
            interaction_mode=getattr(self.args, "interaction_mode", "cafim"),
            interaction_stages=_parse_interaction_stages(
                getattr(self.args, "interaction_stages", (2, 3)),
                default=(2, 3),
            ),
            interaction_reduction=getattr(self.args, "interaction_reduction", 4),
            **get_vssm_kwargs(config),
        )

    def build_train_loader(self):
        return build_train_loader(self.args)

    def build_eval_loaders(self):
        return self.build_runtime_eval_loaders()

    def train_step(self, batch):
        pre_change_imgs, post_change_imgs, labels, _ = batch

        pre_change_imgs = pre_change_imgs.to(self.device).float()
        post_change_imgs = post_change_imgs.to(self.device).float()
        labels = _squeeze_label(labels.to(self.device).long())

        use_uncertainty = getattr(self.args, "use_uncertainty", False)

        if use_uncertainty:
            output_dict = self.model(
                pre_change_imgs,
                post_change_imgs,
                return_aux=True,
            )

            output = output_dict["logits"]
            alpha = output_dict["alpha"]

            ce_loss = F.cross_entropy(
                output,
                labels,
                ignore_index=255,
            )
            lovasz_loss = L.lovasz_softmax(
                F.softmax(output, dim=1),
                labels,
                ignore=255,
            )
            uncertainty_loss = edl_digamma_loss(
                alpha,
                labels,
                num_classes=2,
                ignore_index=255,
            )

            uncertainty_weight = getattr(self.args, "uncertainty_weight", 0.01)

            final_loss = (
                ce_loss
                + 0.75 * lovasz_loss
                + uncertainty_weight * uncertainty_loss
            )

            return {
                "loss": final_loss,
                "log_items": {
                    "loss": final_loss.item(),
                    "ce": ce_loss.item(),
                    "lovasz": lovasz_loss.item(),
                    "uncertainty": uncertainty_loss.item(),
                },
            }

        output = self.model(pre_change_imgs, post_change_imgs)

        ce_loss = F.cross_entropy(
            output,
            labels,
            ignore_index=255,
        )
        lovasz_loss = L.lovasz_softmax(
            F.softmax(output, dim=1),
            labels,
            ignore=255,
        )

        final_loss = ce_loss + 0.75 * lovasz_loss

        return {
            "loss": final_loss,
            "log_items": {
                "loss": final_loss.item(),
                "ce": ce_loss.item(),
                "lovasz": lovasz_loss.item(),
            },
        }

    def evaluate_loader(self, split_name, data_loader):
        evaluator = BinaryChangeEvaluator()

        was_training = self.model.training
        self.model.eval()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        with torch.no_grad():
            for pre_change_imgs, post_change_imgs, labels, _ in data_loader:
                pre_change_imgs = pre_change_imgs.to(self.device).float()
                post_change_imgs = post_change_imgs.to(self.device).float()
                labels = _squeeze_label(labels.to(self.device).long())

                output = self.model(pre_change_imgs, post_change_imgs)
                predictions = torch.argmax(output, dim=1).cpu().numpy()

                evaluator.add_batch(labels.cpu().numpy(), predictions)

        if was_training:
            self.model.train()

        metrics = evaluator.compute()

        return {
            "recall": metrics.recall,
            "precision": metrics.precision,
            "oa": metrics.oa,
            "f1": metrics.f1,
            "iou": metrics.iou,
            "kappa": metrics.kappa,
        }

    def selection_metric(self, eval_results):
        return eval_results["Validation"]["f1"]

    def format_eval_result(self, split_name, iteration, total_iterations, metrics):
        return format_log_block(
            f"EVAL {split_name}",
            {
                "Recall": metrics["recall"],
                "Precision": metrics["precision"],
                "OA": metrics["oa"],
                "F1": metrics["f1"],
                "IoU": metrics["iou"],
                "Kappa": metrics["kappa"],
            },
            meta={"iter": f"{iteration}/{total_iterations}"},
        )

    def format_best_result(self, best_record):
        metrics = best_record["results"]["Validation"]

        return format_log_block(
            "BEST Validation",
            {
                "Recall": metrics["recall"],
                "Precision": metrics["precision"],
                "OA": metrics["oa"],
                "F1": metrics["f1"],
                "IoU": metrics["iou"],
                "Kappa": metrics["kappa"],
            },
            meta={
                "iter": best_record["iteration"],
                "score": best_record["score"],
            },
        )

    def _build_metric_record(self, iteration, split_name, metrics):
        now = time.time()
        elapsed_seconds = now - getattr(self, "train_start_time", now)

        if iteration > 0:
            seconds_per_iter = elapsed_seconds / iteration
            remaining_iters = self.args.max_iters - iteration
            eta_seconds = remaining_iters * seconds_per_iter
            eta_time = datetime.datetime.fromtimestamp(now + eta_seconds).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        else:
            seconds_per_iter = None
            eta_seconds = None
            eta_time = None

        metrics = _metrics_to_jsonable(metrics)

        record = {
            "iter": int(iteration),
            "split": split_name,
            "f1": metrics["f1"],
            "iou": metrics["iou"],
            "recall": metrics["recall"],
            "precision": metrics["precision"],
            "oa": metrics["oa"],
            "kappa": metrics["kappa"],
            "elapsed_seconds": elapsed_seconds,
            "seconds_per_iter": seconds_per_iter,
            "eta_seconds": eta_seconds,
            "estimated_finish_time": eta_time,
        }

        return record

    def _save_metrics(self, iteration, split_name, metrics):
        """
        Save metric history of every evaluation point.
        """
        record = self._build_metric_record(iteration, split_name, metrics)
        self.metrics_history.append(record)

        save_path = os.path.join(
            self.args.model_param_path,
            "metrics_history.json",
        )
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics_history, f, indent=2)

    def _make_best_record(self, iteration, score, metrics):
        """
        Build a complete best-record dict for the current best Validation result.
        """
        return {
            "iteration": int(iteration),
            "score": float(score),
            "selection_metric": "Validation/F1",
            "results": {
                "Validation": _metrics_to_jsonable(metrics),
            },
        }

    def _save_best_metrics(self, best_record):
        """
        Save the complete metric values of the best Validation checkpoint.
        Files:
            best_metrics.json
            best_metrics.txt
        """
        os.makedirs(self.args.model_param_path, exist_ok=True)

        save_json_path = os.path.join(
            self.args.model_param_path,
            "best_metrics.json",
        )
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(best_record, f, indent=2)

        metrics = best_record["results"]["Validation"]

        save_txt_path = os.path.join(
            self.args.model_param_path,
            "best_metrics.txt",
        )
        with open(save_txt_path, "w", encoding="utf-8") as f:
            f.write("BEST Validation Metrics\n")
            f.write(f"Iter      : {best_record['iteration']}\n")
            f.write(f"Metric    : {best_record['selection_metric']}\n")
            f.write(f"Best F1   : {metrics['f1']:.6f}\n")
            f.write(f"Recall    : {metrics['recall']:.6f}\n")
            f.write(f"Precision : {metrics['precision']:.6f}\n")
            f.write(f"OA        : {metrics['oa']:.6f}\n")
            f.write(f"IoU       : {metrics['iou']:.6f}\n")
            f.write(f"Kappa     : {metrics['kappa']:.6f}\n")

    def _save_final_test_metrics(self, test_metrics):
        """
        Save final Test metrics. When best_model.pth exists, final Test uses
        the best Validation checkpoint.
        """
        metrics = _metrics_to_jsonable(test_metrics)

        save_json_path = os.path.join(
            self.args.model_param_path,
            "final_test_metrics.json",
        )
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump({"Test": metrics}, f, indent=2)

        save_txt_path = os.path.join(
            self.args.model_param_path,
            "final_test_metrics.txt",
        )
        with open(save_txt_path, "w", encoding="utf-8") as f:
            f.write("FINAL Test Metrics\n")
            f.write("Checkpoint : best_model.pth if available, otherwise last iteration model\n")
            f.write(f"F1        : {metrics['f1']:.6f}\n")
            f.write(f"Recall    : {metrics['recall']:.6f}\n")
            f.write(f"Precision : {metrics['precision']:.6f}\n")
            f.write(f"OA        : {metrics['oa']:.6f}\n")
            f.write(f"IoU       : {metrics['iou']:.6f}\n")
            f.write(f"Kappa     : {metrics['kappa']:.6f}\n")

    def _load_best_model_for_final_test(self, model):
        """
        Load best_model.pth before final Test evaluation.
        Return True if loaded successfully.
        """
        best_model_path = os.path.join(
            self.args.model_param_path,
            "best_model.pth",
        )

        if not os.path.exists(best_model_path):
            self.emit_log(
                "best_model.pth not found. Final Test will use the last iteration model."
            )
            return False

        checkpoint = torch.load(best_model_path, map_location=self.device)

        # Compatible with raw state_dict and common checkpoint formats.
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=True)

        self.emit_log(f"Loaded best model for final Test: {best_model_path}")
        return True

    def training(self):
        """
        Custom training loop:
            1. show tqdm progress;
            2. evaluate every eval_interval iterations;
            3. save best_model.pth according to Validation F1;
            4. save best_metrics.json / best_metrics.txt;
            5. evaluate Test split using best_model.pth after training.
        """
        args = self.args
        model = self.model
        train_loader = self.build_train_loader()
        optimizer = self.optimizer
        scheduler = self.scheduler

        best_score = -1.0
        best_iter = 0
        best_record = None

        eval_loaders = self.build_eval_loaders()
        val_loader = eval_loaders.get("Validation", None)
        test_loader = eval_loaders.get("Test", None)

        os.makedirs(args.model_param_path, exist_ok=True)

        # Initial validation.
        if val_loader is not None and getattr(args, "start_iter", 0) == 0:
            metrics = self.evaluate_loader("Validation", val_loader)
            self._save_metrics(0, "Validation", metrics)

            best_score = self.selection_metric({"Validation": metrics})
            best_iter = 0
            best_record = self._make_best_record(
                iteration=best_iter,
                score=best_score,
                metrics=metrics,
            )

            torch.save(
                model.state_dict(),
                os.path.join(args.model_param_path, "best_model.pth"),
            )
            self._save_best_metrics(best_record)

            log_msg = self.format_eval_result(
                "Validation",
                0,
                args.max_iters,
                metrics,
            )
            log_msg += "\n" + self.format_best_result(best_record)
            log_msg += "\n[NEW BEST] Best model and best_metrics files have been updated."
            self.emit_log(log_msg)

        model.train()
        self.train_start_time = time.time()

        data_iter = iter(train_loader)
        total_iters = args.max_iters - getattr(args, "start_iter", 0)

        with tqdm(total=total_iters, desc="Training", unit="iter") as pbar:
            for iteration in range(
                getattr(args, "start_iter", 0) + 1,
                args.max_iters + 1,
            ):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    batch = next(data_iter)

                model.train()

                step_output = self.train_step(batch)
                loss = step_output["loss"]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                pbar.update(1)

                log_items = step_output.get("log_items", {})
                postfix = {
                    key: f"{value:.4f}"
                    for key, value in log_items.items()
                    if isinstance(value, float)
                }
                pbar.set_postfix(postfix)

                if val_loader is not None and iteration % self.eval_interval == 0:
                    metrics = self.evaluate_loader("Validation", val_loader)
                    self._save_metrics(iteration, "Validation", metrics)

                    score = self.selection_metric({"Validation": metrics})
                    is_best = score > best_score

                    if is_best:
                        best_score = score
                        best_iter = iteration
                        best_record = self._make_best_record(
                            iteration=best_iter,
                            score=best_score,
                            metrics=metrics,
                        )

                        torch.save(
                            model.state_dict(),
                            os.path.join(args.model_param_path, "best_model.pth"),
                        )
                        self._save_best_metrics(best_record)

                    log_msg = self.format_eval_result(
                        "Validation",
                        iteration,
                        args.max_iters,
                        metrics,
                    )

                    if best_record is not None:
                        log_msg += "\n" + self.format_best_result(best_record)

                    if is_best:
                        log_msg += (
                            "\n[NEW BEST] Best model and best_metrics files "
                            "have been updated."
                        )

                    self.emit_log(log_msg)

                    model.train()

        # Final test: use the best Validation checkpoint when available.
        if test_loader is not None:
            self._load_best_model_for_final_test(model)
            test_metrics = self.evaluate_loader("Test", test_loader)
            self._save_metrics(args.max_iters, "Test", test_metrics)
            self._save_final_test_metrics(test_metrics)

            self.emit_log(
                self.format_eval_result(
                    "Test",
                    args.max_iters,
                    args.max_iters,
                    test_metrics,
                )
            )

        self.emit_log(
            f"Best validation score: {best_score:.4f} at iteration {best_iter}"
        )
        self.emit_log(
            f"Best model saved to "
            f"{os.path.join(args.model_param_path, 'best_model.pth')}"
        )
        self.emit_log(
            f"Best metrics saved to "
            f"{os.path.join(args.model_param_path, 'best_metrics.txt')}"
        )

# ------------------------------------------------------------
# Inferer
# ------------------------------------------------------------
class BCDInferer(BaseInferer):
    task_name = "bcd"

    def __init__(self, args):
        self.evaluator = BinaryChangeEvaluator()
        super().__init__(args)

    def build_model(self, config):
        return ChangeMambaBCD(
            pretrained=self.args.pretrained_weight_path,
            gate_mode=getattr(self.args, "gate_mode", "pixel"),
            use_uncertainty=getattr(self.args, "use_uncertainty", False),
            dropout_rate=getattr(self.args, "dropout_rate", 0.0),
            interaction_mode=getattr(self.args, "interaction_mode", "cafim"),
            interaction_stages=_parse_interaction_stages(
                getattr(self.args, "interaction_stages", (2, 3)),
                default=(2, 3),
            ),
            interaction_reduction=getattr(self.args, "interaction_reduction", 4),
            **get_vssm_kwargs(config),
        )

    def build_data_loader(self):
        return self.build_runtime_data_loader()

    def prepare_output_dirs(self):
        self.change_map_saved_path = os.path.join(
            self.args.result_saved_path,
            self.args.dataset,
            self.args.model_type,
            "change_map",
        )
        os.makedirs(self.change_map_saved_path, exist_ok=True)

        self.uncertainty_saved_path = os.path.join(
            self.args.result_saved_path,
            self.args.dataset,
            self.args.model_type,
            "uncertainty",
        )
        self.gate_weights_path = os.path.join(
            self.args.result_saved_path,
            self.args.dataset,
            self.args.model_type,
            "gate_weights",
        )
        self.prob_saved_path = os.path.join(
            self.args.result_saved_path,
            self.args.dataset,
            self.args.model_type,
            "prob_change",
        )

        if getattr(self.args, "save_prob", False):
            os.makedirs(self.prob_saved_path, exist_ok=True)

        if getattr(self.args, "save_uncertainty", False):
            os.makedirs(self.uncertainty_saved_path, exist_ok=True)

        if getattr(self.args, "save_gate_weights", False):
            os.makedirs(self.gate_weights_path, exist_ok=True)

    def _save_change_maps(self, pred, names):
        """
        pred:
            numpy array, [B, H, W]
        """
        for i, name in enumerate(names):
            binary_change_map = pred[i].astype("uint8")
            binary_change_map[binary_change_map == 1] = 255

            image_name = os.path.splitext(name)[0] + ".png"
            save_path = os.path.join(
                self.change_map_saved_path,
                image_name,
            )
            imageio.imwrite(save_path, binary_change_map)

    def _save_uncertainty_maps(self, uncertainty, names):
        """
        uncertainty:
            torch.Tensor, [B, 1, H, W] or [B, H, W]
        """
        if uncertainty is None:
            return

        if uncertainty.dim() == 4:
            uncertainty = uncertainty.squeeze(1)

        uncertainty = uncertainty.detach().cpu().float().clamp(0, 1).numpy()

        for i, name in enumerate(names):
            unc_map = uncertainty[i]
            unc_img = (unc_map * 255).clip(0, 255).astype("uint8")

            image_name = os.path.splitext(name)[0] + ".png"
            save_path = os.path.join(
                self.uncertainty_saved_path,
                image_name,
            )
            imageio.imwrite(save_path, unc_img)

    def _save_prob_maps(self, prob_change, names):
        """
        prob_change:
            torch.Tensor, [B, H, W]

        Save as uint16 png, value range [0, 65535].
        """
        prob_change = prob_change.detach().cpu().float().clamp(0, 1).numpy()

        for i, name in enumerate(names):
            prob_img = (prob_change[i] * 65535).clip(0, 65535).astype("uint16")

            image_name = os.path.splitext(name)[0] + ".png"
            save_path = os.path.join(
                self.prob_saved_path,
                image_name,
            )
            imageio.imwrite(save_path, prob_img)

    def _save_gate_weight_maps(self, gate_weights, output_size, names):
        """
        gate_weights:
            torch.Tensor, [B, 3, h, w] or [B, 3, 1, 1]

        output_size:
            (H, W)
        """
        if gate_weights is None:
            return

        if gate_weights.shape[-2:] != output_size:
            gate_weights = F.interpolate(
                gate_weights,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

        gate_weights = gate_weights.detach().cpu().float().clamp(0, 1).numpy()
        gate_names = ["seq", "cross", "par"]

        for i, name in enumerate(names):
            base_name = os.path.splitext(name)[0]

            for j, gate_name in enumerate(gate_names):
                if j >= gate_weights.shape[1]:
                    continue

                gate_img = (gate_weights[i, j] * 255).clip(0, 255).astype("uint8")
                save_path = os.path.join(
                    self.gate_weights_path,
                    f"{base_name}_{gate_name}.png",
                )
                imageio.imwrite(save_path, gate_img)

    def _apply_perturbation(self, pre_change_imgs, post_change_imgs):
        perturb_type = getattr(self.args, "perturb_type", "none")

        if perturb_type == "none":
            return pre_change_imgs, post_change_imgs

        if perturb_type == "noise":
            noise_std = getattr(self.args, "noise_std", 0.03)

            pre_change_imgs = pre_change_imgs + torch.randn_like(pre_change_imgs) * noise_std
            post_change_imgs = post_change_imgs + torch.randn_like(post_change_imgs) * noise_std

            pre_change_imgs = pre_change_imgs.clamp(0, 1)
            post_change_imgs = post_change_imgs.clamp(0, 1)

            return pre_change_imgs, post_change_imgs

        if perturb_type == "blur":
            blur_kernel = getattr(self.args, "blur_kernel", 5)

            pre_change_imgs = self._gaussian_blur_tensor(pre_change_imgs, blur_kernel)
            post_change_imgs = self._gaussian_blur_tensor(post_change_imgs, blur_kernel)

            return pre_change_imgs, post_change_imgs

        if perturb_type == "shift":
            shift_pixels = getattr(self.args, "shift_pixels", 4)

            # Only shift post image to simulate bi-temporal misregistration.
            post_change_imgs = self._shift_tensor_zero_pad(
                post_change_imgs,
                shift_pixels,
            )

            return pre_change_imgs, post_change_imgs

        raise ValueError(f"Unsupported perturb_type: {perturb_type}")

    def _gaussian_blur_tensor(self, x, kernel_size=5, sigma=1.0):
        if kernel_size <= 1:
            return x

        if kernel_size % 2 == 0:
            kernel_size += 1

        device = x.device
        dtype = x.dtype

        coords = torch.arange(kernel_size, device=device, dtype=dtype)
        coords = coords - (kernel_size - 1) / 2.0

        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()

        kernel_2d = g[:, None] * g[None, :]
        kernel_2d = kernel_2d.expand(
            x.shape[1],
            1,
            kernel_size,
            kernel_size,
        )

        padding = kernel_size // 2

        return F.conv2d(
            x,
            kernel_2d,
            padding=padding,
            groups=x.shape[1],
        )

    def _shift_tensor_zero_pad(self, x, shift_pixels=4):
        if shift_pixels == 0:
            return x

        b, c, h, w = x.shape
        out = torch.zeros_like(x)

        s = abs(shift_pixels)
        if s >= h or s >= w:
            return out

        if shift_pixels > 0:
            out[:, :, s:, s:] = x[:, :, : h - s, : w - s]
        else:
            out[:, :, : h - s, : w - s] = x[:, :, s:, s:]

        return out

    def infer_batch(self, batch):
        pre_change_imgs, post_change_imgs, labels, names = batch

        pre_change_imgs = pre_change_imgs.to(self.device).float()
        post_change_imgs = post_change_imgs.to(self.device).float()
        labels = _squeeze_label(labels.to(self.device).long())

        save_uncertainty = getattr(self.args, "save_uncertainty", False)
        save_gate_weights = getattr(self.args, "save_gate_weights", False)
        save_prob = getattr(self.args, "save_prob", False)
        use_uncertainty = getattr(self.args, "use_uncertainty", False)

        mc_samples = getattr(self.args, "mc_samples", 0)
        change_threshold = getattr(self.args, "change_threshold", 0.5)

        # Robustness perturbation: none / noise / blur / shift.
        pre_change_imgs, post_change_imgs = self._apply_perturbation(
            pre_change_imgs,
            post_change_imgs,
        )

        self.model.eval()

        prob_change = None
        uncertainty = None
        gate_weights = None

        with torch.no_grad():
            if mc_samples > 0:
                # MC Dropout branch. The model forward supports return_uncertainty
                # for compatibility with older mc_dropout.py.
                from changedetection.utils_func.mc_dropout import mc_dropout_inference

                pred_tensor, uncertainty, gate_weights = mc_dropout_inference(
                    self.model,
                    pre_change_imgs,
                    post_change_imgs,
                    num_samples=mc_samples,
                    return_uncertainty=True,
                )

                if isinstance(pred_tensor, torch.Tensor):
                    pred = pred_tensor.detach().cpu().numpy()
                else:
                    pred = pred_tensor

                output_size = labels.shape[-2:]

            else:
                if use_uncertainty or save_gate_weights:
                    output_dict = self.model(
                        pre_change_imgs,
                        post_change_imgs,
                        return_aux=True,
                    )
                    output = output_dict["logits"]
                    uncertainty = output_dict.get("uncertainty", None)
                    gate_weights = output_dict.get("gate_weights", None)
                else:
                    output = self.model(pre_change_imgs, post_change_imgs)
                    uncertainty = None
                    gate_weights = None

                # Save probability map and threshold prediction by changed-class probability.
                prob_change = F.softmax(output, dim=1)[:, 1]
                pred = (prob_change > change_threshold).long().detach().cpu().numpy()
                output_size = output.shape[-2:]

        # Metrics.
        self.evaluator.add_batch(labels.cpu().numpy(), pred)

        # Save binary change map.
        self._save_change_maps(pred, names)

        # Save changed-class probability for ECE / Risk-Coverage.
        if save_prob and prob_change is not None:
            self._save_prob_maps(prob_change, names)

        # Save evidential or MC uncertainty map.
        if save_uncertainty:
            self._save_uncertainty_maps(uncertainty, names)

        # Save dynamic gate weight maps.
        if save_gate_weights:
            self._save_gate_weight_maps(gate_weights, output_size, names)

    def finish(self):
        metrics = self.evaluator.compute()

        self.emit_log(
            format_log_block(
                "INFER Summary",
                {
                    "Recall": metrics.recall,
                    "Precision": metrics.precision,
                    "OA": metrics.oa,
                    "F1": metrics.f1,
                    "IoU": metrics.iou,
                    "Kappa": metrics.kappa,
                },
            )
        )
        self.emit_log("Inference stage is done!")
