import os
import json

import imageio
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
        torch.lgamma(alpha), dim=1, keepdim=True
    )
    lnB_beta = torch.sum(
        torch.lgamma(beta), dim=1, keepdim=True
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

    alpha:  [B, C, H, W]
    target: [B, H, W]
    """
    valid_mask = target != ignore_index

    if valid_mask.sum() == 0:
        return alpha.sum() * 0.0

    alpha = alpha.permute(0, 2, 3, 1)[valid_mask]  # [N, C]
    target = target[valid_mask]                    # [N]

    y = F.one_hot(target, num_classes=num_classes).float()

    S = torch.sum(alpha, dim=1, keepdim=True)

    data_fit = torch.sum(
        y * (torch.digamma(S) - torch.digamma(alpha)),
        dim=1,
    )

    # 只惩罚错误类别的 evidence
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
        labels = labels.to(self.device).long()

        use_uncertainty = getattr(self.args, "use_uncertainty", False)

        if use_uncertainty:
            output_dict = self.model(
                pre_change_imgs,
                post_change_imgs,
                return_aux=True,
            )

            output = output_dict["logits"]
            alpha = output_dict["alpha"]

            ce_loss = F.cross_entropy(output, labels, ignore_index=255)
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

        ce_loss = F.cross_entropy(output, labels, ignore_index=255)
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
                labels = labels.to(self.device).long()

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
        return eval_results["Validation"]["kappa"]

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

    def _save_metrics(self, iteration, split_name, metrics):
        record = {
            "iter": int(iteration),
            "split": split_name,
            "f1": float(metrics["f1"]),
            "iou": float(metrics["iou"]),
            "recall": float(metrics["recall"]),
            "precision": float(metrics["precision"]),
            "oa": float(metrics["oa"]),
            "kappa": float(metrics["kappa"]),
        }

        self.metrics_history.append(record)

        os.makedirs(self.args.model_param_path, exist_ok=True)

        save_path = os.path.join(
            self.args.model_param_path,
            "metrics_history.json",
        )

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics_history, f, indent=2)

    def training(self):
        """
        自定义训练循环：
        1. tqdm 显示训练进度；
        2. 每 eval_interval 次验证一次；
        3. 根据 Validation Kappa 保存 best_model.pth；
        4. 训练结束后在 Test 上评估。
        """
        args = self.args
        model = self.model
        train_loader = self.build_train_loader()
        optimizer = self.optimizer
        scheduler = self.scheduler

        best_score = -1.0
        best_iter = 0

        eval_loaders = self.build_eval_loaders()
        val_loader = eval_loaders.get("Validation", None)
        test_loader = eval_loaders.get("Test", None)

        os.makedirs(args.model_param_path, exist_ok=True)

        # 初始验证
        if val_loader is not None and getattr(args, "start_iter", 0) == 0:
            metrics = self.evaluate_loader("Validation", val_loader)
            self._save_metrics(0, "Validation", metrics)

            best_score = self.selection_metric({"Validation": metrics})
            best_iter = 0

            torch.save(
                model.state_dict(),
                os.path.join(args.model_param_path, "best_model.pth"),
            )

            self.emit_log(
                self.format_eval_result(
                    "Validation",
                    0,
                    args.max_iters,
                    metrics,
                )
            )

        model.train()

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

                    if score > best_score:
                        best_score = score
                        best_iter = iteration

                        torch.save(
                            model.state_dict(),
                            os.path.join(args.model_param_path, "best_model.pth"),
                        )

                    log_msg = self.format_eval_result(
                        "Validation",
                        iteration,
                        args.max_iters,
                        metrics,
                    )
                    self.emit_log(log_msg)

                    model.train()

        # 最终测试
        if test_loader is not None:
            test_metrics = self.evaluate_loader("Test", test_loader)
            self._save_metrics(args.max_iters, "Test", test_metrics)

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

        if getattr(self.args, "save_uncertainty", False):
            os.makedirs(self.uncertainty_saved_path, exist_ok=True)

        if getattr(self.args, "save_gate_weights", False):
            os.makedirs(self.gate_weights_path, exist_ok=True)

    def _save_change_maps(self, pred, names):
        """
        pred: [B, H, W], numpy array
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
        uncertainty: torch.Tensor, [B, 1, H, W] or [B, H, W]
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

    def _save_gate_weight_maps(self, gate_weights, output_size, names):
        """
        gate_weights: torch.Tensor, [B, 3, h, w]
        output_size: (H, W)
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

    def infer_batch(self, batch):
        pre_change_imgs, post_change_imgs, labels, names = batch

        pre_change_imgs = pre_change_imgs.to(self.device).float()
        post_change_imgs = post_change_imgs.to(self.device).float()
        labels = labels.to(self.device).long()

        save_uncertainty = getattr(self.args, "save_uncertainty", False)
        save_gate_weights = getattr(self.args, "save_gate_weights", False)
        use_uncertainty = getattr(self.args, "use_uncertainty", False)
        mc_samples = getattr(self.args, "mc_samples", 0)

        self.model.eval()

        with torch.no_grad():
            if mc_samples > 0:
                # 注意：你的仓库里 mc_dropout.py 在 utils_func 下
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

                pred = torch.argmax(output, dim=1).detach().cpu().numpy()
                output_size = output.shape[-2:]

        # 计算指标
        self.evaluator.add_batch(labels.cpu().numpy(), pred)

        # 保存变化检测结果
        self._save_change_maps(pred, names)

        # 保存 Evidential 或 MC 不确定性图
        if save_uncertainty:
            self._save_uncertainty_maps(uncertainty, names)

        # 保存门控权重图
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