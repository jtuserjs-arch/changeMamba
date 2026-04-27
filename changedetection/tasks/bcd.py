import os

import imageio
import torch
import torch.nn.functional as F
import json  
from tqdm import tqdm
import changedetection.utils_func.lovasz_loss as L
from changedetection.datasets import build_train_loader
from changedetection.engine import BaseInferer, BaseTrainer
from changedetection.evaluation import BinaryChangeEvaluator
from changedetection.logging_utils import format_log_block
from changedetection.models.ChangeMambaBCD import ChangeMambaBCD
from changedetection.script.script_utils import get_vssm_kwargs


class BCDTrainer(BaseTrainer):
    task_name = "bcd"

    def __init__(self, args):
        super().__init__(args)
        # 设置评估间隔（可以从命令行参数获取，默认为 500）
        self.eval_interval = getattr(args, 'eval_interval', 500)
        # 用于存储指标历史的列表
        self.metrics_history = []
        # 确保保存目录存在
        os.makedirs(self.args.model_param_path, exist_ok=True)

    def build_model(self, config):
        return ChangeMambaBCD(
            pretrained=self.args.pretrained_weight_path,
            gate_mode=getattr(self.args, 'gate_mode', 'pixel'),
            use_uncertainty=getattr(self.args, 'use_uncertainty', False),
            dropout_rate=0.2,
            **get_vssm_kwargs(config),
        )

    def build_train_loader(self):
        return build_train_loader(self.args)

    def build_eval_loaders(self):
        return self.build_runtime_eval_loaders()

    def train_step(self, batch):
        pre_change_imgs, post_change_imgs, labels, _ = batch
        pre_change_imgs = pre_change_imgs.to(self.device).float()
        post_change_imgs = post_change_imgs.to(self.device)
        labels = labels.to(self.device).long()

        output = self.model(pre_change_imgs, post_change_imgs)
        ce_loss = F.cross_entropy(output, labels, ignore_index=255)
        lovasz_loss = L.lovasz_softmax(F.softmax(output, dim=1), labels, ignore=255)
        final_loss = ce_loss + 0.75 * lovasz_loss
        return {
            "loss": final_loss,
            "log_items": {"loss": final_loss.item()},
        }

    def evaluate_loader(self, split_name, data_loader):
        evaluator = BinaryChangeEvaluator()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        with torch.no_grad():
            for pre_change_imgs, post_change_imgs, labels, _ in data_loader:
                pre_change_imgs = pre_change_imgs.to(self.device).float()
                post_change_imgs = post_change_imgs.to(self.device)
                labels = labels.to(self.device).long()

                output = self.model(pre_change_imgs, post_change_imgs)
                predictions = torch.argmax(output, dim=1).cpu().numpy()
                evaluator.add_batch(labels.cpu().numpy(), predictions)

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
            meta={"iter": best_record["iteration"], "score": best_record["score"]},
        )

    def _save_metrics(self, iteration, split_name, metrics):
        """保存当前评估结果到 JSON 文件"""
        record = {
            "iter": iteration,
            "split": split_name,
            "f1": metrics["f1"],
            "iou": metrics["iou"],
            "recall": metrics["recall"],
            "precision": metrics["precision"],
            "oa": metrics["oa"],
            "kappa": metrics["kappa"],
        }
        self.metrics_history.append(record)
        save_path = os.path.join(self.args.model_param_path, "metrics_history.json")
        with open(save_path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
    def training(self):
        """重写训练循环，带 tqdm 进度条，每 eval_interval 次评估并记录指标，保存最佳模型"""
        args = self.args
        model = self.model
        train_loader = self.build_train_loader()
        optimizer = self.optimizer
        scheduler = self.scheduler

        best_score = 0.0
        best_iter = 0

        # 获取验证集加载器
        eval_loaders = self.build_eval_loaders()
        val_loader = eval_loaders.get("Validation", None)
        test_loader = eval_loaders.get("Test", None)

        # 确保保存目录存在
        os.makedirs(args.model_param_path, exist_ok=True)

        # 初始评估（iteration=0）
        if val_loader is not None and args.start_iter == 0:
            metrics = self.evaluate_loader("Validation", val_loader)
            self._save_metrics(0, "Validation", metrics)
            best_score = self.selection_metric({"Validation": metrics})
            # 保存初始模型作为最佳（可选）
            torch.save(model.state_dict(), os.path.join(args.model_param_path, 'best_model.pth'))

        # 训练循环（带 tqdm 进度条）
        from tqdm import tqdm
        data_iter = iter(train_loader)
        total_iters = args.max_iters - args.start_iter
        with tqdm(total=total_iters, desc="Training", unit="iter") as pbar:
            for iteration in range(args.start_iter + 1, args.max_iters + 1):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    batch = next(data_iter)

                # 训练一步
                step_output = self.train_step(batch)
                loss = step_output["loss"]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                # 定期评估
                if val_loader is not None and iteration % self.eval_interval == 0:
                    metrics = self.evaluate_loader("Validation", val_loader)
                    self._save_metrics(iteration, "Validation", metrics)
                    score = self.selection_metric({"Validation": metrics})
                    if score > best_score:
                        best_score = score
                        best_iter = iteration
                        torch.save(model.state_dict(), os.path.join(args.model_param_path, 'best_model.pth'))
                    # 打印评估结果
                    log_msg = self.format_eval_result("Validation", iteration, args.max_iters, metrics)
                    self.emit_log(log_msg)

        # 最终测试评估
        if test_loader is not None:
            test_metrics = self.evaluate_loader("Test", test_loader)
            self._save_metrics(args.max_iters, "Test", test_metrics)
            self.emit_log(self.format_eval_result("Test", args.max_iters, args.max_iters, test_metrics))

        self.emit_log(f"Best validation score: {best_score:.4f} at iteration {best_iter}")
        self.emit_log(f"Best model saved to {os.path.join(args.model_param_path, 'best_model.pth')}")

class BCDInferer(BaseInferer):
    task_name = "bcd"

    def __init__(self, args):
        self.evaluator = BinaryChangeEvaluator()
        super().__init__(args)

    def build_model(self, config):
        return ChangeMambaBCD(
            pretrained=self.args.pretrained_weight_path,
            gate_mode=getattr(self.args, 'gate_mode', 'pixel'),
            use_uncertainty=getattr(self.args, 'use_uncertainty', False),
            **get_vssm_kwargs(config),
        )

    def build_data_loader(self):
        return self.build_runtime_data_loader()

    def prepare_output_dirs(self):
        self.change_map_saved_path = os.path.join(
        self.args.result_saved_path,
        self.args.dataset,
        self.args.model_type,
        "change_map"
        )
        os.makedirs(self.change_map_saved_path, exist_ok=True)
        if self.args.save_uncertainty:
            self.uncertainty_saved_path = os.path.join(self.args.result_saved_path, self.args.dataset, self.args.model_type, "uncertainty")
            os.makedirs(self.uncertainty_saved_path, exist_ok=True)
        if self.args.save_gate_weights:
            self.gate_weights_path = os.path.join(self.args.result_saved_path, self.args.dataset, self.args.model_type, "gate_weights")
            os.makedirs(self.gate_weights_path, exist_ok=True)
    def infer_batch(self, batch):
        pre_change_imgs, post_change_imgs, labels, names = batch
        pre_change_imgs = pre_change_imgs.to(self.device).float()
        post_change_imgs = post_change_imgs.to(self.device)
        labels = labels.to(self.device).long()

        if self.args.mc_samples > 0:
            # MC Dropout 推理
            from changedetection.utils.mc_dropout import mc_dropout_inference
            pred, uncertainty, gate_weights = mc_dropout_inference(
                self.model, pre_change_imgs, post_change_imgs,
                num_samples=self.args.mc_samples,
                return_uncertainty=True
            )
            # 转换为 numpy 数组
            pred = pred.cpu().numpy()
            # 保存不确定性图
            if self.args.save_uncertainty:
                for i, name in enumerate(names):
                    unc_map = uncertainty[i].cpu().numpy()
                    unc_img = (unc_map * 255).astype('uint8')
                    save_path = os.path.join(self.uncertainty_saved_path, f"{os.path.splitext(name)[0]}.png")
                    imageio.imwrite(save_path, unc_img)
            # 保存门控权重图
            if self.args.save_gate_weights and gate_weights is not None:
                for i, name in enumerate(names):
                    for j, ch_name in enumerate(['seq', 'cross', 'par']):
                        gw = gate_weights[i, j].cpu().numpy()
                        gw_img = (gw * 255).astype('uint8')
                        save_path = os.path.join(self.gate_weights_path, f"{os.path.splitext(name)[0]}_{ch_name}.png")
                        imageio.imwrite(save_path, gw_img)
        else:
            # 常规推理
            output = self.model(pre_change_imgs, post_change_imgs)
       
            pred = torch.argmax(output, dim=1).cpu().numpy()

        # 计算指标
        self.evaluator.add_batch(labels.cpu().numpy(), pred)
        # 保存变化检测结果
        binary_change_map = pred.squeeze().astype("uint8")
        binary_change_map[binary_change_map == 1] = 255
        image_name = os.path.splitext(names[0])[0] + ".png"
        imageio.imwrite(os.path.join(self.change_map_saved_path, image_name), binary_change_map)
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
