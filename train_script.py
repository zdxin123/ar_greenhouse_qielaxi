import argparse
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


LABEL_COLUMN = "days_to_harvest"
IMAGE_COLUMN = "image_path"
DROP_COLUMNS = ["date", "time_slot", "种植日期", "设备名称"]
NUMERIC_COLUMNS = [
    "光照",
    "土壤PH",
    "土壤温度",
    "土壤湿度",
    "氮",
    "磷",
    "空气温度",
    "空气湿度",
    "钾",
    "已种植天数",
    LABEL_COLUMN,
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train the greenhouse multimodal days-to-harvest model.")
    parser.add_argument("--data", default="final_train_data_ready.csv", help="Training CSV path.")
    parser.add_argument("--model-path", default="ag_models/days_predictor", help="Where to save the trained model.")
    parser.add_argument("--time-limit", type=int, default=30000, help="Training time limit in seconds.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--metric-interval", type=int, default=30, help="Seconds between train_loss / val_loss prints.")
    parser.add_argument("--no-backup", action="store_true", help="Do not back up an existing model folder.")
    return parser.parse_args()


def log_step(message):
    current_time = time.strftime("%H:%M:%S")
    print(f"[{current_time}] {message}", flush=True)


def progress_iter(items, description):
    if tqdm is None:
        return items
    return tqdm(items, desc=description, unit="row")


def load_and_clean_data(csv_path):
    import pandas as pd

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到训练数据文件：{csv_path}")

    log_step(f"正在读取训练数据：{csv_path}")
    data = pd.read_csv(csv_path)
    original_rows = len(data)

    data = data.dropna(subset=[LABEL_COLUMN, IMAGE_COLUMN])
    data = data[data[IMAGE_COLUMN].astype(str).str.strip() != ""].copy()

    for column in NUMERIC_COLUMNS:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=[column for column in NUMERIC_COLUMNS if column in data.columns])

    missing_image_indices = []
    for idx in progress_iter(data.index, "检查图片路径"):
        image_path = Path(str(data.at[idx, IMAGE_COLUMN]))
        if not image_path.exists():
            missing_image_indices.append(idx)

    if missing_image_indices:
        data = data.drop(index=missing_image_indices)

    for column in DROP_COLUMNS:
        if column in data.columns:
            data = data.drop(columns=[column])

    if data.empty:
        raise RuntimeError("清洗后没有可训练数据，请检查 CSV、图片路径和 days_to_harvest 标签。")

    log_step(
        f"数据清洗完成：原始 {original_rows} 条，保留 {len(data)} 条，"
        f"剔除缺失图片 {len(missing_image_indices)} 条。"
    )
    log_step(f"标签范围：{data[LABEL_COLUMN].min():.0f} 到 {data[LABEL_COLUMN].max():.0f} 天")
    return data


def backup_existing_model(model_path):
    if not model_path.exists():
        return None

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = model_path.with_name(f"{model_path.name}_backup_{timestamp}")
    shutil.move(str(model_path), str(backup_path))
    return backup_path


def build_hyperparameters(args):
    return {
        "model.timm_image.checkpoint_name": "resnet18",
        "env.num_gpus": 0,
        "env.num_workers": args.num_workers,
        "env.num_workers_inference": args.num_workers,
        "env.batch_size": args.batch_size,
        "optim.max_epochs": args.max_epochs,
    }


def start_time_progress(time_limit, stop_event):
    if tqdm is None:
        return None

    def worker():
        with tqdm(total=time_limit, desc="训练进度", unit="s") as bar:
            elapsed = 0
            while not stop_event.is_set() and elapsed < time_limit:
                time.sleep(1)
                elapsed += 1
                bar.update(1)
            if stop_event.is_set() and elapsed < time_limit:
                bar.update(time_limit - elapsed)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


def find_latest_event_file(training_started_at):
    base_dir = Path("AutogluonModels")
    if not base_dir.exists():
        return None

    event_files = []
    for event_file in base_dir.glob("ag-*/events.out.tfevents*"):
        try:
            if event_file.stat().st_mtime >= training_started_at:
                event_files.append(event_file)
        except OSError:
            continue

    if not event_files:
        return None
    return max(event_files, key=lambda path: path.stat().st_mtime)


def read_latest_metrics(event_file):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])

    metrics = {}
    for tag in ("train_loss", "val_loss", "val_rmse", "epoch"):
        if tag not in tags:
            continue
        events = accumulator.Scalars(tag)
        if events:
            metrics[tag] = events[-1].value
    return metrics


def start_metric_monitor(interval, stop_event, training_started_at):
    if interval <= 0:
        return None

    def worker():
        last_printed = None
        warned = False

        while not stop_event.is_set():
            event_file = find_latest_event_file(training_started_at)
            if event_file is None:
                time.sleep(interval)
                continue

            try:
                metrics = read_latest_metrics(event_file)
            except Exception as exc:
                if not warned:
                    log_step(f"训练指标监控暂不可用：{exc}")
                    warned = True
                time.sleep(interval)
                continue

            if metrics:
                signature = tuple(sorted((key, round(value, 6)) for key, value in metrics.items()))
                if signature != last_printed:
                    pieces = []
                    for key in ("epoch", "train_loss", "val_loss", "val_rmse"):
                        if key in metrics:
                            pieces.append(f"{key}={metrics[key]:.4f}")
                    log_step("训练指标 | " + " | ".join(pieces))
                    last_printed = signature

            time.sleep(interval)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


def train(args, train_data):
    log_step("正在导入 AutoGluon 训练模块，首次加载可能需要十几秒。")
    from autogluon.multimodal import MultiModalPredictor

    log_step("AutoGluon 导入完成，正在初始化多模态回归模型。")
    log_step("正在初始化 AutoGluon 多模态回归模型。")
    predictor = MultiModalPredictor(
        label=LABEL_COLUMN,
        problem_type="regression",
        eval_metric="mae",
    )

    hyperparameters = build_hyperparameters(args)
    print("\n" + "=" * 72)
    log_step("开始训练。AutoGluon 会继续输出 epoch / batch 训练日志。")
    print("=" * 72 + "\n", flush=True)

    stop_event = threading.Event()
    training_started_at = datetime.now().timestamp()
    progress_thread = start_time_progress(args.time_limit, stop_event)
    metric_thread = start_metric_monitor(args.metric_interval, stop_event, training_started_at)
    try:
        predictor.fit(
            train_data=train_data,
            time_limit=args.time_limit,
            seed=args.seed,
            hyperparameters=hyperparameters,
        )
    finally:
        stop_event.set()
        if progress_thread is not None:
            progress_thread.join(timeout=3)
        if metric_thread is not None:
            metric_thread.join(timeout=3)

    return predictor


def main():
    args = parse_args()
    csv_path = Path(args.data)
    model_path = Path(args.model_path)

    log_step("系统启动，准备训练温室多模态成熟度预测模型。")
    train_data = load_and_clean_data(csv_path)
    predictor = train(args, train_data)

    backup_path = None
    if not args.no_backup:
        backup_path = backup_existing_model(model_path)
        if backup_path is not None:
            log_step(f"已备份旧模型到：{backup_path}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_step(f"正在保存新模型到：{model_path}")
    predictor.save(str(model_path))
    log_step("训练完成，可以使用 predict_demo.py 或自己的预测脚本加载模型。")


if __name__ == "__main__":
    main()
