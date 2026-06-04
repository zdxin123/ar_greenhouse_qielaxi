import argparse
import contextlib
import gc
import io
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.multimodal import MultiModalPredictor


FACTOR_COLUMNS = ["光照", "土壤PH", "土壤温度", "土壤湿度", "氮", "磷", "空气温度", "空气湿度", "钾"]
DEFAULT_MODEL_PATH = "ag_models/days_predictor"

SHED_CONFIG = {
    "1": {"device": "<DEVICE_1>", "name": "一号棚（混凝土）数据", "plant_date": "<PLANT_DATE>"},
    "2": {"device": "<DEVICE_2>", "name": "二号棚（混凝土）数据", "plant_date": "<PLANT_DATE>"},
    "3": {"device": "<DEVICE_3>", "name": "三号棚（加砌块）数据", "plant_date": "<PLANT_DATE>"},
    "4": {"device": "<DEVICE_4>", "name": "四号棚（加砌块）数据", "plant_date": "<PLANT_DATE>"},
    "5": {"device": "<DEVICE_5>", "name": "五号棚（对称）数据", "plant_date": "<PLANT_DATE>"},
    "6": {"device": "<DEVICE_6>", "name": "六号棚（对称）数据", "plant_date": "<PLANT_DATE>"},
    "7": {"device": "<DEVICE_7>", "name": "七号棚东（大跨度土夯墙）", "plant_date": "<PLANT_DATE>"},
    "8": {"device": "<DEVICE_8>", "name": "七号棚西（大跨度土夯墙）", "plant_date": "<PLANT_DATE>"},
}

CROP_CONFIG = {
    "tomato": {
        "names": {"tomato", "西红柿", "番茄"},
        "display_name": "西红柿",
        "model_path": "ag_models/days_predictor",
        "enabled": True,
    },
    "pepper": {
        "names": {"pepper", "辣椒"},
        "display_name": "辣椒",
        "model_path": "ag_models/pepper_days_predictor",
        "enabled": False,
    },
    "eggplant": {
        "names": {"eggplant", "茄子"},
        "display_name": "茄子",
        "model_path": "ag_models/eggplant_days_predictor",
        "enabled": False,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Predict days to harvest from latest MySQL sensor data and matched image.")
    parser.add_argument("--shed", default="1", choices=sorted(SHED_CONFIG), help="Shed number/folder to predict.")
    parser.add_argument("--crop", default="西红柿", help="Crop name, for example: 西红柿 / 番茄 / 辣椒 / 茄子.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--image-dir", default="predict_need_data")
    parser.add_argument("--host", default="", help="?????????????????")
    parser.add_argument("--port", default="", help="?????????????????")
    parser.add_argument("--user", default="", help="??????????????????")
    parser.add_argument("--password", default="", help="数据库密码，公开仓库中不写默认值。")
    parser.add_argument("--database", default="", help="?????????????????")
    parser.add_argument("--mysql-bin", default="mysql")
    parser.add_argument("--plant-date", default=None, help="Override plant date, YYYY-MM-DD.")
    parser.add_argument("--search-days", type=int, default=1, help="Days around the selected image time to search sensor records.")
    parser.add_argument("--max-history", type=int, default=4000, help="Rows to scan when looking for a complete sensor record.")
    parser.add_argument("--num-workers-inference", type=int, default=0, help="Inference workers. Use 0 on Windows if multiprocessing is blocked.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON for dashboard integration.")
    parser.add_argument("--json-internal", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--allow-tomato-fallback", action="store_true", help="Use the tomato model for unsupported crops. Not recommended for production.")
    return parser.parse_args()


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_header():
    print("================================================================")
    print("       [ 农业大棚多模态成熟度预测系统 - 实时 Demo ]       ")
    print("================================================================")


def simulate_loading(message, delay=0.8):
    print(f"\n[System] {message}", end="", flush=True)
    for _ in range(3):
        time.sleep(delay / 3.0)
        print(".", end="", flush=True)
    print(" [完成]")


def resolve_crop(crop_name, allow_tomato_fallback=False):
    normalized = crop_name.strip().lower()
    for crop_key, config in CROP_CONFIG.items():
        if normalized in {name.lower() for name in config["names"]}:
            if config["enabled"]:
                return crop_key, config
            if allow_tomato_fallback:
                fallback = CROP_CONFIG["tomato"].copy()
                fallback["display_name"] = f"{config['display_name']}（暂用西红柿模型）"
                return "tomato", fallback
            raise ValueError(f"当前还没有训练 {config['display_name']} 模型，请先训练对应作物模型。")
    supported = "、".join(sorted({name for config in CROP_CONFIG.values() for name in config["names"]}))
    raise ValueError(f"未知作物：{crop_name}。可识别作物名：{supported}")


def run_mysql_query(args, device_addr, target_time):
    start_time = target_time - pd.Timedelta(days=args.search_days)
    end_time = target_time + pd.Timedelta(days=args.search_days)
    query = f"""
SELECT d.`时间`, d.`传感器名称`, d.`数据值`
FROM `data` d
WHERE d.`传感器地址ID` LIKE '{device_addr}%'
  AND d.`时间` >= '{start_time:%Y-%m-%d %H:%M:%S}'
  AND d.`时间` <= '{end_time:%Y-%m-%d %H:%M:%S}'
  AND d.`传感器名称` IN ('光照','土壤PH','土壤温度','土壤湿度','氮','磷','空气温度','空气湿度','钾')
ORDER BY ABS(TIMESTAMPDIFF(SECOND, d.`时间`, '{target_time:%Y-%m-%d %H:%M:%S}')), d.`时间` DESC
LIMIT {args.max_history};
"""
    cmd = [
        args.mysql_bin,
        "--default-character-set=utf8mb4",
        "-h",
        args.host,
        "-P",
        str(args.port),
        f"-u{args.user}",
        f"-p{args.password}",
        "-D",
        args.database,
        "--batch",
        "--raw",
        "--skip-column-names",
        "-e",
        query,
    ]
    completed = subprocess.run(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=True,
    )
    return completed.stdout.splitlines()


def run_simple_mysql_query(args, query):
    cmd = [
        args.mysql_bin,
        "--default-character-set=utf8mb4",
        "-h",
        args.host,
        "-P",
        str(args.port),
        f"-u{args.user}",
        f"-p{args.password}",
        "-D",
        args.database,
        "--batch",
        "--raw",
        "--skip-column-names",
        "-e",
        query,
    ]
    completed = subprocess.run(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=True,
    )
    return completed.stdout.splitlines()


def nearest_complete_sensor_row(args, device_addr, target_time):
    grouped = {}

    for line in run_mysql_query(args, device_addr, target_time):
        parts = line.rstrip("\r").split("\t")
        if len(parts) != 3:
            continue

        record_time, sensor_name, value = parts
        if sensor_name not in FACTOR_COLUMNS:
            continue

        try:
            numeric = float(value)
            dt = datetime.strptime(record_time, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

        grouped.setdefault(dt, {})[sensor_name] = numeric

    for dt in sorted(grouped, key=lambda item: abs(item - target_time)):
        values = grouped[dt]
        if all(column in values for column in FACTOR_COLUMNS):
            return dt, values

    raise RuntimeError(f"数据库中没有找到 {target_time} 附近完整的 9 项传感器记录。")


def image_pattern_for_shed(shed):
    if shed in {"1", "2", "3", "4", "5", "6"}:
        return re.compile(rf"^{shed}-\d+_(\d{{8}})_(\d{{6}})\.jpg$", re.IGNORECASE)
    if shed == "7":
        return re.compile(r"^7东-\d+_(\d{8})_(\d{6})\.jpg$", re.IGNORECASE)
    return re.compile(r"^7西-\d+_(\d{8})_(\d{6})\.jpg$", re.IGNORECASE)


def camera_name_matches_shed(camera_name, shed):
    if shed in {"1", "2", "3", "4", "5", "6"}:
        return camera_name.startswith(f"{shed}-")
    if shed == "7":
        return camera_name.startswith("7东-")
    return camera_name.startswith("7西-")


def load_camera_prefixes(args, shed):
    lines = run_simple_mysql_query(args, "SELECT cameraId, cameraName FROM tbcamera;")
    prefixes = []

    for line in lines:
        parts = line.rstrip("\r").split("\t")
        if len(parts) != 2:
            continue
        camera_id, camera_name = parts
        if not camera_name_matches_shed(camera_name, shed):
            continue
        prefixes.append((camera_name, camera_name))
        prefixes.append((camera_id, camera_name))

    if not prefixes:
        raise RuntimeError(f"tbcamera 中没有找到 {shed} 号棚对应的摄像头。")
    return prefixes


def parse_image_time_by_prefix(image_name, prefixes):
    suffix_pattern = re.compile(r"_(\d{8})_(\d{6})\.(jpg|jpeg|png)$", re.IGNORECASE)
    suffix_match = suffix_pattern.search(image_name)
    timestamp_match = None
    timestamp_dt = None
    if not suffix_match:
        timestamp_match = re.search(r"_(.*?)\.(jpg|jpeg|png)$", image_name, re.IGNORECASE)
        if not timestamp_match:
            return None
        for raw_timestamp in re.findall(r"\d{13}", timestamp_match.group(1)):
            candidate_dt = datetime.fromtimestamp(int(raw_timestamp) / 1000)
            if 2020 <= candidate_dt.year <= 2035:
                timestamp_dt = candidate_dt
                break
        if timestamp_dt is None:
            return None

    match_start = suffix_match.start() if suffix_match else timestamp_match.start()
    prefix_text = image_name[:match_start]
    for prefix, camera_name in prefixes:
        if prefix_text == prefix:
            if suffix_match:
                dt = datetime.strptime(
                    f"{suffix_match.group(1)} {suffix_match.group(2)}",
                    "%Y%m%d %H%M%S",
                )
            else:
                dt = timestamp_dt
            return dt, camera_name, prefix
    return None


def find_latest_image(args, image_dir, shed):
    folder = Path(image_dir)
    if not folder.exists():
        raise FileNotFoundError(f"找不到预测图片文件夹：{folder}")

    readable_pattern = image_pattern_for_shed(shed)
    camera_prefixes = load_camera_prefixes(args, shed)
    candidates = []
    image_files = []
    for suffix in ("*.jpg", "*.jpeg", "*.png"):
        image_files.extend(folder.rglob(suffix))

    for image_path in image_files:
        match = readable_pattern.match(image_path.name)
        if match:
            dt = datetime.strptime(f"{match.group(1)} {match.group(2)}", "%Y%m%d %H%M%S")
            candidates.append((dt, image_path, image_path.name.split("_", 1)[0]))
            continue

        parsed = parse_image_time_by_prefix(image_path.name, camera_prefixes)
        if parsed:
            dt, camera_name, _prefix = parsed
            candidates.append((dt, image_path, camera_name))

    if not candidates:
        raise FileNotFoundError(f"{folder} 中没有找到 {shed} 号棚对应的可用原始图片。")

    return max(candidates, key=lambda item: item[0])


def build_predict_row(sensor_time, sensor_values, image_path, plant_date):
    planted_days = (sensor_time.date() - plant_date.date()).days
    row = dict(sensor_values)
    row["已种植天数"] = planted_days
    row["image_path"] = str(image_path)
    return pd.DataFrame([row]), planted_days


def scalar_prediction(prediction):
    return float(np.asarray(prediction).reshape(-1)[0])


def predict_maturity(args):
    crop_key, crop_config = resolve_crop(args.crop, args.allow_tomato_fallback)
    shed_info = SHED_CONFIG[args.shed]
    model_path = args.model_path
    if args.model_path == DEFAULT_MODEL_PATH:
        model_path = crop_config["model_path"]

    plant_date_text = args.plant_date or shed_info["plant_date"]
    plant_date = datetime.strptime(plant_date_text, "%Y-%m-%d")

    image_time, image_path, camera_name = find_latest_image(args, args.image_dir, args.shed)
    sensor_time, sensor_values = nearest_complete_sensor_row(args, shed_info["device"], image_time)
    today_data, planted_days = build_predict_row(sensor_time, sensor_values, image_path, plant_date)

    predictor_days = MultiModalPredictor.load(model_path)
    predictor_days._learner._config.env.num_workers = args.num_workers_inference
    predictor_days._learner._config.env.num_workers_inference = args.num_workers_inference

    predicted_days = scalar_prediction(predictor_days.predict(today_data, as_pandas=False))
    predicted_maturity_date = sensor_time.date() + pd.Timedelta(days=round(predicted_days))

    return {
        "crop": crop_config["display_name"],
        "crop_key": crop_key,
        "shed": args.shed,
        "shed_name": shed_info["name"],
        "device_addr": shed_info["device"],
        "plant_date": plant_date.date().isoformat(),
        "planted_days": planted_days,
        "sensor_time": sensor_time.strftime("%Y-%m-%d %H:%M:%S"),
        "image_time": image_time.strftime("%Y-%m-%d %H:%M:%S"),
        "image_path": str(image_path),
        "camera_name": camera_name,
        "sensor_image_diff_minutes": round(abs((sensor_time - image_time).total_seconds()) / 60, 2),
        "sensor_values": sensor_values,
        "predicted_days_remaining": round(predicted_days, 2),
        "predicted_maturity_date": predicted_maturity_date.isoformat(),
        "model_path": model_path,
    }


def run_json_child(args):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--shed",
        args.shed,
        "--crop",
        args.crop,
        "--model-path",
        args.model_path,
        "--image-dir",
        args.image_dir,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--user",
        args.user,
        "--password",
        args.password,
        "--database",
        args.database,
        "--mysql-bin",
        args.mysql_bin,
        "--search-days",
        str(args.search_days),
        "--max-history",
        str(args.max_history),
        "--num-workers-inference",
        str(args.num_workers_inference),
        "--json-internal",
    ]
    if args.plant_date:
        command.extend(["--plant-date", args.plant_date])
    if args.allow_tomato_fallback:
        command.append("--allow-tomato-fallback")

    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parent,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    output = completed.stdout + "\n" + completed.stderr
    begin = "JSON_RESULT_BEGIN"
    end = "JSON_RESULT_END"
    if begin in output and end in output:
        payload = output.split(begin, 1)[1].split(end, 1)[0].strip()
        try:
            parsed_payload = json.loads(payload)
            print(json.dumps(parsed_payload, ensure_ascii=True, indent=2))
        except json.JSONDecodeError:
            print(payload.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8"))
        return

    print(
        json.dumps(
            {
                "ok": False,
                "error": "内部预测进程没有返回 JSON 结果。",
                "returncode": completed.returncode,
                "tail": output[-1200:],
            },
            ensure_ascii=True,
            indent=2,
        )
    )


def main():
    args = parse_args()
    shed_info = SHED_CONFIG[args.shed]

    if args.json and not args.json_internal:
        run_json_child(args)
        return

    if args.json_internal:
        try:
            result = predict_maturity(args)
            payload = json.dumps({"ok": True, "data": result}, ensure_ascii=True, indent=2)
            print("JSON_RESULT_BEGIN")
            print(payload)
            print("JSON_RESULT_END")
        except Exception as exc:
            print("JSON_RESULT_BEGIN")
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=True, indent=2))
            print("JSON_RESULT_END")
        return

    crop_key, crop_config = resolve_crop(args.crop, args.allow_tomato_fallback)
    plant_date = datetime.strptime(args.plant_date or shed_info["plant_date"], "%Y-%m-%d")
    if args.model_path == DEFAULT_MODEL_PATH:
        args.model_path = crop_config["model_path"]

    clear_screen()
    print_header()

    simulate_loading(f"正在确认作物模型：{crop_config['display_name']}")
    simulate_loading("正在读取最新视觉图像")
    image_time, image_path, camera_name = find_latest_image(args, args.image_dir, args.shed)

    simulate_loading(f"正在匹配 {shed_info['name']} 对应时间的完整传感器数据")
    sensor_time, sensor_values = nearest_complete_sensor_row(args, shed_info["device"], image_time)
    if abs((sensor_time - image_time).total_seconds()) > 90 * 60:
        print(
            f"\n[提示] 图片时间为 {image_time}，匹配传感器时间为 {sensor_time}，"
            "两者相差超过 90 分钟；请确认图片和传感器采集是否同步。"
        )

    today_data, planted_days = build_predict_row(sensor_time, sensor_values, image_path, plant_date)

    simulate_loading("正在初始化 AutoGluon 多模态预测引擎与权重")
    predictor_days = MultiModalPredictor.load(args.model_path)
    predictor_days._learner._config.env.num_workers = args.num_workers_inference
    predictor_days._learner._config.env.num_workers_inference = args.num_workers_inference

    simulate_loading("正在执行视觉通道、时序通道与静态属性联合推理")
    prediction = predictor_days.predict(today_data, as_pandas=False)
    predicted_days = scalar_prediction(prediction)
    predicted_maturity_date = sensor_time.date() + pd.Timedelta(days=round(predicted_days))

    clear_screen()
    print_header()
    print("\n【 输入特征流分析 (Input Stream) 】")
    print(f" > 数据对象 : {shed_info['name']} | 作物 {crop_config['display_name']} | 设备地址 {shed_info['device']}")
    print(f" > 时序通道 : [数据库接入] 最新完整记录 -> {sensor_time}")
    print(f" > 视觉通道 : [图片接入] {camera_name} -> {image_path} ({image_time})")
    print(" > 微环境感知矩阵 ->")
    print(
        f"      ├─ 气候参数 : 空气温度 {today_data['空气温度'].iloc[0]}℃ | "
        f"空气湿度 {today_data['空气湿度'].iloc[0]}% | 光照 {today_data['光照'].iloc[0]} Lx"
    )
    print(
        f"      ├─ 土壤理化 : 土壤温度 {today_data['土壤温度'].iloc[0]}℃ | "
        f"土壤湿度 {today_data['土壤湿度'].iloc[0]}% | pH值 {today_data['土壤PH'].iloc[0]}"
    )
    print(f"      └─ 核心营养 : 氮(N) {today_data['氮'].iloc[0]} | 磷(P) {today_data['磷'].iloc[0]} | 钾(K) {today_data['钾'].iloc[0]}")
    print(f" > 静态属性 : [已接入] 已种植周期 -> {planted_days} 天 (种植日期 {plant_date.date()})")

    print("\n" + "-" * 64)
    print("\n【 多模态联合预测结果 (Output) 】")
    print(f"\n      >>> 预计距离最佳成熟采摘期还剩: [ {predicted_days:.1f} 天 ] <<<\n")
    print(f"      >>> 预计成熟日期: [ {predicted_maturity_date} ] <<<\n")
    print("-" * 64)
    print("[提示] 预测结果基于当前传感器记录、匹配图像和已种植天数综合得出。")
    print("================================================================\n")


if __name__ == "__main__":
    main()

