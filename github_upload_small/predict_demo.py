import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


SHED_CHOICES = [str(i) for i in range(1, 9)]


def parse_args():
    parser = argparse.ArgumentParser(description="农业大棚多模态成熟度预测系统 - 终端可视化 Demo")
    parser.add_argument("--shed", default="1", choices=SHED_CHOICES, help="大棚编号，1-8。")
    parser.add_argument("--crop", default="西红柿", help="作物名，例如：西红柿 / 番茄 / 辣椒 / 茄子。")
    parser.add_argument("--plant-date", default=None, help="种植日期，格式 YYYY-MM-DD。")
    parser.add_argument("--model-path", default="ag_models/days_predictor", help="模型目录。")
    parser.add_argument("--image-dir", default="predict_need_data", help="待预测图片目录。")
    parser.add_argument("--host", default="", help="?????????????????")
    parser.add_argument("--port", default="", help="?????????????????")
    parser.add_argument("--user", default="", help="??????????????????")
    parser.add_argument("--password", default="", help="数据库密码，公开仓库中不写默认值。")
    parser.add_argument("--database", default="", help="?????????????????")
    parser.add_argument("--mysql-bin", default="mysql")
    parser.add_argument("--search-days", type=int, default=1, help="围绕图片时间查找传感器数据的天数。")
    parser.add_argument("--max-history", type=int, default=4000, help="查找完整传感器记录时最多扫描的行数。")
    parser.add_argument("--num-workers-inference", type=int, default=0, help="Windows 下建议保持 0。")
    parser.add_argument("--allow-tomato-fallback", action="store_true", help="暂用西红柿模型预测未训练作物。")
    return parser.parse_args()


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_header():
    print("=" * 68)
    print("          农业大棚多模态成熟度预测系统 - 实时 Demo")
    print("=" * 68)


def show_step(message, delay=0.35):
    print(f"\n[System] {message}", end="", flush=True)
    for _ in range(3):
        time.sleep(delay / 3)
        print(".", end="", flush=True)
    print(" 完成")


def build_backend_command(args):
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "predict.py"),
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
        "--json",
    ]
    if args.plant_date:
        command.extend(["--plant-date", args.plant_date])
    if args.allow_tomato_fallback:
        command.append("--allow-tomato-fallback")
    return command


def run_backend(args):
    completed = subprocess.run(
        build_backend_command(args),
        cwd=Path(__file__).resolve().parent,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    output = completed.stdout.strip()
    if not output:
        raise RuntimeError(f"后台预测没有返回结果。退出码：{completed.returncode}")
    payload = json.loads(output)
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error", "后台预测失败。"))
    return payload["data"]


def print_sensor_values(sensor_values):
    print(" > 微环境感知矩阵")
    print(
        f"      气候参数 : 空气温度 {sensor_values['空气温度']} ℃ | "
        f"空气湿度 {sensor_values['空气湿度']} % | 光照 {sensor_values['光照']} Lx"
    )
    print(
        f"      土壤理化 : 土壤温度 {sensor_values['土壤温度']} ℃ | "
        f"土壤湿度 {sensor_values['土壤湿度']} % | pH {sensor_values['土壤PH']}"
    )
    print(
        f"      核心营养 : 氮 {sensor_values['氮']} | "
        f"磷 {sensor_values['磷']} | 钾 {sensor_values['钾']}"
    )


def render_result(data):
    sensor_time = datetime.strptime(data["sensor_time"], "%Y-%m-%d %H:%M:%S")
    image_time = datetime.strptime(data["image_time"], "%Y-%m-%d %H:%M:%S")

    clear_screen()
    print_header()
    print("\n【 输入特征流 】")
    print(f" > 数据对象 : {data['shed_name']}")
    print(f" > 作物类型 : {data['crop']} ({data['crop_key']})")
    print(f" > 设备地址 : {data['device_addr']}")
    print(f" > 时序通道 : 数据库最新匹配记录 -> {sensor_time:%Y-%m-%d %H:%M:%S}")
    print(f" > 视觉通道 : {data['camera_name']} -> {data['image_path']}")
    print(f" > 图片时间 : {image_time:%Y-%m-%d %H:%M:%S}")
    print(f" > 图数时间差 : {data['sensor_image_diff_minutes']:.1f} 分钟")
    print_sensor_values(data["sensor_values"])
    print(f" > 静态属性 : 已种植 {data['planted_days']} 天，种植日期 {data['plant_date']}")

    print("\n" + "-" * 68)
    print("\n【 多模态联合预测结果 】")
    print(f"\n      >>> 预计距离成熟采摘还剩: {data['predicted_days_remaining']:.1f} 天 <<<")
    print(f"      >>> 预计成熟日期: {data['predicted_maturity_date']} <<<\n")
    print("-" * 68)
    print("提示：结果由当前图片、数据库传感器记录和已种植天数综合预测得出。")
    print("=" * 68 + "\n")


def main():
    args = parse_args()
    clear_screen()
    print_header()

    try:
        show_step(f"确认作物模型：{args.crop}")
        show_step("读取对应大棚的最新视觉图片")
        show_step("匹配数据库中的完整传感器数据")
        show_step("加载多模态模型并执行预测")
        result = run_backend(args)
        render_result(result)
    except Exception as exc:
        print("\n[错误] 预测没有完成。")
        print(f"原因：{exc}")
        print("\n建议先确认：图片目录是否有对应大棚图片、MySQL 是否能连接、模型目录是否存在。")


if __name__ == "__main__":
    main()

