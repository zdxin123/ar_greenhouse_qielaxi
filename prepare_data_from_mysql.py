import argparse
import csv
import re
import subprocess
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path


TEXT = {
    "device_name": "\u8bbe\u5907\u540d\u79f0",
    "light": "\u5149\u7167",
    "soil_ph": "\u571f\u58e4PH",
    "soil_temp": "\u571f\u58e4\u6e29\u5ea6",
    "soil_hum": "\u571f\u58e4\u6e7f\u5ea6",
    "n": "\u6c2e",
    "p": "\u78f7",
    "air_temp": "\u7a7a\u6c14\u6e29\u5ea6",
    "air_hum": "\u7a7a\u6c14\u6e7f\u5ea6",
    "k": "\u94be",
    "plant_date": "\u79cd\u690d\u65e5\u671f",
    "planted_days": "\u5df2\u79cd\u690d\u5929\u6570",
}


SHED_CONFIG = {
    "1": {
        "device": "<DEVICE_1>",
        "name": "\u4e00\u53f7\u68da\uff08\u6df7\u51dd\u571f\uff09\u6570\u636e",
    },
    "2": {
        "device": "<DEVICE_2>",
        "name": "\u4e8c\u53f7\u68da\uff08\u6df7\u51dd\u571f\uff09\u6570\u636e",
    },
    "3": {
        "device": "<DEVICE_3>",
        "name": "\u4e09\u53f7\u68da\uff08\u52a0\u780c\u5757\uff09\u6570\u636e",
    },
    "4": {
        "device": "<DEVICE_4>",
        "name": "\u56db\u53f7\u68da\uff08\u52a0\u780c\u5757\uff09\u6570\u636e",
    },
    "5": {
        "device": "<DEVICE_5>",
        "name": "\u4e94\u53f7\u68da\uff08\u5bf9\u79f0\uff09\u6570\u636e",
    },
    "6": {
        "device": "<DEVICE_6>",
        "name": "\u516d\u53f7\u68da\uff08\u5bf9\u79f0\uff09\u6570\u636e",
    },
    "7": {
        "device": "<DEVICE_7>",
        "name": "\u4e03\u53f7\u68da\u4e1c\uff08\u5927\u8de8\u5ea6\u571f\u592f\u5899\uff09",
    },
    "8": {
        "device": "<DEVICE_8>",
        "name": "\u4e03\u53f7\u68da\u897f\uff08\u5927\u8de8\u5ea6\u571f\u592f\u5899\uff09",
    },
}


TARGET_SLOTS = ["06:30", "10:30", "14:00", "17:30", "23:00"]

DEFAULT_PLANT_DATES = {
    "1": "<DATE>",
    "2": "<DATE>",
    "3": "<DATE>",
    "4": "<DATE>",
    "5": "<DATE>",
}

DEFAULT_HARVEST_DATES = {
    "1": "<DATE>",
    "2": "<DATE>",
    "3": "<DATE>",
    "4": "<DATE>",
    "5": "<DATE>",
}

FACTOR_BY_SUFFIX = {
    "_15_5": TEXT["light"],
    "_4_1": TEXT["soil_ph"],
    "_3_1": TEXT["soil_temp"],
    "_3_2": TEXT["soil_hum"],
    "_28_1": TEXT["n"],
    "_28_2": TEXT["p"],
    "_11_1": TEXT["air_temp"],
    "_11_2": TEXT["air_hum"],
    "_29_2": TEXT["k"],
}

FACTOR_COLUMNS = [
    TEXT["light"],
    TEXT["soil_ph"],
    TEXT["soil_temp"],
    TEXT["soil_hum"],
    TEXT["n"],
    TEXT["p"],
    TEXT["air_temp"],
    TEXT["air_hum"],
    TEXT["k"],
]

OUTPUT_COLUMNS = [
    TEXT["device_name"],
    "date",
    "time_slot",
    *FACTOR_COLUMNS,
    "image_path",
    TEXT["plant_date"],
    TEXT["planted_days"],
    "days_to_harvest",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate final_train_data.csv and final_train_data_ready.csv from MySQL sensor history and greenhouse images."
    )
    parser.add_argument("--host", default="", help="?????????????????")
    parser.add_argument("--port", default="", help="?????????????????")
    parser.add_argument("--user", default="", help="??????????????????")
    parser.add_argument("--password", default="", help="数据库密码，公开仓库中不写默认值。")
    parser.add_argument("--database", default="", help="?????????????????")
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--sheds", default="1,2,3,4,5", help="Comma-separated shed folders, for example: 1,2,3,4,5")
    parser.add_argument("--plant-date", default=None, help="Fallback plant date, YYYY-MM-DD")
    parser.add_argument("--harvest-date", default=None, help="Fallback harvest/maturity date, YYYY-MM-DD")
    parser.add_argument("--start-date", default=None, help="Data start date. Defaults to the earliest selected plant date.")
    parser.add_argument("--end-date", default=None, help="Data end date. Defaults to the latest selected harvest date.")
    parser.add_argument("--plant-dates", default="", help="Per-shed plant dates, for example: 1=<PLANT_DATE>,2=<PLANT_DATE>")
    parser.add_argument("--harvest-dates", default="", help="Per-shed harvest dates, for example: 1=<HARVEST_DATE>,2=<HARVEST_DATE>")
    parser.add_argument("--output", default="final_train_data.csv")
    parser.add_argument("--ready-output", default="final_train_data_ready.csv")
    parser.add_argument("--mysql-bin", default="mysql")
    parser.add_argument("--keep-raw-labels", action="store_true", help="Also fill days_to_harvest in the intermediate output.")
    return parser.parse_args()


def parse_date(value):
    return datetime.strptime(value, "%Y-%m-%d").date()


def parse_date_map(value):
    result = {}
    if not value.strip():
        return result
    for item in value.split(","):
        if not item.strip():
            continue
        shed, day = item.split("=", 1)
        result[shed.strip()] = parse_date(day.strip())
    return result


def closest_slot(dt):
    best_slot = None
    min_diff = timedelta(days=1)
    for slot in TARGET_SLOTS:
        target = datetime.strptime(f"{dt:%Y-%m-%d} {slot}", "%Y-%m-%d %H:%M")
        diff = abs(dt - target)
        if diff < min_diff:
            min_diff = diff
            best_slot = slot
    if min_diff <= timedelta(minutes=90):
        return best_slot
    return None


def factor_column(factor_id):
    for suffix, column in FACTOR_BY_SUFFIX.items():
        if factor_id.endswith(suffix):
            return column
    return None


def run_mysql_query(args, sheds, start_date, end_date):
    devices = ",".join(f"'{SHED_CONFIG[shed]['device']}'" for shed in sheds)
    query = f"""
SELECT h.deviceAddr, h.recordTime, h.factorId, h.data
FROM tbhistory h
WHERE h.deviceAddr IN ({devices})
  AND h.recordTime >= '{start_date.isoformat()} 00:00:00'
  AND h.recordTime <= '{end_date.isoformat()} 23:59:59'
ORDER BY h.deviceAddr, h.recordTime;
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


def build_sensor_rows(lines, device_to_shed):
    buckets = defaultdict(list)

    for line in lines:
        parts = line.rstrip("\r").split("\t")
        if len(parts) != 4:
            continue

        device_addr, record_time, factor_id, value = parts
        shed = device_to_shed.get(device_addr)
        factor = factor_column(factor_id)
        if not shed or not factor:
            continue

        try:
            dt = datetime.strptime(record_time, "%Y-%m-%d %H:%M:%S")
            numeric = float(value)
        except ValueError:
            continue

        slot = closest_slot(dt)
        if not slot:
            continue

        key = (shed, dt.date().isoformat(), slot, factor)
        buckets[key].append(numeric)

    sensor = defaultdict(dict)
    for (shed, day, slot, factor), values in buckets.items():
        sensor[(shed, day, slot)][factor] = round(sum(values) / len(values), 4)

    return sensor


def iter_image_records(image_dir, sheds, start_date, end_date):
    shed_pattern = "|".join(re.escape(shed) for shed in sheds)
    pattern = re.compile(rf"^({shed_pattern})-\d+_(\d{{8}})_(\d{{6}})\.jpg$", re.IGNORECASE)

    for shed in sheds:
        folder = image_dir / shed
        if not folder.exists():
            continue

        for path in sorted(folder.glob("*.jpg")):
            match = pattern.match(path.name)
            if not match:
                continue

            shed_num, date_text, time_text = match.groups()
            try:
                dt = datetime.strptime(f"{date_text} {time_text}", "%Y%m%d %H%M%S")
            except ValueError:
                continue

            if not (start_date <= dt.date() <= end_date):
                continue

            slot = closest_slot(dt)
            if not slot:
                continue

            yield shed_num, dt.date().isoformat(), slot, path


def make_rows(sensor, image_dir, sheds, start_date, end_date, plant_dates, harvest_dates):
    rows = []
    seen_image_paths = set()

    for shed, day, slot, image_path in iter_image_records(image_dir, sheds, start_date, end_date):
        values = sensor.get((shed, day, slot))
        if not values or any(column not in values for column in FACTOR_COLUMNS):
            continue

        image_path_text = str(image_path)
        if image_path_text in seen_image_paths:
            continue
        seen_image_paths.add(image_path_text)

        current_date = parse_date(day)
        plant_date = plant_dates[shed]
        harvest_date = harvest_dates[shed]
        if not (plant_date <= current_date <= harvest_date):
            continue

        row = {
            TEXT["device_name"]: SHED_CONFIG[shed]["name"],
            "date": day,
            "time_slot": slot,
            "image_path": image_path_text,
            TEXT["plant_date"]: plant_date.isoformat(),
            TEXT["planted_days"]: (current_date - plant_date).days,
            "days_to_harvest": (harvest_date - current_date).days,
        }
        for column in FACTOR_COLUMNS:
            row[column] = values[column]

        rows.append(row)

    rows.sort(key=lambda row: (row[TEXT["device_name"]], row["date"], row["time_slot"], row["image_path"]))
    return rows


def write_csv(path, rows, fill_label):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for row in rows:
            output_row = dict(row)
            if not fill_label:
                output_row["days_to_harvest"] = ""
            writer.writerow(output_row)


def print_summary(rows, sensor):
    print(f"Sensor slots: {len(sensor)}")
    print(f"Training rows: {len(rows)}")

    by_shed = defaultdict(int)
    for row in rows:
        by_shed[row[TEXT["device_name"]]] += 1

    for shed_name, count in sorted(by_shed.items()):
        print(f"{shed_name}: {count}")

    if rows:
        min_date = min(row["date"] for row in rows)
        max_date = max(row["date"] for row in rows)
        min_days = min(row["days_to_harvest"] for row in rows)
        max_days = max(row["days_to_harvest"] for row in rows)
        print(f"Date range: {min_date} to {max_date}")
        print(f"days_to_harvest range: {min_days} to {max_days}")


def resolve_crop_dates(sheds, fallback_date, default_dates, override_text):
    dates = {}
    for shed in sheds:
        if shed in default_dates:
            dates[shed] = parse_date(default_dates[shed])
        elif fallback_date:
            dates[shed] = fallback_date
        else:
            raise ValueError(f"No default date configured for shed {shed}; pass --plant-dates or --harvest-dates.")
    dates.update(parse_date_map(override_text))
    return dates


def main():
    args = parse_args()
    sheds = [shed.strip() for shed in args.sheds.split(",") if shed.strip()]
    unknown_sheds = [shed for shed in sheds if shed not in SHED_CONFIG]
    if unknown_sheds:
        raise ValueError(f"Unknown shed numbers: {', '.join(unknown_sheds)}")

    fallback_plant_date = parse_date(args.plant_date) if args.plant_date else None
    fallback_harvest_date = parse_date(args.harvest_date) if args.harvest_date else None
    plant_dates = resolve_crop_dates(sheds, fallback_plant_date, DEFAULT_PLANT_DATES, args.plant_dates)
    harvest_dates = resolve_crop_dates(sheds, fallback_harvest_date, DEFAULT_HARVEST_DATES, args.harvest_dates)
    start_date = parse_date(args.start_date) if args.start_date else min(plant_dates[shed] for shed in sheds)
    end_date = parse_date(args.end_date) if args.end_date else max(harvest_dates[shed] for shed in sheds)

    image_dir = Path(args.image_dir)
    device_to_shed = {SHED_CONFIG[shed]["device"]: shed for shed in sheds}

    lines = run_mysql_query(args, sheds, start_date, end_date)
    sensor = build_sensor_rows(lines, device_to_shed)
    rows = make_rows(sensor, image_dir, sheds, start_date, end_date, plant_dates, harvest_dates)

    write_csv(Path(args.output), rows, fill_label=args.keep_raw_labels)
    write_csv(Path(args.ready_output), rows, fill_label=True)
    print_summary(rows, sensor)


if __name__ == "__main__":
    main()

