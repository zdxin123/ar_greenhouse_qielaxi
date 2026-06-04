温室作物成熟时间预测项目使用说明
====================================

这份说明按“第一次接手项目的人也能照着跑通”的方式写。建议从上到下执行，不要跳步骤。


一、这个项目是做什么的
--------------------

本项目用大棚图片 + 传感器数据，预测作物距离成熟采摘还剩多少天。

目前已经能稳定使用的是“西红柿”模型。后续如果要预测“辣椒”“茄子”，需要先用对应作物的数据重新整理训练集并训练对应模型。

项目主流程分三步：

1. 从 MySQL 和图片文件夹整理训练数据。
2. 用整理好的 CSV 训练模型。
3. 用最新图片和数据库传感器数据进行预测。


二、先确认电脑环境
----------------

进入项目目录：

```bat
D:
cd D:\ar_greenhouse
```

激活 Python 环境。注意：这台电脑不能直接用 `conda activate`，要用下面这个命令：

```bat
C:\Users\win\miniconda3\Scripts\activate.bat ag_greenhouse
```

激活成功后，命令行前面会出现：

```text
(ag_greenhouse)
```

如果没有出现，先不要继续跑训练或预测。


三、项目里几个重要文件夹
----------------------

```text
ar_greenhouse/
├─ images/                 训练图片，已经按 1-8 号棚分类
├─ predict_need_data/      预测时使用的新图片，不需要按棚分类
├─ ag_models/              训练好的模型保存位置
├─ old/                    旧版或暂时不用的脚本
├─ final_train_data.csv    整理后的中间训练数据
├─ final_train_data_ready.csv  最终训练数据
├─ prepare_data_from_mysql.py  从 MySQL + 图片整理训练数据
├─ train_script.py         训练模型
├─ predict.py              给看板/API 调用，输出 JSON
└─ predict_demo.py         给人看的终端演示版
```


四、数据库要求
------------

默认连接信息如下：

```text
数据库类型：xxxx
地址：xxxx
端口：xxxx
用户名：xxxx
密码：xxxx
数据库名：xxxx
```

需要用到的表：

```text
data       当前传感器数据表，预测时使用
history  历史传感器数据表，整理训练集时使用
camera   摄像头编号和大棚名称对应表，预测筛选图片时使用
```

如果 MySQL 命令不可用，先确认电脑里是否能在命令行执行：

```bat
mysql --version
```

如果提示找不到 `mysql`，需要把 MySQL 的 bin 目录加入系统 PATH，或者运行脚本时用 `--mysql-bin` 指定 mysql.exe 的完整路径。


五、图片命名规则
--------------

训练图片放在 `images` 文件夹内，按棚分好：

```text
images/1/
images/2/
images/3/
images/4/
images/5/
images/6/
```


预测图片放在 `predict_need_data` 文件夹内，不需要手动分类。脚本会自动根据图片名筛选对应大棚的最新图片。

预测图片支持两类命名：

```text
1-1_20250726_140004.jpg
5-2_20250726_140004.jpg
```

或者使用数据库 `tbcamera` 表里的 `cameraId` 命名，例如：

```text
03ft817417723926531QbFI_xxx1753410625559xxx.jpg
```


六、重新整理训练数据
------------------

普通情况直接运行：

```bat
python prepare_data_from_mysql.py
```

默认会整理前五个棚：

```text
1, 2, 3, 4, 5
```

默认日期如下：

```text
1号棚：2025-06-10 种植，2025-10-10 成熟
2号棚：2025-06-10 种植，2025-10-10 成熟
3号棚：2025-05-01 种植，2025-08-10 成熟
4号棚：2025-05-01 种植，2025-08-14 成熟
5号棚：2025-03-12 种植，2025-08-01 成熟
```

运行成功后会生成：

```text
final_train_data.csv
final_train_data_ready.csv
```

训练时用的是：

```text
final_train_data_ready.csv
```

如果后续要改某个棚的种植日期和成熟日期，可以这样写：

```bat
python prepare_data_from_mysql.py --plant-dates 1=2025-06-10,2=2025-06-10,3=2025-05-01,4=2025-05-01,5=2025-03-12 --harvest-dates 1=2025-10-10,2=2025-10-10,3=2025-08-10,4=2025-08-14,5=2025-08-01
```

如果要整理更多棚，比如 1-8 号棚：

```bat
python prepare_data_from_mysql.py --sheds 1,2,3,4,5,6,7,8
```

注意：新增棚必须先确认种植日期和成熟日期，否则训练标签会不准。


七、训练模型
----------

确认 `final_train_data_ready.csv` 已经生成后，运行：

```bat
python train_script.py
```

默认训练设置：

```text
最多训练轮数：100 epoch
最长训练时间：30000 秒
模型保存目录：ag_models/days_predictor
训练数据：final_train_data_ready.csv
```

如果想明确指定训练 100 轮：

```bat
python train_script.py --max-epochs 100
```

如果想改最长训练时间，比如最多训练 2 小时：

```bat
python train_script.py --time-limit 7200
```

训练停止条件是谁先到就按谁停：

```text
如果先跑满 --max-epochs，就按 epoch 停。
如果先达到 --time-limit，就按时间停。
```

训练过程中终端会输出进度和 `train_loss`、`val_loss` 等信息。只要这些信息在变化，说明模型确实在训练。

如果终端看起来“卡住不动”，先检查是不是鼠标点选了 cmd 窗口里的文字。Windows 命令行进入选择状态后会暂停程序，按一下 `Esc` 或取消选中即可继续。


八、预测成熟时间
--------------

预测前确认：

```text
1. ag_models/days_predictor 里已有训练好的模型。
2. predict_need_data 文件夹里有对应大棚的新图片。
3. MySQL 的 data 表里有对应时间附近的传感器数据。
```

给人看的终端演示版：

```bat
python predict_demo.py --shed 1 --crop 西红柿 --plant-date 2025-06-10
```

给看板/API 调用的 JSON 版：

```bat
python predict.py --shed 1 --crop 西红柿 --plant-date 2025-06-10 --json
```

看板版成功时会输出类似：

```json
{
  "ok": true,
  "data": {
    "shed": "1",
    "crop": "西红柿",
    "predicted_days_remaining": 76.98,
    "predicted_maturity_date": "2025-10-10"
  }
}
```

参数说明：

```text
--shed          大棚编号，1-8
--crop          作物名，目前正式可用：西红柿 / 番茄 / tomato
--plant-date    实际种植日期，格式 YYYY-MM-DD
--json          输出 JSON，给看板调用时必须加
--image-dir     预测图片目录，默认 predict_need_data
```


九、关于辣椒和茄子
----------------

`predict.py` 已经预留了作物接口：

```text
西红柿 tomato
辣椒 pepper
茄子 eggplant
```

但现在真正训练好的只有西红柿模型。辣椒、茄子后续要能准，需要：

1. 收集对应作物图片和传感器数据。
2. 整理对应作物的 `final_train_data_ready.csv`。
3. 单独训练对应作物模型。
4. 在 `predict.py` 里启用对应作物模型路径。

不要直接拿西红柿模型长期预测辣椒或茄子，这样结果只能当临时参考，不能当正式结果。


十、常见问题
----------

问题 1：`conda` 不是内部或外部命令。

解决：

```bat
C:\Users\win\miniconda3\Scripts\activate.bat ag_greenhouse
```

问题 2：训练几分钟没动。

解决：

```text
先按 Esc，确认不是 cmd 选中文字导致程序暂停。
再看终端有没有 train_loss / val_loss 输出。
```

问题 3：预测提示找不到图片。

解决：

```text
检查 predict_need_data 里是否有该棚对应图片。
检查图片名是否包含 1-1、1-2 这类摄像头名，或包含 tbcamera 表里的 cameraId。
```

问题 4：预测提示传感器数据不完整。

解决：

```text
检查 MySQL 的 data 表。
确认对应大棚、对应图片时间附近有 9 个传感器指标：
光照、土壤PH、土壤温度、土壤湿度、氮、磷、空气温度、空气湿度、钾。
```

问题 5：想删掉“已种植天数”这个静态属性。

说明：

```text
当前模型训练时用了“已种植天数”，预测时也必须提供。
如果要删掉这个属性，必须重新整理数据并重新训练模型。
否则旧模型不能直接正常使用。
```


十一、推荐执行顺序
----------------

第一次完整跑项目，按下面顺序来：

```bat
D:
cd D:\青海_greenhouse\ar_greenhouse
C:\Users\win\miniconda3\Scripts\activate.bat ag_greenhouse
python prepare_data_from_mysql.py
python train_script.py --max-epochs 100
python predict_demo.py --shed 1 --crop 西红柿 --plant-date 2025-06-10
python predict.py --shed 1 --crop 西红柿 --plant-date 2025-06-10 --json
```

到这里如果都能跑通，就说明数据整理、训练、预测、看板接口都正常。

