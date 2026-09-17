# true_high_speed Python 模块与流程说明

本文件用于说明 `true_high_speed` 项目中自有 Python 代码的职责与相互关系。

`third_party/` 目录中的第三方源码不在本文逐文件展开，只作为 LocoTrack、Track-On2 等算法的依赖存在。

---

## 整体结构

项目中的 Python 代码主要分为四条流程：

```text
A. 圆环检测 + 跨帧坐标关联
B. LocoTrack 通用点跟踪
C. TAPNext / TAPNext++ 通用点跟踪
D. Track-On2 通用点跟踪
```

除此之外，还有若干相对独立的视频处理、传统视觉、模型检查和数据辅助脚本。

---

# A. 圆环检测 + 跨帧坐标关联

这一流程包含完整的数据制作、模型训练、PyTorch 推理、ONNX 导出和 ONNX Runtime 推理链路。

## 流程关系

```text
原始视频
   │
   ▼
prepare_ring_pose_dataset.py
   │
   ▼
YOLO Pose 数据集
   │
   ├───────────────┐
   ▼               ▼
train_ring_pose.py
train_ring_pose_by_accuracy.py
   │
   ▼
圆环 Pose 权重
   │
   ├─────────────────────────────────────┐
   │                                     │
   ▼                                     ▼
prepare_coordinate_dataset_from_video.py  infer_ring_points_video.py
   │                                     ▲
   ▼                                     │
坐标关联标注数据                          │
   │                                     │
   ▼                                     │
train_coordinate_association.py          │
   │                                     │
   ↕                                     │
coordinate_association_model.py ─────────┘
   │
   ▼
坐标关联模型权重

圆环 Pose 权重 + 坐标关联模型权重
   │
   ▼
export_models_to_onnx.py
   │
   ├── ring_pose.onnx
   └── coordinate_association.onnx
          │
          ▼
infer_ring_coordinate_onnx_video.py
```

## `dataset_generater/prepare_ring_pose_dataset.py`

圆环 Pose 数据集制作脚本。

主要功能：

- 从输入视频中按设定 FPS 抽取图像；
- 建立 train / val / test 数据目录；
- 使用 Matplotlib 进行交互式人工标注；
- 标注圆环目标框和圆心关键点；
- 支持同时标注易与圆环混淆的其他目标；
- 生成后续 YOLO Pose 训练所需的数据集。

它是 A 流程最前端的数据制作入口。

---

## `trainning_code/train_ring_pose.py`

常规圆环 YOLO Pose 训练脚本。

主要功能：

- 读取 `prepare_ring_pose_dataset.py` 生成的数据集；
- 自动生成/使用 YOLO 数据集配置；
- 检查训练、验证和测试标签；
- 根据设备情况选择训练参数；
- 基于预训练 Pose 模型进行训练；
- 可在测试集上执行评估。

输出的圆环 Pose 权重会继续用于坐标数据制作、PyTorch 推理以及 ONNX 导出。

---

## `trainning_code/train_ring_pose_by_accuracy.py`

另一套圆环 Pose 训练策略。

与 `train_ring_pose.py` 相比，它更关注圆环类别本身的检测效果，并使用自定义指标逻辑监控训练过程。

主要关注：

- Class-0 precision；
- Class-0 recall；
- Class-0 F1；
- 根据圆环类别实际效果保存最佳模型和执行早停。

该脚本与 `train_ring_pose.py` 属于两种可替换的圆环 Pose 训练方案。

---

## `dataset_generater/prepare_coordinate_dataset_from_video.py`

跨帧坐标关联模型的数据制作脚本。

主要功能：

- 使用训练好的圆环 Pose 模型逐帧检测圆环；
- 获取圆环框、圆心和检测特征；
- 使用 Matplotlib 交互界面人工分配持续 ID；
- 记录目标在不同帧中的身份和可见状态；
- 生成用于坐标关联模型训练的 NPZ 场景数据。

它位于圆环检测模型和坐标关联模型之间，是两阶段训练流程的连接点。

---

## `coordinate_association_model.py`

跨帧坐标关联的核心共享模块。

主要包含：

- 检测特征归一化；
- 历史轨迹表示；
- 当前检测表示；
- 21 帧时域上下文；
- 轨迹与检测编码；
- 运动关系建模；
- 刚体/相对位置关系建模；
- 已有轨迹与 `NEW` 目标的匹配判断；
- 一对一 ID 解码；
- 坐标关联模型 checkpoint 加载。

该文件本身不是单独的视频处理入口，而是 A 流程中的核心算法模块，被训练、PyTorch 推理和 ONNX 导出共同使用。

---

## `trainning_code/train_coordinate_association.py`

坐标关联模型训练脚本。

主要功能：

- 读取坐标标注 NPZ 场景；
- 构造历史轨迹、当前检测和时域上下文；
- 生成已有轨迹 / `NEW` 目标监督标签；
- 调用 `coordinate_association_model.py` 中的模型结构；
- 执行数据增强；
- 对错误 `NEW` 判断增加额外惩罚；
- 执行训练、验证和早停；
- 保存最佳坐标关联模型。

它与 `coordinate_association_model.py` 和 `prepare_coordinate_dataset_from_video.py` 共同组成坐标关联训练链。

---

## `infering_code/infer_ring_points_video.py`

圆环方案的 PyTorch / Ultralytics 完整视频推理入口。

处理流程：

```text
视频帧
  ↓
YOLO Pose 圆环检测
  ↓
圆心 / 检测特征
  ↓
时域上下文
  ↓
CoordinateAssociationModel
  ↓
持续 ID
  ↓
标注视频 + CSV
```

该脚本直接复用 `coordinate_association_model.py` 中的关联逻辑，因此与训练阶段共享相同的核心模型定义。

---

## `generate_and_test_onnx/export_models_to_onnx.py`

项目中的 ONNX 导出脚本。

当前 A 流程中主要负责：

- 导出圆环 YOLO Pose 模型；
- 导出坐标关联模型；
- 生成 `ring_pose.onnx`；
- 生成 `coordinate_association.onnx`。

文件中还包含 TAPNext / TAPNext++ 相关 ONNX 包装和兼容代码，但当前主要实际使用的是圆环检测与坐标关联模型导出流程。

---

## `generate_and_test_onnx/infer_ring_coordinate_onnx_video.py`

圆环方案的纯 ONNX Runtime 视频推理版本。

主要功能：

- 使用 ONNX Runtime 执行圆环检测；
- 解析 YOLO Pose 输出；
- 执行 letterbox、NMS 和关键点解码；
- 构造检测特征；
- 维护跨帧轨迹状态；
- 构造时域上下文；
- 使用 `coordinate_association.onnx` 分配持续 ID；
- 输出标注视频和坐标 CSV。

它与 `infer_ring_points_video.py` 完成相同类型的任务，但部署后端从 PyTorch/Ultralytics 切换为 ONNX Runtime。

---

# B. LocoTrack 通用点跟踪

LocoTrack 在项目中同时保留三种运行路径：

```text
LocoTrack PyTorch
    │
    ├── eager 原生推理
    │
    ├── TorchScript 固定帧档位
    │
    └── ONNX 动态帧 / 动态点
```

## 流程关系

```text
LocoTrack 第三方 PyTorch 实现
        │
        ├──────────────┐
        │              │
        ▼              ▼
LocoTrack_B...py   LocoTrack_S...py
   eager Base       eager Small

        │
        ├─────────────────────────────────┐
        │                                 │
        ▼                                 ▼
convert_locotrack...torchscript.py   convert_locotrack...dynamic_onnx.py
        │                                 │
        ▼                                 ▼
多档 TorchScript                    动态 ONNX
        │                                 │
        ▼                                 ▼
infer_LocoTrack_TorchScript_video.py  LocoTrack_ONNX_matplotlib_video_infer.py
```

## `infering_code/LocoTrack_B_matplotlib_video_infer.py`

LocoTrack Base 的原生 PyTorch 视频推理脚本。

主要功能：

- 自动准备 LocoTrack 源码；
- 加载 Base checkpoint；
- 使用 Matplotlib 在首帧选择跟踪点；
- 执行 GPU 推理；
- 支持 FP16；
- 在显存不足时进行分段处理和续接；
- 输出跟踪视频和坐标 CSV。

它主要用于验证 Base 模型原始 PyTorch 行为，也是后续部署模型转换的重要参考基线。

---

## `infering_code/LocoTrack_S_matplotlib_video_infer.py`

LocoTrack Small 的原生 PyTorch 视频推理脚本。

主要功能与 Base 版本类似：

- 加载 Small checkpoint；
- 首帧交互选择跟踪点；
- 执行视频点跟踪；
- 输出标注视频和 CSV。

该文件用于 Small 模型的原生行为验证。

---

## `generate_and_test_torchscript/convert_locotrack_checkpoints_to_torchscript.py`

LocoTrack TorchScript 转换脚本。

它会把 Small / Base 模型转换成多个固定帧数档位：

```text
64
32
16
8
4
2
```

主要功能：

- 自动准备 LocoTrack PyTorch 源码；
- 加载 checkpoint；
- 为不同帧数分别 trace；
- 生成多个 TorchScript 模型；
- 转换过程中若某个大档位显存不足，则跳过该档位并继续尝试更小档位。

这种多档位结构用于后续根据显存情况自动选择可运行的模型。

---

## `generate_and_test_torchscript/infer_LocoTrack_TorchScript_video.py`

LocoTrack TorchScript 视频推理脚本。

主要功能：

- 首帧 Matplotlib 点选；
- 自动选择可用的最大帧数模型；
- 按 `64 → 32 → 16 → 8 → 4 → 2` 的方向进行降档；
- 支持 CUDA OOM 后分段继续处理；
- 根据可见点情况选择下一段的续接帧；
- 合并各段结果；
- 输出标注视频和坐标 CSV。

它与上一条 TorchScript 转换脚本直接配套。

---

## `generate_and_test_onnx/convert_locotrack_checkpoints_to_dynamic_onnx.py`

LocoTrack 动态 ONNX 转换脚本。

目标是将 LocoTrack 转换成：

- 动态帧数；
- 动态跟踪点数量；

的 ONNX 模型。

主要包含：

- 自动准备官方 PyTorch 模型源码；
- 加载 Small / Base checkpoint；
- 针对 ONNX 导出处理 GridSample 和坐标计算；
- 使用动态轴；
- 导出后执行动态输入验证。

该路线与固定帧 TorchScript 路线相互独立，用于探索更加通用的 ONNX 部署形式。

---

## `generate_and_test_onnx/LocoTrack_ONNX_matplotlib_video_infer.py`

LocoTrack ONNX 视频验证脚本。

主要功能：

- 首帧 Matplotlib 交互选点；
- 使用 ONNX Runtime CUDAExecutionProvider；
- 执行动态 ONNX 推理；
- 支持显存不足后的分段处理；
- 根据可见点选择续接位置；
- 合并各段结果；
- 输出标注视频和坐标 CSV。

它与 `convert_locotrack_checkpoints_to_dynamic_onnx.py` 直接配套。

---

# C. TAPNext / TAPNext++

这一部分用于验证 TAP 系列模型的流式点跟踪能力，并进一步转换成适合 C++ / LibTorch 部署的 TorchScript 模型。

## 流程关系

```text
TAP / TAPNext++ checkpoint
       │
       ├──────────────┐
       │              │
       ▼              ▼
TAPNextPP_fast.py  TAPNextPP_stable.py
  eager 实验         eager stable
                         │
                         ▼
convert_tapnextpp_stable_512_torchscript.py
                         │
                         ▼
               TAPNext++ TorchScript
                  ┌──────┴──────┐
                  ▼             ▼
infer_tapnextpp_stable...py   tapnextpp_compare_cuda_profiler.py
```

## `infering_code/TAPNextPP_fast.py`

TAP 系列快速实验脚本。

主要功能：

- 使用较低输入分辨率的模型进行视频跟踪；
- 自动准备 checkpoint；
- 首帧选择跟踪点；
- 维护模型递归 state；
- 使用 CUDA / FP16 autocast 推理；
- 输出视频和 CSV。

它主要承担快速验证和实验用途。

---

## `infering_code/TAPNextPP_stable.py`

TAPNext++ 512 stable 的原生流式推理脚本。

主要功能：

- 加载 stable 模型；
- 首帧选择点；
- 调用模型原始 `track_frame` 流式接口；
- 逐帧维护内部状态；
- 输出标注视频和 CSV。

该文件是后续 TorchScript 转换时的重要原始行为参考。

---

## `generate_and_test_torchscript/convert_tapnextpp_stable_512_torchscript.py`

TAPNext++ stable TorchScript 转换脚本。

主要功能：

- 将原始模型包装成适合部署的两个方法：

```text
initialize
step
```

- 处理不利于 TorchScript 转换的算子实现；
- trace 流式初始化和逐帧递归过程；
- 使用原始模型进行多帧结果对比；
- 检查转换前后输出一致性；
- 保存 TorchScript 模型；
- 生成模型接口信息。

最终模型可以直接在 C++ / LibTorch 中通过 `initialize` 和 `step` 调用。

---

## `generate_and_test_torchscript/infer_tapnextpp_stable_torchscript_video.py`

TAPNext++ TorchScript 视频推理与测速脚本。

主要功能：

- 首帧选择跟踪点；
- 调用 TorchScript 的 `initialize`；
- 后续逐帧调用 `step`；
- 预热两条 JIT 执行路径；
- 检查模型输出是否保持在 CUDA；
- 统计冷启动时间；
- 统计稳态单帧延迟；
- 记录峰值显存；
- 输出标注视频和 CSV。

它用于验证最终 TorchScript 模型是否满足实际流式部署要求。

---

## `generate_and_test_torchscript/tapnextpp_compare_cuda_profiler.py`

TAPNext++ CUDA 性能对比脚本。

主要用于比较：

```text
官方 checkpoint
修改后的 eager 模型
TorchScript 模型
```

在相同输入条件下的性能差异。

主要统计：

- CUDA Event 延迟；
- 单步推理耗时；
- CUDA / PyTorch 环境信息；
- profiler 算子信息；
- JSON / CSV 性能结果；
- profiler trace。

该脚本用于确认模型转换后的性能变化以及定位主要耗时来源。

---

# D. Track-On2 通用点跟踪

Track-On2 路线目前主要围绕 DINOv2 backbone 的流式 TorchScript 部署。

## 流程关系

```text
Track-On2 第三方模型
      +
DINOv2-Small
      +
Track-On2 checkpoint
      │
      ▼
convert_trackon2_dinov2_to_torchscript.py
      │
      ▼
trackon2_dinov2_pure_torchscript.pt
      │
      ▼
infer_trackon2_torchscript_video.py
```

## `generate_and_test_torchscript/convert_trackon2_dinov2_to_torchscript.py`

Track-On2 DINOv2 TorchScript 转换脚本。

主要功能：

- 自动准备 Track-On2 源码；
- 自动准备 DINOv2-Small；
- 加载 Track-On2 checkpoint；
- 使用纯 PyTorch 实现兼容原模型中的多尺度可变形注意力；
- 避免运行时依赖 MMCV 自定义算子；
- 将模型包装为流式：

```text
initialize
step
```

- 导出 TorchScript；
- 使用不同点数量验证动态点维度。

该脚本的目标是得到更适合 Qt / LibTorch 环境使用的 Track-On2 模型。

---

## `generate_and_test_torchscript/infer_trackon2_torchscript_video.py`

Track-On2 TorchScript 视频推理脚本。

主要功能：

- 首帧交互选择跟踪点；
- 将原始视频坐标映射到 Track-On2 模型输入坐标；
- 执行模型初始化；
- 使用 `step` 逐帧递归推理；
- 执行必要的 GPU 预热；
- 将预测坐标映射回原视频分辨率；
- 输出标注视频和坐标 CSV。

它用于验证转换后的 Track-On2 TorchScript 模型和实际流式部署接口。

---

# 独立实验与辅助脚本

下面的文件不属于 A/B/C/D 的核心流水线，主要用于独立实验、数据处理或模型检查。

## `infering_code/lightflow.py`

传统视觉点跟踪实验。

主要流程：

```text
阈值 / 轮廓检测
→ 圆形候选
→ 手工选择目标
→ 邻域角点
→ Lucas-Kanade 金字塔光流
→ 视频跟踪结果
```

用于与深度学习点跟踪方案进行传统视觉方法对照。

---

## `infering_code/yoloe_visual_prompt_ring.py`

YOLOE Visual Prompt 圆环检测实验。

通过参考图像中的手工目标框构造视觉提示，再将提示用于后续视频中的相似目标检测。

该脚本属于独立检测方案实验，不依赖项目自训练的坐标关联模型。

---

## `util/cutting.py`

视频拆帧工具。

将指定视频逐帧导出为 JPG 文件，写入对应输出目录。

---

## `util/generate_print_picture.py`

圆环测试图生成工具。

生成规则排列的黑色外圈 / 白色内圈圆环图片，可用于打印或视觉测试。

---

## `util/onnx_to_json_manifest.py`

ONNX 模型结构检查工具。

可以读取 ONNX 并输出 JSON 描述，包括：

- opset；
- 输入输出张量；
- 节点数量；
- 算子统计；
- initializer；
- metadata；
- 可选 ONNX checker 检查。

适合快速确认导出模型的接口和结构。

---

## `util/reduce_fps.py`

视频降帧率工具。

主要功能：

- 使用 `ffprobe` 获取原视频 FPS 和帧数；
- 交互输入目标 FPS；
- 使用 `ffmpeg` 重新编码；
- 输出降低帧率后的视频。

---

## `util/yolo_red_points_on_white.py`

圆环检测结果可视化工具。

使用圆环 Pose 模型检测视频中的圆心，并：

- 过滤部分重叠或距离过近的候选；
- 将检测到的圆心绘制成白底红点；
- 使用 FFmpeg 输出可视化视频。

主要用于快速检查圆环检测结果和目标分布。

---

# 包初始化文件

以下文件仅用于 Python 包结构，本身没有独立业务逻辑：

```text
__init__.py
dataset_generater/__init__.py
infering_code/__init__.py
trainning_code/__init__.py
```

---

# 第三方依赖说明

`third_party/` 下保存的是项目依赖的外部算法源码，不在本文逐文件介绍。

当前主要包括：

- LocoTrack；
- Track-On2。

本项目自己的脚本主要负责在这些模型基础上完成：

```text
原生模型验证
→ 模型转换
→ TorchScript / ONNX 验证
→ CUDA 推理
→ 视频可视化
→ CSV 输出
→ 性能测试
→ Qt / C++ 部署准备
```

---

# 总结

Python 部分主要承担模型研发、算法验证和部署模型生成工作。

整体覆盖：

```text
数据制作
→ 模型训练
→ PyTorch eager 验证
→ TorchScript / ONNX 转换
→ CUDA 推理
→ OOM 分段与恢复
→ 视频标注
→ CSV 坐标输出
→ 性能 Profiling
→ Qt / C++ 部署模型准备
```

其中：

- **A 流程**：自建圆环检测与跨帧 ID 关联；
- **B 流程**：LocoTrack 的 PyTorch / TorchScript / ONNX 多后端验证；
- **C 流程**：TAPNext / TAPNext++ 的流式推理、TorchScript 转换与性能分析；
- **D 流程**：Track-On2 DINOv2 的纯 TorchScript 流式部署验证。

这些 Python 流程最终为 `true_high_speed` 的 Qt/C++ 高速视频分析程序提供模型、算法验证结果和部署基线。
