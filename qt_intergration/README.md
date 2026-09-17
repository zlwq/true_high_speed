# true_high_speed 构建与使用

## 技术栈与当前构建环境

当前工程为 Windows x64 下的 Qt Widgets / C++ 高速视频运动分析程序，主要技术栈：

- Qt 6.8.3，静态链接版本，MSVC 2022 64-bit。
- CMake + Release 构建。
- Qt 使用 static + `/MD` 运行库配置；Qt 本身不依赖 Qt DLL。
- CUDA 13 + cuDNN 9。
- LibTorch CUDA，用于 LocoTrack、Track-On2、TAPNext++ 等 TorchScript 模型推理。
- ONNX Runtime GPU 1.29.0，用于 ONNX 模型 CUDA 推理。
- OpenCV 4.14.0，用于视频读取、图像处理与视频写出。
- FFmpeg，用于 OpenCV 无法正常解码的视频自动兼容转换。

最终程序虽然使用静态 Qt，但 AI 推理栈仍使用动态库，因此发布时仍需携带
LibTorch、ONNX Runtime、OpenCV、CUDA/cuDNN 等运行时 DLL。由于当前 Qt
静态构建使用 `/MD`，目标 Windows 机器还需要可用的 Microsoft Visual C++
Redistributable。

## 工程结构

- `CMakeLists.txt`：工程构建入口。启用 C++20，查找并链接 Qt6 Core/Gui/Widgets、
  OpenCV、LibTorch 和 ONNX Runtime；登记全部源文件，并为 MSVC 设置 `/utf-8`、
  `/bigobj` 和 `/MD` 运行库。构建完成后会把 LibTorch、ONNX Runtime、OpenCV
  运行时 DLL 以及可找到的 `ffmpeg.exe` 复制到 exe 目录，同时提供 Qt install /
  deploy 脚本。实际使用静态还是动态 Qt 由所选 Qt Kit 决定。
- `main.cpp`：只负责启动 Qt 应用。
- `mainwindow.cpp/.h/.ui`：视频选择、四个方案按钮、首帧点选和界面状态。
- `inferenceworker.cpp/.h`：在独立 `QThread` 中运行推理。
- `ringinference.cpp/.h`：`ring_pose.onnx` 检测和
  `coordinate_association.onnx` 动态 ID 关联。
- `locotrackinference.cpp/.h`：LocoTrack-B 固定帧 TorchScript 分段推理、
  OOM 自动降档和续接合并。
- `trackon2inference.cpp/.h`：Track-On2 DINOv2 TorchScript CUDA 逐帧推理。
- `tapinference.cpp/.h`：TAPNext++ TorchScript CUDA 推理。
- `physicalanalysis.cpp/.h`：读取坐标 CSV、首帧交互标定、短缺失插值与
  速度/加速度计算。
- `reviewdialog.cpp/.h`：物理分析结果回看窗口。读取并校验本程序生成的物理信息
  CSV，解析原视频/处理后视频路径、标定信息和物理量列；用户可选择需要查看的
  线速度、加速度、角速度、角加速度等曲线。窗口使用 OpenCV 播放处理后视频，
  曲线与当前视频帧同步，支持播放/暂停、鼠标悬停查看某帧数值，以及点击曲线
  跳转到对应视频帧。`reviewdialog.h` 对外只暴露 `openReviewDialog(...)` 入口。
- `csvmetadata.h`：在 CSV 顶部写入源视频绝对路径。

## 模型目录

在工程根目录创建 `model` 文件夹，并放入：

```text
model/
  ring_pose.onnx
  coordinate_association.onnx
  tapnextpp_stable_512_q4_fp16_ts.pt
  trackon2_dinov2_pure_torchscript.pt
  localtrack/
    locotrack_base_t64_torchscript.pt
    locotrack_base_t32_torchscript.pt
    locotrack_base_t16_torchscript.pt
    locotrack_base_t8_torchscript.pt
    locotrack_base_t4_torchscript.pt
    locotrack_base_t2_torchscript.pt
```

Qt Creator 中直接运行时，程序会依次查找：exe 旁的 `model`、exe 上一级的
`model`、当前工作目录的 `model`、CMake 源码目录的 `model`。

## FFmpeg 自动兼容转换

程序在启动任意视频处理方案前，都会先用 OpenCV 实际读取第一帧。若视频
容器可以打开但第一帧无法解码，程序会自动调用 FFmpeg，将视频高质量转换为
H.264 `yuv420p`：

```text
<原视频名>_h264.mp4
```

转换文件保存在原视频所在目录，随后四种处理方案都会改用该文件。转换使用
`-vsync 0` 保留原始帧序列，不包含音频。请将 `ffmpeg.exe` 放在程序 exe
旁边，或把 FFmpeg 加入系统 `PATH`。如果 CMake 配置时能在 `PATH` 中找到
FFmpeg，构建后会自动把它复制到 exe 目录。

## Qt Creator 配置

使用自编译的 Qt 6.8.3 静态套件，MSVC 2022 64-bit，`/MD` 运行库。当前已验证的
构建配置对应：

```text
Qt 6.8.3 static
MSVC 2022 64-bit
Release
static runtime: OFF (/MD)
```

Qt Creator 中请选择指向本机 `6.8.3-static-md` 安装目录的 Kit；Kit 的显示名称
可以不同，以实际 Qt 安装路径和编译器配置为准。

工程已给出以下默认依赖路径：

```text
Torch_DIR=D:/QTDEPS/libtorch/share/cmake/Torch
ONNXRUNTIME_ROOT=D:/QTDEPS/ONNXRUNTIME-WIN-X64-GPU_CUDA13-1.29.0
```

还需要让 CMake 找到 OpenCV。例如在 Qt Creator 的 CMake 配置中添加：

```text
OpenCV_DIR=D:/QTDEPS/opencv/build
```

具体值应指向包含 `OpenCVConfig.cmake` 的目录。

构建环境还需要可用的 CUDA 13 工具链，以及与当前 CUDA/cuDNN 运行时兼容的
NVIDIA 驱动。

使用 `Release` 构建。Qt 已静态链接进 `true_high_speed.exe`，因此发布目录
通常不再需要 `Qt6Core.dll`、`Qt6Widgets.dll` 等 Qt 动态库。CMake/发布目录
仍需准备 LibTorch、ONNX Runtime、OpenCV、CUDA/cuDNN 等推理运行时 DLL，
以及 `ffmpeg.exe`。

## GitHub 仓库建议

`.qtcreator` 和 `build` 都属于本机 IDE / 构建生成内容，不建议提交到仓库。
可以在 `.gitignore` 中加入：

```gitignore
.qtcreator/
build/
```

模型权重文件通常较大；如果需要直接保存在 GitHub 仓库中，建议使用 Git LFS。
源码仓库重点保留 `CMakeLists.txt`、`*.cpp`、`*.h`、`*.ui`、README 和必要的
模型/配置说明。

## 当前按钮行为

- “识别圆环方案”：执行两遍视频处理，第一遍检测全视频圆环，第二遍使用
  21 帧上下文动态关联 ID。
- “localtrack通用跟踪点方案”：读取整段视频并缩放到 256×256，优先使用
  LocoTrack-B 固定 64 帧模型；显存不足时自动依次降至 32/16/8/4/2 帧。
  每段使用最后一个全部点可见帧续接；若不存在，则使用最后一个可见点最多帧。
  C++ 端使用 FP32 推理，避免 FP16 数值溢出产生 NaN；若输出仍含 NaN/Inf，
  程序会直接报错，不再写出没有标记的错误结果。
- “trackon通用跟踪点方案”：使用 `384×512` 输入和
  `initialize/step` 状态递推，逐帧执行 Track-On2 DINOv2。为缩短启动时间，
  不执行 `freeze_module` 和额外预热；模型首次使用时加载一次，连续再次使用
  Track-On2 时直接复用显存中的模型。切换到其他方案时会释放该缓存。
- “tapnetpp+通用跟踪点方案”：使用 `512×512` 输入和
  `initialize/step` 状态递推，逐帧执行 TAPNext++；不执行额外预热。
- 三个通用点跟踪方案都会读取原始视频第一帧；左键添加点，右键在点附近
  删除。LocoTrack 最多选择 64 个点。
- “读取csv生成物理信息”：只接受本程序当前版本生成的原始坐标 CSV。
  程序根据 CSV 顶部保存的绝对路径读取源视频第一帧；源视频不存在时直接
  弹窗提示。像素标定可手动填写“像素/米”，也可在画面中拖出一段已知长度，
  输入米或毫米后自动换算。时间标定单位为“帧/秒”，默认使用源视频 FPS。
  单击目标点选择单点；从目标点 A 拖到目标点 B 建立有方向的线段，再按
  “标定并生成 CSV”输出物理信息。
- 关闭正在处理的窗口时会询问是否取消。
- CUDA 不可用时会明确报错，不会自动切换为 CPU 推理。

输出写到输入视频所在目录：

```text
<视频名>_ring_annotated.mp4
<视频名>_ring_raw.csv
<视频名>_locotrack_base_torchscript_annotated.mp4
<视频名>_locotrack_base_torchscript_raw.csv
<视频名>_track_on2_dinov2_annotated.mp4
<视频名>_track_on2_dinov2_raw.csv
<视频名>_tap_annotated.mp4
<视频名>_tap_raw.csv
<原始坐标CSV名>_physical.csv
```

圆环/TAPNext++ CSV 中的 `video_timestamp_sec`，以及 LocoTrack/Track-On2
CSV 中的 `timestamp_sec`，都是 MP4 播放时间。每份 CSV 第一行保存
`#source_video_path=<源视频绝对路径>`。所有原始坐标 CSV 均不输出
`visible` 列；不可见点通过空白坐标单元格表示。TAPNext++ 坐标直接保留
浮点数。

物理信息计算先对每个所需目标点独立处理连续空白：仅当空白长度不超过
3 帧且前后都有有效坐标时，按标定时间进行线性插值；更长空白保持为空。
速度和加速度优先使用中心差分，只有一侧可用时使用单边差分。单点输出
速度大小与加速度大小；线段输出其中点的线速度大小、角速度和角加速度。
任何端点在当前帧为空时，该线段当前帧的全部物理量均为空。角度单位为
弧度，坐标方向沿用视频坐标系（X 向右、Y 向下）。物理信息 CSV 顶部还会
保存本次使用的 `pixels_per_meter` 和 `frames_per_second`。
