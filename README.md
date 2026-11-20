# Tracefolio Models

Tracefolio 独立识别运行包。主程序与资料库不在本仓库。

每个平台一个完整运行包，包含固定 Python 解释器、worker、依赖、OCR/ASR/对象/姿态/本地描述模型，以及 FFmpeg 和 llama-server。模块按任务独立加载，连续文件复用已加载 worker。用户无需安装 Python。

当前准入目标为 Linux amd64/arm64，最低 Ubuntu 22.04（glibc 2.35）。Windows/macOS 的完整模型运行包尚未准入；主程序核心支持不代表这些平台可执行本运行包。

在 Tracefolio「设置 → 模型/运行包」填写 `AnnaofArendelle/tracefolio-models`，检查 Latest，下载后显式激活。也可导入 `tracefolio-models-linux-<arch>.tar.gz`。包导出不包含用户资料或凭据；旧任务保持自己的包身份。

## 构建与验证

在原生 Ubuntu 22.04 runner 安装固定 uv 后，执行 `bash scripts/build_pack.sh`。模型文件固定保存在本地 `assets/models/`，构建缺失时直接报错，不访问上游模型地址。CI 使用本仓库锁定的 model-weights 附件还原同字节副本，随后也从本地文件构建。上游地址仅保留作来源记录。依赖与执行器使用锁定的 wheel、Python standalone 和原生 FFmpeg/llama.cpp 源码；准备好依赖缓存后可执行 `bash scripts/build_pack.sh --offline`，离线缺输入不会回退联网。构建结果、许可证、对应源码及 SHA256SUMS 一起发行；CI 真实模型预检通过后才发布 Latest。

`workers/tests` 检验适配器与文件状态隔离；`packaging/probes` 是固定真实推理预检样本。样本来源见各目录 source/README，未纳入私有研究资料。

本项目源码采用 AGPL-3.0-only。权重、第三方库和原生执行器按 `packaging/licenses`、模型清单和运行包内许可分别分发；包括 GPL FFmpeg/x264、FunASR 模型许可和 LFM Open License，不能把整个包视为单一 AGPL 或 MIT 授权。
