# AutoBIM Studio

AutoBIM Studio 是 Windows + WSLg 或 Linux 上的本地照片建模工作台。选择同一场景的 JPEG 目录和 ADE20K SegFormer-B0 ONNX 权重后，程序依次运行 PyCOLMAP、ONNX Runtime 分割、仓库的几何加权投票、DBSCAN 和 IFC4 导出校验。输出为稀疏点簇的凸包候选，实际尺度需要现场标定。

## 启动

当前开发机双击 AutoBIM-Studio.cmd。新环境需要 WSLg、Ubuntu、Python 3.12、Tk 和可用的 pip。在仓库目录执行以下命令。

    sudo apt install python3-tk python3-pip
    python3 -m pip install --target "$HOME/.local/autobim/lib" -r requirements-studio.txt
    export PYTHONPATH="$HOME/.local/autobim/lib"
    python3 -m autobim_studio.desktop

权重单独提供。测试使用 Xenova/segformer-b0-finetuned-ade-512-512 的 onnx/model.onnx（输入 pixel_values），可在窗口选择，或用 AUTOBIM_MODEL 设置默认路径。照片路径支持 Windows 盘符和 Linux 路径。按单个拍摄场景建项目，照片文件主名应唯一。

## 项目运行与恢复

    python3 -m autobim_studio.jobs run /data/scene/images /models/segformer.onnx
    python3 -m autobim_studio.jobs resume /path/to/project

默认项目位置为 output/studio。每次新建都会分配独立目录，记录输入文件 SHA-256、阶段、状态与日志。继续项目会复核输入并跳过已完成阶段。同一个项目通过文件锁限制并发写入。关闭窗口后任务可继续运行，重开可查看记录。异常中断后可选择项目并点击继续。

result/semantic.ifc 是通过回读和几何检查的模型，semantic.json 记录点数、实例类别和校验结果。pipeline.log 保存处理细节。缺少权重、图像重叠不足或没有有效实例时，请检查错误日志。
