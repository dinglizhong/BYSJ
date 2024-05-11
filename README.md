Ubuntu22.04

CUDA11.8

SAM开源地址：https://yuval-alaluf.github.io/SAM

本人联系方式：dinglizhong666@yeah.net，如果对亚洲人脸数据集有需求，或者其他技术问题，欢迎交流！

本项目部署的人脸区域检测模型和权重，人脸年龄估计模型和权重在 detector_models 目录下，这与 SAM 并无直接联系。

对于人脸年龄变换，本项目对 SAM 进行了数据集扩充和重新训练，并且利用原项目中 scripts/inference_side_by_side.py 进行测试和部署，原项目是在终端用命令进行运行的，本项目部署为 Pycharm 一键运行，执行代码为 scripts/inference_side_by_side.py，此外，本项目中的主要工作也位于该文件内，包含人脸区域检测任务和人脸年龄估计任务的实现，可供 SAM 产出良好变换效果的局部图像的裁剪策略建模，用户友好的交互界面，文件夹及文件操作，结果的展现与存储等。如果您想要实现 SAM 中其他测试任务，可根据 SAM 开源项目自行调整。
