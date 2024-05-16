Ubuntu22.04

CUDA11.8

SAM开源地址：https://yuval-alaluf.github.io/SAM。原项目采用的训练数据集为 FFHQ，测试数据集为 CelebA-HQ，而本人扩充的质量较高的亚洲人脸数据集筛选自 AFAD，AFAD 数据集包含亚洲人脸图像超过160000张，其数据质量良莠不齐，包含很多难以使用的噪声，本人筛选的质量较高的部分 AFAD 数据总量达24000多张，提取方式：链接：https://pan.baidu.com/s/1SfeghxqbWnxbhyBSJqHHzA 提取码：AFAD；本人将这些亚洲人面部特征图像数据扩充到 FFHQ 中并全部投入 SAM 进行模型的重新训练，得到了新的对亚洲人面部特征更加适应，身份保护能力更强的权重文件 best_model.pt，这个权重文件应该置于 experiment_01/checkpoints 目录之下，提取方式：

本人联系方式：dinglizhong666@yeah.net，如果对亚洲人脸数据集有需求，或者其他技术问题，欢迎交流！

本项目部署的人脸区域检测模型和权重，人脸年龄估计模型和权重在 detector_models 目录下，这与 SAM 并无直接联系。

对于人脸年龄变换，本项目对 SAM 进行了数据集扩充和重新训练，并且利用原项目中 scripts/inference_side_by_side.py 进行测试和部署，原项目是在终端用命令进行运行的，本项目将代码重构为 Pycharm 一键运行，执行程序为 scripts/inference_side_by_side.py，此外，本项目中的主要工作以及对应的实现代码也位于该文件内，包含人脸区域检测任务和人脸年龄估计任务的实现，可供 SAM 产出良好变换效果的局部图像的裁剪策略建模，用户友好的交互界面，文件夹及图片文件操作，结果的展现与存储等。如果您想要实现 SAM 中其他测试任务，可根据 SAM 开源项目自行调整。
