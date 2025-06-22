# RRMediCa
#### 项目描述
瑞金医学影像分类框架(RRMediCa)，专为CT、MR等3D医学影像设计，支持多模态输入和多目标预测，提供常用深度学习网络架构，
简化模型训练和预测流程。该框架针对3D医学图像分类场景的特殊需求进行优化，旨在提升科研/实验效率。

#### 软件架构
- 支持单/多模态图像输入（根据扫描场景灵活配置）
- 输入区域图像按指定文件名存储于患者目录，确保框架正确读取
- 可通过修改网络配置快速选择/设计网络架构
- 支持多预测目标配置，提供详细实验记录（损失值、准确率、预测结果）
- 便于快速比较不同网络性能并做出判断

![框架示意图](./images/MediCa.png)

#### 安装指南

```bash
pip install -r requirements.txt
```

#### 数据格式规范: images and annotation
在开始训练或测试前，您需要准备好图像数据和标注文件。数据应按以下方式组织：患者目录位于顶级数据根目录下（分为train_lesion_crops和test_lesion_crops），
每个患者目录中包含所有图像文件及对应的标注文件。
训练/测试前需准备图像数据和标注文件，目录结构如下（以病灶区域为例）：
```commandline
├── train_lesion_crops
│   ├── RJPD_000_lesion0_17.3mm
│   │   ├── annotations.json
│   │   ├── NM.nii.gz
│   │   └── QSM.nii.gz
│   ├── RJPD_001_lesion0_18.0mm
│   │   ├── annotations.json
│   │   ├── NM.nii.gz
│   │   └── QSM.nii.gz
│   ├── RJPD_004_lesion0_16.7mm
│   │   ├── annotations.json
│   │   ├── NM.nii.gz
│   │   └── QSM.nii.gz

├── test_lesion_crops
│   ├── RJPD_003_lesion0_15.3mm
│   │   ├── annotations.json
│   │   ├── NM.nii.gz
│   │   └── QSM.nii.gz
│   ├── RJPD_007_lesion0_16.0mm
│   │   ├── annotations.json
│   │   ├── NM.nii.gz
│   │   └── QSM.nii.gz
│   ├── RJPD_009_lesion0_18.0mm
│   │   ├── annotations.json
│   │   ├── NM.nii.gz
│   │   └── QSM.nii.gz

```

#### annotations.json 文件样例: 
json最顶层为ROI字典，以`id`为key。内容包括：id、tag、预测的所有变量及标注值、输入影像模态列表、各个影像模态的中心坐标、ROI核心大小。若存在多个预测目标，请按照相同命名规范进行定义（例如 label02、label3 等）。在 `nifti_files` 变量中定义不同的模态组合，并保持列表为字符串格式。
`distance` 变量用于定义图像中 ROI 区域（即病灶）的大小，RRMediCa 将自动为每个模态创建对应的掩膜切片。
```commandline
{
  "0": {
    "id": "0",
    "tag": "RJPD",
    "label": "0",
    "nifti_files": "['QSM.nii.gz', 'NM.nii.gz']",
    "phy_loc_xyz": "[[99.7, -100.7, 46.7], [99.3, -100.5, 46.6]]",
    "np_loc_zyx": "[[23, 151, 149], [23, 151, 149]]",
    "distance": "16.7 mm"
  }
}
```
#### 训练设定
输入模态组合调试：修改nifti_files_list参数尝试不同的输入模态组合，可进行多轮对比实验。

关键路径配置：database变量：指定医学影像数据库的存储路径

data_split_file变量：定义训练集/验证集/测试集的病例划分方案

自动处理机制：若未提供预定义的数据划分文件，MediCa将自动生成5折交叉验证的数据划分。

预测目标设置：通过class_names参数定义网络的预测目标类别; 系统将在解析所有患者目录中的annotations.json标注文件后，自动确定预测目标的范围
```set up training path:
    database = '...path/to/train_lesion_crops'
    data_split_file = '...path/to/datasplit_xxx.pkl'
    class_names = ['label', 'label2',... ]
    nifti_files_list = [[...],[...], ...] # different lists of input modalities
```

开始训练:
```Start the training:
python train.py
```

#### 设置测试文件参数并进行测试
在测试阶段，您可以直接在test.py中配置预测输入参数和模型路径：
路径配置：修改database变量指定测试数据集路径

通过ckpts_root变量设置模型检查点的存储位置

模型加载方式：使用find_models函数自动扫描ckpts_root目录下的所有模型，或手动指定需要加载的模型列表

预测集成功能：ensemble函数将对各病例的预测结果进行集成，采用多数表决原则（选择最高频预测值作为最终结果）

自动加载机制：系统将自动加载每个模型的网络配置参数，包括：输入模态组合、网络架构名称、输出层配置
```set up testing path:
    database = '...path/to/train_lesion_crops'
    ckpts_root = '...path/to/ckpts_root'
    model_keys = ['model1...', 'model2...',... ] # specify the models directly
    
```
开始测试:
```Start the testing:
python test.py
```

