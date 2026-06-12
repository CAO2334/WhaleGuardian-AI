# AI护鲸使者 论文写作资料包

本文件用于把当前项目中与毕业论文写作相关的核心信息、关键实验结论、可直接引用的数据文件、CSV、JSON、图片路径整理成一份统一材料，便于交给另一个 AI 或人工写作者直接使用。

适用对象：

- 毕业论文写作
- 大创结题报告写作
- 比赛项目设计说明书写作

---

## 1. 项目基本信息

- 项目名称：AI护鲸使者
- 任务类型：海洋鲸豚类图像物种分类（`species` 分类，不是 `individual_id` 个体重识别）
- 数据集：Kaggle `Happywhale - Whale and Dolphin Identification`
- 图像版本：预处理宽度 512 像素版本
- 最终采用的数据划分方式：`Group Split by individual_id`
- 最终最佳研究模型：`04_transformer_mean_focal_mixup_cutout`
- 最佳轻量/高 Accuracy 基线模型：`01_resnet50_ce_plain`
- 当前 Web 默认部署模型：`artifacts/final_model_04`

---

## 2. 可直接写入论文的核心结论

### 2.1 数据集结论

- 总图像数：51033
- 物种类别数：28
- 最大类别：`bottlenose_dolphin`，10781 张
- 最小类别：`frasiers_dolphin`，14 张
- 类别最大最小样本比：770.07x
- 类别样本中位数：459
- 数据明显存在长尾分布

### 2.2 训练/验证划分结论

- 训练集样本数：40639
- 验证集样本数：10394
- 划分方式：按 `individual_id` 分组划分
- 目的：防止同一鲸鱼个体同时出现在训练集和验证集，降低数据泄漏风险

### 2.3 最终实验结论

- 如果以 `Accuracy` 为主，最佳模型是 `01_resnet50_ce_plain`
  - `best_val_acc = 0.9815`
  - `best_val_macro_f1 = 0.9106`
- 如果以长尾分类更关键的 `Macro F1` 为主，最佳研究模型是 `04_transformer_mean_focal_mixup_cutout`
  - `best_val_acc = 0.9794`
  - `best_val_macro_f1 = 0.9199`
- `CLS Token` 与 `EMA` 在当前任务设定下没有带来额外收益，反而弱于 `mean pooling` 版本

建议论文中的最终模型结论写法：

> 在鲸类物种细粒度分类任务中，ResNet50 baseline 已经具有较高的整体识别准确率；但在更能反映长尾类别识别能力的 Macro F1 指标上，采用多尺度特征融合、Token Pooling 和 mean pooling 的 ResNet50-Transformer 模型表现最优。因此本文将 `04_transformer_mean_focal_mixup_cutout` 作为最终最佳研究模型，而将 `01_resnet50_ce_plain` 作为高效率部署基线模型。

---

## 3. 最终最佳研究模型信息

最终最佳研究模型：

```text
experiment_name: 04_transformer_mean_focal_mixup_cutout
model_type: transformer
pooling: mean
focal: true
mixup: true
cutout: true
transformer: true
cls_token: false
ema: false
backbone_stage: layer3_layer4
token_pool_size: 16
best_epoch: 19
best_val_acc: 0.9794111988
best_val_macro_f1: 0.9199286851
num_classes: 28
train_size: 40639
val_size: 10394
split_strategy: group
group_col: individual_id
```

可直接引用的指标文件：

- `outputs/reports/final_model_04/metrics.json`
- `artifacts/final_model_04/metrics.json`

模型配置文件：

- `artifacts/final_model_04/config.json`

ONNX 部署文件：

- `artifacts/final_model_04/model.onnx`

---

## 4. 消融实验总表

数据来源文件：

- `outputs/ablation_all_results/outputs/ablations/ablation_results.csv`

实验结果：

| 实验名 | 模型 | Pooling | Focal | Mixup | Cutout | EMA | Acc | Macro F1 |
|---|---|---|---|---|---|---|---:|---:|
| 01_resnet50_ce_plain | ResNet50 baseline | gap | 否 | 否 | 否 | 否 | 0.9815 | 0.9106 |
| 02_resnet50_focal_plain | ResNet50 baseline | gap | 是 | 否 | 否 | 否 | 0.9787 | 0.9058 |
| 03_resnet50_focal_mixup_cutout_ema | ResNet50 baseline | gap | 是 | 是 | 是 | 是 | 0.9769 | 0.9048 |
| 04_transformer_mean_focal_mixup_cutout | ResNet50-Transformer | mean | 是 | 是 | 是 | 否 | 0.9794 | **0.9199** |
| 05_transformer_cls_focal_mixup_cutout | ResNet50-Transformer | cls | 是 | 是 | 是 | 否 | 0.9778 | 0.9072 |
| 06_transformer_cls_focal_mixup_cutout_ema | ResNet50-Transformer | cls | 是 | 是 | 是 | 是 | 0.9780 | 0.9022 |

可直接写入论文的分析结论：

1. ResNet50 baseline 在该 `species` 分类任务上已经很强，说明数据集在物种层面可分性较高。
2. 单独引入 Focal Loss 并未优于 CE baseline。
3. 在当前任务下，`mean pooling` 比 `CLS Token` 更适合 Transformer 输出聚合。
4. `EMA` 在该任务配置下没有带来稳定增益。
5. 因此最终最佳研究模型并不是最初设想的 `CLS + EMA` 版本，而是 `04_transformer_mean_focal_mixup_cutout`。

---

## 5. 数据集统计与长尾信息

数据来源：

- `outputs/reports/final_model_04/analysis/dataset_summary.json`
- `outputs/reports/final_model_04/analysis/species_counts.csv`
- `outputs/reports/final_model_04/analysis/train_val_split_stats.csv`
- `outputs/reports/final_model_04/analysis/split_leakage_report.json`

可直接引用的统计值：

```text
num_images = 51033
num_species = 28
max_class = bottlenose_dolphin (10781)
min_class = frasiers_dolphin (14)
imbalance_ratio_max_to_min = 770.07
median_class_count = 459
mean_class_count = 1822.61
```

Top 10 高频类别：

```text
bottlenose_dolphin: 10781
beluga: 7443
humpback_whale: 7392
blue_whale: 4830
false_killer_whale: 3326
dusky_dolphin: 3139
killer_whale: 2455
spinner_dolphin: 1700
melon_headed_whale: 1689
minke_whale: 1608
```

Tail 10 低频类别：

```text
pilot_whale: 262
long_finned_pilot_whale: 238
white_sided_dolphin: 229
brydes_whale: 154
pantropic_spotted_dolphin: 145
globis: 116
commersons_dolphin: 90
pygmy_killer_whale: 76
rough_toothed_dolphin: 60
frasiers_dolphin: 14
```

可直接写入论文的解释：

> 数据集呈现典型长尾分布，头部类别样本数远高于尾部类别。最大类别与最小类别之间的样本规模差异达到 770 倍以上，这会导致模型训练过程中高频类别对总损失的主导作用增强，进而掩盖稀有类别的识别需求。因此本文在模型评估中引入 Macro F1，并在训练中采用 Mixup、Cutout 及 Transformer 全局建模进行对比实验。

---

## 6. 训练/验证集划分信息

数据来源：

- `outputs/reports/final_model_04/analysis/train_val_split_stats.csv`
- `outputs/reports/final_model_04/analysis/split_leakage_report.json`

关键结论：

- 训练集：40639
- 验证集：10394
- `individual_id` 交叉重叠：0
- 使用 `Group Split by individual_id`

可直接写入论文的解释：

> 为避免同一鲸鱼个体的不同图像同时出现在训练集和验证集中，本文未采用简单随机划分或仅按物种分层划分，而是基于 `individual_id` 进行分组划分。这种方式能有效降低个体级信息泄漏，使验证结果更能反映模型对新个体的泛化能力。

---

## 7. 各类别表现与错误分析

数据来源：

- `outputs/reports/final_model_04/analysis/per_class_metrics.csv`
- `outputs/reports/final_model_04/analysis/confusion_top_pairs.csv`

### 7.1 表现较好的类别示例

```text
beluga: F1 = 0.9983
blue_whale: F1 = 0.9979
dusky_dolphin: F1 = 0.9976
sei_whale: F1 = 0.9942
white_sided_dolphin: F1 = 1.0000
commersons_dolphin: F1 = 1.0000
```

### 7.2 表现较弱的类别示例

```text
rough_toothed_dolphin: F1 = 0.5000
pygmy_killer_whale: F1 = 0.6452
globis: F1 = 0.7143
pantropic_spotted_dolphin: F1 = 0.7500
common_dolphin: F1 = 0.8588
```

### 7.3 主要混淆类别对

```text
bottlenose_dolphin -> minke_whale: 12
common_dolphin -> spinner_dolphin: 10
humpback_whale -> gray_whale: 10
common_dolphin -> bottlenose_dolphin: 6
rough_toothed_dolphin -> pygmy_killer_whale: 3
globis -> pilot_whale: 3
```

可直接写入论文的解释：

> 从类别级别指标可以看出，大多数头部类别已经取得较高 F1 值，但部分类间外观相近、样本较少的物种仍存在明显混淆。尤其是 `common_dolphin` 与 `spinner_dolphin`、`rough_toothed_dolphin` 与 `pygmy_killer_whale` 等类别对，其误判反映了细粒度鲸豚识别中局部纹理相似和尾部样本不足的双重挑战。

---

## 8. 可直接引用的 CSV / JSON / 图片清单

### 8.1 论文中需要重点引用的 CSV / JSON

#### 数据集与划分

- `outputs/reports/final_model_04/analysis/dataset_summary.json`
- `outputs/reports/final_model_04/analysis/species_counts.csv`
- `outputs/reports/final_model_04/analysis/train_val_split_stats.csv`
- `outputs/reports/final_model_04/analysis/split_leakage_report.json`

#### 最终模型结果

- `outputs/reports/final_model_04/metrics.json`
- `artifacts/final_model_04/config.json`
- `artifacts/final_model_04/metrics.json`
- `artifacts/final_model_04/manifest.json`

#### 消融实验

- `outputs/ablation_all_results/outputs/ablations/ablation_results.csv`

#### 类别级指标

- `outputs/reports/final_model_04/analysis/per_class_metrics.csv`
- `outputs/reports/final_model_04/analysis/confusion_top_pairs.csv`

### 8.2 论文中建议插入的图片

#### 第三章：数据预处理与数据分析

- `outputs/reports/final_model_04/analysis/species_count_bar.png`
- `outputs/reports/final_model_04/analysis/long_tail_distribution.png`

#### 第五章：实验结果与分析

- `outputs/ablation_all_results/outputs/reports/ablation/ablation_macro_f1_20260421-092959.png`
- `outputs/reports/final_model_04/analysis/per_class_f1_bar.png`
- `outputs/reports/final_model_04/analysis/confusion_top_pairs.png`
- `outputs/reports/final_model_04/evaluation/confusion_matrix.png`

#### 第五章或第六章：可解释性分析

- `outputs/reports/final_model_04/interpretability/gradcam.jpg`
- `outputs/reports/final_model_04/interpretability/attention_map.jpg`

#### 第六章：系统展示

以下图片需要人工补充截图，不在当前自动生成产物中：

- Web 首页截图
- 图片上传页截图
- 识别结果页截图
- `/health` 接口返回截图

---

## 9. 各章节写作时应该使用哪些材料

### 第 1 章 绪论

建议使用：

- 项目背景与意义（来自申请书与 README）
- 任务目标：鲸类 `species` 分类
- 项目定位：兼顾研究与系统实现

### 第 2 章 相关理论与关键技术

建议使用：

- ResNet50
- Transformer / Multi-Head Self-Attention
- Focal Loss
- Mixup / Cutout
- Macro F1
- ONNX Runtime / Flask

### 第 3 章 数据集构建与预处理

必须使用：

- `dataset_summary.json`
- `species_counts.csv`
- `train_val_split_stats.csv`
- `split_leakage_report.json`
- `species_count_bar.png`
- `long_tail_distribution.png`

### 第 4 章 模型设计

建议使用：

- `artifacts/final_model_04/config.json`
- README 中关于模型结构的描述
- 自行绘制模型结构图 / 技术路线图

### 第 5 章 实验设计与结果分析

必须使用：

- `ablation_results.csv`
- `metrics.json`
- `per_class_metrics.csv`
- `confusion_top_pairs.csv`
- `ablation_macro_f1_20260421-092959.png`
- `per_class_f1_bar.png`
- `confusion_top_pairs.png`
- `confusion_matrix.png`
- `gradcam.jpg`
- `attention_map.jpg`

### 第 6 章 系统设计与实现

建议使用：

- `artifacts/final_model_04/model.onnx`
- `artifacts/final_model_04/manifest.json`
- `artifacts/final_model_04/config.json`
- Web 运行截图
- `/health` 返回截图

### 第 7 章 总结与展望

建议使用：

- `metrics.json`
- `ablation_results.csv`
- `per_class_metrics.csv`

---

## 10. 写作者需要特别注意的事实

1. 本项目当前完成的是 `species` 物种分类，不是 `individual_id` 个体识别。
2. 最终最佳研究模型是 `04_transformer_mean_focal_mixup_cutout`，不是最初设想的 `CLS + EMA` 版本。
3. 如果论文强调“部署效率”或“最高 Accuracy”，可以同时提到 `01_resnet50_ce_plain`。
4. 当前 Web 默认部署模型已经切换为 `artifacts/final_model_04`。
5. Kaggle `test_images` 没有物种真值，不能作为独立测试集直接计算 Macro F1。
6. 论文若写“创新点”，应强调：
   - Group Split by `individual_id`
   - 长尾问题驱动的 Macro F1 评价
   - ResNet50 与 ResNet50-Transformer 的系统消融
   - 可解释性分析与 ONNX 部署一体化

---

## 11. 当前本地已确认存在的关键输出

### 报告目录

- `outputs/reports/final_model_04/best_model.pth`
- `outputs/reports/final_model_04/class_to_idx.json`
- `outputs/reports/final_model_04/metrics.json`
- `outputs/reports/final_model_04/analysis/confusion_top_pairs.csv`
- `outputs/reports/final_model_04/analysis/confusion_top_pairs.png`
- `outputs/reports/final_model_04/analysis/dataset_summary.json`
- `outputs/reports/final_model_04/analysis/long_tail_distribution.png`
- `outputs/reports/final_model_04/analysis/per_class_f1_bar.png`
- `outputs/reports/final_model_04/analysis/per_class_metrics.csv`
- `outputs/reports/final_model_04/analysis/species_counts.csv`
- `outputs/reports/final_model_04/analysis/species_count_bar.png`
- `outputs/reports/final_model_04/analysis/split_leakage_report.json`
- `outputs/reports/final_model_04/analysis/train_val_split_stats.csv`
- `outputs/reports/final_model_04/evaluation/confusion_matrix.png`
- `outputs/reports/final_model_04/interpretability/attention_map.jpg`
- `outputs/reports/final_model_04/interpretability/gradcam.jpg`

### Artifact 目录

- `artifacts/final_model_04/class_to_idx.json`
- `artifacts/final_model_04/config.json`
- `artifacts/final_model_04/manifest.json`
- `artifacts/final_model_04/metrics.json`
- `artifacts/final_model_04/model.onnx`

### 消融实验目录

- `outputs/ablation_all_results/outputs/ablations/ablation_results.csv`
- `outputs/ablation_all_results/outputs/reports/ablation/ablation_report_20260421-092959.md`
- `outputs/ablation_all_results/outputs/reports/ablation/ablation_macro_f1_20260421-092959.png`

---

## 12. 最推荐的论文主线

可直接作为论文逻辑主线：

> 针对海洋鲸类图像识别中复杂海况、类别长尾分布和个体泄漏风险等问题，本文基于 Happywhale 数据集构建鲸类物种细粒度识别任务，采用 `individual_id` 分组划分策略防止验证泄漏，并系统比较 ResNet50 baseline 与 ResNet50-Transformer 混合模型。通过消融实验发现，采用多尺度特征融合、Token Pooling 和 mean pooling 的 Transformer 模型在 Macro F1 指标上取得最优结果；在此基础上，进一步完成混淆矩阵、Grad-CAM、Attention Map 可解释性分析，以及 ONNX Runtime + Flask 的可部署识别系统实现。

---

## 13. 毕业论文完整写作框架（可直接交给另一个 AI）

本节给出一套可直接用于毕业论文生成的章节框架。可以把整节内容整体交给另一个 AI，也可以单独抽取其中任意章节，要求其展开撰写。

### 13.1 论文题目

推荐题目：

```text
基于 ResNet50-Transformer 的鲸类物种细粒度识别与可视化系统设计与实现
```

备选题目：

```text
面向海洋生态监测的鲸类图像智能识别系统设计与实现
```

### 13.2 中文摘要写作框架

中文摘要应至少包含以下四部分：

1. 研究背景与目标
   鲸类是海洋生态的关键指示物种，传统人工调查效率低、成本高。本项目旨在利用深度学习技术，解决复杂海况下的鲸类细粒度图像识别难题。

2. 方法概述
   基于 Happywhale 鲸类数据集，提出一种融合局部特征与全局语义的双流混合模型（ResNet50-Transformer）。针对长尾分布与类别不平衡问题，引入 Group Split 划分策略、Focal Loss 以及 Mixup、Cutout 等数据增强方法。

3. 实验结果
   通过多组消融实验，最终确立最佳模型为 `04_transformer_mean_focal_mixup_cutout`，其在 Macro F1 和 Accuracy 等关键指标上表现优异。

4. 系统实现
   将训练好的模型导出为 ONNX 格式，并结合 Flask Web 后端与前端可视化页面，部署开发了一套集图片上传、智能推理、可视化展示为一体的 Web 演示系统。

关键词建议：

```text
鲸类识别；细粒度图像分类；ResNet50；Transformer；长尾分布；模型部署
```

### 13.3 英文摘要写作框架

英文摘要应与中文摘要一一对应翻译，建议包含：

- Research background and objective
- Method overview
- Experimental results
- System implementation
- Keywords

可直接要求另一个 AI：

```text
请将中文摘要严格逐句翻译为英文摘要，保持学术表达风格一致，不新增结论，不删减技术细节。
```

---

## 14. 分章节详细写作提纲

### 第一章 绪论

#### 1.1 研究背景

- 鲸类在海洋生态系统中的核心作用及保护意义
- 传统人工观测与声学监测方式的局限性（效率低、成本高）
- 深度学习与计算机视觉在海洋生态自动监测中的应用前景

#### 1.2 研究意义

- 理论意义：探索 CNN 与 Transformer 混合架构在海洋生物细粒度图像分类中的应用
- 实践意义：提升野外鲸类调查效率和准确率，辅助海洋保护区管理与科研统计

#### 1.3 国内外研究现状

- 生物图像智能识别与细粒度分类研究进展
- CNN 与 Transformer 架构在图像分类中的优劣势与融合趋势
- 海洋生物识别面临的挑战：海面反光、残缺遮挡、相似物种难区分等

#### 1.4 本文研究内容

- 数据集构建、清洗与预处理策略
- ResNet50-Transformer 模型设计、损失函数优化与对比训练
- ONNX 部署与前后端 Web 系统开发

#### 1.5 本文结构安排

简述第二章至第七章的内容安排。

---

### 第二章 相关理论与关键技术

#### 2.1 卷积神经网络基础

- CNN 基本原理与特征提取机制
- ResNet50 残差结构（Residual Block）
- ResNet50 在避免梯度消失和增强深层特征提取方面的优势

#### 2.2 Transformer 与自注意力机制

- Self-Attention 与 Multi-Head Self-Attention 数学原理
- 图像块 Token 化
- Positional Encoding 在视觉任务中的作用

#### 2.3 数据增强技术

- 基础增强：随机旋转、水平翻转、亮度/对比度调整
- 高级增强：Mixup、Cutout
- 这些增强如何缓解过拟合并提升鲁棒性

#### 2.4 长尾分类问题与评价指标

- 长尾分布与类别不平衡问题
- Focal Loss 的核心思想
- 为什么本项目重点关注 Macro F1，而不只看 Accuracy

#### 2.5 模型部署与 Web 开发技术

- ONNX 与 ONNX Runtime 的跨平台推理优势
- Flask/Django 后端与 RESTful API
- 前端页面与模型推理结果展示逻辑

---

### 第三章 数据集构建与预处理

#### 3.1 数据来源与数据结构

- Happywhale 数据集介绍
- 字段说明：`image`、`species`、`individual_id`
- 明确本文任务为 `species` 物种分类

#### 3.2 数据清洗与类别整理

- 拼写噪声修正
- 异常数据处理
- 最终保留类别数量与样本规模

#### 3.3 数据集划分策略

- 为什么不能简单随机划分
- Group Split by `individual_id`
- 训练集/验证集划分结果

**建议插入表格 1：数据集统计表**
数据来源：`outputs/reports/final_model_04/analysis/dataset_summary.json`

**建议插入表格 2：训练/验证集划分表**
数据来源：`outputs/reports/final_model_04/analysis/train_val_split_stats.csv`

#### 3.4 数据增强与预处理方法

- Resize（统一输入尺寸）
- Normalize
- 随机旋转、翻转、亮度对比度调整
- Mixup 与 Cutout 的使用方式

**建议插入图片 1：鲸类数据样本展示图**
来源建议：从 `archive/train_images` 中挑选 6-9 张不同鲸类样本，拼成一张图。

#### 3.5 数据可视化分析

- 类别分布分析
- 长尾分布分析
- 泄漏检测结果

**建议插入图片 2：类别分布柱状图**
图片路径：`outputs/reports/final_model_04/analysis/species_count_bar.png`

**建议插入图片 3：长尾分布图**
图片路径：`outputs/reports/final_model_04/analysis/long_tail_distribution.png`

#### 3.6 本章小结

---

### 第四章 鲸类识别模型设计

#### 4.1 系统任务分析与难点分析

- 海面反光
- 鲸身遮挡，仅露出背鳍或尾鳍
- 相似物种间差异小
- 长尾数据导致模型偏向头部类别

#### 4.2 ResNet50 baseline 模型设计

- 作为基线模型的结构与参数设置
- baseline 的实验意义

#### 4.3 ResNet50-Transformer 混合模型设计

- 串行混合架构总体设计
- ResNet50 提取局部纹理特征
- Transformer 编码全局空间关联
- Mean Pooling / CLS Token 两种聚合方式
- 分类头输出机制

**建议插入图片 4：系统总体架构图**
来源建议：使用 ProcessOn 或 draw.io 绘制系统功能流转图。

**建议插入图片 5：ResNet50-Transformer 模型架构图**
来源建议：绘制特征图转 Token，再输入 Transformer 的网络结构图。

#### 4.4 多尺度特征融合设计

- 选择 `layer3 + layer4` 的原因
- Token Pooling 的作用
- 如何平衡精度与显存消耗

#### 4.5 损失函数与优化策略

- AdamW 优化器
- Warmup + CosineAnnealingLR 学习率策略
- Focal Loss
- EMA 的尝试与实验结果

#### 4.6 本章小结

---

### 第五章 实验设计与结果分析

#### 5.1 实验环境与参数设置

- 硬件资源（GPU 服务器）
- 软件环境（Python、PyTorch、CUDA）
- Batch Size、Epoch、输入尺寸等超参数
- 配置参考：`artifacts/final_model_04/config.json`

#### 5.2 评价指标

- Accuracy
- Macro F1
- 混淆矩阵

#### 5.3 对比实验与消融实验结果

- ResNet50 baseline
- 加入 Focal 的 baseline
- 加入 Mixup/Cutout/EMA 的 baseline
- Transformer + mean
- Transformer + cls
- Transformer + cls + EMA

**建议插入表格 3：消融实验总表**
数据来源：`outputs/ablation_all_results/outputs/ablations/ablation_results.csv`

**建议插入图片 6：消融实验 Macro F1 柱状图**
图片路径：`outputs/ablation_all_results/outputs/reports/ablation/ablation_macro_f1_20260421-092959.png`

分析重点：

- 为什么 `04_transformer_mean_focal_mixup_cutout` 综合表现最好
- 为什么 `CLS Token` 和 `EMA` 没有带来预期增益

#### 5.4 类别级别结果分析

**建议插入表格 4：各类别性能表**
数据来源：`outputs/reports/final_model_04/analysis/per_class_metrics.csv`

**建议插入图片 7：各类别 F1 柱状图**
图片路径：`outputs/reports/final_model_04/analysis/per_class_f1_bar.png`

**建议插入图片 8：最易混淆类别对**
图片路径：`outputs/reports/final_model_04/analysis/confusion_top_pairs.png`

**建议插入图片 9：混淆矩阵**
图片路径：`outputs/reports/final_model_04/evaluation/confusion_matrix.png`

分析重点：

- 长尾类别表现较弱的原因
- 相似物种容易混淆的原因

#### 5.5 可解释性分析

**建议插入图片 10：Grad-CAM 激活图**
图片路径：`outputs/reports/final_model_04/interpretability/gradcam.jpg`

**建议插入图片 11：Attention Map 注意力图**
图片路径：`outputs/reports/final_model_04/interpretability/attention_map.jpg`

分析重点：

- 模型是否关注了背鳍、尾鳍、水柱等关键区域
- Transformer 是否体现了全局关联建模能力

#### 5.6 本章小结

---

### 第六章 鲸类识别系统设计与实现

#### 6.1 系统需求与总体架构

- 用户上传图片
- 模型推理
- Top-3 概率返回
- 低置信度提示
- 前后端系统整体架构

#### 6.2 模型 ONNX 部署实现

- `model.onnx`
- `manifest.json`
- ONNX Runtime 推理流程

#### 6.3 后端核心 API 设计

- `/predict`
- `/health`
- 文件校验
- RESTful JSON 返回

#### 6.4 Web 前端页面展示与测试

**建议插入图片 12-15：系统运行截图组**

建议包含：

- 系统首页 / 图片上传页
- 识别结果页
- Top-3 物种概率柱状图页
- `/health` 接口返回页

#### 6.5 本章小结

---

### 第七章 总结与展望

#### 7.1 全文总结

- 完成了数据分析、模型训练、消融实验、可解释性分析和系统部署
- 最终优选模型为 `04_transformer_mean_focal_mixup_cutout`
- 成功实现可部署的 Web 识别系统

#### 7.2 创新点归纳

- Group Split 防止数据泄漏
- CNN + Transformer 局部与全局特征融合
- 从训练、分析、可解释性到部署的闭环实现

#### 7.3 不足与后续展望

- 当前仅完成 `species` 分类，尚未扩展到 `individual_id`
- 后续可加入 YOLO 等检测模块剥离复杂背景
- 可加入 OOD 检测与多模态数据融合

---

## 15. 参考文献建议

建议至少包含以下经典文献：

1. He K, Zhang X, Ren S, et al. Deep Residual Learning for Image Recognition[C]. CVPR, 2016.
2. Dosovitskiy A, Beyer L, Kolesnikov A, et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale[J]. ICLR, 2021.
3. Lin T Y, Goyal P, Girshick R, et al. Focal Loss for Dense Object Detection[C]. ICCV, 2017.
4. Zhang H, Cisse M, Dauphin Y N, et al. mixup: Beyond Empirical Risk Minimization[J]. ICLR, 2018.
5. DeVries T, Taylor G W. Improved Regularization of Convolutional Neural Networks with Cutout[J]. arXiv preprint, 2017.
6. Kaggle. Happywhale - Whale and Dolphin Identification Dataset.

---

## 16. 附录建议

### 附录 A

主要训练参数与环境依赖清单
参考：`artifacts/final_model_04/config.json`

### 附录 B

后端 API 接口返回数据结构示例
参考：`/predict`、`/health`

### 附录 C

系统核心项目目录结构说明
参考：项目根目录、`README.md`

---

## 17. 可直接交给另一个 AI 的使用方式

如果你只想让另一个 AI 写某一部分，可以直接给它类似提示：

```text
请严格依据 THESIS_WRITING_PACKET.md 中的“第五章 实验设计与结果分析”框架写作，不要虚构实验结果，只能使用文档中提供的数值、CSV、JSON 和图片路径信息，写成学术论文风格正文。
```

或者：

```text
请依据 THESIS_WRITING_PACKET.md 中的“中文摘要写作框架”和“最终最佳研究模型信息”撰写中文摘要，要求语言正式、简洁、符合本科毕业论文摘要风格。
```
