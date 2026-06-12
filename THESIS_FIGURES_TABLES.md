# 论文图表目录清单

本文件用于整理毕业论文中建议使用的图与表，包括编号、建议标题、来源路径、用途说明。
写论文或让另一个 AI 写作时，可直接引用本文件中的编号与说明。

---

## 一、表格目录

### 表 1 数据集统计表

- 建议标题：**表 3-1 数据集整体统计信息**
- 建议位置：第三章 3.3 或 3.5 前
- 主要内容：
  - 总图像数
  - 物种类别数
  - 最大类别及样本数
  - 最小类别及样本数
  - 长尾不平衡比例
  - 类别样本均值与中位数
- 数据来源：
  - `outputs/reports/final_model_04/analysis/dataset_summary.json`
- 建议说明：
  - 用于说明数据集存在显著长尾分布，是后续采用 Macro F1 和数据增强策略的重要依据。

### 表 2 训练集/验证集划分表

- 建议标题：**表 3-2 基于 individual_id 的训练集与验证集划分结果**
- 建议位置：第三章 3.3
- 主要内容：
  - 各类别总样本数
  - train_count
  - val_count
  - train_ratio
  - val_ratio
  - 是否尾部类别
- 数据来源：
  - `outputs/reports/final_model_04/analysis/train_val_split_stats.csv`
- 建议说明：
  - 用于说明 Group Split 划分策略的合理性及训练/验证集分布。

### 表 3 消融实验总表

- 建议标题：**表 5-1 不同模型配置下的消融实验结果对比**
- 建议位置：第五章 5.3
- 主要内容：
  - 实验名
  - 模型类型
  - Pooling
  - Focal
  - Mixup
  - Cutout
  - EMA
  - Accuracy
  - Macro F1
- 数据来源：
  - `outputs/ablation_all_results/outputs/ablations/ablation_results.csv`
- 建议说明：
  - 用于证明最终最佳研究模型为 `04_transformer_mean_focal_mixup_cutout`。

### 表 4 各类别性能表

- 建议标题：**表 5-2 最终最佳模型在各类别上的性能指标**
- 建议位置：第五章 5.4
- 主要内容：
  - species
  - support
  - precision
  - recall
  - f1
- 数据来源：
  - `outputs/reports/final_model_04/analysis/per_class_metrics.csv`
- 建议说明：
  - 用于分析头部类别与尾部类别的识别效果差异。

### 表 5 最易混淆类别对统计表

- 建议标题：**表 5-3 最易混淆类别对统计结果**
- 建议位置：第五章 5.4
- 主要内容：
  - true_species
  - pred_species
  - count
  - true_support
  - error_rate_in_true_class
- 数据来源：
  - `outputs/reports/final_model_04/analysis/confusion_top_pairs.csv`
- 建议说明：
  - 用于分析细粒度鲸豚识别中的典型误判类型。

### 表 6 最终模型训练配置表

- 建议标题：**表 5-4 最终最佳模型训练参数配置**
- 建议位置：第五章 5.1
- 主要内容：
  - image_size
  - batch_size
  - epochs
  - lr
  - weight_decay
  - warmup_epochs
  - transformer_dim
  - transformer_depth
  - transformer_heads
  - token_pool_size
  - backbone_stage
- 数据来源：
  - `artifacts/final_model_04/config.json`
- 建议说明：
  - 用于说明实验环境与参数设置。

---

## 二、图片目录

### 图片 1 数据样本展示图

- 建议标题：**图 3-1 部分鲸类数据样本展示**
- 建议位置：第三章 3.4
- 来源方式：
  - 需人工从 `archive/train_images` 中选取 6-9 张不同鲸类图片，自行拼图生成
- 建议说明：
  - 展示不同鲸类物种图像的外观差异以及海洋场景复杂性。

### 图片 2 类别分布柱状图

- 建议标题：**图 3-2 各鲸类物种样本数量分布**
- 建议位置：第三章 3.5
- 图片路径：
  - `outputs/reports/final_model_04/analysis/species_count_bar.png`
- 建议说明：
  - 展示头部类别与尾部类别的样本数量差异。

### 图片 3 长尾分布图

- 建议标题：**图 3-3 数据集长尾分布特征图**
- 建议位置：第三章 3.5
- 图片路径：
  - `outputs/reports/final_model_04/analysis/long_tail_distribution.png`
- 建议说明：
  - 用于突出数据集类别不平衡问题。

### 图片 4 系统总体架构图

- 建议标题：**图 4-1 鲸类识别系统总体架构图**
- 建议位置：第四章 4.3 或 4.1 后
- 来源方式：
  - 需人工使用 ProcessOn / draw.io 绘制
- 建议说明：
  - 展示数据输入、模型推理、结果输出、Web 展示的整体流程。

### 图片 5 ResNet50-Transformer 模型架构图

- 建议标题：**图 4-2 ResNet50-Transformer 模型结构图**
- 建议位置：第四章 4.3
- 来源方式：
  - 需人工绘制
- 建议说明：
  - 展示 CNN 特征提取、Token 化、Transformer 编码、多尺度融合与分类头结构。

### 图片 6 消融实验 Macro F1 柱状图

- 建议标题：**图 5-1 各消融实验配置的 Macro F1 对比**
- 建议位置：第五章 5.3
- 图片路径：
  - `outputs/ablation_all_results/outputs/reports/ablation/ablation_macro_f1_20260421-092959.png`
- 建议说明：
  - 直观比较不同模型配置在长尾指标上的优劣。

### 图片 7 各类别 F1 柱状图

- 建议标题：**图 5-2 最终最佳模型在各类别上的 F1 分布**
- 建议位置：第五章 5.4
- 图片路径：
  - `outputs/reports/final_model_04/analysis/per_class_f1_bar.png`
- 建议说明：
  - 展示不同鲸类物种的识别难易程度。

### 图片 8 最易混淆类别对图

- 建议标题：**图 5-3 最易混淆类别对统计图**
- 建议位置：第五章 5.4
- 图片路径：
  - `outputs/reports/final_model_04/analysis/confusion_top_pairs.png`
- 建议说明：
  - 用于说明细粒度分类中的典型误判现象。

### 图片 9 混淆矩阵

- 建议标题：**图 5-4 最终最佳模型的混淆矩阵**
- 建议位置：第五章 5.4
- 图片路径：
  - `outputs/reports/final_model_04/evaluation/confusion_matrix.png`
- 建议说明：
  - 综合展示各类别的正确识别与相互混淆情况。

### 图片 10 Grad-CAM 激活图

- 建议标题：**图 5-5 基于 Grad-CAM 的模型关注区域可视化**
- 建议位置：第五章 5.5
- 图片路径：
  - `outputs/reports/final_model_04/interpretability/gradcam.jpg`
- 建议说明：
  - 分析 CNN 对局部关键区域的关注情况。

### 图片 11 Attention Map 注意力图

- 建议标题：**图 5-6 Transformer 注意力可视化结果**
- 建议位置：第五章 5.5
- 图片路径：
  - `outputs/reports/final_model_04/interpretability/attention_map.jpg`
- 建议说明：
  - 展示 Transformer 对全局空间关系的建模能力。

### 图片 12 系统首页截图

- 建议标题：**图 6-1 系统首页与图片上传界面**
- 建议位置：第六章 6.4
- 来源方式：
  - 需人工本地运行 Web 后截图
- 建议说明：
  - 展示系统入口页面与上传交互。

### 图片 13 识别结果页截图

- 建议标题：**图 6-2 鲸类识别结果展示界面**
- 建议位置：第六章 6.4
- 来源方式：
  - 需人工截图
- 建议说明：
  - 展示模型返回的 Top-1 结果、置信度与预测说明。

### 图片 14 Top-3 概率图表截图

- 建议标题：**图 6-3 Top-3 物种概率可视化结果**
- 建议位置：第六章 6.4
- 来源方式：
  - 需人工截图
- 建议说明：
  - 展示前端柱状图或置信度分布。

### 图片 15 /health 接口截图

- 建议标题：**图 6-4 系统健康检查接口返回结果**
- 建议位置：第六章 6.4
- 来源方式：
  - 需人工浏览器访问 `/health` 后截图
- 建议说明：
  - 展示系统当前加载模型、版本信息和运行状态。

---

## 三、写作时的使用建议

1. 第三章优先使用图片 1-3、表 1-2。
2. 第五章优先使用图片 6-11、表 3-6。
3. 第六章优先使用图片 12-15。
4. 如果篇幅有限，至少保留：
   - 表 3（消融实验总表）
   - 图 3（长尾分布图）
   - 图 6（消融 Macro F1 图）
   - 图 9（混淆矩阵）
   - 图 10/11（可解释性分析图）
