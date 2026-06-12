# 论文用 Mermaid 图示代码

本文件整理毕业论文中常用的 Mermaid 图示代码，便于直接复制使用或作为 ProcessOn / draw.io 绘图参考。

建议用途：

- 技术路线图
- 系统架构图
- 模型架构图
- 训练与部署流程图

---

## 1. 技术路线图

```mermaid
flowchart TD
    A[Happywhale 数据集获取] --> B[数据清洗与类别整理]
    B --> C[数据预处理与增强]
    C --> C1[Resize / Normalize]
    C --> C2[随机旋转 / 水平翻转]
    C --> C3[亮度与对比度增强]
    C --> C4[Mixup / Cutout]

    C1 --> D[按 individual_id 进行 Group Split]
    C2 --> D
    C3 --> D
    C4 --> D

    D --> E[构建 ResNet50 baseline]
    D --> F[构建 ResNet50-Transformer 模型]

    E --> G[模型训练与验证]
    F --> G

    G --> H[对比实验与消融实验]
    H --> I[评价指标分析]
    I --> I1[Accuracy]
    I --> I2[Macro F1]
    I --> I3[混淆矩阵]
    I --> I4[per-class F1]

    I1 --> J[最优模型选择]
    I2 --> J
    I3 --> J
    I4 --> J

    J --> K[可解释性分析]
    K --> K1[Grad-CAM]
    K --> K2[Attention Map]

    K1 --> L[ONNX 导出与部署]
    K2 --> L

    L --> M[Flask + ONNX Runtime Web 演示系统]
    M --> N[系统测试与论文撰写]
```

---

## 2. 系统总体架构图

```mermaid
flowchart LR
    U[用户] --> W[Web 前端页面]

    subgraph Frontend[前端展示层]
        W1[图片上传]
        W2[结果展示]
        W3[Top-3 概率图表]
        W4[低置信度提示]
    end

    W --> W1
    W --> W2
    W --> W3
    W --> W4

    W --> A[Flask 后端服务]

    subgraph Backend[后端服务层]
        A1[/predict 接口/]
        A2[/health 接口/]
        A3[文件校验]
        A4[统一 RESTful 返回]
    end

    A --> A1
    A --> A2
    A --> A3
    A --> A4

    A1 --> P[ONNX Runtime 推理模块]

    subgraph Inference[模型推理层]
        P1[图像预处理]
        P2[model.onnx]
        P3[class_to_idx.json]
        P4[Top-3 概率计算]
    end

    P --> P1
    P --> P2
    P --> P3
    P --> P4

    P4 --> A4
    A4 --> W2
    A4 --> W3
    A4 --> W4

    subgraph Artifact[模型版本管理]
        T1[config.json]
        T2[metrics.json]
        T3[manifest.json]
    end

    P2 --> T1
    P2 --> T2
    P2 --> T3
```

---

## 3. ResNet50-Transformer 模型架构图

```mermaid
flowchart TD
    A[输入图像 3x512x512] --> B[ResNet50 Stem + Layer1 + Layer2]
    B --> C[Layer3 特征图]
    C --> D[1x1 Conv 通道映射]
    D --> E[Token Pooling 16x16]
    E --> F[Flatten 为 Token 序列]
    F --> G[加入位置编码]
    G --> H[Transformer Encoder]
    H --> I[Mean Pooling]

    C --> J[Layer4 高层语义特征]
    J --> K[Global Average Pooling]
    K --> L[语义特征投影]

    I --> M[特征拼接 Concat]
    L --> M
    M --> N[分类头 Linear]
    N --> O[物种分类结果]
```

---

## 4. ResNet50 baseline 对比模型图

```mermaid
flowchart TD
    A[输入图像] --> B[ResNet50 Backbone]
    B --> C[Global Average Pooling]
    C --> D[Fully Connected Classifier]
    D --> E[物种分类结果]
```

---

## 5. 训练与消融实验流程图

```mermaid
flowchart TD
    A[数据集与划分] --> B[构建实验配置]

    B --> B1[01 ResNet50 + CE]
    B --> B2[02 ResNet50 + Focal]
    B --> B3[03 ResNet50 + Focal + Mixup/Cutout/EMA]
    B --> B4[04 Transformer + mean]
    B --> B5[05 Transformer + cls]
    B --> B6[06 Transformer + cls + EMA]

    B1 --> C[统一训练流程]
    B2 --> C
    B3 --> C
    B4 --> C
    B5 --> C
    B6 --> C

    C --> D[验证集评估]
    D --> E[记录 Accuracy / Macro F1]
    E --> F[生成消融实验总表]
    F --> G[选择最佳研究模型]
```

---

## 6. ONNX 部署流程图

```mermaid
flowchart TD
    A[最优 PyTorch 模型 checkpoint] --> B[导出 ONNX 模型]
    B --> C[生成 artifact 目录]
    C --> C1[model.onnx]
    C --> C2[class_to_idx.json]
    C --> C3[config.json]
    C --> C4[metrics.json]
    C --> C5[manifest.json]

    C1 --> D[ONNX Runtime 推理]
    C2 --> D
    C3 --> D
    C4 --> D
    C5 --> D

    D --> E[Flask Web 服务接入]
    E --> F[用户上传图片]
    F --> G[Top-1/Top-3 分类结果展示]
```

---

## 7. 可解释性分析流程图

```mermaid
flowchart TD
    A[输入测试图片] --> B[最优模型推理]
    B --> C[预测类别输出]
    C --> D[Grad-CAM]
    C --> E[Attention Map]
    D --> F[局部关注区域分析]
    E --> G[全局空间关联分析]
    F --> H[模型可解释性结论]
    G --> H
```

---

## 8. 写论文时的使用建议

1. 第三章可使用“技术路线图”与“训练与消融实验流程图”。
2. 第四章可使用“ResNet50-Transformer 模型架构图”与“ResNet50 baseline 对比模型图”。
3. 第六章可使用“系统总体架构图”与“ONNX 部署流程图”。
4. 第五章可在可解释性分析部分使用“可解释性分析流程图”。
