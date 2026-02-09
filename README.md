<div align="center">
  <h1>🌌 News-Prism | 极速中文新闻智能分类系统</h1>
  <p>
    <b>Enterprise-Grade Chinese News Classification System based on FastText</b>
  </p>
  
  <p>
    <a href="https://www.python.org/">
      <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python">
    </a>
    <a href="https://fasttext.cc/">
      <img src="https://img.shields.io/badge/Engine-FastText-red.svg" alt="FastText">
    </a>
    <a href="https://flask.palletsprojects.com/">
      <img src="https://img.shields.io/badge/Microservice-Flask-green.svg" alt="Flask">
    </a>
    <a href="https://github.com/facebookresearch/fastText">
      <img src="https://img.shields.io/badge/AutoML-Supported-orange.svg" alt="AutoML">
    </a>
    <a href="LICENSE">
      <img src="https://img.shields.io/badge/License-MIT-lightgrey.svg" alt="License">
    </a>
  </p>
</div>

---

## 📖 项目背景 (Background)

在海量非结构化文本数据爆发的今天，传统的基于规则或统计机器学习（如 SVM、RandomForest）的分类方案已难以满足**亿级数据量**下的实时处理需求。而基于 Transformer（如 BERT/GPT）的重型模型虽然精度高，但推理成本高昂，难以在边缘设备或低延迟要求的场景下落地。

**News-Prism** 应运而生。本项目基于 Facebook Research 开源的 **FastText** 深度表征学习框架，旨在构建一个**轻量级、毫秒级响应、高精度**的中文新闻文本分类引擎。它不仅支持传统的词袋模型（BoW），还引入了 N-gram 特征与层级 Softmax 加速技术，在保持与深度神经网络相当精度的同时，训练与推理速度提升了 **100x - 1000x** 倍。

---

## 🏗️ 系统架构与核心特性 (Architecture)

### 1. 核心技术栈
* **Deep Learning Engine**: `Facebook FastText` (高效文本分类与表征学习)
* **NLP Tools**: `Jieba` (精准中文分词), `N-gram` (多尺度特征提取)
* **AutoML**: `FastText Autotune` (自动化模型超参数调优)
* **Microservice**: `Flask` (轻量级 RESTful API 网关)
* **ETL Pipeline**: 自研数据清洗与格式化管道

### 2. 双模驱动 (Dual-Mode Engine)
系统创新性地支持两种特征工程模式，适应不同业务场景：
* **⚡ 字符级模式 (Char-Level Mode)**:
    * **原理**: 将每个汉字作为 Token，结合 Bi-gram/Tri-gram 捕捉局部语义。
    * **优势**: 彻底解决 OOV (Out-of-Vocabulary) 问题，抗噪声能力强，适合微博、评论等口语化文本。
* **🧩 词级模式 (Word-Level Mode)**:
    * **原理**: 使用 `Jieba` 进行语义分词，基于词向量（Word Embedding）进行表征。
    * **优势**: 语义边界清晰，适合长篇正规新闻报道。

---

## 📂 项目模块深度解析 (Module Manifest)

本项目采用模块化设计，遵循 **"Configuration over Convention"** 原则。

| 文件名 (Filename) | 模块类型 | 深度功能解析 (Description) |
| :--- | :--- | :--- |
| **`01-data_preprocess.py`** | 🧹 **ETL 引擎** | **数据清洗与格式化入口**。<br>1. **Pipeline**: 原始语料 -> 正则清洗 -> 分词/分字 -> Label格式化。<br>2. **Router**: 根据配置自动分流训练集、验证集与测试集。<br>3. **Output**: 生成 FastText 标准格式 `__label__Category Text`。 |
| **`02-fasttext_char_2_auto.py`** | 🧠 **训练核心 (Pro)** | **AutoML 训练脚本 (生产环境推荐)**。<br>集成 `autotuneValidationFile` 机制，在设定时间预算（Duration）内，利用遗传算法自动搜索最优超参数组合（Learning Rate, Epoch, Dim, Bucket）。确保模型在体积与精度间取得最佳平衡。 |
| **`02-fasttext_char_1_default.py`** | 🧪 **训练核心 (Lite)** | **Baseline 训练脚本**。<br>使用标准参数快速构建基准模型，用于验证数据管道连通性或进行快速迭代实验。 |
| **`api.py`** | 🚀 **服务网关** | **推理服务主程序**。<br>基于 Flask 构建的高并发 RESTful API。内置全局模型预加载（Pre-loading）机制，避免重复 I/O，实现亚毫秒级推理延迟。监听端口: `8003`。 |
| **`predict.py`** | ⚙️ **推理逻辑** | **预测引擎封装层**。<br>负责模型生命周期管理。包含模型加载、输入文本预处理（去噪、截断）、向量化推理及结果后处理（Label 解码）。 |
| **`config.py`** | 🔧 **配置中心** | **全局环境配置**。<br>统一管理所有 I/O 路径、超参数默认值及环境常量。实现“代码与配置分离”，增强系统的可移植性与可维护性。 |
| **`api_test.py`** | 🚦 **质量探针** | **端到端自动化测试**。<br>模拟真实客户端请求，监控 API 的可用性（Availability）、准确性（Accuracy）及延迟（Latency）。 |

---

## ⚡ 快速启动指南 (Quick Start)

### 环境准备 (Prerequisites)
推荐使用 `Linux` 或 `macOS` 环境，并安装 Python 3.8+。

```bash
# 安装依赖
pip install fasttext-wheel flask jieba pandas requests
```

### Phase 1: 数据流构建 (Data Pipeline)
将原始语料转化为模型可读的格式。

1.  编辑 `01-data_preprocess.py`，配置输入源与分词模式。
2.  执行 ETL 脚本：
    ```bash
    python 01-data_preprocess.py
    ```
    *Check: 检查 `result/` 目录下是否生成了 `train_fastText_char.txt` 等文件。*

### Phase 2: AutoML 模型训练 (Training)
让算法自动寻找最佳参数组合。

```bash
python 02-fasttext_char_2_auto.py
```
*Log: 系统将自动进行多轮参数搜索，最终模型将保存至 `save_models/`。*

### Phase 3: 服务微服务化部署 (Deployment)
启动高性能预测服务。

```bash
python api.py
```
*Status: 服务将运行在 `http://0.0.0.0:8003`，等待调用。*

---

## 📡 API 接口规范 (Interface Specification)

**Base URL**: `http://127.0.0.1:8003`

### 1. 新闻分类预测 (Predict)

* **Endpoint**: `/predict`
* **Method**: `POST`
* **Content-Type**: `application/json`

**请求参数 (Request):**

| 参数名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| `text` | string | 是 | 需要分类的新闻文本内容（建议长度 > 10字符） |

**请求示例 (cURL):**

```bash
curl -X POST http://127.0.0.1:8003/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "SpaceX 星舰今日成功发射，开启火星移民新篇章，马斯克发文庆祝。"}'
```

**响应示例 (Response):**

```json
{
    "text": "SpaceX 星舰今日成功发射...",
    "pred_class": "科技",
    "meta": {
        "model": "fastText_char_v2",
        "latency_ms": "0.42ms"
    }
}
```

---

## 🗺️ 路线图 (Roadmap)

我们致力于将 News-Prism 打造为最通用的文本分类脚手架：

* [x] **v1.0**: 完成 FastText 核心算法集成，支持字/词双模式。
* [x] **v1.1**: 实现 Flask 微服务化封装与 AutoML 自动调参。
* [ ] **v2.0**: 增加 Docker 容器化支持与 Docker-Compose 编排文件。
* [ ] **v2.1**: 集成 Prometheus + Grafana 监控，实现服务可观测性。
* [ ] **v3.0**: 探索模型量化（Quantization）技术，适配移动端部署。

---

## 📚 参考文献 (References)

本项目的核心算法实现参考了以下学术论文与开源项目：

1.  **Enriching Word Vectors with Subword Information**
    * *P. Bojanowski, E. Grave, A. Joulin, T. Mikolov*, 2017
    * [Link to Paper](https://arxiv.org/abs/1607.04606)
2.  **Bag of Tricks for Efficient Text Classification**
    * *A. Joulin, E. Grave, P. Bojanowski, T. Mikolov*, 2016
    * [Link to Paper](https://arxiv.org/abs/1607.01759)
3.  **FastText Official Documentation**
    * https://fasttext.cc/

---

## ❤️ 致谢 (Acknowledgments)

本项目在开发过程中，深受以下社区与课程的启发，特此致谢：

* **黑马程序员 (Dark Horse Programmer)**: 感谢其优秀的 AI 与大数据课程体系，为本项目提供了坚实的理论基础与工程化灵感。本项目部分数据处理逻辑参考了其教学案例。
* **FastText Community**: 感谢 Facebook Research 开源了如此优秀的 NLP 工具。
* **Open Source Contributors**: 感谢所有为 Python NLP 生态做出贡献的开发者。

---

## 📄 版权说明 (License)

本项目采用 **MIT License** 开源协议。
这意味着你可以自由地使用、复制、修改、合并、出版发行、散布、再授权及贩售本软件的副本。

> **News-Prism** - *Lighting up the dark corners of unstructured data.*
>
> 2026 © Developed by News-Prism Team
