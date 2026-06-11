# restaurant_project餐厅评论情感分析与评分预测项目 README
项目概述
本项目基于餐厅用户评论文本与分项环境 / 口味 / 服务评分，搭建二分类情感模型（好评 / 差评判别）与回归评分预测模型（1–5 分总体评分预测），完整覆盖文本预处理、探索性数据分析、TF-IDF 文本向量化、逻辑回归分类、随机森林多特征回归、网格搜索超参调优、模型可视化、新样本推理与结果导出全流程，适用于餐饮口碑挖掘、用户情感识别、菜品质量量化评估场景。
项目包含两大核心任务：
1.文本情感二分类：过滤 3 分中性评论，≥4 分为好评 (1)、≤2 分为差评 (0)，采用 TF-IDF + 逻辑回归，GridSearchCV 自动寻优，输出分类指标与关键词权重；
2.多特征评分回归：融合评论文本 TF-IDF 特征 + 环境 / 口味 / 服务分项标准化数值特征，使用随机森林回归预测餐厅总体评分，输出 MAE/RMSE/R² 回归评估指标。
一、项目目录结构
plaintext
restaurant_project/
├── ratings_sample.csv                        # 餐厅评论数据集
├── stopwords.txt                             # 中文停用词表
├── restaurant_project_results.ipynb          # 完整分析主程序
├── LICENSE
└── README.md                                 # 项目说明文档
二、环境依赖与安装指南
基础 Python 版本要求
Python >= 3.8（推荐 3.9/3.10，兼容 sklearn、matplotlib 全部接口）
完整依赖清单
表格
库名称	用途	安装命令
numpy	数值计算、矩阵处理	pip install numpy
pandas	表格数据读写、清洗统计	pip install pandas
matplotlib	数据可视化（分布柱状图、热力图、混淆矩阵、散点图）	pip install matplotlib
scikit-learn	TF-IDF、划分数据集、逻辑回归、随机森林、网格搜索、评估指标	pip install scikit-learn
jieba	中文精准分词（可选，未安装程序自动启用简易分词兜底）	pip install jieba
IPython	Notebook 环境表格展示 display 函数（纯脚本运行无依赖报错）	pip install ipython
三、运行方式
Jupyter Notebook（推荐，可视化完整）
将代码全量复制至.ipynb文件；
确认同目录存在ratings_sample.csv，可选放入stopwords.txt；
执行，自动输出表格、图表、模型评估日志；
运行前置检查项
目录存在ratings_sample.csv且字段符合规范；
已安装全部依赖库；