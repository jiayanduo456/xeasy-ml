xeasy-ml
====
## 1. What is xeasy-ml
Xeasy-ml is a packaged machine learning framework. It allows a user to quickly build a machine learning model only with configuration and use the model to process and analyze his own data. At the same time, we have also realized the automatic analysis of data. During data processing, xeasy-ml will automatically draw data box plots, distribution histograms, etc., and perform feature correlation analysis to help users quickly discover the value of data.

## 2.Installation
### Dependencies
xeasy-ml requires:

    Scikit-learn >= 0.24.1
    
    Pandas >= 0.24.2
    
    Numppy >= 1.19.5
    
    Matplotlib >= 3.3.4
    
    Pydotplus >= 2.0.2
    
    Xgboost >= 1.4.2
### User installation
    pip install xeasy-ml

## 3. Quick Start


### 1.Create a new project

#### Create a new python file named pro_init.py to initialize the project.
```Bash
from xeasy_ml.project_init.create_new_demo import create_project
import os

pro_path = os.getcwd()
create_project(pro_path)
```

#### Now you can see the following file structure in your project.
```
├── Your_project
     ...
│   ├── pro_init.py
│   ├── project
│   │   └── your_project
```
### 2.Run example
```Bash
cd project/your_project

python __main__.py
```

### 3.View Results

```Bash
cd project/your_project_name/result/v1
ls -l
```
    ├── box   (Box plot)
    ├── cross_predict.txt （Cross-validation prediction file）
    ├── cross.txt  （Cross validation effect evaluation）
    ├── deleted_feature.txt  （Features that need to be deleted）
    ├── demo_feature_weight.txt  （Feature weights）
    ├── demo.m   （Model）
    ├── feature_with_feature  （Feature similarity）
    ├── feature_with_label   （Similarity between feature and label ）
    ├── hist    （Distribution histogram）
    ├── model
    ├── predict_result.txt  （Test set prediction results）
    └── test_score.txt      （Score on the test set）

<br/>

---
xeasy-ml中文文档
====
### 1. 简介

​		xeasy-ml是一个机器学习框架。它封装了常用的机器学习学习组件和方法，允许使用者通过配置方式快速建立机器学习模型，并使用该模型处理和分析自己的数据，进行模型选择和参数调优。在数据处理和分析过程中，xeasy-ml会自动绘制数据的箱线图、分布直方图等，并进行特征相关性分析，帮助用户快速发现数据的价值。

### 2.安装

依赖包：

```
Scikit-learn >= 0.24.1
Pandas >= 0.24.2
Numppy >= 1.19.5
Matplotlib >= 3.3.4
Pydotplus >= 2.0.2
Xgboost >= 1.4.2
```

​	安装：

```
pip install xeasy-ml
```

### 3.如何使用

#### 1.创建自己的项目

```bash
#创建一个名为pro_init.py的新python文件来初始化项目。
from xeasy_ml.project_init.create_new_demo import create_project
import os
pro_path = os.getcwd()
create_project(pro_path)
```

```python
#在pro_init.py同级目录下可以看到以下目录结构：
├── Your_project
 	 ...
	├── pro_init.py
	├── project
	│  └── your_project
```

#### 	2.运行	

```bash
cd project/your_project
python __main__.py
```

#### 	3.查看结果

~~~bash
cd project/your_project_name/result/v1
ls -l

  ├── box  (箱线图)

  ├── cross_predict.txt （交叉验证预测文件）

  ├── cross.txt （交叉验证评估）

  ├── deleted_feature.txt （需要被删除的特征）

  ├── demo_feature_weight.txt （模型特征权重）

  ├── demo.m  （保存的模型文件）

  ├── feature_with_feature （特征相似度）

  ├── feature_with_label  （特征与标签相似度）

  ├── hist  （分布直方图）

  ├── model

  ├── predict_result.txt （测试集预测结果）

  └── test_score.txt   （测试集评价指标得分）
~~~

### 4.线上使用手册

​		假设你已经按照3.1的指引生成了你的个人项目文件夹，文件的目录结构为：

```
|———— Your_project
 	 ...
	| |———— pro_init.py
	| |———— project
	| |	└──your_project
	| |	   └──config
	| |	      └──demo
	| |		 └──ml.conf
	| |		 └──model.conf
	| |		 ...
	| |	      |——log.conf
	| |	   |——data
	| |	      └──sample.txt
	| |        |——log
	| |        |——result
	| |        |——__main__.py									
```

#### 	1.训练

​		上述project结构中，config文件夹下为模型配置文件和日志配置文件；data为训练集；log是训练过程储存日志的文件夹，你可以在这里查看你的模型运行日志；result用于储存模型运行过程产生的数据分析资料，模型文件等；

​		训练时，你可以根据自己的任务对配置文件进行调整，数据需存放在data文件夹下；模型训练和预测的结果在result内；加入你已经完成了模型的训练过程，你最需要关注的是result下的变化，其中最重要的是model文件下的demo.m，这是模型训练后的储存文件。

```
|——result
   |——v1
      |——box
      |——hist
      |——model
```

####  2.工程预测

​	    线上使用xeasy-ml时，你需要准备三个文件：demo.m , log.conf 和feature_enginnering.conf；在完成训练步骤后，你可以在project文件夹下找到它们；将这三个文件放在你的工程目录下，接着你需要做的就是写出你自己的predict.py(或者调用xeasy-ml.predict()方法，传入上述三个参数),这个文件包括xeasy-ml中的prediction_ml.PredictionML类用以初始化模型，PredictionML(config=conf, xeasy_log_path = xeasy_log_path)有两个参数：config是用于模型初始化的文件，easy_log_path是模型的日志配置文件；这里有个要注意的地方是我们可以根据自己的需要决定是否传入模型的配置文件（训练中的ml.conf）文件的作用是根据配置信息初始化模型（包括数据处理等），如果执行这一步操作，你需要在与启动文件相同目录下添加’./config/demo/model.conf‘和'’./config/demo/feature_enginnering.conf‘'；需要注意的是ml.conf和model.conf的参数调整

```python
self.ml = xml.prediction_ml.PredictionML(config = 'your ml.conf path', xeasy_log_path=xeasy_log_path)
```

如果只是使用XGBClassifier模型，不需要传入模型初始化文件，也不需要额外建立’./config/demo/model.conf‘文件目录；仅传入日志配置文件即可，但是需要自定义数据处理，代码形式如下：

```python
self.ml = xml.prediction_ml.PredictionML(xeasy_log_path=xeasy_log_path)
self.ml._model = XGBClassifier()
self.ml._model.load_model(model_path)

self.ml._feature_processor = xml.data_processor.DataProcessor(conf=ml_config, log_path=xeasy_log_path)
self.ml._feature_processor.init()
```

以上步骤是线上模型的两种初始化方式；初始化后，对预测数据进行预测前需要进行数据处理，例：

```python
self.ml._feature_processor.test_data = data_frame
self.ml._feature_processor.execute()
# 测试数据
test_feature = self.ml._feature_processor.test_data_feature.astype("float64", errors='ignore')
# 预测结果
predict_res = self.ml._model.predict(test_feature)
```
