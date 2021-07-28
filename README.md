xeasy-ml
====
## 1. What is xeasy-ml
Xeasy-ml is a packaged machine learning framework. It allows a beginner to quickly build a machine learning model and use the model to process and analyze his own data. At the same time, we have also realized the automatic analysis of data. During data processing, xeasy-ml will automatically draw data box plots, distribution histograms, etc., and perform feature correlation analysis to help users quickly discover the value of data.

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