import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xeasy-ml",
    version="0.1.0",
    author="X",
    author_email="author@example.com",
    description="An integrated machine learning analysis framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src1"},
    packages=setuptools.find_packages(where="src1"),
    package_data={'xeasy_ml': ['project_init/*','xes_ml_arch/*','xes_ml_arch/config/*','xes_ml_arch/config/demo/*','xes_ml_arch/data/*','xes_ml_arch/log/*','xes_ml_arch/result/*',
                             'xes_ml_arch/src/tests/analysis_test/config/*','xes_ml_arch/src/tests/analysis_test/log/*','xes_ml_arch/src/tests/analysis_test/*',
                             'xes_ml_arch/src/tests/config/*', 'xes_ml_arch/src/tests/cross_validation_test/*','xes_ml_arch/src/tests/cross_validation_test/model/*',
                             'xes_ml_arch/src/tests/data/*', 'xes_ml_arch/src/tests/feature_enginner_test/*','xes_ml_arch/src/tests/feature_enginner_test/conf/*',
                             'xes_ml_arch/src/tests/feature_enginner_test/config/*','xes_ml_arch/src/tests/feature_enginner_test/data/*','xes_ml_arch/src/tests/feature_enginner_test/model/*',
                             'xes_ml_arch/src/tests/ml_test/config/*', 'xes_ml_arch/src/tests/ml_tests/*', 'xes_ml_arch/src/tests/model/*','xes_ml_arch/src/tests/model_test/*',
                             'xes_ml_arch/src/tests/model_test/config/*', 'xes_ml_arch/src/tests/*']
                  },

    install_requires=[ 'scikit-learn==0.24.1',
                    'pandas==0.24.2',
                    'matplotlib==3.3.4',
                    'pydotplus==2.0.2',
                    'xgboost==1.4.2',
                    'numpy==1.19.5'
                ],
    python_requires=">=3.6",
)
