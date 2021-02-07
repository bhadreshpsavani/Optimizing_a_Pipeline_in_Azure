# Optimizing an ML Pipeline in Azure

## Overview
In this project, We have build and optimized an ML pipeline. We have used a custom-coded model—a standard Scikit-learn Logistic Regression of which we have optimized hyperparameters using HyperDrive. We have also used AutoML to build and optimize a model on the same dataset, so that we can compare the results of the two methods.

You can see the main steps that you'll be taking in the diagram below:

![project-flow](/Images/creating-and-optimizing-an-ml-pipeline.png)

## Summary
We are going to use data about Banking Marketing. It has 39 features about target customers and we will predict If marketing is affective or not based on the 39 Features.
```
# features
age                              32950 non-null int64
marital                          32950 non-null int64
default                          32950 non-null int64
housing                          32950 non-null int64
loan                             32950 non-null int64
month                            32950 non-null int64
day_of_week                      32950 non-null int64
duration                         32950 non-null int64
campaign                         32950 non-null int64
pdays                            32950 non-null int64
previous                         32950 non-null int64
poutcome                         32950 non-null int64
emp.var.rate                     32950 non-null float64
cons.price.idx                   32950 non-null float64
cons.conf.idx                    32950 non-null float64
euribor3m                        32950 non-null float64
nr.employed                      32950 non-null float64
job_admin.                       32950 non-null uint8
job_blue-collar                  32950 non-null uint8
job_entrepreneur                 32950 non-null uint8
job_housemaid                    32950 non-null uint8
job_management                   32950 non-null uint8
job_retired                      32950 non-null uint8
job_self-employed                32950 non-null uint8
job_services                     32950 non-null uint8
job_student                      32950 non-null uint8
job_technician                   32950 non-null uint8
job_unemployed                   32950 non-null uint8
job_unknown                      32950 non-null uint8
contact_cellular                 32950 non-null uint8
contact_telephone                32950 non-null uint8
education_basic.4y               32950 non-null uint8
education_basic.6y               32950 non-null uint8
education_basic.9y               32950 non-null uint8
education_high.school            32950 non-null uint8
education_illiterate             32950 non-null uint8
education_professional.course    32950 non-null uint8
education_university.degree      32950 non-null uint8
education_unknown                32950 non-null uint8
```

Targets:
```
# Python code:
y.value_counts(normalize=True)

Output:
0    0.887951
1    0.112049
Name: y, dtype: float64
```

We can say that our data is imbalence. We have `88.78%` 'No' as target and `11.28%` 'YES' as target inside data.

The best performing Model was **VotingEnsemble** model. It give `91.72%` Accuracy on Validation dataset. It was finetune using AutoML.

## Part1. Scikit-learn Pipeline (HyperDrive based Hyperparameter Tuning)

We used Logistic Regression based model for this binary classfication problem

The steps performed in SKLearn Pipeline are,
1. Clean Data and Convert into DataFrame object
2. Get Train and Validation Set
3. Do Hyperparameter Tuning using Hyperdrive
4. Get The Best Performing model based on Accuracy

We have used Logistic Regression model in the sklearn pipeline. This model is considered as first choice for simple classification problem. It gives probability of classes. We can choose class with highest probabality. Our problem is Binary CLassification so we have used this model.

We finetune Two parameters of Logistic Regression:
1. C (default=1.0) : It is Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
2. max_iter (default=100) : Maximum number of iterations taken for the solvers to converge.

### Benefits of Random sampling
Random sampling supports discrete and continuous hyperparameters. It supports early termination of low-performance runs. 

We can do an initial search with random sampling and then refine the search space to improve results, while Grid sampling performs a simple grid search over all possible values. If we use Grid Search It needs to go over 44 different hyerparameter combinations. Bayesian sampling is recommended if you have enough budget to explore the hyperparameter space. For best results, it was recommend a maximum number of runs greater than or equal to 20 times the number of hyperparameters being tuned. So It will take longer time.

We are having `max_total_runs=16` runs so Random sampling seems to be better option.

### Benefits of Bandit policy
Early termination improves computational efficiency. Bandit policy is based on slack factor/slack amount and evaluation interval. Bandit terminates runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run.
```
BanditPolicy(evaluation_interval=2, slack_factor=0.1)
```
We used `slack_factor=0.1`. So, Any run whose best metric is less than (1/(1+0.1) or 91% of the best performing run will be terminated.

### Best Model Hyperparameters Selected:
```
Model Name: LogisticRegression 
Accuracy: 0.9142995872784656
C: 0.001
Max Iter: 50
```

### Output of Hyperdrive Run

![hyperdrive-logs](/Images/1111.PNG)
![hyperdrive-runs](/Images/2222.PNG)
![hyperdrive-accuracy](/Images/3333.PNG)
![hyperdrive-2d](/Images/4444.PNG)
![hyperdrive-3d](/Images/5555.PNG)
![hyperdrive-3d](/Images/6666.PNG)

## Part2. AutoML
Using AutoML with just One Click we can try different Powerful Models based on Emsembling, Boosting, Randomforest etc. It will internally select Hyperparameters and Model.  

### Best Model Hyperparameters Selected:
```
Model Name: VotingEnsemble
Accuracy: 0.9172382397572079
```
A Voting Classifier is a machine learning model that trains on an ensemble of numerous models and predicts an output (class) based on their highest probability of chosen class as the output.

It simply aggregates the findings of each classifier passed into Voting Classifier and predicts the output class based on the highest majority of voting. The idea is instead of creating separate dedicated models and finding the accuracy for each them, we create a single model which trains by these models and predicts output based on their combined majority of voting for each output class.

Voting Classifier supports two types of votings.

1. Hard Voting: In hard voting, the predicted output class is a class with the highest majority of votes i.e the class which had the highest probability of being predicted by each of the classifiers. Suppose three classifiers predicted the output class(A, A, B), so here the majority predicted A as output. Hence A will be the final prediction.

2. Soft Voting: In soft voting, the output class is the prediction based on the average of probability given to that class. Suppose given some input to three models, the prediction probability for class A = (0.30, 0.47, 0.53) and B = (0.20, 0.32, 0.40). So the average for class A is 0.4333 and B is 0.3067, the winner is clearly class A because it had the highest probability averaged by each classifier.

The Best Model given by AutoML is using SoftVoting.

The VotingClassifier Used below list of estimators
* XGBoostClassifier
* LightGBM
* LogisticRegression
* RandomForest

Note: The best model used Many XGBoostClassifier with Different set of parameters 

Our best Fitted Model Parameters:

```
prefittedsoftvotingclassifier
{'estimators': ['1', '0', '14', '11', '6', '9', '5'],
 'weights': [0.15384615384615385,
             0.15384615384615385,
             0.15384615384615385,
             0.07692307692307693,
             0.15384615384615385,
             0.07692307692307693,
             0.23076923076923078]}

Pipeline(memory=None,
         steps=[('maxabsscaler', MaxAbsScaler(copy=True)),
                ('xgboostclassifier',
                 XGBoostClassifier(base_score=0.5, booster='gbtree',
                                   colsample_bylevel=1, colsample_bynode=1,
                                   colsample_bytree=1, gamma=0,
                                   learning_rate=0.1, max_delta_step=0,
                                   max_depth=3, min_child_weight=1, missing=nan,
                                   n_estimators=100, n_jobs=1, nthread=None,
                                   objective='binary:logistic', random_state=0,
                                   reg_alpha=0, reg_lambda=1,
                                   scale_pos_weight=1, seed=None, silent=None,
                                   subsample=1, tree_method='auto', verbose=-10,
                                   verbosity=0))],
         verbose=False)
Pipeline(memory=None,
         steps=[('maxabsscaler', MaxAbsScaler(copy=True)),
                ('lightgbmclassifier',
                 LightGBMClassifier(boosting_type='gbdt', class_weight=None,
                                    colsample_bytree=1.0,
                                    importance_type='split', learning_rate=0.1,
                                    max_depth=-1, min_child_samples=20,
                                    min_child_weight=0.001, min_split_gain=0.0,
                                    n_estimators=100, n_jobs=1, num_leaves=31,
                                    objective=None, random_state=None,
                                    reg_alpha=0.0, reg_lambda=0.0, silent=True,
                                    subsample=1.0, subsample_for_bin=200000,
                                    subsample_freq=0, verbose=-10))],
         verbose=False)
Pipeline(memory=None,
         steps=[('sparsenormalizer',
                 <azureml.automl.runtime.shared.model_wrappers.SparseNormalizer object at 0x7f9c69150b70>),
                ('xgboostclassifier',
                 XGBoostClassifier(base_score=0.5, booster='gbtree',
                                   colsample_bylevel=1, colsample_bynode=1,
                                   colsample_bytree=0.6, eta=0.3, gamma=1,
                                   learning_rate=0.1, max_delta_step=0,
                                   max_depth=6, max_leaves=31,
                                   min_child_weight=1, missing=nan,
                                   n_estimators=200, n_jobs=1, nthread=None,
                                   objective='reg:logistic', random_state=0,
                                   reg_alpha=0, reg_lambda=1.6666666666666667,
                                   scale_pos_weight=1, seed=None, silent=None,
                                   subsample=0.9, tree_method='auto',
                                   verbose=-10, verbosity=0))],
         verbose=False)
Pipeline(memory=None,
         steps=[('sparsenormalizer',
                 <azureml.automl.runtime.shared.model_wrappers.SparseNormalizer object at 0x7f9c688db128>),
                ('xgboostclassifier',
                 XGBoostClassifier(base_score=0.5, booster='gbtree',
                                   colsample_bylevel=1, colsample_bynode=1,
                                   colsample_bytree=0.9, eta=0.3, gamma=0,
                                   learning_rate=0.1, max_delta_step=0,
                                   max_depth=10, max_leaves=15,
                                   min_child_weight=1, missing=nan,
                                   n_estimators=25, n_jobs=1, nthread=None,
                                   objective='reg:logistic', random_state=0,
                                   reg_alpha=0, reg_lambda=0.5208333333333334,
                                   scale_pos_weight=1, seed=None, silent=None,
                                   subsample=0.6, tree_method='auto',
                                   verbose=-10, verbosity=0))],
         verbose=False)
Pipeline(memory=None,
         steps=[('sparsenormalizer',
                 <azureml.automl.runtime.shared.model_wrappers.SparseNormalizer object at 0x7f9c688df588>),
                ('xgboostclassifier',
                 XGBoostClassifier(base_score=0.5, booster='gbtree',
                                   colsample_bylevel=1, colsample_bynode=1,
                                   colsample_bytree=0.9, eta=0.3, gamma=0,
                                   learning_rate=0.1, max_delta_step=0,
                                   max_depth=9, max_leaves=0,
                                   min_child_weight=1, missing=nan,
                                   n_estimators=25, n_jobs=1, nthread=None,
                                   objective='reg:logistic', random_state=0,
                                   reg_alpha=0, reg_lambda=0.7291666666666667,
                                   scale_pos_weight=1, seed=None, silent=None,
                                   subsample=0.9, tree_method='auto',
                                   verbose=-10, verbosity=0))],
         verbose=False)
Pipeline(memory=None,
         steps=[('maxabsscaler', MaxAbsScaler(copy=True)),
                ('logisticregression',
                 LogisticRegression(C=2.559547922699533, class_weight=None,
                                    dual=False, fit_intercept=True,
                                    intercept_scaling=1, l1_ratio=None,
                                    max_iter=100, multi_class='ovr', n_jobs=1,
                                    penalty='l2', random_state=None,
                                    solver='saga', tol=0.0001, verbose=0,
                                    warm_start=False))],
         verbose=False)
Pipeline(memory=None,
         steps=[('maxabsscaler', MaxAbsScaler(copy=True)),
                ('randomforestclassifier',
                 RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
                                        class_weight='balanced',
                                        criterion='entropy', max_depth=None,
                                        max_features='sqrt',
                                        max_leaf_nodes=None, max_samples=None,
                                        min_impurity_decrease=0.0,
                                        min_impurity_split=None,
                                        min_samples_leaf=0.01,
                                        min_samples_split=0.2442105263157895,
                                        min_weight_fraction_leaf=0.0,
                                        n_estimators=10, n_jobs=1,
                                        oob_score=False, random_state=None,
                                        verbose=0, warm_start=False))],
         verbose=False)
```

### Output of AutoML Run

![AutoML-runs](/Images/7777.PNG)
![AutoML-Accuracy](/Images/8888.PNG)
![AutoML-auc-weighted](/Images/99999999.PNG)
![AutoML-f1-weighted](/Images/1111111100000000.PNG)

## Part3. Pipeline comparison
* While using HyperDrive based pipeline, We can only use one model at a time. 
* We need to define the range of Hyperparameters, Sampling Policy and Early Stopping Policy but in AutoML we just need to give input details about task. 
* AutoML based pipeline can use other models like Deep Learning based or from other library.

## Part4. Future work

1. This dataset is imbalanced as mentioned earlier. If model is predicting well for mejority class. It will give better Accuracy and still fail to perform well on minority class. We should consider this problem as Class Imbalance Classfication problem. Accuracy is not the right metric to be considered in such cases.

   There are several ways to handle this probelm, 
   1. Consider F1 score or Recall or AUC as primary Matrix for model Selection since it is less affected by Imbalance data

   2. Change the algorithm : While in every machine learning problem, it’s a good rule of thumb to try a variety of algorithms, it can be especially beneficial with imbalanced datasets. Decision trees frequently perform well on imbalanced data. They work by learning a hierarchy of if/else questions and this can force both classes to be addressed.

   3. Try Upsampling(Oversampling) or Downsampling(Undersampling) techniques to reduce imbalance in data

       ![resampling](/Images/resampling.png)

       i. Oversampling: As shown in the image Oversampling will increase minority class data by duplicating it. We should try this when we has relatively small amount of data

       ii. Undersampling: It will reduce data in Majority class data. THis is useful when we have huge amount of data and we can afford to lose data.

       This way model will be able to see equal distribution of classes and it will be unbias while prediction

   4. Generate synthetic samples : A technique similar to upsampling is to create synthetic samples. Here we will use imblearn’s SMOTE or Synthetic Minority Oversampling Technique. SMOTE uses a nearest neighbors algorithm to generate new and synthetic data we can use for training our model.

       Again, it’s important to generate the new samples only in the training set to ensure our model generalizes well to unseen data.
       ```
       from imblearn.over_sampling import SMOTE
       # Separate input features and target
       y = df.Class
       X = df.drop('Class', axis=1)

       # setting up testing and training sets
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)

       sm = SMOTE(random_state=27, ratio=1.0)
       X_train, y_train = sm.fit_sample(X_train, y_train)
       ```

2. We are having 39 features in the dataset. All the features might not be equally important. 

   Model-based feature selection can be tried using sklearn. We will know important of each of this feature. We can eleminate less important features and use only important features. 

   If might help us to speed up inference time. It might be a boost when we will use this model in production. 

## Reference:
* [how-to-configure-auto-train-in-azure](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train)
* [how-to-tune-hyperparameters-in-hyperdrive](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)
* [ml-voting-classifier-using-sklearn](https://www.geeksforgeeks.org/ml-voting-classifier-using-sklearn/)
* [Why-is-XGBoost-better-than-logistic-regression](https://www.quora.com/Why-is-XGBoost-better-than-logistic-regression)
* [methods-for-dealing-with-imbalanced-data](https://towardsdatascience.com/methods-for-dealing-with-imbalanced-data-5b761be45a18)
* [feature-selection-dimensionality-reduction-techniques-to-improve-model-accuracy](https://towardsdatascience.com/feature-selection-dimensionality-reduction-techniques-to-improve-model-accuracy-d9cb3e008624)
