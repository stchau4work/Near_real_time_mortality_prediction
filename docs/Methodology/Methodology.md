<h1 align="center">Methodology</h1>

## Higher frequency data

Machine Learning is to learn patterns from data. Living in a world that is full of data points, insurance companies could leverage on this to devise better algorithms for various decision making. To illustrate this, I have first sourced the dataset from [Google trend](https://trends.google.com/trends).

Knowing that there are more channels to provide higher frequency dataset, given the time constraint, one who in the further look for the more alternative dataset

![googleTrend.png](https://raw.githubusercontent.com/stchau4work/Near_real_time_mortality_prediction/master/docs/Methodology/googleTrend.png)

Google Trend provide high-frequency dataset up to hourly basis and the data coverage is since 2004

## First step to identify the web query - feature extraction

We start with something simple. By taking a static screenshot at the year of 2014 and with a dedicated focus on Neoplasms (one of the major causes of death), we try to relate the queries with the IHME dataset

![geoMap.png](https://raw.githubusercontent.com/stchau4work/Near_real_time_mortality_prediction/master/docs/Methodology/geoMap.png)

Now, it is where the creativity comes in, we need to make some educated guess on the web queries and extract their trends for around 50 regions in the United State

      x1 = # of breast cancer queries in 2014
      x2 = # of oral cancer queries in 2014
      x3 = # of esophageal cancer in 2014
      x4 = # of lymphoma cancer queries in 2014
      x5 = # of leukemia cancer queries in 2014
      x6 = # of stomach cancer queries in 2014
      x7 = # of liver cancer queries in 2014
      x8 = # of pancreatkc cancer queries in 2014
      x9 = # of lung cancer queries in 2014
     x10 = # of skin cancer queries in 2014
     x11 = # of colon cancer queries in 2014
     x12 = # of cavocal cancer queries in 2014
     x13 = # of ovarian cancer queries in 2014
     x14 = # of prostate cancer queries in 2014
     x15 = # of testicular cancer queries in 2014
     x16 = # of kidney cancer queries in 2014
     x17 = # of bladder cancer queries in 2014
     x18 = # of brain cancer queries in 2014
     x19 = # of thyroid cancer queries in 2014

Then we apply machine learning models to find the relationship between these counts of queries and the corresponding mortality rate from IHME dataset

$$ mortality = y = f(x1,x2, ..., x19) $$

## Supervised Machine Learning

First of all, we are going to split the dataset into training, development set and testing set in ratio 8-1-1

Hyper-parameter turning in training and development sets which aim to maximise the predictive power of the models

In this exercise, we try to play with Random Forest Regression and Neural Network to predict the Mortality Rate

In Random Forest Regression, despite it won't overfit, but it has been suggested to keep the max-features as one-third of the variables so as to reduce the bias

![randomForest.png](https://raw.githubusercontent.com/stchau4work/Near_real_time_mortality_prediction/master/docs/Methodology/randomForest.png)

In Artificial Neural Network, there are numerous hyper parameters to turn such as the architecture of the hidden layers, activation functions and so on. We will apply stochastic gradient descent optimization method and drop out layer to prevent get stuck at the local minimum / not overfitting

![ANN.png](https://raw.githubusercontent.com/stchau4work/Near_real_time_mortality_prediction/master/docs/Methodology/ANN.png)

When selecting the models, we consider the errors

$$ Errors = Bias + Variance + Irreducible error $$

Out-of-sample mean absolute error:

    Random Forest: 5.720633
              ANN: 6.549579

However, as the size of the data is not big enough to draw any conclusion, the process has demonstrated the essential ideas for choosing the model

In addition, a closer look at the training result of random forest we observe that there are 50 trees with max_depth as 7

The errors in both training set and validation set are roughly the same, which is a good sign

```R

H2ORegressionModel: drf
Model Key:  DRF_model_R_1522509496627_19
Model Summary:
  number_of_trees number_of_internal_trees model_size_in_bytes min_depth max_depth mean_depth min_leaves max_leaves mean_leaves
1              50                       50               16548         6         7    6.90000         16         28    21.60000

H2ORegressionMetrics: drf
** Reported on training data. **
** Metrics reported on Out-Of-Bag training samples **

MSE:  197.1279
RMSE:  14.04022
MAE:  11.4405
RMSLE:  0.07444347
Mean Residual Deviance :  197.1279


H2ORegressionMetrics: drf
** Reported on validation data. **

MSE:  198.3843
RMSE:  14.0849
MAE:  10.38151
RMSLE:  0.06626213
Mean Residual Deviance :  198.3843


```

From the result of ANN we have 2 hidden layers and there are over 400 nodes which makes it impossible to train the model with the given amount of data

As a result, the validation set has higher errors and it is suggested the model has memorized the patterns rather than learning

```R

H2ORegressionModel: deeplearning
Model Key:  DeepLearning_model_R_1522509496627_20
Status of Neuron Layers: predicting Mortality_Rate, regression, gaussian distribution, Quadratic loss, 44,201 weights/biases, 529.8 KB, 370 training samples, mini-batch size 1
  layer units             type dropout       l1       l2 mean_rate rate_rms momentum mean_weight weight_rms mean_bias bias_rms
1     1    18            Input  0.00 %
2     2   200 RectifierDropout 50.00 % 0.000000 0.000000  0.005742 0.003480 0.000000    0.001595   0.098558  0.497391 0.014043
3     3   200 RectifierDropout 50.00 % 0.000000 0.000000  0.015154 0.027960 0.000000   -0.000927   0.069770  0.993742 0.016470
4     4     1           Linear         0.000000 0.000000  0.000376 0.000199 0.000000    0.003967   0.097861 -0.005620 0.000000

H2ORegressionMetrics: deeplearning
** Reported on training data. **
** Metrics reported on full training frame **

MSE:  130.5378
RMSE:  11.42532
MAE:  9.314231
RMSLE:  0.06067393
Mean Residual Deviance :  130.5378


H2ORegressionMetrics: deeplearning
** Reported on validation data. **
** Metrics reported on full validation frame **

MSE:  242.1336
RMSE:  15.56064
MAE:  12.80417
RMSLE:  0.074033
Mean Residual Deviance :  242.1336



```


In conclude, we favor Random Forest for this small dataset

## Variable Importance

Continue with the model selection and feature selections, we can check the variable importance from the model we just trained

In this case, we listed the variable importance computed by Random Forest Regression model

```R
Variable Importances: (Extract with `h2o.varimp`)
=================================================

Variable Importances:
                    variable relative_importance scaled_importance percentage
1     stomach cancer: (2014)       112874.351562          1.000000   0.233193
2     ovarian cancer: (2014)        63937.492188          0.566448   0.132092
3  pancreatic cancer: (2014)        53483.789062          0.473835   0.110495
4        skin cancer: (2014)        34494.417969          0.305600   0.071264
5      breast cancer: (2014)        33791.617188          0.299374   0.069812
6       liver cancer: (2014)        31891.388672          0.282539   0.065886
7    cervical cancer: (2014)        24498.271484          0.217040   0.050612
8        lung cancer: (2014)        23561.511719          0.208741   0.048677
9      colon  cancer: (2014)        20909.589844          0.185247   0.043198
10 testicular cancer: (2014)        15674.990234          0.138871   0.032384
11     kidney cancer: (2014)        12881.189453          0.114120   0.026612
12       oral cancer: (2014)         9982.496094          0.088439   0.020623
13      brain cancer: (2014)         9459.253906          0.083803   0.019542
14    thyroid cancer: (2014)         9090.477539          0.080536   0.018781
15   prostate cancer: (2014)         8613.831055          0.076313   0.017796
16 esophageal cancer: (2014)         7718.527832          0.068382   0.015946
17    bladder cancer: (2014)         6169.616211          0.054659   0.012746
18   lymphoma cancer: (2014)         5004.554688          0.044337   0.010339

```

## Second step - time series data analysis

With the features we extracted the previous step, we carry out time series forecast for the US mortality

As we have monthly Google Trend from the top five features and the annual data from IHME, we could apply [Mixed Data Sampling Regression](https://cran.r-project.org/web/packages/midasr/midasr.pdf)

The general idea is to aggregate higher frequency data set and apply multivariate time series analysis

![Rplot.png](https://raw.githubusercontent.com/stchau4work/Near_real_time_mortality_prediction/master/docs/Methodology/Rplot.png)


<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [ ['$', '$'] ],
        displayMath: [ ['$$', '$$']]}
});
</script>

<script type="text/javascript" src="https://cdn.bootcss.com/mathjax/2.7.2/MathJax.js?config=default"></script>
