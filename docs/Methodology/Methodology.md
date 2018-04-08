<h1 align="center">Methodology</h1>

## Higher frequency data

Machine Learning is to learn patterns from data. Living in a world that is full of data points, insurance companies could leverage on this to devise better algorithms for various decision making. To illustrate this, I have first sourced the dataset from [Google trend](https://trends.google.com/trends).

Knowing that there are more channels to provide higher frequency dataset, given the time constrain, one who in the further look for more alternative dataset

![googleTrend.png](https://raw.githubusercontent.com/stchau4work/Near_real_time_mortality_prediction/master/docs/Methodology/googleTrend.png)

Google Trend provide high frequency dataset up to hourly basis and the data coverage is since 2004

## First step to identify the web query - feature extraction

We start with something simple. By taking a static screenshot at year of 2014 and with a dedicated focus on Neoplasms (one of the major causes of death), we try to relate the queries with the IHME dataset

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

In this exercise we try to play around Random Forest Regression and Neural Network to predict the Mortality Rate

In Random Forest Regression, despite it won't overfit, but it has been suggested to keep the max-features as one-third of the variables so as to reduce the bias

![randomForest.png](https://raw.githubusercontent.com/stchau4work/Near_real_time_mortality_prediction/master/docs/Methodology/randomForest.png)

In Artificial Neutral Network, there are numerous hyper parameters to turn such as the architecture of the hidden layers, activation functions and so on. We will apply stochastic gradient descent optimization method and drop out layer to prevent get stuck at the local minimum / not overfiting

![ANN.png](https://raw.githubusercontent.com/stchau4work/Near_real_time_mortality_prediction/master/docs/Methodology/ANN.png)

We can think of the total error as

$$ Errors = Bias + Variance + Irreducible error $$

Out-of-sample mean absolute error:

    Random Forest: 5.720633
              ANN: 6.549579

However, as the dataset is small and hence the above is not reliable result but for illustration purpose

From the result of random forest we observe that there are 50 trees with max_depth as 7

The errors in both training set and validation set are roughly the same, which is a good sign

```r

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

From the result of ANN we have 2 hidden layers and there are over 400 nodes which makes it impossible to train the model as the parameters are way more than the training set.

As a result the validation set has higher error than training set and it is suggested the model has memorized the pattern

```r

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


In conclude, we favor Random Forest for this small data set.

## Variable selections/Importance


## Second step - time series data

## Applying nowcasting technique to nowcast the mortality rate

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [ ['$', '$'] ],
        displayMath: [ ['$$', '$$']]}
});
</script>

<script type="text/javascript" src="https://cdn.bootcss.com/mathjax/2.7.2/MathJax.js?config=default"></script>

