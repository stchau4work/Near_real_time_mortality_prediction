<h1 align="center">Methodology</h1>

## Higher frequency data

Machine Learning is to learn patterns from data. Living in a world that is full of data points, insurance companies could leverage on this to devise better algorithms for various decision making. To illustrate this, I have first sourced the dataset from [Google trend](https://trends.google.com/trends).

Knowing that there are more channels to provide higher frequency dataset, given the time constrain, one who in the further look for more alternative dataset

![googleTrend.png](https://raw.githubusercontent.com/stchau4work/Near_real_time_mortality_prediction/master/docs/Methodology/googleTrend.png)

Google Trend provide high frequency dataset up to hourly basis and the data coverage is since 2004

## First step to identify the web query

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

## Choosing the right model & models turning

First of all, we are going to split the dataset into training, development set and testing set in ratio 8:1:1

We can think of the total error as

$$ Errors = Bias + Variance + Irreducible error $$

By comparing Out-of-sample testing from the test size, we are able to tell which models work the best

Below is a summary of the result using mean absolute error (mae)

Moreover for the model tuning, by compare the training set and development set, we are able to test the bias and overfitting

and there are some hyper-parameter turning for various models.

In Random Forest Regression, despite won't over fit, but research has suggested to keep the max-features as one-third of the variables so as to bring more variance to the each of the tree and hence less bias.

![randomForest.png](https://raw.githubusercontent.com/stchau4work/Near_real_time_mortality_prediction/master/docs/Methodology/randomForest.png)

In Artificial Neutral Network, there are numerous hyper parameters to turn such as the architecture of the hidden layers, activation functions and so on. We can also apply stochastic gradient descent optimization method to prevent get track at the local minimum and also random drop out layer to make the neural network less prone to overfit.

![ANN.png](https://raw.githubusercontent.com/stchau4work/Near_real_time_mortality_prediction/master/docs/Methodology/ANN.png)


Out-of-sample MAE

    Random Forest:
              ANN:

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

