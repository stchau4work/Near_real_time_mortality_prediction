<h1 align="center">Methodology</h1>

## Higher frequency data

Machine Learning is to learn patterns from data. Living in a world that is full of data points, insurance companies could leaverage on this to devise better algorthims for various decision making. To illustrustrate this, I have first sourced the dataset from [Google trend](https://trends.google.com/trends).

Knowing that there are more channels to provide higher frequency dataset, given the time constrain, one who in the further look for more alternative dataset

![googleTrend.png](https://raw.githubusercontent.com/stchau4work/Near_real_time_mortality_prediction/master/docs/Methodology/googleTrend.png)

Google Trend provide high frequency dataset up to hourly basis and the data coverage is since 2004

## First step to identify the web query

We start with something simple. By taking a static sreenshot at year of 2014 and with a focus on Neoplasms (one of the major causes of dealth), we are trying to relate the queries to the IHME dataset.

Now, it is where the creativity comes in, we need to make some edecated guess on the web queryies and extract their trends for 51 county in the United State
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
   x14 = # ofprostate cancer queries in 2014
   x15 = # of testicular cancer queries in 2014
   x16 = # of kidney cancer queries in 2014
   x17 = # of bladder cancer queries in 2014
   x18 = # of brain cancer queries in 2014
   x19 = # of thyroid cancer queries in 2014

Then we apply machine learning models to find the relationship between these queries and the corresponding mortality rate from IHME dataset

i.e. $$ mortality rate = y = f(x1,x2, ..., x19) $$

## Choosing the right models

** 

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