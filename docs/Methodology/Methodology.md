<h1 align="center">Methodology</h1>

## Higher frequency data

** Machine Learning is to learn patterns from data. Living in a world that is full of data points, insurance companies could leaverage on this to devise better algorthims for various decision making. To illustrustrate this, I have first sourced the dataset from [Google trend](https://trends.google.com/trends).

** Knowing that there are more channels to provide higher frequency dataset, given the time constrain, one who in the further look for more alternative dataset

![googleTrend.png](https://raw.githubusercontent.com/stchau4work/Near_real_time_mortality_prediction/master/docs/Methodology/googleTrend.png)

** Google Trend provide high frequency dataset up to hourly basis and the data coverage is since 2004

## First step to identify the web query

** We start with something simple. By taking a static sreenshot at year of 2014 and with a focus on Neoplasms (one of the major causes of dealth), we are trying to relate the queries to the IHME dataset.

** Now, it is where the creativity comes in, we need to make some edecated guess on the web queryies and extract their trends for 51 county in the United State
	1. breast cancer
	2. oral cancer
	3. esophageal cancer
	4. lymphoma cancer
	5. leukemia cancer
	6. stomach cancer
	7. liver cancer
	8. pancreatkc cancer
	9. lung cancer
	10.skin cancer
	11.colon cancer
	12.cavocal cancer
	13.ovarian cancer
	14.prostate cancer
	15.testicular cancer
	16.kidney cancer
	17.bladder cancer
	18.brain cancer
	19.thyroid cancer

** With a bit of the works, we are able to apply machine learning models to find the relationship between these queries and the corresponding mortality rate from IHME dataset

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