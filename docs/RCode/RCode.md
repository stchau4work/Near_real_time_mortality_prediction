<h1 align="center">R Code</h1>

```R
setwd('/Users/steven/Documents/myWorks/dataMining/R/Prudential')

lapply(c('data.table','ggplot2','h2o','ggthemes','midasr'), require, character.only=T)

# data from Google Trend
g<-fread('data/google/cancer.csv')

# data from IHME
m<-fread('data/mortality/2014.csv')

# joining the dataset
df <- merge(x=g, y=m, by.x="Region", by.y="Location", all.y=T )

# NA actions
df <- replace(df, is.na(g), 0)

mae <- function(y, y_hat ){
  mean(abs(y - y_hat)/y*100)
}

# Machine Learning using H2o framework
h2o.init()
set.seed(-1)
df.h2o <- as.h2o(df)

splits <- h2o.splitFrame(
  df.h2o,
  c(0.8,0.1),
  seed=1234
)

# splitting training-dev-test set
train <- h2o.assign(splits[[1]], "train.hex")
valid <- h2o.assign(splits[[2]], "valid.hex")
test <- h2o.assign(splits[[3]], "test.hex")

#model 1
rf_fit <- h2o.randomForest(
        training_frame = train,
        validation_frame = valid,
        x=1:20,
        y=21,
        max_depth = length(names(df))/3,
        score_each_iteration = T)

summary(rf_fit)

y_hat <- h2o.predict(rf_fit, newdata=test[-21])

#6.813934
mae(test[21],y_hat)

#model 2
ann_fit <- h2o.deeplearning(
  training_frame = train,
  validation_frame = valid,
  x=1:20,
  y=21,
  hidden = c(50,50),
  activation = "RectifierWithDropout",
  score_each_iteration = T
)

summary(ann_fit)

y_hat <- h2o.predict(ann_fit, newdata=test[-21])

#4.387634
mae(test[21],y_hat)


####### time series analysis
# data from Google Trend
t <-fread('data/google/timeseries.csv')
t <-t[,Month:=as.yearmon(Month, "%Y-%m")]
# convert to long form
tm <- melt(t, id.vars= 'Month')

# mannually adding the IHME dataset from 2005-2014
u<- data.table(Month=c(as.yearmon("2005-12", "%Y-%m"),as.yearmon("2010-12", "%Y-%m"),as.yearmon("2014-12", "%Y-%m")),
              variable = replicate(3, "Mortality_rate"),
              value = c(4.44*20,3.75*20,3.32*20)
              )

#joing the tables
tu <- rbindlist(list(tm,u))

# Visualization
ggplot(tu,aes(x=Month, y=value, color=variable)) +
  geom_point() +
  geom_line() +
  ggtitle("Annual US Mortality Rate vs Googel Monthly Trend") +
  theme_economist() +
  scale_colour_economist()

# midasr
 x1 <-ts(t$`breast cancer`, start=c(2005,1),frequency = 12)
 y <- ts(u$value[1:2], start=2005, end=2010, frequency = 1/5)
 mr <- midas_r (y ~ mls(x1,0:59,60) , start = list(x1 = rep(0, 2)))

```

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [ ['$', '$'] ],
        displayMath: [ ['$$', '$$']]}
});
</script>

<script type="text/javascript" src="https://cdn.bootcss.com/mathjax/2.7.2/MathJax.js?config=default"></script>