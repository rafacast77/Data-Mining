---
  title: "1600389 Coursework"
author: "Rafael Castillo"
date: "30/11/2020"

---
 ### Setup
library(caret)
library(RWeka)
library(partykit)

setwd("D:/OneDrive/Desktop/MASTERS/Data Mining/Coursework")
water1 = read.csv("sampleWater1.csv", header = TRUE, stringsAsFactors = TRUE)
water2 = read.csv("sampleWater2.csv", header = TRUE, stringsAsFactors = TRUE)
testWater = read.csv("testWater.csv", header = TRUE, stringsAsFactors = TRUE)

# Water 1 dimension
dim(water1)
# Water 2 dimension
dim(water2)
# Water 3 dimension
dim(testWater)
#### Display the attributes and top 6 instances.

# water 1 preview
head(water1)
# water 2 preview
head(water2)
# testWater preview
head(testWater)

#### Summary of the datasets

# Water 1 summary
summary(water1)
# Water 2 summary
summary(water2)
# testWater summary
summary(testWater)
#### Checks for duplicates on the dataset.
# Water 1 duplicate count
nrow(water1[duplicated(water1),])
# Water 1 duplicate count
nrow(water2[duplicated(water2),])
# Water 1 duplicate count
nrow(testWater[duplicated(testWater),])
```


### Interesting Facts
#### Analysis of WBCategories
par(mfrow=c(1,3))
# Water 1 WBCategories analysis
plot(water1$WBCategory, main = "Water 1 categories", xlab = "Category", 
     ylab = "Number of Instances", col = c("green", "grey", "red", "blue"))
# Water 2 WBCategories analysis
plot(water2$WBCategory, main = "Water 2 categories", xlab = "Category", 
     ylab = "Number of Instances", col = c("green", "grey", "red", "blue"))
# testWater WBCategories analysis
plot(testWater$WBCategory, main = "Test Water categories", xlab = "Category", 
     ylab = "Number of Instances", col = c("green", "grey", "red", "blue"))


#### Analysis of NSamples
par(mfrow=c(1,3))
# Water 1 NSamples analysis
hist(water1$NSamples, main = "Water 1 Samples", xlab = "Number of Samples", 
     ylab = "Number of Instances")
# Water 2 NSamples analysis
hist(water2$NSamples, main = "Water 2 Samples", xlab = "Number of Samples", 
     ylab = "Number of Instances")
# testWater NSamples analysis
hist(testWater$NSamples, main = "Test Water Samples", xlab = "Number of Samples", 
     ylab = "Number of Instances")


#### Analysis of Mean Values
par(mfrow=c(1,3))
# Water 1 mean value analysis
hist(water1$meanValue, main = "Water 1 Mean Values", xlab = "Values", 
     ylab = "Number of Instances")
# Water 2 mean value analysis
hist(water2$meanValue, main = "Water 2 Mean Values", xlab = "Values", 
     ylab = "Number of Instances")
# testWater mean value analysis
hist(testWater$meanValue, main = "Test Water Mean Values", xlab = "Values", 
     ylab = "Number of Instances")

#### Analysis of Standard deviation.
par(mfrow=c(1,3))
# Water 1 standart deviation analysis
plot(water1$sd, water1$meanValue, main = "Water 1 Standard Deviation", 
     xlab = "Standard deviation", ylab = "Mean value")
# Water 2 standart deviation analysis
plot(water2$sd, water2$meanValue, main = "Water 2 Standard Deviation", 
     xlab = "Standard deviation", ylab = "Mean value")
# testWater standart deviation analysis
plot(testWater$sd, testWater$meanValue, main = "Test Water Standard Deviation", 
     xlab = "Standard deviation", ylab = "Mean value")


## Pre-prossesing

### Remove useless columns from datasets
water1$analysed = NULL
water1$media = NULL
water1$siteIDScheme = NULL
water2$analysed = NULL
water2$media = NULL
water2$siteIDScheme = NULL
testWater$analysed = NULL
testWater$media = NULL
testWater$siteIDScheme = NULL

### Replacing NA values
# Replacing NA values for water 1
for(i in 1:ncol(water1)){
  if(is.numeric(water1[,i])){
    water1[is.na(water1[,i]), i] = mean(water1[,i], na.rm = TRUE)
  }
}
# Replacing NA values for water 2
for(i in 1:ncol(water2)){
  if(is.numeric(water2[,i])){
    water2[is.na(water2[,i]), i] = mean(water2[,i], na.rm = TRUE)
  }
}
# Replacing NA values for testWater
for(i in 1:ncol(testWater)){
  if(is.numeric(testWater[,i])){
    testWater[is.na(testWater[,i]), i] = mean(testWater[,i], na.rm = TRUE)
  }
}

### Joining water1 with wate2
water = rbind(water1,water2)
dim(water)


## Part B

### Information gained.

# Checks the attributes that contain the higher information gained
GainRatioAttributeEval(WBCategory ~., data=water1)
GainRatioAttributeEval(WBCategory ~., data=water)

### Water 1 Classifiers
# setting seed for reproducible experiment
set.seed(34)
# Preparing training method
ctrl = trainControl(method="repeatedcv", number=10, repeats=3)

# Tree Classifier J48
J48Water1 = train(WBCategory ~., data=water1, method = "J48", metric = "Accuracy", 
                  trControl = ctrl)
# Tree classifier CART
cartWater1 = train(WBCategory ~., data= water1, method = "rpart", metric = "Accuracy", 
                   trControl = ctrl)
# Instance Base classifier KNN
knnWater1 = train(WBCategory ~., data= water1, method = "knn", metric = "Accuracy", 
                  trControl = ctrl)
### Water 1 Results
# water2 J48 run values
J48Water1
# water1 CART run values
cartWater1
# water1 KNN run values
knnWater1

# water1 J48 outputs values for final model
print(J48Water1$finalModel)
# water1 CART outputs values for final model
print(cartWater1$finalModel)
# water1 KNN outputs values for final model
print(knnWater1$finalModel)

# water1 J48 output results
print(J48Water1$results)
# water1 CART output results
print(cartWater1$results)
# water1 KNN output results
print(knnWater1$results)

### Water classifiers

# setting seed for reproducible experiment
set.seed(34)
# Preparing training method
ctrl = trainControl(method="repeatedcv", number=10, repeats=3)


# Tree classifier for water
J48Water = train(WBCategory ~., data=water,method = "J48", metric = "Accuracy",
                 trControl = ctrl)
# Tree classifier for water
cartWater = train(WBCategory ~., data=water,method = "rpart", metric = "Accuracy", 
                  trControl = ctrl)
# Instance base classifier for water
knnWater = train(WBCategory ~., data=water,method = "knn", metric = "Accuracy", 
                 trControl = ctrl)


# water1 run values
J48Water
# water run values
cartWater
# water run values
knnWater


# water outputs values for final model
print(J48Water$finalModel)
# water outputs values for final model
print(cartWater$finalModel)
# water outputs values for final model
print(knnWater$finalModel)


# water output results
print(J48Water$results)
# water output results
print(cartWater$results)
# water output results
print(knnWater$results)
```
### Comparing results for water1 and water
# collect resamples
results = resamples(list(Water1_j48=J48Water1,Water1_CART=cartWater1,
                         Water1_KNN=knnWater1, Water_j48=J48Water, 
                         Water_CART=cartWater, Water_KNN=knnWater))
# show accuracy and kappa details
results
summary(results)

# Building dotplot with different confidence levels
scales = list(x=list(relation="free"), y=list(relation= "free"))
dotplot(results, scales=scales, conf.level = 0.99)
dotplot(results, scales=scales, conf.level = 0.97)
dotplot(results, scales=scales, conf.level = 0.95)
dotplot(results, scales=scales, conf.level = 0.90)


### Testing datasets
#Testing J48
J48TestRes = predict(J48Water, newdata = testWater, type="raw")
confusionMatrix(J48TestRes, testWater$WBCategory)

#Testing CART
cartWaterTestRes = predict(cartWater, newdata = testWater, type="raw")
confusionMatrix(cartWaterTestRes, testWater$WBCategory)

#Testing KNN
knnWaterTestRes = predict(knnWater, newdata = testWater, type="raw")
confusionMatrix(knnWaterTestRes, testWater$WBCategory)

## Clustering
### Pre-processing
#binarising nominal attributes - one-hot encoding
# make a copy
noClass = water

# remove the class - it is not transformed
noClass$WBCategory = NULL

set.seed(34)
#binarise nominal attributes - one-hot encoding
binaryVars = dummyVars(~ ., data = noClass)
newWater = predict(binaryVars, newdata = noClass)

# add the class to the binarised dataset
binWater = cbind(newWater, water[1])
# check the result. 
# View(binWeather) could have been used instead but it does not appear in the knitted document.
# previewing dataset
head(binWater, 10)
### Finding ideal number of clusters
# Data standardization
pca_water = preProcess(binWater, 
                        method = c("center", "scale", "pca"))
# checking standardlised data
waterTreated = predict(pca_water, newdata = binWater)
summary(waterTreated)

set.seed(34)
# Using the Elbow approach to find the number of K to use for clsutering.
wss = (nrow(waterTreated[, -1])-1)*sum(apply(waterTreated[,-1],2,var))

for (i in 2:15) 
  wss[i] = sum(kmeans(waterTreated[, -1], centers=i, nstart=100, iter.max=1000)$withinss)
plot(1:15, wss, type="b", xlab="k= Number of Clusters", ylab="Within groups sum of squares")
```

### Clustering algorithm
set.seed(34)
# Building clusters
km = kmeans(waterTreated[, -1], 4, nstart=25, iter.max=1000)
# Plotting the different clusters
plot(waterTreated[, -1], col=km$clust, pch=16)