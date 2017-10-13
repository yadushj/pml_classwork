---
title: "Predicting Efficient Barbell Lifts"
author: "Joe Yadush"
date: "October 12, 2017"
output: html_document
---
##Executive Summary
This excercise will use R and some basic machine learning algorithms to predict how well barbell lifts were accomplished from a set of 20 test records. The data is grouped into 5 classifications with the 'A' class being the group of roll, pitch, and yaw measurements from wearable recording devices that define a correct executement of the excercise. For more information on the classifications, data, and collection methods please refer to the Human Actvity Recognition paper which was the source for this assignment.


**Paper citation:**
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.


##Getting and Cleaning the Data
We will download the data from the aforementioned site

```r
library(randomForest)
library(caret)
library(rattle)
library(gbm)

if (!file.exists("pml-training.csv")) {
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "pml-training.csv")
}
if (!file.exists("pml-testing.csv")) {
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", "pml-testing.csv")
}
```
Using R, read the data into data frames, identifying missing data

```r
df_training_data <- read.csv("pml-training.csv", header=TRUE, na.strings=c("NA", "#DIV/0!", ""))
df_testing_data <- read.csv("pml-testing.csv", header=TRUE, na.strings=c("NA", "#DIV/0!", ""))
```
Clean out the records with missing data

```r
df_training_data <- df_training_data[, colSums(is.na(df_training_data)) == 0]
df_testing_data  <- df_testing_data[, colSums(is.na(df_testing_data)) == 0]
```
Now we will remove the first seven columns that are essentially row identifiers with no relevance

```r
df_training_data <- df_training_data[, -c(1:7)]
df_testing_data  <- df_testing_data[, -c(1:7)]
```
Examine data for any zero covariates

```r
nsv <- nearZeroVar(df_training_data, saveMetrics = TRUE)
nsv <- nearZeroVar(df_testing_data, saveMetrics = TRUE)
```
For **cross-validation** approach we will split the training set into a train and test
while leaving original test set untouched for unbiased measurement

```r
inTrain <- createDataPartition(df_training_data$classe, p = 0.7, list = FALSE)
train <- df_training_data[inTrain, ]
test  <- df_training_data[-inTrain, ]
```
##Model Selection
Attempting three different models to examine for accuracy learned from the course.
Decision Tree (Dendograms), Random Forests, and Boosting

Dendogram

```r
set.seed(9798)
treeFit <- train(classe ~., method = "rpart", data = train)
fancyRpartPlot(treeFit$finalModel)
```

![plot of chunk unnamed-chunk-15](figure/unnamed-chunk-15-1.png)

```r
predictTree <- predict(treeFit, test)
matrixTree <- confusionMatrix(predictTree, test$classe)
matrixTree
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1526  465  455  419  141
##          B   30  374   32  161  138
##          C  114  300  539  384  312
##          D    0    0    0    0    0
##          E    4    0    0    0  491
## 
## Overall Statistics
##                                          
##                Accuracy : 0.4979         
##                  95% CI : (0.485, 0.5107)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.3447         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9116  0.32836  0.52534   0.0000  0.45379
## Specificity            0.6485  0.92394  0.77156   1.0000  0.99917
## Pos Pred Value         0.5077  0.50884  0.32686      NaN  0.99192
## Neg Pred Value         0.9486  0.85146  0.88503   0.8362  0.89035
## Prevalence             0.2845  0.19354  0.17434   0.1638  0.18386
## Detection Rate         0.2593  0.06355  0.09159   0.0000  0.08343
## Detection Prevalence   0.5108  0.12489  0.28020   0.0000  0.08411
## Balanced Accuracy      0.7801  0.62615  0.64845   0.5000  0.72648
```

Random Forest

```r
set.seed(9798)
##forestFit <- train(classe ~., data = train, method = "rf", prox = TRUE)
forestFit <- randomForest(classe~., data = train)
predictForest <- predict(forestFit, test)
matrixForest <- confusionMatrix(predictForest, test$classe)
matrixForest
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    2    0    0    0
##          B    1 1137    7    0    0
##          C    0    0 1018   11    0
##          D    0    0    1  950    0
##          E    1    0    0    3 1082
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9956          
##                  95% CI : (0.9935, 0.9971)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9944          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9988   0.9982   0.9922   0.9855   1.0000
## Specificity            0.9995   0.9983   0.9977   0.9998   0.9992
## Pos Pred Value         0.9988   0.9930   0.9893   0.9989   0.9963
## Neg Pred Value         0.9995   0.9996   0.9984   0.9972   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2841   0.1932   0.1730   0.1614   0.1839
## Detection Prevalence   0.2845   0.1946   0.1749   0.1616   0.1845
## Balanced Accuracy      0.9992   0.9983   0.9950   0.9926   0.9996
```

Boosting with Regression Trees

```r
set.seed(9798)
gbmFit <- train(classe ~., data = train, method = "gbm", verbose = FALSE)
gbmPredict <- predict(gbmFit, newdata = test)
gbmMatrix <- confusionMatrix(gbmPredict, test$classe)
gbmMatrix
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1647   34    0    0    1
##          B   23 1072   23    2   12
##          C    2   32  990   28   12
##          D    2    0   11  925   10
##          E    0    1    2    9 1047
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9653          
##                  95% CI : (0.9603, 0.9699)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9561          
##  Mcnemar's Test P-Value : 0.0001514       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9839   0.9412   0.9649   0.9595   0.9677
## Specificity            0.9917   0.9874   0.9848   0.9953   0.9975
## Pos Pred Value         0.9792   0.9470   0.9305   0.9757   0.9887
## Neg Pred Value         0.9936   0.9859   0.9925   0.9921   0.9927
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2799   0.1822   0.1682   0.1572   0.1779
## Detection Prevalence   0.2858   0.1924   0.1808   0.1611   0.1799
## Balanced Accuracy      0.9878   0.9643   0.9748   0.9774   0.9826
```
##Conclusion and Prediction Results
The _Decision Tree_ was excellant for a quick visual to give an indication of multiple variables, but had left a lot of room for uncertainty and the accuracy was very low. The _Boosting_, which is very popular and takes resampling to a good level did produce a much better result, but the _Random Forest_ model seemed to provide the most accurate with an **accuracy of 99** and an **out of sample error of ** measurements with which to apply against the final test set.
The results applied against the Random Forest are shown below:


```r
finalPrediction <- predict(forestFit, df_testing_data)
finalPrediction
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
