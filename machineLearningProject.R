library(randomForest)
library(caret)
library(rattle)
library(gbm)


## getting the data
if (!file.exists("pml-training.csv")) {
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "pml-training.csv")
}
if (!file.exists("pml-testing.csv")) {
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", "pml-testing.csv")
}

##read data into R data frames, identifying missing data
df_training_data <- read.csv("pml-training.csv", header=TRUE, na.strings=c("NA", "#DIV/0!", ""))
df_testing_data <- read.csv("pml-testing.csv", header=TRUE, na.strings=c("NA", "#DIV/0!", ""))


##clean out the data remove columna that contain NA values
df_training_data <- df_training_data[, colSums(is.na(df_training_data)) == 0]
df_testing_data  <- df_testing_data[, colSums(is.na(df_testing_data)) == 0]

## now remove the first seven columns that are essentially row identifiers with no relevance
df_training_data <- df_training_data[, -c(1:7)]
df_testing_data  <- df_testing_data[, -c(1:7)]

##rexamine data for any zero covariates
nsv <- nearZeroVar(df_training_data, saveMetrics = TRUE)
nsv

nsv <- nearZeroVar(df_testing_data, saveMetrics = TRUE)
nsv

##nothing from nsv analysis appears to be discarded

##for cross-validation approach we will split the training set into a train and test
##leaving original test set untouched for unbiased measurement

inTrain <- createDataPartition(df_training_data$classe, p = 0.7, list = FALSE)
train <- df_training_data[inTrain, ]
test  <- df_training_data[-inTrain, ]



##attempt various models
set.seed(9798)

##dendogram
treeFit <- train(classe ~., method = "rpart", data = train)
fancyRpartPlot(treeFit$finalModel)
predictTree <- predict(treeFit, test)
matrixTree <- confusionMatrix(predictTree, test$classe)
matrixTree

##good for quick grouping but hard to understand uncertainty.

##random forest
set.seed(9798)
##forestFit <- train(classe ~., data = train, method = "rf", prox = TRUE)
forestFit <- randomForest(classe~., data = train)
predictForest <- predict(forestFit, test)
matrixForest <- confusionMatrix(predictForest, test$classe)
matrixForest
##much better accuracy ad confidence

##boosting with regression trees
set.seed(9798)
gbmFit <- train(classe ~., data = train, method = "gbm", verbose = FALSE)
gbmPredict <- predict(gbmFit, newdata = test)
gbmMatrix <- confusionMatrix(gbmPredict, test$classe)
gbmMatrix

## random forest produces best accuracy of 99%
##apply to final test set
finalPrediction <- predict(forestFit, df_testing_data)
finalPrediction