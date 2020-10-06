---
title: "Practical Machine Learning Course Project"
author: "Tina"
date: "05/10/2020"
output: html_document
---

```{r setup, include=FALSE}
library(knitr)
knitr::opts_chunk$set(echo = TRUE)
```

## Project Outline  

Using tracking devices, it is now possible to collect a large amount of data about personal activity. This project uses data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants in order to predict the manner in which they did the exercise (barbell lifts correctly and incorrectly in 5 different ways).

A Random Forest model was used, which had 99.2% accuracy and an out-of-sample error of 0.8%.

***

## Loading the data

The training data for this project are available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv) 

The test data are available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv) 

```{r, echo = TRUE}
if(!file.exists("pml-training.csv"))
{
  download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "pml-training.csv", method = 'curl')
}
dataset <- read.csv("pml-training.csv", na.strings = c("NA", ""))
if(!file.exists("pml-testing.csv"))
{
  download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", "pml-testing.csv", method = 'curl')
}
validation <- read.csv("pml-testing.csv")
```

***

## Data preparation 

First we load the packages needed for analysis.

```{r, echo = TRUE, message = FALSE, warning = FALSE}
library(caret)
library(randomForest)
library(e1071)
library(rattle)
library(rpart)
```

We then set a seed for reproducibility:

```{r, echo = TRUE}
set.seed(17)
```

Next, we need to split the training set into two - a smaller training set and a test set. This splitting will also allow us to compute the out-of-sample errors.

```{r, echo = TRUE}
inTrain = createDataPartition(y=dataset$classe, p=0.7, list=FALSE)
training = dataset[inTrain,]
testing = dataset[-inTrain,]
```

Looking at the original training data, it is possible to see lots of NA entries. 

```{r, echo = TRUE, results = "hide"}
head(dataset)
```

These can be removed, along with variables that are not useful for these analyses.

Training subset:
```{r, echo = TRUE}
# Make a vector of all the columns and the number of NA entries
naColumns = sapply(training, function(x) {sum(is.na(x))}) 
columnsWithNA = names(naColumns[naColumns > 0])  #Vector with all the columns that has NA values
training = training[, !names(training) %in% columnsWithNA] # Remove those columns from the training set

# Remove unnecessary columns (the first 7 columns)
training <- training[, !names(training) %in% c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "new_window", "num_window")]
```

Testing subset:

```{r, echo = TRUE}
# Make a vector of all the columns and the number of NA entries
naColumns = sapply(testing, function(x) {sum(is.na(x))}) 
columnsWithNA = names(naColumns[naColumns > 0]) # Vector with all the columns that has NA values
testing = testing[, !names(testing) %in% columnsWithNA] # Remove those columns from the testing set

# Remove unnecessary columns
testing <- testing[, !names(testing) %in% c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "new_window", "num_window")]
```

Validation set:

```{r, echo = TRUE}
# Make a vector of all the columns and the number of NA entries
naColumns = sapply(validation, function(x) {sum(is.na(x))}) 
columnsWithNA = names(naColumns[naColumns > 0]) # Vector with all the columns that has NA values
validation = validation[, !names(validation) %in% columnsWithNA] # Remove those columns from the validation set

# Remove unnecessary columns 
validation <- validation[, !names(validation) %in% c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "new_window", "num_window")]
```

***

## Model building

First, a dendogram is created using a classification tree.

```{r, echo = TRUE}
classTree <- rpart(classe ~ ., data=training, method="class")
fancyRpartPlot(classTree)
```

This is then validated on the testing subset and the accuracy of the model is assessed.

```{r, echo = TRUE}
predTree <- predict(classTree, testing, type = "class")
cmtree <- confusionMatrix(predTree, as.factor(testing$classe))
cmtree
```
The accuracy is 73.9%, thus the predicted accuracy for the out-of-sample error is 26.1%, which is considerable and a different model should be considered.

Next, a Random Forest model is used to assess performance for prediction. The model is fitted on training and the "train" function uses 3-fold cross-validation to select optimal tuning parameters for the model.

``` {r, echo = TRUE}
# instruct train to use 3-fold CV to select optimal tuning parameters
fitControl <- trainControl(method="cv", number=3, verboseIter=F)

# fit model on training subset
fit <- train(classe ~ ., data=training, method="rf", trControl=fitControl)

# check parameters
fit$finalModel
```

This model has 500 trees and 27 variables at each split.

***

## Model cross-evaluation

The fitted model is used to predict the variable "classe" in the testing subset and the "confusionMatrix" function is used to compare the accuracy of the prediction.

``` {r, echo = TRUE}
# use model to predict classe in testing subset
pred <- predict(fit, newdata=testing)

# show confusion matrix to get estimate of out-of-sample error
coMa <- confusionMatrix(pred, as.factor(testing$classe))
acc <- coMa$overall["Accuracy"]
acc
```

```{r, echo = TRUE}
plot(fit)
```

The accuracy is 99.2%, thus the predicted accuracy for the out-of-sample error is 0.8% and we can proceed with the Random Forest model for prediction.

***

## Model re-fitting

Before predicting on the validation test set, we can look at the model in the full training set.

``` {r, echo = TRUE}
# Make a vector of all the columns and the number of NA entries
naColumns = sapply(dataset, function(x) {sum(is.na(x))}) 
columnsWithNA = names(naColumns[naColumns > 0])  #Vector with all the columns that has NA values
dataset = dataset[, !names(dataset) %in% columnsWithNA] # Remove those columns from the full dataset

# Remove unnecessary columns (the first 7 columns)
dataset <- dataset[, !names(dataset) %in% c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "new_window", "num_window")]

# re-fit model using full training set
fitallControl <- trainControl(method="cv", number=3, verboseIter=F)
fitall <- train(classe ~ ., data=dataset, method="rf", trControl=fitControl)
fitall$finalModel
```

***

## Predicting validation testing data

Finally, we use the model fit on the full training set to predict the label for the observations in the validation testing set.

```{r, echo = TRUE}
# predict on validation set
preds <- predict(fitall, newdata=validation)

# convert predictions to character vector
preds <- as.character(preds)

# create function to write predictions to files
pred_write_files <- function(x) {
    n <- length(x)
    for(i in 1:n) {
        filename <- paste0("problem_id_", i, ".txt")
        write.table(x[i], file=filename, quote=F, row.names=F, col.names=F)
    }
}

# create prediction files to submit
pred_write_files(preds)
```

The prediction model was used to predict 20 test cases. The results of these are:

B A B A A E D B A A B C B A E E A B B B



