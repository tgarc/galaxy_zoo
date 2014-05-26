################################################################################
# Dylan Anderson
# University of Texas at Austin
# May 6, 2014
# 
# This script trains a gradient boosting machine model. Parameters are selected
# via cross validation. You must first run preProcessing.R to generate the 
# requisite data files before running this script.
################################################################################

################################################################################
# Set up workspace. Load data.
#
################################################################################
# Set up the workspace
rm(list = ls())
# Get the startup par settings
.pardefault <- par(no.readonly = T)

load('results//ProcessedData.RData')

# Using transformed data
rm(train, test)

################################################################################
# Examine correlations between predictors 
#
################################################################################
# require(corrplot)

targetIdx <- grep('Class', names(train.transf))

# Chop up the data frames
train.predictors <- train.transf[,-targetIdx]
train.targets <- train.transf[, targetIdx]
test.predictors <- test.transf[,-targetIdx]
test.targets <- test.transf[, targetIdx]

# Plot the pair-wise correlations
# corrplot(cor(train.predictors), order="hclust")

# As seen in the above plot, the different channel measurements for 
# some of the features are highly self correlated. We will just average these 
# values across the channels.
variables <- c('bulge_presence', 'bulge_prominence', 'area', 'm2', 'm1',
               'ellipticity', 'disc_dev')
for (variable in variables) {
  idx <- grep(variable, names(train.predictors))
  train.predictors[,variable] <- apply(train.predictors[,idx], 1, mean)
  train.predictors[,idx] <- list(NULL)  
  
  test.predictors[,variable] <- apply(test.predictors[,idx], 1, mean)
  test.predictors[,idx] <- list(NULL)  
}

# Replot the pair-wise correlations
# Looks like we are generally all uncorrelated now, good
# corrplot(cor(train.predictors), order="hclust")

# Dump the galaxy ID (we don't need it for our assessment since we aligned the
# training and testing data in preprocessing)
train.predictors <- train.predictors[,names(train.predictors)!="GalaxyID"]
test.predictors <- test.predictors[,names(test.predictors)!="GalaxyID"]

# Rebind the datasets
train <- cbind(train.predictors, train.targets)
test <- cbind(test.predictors, test.targets)


################################################################################
# Select model parameters with cross validation
# 
################################################################################
install.packages('gbm')
install.packages('caret')
install.packages('doMC')
require(gbm)
require(caret)
require(doMC)
registerDoMC(cores=4)

# RMSE function
rmse <- function(predicted, true) {
  if (nrow(predicted)!=nrow(true) || ncol(predicted)!=ncol(true)) {
    stop("Error: Calculating rmse predicted and true must have the same dimensions")
  }
  sqrt(sum((predicted-true)^2)/(nrow(true)*ncol(true)))
}

# Set up the training parameters
param.grid <- expand.grid(n.trees=5000, interaction.depth=3, 
                          shrinkage=c(0.01, 0.1, 0.25, 0.5))

# Select params using 10-fold cross validation
trCtrl <- trainControl(method="cv", number=10, allowParallel=FALSE)

# For code testing purposes only
# train.targets <- train.targets[,1:2]

# Container for models
gbm.models <- list()

# Build an individual model for each branch of each node 
# We are essentially ignoring the DT for this method...
gbm.models <- foreach(t=1:length(names(train.targets))) %dopar% {
  # Build df for formula
  df <- train.predictors
  df$target <- train.targets[,t]
  
  # Try to reduce the amount of models generated to keep it under control
  model <- train(target~., data=df, method="gbm",
                       tuneGrid=param.grid, trControl=trCtrl, 
                       distribution="gaussian", verbose=FALSE, 
                       keep.data=FALSE)
  model <- model$finalModel
  model$fit <- 0
}

# Compute predictions
gbm.pred.train <- foreach(i=1:length(gbm.models), .combine=cbind) %dopar% {
    pred <- predict(gbm.models[[i]], newdata=train.predictors, type="response",
                    n.trees=gbm.perf(gbm.models[[i]], plot.it=FALSE, 
                                     method="OOB"))
}
gbm.pred.test <- foreach(i=1:length(gbm.models), .combine=cbind) %dopar% {
  pred <- predict(gbm.models[[i]], newdata=test.predictors, type="response",
                  n.trees=gbm.perf(gbm.models[[i]], plot.it=FALSE, 
                                   method="OOB"))
}

# Error predictions
train.rmse <- rmse(gbm.pred.train, train.targets)
test.rmse <- rmse(gbm.pred.test, test.targets)

# Save data for the reset
save(gbm.models, gbm.pred.train, gbm.pred.test, train.rmse, test.rmse, 
     file="results/GBM.RData")
