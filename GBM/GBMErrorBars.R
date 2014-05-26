################################################################################
# Dylan Anderson
# University of Texas at Austin
# May 6, 2014
# 
# This script generates error bar predictions for the GBM model. You must have 
# first run preProcessing.R and GBM.R to generate the required data files.
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
# Use bag of little bootstraps to generate prediction error bars
# 
################################################################################
require(gbm)
require(caret)
## TODO get parameters

n.bags <- 10
idx <- sample(1:nrow(train.predictors)) # random shuffle
bags <- matrix(idx, ncol=n.bags)

train.rmse <- rep(0, n.bags)
test.rmse <- rep(0, n.bags)

for (j in 1:n.bags) {
  # Generate the bag
  bag <- train.predictors[bags[,j],]
  bagtargets <- train.targets[bags[,j],]
  idx <- sample(1:nrow(bag), size=nrow(train), replace=TRUE)
  bag <- bag[idx,]
  bagtargets <- bagtargets[idx,]
  
  models <- foreach(i=1:ncol(train.targets)) %dopar% {
    # Build df for formula
    df <- bag
    df$target <- bagtargets[,t]
    
    # Try to reduce the amount of models generated to keep it under control
    # TODO select params
    model <- train(target~., data=df, method="gbm",
                   tuneGrid=param.grid, trControl=trCtrl, 
                   distribution="gaussian", verbose=FALSE, 
                   keep.data=FALSE)
    model <- model$finalModel
    model$fit <- 0
    
  }
  # Create predictions
  train.predictions <- foreach(model=models) %dopar% {
    pred <- predict(model, newdata=train)
  }
  train.predictions <- Reduce("+", train.predictions)/
    length(train.predictions)
  
  test.predictions <- foreach(model=models) %dopar% {
    pred <- predict(model, newdata=test)
  }
  test.predictions <- Reduce("+", test.predictions)/
    length(test.predictions)
  
  # Compute errors
  train.rmse[j] <- rmse(train.predictions, train.targets)
  test.rmse[j] <- rmse(test.predictions, test.targets)
}

# Compute error bars
gbm.train.bagged.rmse <- mean(train.rmse)
gbm.train.bagged.sd <- sd(train.rmse)
gbm.test.bagged.rmse <- mean(test.rmse)
gbm.test.bagged.sd <- sd(test.rmse)

# Save 
save(gbm.train.bagged.rmse, gbm.train.bagged.sd, gbm.test.bagged.rmse, 
     gbm.test.bagged.sd, file="results/GBMErrorBars.RData")
