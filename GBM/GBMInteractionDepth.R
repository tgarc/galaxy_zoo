################################################################################
# Dylan Anderson
# University of Texas at Austin
# May 5, 2014
# 
# This script analyzes different interaction depth levels for a gradient boosted
# machine applied to the data. It requires first running preProccessing.R to 
# generate the requisite data files.
################################################################################

################################################################################
# Set up workspace. Load data.
#
################################################################################
# Set up the workspace
rm(list = ls())
# Get the startup par settings
.pardefault <- par(no.readonly = T)

load('results/ProcessedData.RData')

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

# Remove stuff we don't need
rm(train.transf, test.transf)

################################################################################
# Build a gradient boosting machine for different levels of interaction effects
#   between predictor variables
################################################################################
install.packages('gbm')
install.packages('caret')
install.packages('doMC')
install.packages('foreach')
require(gbm)
require(caret)
require(foreach)
require(doMC)
registerDoMC(cores=4)

# RMSE function
rmse <- function(predicted, true) {
  if (nrow(predicted)!=nrow(true) || ncol(predicted)!=ncol(true)) {
    stop("Error: Calculating rmse predicted and true must have the same dimensions")
  }
  sqrt(sum((predicted-true)^2)/(nrow(true)*ncol(true)))
}

# Model params
shrinkage <- 0.05
n.trees <- 2500
interaction.depth <- c(1,2,3,4,5)
n.cores <- 4

train.pred <- as.data.frame(matrix(nrow=nrow(train.predictors), 
                                       ncol=ncol(train.targets)))
test.pred <- as.data.frame(matrix(nrow=nrow(test.predictors), 
                                   ncol=ncol(test.targets)))

train.rmse <- rep(0, length(interaction.depth))
test.rmse <- rep(0, length(interaction.depth))

for (j in 1:length(interaction.depth)) {
  gbm.models <- foreach(i=1:ncol(train.targets)) %dopar% {
    df <- train.predictors[,]
    df$target <- train.targets[,i]
    model <- gbm(target~., data=df, n.trees=n.trees, 
                 shrinkage=shrinkage,
                 interaction.depth=interaction.depth[j], 
                 distribution='gaussian')
    return(model)
  }
  train.predictions <- data.frame(foreach(i=1:ncol(train.targets), .combine=cbind) %dopar% {
    train <- predict(gbm.models[[i]], newdata=train.predictors, n.trees=n.trees)
  })
    
  test.predictions <- data.frame(foreach(i=1:ncol(train.targets), .combine=cbind) %dopar% {
    test <- predict(gbm.models[[i]], newdata=test.predictors, n.trees=n.trees)
  })
  
  train.rmse[j] <- rmse(train.predictions,train.targets)
  test.rmse[j] <- rmse(test.predictions, test.targets)
}

save(train.rmse, test.rmse, file='results/GBMInteractionEffects.RData')