################################################################################
# Dylan Anderson
# University of Texas at Austin
# May 4, 2014
# 
# This script explores multivariate Neural Networks applied to the dataset 
# (transformed data)
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
#     Neural Nets are very senstitive to correlations between predictors. As 
#     such, we should try to reduce this before training a model. Using 
#     correlation matrices, I saw that bulge presence, bulge prominence, 
#     ellipticity, area, m1, m2, and disc_dev were all highly correlated with
#     themselves across the three channels (which makes sense). So rather than 
#     use all of them, I averaged the no. across the channels.
#     This step also greatly reduces the computational burden. As we see below
#     in the analysis plots, it appears that the majority of the model 
#     predictive power is concentrated in just a few of the predictors anyways.
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
train <- data.frame(train.predictors, train.targets)
test <- data.frame(test.predictors, test.targets)

################################################################################
# Select neural net parameters via cross validation.
#
################################################################################
require(nnet)
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

# Create CV folds
set.seed(1234)
idx <- sample(1:nrow(train.predictors)) # random shuffle
n.folds <- 3
folds <- matrix(idx, ncol=n.folds)

# Tuning parameters
tuning.parameters <- expand.grid(decay=c(1e-3, 0.5, 1), size=c(1,3,5,7,9))

# CV Tuning
best.error = 1e10
best.idx = 0
for (i in 1:nrow(tuning.parameters)) {
  params <- tuning.parameters[i,]
  train.rmse <- foreach(j=1:n.folds, .combine=cbind) %dopar% {
    df <- train[-folds[,j],]
    model <- nnet(x=train.predictors, y=train.targets, data=df, 
                  size=params$size, maxit=500, decay=params$decay)
    error <- rmse(predict(model, newdata=train.predictors[folds[,j],]),
                  train.targets[folds[,j],])
  }
  error <- mean(train.rmse)
  print(error)
  if (error < best.error) {
    best.idx <- i
    best.error <- error
  }
}

# Get the best performing model
best.parameters <- tuning.parameters[best.idx,]

################################################################################
# Generate a single best model for analysis
#   Plot the model structure and relative importance of weights for each
#   response variable.
################################################################################
# Build a "best" model on all of the available training data
model <- nnet(x=train.predictors, y=train.targets, data=train,
              size=best.parameters$size, maxit=500, decay=best.parameters$decay)
# Import the analysis functions
require(ggplot2)
require(gridExtra)
# Plotting function
source('nnet_plot_fun.r')

# Relative predictor importance
source('gar_fun.r')

# Predictor importance
for (i in 1:11) {
  class.idx <- grep(paste('Class',i,'[.]+', sep=''), names(train))
  weightplot <- list()
  for (j in 1:length(class.idx)) {
    weightplot[[j]] <- gar.fun(out.var=names(train)[class.idx[j]], model) 
  }
  print(length(class.idx))
  png(file=paste("ImportanceWeightsNode",i,".png", sep=""))
  if (length(class.idx) < 3) {
    h <- do.call(grid.arrange, c(weightplot, ncol=length(class.idx)))
  } else {
    h <- do.call(grid.arrange, c(weightplot, nrow=ceiling(length(class.idx)/3), ncol=3))    
  }
  dev.off()

  rm(weightplot)
}

# Plot the model
outnodes <- names(train)[grep('Class(\\d{1})+\\.1', names(train))]
png(file='MultivariateNN.png', width=11, height=20)
plot.nnet(model, circle.col='brown', pos.colo='darkgreen', 
          neg.col='darkblue', alpha.val=0.7, all.out=outnodes, rel.rsc=10)
dev.off()

################################################################################
# Create bagged ensemble and average for prediction
################################################################################
n.models <- 15

models <- foreach(i=1:n.models) %dopar% {
  # bootstrapped sample
  idx <- sample(1:nrow(train.predictors), size=nrow(train.predictors), 
                replace=TRUE)
  df <- train[idx,]
  model <- nnet(x=train.predictors, y=train.targets, data=df,
                size=best.parameters$size, maxit=500,
                decay=best.parameters$decay)
  # Decrease size (don't keep internal data)
  model$fitted.values <- 0
  model$residuals <- 0
  return(model)
}