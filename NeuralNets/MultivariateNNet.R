################################################################################
# Dylan Anderson
# University of Texas at Austin
# May 6, 2014
# 
# This script explores multivariate Neural Networks applied to the dataset 
# (transformed data). You must have first run preProcessing.R to generate the
# requisite data files.
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
n.folds <- 5
folds <- matrix(idx, ncol=n.folds)

# Tuning parameters
tuning.parameters <- expand.grid(decay=c(0.5, 1), size=c(11,15,20,25))

# CV Tuning
best.error = 1e10
best.idx = 0
for (i in 1:nrow(tuning.parameters)) {
  params <- tuning.parameters[i,]
  train.rmse <- foreach(j=1:n.folds, .combine=cbind) %dopar% {
    df <- train[-folds[,j],]
    model <- nnet(x=train.predictors, y=train.targets, data=df, 
                  size=params$size, maxit=2500, decay=params$decay, 
                  MaxNWts=1500)
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
              size=best.parameters$size, maxit=2500, decay=best.parameters$decay,
              MaxNWts=1500)

# Import the analysis functions
require(ggplot2)
# Plotting function: this function is adapted from code provided here:
# https://gist.githubusercontent.com/fawda123/7471137/raw/c720af2cea5f312717f020a09946800d55b8f45b/nnet_plot_update.r
source('nnet_plot_fun.r')
# Relative predictor importance: this function is adapted from code provided here:
# https://gist.githubusercontent.com/fawda123/6206737/raw/2e1bc9cbc48d1a56d2a79dd1d33f414213f5f1b1/gar_fun.r
source('gar_fun.r')

# Predictor importance
targets <- names(train.transf)[grep('Class', names(train.transf))]
wts <- list()
for (i in 1:length(targets)) {
  wts[[i]] <- as.data.frame(t(gar.fun(out.var=targets[i], model, wts.only=T)$rel.imp)) 
}

wts <- melt(wts)
targets <- gsub("Class","", targets)

ggplot(wts, aes(x=L1, y=value, fill=variable)) + 
  geom_density(alpha=0.3, stat="identity") + 
  labs(title='Neural Net Relative Factor Importance', x='Target', y='Relative Importance')+
  scale_x_discrete(labels=targets) + 
  theme(title=element_text(size=24), 
        axis.text.x=element_text(size=15, angle=90, hjust=1, vjust=0.5),
        axis.text.y=element_blank(), legend.title=element_blank(), 
        legend.text=element_text(size=16))
ggsave(filename="results/NNetImportanceFactors.png")


# Plot the model
outnodes <- T
pdf(file='results/MultivariateNN.pdf', width=12, height=12)
plot.nnet(model, circle.col='brown', pos.colo='darkgreen', 
          neg.col='darkblue', alpha.val=0.7, all.out=outnodes, rel.rsc=10)
dev.off()

################################################################################
# Create bagged ensemble and average for prediction
# 
################################################################################
n.models <- 15

nn.models <- foreach(i=1:n.models) %dopar% {
  # bootstrapped sample
  idx <- sample(1:nrow(train.predictors), size=nrow(train.predictors), 
                replace=TRUE)
  df <- train[idx,]
  model <- nnet(x=train.predictors, y=train.targets, data=df,
                size=best.parameters$size, maxit=2500,
                decay=best.parameters$decay, MaxNWts=1500)
  # Decrease size (don't keep internal data)
  model$fitted.values <- 0
  model$residuals <- 0
  return(model)
}

# Create predictions
nn.train.predictions <- foreach(model=nn.models) %dopar% {
  pred <- predict(model, newdata=train)
}
nn.train.predictions <- Reduce("+", nn.train.predictions)/
  length(nn.train.predictions)

nn.test.predictions <- foreach(model=nn.models) %dopar% {
  pred <- predict(model, newdata=test)
}
nn.test.predictions <- Reduce("+", nn.test.predictions)/
  length(nn.test.predictions)

# Compute errors
nn.train.rmse <- rmse(nn.train.predictions, train.targets)
nn.test.rmse <- rmse(nn.test.predictions, test.targets)

################################################################################
# Use bag of little bootstraps to generate prediction error bars
# 
################################################################################
n.bags <- 10
idx <- sample(1:nrow(train.predictors)) # random shuffle
bags <- matrix(idx, ncol=n.bags)

train.rmse <- rep(0, n.bags)
test.rmse <- rep(0, n.bags)

for (j in 1:n.bags) {
  # Generate the bag
  bag <- train[bags[,j],]
  idx <- sample(1:nrow(bag), size=nrow(train), replace=TRUE)
  bag <- bag[idx,]
  
  models <- foreach(i=1:n.models) %dopar% {
    # bootstrapped sample
    idx <- sample(1:nrow(train.predictors), size=nrow(bag), 
                  replace=TRUE)
    df <- bag[idx,]
    model <- nnet(x=train.predictors, y=train.targets, data=bag,
                  size=best.parameters$size, maxit=2500,
                  decay=best.parameters$decay, MaxNWts=1500)
    # Decrease size (don't keep internal data)
    model$fitted.values <- 0
    model$residuals <- 0
    return(model)
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
nn.train.bagged.rmse <- mean(train.rmse)
nn.train.bagged.sd <- sd(train.rmse)
nn.test.bagged.rmse <- mean(test.rmse)
nn.test.bagged.sd <- sd(test.rmse)

# Save 
save(nn.models, nn.train.predictions, nn.test.predictions, nn.train.rmse, 
     nn.test.rmse, nn.test.bagged.rmse, nn.test.bagged.sd, nn.train.bagged.rmse, 
     nn.train.bagged.sd, best.parameters, file="results/MultivariateNNet.RData")
