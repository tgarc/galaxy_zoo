################################################################################
# David Moon
# University of Texas at Austin
# April 18, 2014
# 
# RandomForest
################################################################################

################################################################################
# Set up workspace. Load data.
#
################################################################################
# Set Working Directory
setwd("C:/Users/David/Dropbox/ee-381v-project/") #CHANGE1
# Set up the workspace
rm(list = ls())
# Get the startup par settings
.pardefault <- par(no.readonly = T)

load('results//ProcessedData.RData')

# Using transformed data
rm(train, test)

targetIdx <- grep('Class', names(train.transf))

# Chop up the data frames
train.predictors <- train.transf[,-targetIdx]
train.targets <- train.transf[, targetIdx]
test.predictors <- test.transf[,-targetIdx]
test.targets <- test.transf[, targetIdx]

# Set whether to combine predictors with high correlation
corr.combine = FALSE
if (corr.combine == TRUE) {
  variables <- c('bulge_presence', 'bulge_prominence', 'area', 'm2', 'm1', 'ellipticity', 'disc_dev')
  for (variable in variables) {
    idx <- grep(variable, names(train.predictors))
    train.predictors[,variable] <- apply(train.predictors[,idx], 1, mean)
    train.predictors[,idx] <- list(NULL)  
    
    test.predictors[,variable] <- apply(test.predictors[,idx], 1, mean)
    test.predictors[,idx] <- list(NULL)  
  }
}

# Replot the pair-wise correlations
# Looks like we are generally all uncorrelated now, good
# corrplot(cor(train.predictors), order="hclust")

# Rebind the datasets
# train <- cbind(train.predictors, train.targets)
# test <- cbind(test.predictors, test.targets)
# rm(train.predictors, train.targets, test.predictors, test.targets)


# ################################################################################
# # Build 'MODEL NAME'
# #
# ################################################################################
require(randomForest) #CHANGE2
require(caret) #CHANGE2
require(doParallel)
registerDoParallel()

# RMSE function
rmse <- function(predicted, true) {
  predicted <- data.frame(predicted)
  true <- data.frame(true)
  if (nrow(predicted)!=nrow(true) || ncol(predicted)!=ncol(true)) {
    stop("Error: Calculating rmse predicted and true must have the same dimensions")
  }
  sqrt(sum((predicted-true)^2)/(nrow(true)*ncol(true)))
}

# Dump the galaxy ID
train.predictors <- train.predictors[,names(train.predictors)!="GalaxyID"]
test.predictors <- test.predictors[,names(test.predictors)!="GalaxyID"]

# Set whether to build model on each leaf, each node, or overall tree
tf.leaf = TRUE #KNOB1
tf.node = FALSE #KNOB2
tf.tree = FALSE #KNOB3

# Set whether to sample the data and what the sample size should be
tf.sample = FALSE #KNOB4
num.sample = 3000  #KNOB5

# IF SAMPLING, set whether to perform bagging and how many models to build
tf.bagging = TRUE #KNOB6
num.model = 5 #KNOB7 # Only applies to tf.leaf models

# Set whether to scale the error
tf.errorscale = FALSE; #KNOB8

if (tf.errorscale == TRUE) {
  for (i in 1:11) {
    # Get the indeces of the current target columns  
    class.idx <- grep(paste('Class',i,'[.]+', sep=''), names(train.targets))
    # Scale the targets to be pdfs
    # Not sure why we need the transpose...
    train.targets[,class.idx] <- t(apply(train.targets[,class.idx], MARGIN=1, function(row) {
      if (sum(row) == 0) {
        return(row)
      } 
      return(row/sum(row))
    }))
    test.targets[,class.idx] <- t(apply(test.targets[,class.idx], MARGIN=1, function(row) {
      if (sum(row) == 0) {
        return(row)
      } 
      return(row/sum(row))
    }))
  }
}

# Initialize Error Vector
error <- as.data.frame(matrix(nrow=1, ncol=2))

# Determine whether to build models on the leaves
if (tf.leaf == TRUE && tf.node == FALSE && tf.tree == FALSE) {
  # Unscaled predictions (raw numbers, doesn't use DT weightings)
  colnames(error) <- c('train', 'test')
  
  # Sampling and building leaves models
  if (tf.sample == TRUE) {
    pred.train.mat <- array(data=c(0), dim=c(29982,37,num.model))
    pred.test.mat <- array(data=c(0), dim=c(29982,37,num.model))
    for (m in 1:num.model) {
      sample.idx <- sample(x = 1:29982, size = num.sample, replace = tf.bagging)
      train.predictors.sample = train.predictors[sample.idx,]
      train.targets.sample = train.targets[sample.idx,]
      for (i in 1:37) {
        # Train the MODEL
        class.idx <- i
        # Parameters for Tuning #CHANGE
        set.seed(100) #CHANGE
        require(caret)
        indx <- createFolds(train.targets.sample[,class.idx], k = 10, returnTrain = TRUE) #CHANGE
        ctrl <- trainControl(method = "cv", index = indx) #CHANGE
        mtryGrid <- data.frame(mtry = seq(1, 5)) #CHANGE
        # Tune the model using cross-validation #CHANGE
        set.seed(100)
        forest.tune <- train(x = train.predictors.sample, y = train.targets.sample[,class.idx],
                             method = "rf",
                             tuneGrid = mtryGrid,
                             ntree = 30,
                             importance = TRUE,
                             trControl = ctrl) #CHANGE
        forest.model <- forest.tune$finalModel #CHANGE
        save(forest.tune, file = paste('results/tuneRandomForest',i,'.RData',sep='')) #CHANGE
        save(forest.model, file = paste('results/modelRandomForest',i,'.RData',sep='')) #CHANGE
        # Make predictions
        pred.train.mat[,class.idx,m] <- predict(forest.model, newdata=train.predictors) #CHANGE3
        pred.test.mat[,class.idx,m] <-  predict(forest.model, newdata=test.predictors) #CHANGE4
      }
    }
    # Take the Mean of the predictions over the total sample models 
    pred.train <- apply(pred.train.mat, c(1,2), mean)
    pred.test <- apply(pred.test.mat, c(1,2), mean)
    # Compute Errors
    error[1,1] <- rmse(pred.train, train.targets)
    error[1,2] <- rmse(pred.test, test.targets)
  } 
  
  # NOT Sampling and building leaves models
  else {
    pred.train <- array(data=c(0), dim=c(29982,37))
    pred.test <- array(data=c(0), dim=c(29982,37))
    foreach (i = 1:37) %dopar% {
      # Train the MODEL
      class.idx <- i
      # Parameters for Tuning #CHANGE
      set.seed(100) #CHANGE
      require(caret)
      indx <- createFolds(train.targets[,class.idx], k = 3, returnTrain = TRUE) #CHANGE
      ctrl <- trainControl(method = "cv", index = indx) #CHANGE
      mtryGrid <- data.frame(mtry = seq(1, 7)) #CHANGE
      # Tune the model using cross-validation #CHANGE
      set.seed(100)
      forest.tune <- train(x = train.predictors, y = train.targets[,class.idx],
                           method = "rf",
                           tuneGrid = mtryGrid,
                           ntree = 30,
                           importance = TRUE,
                           trControl = ctrl) #CHANGE
      forest.model <- forest.tune$finalModel #CHANGE
      save(forest.tune, file = paste('results/tuneRandomForest',i,'.RData',sep='')) #CHANGE
      save(forest.model, file = paste('results/modelRandomForest',i,'.RData',sep='')) #CHANGE
      # Make predictions
      pred.train[,class.idx] <- predict(forest.model, newdata=train.predictors) #CHANGE5
      pred.test[,class.idx] <- predict(forest.model, newdata=test.predictors) #CHANGE6
      print(i)
    }
    #Compute Errors
    error[1,1] <- rmse(pred.train, train.targets)
    error[1,2] <- rmse(pred.test, test.targets)
  }
  # Determine whether to build models on the nodes
} else if (tf.node == TRUE && tf.leaf == FALSE && tf.tree == FALSE) {
  print('Cannot Build Random Forest Over Nodes')    
  # Determine whether to build models on the tree
} else if (tf.tree == TRUE && tf.node == FALSE && tf.leaf == FALSE) {
  print('Cannot Build Random Forest Over Tree')
} else {
  print('Please check the variables tf.leaf, tf.node, and tf.tree and make sure only one of them is selected.')
}

write.csv(pred.train, file='results/trainPredictionRandomForest.csv', quote=FALSE, row.names=TRUE) #CHANGE7
save(pred.train, file = 'results/trainPredictionRandomForest.RData') #CHANGE7
write.csv(pred.test, file='results/testPredictionRandomForest.csv', quote=FALSE, row.names=TRUE) #CHANGE8
save(pred.test, file = 'results/testPredictionRandomForest.RData') #CHANGE8
write.csv(error, file='results/errorsRandomForest.csv', quote=FALSE, row.names=TRUE) #CHANGE9
save(error, file = 'results/errorsRandomForest.RData') #CHANGE9
