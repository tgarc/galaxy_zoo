################################################################################
# David Moon
# University of Texas at Austin
# April 18, 2014
# 
# Template 
################################################################################

################################################################################
# Set up workspace. Load data.
#
################################################################################
# Set Working Directory
# setwd("C:/Users/David/Desktop/GalaxyProject/ee-381v-project") #CHANGE1
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
# namestargets <- names(test.targets)
# write.csv(namestargets, file='results/namestargets.csv', quote=FALSE, row.names=TRUE) #CHANGE9
# save(namestargets, file = 'results/namestargets.RData') #CHANGE9



# Dump the galaxy ID
train.predictors <- train.predictors[,names(train.predictors)!="GalaxyID"]
test.predictors <- test.predictors[,names(test.predictors)!="GalaxyID"]

# # Replot the pair-wise correlations
# # Looks like we are generally all uncorrelated now, good
# require(corrplot)
# png('presentation/DataCorrelation.png')
# corrplot(cor(train.predictors), order="hclust")
# dev.off()

# Set whether to combine predictors with high correlation
corr.combine = TRUE
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

# # Feature Selection
# train.predictors<-train.predictors[,!(colnames(train.predictors) %in% c("bulge_prominence","ellipticity"))]
# test.predictors<-test.predictors[,!(colnames(test.predictors) %in% c("bulge_prominence","ellipticity"))]

# # Log Transform
# for (j in 1:ncol(train.predictors)) {
#   train.predictors[,j] <- log(30*train.predictors[,j]+200)
#   test.predictors[,j] <- log(30*test.predictors[,j]+200)
# }

# require(corrplot)
# png('presentation/ReducedDataCorrelation.png')
# corrplot(cor(train.predictors), order="hclust")
# dev.off()
# 
# require(corrplot)
# png('presentation/TargetsCorrelation.png')
# corrplot(cor(train.targets), order="hclust")
# dev.off()
# 
# corrplot(cor(cbind(train.targets,train.predictors),cbind(train.targets,train.predictors)), order="hclust")
# 
# for (i in ncol(train.targets)) {
#   par(mfrow=c(3,3))
#   for (j in ncol(train.predictors)) {
#     plot(train.predictors[sample(x = 1:29982, size = 3000, replace = FALSE),j],train.targets[sample(x = 1:29982, size = 3000, replace = FALSE),i], xlab=paste(names(train.predictors[,j])))
#   } 
#   mtext(paste("Correlation of ", names(train.targets[,i]), "vs. Features", sep = ''))
# }


# Rebind the datasets
# train <- cbind(train.predictors, train.targets)
# test <- cbind(test.predictors, test.targets)
# rm(train.predictors, train.targets, test.predictors, test.targets)


# ################################################################################
# # Build 'MLR & MVLR'
# #
# ################################################################################
require(stats) #CHANGE2
# require(doParallel)
# registerDoParallel()
# RMSE function
rmse <- function(predicted, true) {
  predicted <- data.frame(predicted)
  true <- data.frame(true)
  if (nrow(predicted)!=nrow(true) || ncol(predicted)!=ncol(true)) {
    stop("Error: Calculating rmse predicted and true must have the same dimensions")
  }
  sqrt(sum((predicted-true)^2)/(nrow(true)*ncol(true)))
}



# Set whether to build model on each leaf, each node, or overall tree
tf.leaf = TRUE #KNOB1
tf.node = FALSE #KNOB2
tf.tree = FALSE #KNOB3

# Set whether to sample the data and what the sample size should be
tf.sample = TRUE #KNOB4
num.sample = 3000  #KNOB5

# IF SAMPLING, set whether to perform bagging and how many models to build
tf.bagging = TRUE #KNOB6
num.model = 1 #KNOB7 # Only applies to tf.leaf models

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
    print("HERE")
    pred.train.mat <- array(data=c(0), dim=c(29982,37,num.model))
    pred.test.mat <- array(data=c(0), dim=c(29982,37,num.model))
    for (m in 1:num.model) {
      sample.idx <- sample(x = 1:29982, size = num.sample, replace = tf.bagging)
      train.predictors.sample = train.predictors[sample.idx,]
      train.targets.sample = train.targets[sample.idx,]
      for (i in 1:37) {
        # Train the MODEL
        class.idx <- i
        lm.formula <- as.formula(paste("cbind(", paste( names(train.targets.sample)[class.idx], collapse=","),") ~.",sep=" ")) #CHANGE3
        lm.model <- lm(lm.formula, data = c(subset(train.targets.sample, select = names(train.targets.sample)[class.idx]), train.predictors.sample)) #CHANGE4 
        save(lm.model, file = paste('results/modelLinearRegression',i,'.RData',sep='')) #CHANGE
        # Make predictions
        pred.train.mat[,class.idx,m] <- predict(lm.model, newdata=train.predictors) #CHANGE3
        pred.test.mat[,class.idx,m] <-  predict(lm.model, newdata=test.predictors) #CHANGE4
        # Get Coefficients
        lm.coefficients <- lm.model$coefficients[-1]
        current <- as.data.frame(t(unname(lm.coefficients)))
        names(current) <- names(lm.coefficients)
        if (i == 1) {
          current <- current
        }
        else {
          current <- rbind(current,past)
        }
        past <- current
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
    print("THERE")
    pred.train <- array(data=c(0), dim=c(29982,37))
    pred.test <- array(data=c(0), dim=c(29982,37))
    for (i in 1:37) {
      # Train the MODEL
      class.idx <- i
      lm.formula <- as.formula(paste("cbind(", paste( names(train.targets)[class.idx], collapse=","),") ~.",sep=" ")) #CHANGE3
      lm.model <- lm(lm.formula, data = c(subset(train.targets, select = names(train.targets)[class.idx]), train.predictors)) #CHANGE4 
      save(lm.model, file = paste('results/modelLinearRegression',i,'.RData',sep='')) #CHANGE
      # Make predictions
      pred.train[,class.idx] <- predict(lm.model, newdata=train.predictors) #CHANGE5
      pred.test[,class.idx] <- predict(lm.model, newdata=test.predictors) #CHANGE6
    }
    #Compute Errors
    error[1,1] <- rmse(pred.train, train.targets)
    error[1,2] <- rmse(pred.test, test.targets)
  }
# Determine whether to build models on the nodes
} else if (tf.node == TRUE && tf.leaf == FALSE && tf.tree == FALSE) {
  print("HIER")
  # Unscaled predictions (raw numbers, doesn't use DT weightings)
  colnames(error) <- c('train', 'test')
  pred.train <- array(data=c(0), dim=c(29982,37))
  pred.test <- array(data=c(0), dim=c(29982,37))
  for (i in 1:11) {
    # Get the indeces of the current target columns
    class.idx <- grep(paste('Class',i,'[.]+', sep=''), names(train.targets))
    # Train the MODEL
    lm.formula <- as.formula(paste("cbind(", paste( names(train.targets)[class.idx], collapse=","),") ~.",sep=" ")) #CHANGE3
    lm.model <- lm(lm.formula, data = c(subset(train.targets, select = names(train.targets)[class.idx]), train.predictors)) #CHANGE4 
    # Make predictions
    pred.train[,class.idx] <- predict(lm.model, newdata=train.predictors) #CHANGE5
    pred.test[,class.idx] <- predict(lm.model, newdata=data.frame(test.predictors)) #CHANGE6
  }
  #Compute Errors
  error[1,1] <- rmse(pred.train, train.targets)
  error[1,2] <- rmse(pred.test, test.targets)
# Determine whether to build models on the nodes
} else if (tf.tree == TRUE && tf.node == FALSE && tf.leaf == FALSE) {
  print("ZEYER")
  # Unscaled predictions (raw numbers, doesn't use DT weightings)
  colnames(error) <- c('train', 'test')
  # Get the indeces of the current target columns
  class.idx <- grep('Class', names(train.targets))
  # Train the MODEL
  lm.formula <- as.formula(paste("cbind(", paste( names(train.targets)[class.idx], collapse=","),") ~.",sep=" ")) #CHANGE3
  lm.model <- lm(lm.formula, data = c(subset(train.targets, select = names(train.targets)[class.idx]), train.predictors)) #CHANGE4 
  save(lm.model, file = paste('results/modelMultiVariateLinearRegression.RData',sep='')) #CHANGE
  # Make predictions and compute errors
  pred.train <- predict(lm.model, newdata=train.predictors) #CHANGE5
  error[1,1] <- rmse(pred.train, train.targets[,class.idx])
  pred.test <- predict(lm.model, newdata=test.predictors) #CHANGE6
  error[1,2] <- rmse(pred.test, test.targets[,class.idx])
  # Get Coefficients
  lm.coefficients <- lm.model$coefficients[-1]
  current <- as.data.frame(t(unname(lm.coefficients)))
  names(current) <- names(lm.coefficients)
} else {
  print('Please check the variables tf.leaf, tf.node, and tf.tree and make sure only one of them is selected.')
}

#Produce Heatmap of coefficients
require(gplots)
require(MASS)
load('results//namestargets.RData')
rownames(current)<-namestargets
scale<-apply(abs(data.matrix(current)),1,max)*diag(37)
norm.current <-ginv(scale)%*%abs(data.matrix(current))
rownames(norm.current)<-namestargets
RF_heatmap <- heatmap.2(t(data.matrix(norm.current)), Rowv=NA, Colv=NA, dendrogram = "none", col = cm.colors(256), trace = "none")

# write.csv(pred.train, file='results/trainPredictionMultivariateLinearRegressionLeavesSample3000.csv', quote=FALSE, row.names=TRUE) #CHANGE7
# save(pred.train, file = 'results/trainPredictionMultivariateLinearRegressionLeavesSample3000.RData') #CHANGE7
# write.csv(pred.test, file='results/testPredictionMultivariateLinearRegressionLeavesSample3000.csv', quote=FALSE, row.names=TRUE) #CHANGE8
# save(pred.test, file = 'results/testPredictionMultivariateLinearRegressionLeavesSample3000.RData') #CHANGE8
# write.csv(error, file='results/errorsMultivariateLinearRegressionLeavesSample3000.csv', quote=FALSE, row.names=TRUE) #CHANGE9
# save(error, file = 'results/errorsMultivariateLinearRegressionLeavesSample3000.RData') #CHANGE9
