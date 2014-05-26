# Set Working Directory
setwd("C:/Users/David/Desktop/GalaxyProject/ee-381v-project") #CHANGE1

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

# RMSE function
rmse <- function(predicted, true) {
  predicted <- data.frame(predicted)
  true <- data.frame(true)
  if (nrow(predicted)!=nrow(true) || ncol(predicted)!=ncol(true)) {
    stop("Error: Calculating rmse predicted and true must have the same dimensions")
  }
  sqrt(sum((predicted-true)^2)/(nrow(true)*ncol(true)))
}

require(randomForest)
pred.train <- array(data=c(0), dim=c(29982,37))
pred.test <- array(data=c(0), dim=c(29982,37))
print("HERE")
setwd("C:/Users/David/Desktop/RandomForest500Files") #CHANGE1
load("namestargets.RData")
for (i in 1:37) {
  class.idx <- i
  load(paste('modelRandomForest',i,'.RData',sep=''))
  # Make predictions
  pred.train[,class.idx] <- predict(forest.model, newdata=train.predictors) #CHANGE3
  pred.test[,class.idx] <-  predict(forest.model, newdata=test.predictors) #CHANGE4
  print(forest.model$mtry)
  current <- as.data.frame(t(unname(forest.model$importance[,2])))
  names(current) <- names(forest.model$importance[,2])
  
  if (i == 1) {
    current <- current
  }
  else {
    current <- rbind(current,past)
  }
  past <- current
}
require(gplots)
require(MASS)
rownames(current)<-namestargets
scale<-apply(data.matrix(current),1,max)*diag(37)
norm.current <-ginv(scale)%*%data.matrix(current)
rownames(norm.current)<-namestargets
RF_heatmap <- heatmap(t(data.matrix(norm.current)), Rowv=NA, Colv=NA, col = cm.colors(256), key=TRUE, trace = "none")
# error <- as.data.frame(matrix(nrow=1, ncol=2))
# error[1,1] <- rmse(pred.train[,1:37], train.targets[,1:37])
# error[1,2] <- rmse(pred.test[,1:37], test.targets[,1:37])

#Plot Variable Importance

forest.RSS <- round(sum((pred.train[,30]-train.targets[,30])^2)*1000)/1000
# pdf("presentation/RandomForestVI500.pdf")    
# print(dotplot(sort(forest.varimp),main=paste("Variable Importance: RSS Decrease in Predicting Target 37
#             (RSS of the Random Forest is ",forest.RSS,")",sep="")))    
# dev.off() 



# write.csv(pred.train, file='results/trainPredictionRandomForest500.csv', quote=FALSE, row.names=TRUE) #CHANGE7
# save(pred.train, file = 'results/trainPredictionRandomForest500.RData') #CHANGE7
# write.csv(pred.test, file='results/testPredictionRandomForest500.csv', quote=FALSE, row.names=TRUE) #CHANGE8
# save(pred.test, file = 'results/testPredictionRandomForest500.RData') #CHANGE8
# write.csv(error, file='results/errorsRandomForest500.csv', quote=FALSE, row.names=TRUE) #CHANGE9
# save(error, file = 'results/errorsRandomForest500.RData') #CHANGE9