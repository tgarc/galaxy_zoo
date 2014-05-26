################################################################################
# Dylan Anderson
# University of Texas at Austin
# May 6, 2014
# 
# This script is used to generate the aggregate analysis error plots used in the
# final report.
################################################################################

################################################################################
# Set up workspace. Load data.
#
################################################################################
# Set up the workspace
rm(list = ls())
# Get the startup par settings
.pardefault <- par(no.readonly = T)

# Load the data (need the true values)
load('results//ProcessedData.RData')
targetIdx <- grep('Class', names(train.transf))
# Chop up the data frames
# For the analysis done here, we need only the true test values
test.targets <- test.transf[, targetIdx]
rm(test, train, test.transf, train.transf)

################################################################################
# Error Functions 
#
################################################################################
# RMSE function
rmse <- function(predicted, true) {
  if (nrow(predicted)!=nrow(true) || ncol(predicted)!=ncol(true)) {
    stop("Error: Calculating rmse predicted and true must have the same dimensions")
  }
  sqrt(sum((predicted-true)^2)/(nrow(true)*ncol(true)))
}

# MSE (split across nodes) - returns a vector of length 11
mse <- function(predicted, true) {
  if (nrow(predicted)!=nrow(true) || ncol(predicted)!=ncol(true)) {
    stop("Error: Calculating mse predicted and true must have the same dimensions")
  }
  error <- rep(0, 11)
  for (i in 1:11) {
    # Get the indeces of the current target columns
    class.idx <- grep(paste('Class',i,'[.]+', sep=''), names(test.targets))
    error[i] <- sum((predicted[,class.idx] - true[,class.idx])^2) /
      (nrow(true) * ncol(true))
  }
  return(error)
}

models <- list()
################################################################################
# Dylan's Models - Calculate errors for the models used (rmse and mse) 
#
################################################################################
load('results/MultivariateNNet.RData')
models[[1]] <- data.frame(mse=mse(nn.test.predictions, test.targets), 
                          rmse=rmse(nn.test.predictions, test.targets),
                          sd=nn.test.bagged.sd,
                          model="MV NNet")
rm(list=ls(pattern="nn"))
load('results/GBM.RData')
models[[2]] <- data.frame(mse=mse(gbm.pred.test, test.targets), 
                          rmse=rmse(gbm.pred.test, test.targets),
                          sd=0,
                          model="GBM")
rm(list=ls(pattern="gbm"))

################################################################################
# David's Models 
#
################################################################################
# load('results/MultipleLinearRegressionTestPredictionSample10.RData')
# models[[3]] <- data.frame(mse=mse(pred.train, test.targets), 
#                           rmse=rmse(pred.train, test.targets),
#                           sd=0,
#                           model="MLR n = 10")
load('results/MultipleLinearRegressionTestPredictionSample100.RData')
models[[3]] <- data.frame(mse=mse(pred.train, test.targets), 
                          rmse=rmse(pred.train, test.targets),
                          sd=0,
                          model="MLR n = 100")
load('results/MultivariateLinearRegressionTestPrediction.RData')
models[[4]] <- data.frame(mse=mse(pred.test, test.targets), 
                          rmse=rmse(pred.test, test.targets),
                          sd=0,
                          model="MV LR")
load('results/testPredictionRandomForest.RData')
models[[5]] <- data.frame(mse=mse(pred.test, test.targets), 
                          rmse=rmse(pred.test, test.targets),
                          sd=0,
                          model="RF 30")

load('results/RandomForestTestPrediction500.RData')
models[[6]] <- data.frame(mse=mse(pred.test, test.targets), 
                          rmse=rmse(pred.test, test.targets),
                          sd=0,
                          model="RF 500")
rm(list=ls(pattern="pred"))

################################################################################
# Tzu Ling's Models 
#
################################################################################
load('results/prediction_svmLinear.RData')
models[[7]] <- data.frame(mse=mse(pred.lin, test.targets), 
                          rmse=rmse(pred.lin, test.targets),
                          sd=0,
                          model="SVR Linear")
load('results/prediction_svmRadial.RData')
models[[8]] <- data.frame(mse=mse(pred.rad, test.targets), 
                          rmse=rmse(pred.rad, test.targets),
                          sd=0,
                          model="SVR Radial")
load('results/prediction_bestSubset.RData')
pred_bestsub <- do.call(cbind, pred_bestsub)
models[[9]] <- data.frame(mse=mse(pred_bestsub, test.targets), 
                          rmse=rmse(pred_bestsub, test.targets),
                          sd=0,
                          model="Best Subset")
load('results/prediction_lasso.RData')
models[[10]] <- data.frame(mse=mse(pred_lasso, test.targets), 
                          rmse=rmse(pred_lasso, test.targets),
                          sd=0,
                          model="MLR Lasso")
load('results/prediction_poly_low.RData')
models[[11]] <- data.frame(mse=mse(pred_poly_low, test.targets), 
                           rmse=rmse(pred_poly_low, test.targets),
                           sd=0,
                           model="Poly Reg Low")
load('results/prediction_poly_high.RData')
models[[12]] <- data.frame(mse=mse(pred_poly_high, test.targets), 
                           rmse=rmse(pred_poly_high, test.targets),
                           sd=0,
                           model="Poly Reg High")
load('results/prediction_hierachy.RData')
models[[13]] <- data.frame(mse=mse(pred_hier, test.targets), 
                           rmse=rmse(pred_hier, test.targets),
                           sd=0,
                           model="Hierarchy")
################################################################################
# Benchmarks 
#
################################################################################
bench <- list()
bench[[1]] <- data.frame(mse=rep(0,11), 
                         rmse=0.07492,
                         sd=0,
                         model="Top Contest Entry: sedielem")
bench[[2]] <- data.frame(mse=rep(0,11), 
                         rmse=0.16194,
                         sd=0,
                         model="Central Pixel Benchmark")
bench[[3]] <- data.frame(mse=rep(0,11), 
                         rmse=0.27160,
                         sd=0,
                         model="All Zeros Benchmark")

################################################################################
# Evaluation Plots 
#
################################################################################
require(ggplot2)
require(reshape)
require(grid)

# RMSE plots (with error bars)
# Reshape data for plotting
rmse.df <- melt(lapply(append(models, bench), function(x) {
  data.frame(model=x$model[1], rmse=x$rmse[1])
  }))
sd.df <- do.call(rbind, lapply(append(models, bench), function(x) {
  data.frame(model=x$model[1], ymin=x$rmse[1]-x$sd[1], ymax=x$rmse[1]+x$sd[1])
  }))
rmse <- ggplot(data=rmse.df, aes(x=model, y=value, fill=model)) + 
        geom_bar(stat="identity") + xlab("Model") + ylab("Test Set Error") +
        ggtitle("Test Set RMSE") + 
        theme(axis.text.x=element_blank(),
              axis.text.y=element_text(size=24, face="bold"),
              axis.title=element_text(size=24, face="bold"),
              title=element_text(size=24, face="bold"),
              legend.position="bottom",
              legend.title=element_blank(),
              legend.text=element_text(size=24),
              legend.key.height=unit(0.5, "in")) +
        guides(fill=guide_legend(ncol=3)) + 
  geom_errorbar(data=sd.df, 
                aes(x=model, ymin=ymin, ymax=ymax, y=NULL, fill=NULL))
print(rmse)
ggsave(rmse, file="final_report/RMSE.png", width=12, height=8, units="in")

# Reshape for csv
rmse.df$L1 <- NULL
rmse.df$variable<-NULL
rmse.df$value <- round(rmse.df$value, digits=4)
colnames(rmse.df) <- c("Model", "Test Set RMSE")
write.csv(rmse.df, file='final_report/rmse.csv', quote=F, row.names=F)

# MSE (Class relative) performance plot
names <- c("Class 1", "Class 2", "Class 3", "Class 4", "Class 5", 
           "Class 6", "Class 7", "Class 8", "Class 9", "Class 10", 
           "Class 11")
mse.df <- do.call(rbind, lapply(models, function(x) {
  df <- data.frame(model=x$model, mse=x$mse)
  df$class <- factor(names, levels=names, ordered=TRUE)
  df
  }))

# Frequency polygon
mse.densityStack <- ggplot(mse.df, aes(x=model, y=mse, group=class, fill=class)) + 
        geom_area(stat="identity", colour="black", position="stack") + 
  ylab("Test Set Mean Square Error") + ggtitle("Test Set MSE") + 
  theme(axis.title.x=element_blank(),
        axis.title.y=element_text(size=24, face="bold"),
        title=element_text(size=24, face="bold"),
        legend.title=element_blank(),
        legend.text=element_text(size=24),
        legend.key.height=unit(0.5,"in"),
        axis.text.x=element_text(size=20, face="bold", angle=60, hjust=1))
print(mse.densityStack)


ggsave(mse.densityStack, file="final_report/MSEDensityStack.png", width=12,
       height=8, units="in")

################################################################################
# Below this are different ways to display the MSE data. Note that they expect
#   a slightly different data format than the stacked density above. We felt 
#   stacked density best conveyed the data we were trying to show.
################################################################################


# # MSE (Node relative) performance plot
# colnames(mse.df) <- c("Node 1", "Node 2", "Node 3", "Node 4", "Node 5", 
#                       "Node 6", "Node 7", "Node 8", "Node 9", "Node 10", 
#                       "Node 11", "Model")
# mse.df <- melt(mse.df, id.vars="Model")
# 
# # Below are different plots of the same information - relative error contributed
# # by each node to the model test error.
# # Stacked bar plot
# mse <- ggplot(mse.df, aes(x=Model, y=value, fill=variable)) + 
#         geom_bar(stat="identity") + ylab("Test Set Mean SquareError") +
#         ggtitle("Test Set MSE") + 
#         theme(axis.text.x=element_text(size=20, face="bold", angle=60, hjust=1),
#               axis.title.x=element_blank(),
#               axis.text.y=element_text(size=24, face="bold"),
#               axis.title=element_text(size=24, face="bold"),
#               title=element_text(size=24, face="bold"),
#               legend.title=element_blank(),
#               legend.text=element_text(size=24),
#               legend.key.height=unit(0.5, "in"))
# print(mse)
# ggsave(mse, file="presentation/MSE.pdf", width=16, height=8, units="in")
# 
# # Dodged bar plot
# 
# mse.dodged <- ggplot(mse.df, aes(x=Model, y=value, fill=variable)) + 
#         geom_bar(stat="identity", position="dodge") +
#         ylab("Test Set Mean SquareError") + ggtitle("Test Set MSE") +
#         theme(axis.text.x=element_text(size=20, face="bold", angle=60, hjust=1),
#               axis.title.x=element_blank(),
#               axis.text.y=element_text(size=24, face="bold"),
#               axis.title=element_text(size=24,face="bold"),
#               title=element_text(size=24, face="bold"),
#               legend.title=element_blank(), 
#               legend.text=element_text(size=24), 
#               legend.key.height=unit(0.5, "in"))
# print(mse.dodged)
# ggsave(mse.dodged, file="presentation/MSEDodged.pdf", width=16, height=8, 
#        units="in")
# 
# # Faceting bar plots
# mse.facet <- ggplot(mse.df, aes(x=Model, y=value, fill=Model)) + 
#         geom_bar(stat="identity") + xlab("Model") + 
#   ylab("Test Set Mean SquareError") + ggtitle("Test Set MSE") +
#   facet_wrap(~ variable) + guides(fill=guide_legend(ncol=2)) + 
#   theme(axis.text.x=element_blank(),
#         axis.title.x=element_blank(),
#         axis.title.y=element_text(size=24, face="bold"),
#         title=element_text(size=24, face="bold"),
#         legend.title=element_blank(),
#         legend.text=element_text(size=24),
#         legend.key.height=unit(0.5,"in"),
#         legend.position="bottom")
# 
# print(mse.facet)
# ggsave(mse.facet, file="presentation/MSEFacet.pdf", width=16, height=8, 
#        units="in")
# 
# # Frequency polygon
# mse.densityStack <- ggplot(mse.df, aes(x=Model, y=value, fill=variable)) + 
#         geom_area(aes(y=value, fill=variable, group=variable), stat="identity",
#                   colour="black") + geom_line(aes(position='stack')) + 
#   ylab("Test Set Mean Square Error") + ggtitle("Test Set MSE") + 
#   theme(axis.title.x=element_blank(),
#         axis.title.y=element_text(size=24, face="bold"),
#         title=element_text(size=24, face="bold"),
#         legend.title=element_blank(),
#         legend.text=element_text(size=24),
#         legend.key.height=unit(0.5,"in"),
#         axis.text.x=element_text(size=20, face="bold", angle=60, hjust=1))
# print(mse.densityStack)
# 
# 
# ggsave(mse.densityStack, file="presentation/MSEDensityStack.pdf", width=16,
#        height=8, units="in")











# ################################################################################
# # David's Models - Calculate errors for the models used (rmse and mse) 
# #
# ################################################################################
# # MLR
# load('results/testPredictionMultivariateLinearRegressionTree.RData')
# rmse.df <- rbind(rmse.df,
#             data.frame(#train=rmse(gbm.pred.train, train.targets),
#                        test=rmse(pred.test, test.targets),
#                        model="Multivariate Lin Reg"))
# mse.df <- rbind(mse.df,
#                 data.frame(#train=mse(gbm.pred.train, train.targets),
#                      test=mse(pred.test, test.targets),
#                      model="Multivariate Lin Reg"))
# 
# load('results/testPredictionRandomForest.RData')
# 
# rmse.df <- rbind(rmse.df,
#             data.frame(#train=rmse(gbm.pred.train, train.targets),
#                        test=rmse(pred.test, test.targets),
#                        model="Random Forest"))
# mse.df <- rbind(mse.df,
#                 data.frame(#train=mse(gbm.pred.train, train.targets),
#                      test=mse(pred.test, test.targets),
#                      model="Random Forest"))
# 
# rf.test <- pred.test
# 
# ################################################################################
# # Tzu Ling's Models - Calculate errors for the models used (rmse and mse) 
# #
# ################################################################################
# # Trimmed outlier test data
# load('results/TzuLingTestVal.RData')
# # Best subset prediction
# load('results/prediction_bestSubset.RData')
# pred_bestsub <- as.data.frame(pred_bestsub)
# rmse.df <- rbind(rmse.df,
#             data.frame(test=rmse(pred_bestsub, y.test),
#                        model="Best feature subset MLR"))
# mse.df <- rbind(mse.df,
#                 data.frame(test=mse(pred_bestsub, y.test),
#                      model="Best feature subset MLR"))
# 
# # Linear regression
# load('results/prediction_linearRegression.RData')
# rmse.df <- rbind(rmse.df,
#             data.frame(test=rmse(pred_lin_reg, y.test),
#                        model="MLR"))
# mse.df <- rbind(mse.df,
#                 data.frame(test=mse(pred_lin_reg, y.test),
#                      model="MLR"))
# 
# # Polynomial regression
# load('results/prediction_poly_high.RData')
# load('results/prediction_poly_low.RData')
# load('results/prediction_poly_all.RData')
# 
# rmse.df <- rbind(rmse.df,
#             data.frame(test=rmse(pred_poly_high, y.test),
#                        model="MLR High Corr Interaction"))
# mse.df <- rbind(mse.df,
#                 data.frame(test=mse(pred_poly_high, y.test),
#                      model="MLR High Corr Interaction"))
# 
# rmse.df <- rbind(rmse.df,
#             data.frame(test=rmse(pred_poly_low, y.test),
#                        model="MLR Low Corr Interaction"))
# mse.df <- rbind(mse.df,
#                 data.frame(test=mse(pred_poly_low, y.test),
#                      model="MLR Low Corr Interaction"))
# 
# # rmse.df <- rbind(rmse.df,
# #             data.frame(test=rmse(pred_poly_all, y.test),
# #                        model="Polynomial Regression All"))
# # mse.df <- rbind(mse.df,
# #                 data.frame(test=mse(pred_poly_all, y.test),
# #                      model="Polynomial Regression All"))
# # LASSO Regression
# load('results/prediction_lasso.RData')
# 
# rmse.df <- rbind(rmse.df,
#             data.frame(test=rmse(pred_lasso, y.test),
#                        model="LASSO MLR"))
# mse.df <- rbind(mse.df,
#                 data.frame(test=mse(pred_lasso, y.test),
#                      model="LASSO MLR"))
# 

# # Reorder to the order models were presented
# # idx <- c(4, 5, 6, 7, 8, 9, 1, 2)
# # rmse.df <- rmse.df[idx,]
# # idx <- c(4, 5, 6, 7, 8, 1, 2)
# # mse.df <- mse.df[idx,]

# ################################################################################
# # Evaluation Plots 
# #
# ################################################################################
# require(ggplot2)
# require(reshape2)
# require(grid)
# # RMSE Performance plot
# rmse <- ggplot(rmse.df, aes(x=model, y=test, fill=model)) + 
#         geom_bar(stat="identity") + xlab("Model") + ylab("Test Set Error") +
#         ggtitle("Test Set RMSE") + 
#         theme(axis.text.x=element_blank(),
#               axis.text.y=element_text(size=24, face="bold"),
#               axis.title=element_text(size=24, face="bold"),
#               title=element_text(size=24, face="bold"),
#               legend.position="bottom",
#               legend.title=element_blank(),
#               legend.text=element_text(size=24),
#               legend.key.height=unit(0.5, "in")) +
#         guides(fill=guide_legend(ncol=3))
# print(rmse)
# ggsave(rmse, file="presentation/RMSE.pdf", width=16, height=8, units="in")
# 
# # MSE (Node relative) performance plot
# colnames(mse.df) <- c("Node 1", "Node 2", "Node 3", "Node 4", "Node 5", 
#                       "Node 6", "Node 7", "Node 8", "Node 9", "Node 10", 
#                       "Node 11", "Model")
# mse.df <- melt(mse.df, id.vars="Model")
# 
# # Below are different plots of the same information - relative error contributed
# # by each node to the model test error.
# # Stacked bar plot
# mse <- ggplot(mse.df, aes(x=Model, y=value, fill=variable)) + 
#         geom_bar(stat="identity") + ylab("Test Set Mean SquareError") +
#         ggtitle("Test Set MSE") + 
#         theme(axis.text.x=element_text(size=20, face="bold", angle=60, hjust=1),
#               axis.title.x=element_blank(),
#               axis.text.y=element_text(size=24, face="bold"),
#               axis.title=element_text(size=24, face="bold"),
#               title=element_text(size=24, face="bold"),
#               legend.title=element_blank(),
#               legend.text=element_text(size=24),
#               legend.key.height=unit(0.5, "in"))
# print(mse)
# ggsave(mse, file="presentation/MSE.pdf", width=16, height=8, units="in")
# 
# # Dodged bar plot
# 
# mse.dodged <- ggplot(mse.df, aes(x=Model, y=value, fill=variable)) + 
#         geom_bar(stat="identity", position="dodge") +
#         ylab("Test Set Mean SquareError") + ggtitle("Test Set MSE") +
#         theme(axis.text.x=element_text(size=20, face="bold", angle=60, hjust=1),
#               axis.title.x=element_blank(),
#               axis.text.y=element_text(size=24, face="bold"),
#               axis.title=element_text(size=24,face="bold"),
#               title=element_text(size=24, face="bold"),
#               legend.title=element_blank(), 
#               legend.text=element_text(size=24), 
#               legend.key.height=unit(0.5, "in"))
# print(mse.dodged)
# ggsave(mse.dodged, file="presentation/MSEDodged.pdf", width=16, height=8, 
#        units="in")
# 
# # Faceting bar plots
# mse.facet <- ggplot(mse.df, aes(x=Model, y=value, fill=Model)) + 
#         geom_bar(stat="identity") + xlab("Model") + 
#   ylab("Test Set Mean SquareError") + ggtitle("Test Set MSE") +
#   facet_wrap(~ variable) + guides(fill=guide_legend(ncol=2)) + 
#   theme(axis.text.x=element_blank(),
#         axis.title.x=element_blank(),
#         axis.title.y=element_text(size=24, face="bold"),
#         title=element_text(size=24, face="bold"),
#         legend.title=element_blank(),
#         legend.text=element_text(size=24),
#         legend.key.height=unit(0.5,"in"),
#         legend.position="bottom")
# 
# print(mse.facet)
# ggsave(mse.facet, file="presentation/MSEFacet.pdf", width=16, height=8, 
#        units="in")
# 
# # Frequency polygon
# mse.densityStack <- ggplot(mse.df, aes(x=Model, y=value, fill=variable)) + 
#         geom_area(aes(y=value, fill=variable, group=variable), stat="identity",
#                   colour="black") + geom_line(aes(position='stack')) + 
#   ylab("Test Set Mean Square Error") + ggtitle("Test Set MSE") + 
#   theme(axis.title.x=element_blank(),
#         axis.title.y=element_text(size=24, face="bold"),
#         title=element_text(size=24, face="bold"),
#         legend.title=element_blank(),
#         legend.text=element_text(size=24),
#         legend.key.height=unit(0.5,"in"),
#         axis.text.x=element_text(size=20, face="bold", angle=60, hjust=1))
# print(mse.densityStack)
# 
# 
# ggsave(mse.densityStack, file="presentation/MSEDensityStack.pdf", width=16,
#        height=8, units="in")
