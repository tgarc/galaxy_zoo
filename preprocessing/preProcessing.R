################################################################################
# Dylan Anderson
# University of Texas at Austin
# April 10,2014
# 
# This script processes the data from the output of the image processing. It
# removes galaxies whose values contain NA's/inf's. It performs a YeoJohnson
# transform for skewness and centers scales the data. It also combines the 
# true values of the distribution with the predictors into a single data frame.
# Finally, it splits the data in half into testing and training. The outputs of
# this script are as follows:
#   - results/ProcessedData.RData
#         - train - untransformed training data
#         - test  - untransformed testing data
#         - train.transf - transformed training data
#         - test.transf  - transformed testing data
#   - results/BadGalaxyIDs.RData
#         - na.GalaxyID - galaxy IDs who had NA (or None) features
#         - inf.GalaxyID - galaxys IDs who had inf features
################################################################################

################################################################################
# Set up workspace. Load data.
# setwd("C:/Users/David/Desktop/GalaxyProject/ee-381v-project") #CHANGE1
################################################################################
# Set up the workspace
rm(list = ls())
# Get the startup par settings
.pardefault <- par(no.readonly = T)

d1 <- read.csv('results/revised1.csv')
d2 <- read.csv('results/revised2.csv')
d3 <- read.csv('results/revised3.csv')
d4 <- read.csv('results/revised4.csv')
data <- rbind(d1, d2, d3, d4)

#load('results//train.RData')
trueVal <- read.csv('Data/training_solutions_rev1.csv')

################################################################################
# Basic data cleaning. Reorders data and the true distributions
#     so that the rows correspond with each other. Removes the 
#     galaxies that have NA's and inf's from both data and trueVal.
#
################################################################################
require(stats)

# Reorder data to match the true distributions
data <- data[order(data$gid),]
trueVal <- trueVal[order(trueVal$GalaxyID),]

# Remove stupid row.names...
row.names(data) <- NULL
row.names(trueVal) <- NULL

# Remove the sample from the training set that we don't have a 
# distribution for
data <- data[-(which(data$gid!=trueVal$GalaxyID)[1]),]

# Remove NA's
na.idx <- complete.cases(data)
na.GalaxyID <- trueVal$GalaxyID[!na.idx]
data <- data[na.idx,]
trueVal <- trueVal[na.idx,]

# Remove inf's
inf.idx <- which(apply(is.infinite(as.matrix(data)), 1, any), arr.ind=TRUE, 
                 useNames=FALSE) 
inf.GalaxyID <- trueVal$GalaxyID[inf.idx]
data <- data[-inf.idx,]
trueVal <- trueVal[-inf.idx,]

# Remove some arbitrarily large numbers
large.idx <- union(union(which(data$ellipticity_i > 1e+3, arr.ind=TRUE),
                         which(data$ellipticity_g > 1e+3, arr.ind=TRUE)),
                   which(data$ellipticity_r > 1e+3, arr.ind=TRUE))
data <- data[-large.idx,]
trueVal <- trueVal[-large.idx,]

################################################################################
# Transform the data for skewness. Center and scale to zero mean/unit variance.
#     Bind the true values in with the predictors into a single data frame.
#
################################################################################
require(e1071)
require(caret)

# Delete the galaxy ID col (it will become redundant soon...)
data$gid <- NULL

# Note that the data is significantly skewed and scaled. Some predictors have 
# max vals as high as 1e15 while others have max 5e-1. See:
# print(apply(data, 2, max))

# The data is also significantly skewed. See:
# print(apply(data, 2, skewness))

# Use YeoJohnson transform for skewness because BoxCox requires strictly 
# positive predictors. Center/Scale to zero mean/unit variance.
proc <- preProcess(data, method=c("YeoJohnson", "center", "scale"))
data.transf <- predict(proc, data)

# Bind the data so true values and predictors are in the same data frame.
data.trans <- cbind(data.transf, trueVal)
data <- cbind(data, trueVal)

################################################################################
# Split the data into training and testing sets. Save the output data.
#
################################################################################
# Generate the split
train.idx <- sample(1:nrow(data), size=nrow(data)%/%2, replace=FALSE)

train <- data[train.idx,]
test <- data[-train.idx,]

train.transf <- data[train.idx,]
test.transf <- data[-train.idx,]

# Save the data so we don't always have to rerun this script...
save(train, test, train.transf, test.transf, file="results/ProcessedData.RData")

# Save the Galaxy ID's of the NA's and inf's for image processing diagnostics
save(inf.GalaxyID, na.GalaxyID, file="results/BadGalaxyIDs.RData")
