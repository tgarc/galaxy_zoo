#######################################################################
# FindDistributions.R
# 
# Purpose: The purpose of this script is to find and plot the relative
#   probability distributions of the training data.
#
# Written by: Dylan Anderson
#   dylan.anderson@utexas.edu
#   University of Texas at Austin
#   April 03, 2014
#   Galaxy Zoo Classification Challenge
#######################################################################

#######################################################################
# Set up workspace. Load data
#
#######################################################################
# Set up the workspace
rm(list = ls())
# Get the startup par settings
.pardefault <- par(no.readonly = T)

data <- read.csv('../Data/training_solutions_rev1.csv')

#######################################################################
# Find distributions
#
#######################################################################
library('ggplot2')
library('reshape2')
# Sum down each column (except for the galaxy ID)
dataSums <- apply(data[,-1], 2, sum)

# Distribution for each main class (corresponds to a node in DT)
class.df = list()
class.distrib = list()
for (i in 1:11) {
  # Class i distribution
  class.idx <- grep(paste('Class',i,'[.]+', sep=''), names(dataSums))
  class <- dataSums[class.idx]/sum(dataSums[class.idx])
  df <- data.frame(x=names(class), y=class[])
  df$nodeID <- i
  class.df[[i]] <- df
  class.distrib[[i]] <- ggplot(df, aes(x,y)) + geom_bar(stat="identity") +
    labs(title=paste('Class ', i, ' Distributions',sep=''), y='Probability',x='')
  print(class.distrib[[i]])
}


#######################################################################
# Aggregate distributions
#
#######################################################################
# We are going to give a distribution of the leaf nodes
# The leaves are {1.3, 6.2, 8.x}
# We have to weight each leaf by all the probabilities that lead to it

# Start with the easy ones first
# 1.3
df <- class.df[[1]][3,]

# 6.2 <- 6.2 * 1.1
df <- rbind(df, class.df[[6]][2,])
df$y[2] <- df$y[2] * class.df[[1]]$y[1]

# The probabilities of class 8 are more difficult since there are
# multiple paths that lead  to this node
pathProbability <- 1 - sum(df$y)
for (i in 1:dim(class.df[[8]])[1]) {
  df <- rbind(df, class.df[[8]][i,])
  df$y[i+2] <- df$y[i+2] * pathProbability
}

leaf.distrib <- ggplot(df, aes(x,y)) + geom_bar(stat="identity") +
  labs(title='Leaf Distributions', y='Probability',x='')
print(leaf.distrib)
