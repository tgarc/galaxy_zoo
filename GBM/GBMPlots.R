################################################################################
# Dylan Anderson
# University of Texas at Austin
# May 5, 2014
# 
# This script creates analysis plots for the GBM data created in GBM.R and 
# GBMInteractionDepth.R
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

################################################################################
# Interaction depth effects
#
################################################################################
require(ggplot2)
load('results//GBMInteractionEffects.RData')
df <- data.frame(rmse=test.rmse, interaction.depth=1:length(test.rmse))
ggplot(df, aes(x=interaction.depth, y=rmse)) + geom_line(colour='blue') +
  geom_point(size=3) + labs(title='GBM Training Error vs Interaction Effects', 
                            x='Interaction Depth', y='RMSE') +
  theme(title=element_text(size=24), axis.text=element_text(size=16))
ggsave(filename="results/GBMInteractionEffects.png")


################################################################################
# Parameter Selection
#
################################################################################
load('results/GBM.RData')
require('gbm')
require('reshape')

targets <- names(train.transf)[grep('Class', names(train.transf))]

gbm.models <- lapply(gbm.models, function(model) { 
  model$fit <- 0
  return(model)
  })
predictor.importance <- lapply(gbm.models, summary, plotit=FALSE)
for (i in 1:length(predictor.importance)) {
  predictor.importance[[i]]$target <- targets[i]
}


predictor.importance <- melt(predictor.importance)
predictor.importance$variable <- NULL
targets <- gsub("Class","", targets)

ggplot(predictor.importance, aes(x=L1, y=value, fill=var)) + 
  geom_density(alpha=0.3, stat="identity") + 
  labs(title='GBM Relative Factor Importance', x='Target', y='Relative Importance')+
  scale_x_discrete(labels=targets) + 
  theme(title=element_text(size=24), 
        axis.text.x=element_text(size=15, angle=90, hjust=1, vjust=0.5),
        axis.text.y=element_blank(), legend.title=element_blank(), 
          legend.text=element_text(size=16))
ggsave(filename="results/GBMFactorImportance.png")


