library(leaps)
library(caret)
library(corrplot)
load("/Users/Tzu-Ling/Downloads/Galaxy Zoo/ProcessedData.RData")
#remove Galaxy ID
train.transf<-train.transf[,-22]
test.transf<-test.transf[,-22]
#remove the outlier
remove_outliers <- function(x, na.rm = TRUE) 
{
  qnt <- quantile(x, probs=c(.00001, .99999), na.rm = na.rm)
  H <- 1.5 * IQR(x, na.rm = na.rm)
  y <- x
  y[x < (qnt[1] - H)] <- NA
  y[x > (qnt[2] + H)] <- NA
  y
}
train.transf<-remove_outliers(as.matrix(train.transf)); 
test.transf<-remove_outliers(as.matrix(test.transf));
train.transf<-na.omit(train.transf); train.transf<-data.frame(train.transf);
test.transf<-na.omit(test.transf); test.transf<-data.frame(test.transf);
#generate the target and data of training/testing sets
x.train<-train.transf[,1:21]; x.train<-data.frame(x.train);
x.test<-test.transf[,1:21]; x.test<-data.frame(x.test);
y.train<-train.transf[,22:ncol(train.transf)]; y.train<-data.frame(y.train);
y.test<-test.transf[,22:ncol(train.transf)]; y.test<-data.frame(y.test);
#remove predictors with near-zero variance predictors
nearZero.idx<-nearZeroVar(x.train)
if (length(nearZero.idx)>0)x.train<-x.train[,-nearZero.idx]
#remove predictors with high correlations
train.corr<-cor(x.train)
highCorr.idx<-findCorrelation(train.corr,cutoff=0.8)
x.train<-x.train[,-highCorr.idx]
names(x.train)
###############################best subset selection######################################
num_pre<-ncol(x.train)
##Since the subset selection can only target at one variable, we iterate through all the y's
##Try to find best subsets for each y
coeff.result<-vector("list",ncol(y.train))
error.result<-c()
pred_bestsub<-vector("list",ncol(y.train))
for (j in 1:ncol(y.train))
{
  subset<-regsubsets(as.matrix(y.train[,j])~.,x.train,nvmax=num_pre,method="exhaustive")
  error<-rep(NA,num_pre)
  test.mat<-model.matrix(test.transf[,21+j]~.,data=test.transf[,1:21])
  for (i in 1:num_pre)
  {
    coeff<-coef(subset, id=i)
    pred<-test.mat[,names(coeff)]%*%coeff
    error[i]<-mean((test.transf[,21+j]-pred)^2)
  }
  error.result<-c(error.result,min(error))
  coeff.result[[j]]<-coef(subset,id=which.min(error))
  pred_bestsub[[j]]<-test.mat[,names(coeff.result[[j]])]%*%coeff.result[[j]]
}  
#test it on test sets
RMSE_bestsub<-sqrt(mean(error.result))
save(pred_bestsub,file="prediction_bestSubset.RData")
##################################Basic Regression########################################
linear.fit<-lm(as.matrix(y.train)~.,x.train)
pred_lin_reg<-predict(linear.fit,newdata=x.test)
err<-as.matrix(pred-y.test)
RMSE_linear<-sqrt(sum(err^2)/(nrow(err)*ncol(err)))
save(pred_lin_reg,file="prediction_linearRegression.RData")
################################Poly Regression###########################################
library(MASS)
##From the result below, the error rates are most probably affected by overfitting
##But we can see that add interaction terms improves the performance, so we don't consider 
##PCR, since just by using linear regression, we are not overfitting
#error rate (high) cutoff=0.53
poly.fit_high<-lm(as.matrix(y.train)~ellipticity_g+bulge_presence_g+disc_dev_g+
                    disc_dev_r+area_i+m2_i*area_g*bulge_prominence_g*area_r*
                    bulge_prominence_i*bulge_presence_i*m1_i,x.train)
#error rate (all) 
poly.fit_all<-lm(as.matrix(y.train)~ellipticity_g*area_g*bulge_prominence_g*
                   bulge_presence_g*disc_dev_g*area_r*disc_dev_r*area_i*bulge_prominence_i*
                   bulge_presence_i*m1_i*m2_i,x.train)
#error rate (low)
poly.fit_low<-lm(as.matrix(y.train)~ellipticity_g*bulge_presence_g*disc_dev_g*
                   disc_dev_r*area_i*m2_i+area_g+bulge_prominence_g+area_r+
                   bulge_prominence_i+bulge_presence_i+m1_i,x.train)
#prediction
pred_poly_high<-predict(poly.fit_high,newdata=x.test)
pred_poly_all<-predict(poly.fit_all,newdata=x.test)
pred_poly_low<-predict(poly.fit_low,newdata=x.test)
err_poly_high<-as.matrix(pred_poly_high-y.test)
err_poly_all<-as.matrix(pred_poly_all-y.test)
err_poly_low<-as.matrix(pred_poly_low-y.test)
#calculate RMSE
RMSE_poly_high<-sqrt(sum(err_poly_high^2)/(nrow(err_poly_high)*ncol(err_poly_high)))
RMSE_poly_all<-sqrt(sum(err_poly_all^2)/(nrow(err_poly_all)*ncol(err_poly_all)))
RMSE_poly_low<-sqrt(sum(err_poly_low^2)/(nrow(err_poly_low)*ncol(err_poly_low)))
#save the data
save(pred_poly_high,file="prediction_poly_high.RData")
save(pred_poly_all,file="prediction_poly_all.RData")
save(pred_poly_low,file="prediction_poly_low.RData")
###############################LASSO Regression##########################################
library(glmnet)
f.train<-as.formula(as.matrix(y.train)~ellipticity_g*area_g*bulge_prominence_g*
                      bulge_presence_g*disc_dev_g*area_r*disc_dev_r*area_i*
                      bulge_prominence_i*bulge_presence_i*m1_i*m2_i,x.train)
f.test<-as.formula(as.matrix(y.test)~ellipticity_g*area_g*bulge_prominence_g*
                     bulge_presence_g*disc_dev_g*area_r*disc_dev_r*area_i*
                     bulge_prominence_i*bulge_presence_i*m1_i*m2_i,x.test)
train.mat<-model.matrix(f.train,train.transf)
test.mat<-model.matrix(f.test,test.transf)
err_lasso<-0
pred_lasso<-matrix(data=NA,nrow=nrow(y.test),ncol(y.test))
for (i in 1:ncol(y.train))
{
  lasso.fit<-cv.glmnet(train.mat,y.train[,i],nfolds=3,nlambda=20,alpha=0)
  pred_lasso[,i]<-predict(lasso.fit,test.mat,s="lambda.min")
  err_lasso<-err_lasso+sum((pred_lasso[i]-y.test[,i])^2)
}
#error rate
RMSE_lasso<-sqrt(err_lasso/(nrow(y.test)*ncol(y.test)))
#save the data
save(pred_lasso,file="prediction_lasso.RData")
###############################SVM Regression############################################
library(caret)
library(kernlab)
library(doMC)
registerDoMC()
ctrl<-trainControl(method="cv",allowParallel=TRUE)
svm.lin.fit<-train(x.train,as.matrix(y.train),method="svmLinear",trControl=ctrl)
svm.rad.fit<-train(x.train,as.matrix(y.train),method="svmRadial",trControl=ctrl)
svm.gaulin.fit<-train(x.train,as.matrix(y.train),method="gaussprLinear",trControl=ctrl)
svm.gaupol.fit<-train(x.train,as.matrix(y.train),method="gaussprPoly",trControl=ctrl)
svm.gaurad.fit<-train(x.train,as.matrix(y.train),method="gaussprRadial",trControl=ctrl)

pred.lin<-predict(svm.lin.fit,newdata=x.test)
pred.rad<-predict(svm.rad.fit,newdata=x.test)
pred.gaulin<-predict(svm.gaulin.fit,newdata=x.test)
pred.gaupol<-predict(svm.gaupol.fit,newdata=x.test)
pred.gaurad<-predict(svm.gaurad.fit,newdata=x.test)

num=nrow(y.test)*ncol(y.test)
RMSE_svm_lin<-sqrt(sum((pred.lin-y.test)^2)/num)
RMSE_svm_rad<-sqrt(sum((pred.rad-y.test)^2)/num)
RMSE_svm_gaulin<-sqrt(sum((pred.gaulin-y.test)^2)/num)
RMSE_svm_gaupol<-sqrt(sum((pred.gaupol-y.test)^2)/num)
RMSE_svm_gaurad<-sqrt(sum((pred.gaurad-y.test)^2)/num)

save(pred.lin,file="prediction_svmLinear.RData")
save(pred.rad,file="prediction_svmRadial.RData")
save(pred.gaulin,file="prediction_svmGauLinear.RData")
save(pred.gaupol,file="prediction_svmGauPoly.RData")
save(pred.gaurad,file="prediction_svmGauRadial.RData")
