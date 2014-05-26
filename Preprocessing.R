library(leaps)
library(caret)
library(corrplot)
load("ProcessedData.RData")
#remove Galaxy ID
train.transf<-train.transf[,-22]
test.transf<-test.transf[,-22]
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
x.test<-x.test[,-highCorr.idx]
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
err<-as.matrix(pred_lin_reg-y.test)
RMSE_linear<-sqrt(mean(err^2))
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
pred.lin<-matrix(data=NA,nrow=nrow(y.test),ncol(y.test))
pred.rad<-matrix(data=NA,nrow=nrow(y.test),ncol(y.test))
for (i in 1:37)
{
  svm.lin.fit<-train(x.train,as.matrix(y.train[,i]),method="svmLinear",trControl=ctrl)
  svm.rad.fit<-train(x.train,as.matrix(y.train[,i]),method="svmRadial",trControl=ctrl)
  pred.lin[,i]<-predict(svm.lin.fit,newdata=x.test)
  pred.rad[,i]<-predict(svm.rad.fit,newdata=x.test)
}

num=nrow(y.test)*ncol(y.test)
RMSE_svm_lin<-sqrt(sum((pred.lin-y.test)^2)/num)
RMSE_svm_rad<-sqrt(sum((pred.rad-y.test)^2)/num)

save(pred.lin,file="prediction_svmLinear.RData")
save(pred.rad,file="prediction_svmRadial.RData")
############################Hierachy##################################################
dia<-c(1,1,1,2,2,4,4,8,8,64,64,64,64,128,128,2,2,2,256,256,256,256,256,256,256,4,4,4,16,16,16,32,32,32,32,32,32)
penalty<-diag(dia)
y.temp.train<-as.matrix(y.train)%*%penalty; 
y.temp.test<-as.matrix(y.test)%*%penalty;
hier.fit<-lm(as.matrix(y.temp.train)~.,x.train)
pred_hier<-predict(hier.fit,newdata=x.test)
err<-as.matrix(pred_hier-y.temp.test)%*%solve(penalty)
RMSE_hier<-sqrt(mean(err^2))
save(pred_hier,file="prediction_hierachy.RData")
############################Inverse Hierachy##########################################
dia<-c(256,256,256,128,128,64,64,32,32,4,4,4,4,2,2,128,128,128,1,1,1,1,1,1,1,64,64,64,16,16,16,8,8,8,8,8,8)
penalty<-diag(dia)
y.temp.train<-as.matrix(y.train)%*%penalty; 
y.temp.test<-as.matrix(y.test)%*%penalty;
hierInver.fit<-lm(as.matrix(y.temp.train)~.,x.train)
pred_hierInver<-predict(hierInver.fit,newdata=x.test)
err<-as.matrix(pred_hierInver-y.temp.test)%*%solve(penalty)
RMSE_hierInver<-sqrt(mean(err^2))
save(pred_hierInver,file="prediction_hierachyInverse.RData")