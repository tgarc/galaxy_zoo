library(leaps)
library(caret)
library(corrplot)
load("/Users/Tzu-Ling/Downloads/Galaxy Zoo/ProcessedData.RData")
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
train.transf<-train.transf[,-highCorr.idx]
test.transf<-test.transf[,-highCorr.idx]
#sample train set
permut<-train.transf[sample(nrow(train.transf)),]
subtrain1<-permut[c(1:3000),];train1<-subtrain1[sample(nrow(subtrain1),size=29982,replace=T),];
subtrain2<-permut[c(3001:6000),];train2<-subtrain2[sample(nrow(subtrain2),size=29982,replace=T),];
subtrain3<-permut[c(6001:9000),];train3<-subtrain3[sample(nrow(subtrain3),size=29982,replace=T),];
subtrain4<-permut[c(9001:12000),];train4<-subtrain4[sample(nrow(subtrain4),size=29982,replace=T),];
subtrain5<-permut[c(12001:15000),];train5<-subtrain5[sample(nrow(subtrain5),size=29982,replace=T),];
subtrain6<-permut[c(15001:18000),];train6<-subtrain6[sample(nrow(subtrain6),size=29982,replace=T),];
subtrain7<-permut[c(18001:21000),];train7<-subtrain7[sample(nrow(subtrain7),size=29982,replace=T),];
subtrain8<-permut[c(21001:24000),];train8<-subtrain8[sample(nrow(subtrain8),size=29982,replace=T),];
subtrain9<-permut[c(24001:27000),];train9<-subtrain9[sample(nrow(subtrain9),size=29982,replace=T),];
subtrain10<-permut[c(27001:29982),];train10<-subtrain10[sample(nrow(subtrain10),size=29982,replace=T),]; 

num<-ncol(train.transf)
x.train1<-train1[,1:12]; y.train1<-train1[,13:num]; 
x.train2<-train2[,1:12]; y.train2<-train2[,13:num];
x.train3<-train3[,1:12]; y.train3<-train3[,13:num];
x.train4<-train4[,1:12]; y.train4<-train4[,13:num];
x.train5<-train5[,1:12]; y.train5<-train5[,13:num];
x.train6<-train6[,1:12]; y.train6<-train6[,13:num];
x.train7<-train7[,1:12]; y.train7<-train7[,13:num];
x.train8<-train8[,1:12]; y.train8<-train8[,13:num];
x.train9<-train9[,1:12]; y.train9<-train9[,13:num];
x.train10<-train10[,1:12]; y.train10<-train10[,13:num];
###############################BASIC LINEAR REGRESSION######################################

lin.fit1<-lm(as.matrix(train1[,13:num])~.,train1[,1:12]);lin.pred1<-predict(lin.fit1,newdata=test.transf);
lin.fit2<-lm(as.matrix(train2[,13:num])~.,train2[,1:12]);lin.pred2<-predict(lin.fit2,newdata=test.transf);
lin.fit3<-lm(as.matrix(train3[,13:num])~.,train3[,1:12]);lin.pred3<-predict(lin.fit3,newdata=test.transf);
lin.fit4<-lm(as.matrix(train4[,13:num])~.,train4[,1:12]);lin.pred4<-predict(lin.fit4,newdata=test.transf);
lin.fit5<-lm(as.matrix(train5[,13:num])~.,train5[,1:12]);lin.pred5<-predict(lin.fit5,newdata=test.transf);
lin.fit6<-lm(as.matrix(train6[,13:num])~.,train6[,1:12]);lin.pred6<-predict(lin.fit6,newdata=test.transf);
lin.fit7<-lm(as.matrix(train7[,13:num])~.,train7[,1:12]);lin.pred7<-predict(lin.fit7,newdata=test.transf);
lin.fit8<-lm(as.matrix(train8[,13:num])~.,train8[,1:12]);lin.pred8<-predict(lin.fit8,newdata=test.transf);
lin.fit9<-lm(as.matrix(train9[,13:num])~.,train9[,1:12]);lin.pred9<-predict(lin.fit9,newdata=test.transf);
lin.fit10<-lm(as.matrix(train10[,13:num])~.,train10[,1:12]);lin.pred10<-predict(lin.fit10,newdata=test.transf);
save(lin.pred1,file="pred1_linSub.RData");
save(lin.pred2,file="pred2_linSub.RData");
save(lin.pred3,file="pred3_linSub.RData");
save(lin.pred4,file="pred4_linSub.RData");
save(lin.pred5,file="pred5_linSub.RData");
save(lin.pred6,file="pred6_linSub.RData");
save(lin.pred7,file="pred7_linSub.RData");
save(lin.pred8,file="pred8_linSub.RData");
save(lin.pred9,file="pred9_linSub.RData");
save(lin.pred10,file="pred10_linSub.RData");
err1<-as.matrix(lin.pred1-y.test)
RMSE_lin1<-sqrt(sum(err1^2)/(nrow(err1)*ncol(err1)))
############################Hierachy##################################################
dia<-c(1,1,1,2,2,4,4,8,8,64,64,64,64,128,128,2,2,2,256,256,256,256,256,256,256,4,4,4,16,16,16,32,32,32,32,32,32)
penalty<-diag(dia)
y.temp.train1<-as.matrix(y.train1)%*%penalty; 
y.temp.train2<-as.matrix(y.train2)%*%penalty; 
y.temp.train3<-as.matrix(y.train3)%*%penalty; 
y.temp.train4<-as.matrix(y.train4)%*%penalty; 
y.temp.train5<-as.matrix(y.train5)%*%penalty; 
y.temp.train6<-as.matrix(y.train6)%*%penalty; 
y.temp.train7<-as.matrix(y.train7)%*%penalty; 
y.temp.train8<-as.matrix(y.train8)%*%penalty; 
y.temp.train9<-as.matrix(y.train9)%*%penalty; 
y.temp.train10<-as.matrix(y.train10)%*%penalty; 
y.temp.test<-as.matrix(y.test)%*%penalty;

hier.fit1<-lm(as.matrix(y.temp.train1)~.,x.train1)
hier.fit2<-lm(as.matrix(y.temp.train2)~.,x.train2)
hier.fit3<-lm(as.matrix(y.temp.train3)~.,x.train3)
hier.fit4<-lm(as.matrix(y.temp.train4)~.,x.train4)
hier.fit5<-lm(as.matrix(y.temp.train5)~.,x.train5)
hier.fit6<-lm(as.matrix(y.temp.train6)~.,x.train6)
hier.fit7<-lm(as.matrix(y.temp.train7)~.,x.train7)
hier.fit8<-lm(as.matrix(y.temp.train8)~.,x.train8)
hier.fit9<-lm(as.matrix(y.temp.train9)~.,x.train9)
hier.fit10<-lm(as.matrix(y.temp.train10)~.,x.train10)

pred_hier1<-predict(hier.fit1,newdata=x.test)
pred_hier2<-predict(hier.fit2,newdata=x.test)
pred_hier3<-predict(hier.fit3,newdata=x.test)
pred_hier4<-predict(hier.fit4,newdata=x.test)
pred_hier5<-predict(hier.fit5,newdata=x.test)
pred_hier6<-predict(hier.fit6,newdata=x.test)
pred_hier7<-predict(hier.fit7,newdata=x.test)
pred_hier8<-predict(hier.fit8,newdata=x.test)
pred_hier9<-predict(hier.fit9,newdata=x.test)
pred_hier10<-predict(hier.fit10,newdata=x.test)

save(pred_hier1,file="pred1_hierSub.RData");
save(pred_hier2,file="pred2_hierSub.RData");
save(pred_hier3,file="pred3_hierSub.RData");
save(pred_hier4,file="pred4_hierSub.RData");
save(pred_hier5,file="pred5_hierSub.RData");
save(pred_hier6,file="pred6_hierSub.RData");
save(pred_hier7,file="pred7_hierSub.RData");
save(pred_hier8,file="pred8_hierSub.RData");
save(pred_hier9,file="pred9_hierSub.RData");
save(pred_hier10,file="pred10_hierSub.RData");

err<-as.matrix(pred_hier-y.temp.test)%*%solve(penalty)
RMSE_hier<-sqrt(sum(err^2)/(nrow(err)*ncol(err)))
############################Inverse Hierachy##########################################
dia<-c(256,256,256,128,128,64,64,32,32,4,4,4,4,2,2,128,128,128,1,1,1,1,1,1,1,64,64,64,16,16,16,8,8,8,8,8,8)
penalty<-diag(dia)
y.temp.train1<-as.matrix(y.train1)%*%penalty; 
y.temp.train2<-as.matrix(y.train2)%*%penalty; 
y.temp.train3<-as.matrix(y.train3)%*%penalty; 
y.temp.train4<-as.matrix(y.train4)%*%penalty; 
y.temp.train5<-as.matrix(y.train5)%*%penalty; 
y.temp.train6<-as.matrix(y.train6)%*%penalty; 
y.temp.train7<-as.matrix(y.train7)%*%penalty; 
y.temp.train8<-as.matrix(y.train8)%*%penalty; 
y.temp.train9<-as.matrix(y.train9)%*%penalty; 
y.temp.train10<-as.matrix(y.train10)%*%penalty; 
y.temp.test<-as.matrix(y.test)%*%penalty;

hierInver.fit1<-lm(as.matrix(y.temp.train1)~.,x.train1)
hierInver.fit2<-lm(as.matrix(y.temp.train2)~.,x.train2)
hierInver.fit3<-lm(as.matrix(y.temp.train3)~.,x.train3)
hierInver.fit4<-lm(as.matrix(y.temp.train4)~.,x.train4)
hierInver.fit5<-lm(as.matrix(y.temp.train5)~.,x.train5)
hierInver.fit6<-lm(as.matrix(y.temp.train6)~.,x.train6)
hierInver.fit7<-lm(as.matrix(y.temp.train7)~.,x.train7)
hierInver.fit8<-lm(as.matrix(y.temp.train8)~.,x.train8)
hierInver.fit9<-lm(as.matrix(y.temp.train9)~.,x.train9)
hierInver.fit10<-lm(as.matrix(y.temp.train10)~.,x.train10)

pred_hierInver1<-predict(hierInver.fit1,newdata=x.test)
pred_hierInver2<-predict(hierInver.fit2,newdata=x.test)
pred_hierInver3<-predict(hierInver.fit3,newdata=x.test)
pred_hierInver4<-predict(hierInver.fit4,newdata=x.test)
pred_hierInver5<-predict(hierInver.fit5,newdata=x.test)
pred_hierInver6<-predict(hierInver.fit6,newdata=x.test)
pred_hierInver7<-predict(hierInver.fit7,newdata=x.test)
pred_hierInver8<-predict(hierInver.fit8,newdata=x.test)
pred_hierInver9<-predict(hierInver.fit9,newdata=x.test)
pred_hierInver10<-predict(hierInver.fit10,newdata=x.test)

save(pred_hierInver1,file="pred1_hierInverSub.RData");
save(pred_hierInver2,file="pred2_hierInverSub.RData");
save(pred_hierInver3,file="pred3_hierInverSub.RData");
save(pred_hierInver4,file="pred4_hierInverSub.RData");
save(pred_hierInver5,file="pred5_hierInverSub.RData");
save(pred_hierInver6,file="pred6_hierInverSub.RData");
save(pred_hierInver7,file="pred7_hierInverSub.RData");
save(pred_hierInver8,file="pred8_hierInverSub.RData");
save(pred_hierInver9,file="pred9_hierInverSub.RData");
save(pred_hierInver10,file="pred10_hierInverSub.RData");

err<-as.matrix(pred_hier-y.temp.test)%*%solve(penalty)
RMSE_hierInver<-sqrt(sum(err^2)/(nrow(err)*ncol(err)))
##########################BEST SUBSET SELECTION#############################################
x_iter<-list(x.train1,x.train2,x.train3,x.train4,x.train5,x.train6,x.train7,x.train8,x.train9,x.train10)
y_iter<-list(y.train1,y.train2,y.train3,y.train4,y.train5,y.train6,y.train7,y.train8,y.train9,y.train10)
for (k in 1:10)
{
  x<-data.frame(x_iter[k]);y<-data.frame(y_iter[k]);
  num_pre<-ncol(x)
  ##Since the subset selection can only target at one variable, we iterate through all the y's
  ##Try to find best subsets for each y
  coeff.result<-vector("list",12)
  error.result<-c()
  pred_bestsub<-vector("list",37)
  for (j in 1:37)
  {
    subset<-regsubsets(as.matrix(y[,j])~.,x,nvmax=num_pre,method="exhaustive")
    error<-rep(NA,num_pre)
    test.mat<-model.matrix(test.transf[,12+j]~.,data=test.transf[,1:12])
    for (i in 1:num_pre)
    {
      coeff<-coef(subset, id=i)
      pred<-test.mat[,names(coeff)]%*%coeff
      error[i]<-mean((test.transf[,12+j]-pred)^2)
    }
    error.result<-c(error.result,min(error))
    coeff.result[[j]]<-coef(subset,id=which.min(error))
    pred_bestsub[[j]]<-test.mat[,names(coeff.result[[j]])]%*%coeff.result[[j]]
  }  
  save(pred_bestsub,file=paste("pred_bestSub",k,".RData",sep=""))
}
################################Poly Regression###########################################
library(MASS)
x_iter<-list(x.train1,x.train2,x.train3,x.train4,x.train5,x.train6,x.train7,x.train8,x.train9,x.train10)
y_iter<-list(y.train1,y.train2,y.train3,y.train4,y.train5,y.train6,y.train7,y.train8,y.train9,y.train10)
for (k in 1:10)
{
  #error rate (high) cutoff=0.53
  poly.fit_high<-lm(as.matrix(data.frame(y_iter[k]))~ellipticity_g+bulge_presence_g+disc_dev_g+
                      disc_dev_r+area_i+m2_i*area_g*bulge_prominence_g*area_r*
                      bulge_prominence_i*bulge_presence_i*m1_i,data.frame(x_iter[k]))
  #error rate (low)
  poly.fit_low<-lm(as.matrix(data.frame(y_iter[k]))~ellipticity_g*bulge_presence_g*disc_dev_g*
                     disc_dev_r*area_i*m2_i+area_g+bulge_prominence_g+area_r+
                     bulge_prominence_i+bulge_presence_i+m1_i,data.frame(x_iter[k]))
  pred_poly_high<-predict(poly.fit_high,newdata=x.test)
  pred_poly_low<-predict(poly.fit_low,newdata=x.test)
  save(pred_poly_high,file=paste("pred_poly_high",k,".RData",sep=""))
  save(pred_poly_low,file=paste("pred_poly_low",k,".RData",sep=""))
}
###############################LASSO Regression##########################################
library(glmnet)
x_iter<-list(x.train1,x.train2,x.train3,x.train4,x.train5,x.train6,x.train7,x.train8,x.train9,x.train10)
y_iter<-list(y.train1,y.train2,y.train3,y.train4,y.train5,y.train6,y.train7,y.train8,y.train9,y.train10)

f.test<-as.formula(as.matrix(y.test)~ellipticity_g*area_g*bulge_prominence_g*
                     bulge_presence_g*disc_dev_g*area_r*disc_dev_r*area_i*
                     bulge_prominence_i*bulge_presence_i*m1_i*m2_i,x.test)
test.mat<-model.matrix(f.test,test.transf)

for(k in 1:10)
{
  y<-data.frame(y_iter[k]);x<-data.frame(x_iter[k]);
  f.train<-as.formula(as.matrix(y)~ellipticity_g*area_g*bulge_prominence_g*
                        bulge_presence_g*disc_dev_g*area_r*disc_dev_r*area_i*
                        bulge_prominence_i*bulge_presence_i*m1_i*m2_i,x)
  train.mat<-model.matrix(f.train,train.transf)
  err_lasso<-0
  pred_lasso<-matrix(data=NA,nrow=nrow(y.test),ncol(y.test))
  for (i in 1:ncol(y.train))
  {
    lasso.fit<-cv.glmnet(train.mat,y[,i],nfolds=3,nlambda=20,alpha=0)
    pred_lasso[,i]<-predict(lasso.fit,test.mat,s="lambda.min")
    err_lasso<-err_lasso+sum((pred_lasso[i]-y.test[,i])^2)
  }
  save(pred_lasso,file=paste("pred",k,"_lasso.RData",sep=""))
}

