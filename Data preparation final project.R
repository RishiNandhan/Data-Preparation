#install the necessary packages
install.packages("dummies") #create dummy variables
library(dummies)
install.packages("ggplot2") #create graphics
library(ggplot2)
install.packages("reshape") #used to reshape variables into heatmap
library(reshape)
install.packages("GGally") #extension of ggplot2 package
library(GGally)
install.packages("randomForest") #Package for Random Forest model
library(randomForest)
install.packages("MASS")
library(MASS)
install.packages("neuralnet") #Package for neural network
library(neuralnet)
install.packages("caret")
library(caret)
install.packages("ROSE") #Package for over and under sampling
library(ROSE)

#load the data
setwd("E://") #set the working directory
liver.df<-read.csv("liver.csv") #read the csv file
#structure and summary of the dataset
summary(liver.df) #summary of the dataset
str(liver.df) #structure of the dataset
#find the number of NA values and remove
sum(is.na(liver.df)) #total sum of NA values in dataset
liver.df<-na.omit(liver.df) #remove the NA values
#create dummy variables for categorical variables
liver.df$Dataset<-ifelse(liver.df$Dataset==1,1,0)
liver_dummy.df<-dummy.data.frame(liver.df,sep="_") #create dummy variables
str(liver_dummy.df)
tab8<-table(liver.df$Gender)
tab8
tab9<-table(liver.df$Gender,liver.df$Dataset)
tab9

##
##
#create some charts to explain the relationship of your dataset
##
##
liver.df$Dataset<-as.factor(liver.df$Dataset)
#barchart to see how many patients and non-patients are there.
ggplot(liver.df)+geom_bar(aes(Dataset),fill=c("brown","orange"))
#barchart to see how many male and female are there.
ggplot(liver.df)+geom_bar(aes(Gender),fill=c("red","blue"))
ggplot(liver.df)+geom_bar(aes(Age, fill=Dataset))
#scatter plot between total and direct bilirubin.
lm(liver_dummy.df$Direct_Bilirubin~liver_dummy.df$Total_Bilirubin)
ggplot(liver_dummy.df)+geom_point(aes(x=Total_Bilirubin,y=Direct_Bilirubin),color="red")+
  geom_abline(intercept = 0.18,slope = 0.47,size=1,color="blue",linetype="dashed")
#scatter plot between alamine and aspartate aminotransferase.
lm(liver_dummy.df$Alamine_Aminotransferase~liver_dummy.df$Aspartate_Aminotransferase)
ggplot(liver_dummy.df)+geom_point(aes(x=Alamine_Aminotransferase,y=Aspartate_Aminotransferase),color="brown")+
  geom_abline(intercept = 25.86,slope = 0.5,size=1,color="blue",linetype="dashed")
#scatter plot between total_protein and albumin.
lm(liver_dummy.df$Total_Protiens~liver_dummy.df$Albumin)
ggplot(liver_dummy.df)+geom_point(aes(x=Total_Protiens,y=Albumin),color="blue")+
  geom_abline(intercept = -1.32,slope = 0.7,size=1,color="red",linetype="dashed")
#find the correlation of variables in heatmap and matrix form.
cor.mat<-round(cor(liver_dummy.df),2) #correlation table 
cor.mat
melted.cor.mat<-melt(cor.mat)
ggplot(melted.cor.mat,aes(x=X1, y=X2, fill=value))+
  geom_tile()+
  geom_text(aes(x=X1, y=X2, label=value)) #heatmap visualizing correlation
#drop the variables which are highly correlated to each other
#droping variables
liver_dummy.df$Total_Bilirubin<-NULL
liver_dummy.df$Alamine_Aminotransferase<-NULL
liver_dummy.df$Albumin<-NULL


#Neural Network Model
norm_values<-preProcess(liver_dummy.df,method = "range")
liver_norm<-predict(norm_values,liver_dummy.df)
set.seed(2010)
train.index<-sample(rownames(liver_norm), 0.7*dim(liver_norm)[1])
valid.index<-setdiff(row.names(liver_norm),train.index)
train.df<-liver_norm[train.index,]
valid.df<-liver_norm[valid.index,]

allvars1<-colnames(liver_norm)
predictorvars1<-allvars1[!allvars1%in%"Dataset"]
predictorvars1<-paste(predictorvars1,collapse = "+")
formula<-as.formula(paste("Dataset~",predictorvars1, collapse = "+"))
nn1<-neuralnet(formula = formula, data = train.df,hidden = 6, linear.output = F)
plot(nn1)
#training data
#denormalize the data and then find the error.
predictions0<-compute(nn1,train.df[,-9])
predictions0<-predictions0$net.result*(max(train.df$Dataset)-min(train.df$Dataset))+min(train.df$Dataset)
actualvalues0<-(train.df$Dataset)*(max(train.df$Dataset)-min(train.df$Dataset))+min(train.df$Dataset)
#RMSE(predictions0,actualvalues0)
output<-compute(nn1,train.df[,-9])
p1<-output$net.result
pred1<-ifelse(p1>0.5, 1,0)
tab1<-table(pred1, train.df$Dataset)
tab1
accuracy_neuralnet_training<-sum(diag(tab1))/sum(tab1)
accuracy_neuralnet_training
error_neuralnet_training<-1-sum(diag(tab1))/sum(tab1)

plot(train.df$Dataset,predictions0)
#validation data
#denormalize the data and find the error.
predictions1<-compute(nn1,valid.df[,-9])
predictions1<-predictions1$net.result*(max(valid.df$Dataset)-min(valid.df$Dataset))+min(valid.df$Dataset)
actualvalues1<-(valid.df$Dataset)*(max(valid.df$Dataset)-min(valid.df$Dataset))+min(valid.df$Dataset)
#RMSE(predictions1,actualvalues1)

output1<-compute(nn1,valid.df[,-9])
p2<-output1$net.result
pred2<-ifelse(p2>0.5, 1,0)
tab2<-table(pred2, valid.df$Dataset)
tab2
accuracy_neuralnet_validation<-sum(diag(tab2))/sum(tab2)
accuracy_neuralnet_validation
error_neuralnet_validation<-1-sum(diag(tab2))/sum(tab2)

plot(valid.df$Dataset,predictions1)


#logistic regression model
str(liver_dummy.df)
liver_dummy.df$Dataset<-as.factor(liver_dummy.df$Dataset)
#split your model into training and testing
set.seed(1000)
ind<-sample(2, nrow(liver_dummy.df),replace = T, prob = c(0.7,0.3))
train<-liver_dummy.df[ind==1,]
test<-liver_dummy.df[ind==2,]
#run logistic regression model
mymodel<-glm(Dataset~., data= train, family ='binomial' )
summary(mymodel)

#predict with training data
p1<-predict(mymodel, train, type = 'response')
head(p1)
#check for accuracy and error percentage
pred1<-ifelse(p1>0.5,1,0)
tab1<-table(predicted=pred1, Actual=train$Dataset)
tab1
accuracy_logit_train<-sum(diag(tab1))/sum(tab1)
accuracy_logit_train
error_logit_train<-1-sum(diag(tab1))/sum(tab1)
#predict wit test data
p2<-predict(mymodel, test, type='response')
head(p2) 
#check for accuracy and error
pred2<-ifelse(p2>0.5,1,0)
tab2<-table(prediction=pred2, Actual=test$Dataset)
tab2
accuracy_logit_test<-sum(diag(tab2))/sum(tab2)
accuracy_logit_test
error_logit_test<-1-sum(diag(tab2))/sum(tab2)
# p-value to find your is significant or not.(if p-value is very less then significant)
with(mymodel, pchisq(null.deviance-deviance, df.null-df.residual, lower.tail = F))

#Random Forest
#split into training and testin data
set.seed(2337)
ind<-sample(2,nrow(liver_dummy.df),replace = T,prob = c(0.7,0.3))
training<-liver_dummy.df[ind==1,]
testing<-liver_dummy.df[ind==2,]
# run Random Forest model
set.seed(985)
rf<-randomForest(Dataset~., data = training,
                 importance=T, proximity=T)
rf
#predict with traininf data and plot confusion matrix
p1<-predict(rf, training)
confusionMatrix(p1, training$Dataset)
#predict with testing data and plot confusion matrix
p2<-predict(rf,testing)
confusionMatrix(p2,testing$Dataset)
table(training$Dataset)
#oversampling the training data and using random forest.
over<-ovun.sample(Dataset~., data = training,method = "over",N=560)$data
table(over$Dataset)
set.seed(876)
rf<-randomForest(Dataset~., data = over,
                 importance=T, proximity=T)
rf
#predict with training data and plot confusion matrix
p1<-predict(rf, over)
confusionMatrix(p1, over$Dataset)
#predict with testing data and plot confusion matrix
p2<-predict(rf,test)
confusionMatrix(p2,test$Dataset)

