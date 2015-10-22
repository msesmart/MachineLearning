getwd()
Spam <- read.table("~/Documents/Spring 15/Machine Learning/Spam.txt", quote="\"")
dim(Spam)
str(Spam)
cor(Spam)
library(dplyr)
train <- sample_frac(Spam,0.7)
test <- setdiff(Spam,train)
# try logistic regression for training data
glm.fit <- glm(V58~.,data=train,family = binomial)
summary(glm.fit)
glm.probs <- predict(glm.fit,test,type="response")
head(glm.probs)
glm.pred <- rep("yes",1219)
glm.pred[glm.probs<0.5] <- "no"
table(glm.pred)
table(glm.pred,test.result)
mean(glm.pred!=test.result)

# The test error rate for logistic regression is 8.6%
# try LDA for training data 
library(MASS)
lda.fit <- lda(V58~.,data=train)
lda.fit
plot(lda.fit)
lda.predict <- predict(lda.fit,test)
names(lda.predict)
lda.class <- lda.predict$class
table(lda.class)
table(lda.class,test$V58)
mean(lda.class!=test$V58)
# The test error rate for LDA is 12.4%
# Use decision tree for training
attach(train)
result <- ifelse(train$V58==0,"no","yes")
train <- data.frame(train,result)
library(tree)
tree.spam <- tree(result~.-V58,data=train)
summary(tree.spam)
plot(tree.spam)
text(tree.spam,pretty=0)
tree.pred <- predict(tree.spam,test,type="class")
table(tree.pred,test.result)
mean(tree.pred!=test.result)
# The test error rate is 9.76%
cv.spam <- cv.tree(tree.spam,FUN=prune.misclass)
names(cv.spam)
cv.spam
# plot error rate as a function of both size and k
plot(cv.spam$size,cv.spam$dev,type="b")
plot(cv.spam$k,cv.spam$dev,type="b")
# we choose the pruned tree with 14 nodes
prune.spam <- prune.misclass(tree.spam,best=14)
plot(prune.spam)
text(prune.spam,pretty=0)
tree.pred.prune <- predict(prune.spam,test,type="class")
table(tree.pred.prune,test.result)
mean(tree.pred.prune!=test.result)
# The error rate still 9.76%, the pruned tree does not make much sense

# Bagging and Random Forest
library(randomForest)
train <- train[-58]
bag.spam <- randomForest(result~.,data=train,mtry=57,importance=TRUE)
bag.spam
yhat.bag <- predict(bag.spam,test)
table(yhat.bag)
table(yhat.bag,test.result)
mean(yhat.bag!=test.result)
# The test error rate is 6.4% for bagging

# try random forest 
rf.spam <- randomForest(result~.,data=train,importance=T)
rf.spam
yhat.rf <- predict(rf.spam,test)
table(yhat.rf,test.result)
mean(yhat.rf!=test.result)
# The test error rate is 5.5% for random forest

# Try boosting
install.packages("gbm")
library(gbm)
train <- sample_frac(Spam,0.7)
test <- setdiff(Spam,train)
boost.spam <- gbm(V58~.,data=train,distribution="bernoulli",n.trees=5000)
summary(boost.spam)
yhat.boost <- predict(boost.spam,test,n.trees=5000)
unique(yhat.boost)
table(yhat.boost,test$V58)
