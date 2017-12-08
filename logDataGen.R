set.seed(1408);
n <- 1000
p <- 20
s <- 2
beta <- c(rnorm(s),rep(0,p-s))
X <- matrix(rnorm(n*p), ncol = p)
Z= X %*% beta
pr = 1/(1+exp(-Z))         # pass through an inv-logit function
Y = rbinom(n,1,pr) 
write.table(X, file="Xlog.txt", row.names=FALSE, col.names=FALSE)
write.table(Y, file="Ylog.txt", row.names=FALSE, col.names=FALSE)
write.table(beta, file="Betalog.txt", row.names=FALSE, col.names=FALSE)
model<-cv.glmnet(X,Y,family="binomial",intercept=FALSE)
plot(model)
coef(model)
modelcoef<-coef(model, s="lambda.min",intercept=FALSE)
modelcoef<-as.matrix(modelcoef)
modelcoef<-modelcoef[-1]
write.table(modelcoef, file="logbest.txt", row.names=FALSE, col.names=FALSE)