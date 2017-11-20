set.seed(1408);
n <- 30
p <- 40
s <- 2
beta <- c(rnorm(s),rep(0,p-s))

X <- matrix(rnorm(n*p), ncol = p)
Y <- X %*% beta + rnorm(n)
library(glmnet)
fit<-cv.glmnet(X,Y)
plot(fit)
coef(fit)
write.table(X, file="X.txt", row.names=FALSE, col.names=FALSE)
write.table(Y, file="Y.txt", row.names=FALSE, col.names=FALSE)
write.table(beta, file="Beta.txt", row.names=FALSE, col.names=FALSE)