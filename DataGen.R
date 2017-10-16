set.seed(1408);
n <- 100
p <- 1000
s <- 3
beta <- c(rnorm(s),rep(0,p-s))

X <- matrix(rnorm(n*p), ncol = p)
y <- X %*% beta + rnorm(n)
write.table(X, file="X.txt", row.names=FALSE, col.names=FALSE)
write.table(y, file="Y.txt", row.names=FALSE, col.names=FALSE)
write.table(beta, file="Beta.txt", row.names=FALSE, col.names=FALSE)