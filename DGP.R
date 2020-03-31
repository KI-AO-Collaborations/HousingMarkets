library(data.table)

size <- sample(1:3, 1000, replace=T)
beds <- c()
baths <- c()

for (i in 1:length(size)) {
  if(size[i]==1){
    beds[i]<- sample(1:2)
    baths[i]<- sample(1:2)
  }
  if(size[i]==2){
    beds[i]<- sample(2:3)
    baths[i]<- sample(2:3)
  }
  if(size[i]==3){
    beds[i]<- sample(3:4)
    baths[i]<- sample(3:4)
  }
}

data <- data.table(beds, baths, size)

beta <- c(2, 1.75, 3)

mu_PR <- beta[1]*baths+beta[2]*beds+beta[3]*size

sigma_PR <- .25*size

P_R <- rnorm(1000, mean = mu_PR, sd = sigma_PR)

alpha <- 1
gamma <- 1
c <- 0
epsilon <- 1
lambda <- 10

sigma_D <- ((alpha/P_R)-c)^(1/(1+gamma))

D <- exp(rnorm(1000, 0, sigma_D))

P <- P_R - lambda*(D)^(1/(1+epsilon)) + rnorm(1000, 0, .25*size)


hist(P_R, breaks=30)
hist(sigma_D, breaks=50)
hist(D, breaks=50)
hist(P, breaks=50)

plot(P_R,P)
plot(P_R,D)
plot(D,P)