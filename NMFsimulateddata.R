install.packages("NMF")
library(NMF)

setwd("C:/Users/danie/Documents/AutoMF/NMFdir")

n <- 50
counts <- c(5, 5, 8)

# no noise

for (x in 1:3){ 
  V <- syntheticNMF(n, r = 2, counts, noise=FALSE)
  
  write.table(V, file=paste('NMFsim.tsv',as.character(x)), quote=FALSE, sep='\t')
  
  }

