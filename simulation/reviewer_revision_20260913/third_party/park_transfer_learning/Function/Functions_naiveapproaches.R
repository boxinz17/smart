library(corpcor)       # For fast SVD: fast.svd()
library(stats)         # For basic functions like apply(), chol(), etc. (usually loaded by default)


################################## ADMM update codes ###############################################
######################## Upadting B #########################
B.update <- function(L,U,eta, covXY, gramridgeInv){
  B_pred <- covXY - U + eta*L
  return( (gramridgeInv %*% B_pred) )
}
#############################################################

########### Updating C in the ADMM algorithm described in Section S2.1 ####################
L.update <- function(B,U,eta,lambda,p,q){
  Lpred <- B + U/eta
  svdres <- corpcor::fast.svd(Lpred)
  d <- svdres$d
  dthres <- pmax(d-(lambda/eta),0)
  return( list(L = (svdres$u %*% diag(dthres) %*% t(svdres$v)),d=dthres ))
}
#############################################################################################

############# ADMM for nuclear norm regularized multiple response regression ################
ADMM.nuclear <- function(Y,X,B,L,eta,lambda,maxiter,tol=1e-04,
                         standardize=TRUE){
  
  
  n <- nrow(Y)
  p <- ncol(X)
  q <- ncol(Y)
  # Y: n by q matrix of responses
  # X: matrix of p covariates
  # B: A in the ADMM algorithm described in Section S2.1
  # L: C in the ADMM algorithm described in Section S2.1
  # eta: rho in the ADMM algorithm described in Section S2.1
  # maxiter: the maximum number of interations
  # tol: tolerance
  
  
  if(standardize){
    Ymean <- apply(Y,2,mean)
    Ycenter <- Y - matrix(Ymean,n,q,byrow=T)
    Xmean <- apply(X,2,mean)
    Xcenter <- X - matrix(Xmean,n,p,byrow=T)
    Xcenternorm <- apply(Xcenter,2,function(x){sqrt(sum(x^2)/length(x))})
    Xstd <- Xcenter / matrix(Xcenternorm,n,p,byrow=T)
    X <- Xstd
    Y <- Ycenter
  }
  
  
  Onematrix <- matrix(1,p,q)
  gramridge <- diag(eta,p,p) + (1/n)*t(X)%*%X
  gramridgeInv <- chol2inv(chol(gramridge))
  covXY <- (1/n)*t(X)%*%Y
  
  iter <- 1
  isstop <- FALSE
  obj <- NULL
  errB <- errL <- errU <- NULL
  
  if(is.null(B)){
    B <- matrix(0,p,q)
  }
  if(is.null(L)){
    L <- matrix(0,p,q)
  }
  U <- B-L # Z in in the ADMM algorithm described in Section S2.1
  
  while(iter <= maxiter & !(isstop)){
    Bold <- B; Lold <- L; Uold <- U
    B <- B.update(L,U,eta, covXY, gramridgeInv)
    Lres <- L.update(B,U,eta,lambda,p,q)
    L <- Lres$L
    U <- U+eta*(B-L)
    res <- Y - X %*% B
    obj[iter] <- (1/(2*n))*sum(res^2) + lambda * sum(Lres$d )
    errB[iter] <-  sum((Bold -B)^2)
    errL[iter] <-  sum((Lold -L)^2)
    errU[iter] <-  sum((Uold -U)^2)
    isstop <- (max(errB[iter],errL[iter],errU[iter]) <= tol) # stopping criterion
    iter <- iter + 1
  }
  if(standardize){
    B <- matrix(1/Xcenternorm,p,q) * B # rescale
    L <- matrix(1/Xcenternorm,p,q) * L
    Bint <- matrix(Ymean,1,q) - matrix(Xmean,1,p) %*% B
    Lint <- matrix(Ymean,1,q) - matrix(Xmean,1,p) %*% L
    B <- rbind(Bint,B)
    L <- rbind(Lint,L)
    
  }
  
  return(list(B=B,L=L,obj=obj,errB=errB,errL=errL,errU=errU,iter=iter-1,d=Lres$d))
  
}
#############################################################################################



################# Cross-validation for nuclear norm regulairzed regression #################################

cv.nuclear <- function(Y,X,B,L,eta,lamseq,maxiter=100,tol=1e-04,nfold=5){
  Y <- as.matrix(Y)
  X <- as.matrix(X)
  ######################### Divide index ##################################
  set.seed(100)
  n <- nrow(Y)
  ind <- 1:n
  foldset <- list()
  samplesizes <- rep(n %/% nfold, nfold)
  if( (n %% nfold) > 0 ){
    samplesizes[1:(n %% nfold)] <- samplesizes[1:(n %% nfold)] + 1
  }
  
  for( k in 1:(nfold-1) ){
    foldset[[k]] <- sample(ind, samplesizes[[k]])
    ind <- setdiff(ind, foldset[[k]])
  }
  foldset[[nfold]] <- ind
  ##############################################################################
  Xtra <- Ytra <- list()
  Xtest <- Ytest <- list()
  ################### Standardization step ################################
  p <- ncol(X)
  q <- ncol(Y)
  Ymean <- apply(Y,2,mean)
  Ycenter <- Y-matrix(Ymean,byrow=T,n,q)
  Xmean <- apply(X,2,mean)
  Xcenter <- X-matrix(Xmean,n,p,byrow=T)
  Xcenternorm <- apply(Xcenter, 2, function(x){sqrt(sum(x^2)/n)})
  Xstd <- Xcenter / matrix(Xcenternorm,n,p, byrow=T)
  for(k in 1:nfold){
    Xtra[[k]] <- X[-foldset[[k]],]
    Ytra[[k]] <- Y[-foldset[[k]],]
    if(length(foldset[[k]]) == 1 ){
      Xtest[[k]] <- matrix(X[foldset[[k]],],nrow=1)
      Ytest[[k]] <- matrix(Y[foldset[[k]],],nrow=1)
    }else{
      Xtest[[k]] <- X[foldset[[k]],]
      Ytest[[k]] <- Y[foldset[[k]],]
    }
    
  }
  ##########################################################################
  
  
  ######################## K-fold validation ##############################
  lamfold <- expand.grid(lamseq, 1:nfold)
  lamfold <- as.matrix(lamfold)
  nlamfold <- nrow(lamfold)
  cverr <- lapply(1:nlamfold, function(x){
    lambda <- lamfold[x,1]
    fold <- lamfold[x,2]
    Ytrax <- Ytra[[fold]]
    Xtrax <- Xtra[[fold]]
    Ytestx <- Ytest[[fold]]
    Xtestx <- Xtest[[fold]]
    
    resx <- ADMM.nuclear(Y=Ytrax, X=Xtrax, B=NULL, L=NULL,
                         eta=eta, lambda = lambda,
                         maxiter=maxiter, tol=tol, standardize = TRUE)
    Bhat <- resx$L
    predY <- cbind(1,Xtestx) %*% Bhat
    avgerr <- mean( (Ytestx -predY)^2 )
    return( avgerr )
  })
  cverr <- unlist(cverr)
  cverrmean <- tapply(cverr, lamfold[,1], mean)
  lamopt <- lamseq[which.min(cverrmean)]
  errors <- cverr[lamfold[,1] == lamopt]
  ##########################################################################
  
  ############################# Fit model ##################################
  finalfit <- ADMM.nuclear(Y=Ycenter, X=Xstd, B=B, L=L, eta=eta,
                           lambda=lamopt, maxiter=maxiter, tol=tol,
                           standardize = FALSE)
  Bhat <- finalfit$L
  Slope <- 1/matrix(Xcenternorm,p,q) * Bhat
  Intercept <- matrix(Ymean, 1, q) -  matrix(Xmean,1,p) %*% Slope
  Bhat <- rbind(Intercept, Slope)
  ###########################################################################
  return( list(B=Bhat, d=finalfit$d, foldset=foldset, lamopt=lamopt,
               opterr=min(cverrmean), errrorsd= sd(errors)))
}

###########################################################################################################


################# Cross-validation for nuclear norm regulairzed regression #################################

cv.pooledNR <- function(Y,X,auxYlist,auxXlist,B,L,eta,lamseq,maxiter=100,tol=1e-04,nfold=5){
  # Y: Y^{(0)}
  # X: X^{(0)}
  # auxYlist[[k]]: Y^{(k)} for k=1,..., K
  # auxXlist[[k]]: X^{(k)} for k=1,..., K
  # B,L : A,C in the admm algorithm (Section S2.1)
  Yauxmat <- do.call("rbind", auxYlist)
  Xauxmat <- do.call("rbind", auxXlist)
  allY <- rbind(as.matrix(Y), do.call('rbind',auxYlist) )
  allX <- rbind(as.matrix(X), do.call('rbind',auxXlist) )
  cv.nuclear(Y=allY, X=allX, B=B,L=L, eta=eta, lamseq = lamseq,
             maxiter=maxiter, tol=tol, nfold=nfold) -> cvres
  return(cvres)
}

###########################################################################################################






##################### Cross-validation for two-step transfer learning  ####################################
cv.twostep <- function(Y,X,auxYlist,auxXlist,B,L,eta,
                                      lamseq_w, lamseq_delta,
                                      maxiter=100,tol=1e-04,nfold=5){
  # auxYlist[[k]]: Y^{(k)} for k=1,..., K
  # auxXlist[[k]]: X^{(k)} for k=1,..., K
  # B,L : A,C in the admm algorithm (Section S2.1)
  
  
  Y <- as.matrix(Y)
  X <- as.matrix(X)
  K <- length(auxYlist)
  q <- ncol(Y)
  p <- ncol(X)
  n <- nrow(X)
  
  ####### Centering each data to prevent performance degreatuion due to covariate shift ##############
  auxYcenterlist <- list()
  for(k in 1:K){
    auxY_k <- auxYlist[[k]]
    auxY_kmean <- apply(auxY_k,2,mean)
    auxY_cneter <- auxY_k - matrix(auxY_kmean, nrow(auxY_k), q,byrow=T)
    auxYcenterlist[[k]] <- auxY_cneter
  }
  auxXcenterlist <- list()
  for(k in 1:K){
    auxX_k <- auxXlist[[k]]
    nk <- nrow(auxX_k)
    auxX_kmean <- apply(auxX_k,2,mean)
    auxX_center <- auxX_k - matrix(auxX_kmean, nk, p,byrow=T)
    auxXcenterlist[[k]] <- auxX_center
  }
  ############################# Stacking ########################################
  auxX <- do.call("rbind", auxXcenterlist) # X^{A}
  auxY <- do.call("rbind", auxYcenterlist) # Y^{A}
  auxY <- as.matrix(auxY)
  auxX <- as.matrix(auxX)
  
  ########################## Centering target data #########################
  Xmean <- apply(X,2,mean)
  Xcenter <- X-matrix(Xmean,n,p,byrow=T)
  Ymean <- apply(Y,2,mean)
  Ycenter <- Y - matrix(Ymean,n,q,byrow=T)
  Xcenter <- as.matrix(Xcenter)
  Ycenter <- as.matrix(Ycenter)
  ##########################################################################
  
  
  #################### Standardization aggregate samples(Y^{0} U Y^{A}) ###################
  allY <- rbind(Ycenter,auxY)
  allX <- rbind(Xcenter,auxX)
  
  nall <- nrow(allY)
  allYmean <- apply(allY,2,mean)
  allYcenter <- allY-matrix(allYmean,byrow=T,nall,q)
  allXmean <- apply(allX,2,mean)
  allXcenter <- allX-matrix(allXmean,nall,p,byrow=T)
  
  #########################################################################################
  
  cv.nuclear(Y=allY, X=allX, B=B,L=L, eta=eta, lamseq = lamseq_w,
             maxiter=maxiter, tol=tol, nfold=nfold) -> cvW
  What <- cvW$B[-1,] # What where tuning paremter is detrimed by CV
  Yres <- Ycenter - Xcenter %*% What
  
  cv.nuclear(Y=Yres,X=Xcenter, B=B, L=L, eta=eta, lamseq=lamseq_delta,
             maxiter=maxiter, tol=tol, nfold=nfold) -> rescorrection
  
  Deltahat <- rescorrection$B[-1,] # Delta hat where tuning paremter is detrimed by CV
  intercept <-  matrix(Ymean, 1, q) -  matrix(Xmean,1,p) %*% (Deltahat+What) #B= Deltahat + What
  Slope <- Deltahat + What #B= Deltahat + What
  Bhat <- rbind(intercept, Slope)
  
  ###########################################################################
  return( list(B=Bhat, Deltahat=Deltahat, What=What,
               lamoptw = cvW$lamopt, lamoptdelta= rescorrection$lamopt))
}
############################################################################################################
