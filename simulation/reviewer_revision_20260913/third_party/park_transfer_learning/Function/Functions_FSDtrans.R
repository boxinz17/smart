
library(corpcor)       # fast.svd()
library(foreach)       # foreach loop (used for parallel computation)
library(doParallel)    # for registering parallel backends
library(splitTools)    # create_folds()


FSDtrans <- function(Y,X,auxYlist,auxXlist,B,L,eta,
                     lamseq_w, lamseq_delta,
                     maxiter=100,tol=1e-04,nfold=5,nfold_choiceforC=3,
                     nfold_selectionstep_pe=3, C){
  
  
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
  
  traind <- 1:nrow(Y)
  set.seed(100)
  folds <- splitTools::create_folds(traind,k=nfold_choiceforC, type = "basic")
  auxvec <- 1:length(auxYlist)
  # to save validation prediction error for each of C
  errlist_forward <- list() # Forward source detection
  for(k1 in 1:nfold_choiceforC){
    Y_tra <- Y[folds[[k1]],]
    Y_val <- Y[-folds[[k1]],]
    
    X_tra <- X[folds[[k1]],]
    X_val <- X[-folds[[k1]],]
    
    ####################### Creating folds that will be used to select sources  ##############################
    set.seed(100)
    ni <- nrow(Y)
    indi <- (1:ni)
    Ytraset <- Xtraset <- list()
    Ytestset <- Xtestset <- list()
    traind_k1 <- 1:nrow(Y_tra)
    folds_k1 <- splitTools::create_folds(traind_k1,k=nfold_selectionstep_pe, type = "basic")
    for(ss in 1:nfold_selectionstep_pe){
      Ytestset[[ss]] <- Y_tra[-folds_k1[[ss]],]
      Ytraset[[ss]] <-  Y_tra[folds_k1[[ss]],]
      
      Xtestset[[ss]] <- X_tra[-folds_k1[[ss]],]
      Xtraset[[ss]] <-  X_tra[folds_k1[[ss]],]
      
    }
    ##########################################################################################################
    
    
    ################################## To compute validation errors for  NR ##########################################################################
    p <- ncol(Xtraset[[ss]])
    q <- ncol(Ytraset[[ss]])
    lassoerr_vec <- foreach::foreach(ss=1:nfold_selectionstep_pe) %dopar%{
      Lasso_val <-    cv.nuclear(Y=Ytraset[[ss]], X=Xtraset[[ss]], B=NULL,
                                 L=NULL,eta=eta,lamseq = lamseq_w, tol=tol,maxiter=maxiter)
      mean((Ytestset[[ss]]  - cbind(1,Xtestset[[ss]])  %*%  Lasso_val$B )^2)
    }
    lassoerr_vec <- unlist(lassoerr_vec) # validation errors when using only target data
    
    
    cv.nuclear(Y=Y_tra, X=X_tra, B=NULL,
               L=NULL,eta=eta,lamseq = lamseq_w,
               tol=tol,maxiter=maxiter) -> cvLasso_tra
    predLassoi <- cbind(1,X_val) %*% cvLasso_tra$B
    Lassoerr_tra <- mean( (Y_val-predLassoi)^2 )
    ######################################################################################################################################################
    
    ################### Computing validation errors when each source is added ###############################################
    
    foldbyaux <- expand.grid(1:nfold_selectionstep_pe, 1:length(auxvec))
    foldbyaux <- as.matrix(foldbyaux)
    nauxfold <- nrow(foldbyaux)
    
    errorvec <- foreach::foreach(ss=1:nauxfold) %dopar%{
      
      fold <- foldbyaux[ss,1]
      g <- foldbyaux[ss,2]
      
      Y_ss <- rbind(Ytraset[[fold]], auxYlist[[ g ]] )
      X_ss <- rbind(Xtraset[[fold]], auxXlist[[ g ]] )
      
      
      cv.nuclear(Y=Y_ss, X=X_ss, B=NULL,
                 L=NULL,eta=eta,lamseq = lamseq_w,
                 tol=tol,maxiter=maxiter) -> cvLasso_ss
      
      mean((Ytestset[[fold]] - cbind(1,Xtestset[[fold]]) %*% cvLasso_ss$B)^2)
    }
    errorvec <- unlist(errorvec)
    errorvecmean <- tapply(errorvec, foldbyaux[,2], mean) # K-dimensinal vector: the kth component correspond to CV errors when combining target with the kth source
    
    sellist_list <- list() # to save source
    errlist <- list() # to save validation error correpoding to each of candiadtes of C
    
    
    # errorvecmean:  K-dimensinal vector: the kth component correspond to CV errors when combining target with the kth source
    errorvecmean_init <- errorvecmean
    for(sss in 1:length(C)){
      errorvecmean <- errorvecmean_init 
      detection <- any(errorvecmean < ((1+C[sss])*mean(lassoerr_vec))) # TRUE: go to the next stage (finidng an additional source)
      sources_cands <- auxvec
      auxYcand <- auxXcand <- NULL
      if(detection){
        auxvec_old <- sources_cands
        sources <- auxvec[which.min(errorvecmean)] # a detected source
        sources_cands <- auxvec[-which.min(errorvecmean)] # cancdidates 
        auxYcand <- auxYlist[[which.min(errorvecmean)]]
        auxXcand <- auxXlist[[which.min(errorvecmean)]]
        
        ############### forward step: from line 3 to line 16 in Algorithm 2 in the main paper ############### 
        for(kkk in 1:(length(auxvec)-1)){
          
          
          cri_err <- errorvecmean[which.min(errorvecmean)]
          foldbyaux2 <- expand.grid(1:nfold_selectionstep_pe, sources_cands  )
          foldbyaux2 <- as.matrix(foldbyaux2)
          nauxfold2 <- nrow(foldbyaux2)
          errorvec <- foreach::foreach(ss=1:nauxfold2) %dopar%{
            
            fold <- foldbyaux2[ss,1]
            g <- foldbyaux2[ss,2]
            auxYcand_g <- rbind(auxYcand, auxYlist[[ g ]])
            auxXcand_g <- rbind(auxXcand, auxXlist[[ g ]])
            
            Y_ss <- rbind(Ytraset[[fold]], auxYcand_g )
            X_ss <- rbind(Xtraset[[fold]], auxXcand_g )
            
            val_g <- cv.nuclear(Y=Y_ss, X=X_ss, B=NULL,L=NULL,eta=eta,lamseq = lamseq_w,
                                tol=tol,maxiter=maxiter)
            
            mean((Ytestset[[fold]]  - cbind(1,Xtestset[[fold]])  %*%  val_g$B )^2)
          }
          errorvec <- unlist(errorvec)
          errorvecmean <- tapply(errorvec, foldbyaux2[,2], mean) # a componenet is an validation error when adding each of candidate sources
          if(all( errorvecmean > ( (1+C[sss]) * cri_err)  ) ){
            break
          }else{
            
            sources <- c(sources, sources_cands[which.min(errorvecmean)])
            auxYcand <- rbind(auxYcand, auxYlist[[ sources_cands[which.min(errorvecmean)] ]])
            auxXcand <- rbind(auxXcand, auxXlist[[ sources_cands[which.min(errorvecmean)] ]])
            sources_cands <- sources_cands[-which.min(errorvecmean)]
          }
          
        }
        #################################################################################################
        
        selist <- sources
        auxvec1 <- (1:length(auxvec))[which(auxvec %in% selist )]
        auxYlist1 <- auxXlist1 <- list()
        if(length(auxvec1)==0){
          cv1err <- Lassoerr_tra # when anay source isn't selected 
        }else{
          for(g in 1:length(auxvec1)){
            auxYlist1[[g]] <- auxYlist[[ auxvec1[g] ]] # saving detected sources
            auxXlist1[[g]] <- auxXlist[[ auxvec1[g] ]]
          }
        }
        
        cv1 <- cv.twostep(Y=Y_tra, X=X_tra, B=NULL,
                          L=NULL,eta=eta,
                          lamseq_w = lamseq_w,
                          lamseq_delta=lamseq_delta,
                          tol=tol,maxiter=maxiter,
                          auxYlist=auxYlist1, auxXlist=auxXlist1)
        cv1_pred <- cbind(1,X_val) %*% cv1$B
        cv1err <-   mean( (Y_val -    cv1_pred)^2 )
        errlist[sss] <- cv1err
        sellist_list[[sss]] <- sources
        
        
      }else{
        errlist[sss] <- Lassoerr_tra
        sellist_list[[sss]] <- NULL
      }
      
    }
    
    errlist_forward[[k1]] <- errlist
    
    
  }
  ###################################################################################################################################################
  
  
  ############################################# Source detection step based on optimal C ###############################################################################
  err_forward_mean <- apply(do.call('rbind', lapply(errlist_forward, unlist)),2,mean)
  ############# Dividing data to perform source selections based on the optimal C ############
  Ytraset <- Xtraset <- list()
  Ytestset <- Xtestset <- list()
  for(ss in 1:nfold_choiceforC){
    Ytestset[[ss]] <- Y[-folds[[ss]],]
    Ytraset[[ss]] <-  Y[folds[[ss]],]
    
    Xtestset[[ss]] <- X[-folds[[ss]],]
    Xtraset[[ss]] <-  X[folds[[ss]],]
    
  }
  ###########################################################################################
  
  
  ########################## Source detection ###############################################
  foldbyaux <- expand.grid(1:nfold_choiceforC, 1:length(auxvec))
  foldbyaux <- as.matrix(foldbyaux)
  nauxfold <- nrow(foldbyaux)
  
  errorvec <- foreach::foreach(ss=1:nauxfold) %dopar%{
    
    fold <- foldbyaux[ss,1]
    g <- foldbyaux[ss,2]
    
    Y_ss <- rbind(Ytraset[[fold]], auxYlist[[ g ]] )
    X_ss <- rbind(Xtraset[[fold]], auxXlist[[ g ]] )
    
    cv.nuclear(Y=Y_ss, X=X_ss, B=NULL,
               L=NULL,eta=eta,lamseq = lamseq_w,
               tol=tol,maxiter=maxiter) -> cvLasso_ss
    
    mean((Ytestset[[fold]] - cbind(1,Xtestset[[fold]]) %*% cvLasso_ss$B)^2)
  }
  errorvec <- unlist(errorvec)
  errorvecmean <- tapply(errorvec, foldbyaux[,2], mean)
  
  
  lassoerr_vec <- foreach::foreach(ss=1:3) %dopar%{
    Lasso_val <-    cv.nuclear(Y=Ytraset[[ss]], X=Xtraset[[ss]], B=NULL,
                               L=NULL,eta=eta,lamseq = lamseq_w, tol=tol,maxiter=maxiter)
    mean((Ytestset[[ss]]  - cbind(1,Xtestset[[ss]])  %*%  Lasso_val$B )^2)
  }
  lassoerr_vec <- unlist(lassoerr_vec)
  
  ############################# Forward source detection #####################################
  optC_sequence <-  C[which.min( unlist(err_forward_mean) )]
  detection <- any(errorvecmean < ((1+optC_sequence)*mean(lassoerr_vec)))
  sources_cands <- auxvec
  auxYcand <- auxXcand <- NULL
  selist_opt <- NULL
  if(detection){
    auxvec_old <- sources_cands
    sources <- auxvec[which.min(errorvecmean)]
    sources_cands <- auxvec[-which.min(errorvecmean)]
    #auxvec_new <- auxvec_new[-which.min(errorvecmean)]
    
    auxYcand <- auxYlist[[which.min(errorvecmean)]]
    auxXcand <- auxXlist[[which.min(errorvecmean)]]
    selist_opt <- sources
    
    for(kkk in 1:(length(auxvec)-1)){
      
      cri_err <- errorvecmean[which.min(errorvecmean)]
      foldbyaux2 <- expand.grid(1:nfold_selectionstep_pe, sources_cands  )
      foldbyaux2 <- as.matrix(foldbyaux2)
      nauxfold2 <- nrow(foldbyaux2)
      errorvec <- foreach::foreach(ss=1:nauxfold2) %dopar%{
        
        fold <- foldbyaux2[ss,1]
        g <- foldbyaux2[ss,2]
        auxYcand_g <- rbind(auxYcand, auxYlist[[ g ]])
        auxXcand_g <- rbind(auxXcand, auxXlist[[ g ]])
        
        Y_ss <- rbind(Ytraset[[fold]], auxYcand_g )
        X_ss <- rbind(Xtraset[[fold]], auxXcand_g )
        
        val_g <- cv.nuclear(Y=Y_ss, X=X_ss, B=NULL,L=NULL,eta=eta,lamseq = lamseq_w,
                            tol=tol,maxiter=maxiter)
        
        mean((Ytestset[[fold]]  - cbind(1,Xtestset[[fold]])  %*%  val_g$B )^2)
      }
      errorvec <- unlist(errorvec)
      errorvecmean <- tapply(errorvec, foldbyaux2[,2], mean)
      if(all( errorvecmean > ( (1+optC_sequence) * cri_err)  ) ){
        break
      }else{
        sources <- c(sources, sources_cands[which.min(errorvecmean)])
        auxYcand <- rbind(auxYcand, auxYlist[[ sources_cands[which.min(errorvecmean)] ]])
        auxXcand <- rbind(auxXcand, auxXlist[[ sources_cands[which.min(errorvecmean)] ]])
        sources_cands <- sources_cands[-which.min(errorvecmean)]
        selist_opt <- sources
      }
      
    }
  }
  
  auxvec1_opt <- (1:length(auxvec))[auxvec %in% selist_opt ]
  if(length(auxvec1_opt)==0){
    What_1_opt <- cvLasso$B
    Bhat_1_opt <- cvLasso$B
    Delthat_1_opt <- matrix(0,p,q)
    lamopt_1_W <- cvLasso$lamopt
    lamoptdelta_1 <- 0
    
  }else{
    auxYlist1 <- auxXlist1 <- list()
    for(g in 1:length(auxvec1_opt)){
      auxYlist1[[g]] <- auxYlist[[ auxvec1_opt[g] ]]
      auxXlist1[[g]] <- auxXlist[[ auxvec1_opt[g] ]]
    }
    cv1_opt <- cv.twostep(Y=Y, X=X, B=NULL,
                          L=NULL,eta=eta,
                          lamseq_w = lamseq_w,
                          lamseq_delta=lamseq_delta,
                          tol=tol,maxiter=maxiter,
                          auxYlist=auxYlist1, auxXlist=auxXlist1)
    What_1_opt <- cv1_opt$W
    Delthat_1_opt <- cv1_opt$Delta
    Bhat_1_opt <- cv1_opt$B
    lamopt_1_W <- cv1_opt$lamoptw
    lamoptdelta_1 <- cv1_opt$lamoptdelta
  }
  
  return( list(B=Bhat_1_opt, Deltahat=Delthat_1_opt, What=What_1_opt,
               lamoptw = lamopt_1_W, lamoptdelta= lamoptdelta_1,
               detectedsources=auxvec1_opt))
}
