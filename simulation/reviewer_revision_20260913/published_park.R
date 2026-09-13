# I/O and observational diagnostics around unmodified authors' cv.twostep.
# The vendored implementation is hash-checked by the Python caller.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2L) stop("usage: published_park.R data_directory vendor_directory")
directory <- normalizePath(args[[1]], mustWork=TRUE)
vendor <- normalizePath(args[[2]], mustWork=TRUE)
base::options(digits=17)
if (!requireNamespace("jsonlite", quietly=TRUE)) stop("R package jsonlite is required")
if (!requireNamespace("corpcor", quietly=TRUE)) stop("R package corpcor is required")
options <- jsonlite::fromJSON(file.path(directory, "options.json"))
read_matrix <- function(name) as.matrix(read.csv(file.path(directory, paste0(name, ".csv")), header=FALSE))
X <- read_matrix("X")
Y <- read_matrix("Y")
X0 <- read_matrix("X0")
Y0 <- read_matrix("Y0")
author_environment <- new.env(parent=globalenv())
source(file.path(vendor, "Function", "Functions_naiveapproaches.R"), local=author_environment)

# The author's coefficient calculations and CV are unchanged. This wrapper
# records the original solver's returned update residuals for every invocation.
original_admm <- author_environment$ADMM.nuclear
admm_audit <- list()
author_environment$ADMM.nuclear <- function(...) {
  input <- list(...)
  result <- original_admm(...)
  last_update <- max(tail(result$errB, 1), tail(result$errL, 1), tail(result$errU, 1))
  audit <- list(lambda=input$lambda, iterations=as.integer(result$iter),
                max_iter=as.integer(input$maxiter), tolerance=input$tol,
                last_max_squared_update=last_update,
                stopping_criterion_met=is.finite(last_update) && last_update <= input$tol,
                returned_scale_squared_B_minus_L=sum((result$B - result$L)^2),
                authors_last_objective=tail(result$obj, 1))
  # When standardize=FALSE the returned L is in the objective's exact input
  # coordinates, so observe an actual primal/feasible-dual objective pair.
  # This does not change the original solver's stopping rule or its iterates.
  if (identical(input$standardize, FALSE)) {
    residual <- input$Y - input$X %*% result$L
    nfit <- nrow(input$X)
    primal <- sum(residual^2)/(2*nfit) + input$lambda*sum(svd(result$L, nu=0, nv=0)$d)
    dual <- residual/nfit
    dualnorm <- max(svd(t(input$X) %*% dual, nu=0, nv=0)$d)
    if (dualnorm > input$lambda && dualnorm > 0) dual <- dual*(input$lambda/dualnorm)
    dualobjective <- sum(input$Y*dual) - nfit*sum(dual^2)/2
    audit$standardized_primal_objective <- primal
    audit$feasible_dual_objective <- dualobjective
    audit$duality_gap <- max(0, primal-dualobjective)
    audit$duality_gap_scope <- "normalized_nuclear_objective_on_supplied_standardized_design"
  }
  admm_audit[[length(admm_audit) + 1L]] <<- audit
  result
}

# Fixed-lambda final-fit portion of the authors' cv.nuclear, with exactly the
# original arithmetic order for centering, standardization, rescaling and the
# intercept. Inner-CV fits are omitted because a singleton grid cannot select a
# different lambda. The original ADMM.nuclear coefficient algorithm is called.
fixed_nuclear <- function(Y, X, lambda) {
  Y <- as.matrix(Y)
  X <- as.matrix(X)
  n <- nrow(Y)
  p <- ncol(X)
  q <- ncol(Y)
  Ymean <- apply(Y,2,mean)
  Ycenter <- Y-matrix(Ymean,byrow=TRUE,n,q)
  Xmean <- apply(X,2,mean)
  Xcenter <- X-matrix(Xmean,n,p,byrow=TRUE)
  Xcenternorm <- apply(Xcenter,2,function(x) sqrt(sum(x^2)/n))
  Xstd <- Xcenter/matrix(Xcenternorm,n,p,byrow=TRUE)
  if (any(!is.finite(Xstd))) stop("zero-variance training predictor under authors' scaling")
  finalfit <- author_environment$ADMM.nuclear(
    Y=Ycenter,X=Xstd,B=NULL,L=NULL,eta=options$eta,lambda=lambda,
    maxiter=options$max_iter,tol=options$tolerance,standardize=FALSE)
  Bhat <- finalfit$L
  Slope <- 1/matrix(Xcenternorm,p,q)*Bhat
  Intercept <- matrix(Ymean,1,q)-matrix(Xmean,1,p)%*%Slope
  list(B=rbind(Intercept,Slope),admm_index=length(admm_audit))
}

candidate_results <- list()
validation_diagnostics <- list()
external_validation_fit <- function() {
  Xv <- read_matrix("Xv")
  Yv <- read_matrix("Yv")
  n <- nrow(X)
  p <- ncol(X)
  q <- ncol(Y)
  # Identical one-source preparation to original cv.twostep.
  auxYmean <- apply(Y0,2,mean)
  auxY <- Y0-matrix(auxYmean,nrow(Y0),q,byrow=TRUE)
  auxXmean <- apply(X0,2,mean)
  auxX <- X0-matrix(auxXmean,nrow(X0),p,byrow=TRUE)
  Xmean <- apply(X,2,mean)
  Xcenter <- as.matrix(X-matrix(Xmean,n,p,byrow=TRUE))
  Ymean <- apply(Y,2,mean)
  Ycenter <- as.matrix(Y-matrix(Ymean,n,q,byrow=TRUE))
  allY <- rbind(Ycenter,as.matrix(auxY))
  allX <- rbind(Xcenter,as.matrix(auxX))
  best <- NULL
  best_loss <- Inf
  best_index <- NULL
  for (lambda_w in options$lambda_w) {
    pooled_started <- proc.time()[["elapsed"]]
    pooled <- tryCatch(fixed_nuclear(allY,allX,lambda_w),error=function(e)e)
    pooled_elapsed <- proc.time()[["elapsed"]]-pooled_started
    for (lambda_delta in options$lambda_delta) {
      candidate_index <- length(candidate_results)
      record <- list(candidate_index=candidate_index,lambda_w=lambda_w,
                     lambda_delta=lambda_delta,pooled_fit_seconds=pooled_elapsed)
      if (inherits(pooled,"error")) {
        record$status <- "failed"
        record$error <- conditionMessage(pooled)
        record$failure_stage <- "pooled_nuclear_fit"
        candidate_results[[candidate_index+1L]] <<- record
        next
      }
      correction_started <- proc.time()[["elapsed"]]
      candidate <- tryCatch({
        What <- pooled$B[-1,,drop=FALSE]
        Yres <- Ycenter-Xcenter%*%What
        correction <- fixed_nuclear(Yres,Xcenter,lambda_delta)
        Deltahat <- correction$B[-1,,drop=FALSE]
        Slope <- Deltahat+What
        intercept <- matrix(Ymean,1,q)-matrix(Xmean,1,p)%*%Slope
        Bhat <- rbind(intercept,Slope)
        loss <- sum((Yv-cbind(1,Xv)%*%Bhat)^2)/(2*nrow(Xv))
        if (any(!is.finite(Bhat)) || !is.finite(loss)) stop("nonfinite candidate or validation loss")
        list(B=Bhat,lamoptw=lambda_w,lamoptdelta=lambda_delta,loss=loss,
             correction_admm_index=correction$admm_index)
      },error=function(e)e)
      record$correction_fit_and_validation_seconds <- proc.time()[["elapsed"]]-correction_started
      record$pooled_admm_index <- pooled$admm_index
      if (inherits(candidate,"error")) {
        record$status <- "failed"
        record$error <- conditionMessage(candidate)
        record$failure_stage <- "correction_nuclear_fit"
      } else {
        record$validation_loss <- candidate$loss
        record$correction_admm_index <- candidate$correction_admm_index
        record$pooled_stopping_criterion_met <- admm_audit[[pooled$admm_index]]$stopping_criterion_met
        record$correction_stopping_criterion_met <- admm_audit[[candidate$correction_admm_index]]$stopping_criterion_met
        converged <- record$pooled_stopping_criterion_met && record$correction_stopping_criterion_met
        record$status <- if (!isTRUE(options$require_convergence) || converged) "ok" else "uncertified"
        if (record$status == "uncertified") record$exclusion_reason <- "authors_ADMM_stopping_criterion_not_met"
        if (record$status == "ok" && candidate$loss < best_loss) {
          best <- candidate
          best_loss <- candidate$loss
          best_index <- candidate_index
        }
      }
      candidate_results[[candidate_index+1L]] <<- record
    }
  }
  if (is.null(best)) stop("no eligible externally validated Park candidate; inspect candidate_results")
  selected_record <- candidate_results[[best_index+1L]]
  selected_pooled_audit <- admm_audit[[selected_record$pooled_admm_index]]
  selected_correction_audit <- admm_audit[[selected_record$correction_admm_index]]
  validation_diagnostics <<- list(
    candidate_results=candidate_results,selected_index=best_index,validation_loss=best_loss,
    candidate_count=length(candidate_results),selection_tie_rule="first_in_increasing_lambda_w_then_delta_order",
    pooled_fit_reuse="one_pooled_fit_per_lambda_w_shared_across_correction_grid",
    selected_admm_stopping_criteria_met=selected_pooled_audit$stopping_criterion_met && selected_correction_audit$stopping_criterion_met,
    selected_pooled_duality_gap=selected_pooled_audit$duality_gap,
    selected_correction_duality_gap=selected_correction_audit$duality_gap,
    timing_note="pooled_fit_seconds_repeated_in_records_do_not_sum_across_delta_candidates",
    preprocessing_scope="target_training_and_raw_source_only_validation_rows_excluded",
    refit=FALSE,standardization="authors_final_fit_transformations_without_unused_inner_CV")
  best
}

started <- proc.time()[["elapsed"]]
result <- tryCatch(
  if (identical(options$mode,"external_validation")) external_validation_fit() else author_environment$cv.twostep(
    Y=Y, X=X, auxYlist=list(Y0), auxXlist=list(X0), B=NULL, L=NULL,
    eta=options$eta, lamseq_w=options$lambda_w,
    lamseq_delta=options$lambda_delta, maxiter=options$max_iter,
    tol=options$tolerance, nfold=options$nfold),
  error=function(error) {
    jsonlite::write_json(list(status="failed", error=conditionMessage(error),
                              admm_fits=admm_audit,candidate_results=candidate_results),
                        file.path(directory, "diagnostics.json"), auto_unbox=TRUE,
                        pretty=TRUE, digits=17, null="null")
    stop(error)
  }
)
elapsed <- proc.time()[["elapsed"]] - started
coefficient <- result$B[-1, , drop=FALSE]
intercept <- result$B[1, , drop=FALSE]
write.table(coefficient, file.path(directory, "coefficient.csv"), sep=",", row.names=FALSE, col.names=FALSE)
write.table(intercept, file.path(directory, "intercept.csv"), sep=",", row.names=FALSE, col.names=FALSE)
diagnostics <- list(
  r_elapsed_seconds=elapsed, selected_lambda_w=result$lamoptw,
  selected_lambda_delta=result$lamoptdelta,
  admm_fit_count=length(admm_audit), admm_fits=admm_audit,
  all_admm_stopping_criteria_met=all(vapply(admm_audit, function(x) x$stopping_criterion_met, logical(1))),
  admm_iteration_cap_count=sum(!vapply(admm_audit, function(x) x$stopping_criterion_met, logical(1))),
  convergence_scope="authors_squared_B_L_U_update_rule_not_a_duality_gap_certificate",
  standardization="unmodified_authors_internal_training_fold_centering_and_scaling",
  intercept_fitted=TRUE, r_version=R.version.string,
  r_packages=list(corpcor=as.character(packageVersion("corpcor")), jsonlite=as.character(packageVersion("jsonlite"))),
  rng_note="authors_cv_nuclear_resets_R_seed_to_100_for_each_internal_CV"
)
if (identical(options$mode,"external_validation")) {
  diagnostics <- modifyList(diagnostics,validation_diagnostics)
  diagnostics$rng_note <- "fixed_lambda_final_fits_do_not_use_R_randomness"
}
jsonlite::write_json(diagnostics, file.path(directory, "diagnostics.json"),
                    auto_unbox=TRUE, pretty=TRUE, digits=17, null="null")
