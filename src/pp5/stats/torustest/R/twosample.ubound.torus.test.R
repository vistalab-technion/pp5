#' Asymptotic two-sample goodness-of-fit on the torus
#'
#' Performs a two-sample goodness-of-fit test for measures supported on the torus, using a p-value upper bound. The
#' test is asymptotically consistent at level alpha.
#'
#' @param sample_1 n x 2 matrix containing n observations in the two-dimensional flat torus, parametrized as the periodic [0,1) x [0,1).
#' @param sample_2 n x 2 matrix containing n observations in the two-dimensional flat torus, parametrized as the periodic [0,1) x [0,1).
#' @param return_stat If TRUE, both the the test statistic and pvalue are returned, otherwise only the pvalue is returned.
#'
#' @return The p-value for the two-sample test on the torus.
#'
#' @examples
#'
#' n <- 2000 # Sample size
#'
#' set.seed(10)
#'  # Bivariate von Mises distribution
#' samp_1 <- BAMBI::rvmcos(n, kappa1 = 1, kappa2 = 1, mu1 = 0, mu2 = 0)/(2*pi)
#' samp_2 <- BAMBI::rvmcos(n, kappa1 = 1, kappa2 = 1, mu1 = 0, mu2 = 0)/(2*pi)
#' twosample.ubound.torus.test(samp_1, samp_2)
#'
#' samp_1 <- BAMBI::rvmcos(n ,kappa1 = 0, kappa2 = 0, mu1 = 0.5, mu2 = 0.5)/(2*pi)
#' samp_2 <- BAMBI::rvmcos(n, kappa1 = 1, kappa2 = 1, mu1 = 0.5, mu2 = 0.5)/(2*pi)
#' twosample.ubound.torus.test(samp_1, samp_2)
#'
#' @export

twosample.ubound.torus.test <- function(sample_1, sample_2, return_stat=FALSE){

  n <- nrow(sample_1); m <- nrow(sample_2)
  costmatrix <- proxy::dist(x = sample_1, y = sample_2, method = dist.torus, diag = TRUE)  # Cost matrix
  wdis <- transport::wasserstein(rep(1/n, n), rep(1/m, m), costm = costmatrix, p = 2, method = 'networkflow') # Wasserstein distance

  u_bound <- function(t, n, m){
    exp(-8 * t^2 * m * n / (n+m))
  }

  pval <- u_bound(wdis^2, n, m) #p-value

  if(return_stat){
      return(list(pval=pval, stat=wdis))
  }
  else{
      return(pval)
  }

}
