#' Asymptotic two-sample goodness-of-fit on the torus
#'
#' Performs a two-sample goodness-of-fit test for measures supported on the torus, using a p-value upper bound. The
#' test is asymptotically consistent at level alpha.
#'
#' @param sample_1 n x 2 matrix containing n observations in the two-dimensional flat torus, parametrized as the periodic [0,1) x [0,1).
#' @param sample_2 n x 2 matrix containing n observations in the two-dimensional flat torus, parametrized as the periodic [0,1) x [0,1).
#'
#' @return The p-value for the two-sample test on the torus.
#'
#' @examples
#'
#' n <- 2000 # Sample size
#'
#' set.seed(10)
#' samp_1 <- BAMBI::rvmcos(n, kappa1=1, kappa2=1, mu1=0, mu2=0)/(2*pi) # Bivariate von Mises distribution
#' samp_2 <- BAMBI::rvmcos(n, kappa1=1, kappa2=1, mu1=0, mu2=0)/(2*pi)
#' twosample.ubound.torus.test(samp_1, samp_2)
#' 0.9963195
#'
#' samp_1 <- BAMBI::rvmcos(n ,kappa1=0, kappa2=0, mu1=0.5, mu2=0.5)/(2*pi)
#' samp_2 <- BAMBI::rvmcos(n, kappa1=1, kappa2=1, mu1=0.5, mu2=0.5)/(2*pi)
#' twosample.ubound.torus.test(samp_1, samp_2)
#' 0.02360551
#'
#' @export
twosample.ubound.torus.test <- function(sample_1, sample_2){
  n <- nrow(sample_1); m <- nrow(sample_2)
  costmatrix <- proxy::dist(x = sample_1, y = sample_2, method = dist.torus, diag = TRUE)  # Cost matrix
  wdis <- transport::wasserstein(rep(1/n,n), rep(1/m,m), costm = costmatrix, p=2, method='networkflow') # Wasserstein distance

  u_bound <- function(t, n, m){
    exp(-8 * t^2 * m * n / (n+m))
  }
  return(u_bound(wdis^2, n, m)) #p-value
}

#' Distance on the two-dimensional torus
#'
#' Distance between two points on the two-dimensional torus (periodic [0,1] x [0,1])
#'
#' @param x A vector of two coordinates in the two-dimensional flat torus, parameterized as the periodic [0,1) x [0,1).
#' @param y A vector of two coordinates in the two-dimensional flat torus, parameterized as the periodic [0,1) x [0,1).
#'
#' @return The distance on the torus between x and y
#'
#' @examples
#' set.seed(10)
#' x <- uniformly::runif_in_cube(1,2,c(0.5,0.5),0.5) # Uniform distribution on [0,1] x [0,1]
#' y <- uniformly::runif_in_cube(1,2,c(0.5,0.5),0.5)
#' dist.torus(x, y)
#' 0.3946457
#'
#' x <- uniformly::runif_in_cube(1,2,c(0.5,0.5),0.5)
#' y <- BAMBI::rvmcos(1, kappa1=1, kappa2=1, mu1=0, mu2=0)/(2*pi) # Bivariate von Mises distribution
#' dist.torus(x, y)
#' 0.5014811
#'
#' @export
dist.torus<-function(x, y){
  x1 <- x[1]; x2 <- x[2]; y1 <- y[1]; y2 <- y[2]
  dis <- sqrt(pmin(abs(x1 - y1), 1 - abs(x1 - y1))^2 + pmin(abs(x2 - y2), 1 - abs(x2 - y2))^2)
  return(dis)

}
