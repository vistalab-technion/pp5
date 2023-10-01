#' Closed geodesics sampling
#'
#' Samples a number of different closed geodesics on the torus. Each geodesic is represented by a vector of integers (B, A),
#' where B follows a geometric distribution of parameter p and, for a given B=b, A is uniform in (0,1,...,b).
#'
#' @param ng The size of the geodesics sample.
#' @param p The parameter p of the geometric distribution of B.
#'
#' @return A ng x 2 matrix containing the vectors corresponding to the sampled closed geodesics.
#'
#' @details For big sample sizes, the value of p must be reduced in order to get a sample of different
#' geodesics. This yields to less simple geodesics, with more revolutions over the torus.
#'
#' @references [1] Petroni, N. C.(2019). Taking rational numbers at random. arXiv:1908.06944v1.
#'
#' @examples
#'
#' rgeodesic(5, p = 0.1)
#'
#' # For bigger sample sizes, p must be reduced.
#' rgeodesic(100, p = 0.01)
#'
#' @export


rgeodesic <- function(ng, p = 0.1){

  rgeodesic_one <- function(p){

    m <- 1 + stats::rgeom(1, p)
    n <- sample(0:m, size = 1)
    u <- c(n, m) / FRACTION::gcd(n, m)

    oct <- sample(1:4, size = 1)
    if(oct == 1){u <- c(-u[2], u[1])}
    if(oct == 2){u[1] <- -u[1]}
    if(oct == 4){u <- c(u[2], u[1])}

    return(u)

  }

  gsamp <- c(0, 0)
  while(sum(duplicated(gsamp)) > 0){ # Choose only samples with ng different geodesics
    gsamp <- t(replicate(ng, rgeodesic_one(p)))}

  return(gsamp)

}
