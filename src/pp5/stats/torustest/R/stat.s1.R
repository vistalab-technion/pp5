#' Wasserstein distribution-free statistic on S^1
#'
#' Computes Wasserstein statistic between two samples on S^1, parameterized as a periodic [0,1).
#'
#' @param x numerical vector of sample in [0,1).
#' @param y numerical vector of sample in [0,1).
#'
#' @return The statistic realization of samples x, y.
#'
#' @references [1] Delon, J., Salomon, J., Sobolevskii, A.: Fast transport optimization for Monge costs on the circle. SIAM J. Appl. Math. 70(7), 2239–2258 (2010).
#' [2] RAMDAS, A., GARCIA, N. and CUTURI, M. (2015). On Wasserstein Two Sample Testing and Related Families of Nonparametric Tests. Entropy 19.
#'
#' @examples
#' set.seed(10)
#' stat.s1(runif(50), runif(50))
#' stat.s1(runif(50), as.numeric(circular::rvonmises(50, pi, 1)/(2*pi)))
#'
#' @export

stat.s1 <- function(x, y){

  fy <- stats::ecdf(y)
  h <- function(t){(fy(stats::quantile(x, t)) - t)}
  optimal_alpha <- pracma::integral(h, 0, 1)

  h_alpha <- function(t){(fy(stats::quantile(x, t)) - t - optimal_alpha)^2}

  n <- length(x);  m <- length(y)
  return(n*m / (n + m) * pracma::integral(h_alpha, 0, 1))

}

