
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
#' # Uniform distribution on [0,1] x [0,1]
#' x <- uniformly::runif_in_cube(1, 2, c(0.5, 0.5), 0.5) 
#' y <- uniformly::runif_in_cube(1, 2, c(0.5, 0.5), 0.5)
#' dist.torus(x, y)
#' 
#' # Bivariate von Mises distribution
#' x <- uniformly::runif_in_cube(1, 2, c(0.5, 0.5), 0.5) 
#' y <- BAMBI::rvmcos(1, kappa1 = 1, kappa2 = 1, mu1 = 0, mu2 = 0)/(2*pi)
#' dist.torus(x, y)
#' 
#' @export

dist.torus<-function(x, y){
  
  x1 <- x[1]; x2 <- x[2]; y1 <- y[1]; y2 <- y[2]
  dis <- sqrt(pmin(abs(x1 - y1), 1 - abs(x1 - y1))^2 + pmin(abs(x2 - y2), 1 - abs(x2 - y2))^2)
  return(dis)
  
}
