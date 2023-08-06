#' Two-sample goodness-of-fit on the circle
#'
#' Performs a two-sample goodness-of-fit test for measures supported on the circle, based on a
#' distribution-free Wasserstein statistic.
#'
#' @param x numerical vector of sample in [0,1).
#' @param y numerical vector of sample in [0,1).
#' @param sim_null The simulated null distribution of the statistic. If NULL, the distribution is simulated with the given parameters (very time consuming).
#' @param NR The number of replicas if simulation is required.
#' @param NC The number of cores if parallel simulation is required.
#' @param n The sample sizes of the simulated samples is simulation is required.
#' 
#' @return
#' \itemize{
#'   \item stat - The test statistic.
#'   \item pvalue - The test p-value.
#' }
#'
#' @references [1] Delon, J., Salomon, J., Sobolevskii, A.: Fast transport optimization for Monge costs on the circle. SIAM J. Appl. Math. 70(7), 2239–2258 (2010).
#' [2] RAMDAS, A., GARCIA, N. and CUTURI, M. (2015). On Wasserstein Two Sample Testing and Related Families of Nonparametric Tests. Entropy 19.
#' 
#' @examples
#' 
#' n <- 50 # Sample size
#' 
#' # Simulate the statistic null distribution
#' NR <- 100
#' sim_free_null <- sim.null.stat(500, NC = 2)
#' 
#' x <- runif(n, 0, 1)
#' y <- runif(n, 0, 1)
#' twosample.test.s1(x, y, sim_free_null) 
#' 
#' x <- as.numeric(circular::rvonmises(n, pi, 1)/(2*pi))
#' y <- as.numeric(circular::rvonmises(n, pi, 0)/(2*pi))
#' twosample.test.s1(x, y, sim_free_null) 
#' 
#' @export

twosample.test.s1 <- function(x, y, sim_null = NULL, NR = 500, NC = 1, n = 30){
  
  if(is.null(sim_null)){
    
    cat('No null distribution given as an argument. Simulating with default parameters...\n')
    sim_null <- sim.null.stat(NR, NC, n)
    cat('Done.\n')
  }
  
  statistic <- stat.s1(x, y) # Statistic on [0,1)
  pv <- mean(sim_null > statistic) # p-value
  
  return(list(stat = statistic, pvalue = pv))
}
