#' Geodesic two-sample goodness-of-fit on the torus
#'
#' Performs a two-sample goodness-of-fit test for measures supported on the torus, by testing
#' the equality of their projected distributions on ng closed geodesics and combining the results into a global p-value.
#'
#' @param sample_1 n x 2 matrix containing n observations in the two-dimensional flat torus, parameterized as the periodic [0,1) x [0,1).
#' @param sample_2 n x 2 matrix containing n observations in the two-dimensional flat torus, parameterized as the periodic [0,1) x [0,1).
#' @param n_geodesics The number of closed geodesics where samples must be projected.
#' @param NC_geodesic The number of cores for computing parallelly the circular tests.
#' @param geodesic_list If NULL, geodesics are chosen randomly with the function rgeodesic.
#' Else, a list of vectors containing the director vectors of the chosen geodesics.
#' @param sim_null The simulated null distribution of the circle test statistic. It NULL, the null distribution is simulated with the given parameters (very time consuming).
#' @param NR The number of replicas if simulation is required.
#' @param NC The number of cores if parallel simulation for sim_null is required.
#' @param n The sample sizes of the simulated samples is simulation is required.
#' @param return_stat If TRUE, both the the test statistic and pvalue are returned, otherwise only the pvalue is returned.
#'
#' @return The p-value for the two-sample test on the torus.
#'
#' @examples
#'
#' n <- 100 # Sample size
#'
#' # Simulate the null distribution of the circle test statistic
#'
#' sim_free_null <- sim.null.stat(500, NC = 2)
#'
#' # Bivariate von Mises distributions
#' samp_1 <- BAMBI::rvmcos(n) / (2 * pi)
#' samp_2 <- BAMBI::rvmcos(n) / (2 * pi)
#'
#' #4 geodesics are chosen randomly
#' twosample.geodesic.torus.test(samp_1, samp_2, n_geodesics = 3,
#' NC_geodesic = 2, sim_null = sim_free_null)
#'
#' #4 geodesics are chosen a priori
#' glist <- list(c(1, 0), c(0, 1), c(1, 1), c(2, 3))
#' twosample.geodesic.torus.test(samp_1, samp_2, geodesic_list = glist,
#' NC_geodesic = 2, sim_null = sim_free_null)
#'
#'
#' @export

twosample.geodesic.torus.test<-function(sample_1, sample_2, n_geodesics = 1, NC_geodesic = 1, geodesic_list = NULL, sim_null = NULL, NR = 500, NC = 1, n = 30, return_stat=FALSE){

  one_geodesic_test <- function(u, data_1, data_2, sim_null_free){
    p <- geodesic.projection(u, data_1, data_2, do_plots = FALSE)
    test_u <- twosample.test.s1(p$circle_one, p$circle_two, sim_null_free,  NR, NC, n)
    return(test_u)
  }

  if(is.null(geodesic_list)){
    samp <- rgeodesic(n_geodesics)
  }
  else {
    samp <- matrix(unlist(geodesic_list), ncol=2, byrow = TRUE)
    n_geodesics <- nrow(samp)
  }

  cl <- parallel::makeCluster(NC_geodesic, type = "PSOCK")
  parallel::clusterExport(cl = cl, varlist = c('geodesic.projection', 'twosample.test.s1',
                                           'stat.s1', 'sim.null.stat'))
  parallel::clusterExport(cl = cl, varlist = c('sim_null'), envir = environment())
  results_list <- parallel::parApply(cl, FUN = one_geodesic_test, X = samp, MARGIN = 1,
  data_1 = sample_1, data_2 = sample_2, sim_null_free = sim_null)
  parallel::stopCluster(cl)

  # obtain all pvalues and statistic values
  results_binded <- Reduce("c", results_list)
  results <- tapply(results_binded, names(results_binded), function(x) unlist(x, FALSE, FALSE))
  pvals <- results$pvalue
  stats <- results$stat

  pval <- min(1, n_geodesics*(min(pvals)))
  stat <- mean(stats)
  if(return_stat){
      return(list(pval=pval, stat=stat))
  }
  else {
      return(pval)
  }
}
