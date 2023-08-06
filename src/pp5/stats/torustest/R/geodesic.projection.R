#' Geodesic projection of a pair of samples on the torus
#'
#' Given a vector of integers u=(a,b), two samples on the torus are projected to the
#' closed geodesic given by the canonical projection of the straight line with director
#' vector u and containing the origin.
#'
#' @param u A vector of two integers (a,b), with b non-zero.
#' @param data_1 n x 2 matrix containing n observations in the two-dimensional flat torus, parameterized as the periodic [0,1) x [0,1).
#' @param data_2 m x 2 matrix containing m observations in the two-dimensional flat torus, parameterized as the periodic [0,1) x [0,1).
#' @param do_plots Whether to produce plots illustrating the steps of the algorithm for the given geodesic and samples.
#' @param size_points If do_plots is TRUE, the size of the points to plot (to choose according to n,m).
#'
#' @return  \itemize{
#'   \item plot_list - If do_plots is TRUE, a list with the corresponding plots.
#'   \item circle_one - The projected sample data_1 parameterized on the circle (periodic [0,1)).
#'   \item circle_two - The projected sample data_2 parameterized on the circle (periodic [0,1)).
#'   \item proj_data - A data frame containing the coordinates on [0,1]x[0,1] of both samples and the corresponding projections.
#'
#' }
#'
#' @examples
#'
#' data_1 <- BAMBI::rvmcos(30) / (2 * pi)
#' data_2 <- BAMBI::rvmcos(30) / (2 * pi)
#' pr <- geodesic.projection(u = c(-1, -2), data_1, data_2, do_plots = TRUE)
#' pr$plot_list[[1]]
#' pr$plot_list[[2]]
#' pr$plot_list[[3]]
#' pr$plot_list[[4]]
#' pr$plot_list[[5]]
#' pr$circle_one; pr$circle_two
#'
#' @export

geodesic.projection<-function(u, data_1, data_2, do_plots = FALSE, size_points = 1){

  vec.all.equal <- function(a, b){ # Internal usage. Corrects numerical errors due to rounding.
    logic <- as.logical(apply(X = as.matrix(a), FUN = all.equal, MARGIN = 1, current = b))
    logic[is.na(logic)] <- FALSE
    return(logic)
  }

  if(sum(floor(u) == u & ceiling(u) == u) != 2){stop('u must be given as a vector of two integers')}
  if(u[1] == 0 & u[2] == 0){stop('u must have non-zero module')}
  if(u[1] == 0){return(list(circle_one = as.numeric(data_1[,2]), circle_two = as.numeric(data_2[,2])))} #phi (x) marginal
  if(u[2] == 0){return(list(circle_one = as.numeric(data_1[,1]), circle_two = as.numeric(data_2[,1])))} #psi (y) marginal

  x_split <- FALSE

  data_1 <- as.data.frame(data_1)
  colnames(data_1) <- c('phi','psi')
  data_1$sample <- 1 #Sample label

  data_2 <- as.data.frame(data_2)
  colnames(data_2) <- c('phi','psi')
  data_2$sample <- 2 #Sample label

  # Octant corrections for computational convenience

  #(-u1,-u2) equals (u1,u2)
  if(u[1] < 0 & u[2] < 0){u[1] <- -u[1]; u[2] <- -u[2]}
  #(-u1,u2) equals (u1,-u2)
  if(u[1] < 0 & u[2] > 0){u[1]<- -u[1]; u[2]<- -u[2]}
  if(u[1] > 0 & u[2] < 0){x_split <- TRUE; u[2] <- -u[2]}

  # Define the set P_u and transfer it to [0,1]x[0,1]

  points <- matrix(c(0, 0), ncol=2, nrow=1)
  colnames(points) <- c('x','y')

  points <- as.data.frame(rbind(points, rbind(cbind(u[1] / u[2] * (1:u[2]), 1:u[2]), cbind(1:u[1], u[2] / u[1] * (1:u[1])))))
  points <- points - floor(points)
  points <- points[order(points$x),]
  if(sum(duplicated(round(points, 20)))>0){points <- points[!duplicated(round(points, 20)),]} #Correct numerical errors due to rounding

  #Define the straight lines of L_u

  diag_list <- c()
  intercepts <- c()
  if(x_split){

    for(i in 1:nrow(points)){

      int_i <- points[i,2] + u[2] / u[1] * (1 - points[i,1])
      intercepts <- c(intercepts, int_i)
      diag_list <- c(diag_list, ggplot2::geom_abline(intercept = int_i, slope = -u[2] / u[1], size = 0.2))
    }
  }else{

    for(i in 1:nrow(points)){

      int_i <- points[i,2] - u[2] / u[1] * points[i,1]
      intercepts <- c(intercepts, int_i)
      diag_list <- c(diag_list, ggplot2::geom_abline(intercept = int_i, slope = u[2] / u[1], size = 0.2))
    }
  }

  int_max <- ifelse(length(intercepts) == 1,
                  ifelse(x_split, 2, 1),
                  max(intercepts) + max(diff(sort(intercepts))))

  int_min <- ifelse(length(intercepts) == 1,
                  ifelse(x_split, 0, -1),
                  min(intercepts) - max(diff(sort(intercepts))))

  intercepts <- c(intercepts, int_max, int_min)
  intercepts <- round(intercepts, 22) #Avoid numerical errors due to rounding

  diag_list <- c(diag_list, ggplot2::geom_abline(intercept = int_max, slope = ifelse(x_split, -u[2] / u[1], u[2] / u[1]), size = 0.2))
  diag_list <- c(diag_list, ggplot2::geom_abline(intercept = int_min, slope = ifelse(x_split, -u[2] / u[1], u[2] / u[1]), size = 0.2))

  both_data <- rbind(data_1, data_2)
  both_data$sample <- as.factor(both_data$sample)

  if(do_plots){ # Produce plots illustrating the algorithm for the given samples and geodesic

    plot_list <- list()

    plot_list[[1]] <- ggplot2::ggplot(both_data, ggplot2::aes(x = phi, y = psi, col = sample))+
      ggplot2::geom_point(size = size_points)+
      diag_list+
      ggplot2::scale_x_continuous(expand = c(0, 0))+
      ggplot2::scale_y_continuous(expand = c(0, 0))+
      ggplot2::theme(legend.position='none')+
      ggplot2::ggtitle(paste0('Geodesic projection to (',u[1],',',ifelse(x_split, -u[2], u[2]),')'))+
      ggplot2::labs(x = expression(phi), y = expression(psi))

    plot_list[[2]] <- ggplot2::ggplot(both_data, ggplot2::aes(x = phi, y = psi, col = sample))+
      ggplot2::geom_point(size = size_points)+
      diag_list+
      ggplot2::scale_x_continuous(expand = c(-1, 2))+
      ggplot2::scale_y_continuous(expand = c(-1, 2))+
      ggplot2::theme(legend.position='none')+
      ggplot2::ggtitle(paste0('Geodesic projection to (',u[1],',',ifelse(x_split, -u[2], u[2]),')'))+
      ggplot2::labs(x = expression(phi), y = expression(psi))

  }

  d_point_line <- function(point, intercept, slope){ # Distance point - straight line
      return(abs(slope * point[1] + intercept - point[2]) / sqrt(1 + slope^2))
  }

  d_twodiag <- max(diff(sort(intercepts)))/sqrt(1 + (u[2] / u[1])^2) # Distance between two lines of L_u

  intercepts <- sort(intercepts)
  mat_dis <- matrix(NA, ncol = length(intercepts), nrow = nrow(both_data)) # Distances to each line of L_u
  for(k in 1:length(intercepts)){
    mat_dis[, k] <- apply(FUN = d_point_line, X = both_data[, 1:2], MARGIN = 1, intercept = intercepts[k], slope = ifelse(x_split, -u[2]/u[1], u[2]/u[1]))
  }
  both_data$diag_proj <- apply(X = mat_dis, FUN = which.min, MARGIN=1) # Closest line of L_u to each point

  if(x_split){u_norm <- c(u[2], u[1]) / sqrt(sum(u^2))}else{
    u_norm <- c(-u[2], u[1]) / sqrt(sum(u^2))}

  dot_u <- function(v, u, split = x_split){
    if(split){u[2] <- -u[2]}
    return(v%*%u)}

  both_data$diag_proj <- both_data$diag_proj - which(vec.all.equal(intercepts, 0))

  # Project points to the closest line of L_u

  k <- apply(X = both_data[, 1:2], FUN = dot_u, MARGIN = 1, u = u) / sum(u^2)
  both_data$proj_phi <- k * u[1] + both_data$diag_proj * d_twodiag * u_norm[1]
  both_data$proj_psi <- k * ifelse(x_split, -u[2], u[2]) + both_data$diag_proj * d_twodiag * u_norm[2]

  if(do_plots){ # Produce plots illustrating the algorithm for the given samples and geodesic

    plot_list[[3]]  <- ggplot2::ggplot(both_data, ggplot2::aes(x = phi, y = psi, col = sample))+
      ggplot2::geom_point(size = size_points)+
      diag_list+
      ggplot2::scale_x_continuous(expand = c(-1,2))+
      ggplot2::scale_y_continuous(expand = c(-1,2))+
      ggplot2::theme(legend.position='none')+
      ggplot2::geom_point(ggplot2::aes(x = proj_phi, y = proj_psi, col = sample), size = 0.5)+
      ggplot2::ggtitle(paste0('Geodesic projection to (', u[1],',',ifelse(x_split, -u[2], u[2]),')'))+
      ggplot2::labs(x=expression(phi),y=expression(psi))

  }

  # Transfer projections outside [0,1]x[0,1] to [0,1] x[0,1]

  if(length(which(both_data$proj_phi > 1)) > 0){both_data$proj_phi[which(both_data$proj_phi > 1)] <- both_data$proj_phi[which(both_data$proj_phi > 1)] - 1}
  if(length(which(both_data$proj_phi < 0)) > 0){both_data$proj_phi[which(both_data$proj_phi < 0)] <- both_data$proj_phi[which(both_data$proj_phi < 0)] + 1}
  if(length(which(both_data$proj_psi > 1)) > 0){both_data$proj_psi[which(both_data$proj_psi > 1)] <- both_data$proj_psi[which(both_data$proj_psi > 1)] - 1}
  if(length(which(both_data$proj_psi < 0)) > 0){both_data$proj_psi[which(both_data$proj_psi < 0)] <- both_data$proj_psi[which(both_data$proj_psi < 0)] + 1}

  if(do_plots){ # Produce plots illustrating the algorithm for the given samples and geodesic

    plot_list[[4]] <- ggplot2::ggplot(both_data, ggplot2::aes(x = phi, y = psi, col = sample))+
      ggplot2::geom_point(size = size_points)+
      diag_list+
      ggplot2::scale_x_continuous(expand = c(-1, 2))+
      ggplot2::scale_y_continuous(expand = c(-1, 2))+
      ggplot2::theme(legend.position = 'none')+
      ggplot2::geom_point(ggplot2::aes(x = proj_phi, y = proj_psi, col = sample), size = 0.5)+
      ggplot2::ggtitle(paste0('Geodesic projection to (', u[1],',', ifelse(x_split, -u[2], u[2]),')'))+
      ggplot2::labs(x = expression(phi),y = expression(psi))

    plot_list[[5]] <- ggplot2::ggplot(both_data, ggplot2::aes(x = phi, y = psi, col = sample))+
      ggplot2::geom_point(size = size_points)+
      diag_list+
      ggplot2::scale_x_continuous(expand = c(0, 0))+
      ggplot2::scale_y_continuous(expand = c(0, 0))+
      ggplot2::theme(legend.position = 'none')+
      ggplot2::geom_point(ggplot2::aes(x = proj_phi, y = proj_psi, col = sample), size = 0.8, alpha = 0.5)+
      ggplot2::ggtitle(paste0('Geodesic projection to (', u[1], ',', ifelse(x_split, -u[2], u[2]),')'))+
      ggplot2::labs(x = expression(phi), y = expression(psi))

  }

  if(x_split){points$x <- 1 - points$x}
  points$x_end <- (1 - points$y + ifelse(x_split, -u[2] / u[1], u[2] / u[1]) * points$x) / ifelse(x_split, -u[2] / u[1], u[2] / u[1])
  points$y_end <- 1

  if(length(which(points$x_end > 1)) > 0){
    points$x_end[which(points$x_end > 1)] <- 1
    points$y_end[which(points$x_end == 1)] <- points$y[which(points$x_end == 1)] + ifelse(x_split, -u[2] / u[1], u[2] / u[1]) * (1 - points$x[which(points$x_end == 1)])}

  if(length(which(points$x_end<0)) > 0){
    points$x_end[which(points$x_end < 0)] <- 0
    points$y_end[which(points$x_end == 0)] <- points$y[which(points$x_end == 0)] - ifelse(x_split, -u[2] / u[1], u[2] / u[1]) * points$x[which(points$x_end == 0)]}

  if(!x_split){
    points$x_end_new <- ifelse(0 <= points$x_end - 1 & points$x_end - 1 <= 1, points$x_end - 1, points$x_end)}else{
      points$x_end_new <- ifelse(0 <= points$x_end + 1 & points$x_end + 1 <= 1, points$x_end + 1, points$x_end)
    }
  points$y_end_new <- ifelse(0 <= points$y_end - 1 & points$y_end - 1 <= 1, points$y_end - 1, points$y_end)

  if(!x_split){
    one_one <- which(vec.all.equal(points$x_end, 1) & vec.all.equal(points$y_end, 1))
    if(length(one_one) > 0){points[one_one, c('x_end_new', 'y_end_new')] <- c(0,0)}}else{
      one_one <- which(vec.all.equal(points$x_end, 0) & vec.all.equal(points$y_end, 1))
      if(length(one_one) > 0){points[one_one, c('x_end_new', 'y_end_new')] <- c(1,0)}
    }

  points$len_diag <- sqrt((points$x_end - points$x)^2 + (points$y_end - points$y)^2)
  if(sum(duplicated(round(points[, c('x', 'y')], 10))) > 0){points <- points[!duplicated(round(points[, c('x', 'y')], 10)), ]}

  ordered_points <- matrix(NA, ncol = 3, nrow = nrow(points))
  ordered_points[1, ] <- as.numeric(points[1, c('x', 'y', 'len_diag')])

  if(nrow(ordered_points) > 1){
    for(i in 2:nrow(ordered_points)){
      ordered_points[i, 1:2] <- as.numeric(points[which(vec.all.equal(points$x, ordered_points[i - 1, 1]) & vec.all.equal(points$y, ordered_points[i - 1, 2])), c('x_end_new', 'y_end_new')])
      ordered_points[i, 3] <- as.numeric(points$len_diag[which(vec.all.equal(points$x, ordered_points[i, 1]) & vec.all.equal(points$y, ordered_points[i, 2]))])
    }}

  if(sum(duplicated(ordered_points)) > 0){ordered_points <- ordered_points[!duplicated(ordered_points), ]}

  ordered_points <- as.data.frame(ordered_points)
  colnames(ordered_points) <- c('x0', 'y0', 'len')
  ordered_points$intercept <- ordered_points$y0 - ifelse(x_split, -u[2] / u[1], u[2] / u[1]) * ordered_points$x0
  int_order <- ordered_points$intercept

  both_data$intercept <- intercepts[which(vec.all.equal(intercepts, 0)) + both_data$diag_proj]
  # Correct outsiders' intercepts
  both_data$intercept <- round(both_data$proj_psi - ifelse(x_split, -u[2] / u[1], u[2] / u[1]) * both_data$proj_phi, 20)
  sample_one <- both_data[which(both_data$sample == 1), ]
  sample_two <- both_data[which(both_data$sample == 2), ]

  sample_one_circle <- c(); sample_two_circle <- c() # Parameterize samples on the circle
  for(i in 1:length(int_order)){

    ni1 <- which(vec.all.equal(sample_one$intercept, int_order[i]))
    if(length(ni1) != 0){
      proj_i <- sample_one[ni1, c('proj_phi', 'proj_psi')]
      origin_i <- matrix(rep(as.numeric(ordered_points[which(vec.all.equal(ordered_points$intercept, int_order[1])), c('x0', 'y0')]), nrow(proj_i)), nrow = nrow(proj_i), ncol = 2, byrow = TRUE)
      dis_i1 <- sqrt((proj_i - origin_i)[, 1]^2 + (proj_i - origin_i)[, 2]^2) +
        ifelse(i > 1, sum(ordered_points$len[1:(which(vec.all.equal(ordered_points$intercept, int_order[i])) - 1)]), 0)
      sample_one_circle <- c(sample_one_circle, dis_i1)}

    ni2 <- which(vec.all.equal(sample_two$intercept, int_order[i]))
    if(length(ni2) != 0){
      proj_i <- sample_two[ni2, c('proj_phi','proj_psi')]
      origin_i <- matrix(rep(as.numeric(ordered_points[which(vec.all.equal(ordered_points$intercept, int_order[1])), c('x0', 'y0')]), nrow(proj_i)), nrow = nrow(proj_i), ncol = 2, byrow = TRUE)
      dis_i2 <- sqrt((proj_i - origin_i)[, 1]^2 + (proj_i - origin_i)[, 2]^2) + ifelse(i > 1, sum(ordered_points$len[1:(which(vec.all.equal(ordered_points$intercept, int_order[i])) - 1)]), 0)
      sample_two_circle <- c(sample_two_circle, dis_i2)}
  }

  L <- sum(ordered_points$len)
  sample_one_circle <- sample_one_circle / L
  sample_two_circle <- sample_two_circle / L

  if(do_plots){
    return(list(plot_list = plot_list, circle_one = sample_one_circle, circle_two = sample_two_circle,
                proj_data = both_data[, c('phi', 'psi', 'sample', 'proj_phi', 'proj_psi')]))}else{
      return(list(circle_one = sample_one_circle, circle_two = sample_two_circle,
                  proj_data = both_data[, c('phi', 'psi', 'sample', 'proj_phi', 'proj_psi')]))
    }
}

