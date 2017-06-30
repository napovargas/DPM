#include <RcppArmadillo.h>
#include <cmath>
#include <rtnorm.hpp>

// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

/* SImple function to tabulate integer vectors */
arma::uvec tabulate(arma::uvec y){
  arma::uvec uniq = unique(y);
  arma::uword k   = uniq.n_rows;
  arma::uvec x     = zeros<uvec>(k, 1);
  for(uword i = 0; i < k; i++){
    x(i) = sum(y.elem(find(y == (i + 1))))/(i + 1.0);
  }
  return(x);
}

/* Update fixed effects */
arma::vec UpdateBetaCpp(arma::vec y, arma::mat X, arma::mat XtX, arma::mat Z, arma::vec b, double sigma2e, 
                        arma::vec mu_beta, arma::mat Sigma_betainv){
  arma::uword p = X.n_cols;
  arma::vec   tmp;
  arma::vec   beta_hat;
  arma::vec   mu_beta_tilde;
  arma::mat   Sigma_beta_tilde;
  
  Sigma_beta_tilde = inv(XtX*(1/sigma2e) + Sigma_betainv);
  mu_beta_tilde    = Sigma_beta_tilde*(trans(X)*(y - Z*b)*(1/sigma2e) + Sigma_betainv*mu_beta);
  tmp              = rnorm(p);
  beta_hat         = chol(Sigma_beta_tilde)*tmp + mu_beta_tilde;
  return(beta_hat);
}

/* Update random effects */
arma::vec UpdateBCpp_R(arma::vec y, arma::mat X, arma::vec beta, arma::mat Z, arma::mat ZtZ, 
                       double sigma2e, arma::vec theta, arma::mat Sigma_binv){
  arma::uword q = Z.n_cols;
  arma::vec   tmp;
  arma::vec   mu_b_tilde;
  arma::mat   Sigma_b_tilde;
  arma::vec   b_hat;
  
  Sigma_b_tilde = inv(Sigma_binv + ZtZ*(1/sigma2e));
  mu_b_tilde    = Sigma_b_tilde*(trans(Z)*(y - X*beta)*(1/sigma2e) + Sigma_binv*theta);
  tmp           = rnorm(q);
  b_hat         = chol(Sigma_b_tilde)*tmp + mu_b_tilde;
  return(b_hat);
}

/* Update location parameter for each cluster */
arma::vec UpdatePhiCpp(arma::uword H, arma::uvec C, arma::vec b, double mu_0, double sigma2_0, double sigma2b){
  double mu0_star    = 0;
  double sigma0_star = 0;
  
  arma::vec   phi = zeros(H);
  arma::uvec  nh  = zeros<uvec>(H);
  
  for(uword h = 0; h < H; h++){
    nh(h) = sum(find(C == (h + 1)));
    if(nh(h) == 0){
      phi(h) = rnorm(1, mu_0, sqrt(sigma2_0))[0];
    } else {
      sigma0_star = 1/(nh(h)/sigma2b + 1/sigma2_0);
      mu0_star    = sigma0_star*(nh(h)/sigma2b*mean(b.elem(find(C == (h + 1)))) + mu_0/sigma2_0);
      phi(h)      = rnorm(1, mu0_star, sqrt(sigma0_star))[0];
    }
  }
  return(phi); 
}

/* Draw samples from a discrete deistribution */
arma::uword rdiscrete(arma::uvec value, arma::vec probs){
  if(accu(probs) != 1){
    probs = probs/accu(probs);
  }
  
  int l           = probs.n_rows;
  arma::vec p_upp = cumsum(probs);
  arma::vec p_low = join_vert(zeros(1,1), p_upp(span(0, l - 2)));
  arma::vec U     = ones(l)*runif(1)[0];
  arma::uvec C    = (U > p_low) && (U < p_upp);
  arma::uword X   = 0;
  for (int i = 0; i < l; i++){
    X = X + value(i)*(double)C(i);
  }
  return(X);  
}

/* Update Class (cluster) ID for each sire */
arma::uvec UpdateCCpp(arma::uword H, arma::vec b, arma::vec phi, double sigma2b, arma::vec Pi){
  arma::uword q     = b.n_rows;
  arma::uvec  C     = zeros<uvec>(q);
  arma::mat   D     = zeros(q, H);
  arma::uvec  value = linspace<uvec>(1, H, H);
  for(uword h = 0; h < H; h++){
    for(uword j = 0; j < q; j++){
      D(j, h) = R::dnorm(b(j), phi(h), sqrt(sigma2b), 0);
    }
  }
  for(uword j = 0; j < q; j++){
    C(j) = rdiscrete(value, trans(D.row(j)) % Pi);
  }
  return(C);
}

/* Update cluster probabilities */
arma::cube UpdatePiCpp(arma::uvec C, double alpha, uword H){
  arma::vec  Pi      = zeros(H);
  arma::vec  V       = zeros(H - 1);
  arma::uvec nh      = zeros<uvec>(H);
  arma::cube Out     = zeros(H, 1, 2);
  double     Product = 0;
  
  for(uword h = 0; h < H; h++){
    nh(h) = sum(find(C == (h + 1)));
  }
  for(uword h = 0; h < (H - 1); h++){
    V(h) = rbeta(1, 1 + nh(h), alpha + sum(nh(span(h + 1, H - 1))))[0];
    if(V(h) == 0){
      V(h) = 0.00001;
    } else if(V(h) == 1){
      V(h) = 0.9999;
    }
  }
  Pi(0)   = V(0);
  Product = 1.0;
  for(uword h = 1; h < (H - 1); h++){
    Product  = Product*(1 - V(h - 1));
    Pi(h)    = V(h)*Product; 
  }
  if((1 - sum(Pi(span(0, H - 2)))) >= 0.0) {
    Pi(H - 1) = 1 - sum(Pi(span(0, H - 2)));
  } else {
    Pi(H - 1) = 0;
  }
  Out.slice(0) = Pi;
  Out(span(0, H - 2), span(0, 0), span(1, 1)) = V;
  return(Out);
}

/* Update residual variance */
double UpdateSigma2eCpp(arma::vec y, arma::vec e, double ae, double be){
  arma::uword   n       = y.n_rows;
  double        atilde  = ae + 0.5*n;
  double        btilde  = be + 0.5*accu(arma::pow(e, 2));
  double        sigma2e = 1/rgamma(1, atilde, 1/btilde)[0];
  return(sigma2e);
}

/* Update additive variance */
double UpdateSigma2bCpp(arma::vec b, arma::vec theta, double ab, double bb){
  arma::uword q  = b.n_rows;
  double atilde  = ab + 0.5*q;
  double btilde  = bb + 0.5*accu(arma::pow(b - theta, 2));
  double sigma2b = 1/rgamma(1, atilde, 1/btilde)[0];
  return(sigma2b);
}

/* random samples from Dirichlet distribution with concentration parameter alpha */
arma::vec Dirichlet(arma::vec alpha){
  arma::uword k = alpha.n_rows;
  arma::vec y = zeros(k);
  arma::vec x = zeros(k);
  for(uword j = 0; j < k; j++){
    y(j) = rgamma(1, alpha(j), 1)[0];
  }
  for(uword i = 0; i < k; i++){
    x(i) = y(i)/arma::accu(y);
  }
  return(x);
}

/* Nandran and Chen's (1996) algorithm to sample tau */
// [[Rcpp::export]]
arma::mat UpdateTauNCCpp(arma::uvec y, arma::vec mu, arma::vec prob, arma::vec tau){
  double      A           = 0;
  double      accept      = 0;
  arma::uword n           = y.n_rows;
  arma::uvec  Category    = sort(unique(y));
  arma::uword K           = Category.n_rows;
  arma::uword Kmax        = arma::max(Category);
  arma::uword Kmin        = arma::min(Category);
  arma::vec   nk          = zeros(K);
  arma::vec   ftau_new    = zeros(n);
  arma::vec   ftau        = zeros(n);
  arma::vec   alpha       = zeros(K - 2);
  arma::vec   prob_new    = zeros(K - 2);
  arma::vec   tau_new     = zeros(K - 1);
  arma::vec   loglp;
  arma::vec   loglp_new;
  //arma::vec   mu;
  arma::vec   logftau;
  arma::vec   logftau_new;
  
  arma::mat   Out(K - 1, 3);
  for(uword k = 0; k < K; k++){
    nk(k) = accu(y.elem(find(y == Category(k))))/(1.0*Category(k));
  }
  alpha                   = 0.80*nk(span(1, K - 2));
  prob_new                = Dirichlet(alpha);
  tau_new(0)              = 0.00;
  tau_new(K - 2)          = 1.00;
  if(K > 3){
    tau_new(span(1, K - 2)) = cumsum(prob_new);
  }
  loglp                   = alpha%log(prob);
  loglp_new               = alpha%log(prob_new);
  //mu                      = X*beta + Z*b;
  
  for(uword i = 0; i < n; i++){
    if(y(i) == Kmax){
      ftau(i)     = 1 - R::pnorm(tau(K - 2) - mu(i), 0, 1, 1, 0);
      ftau_new(i) = 1 - R::pnorm(tau_new(K - 2) - mu(i), 0, 1, 1, 0);
    } else if(y(i) == Kmin){
      ftau(i) = R::pnorm(tau(0) - mu(i), 0, 1, 1, 0);
      ftau(i) = R::pnorm(tau_new(0) - mu(i), 0, 1, 1, 0);
    } else {
      for(uword k = 1; k < (K - 2); k++){
        if(y(i) == Category(k)){
          ftau(i)     = R::pnorm((tau(k) - mu(i)), 0, 1, 1, 0) - R::pnorm((tau(k - 1) - mu(i)), 0, 1, 1, 0);
          ftau_new(i) = R::pnorm((tau_new(k) - mu(i)), 0, 1, 1, 0) - R::pnorm((tau_new(k - 1) - mu(i)), 0, 1, 1, 0);
        }
      }
    }
  }
  logftau     = arma::log(ftau);
  logftau_new = arma::log(ftau_new);
  logftau.replace(datum::nan, log(1e-15));
  logftau_new.replace(datum::nan, log(1e-20));
  A           = exp(accu(logftau_new - logftau) - accu(loglp_new - loglp));
  if(runif(1)[0] < std::min(1.0, A)){
    prob   = prob_new;
    tau    = tau_new;
    accept = 1;
  }
  /*Out["tau"]              = tau;
  Out["prob"]             = prob;
  Out["accept"] 		      = accept;*/
  Out.col(0)              = tau;
  Out(span(0, K - 3), 1)  = prob;
  Out(0, 2)               = accept;
  return(Out);
}

/* Draw samples from truncated normal */
arma::vec TruncatedNormal(arma::vec mu, double sigma, arma::uvec y, arma::vec cutoffs){
  arma::uword n         = y.n_rows;
  arma::vec   a(n);
  arma::vec   b(n);
  arma::vec   FA(n);
  arma::vec   FB(n);
  arma::vec   out(n);
  
  for (uword i= 0; i < n; i++){
    a(i) = cutoffs(y(i) - 1);
    b(i) = cutoffs(y(i));
  }
  for (uword i = 0; i < n; i++) {
    FA(i)  = R::pnorm((a(i) - mu(i))/sigma,0,1,1,0);
    FB(i)  = R::pnorm((b(i) - mu(i))/sigma,0,1,1,0);
    out(i) = mu[i] + sigma*R::qnorm(runif(1)[0]*(FB(i) - FA(i)) + FA(i),0,1,1,0);
  }
  return(out);
}

// [[Rcpp::export]]
List DPMMOrdinal(arma::uvec z, arma::mat X, arma::mat Z, arma::uword N, arma::vec mbeta, arma::mat Vbeta, 
                 double mu0, double sigma2_0, double alpha, double ae, double be, double ce, double aa, double ba, double ca, 
                 arma::uword nIter){
  
  /* Integer variables and vectors */
  arma::uword n             = z.n_rows;
  arma::uword p             = X.n_cols;
  arma::uword q             = Z.n_cols;
  arma::uvec  Category      = sort(unique(z));
  arma::uword K             = Category.n_rows; 
  arma::uvec  C             = ones<uvec>(q);
  arma::uvec  ActiveC;
  arma::uvec  nk;
  arma::uvec  idx           = linspace<uvec>(1, K - 1, K - 1);
  
  /* Real variables */
  double  sigma2e           = 1;
  double  sigma2a           = 1;
  double  lambdae           = 1;
  double  lambdaa           = 1;
  double  accept            = 0;
  double  pct               = 0;
  double  ttime             = 0;
  clock_t start             = 0;
  clock_t end               = 0;
  
  /* Real vectors and matrices */
  arma::vec   beta          = zeros(p);
  arma::vec   tau           = zeros(K - 1);
  arma::vec   prob          = zeros(K - 3);
  arma::vec   theta         = zeros(q);
  arma::vec   b             = zeros(q);
  arma::vec   Pi            = zeros(N);
  arma::vec   phi           = zeros(N);
  arma::vec   mu            = zeros(n);
  arma::vec   V             = zeros(N);
  arma::vec   liability     = zeros(n);
  arma::vec   residual      = zeros(n);
  arma::vec   conc          = zeros(K - 2);
  arma::vec   cuttoffs      = zeros(K + 1);
  arma::mat   Vbetainv      = inv(Vbeta);
  arma::cube  Pi_tmp;
  arma::mat   tau_tmp;
  arma::mat   XtX           = trans(X)*X;
  arma::mat   ZtZ           = trans(Z)*Z;
  arma::mat   Sbinv         = eye(q, q);
  
  /* Vectors and matrices for storing parameter draws */
  arma::mat   store_beta    = zeros(nIter, p);
  arma::mat   store_tau     = zeros(nIter, K - 1);
  arma::mat   store_b       = zeros(nIter, q);
  arma::mat   store_theta   = zeros(nIter, q);
  arma::mat   store_VC      = zeros(nIter, 4);
  arma::umat  store_N       = zeros<umat>(nIter, 1);
  arma::umat  store_C       = zeros<umat>(nIter, q);
  arma::mat   store_phi     = zeros(nIter, N);
  
  /* List for final output */
  List Out;
  
  /* Initial values for remaining parameters */
  liability                 = runif(n);
  Pi.fill(1/(N*1.0));
  for(uword i = 0; i < q; i++){
    C(i)                    = rdiscrete(linspace<uvec>(1, N, N), Pi);
  }
  tau(0)                    = 0.0;
  tau(span(1, K - 3))       = arma::sort(vec(runif(K - 3)));
  tau(K - 2)                = 1.0;
  nk                        = tabulate(z);
  for(uword j = 1; j < (K - 1); j++){
    conc(j - 1)                      = 0.80*nk(j);
  }
  prob                      = Dirichlet(conc);
  cuttoffs(0)               = -100.0;
  cuttoffs(K)               = 100.0;
  
  /* Main MCMC loop */
  /* start clock */
  start = clock();
  for(uword iter = 0; iter < nIter; iter++){
    beta                = UpdateBetaCpp(liability, X, XtX, Z, b, sigma2e, mbeta, Vbeta);
    b                   = UpdateBCpp_R(liability, X, beta, Z, ZtZ, sigma2e, theta, eye(q, q)*1/sigma2a);
    mu                  = X*beta + Z*b;
    phi                 = UpdatePhiCpp(N, C, b, mu0, sigma2_0, sigma2a);
    C                   = UpdateCCpp(N, b, phi, sigma2a, Pi);
    for(uword i = 0; i < q; i++){
      theta(i)  = phi(C(i) - 1);
    }
    Pi_tmp              = UpdatePiCpp(C, alpha, N);
    Pi                  = Pi_tmp.slice(0);
    residual            = liability - mu;
    sigma2e             = UpdateSigma2eCpp(liability, residual, ae, lambdae);
    sigma2a             = UpdateSigma2bCpp(b, theta, aa, lambdaa);
    tau_tmp             = UpdateTauNCCpp(z, mu, prob, tau);
    tau                 = tau_tmp.col(0);
    prob                = tau_tmp(span(0, K - 3), 1);
    accept              = accept + tau_tmp(0, 2);
    cuttoffs.elem(idx)  = tau;
    liability           = TruncatedNormal(mu, std::sqrt(sigma2e), z, cuttoffs);
    liability.replace(datum::nan, 1e-10);
    ActiveC             = unique(C);
    lambdae             = rgamma(1, ae + be, 1/(ce + 1/sigma2e))[0];
    lambdaa             = rgamma(1, aa + ba, 1/(ca + 1/sigma2a))[0];
    
    /* Storing results */
    store_beta.row(iter)  = trans(beta);
    store_b.row(iter)     = trans(b);
    store_VC(iter, 0)     = sigma2e;
    store_VC(iter, 1)     = sigma2a;
    store_VC(iter, 2)     = lambdae;
    store_VC(iter, 3)     = lambdaa;
    store_theta.row(iter) = trans(theta);
    store_C.row(iter)     = trans(C);
    store_N.row(iter)     = ActiveC.n_rows;
    store_phi.row(iter)   = trans(phi);
    store_tau.row(iter)   = trans(tau);
    /* Printing number of iterations to console */
    if(iter % 1000 == 0){
      Rcpp::Rcout.precision(2);
      Rcpp::checkUserInterrupt();
      pct = (double)iter/(double)nIter;
      Rcpp::Rcout << " Iteration: " << iter << "/" << nIter << " [" << std::fixed << (pct*100.00) << " %] "<< std::endl;
    }
  }
  /* timing */
  end   = clock();
  ttime = ((double) (end - start)) / CLOCKS_PER_SEC;
  
  /* Preparing to exit program                           */
  /* Printing final message to console with time elapsed */
  Rcpp::Rcout << "                                               " << std::endl;
  Rcpp::Rcout << " Done! " << std::endl;
  Rcpp::Rcout << "                                               " << std::endl;
  Rcpp::Rcout << nIter << " iterations in " << ttime << " seconds" << std::endl;
  
  /* Remove burnin and filling list */ 
  Out["beta"]     = store_beta;
  Out["b"]        = store_b;
  Out["VC"]       = store_VC;
  Out["tau"]      = store_tau;
  Out["accept"]   = accept/(nIter*1.0);
  Out["theta"]    = store_theta; 
  Out["phi"]      = store_phi; 
  Out["cluster"]  = store_C; 
  Out["N"]        = store_N; 
  Out["time"]     = ttime;
  return(wrap(Out));
}