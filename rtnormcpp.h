#include <cmath>
#include <rtnorm.hpp>
#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

int N = 4001;   // Index of the right tail
double yl(int k)
{
  double yl0 = 0.053513975472;                  // y_l of the leftmost rectangle
  double ylN = 0.000914116389555;               // y_l of the rightmost rectangle
  
  if (k == 0)
    return yl0;
    
  else if(k == N-1)
    return ylN;
          
  else if(k <= 1953)
    return Rtnorm::yu[k-1];
          
  else
    return Rtnorm::yu[k+1];
}

double rtexp(double a, double b)
{
  int stop = false;
  double twoasq = 2*pow(a,2);
  double expab = exp(-a*(b-a)) - 1;
  double z, e;

  while(!stop)
  {
    z = log(1 + runif(1)[0]*expab);
    e = -log(runif(1)[0]);
    stop = (twoasq*e > pow(z,2));
  }
  return a - z/a;
}

// [[Rcpp::export]]
double rtnorm(double a, double b, const double mu, const double sigma){
  // Design variables
  double xmin = -2.00443204036;                 // Left bound
  double xmax =  3.48672170399;                 // Right bound
  int kmin = 5;                                 // if kb-ka < kmin then use a rejection algorithm
  double INVH = 1631.73284006;                  // = 1/h, h being the minimal interval range
  int I0 = 3271;                                // = - floor(x(0)/h)
  double ALPHA = 1.837877066409345;             // = log(2*pi)
  int xsize=sizeof(Rtnorm::x)/sizeof(double);   // Length of table x
  int stop = false;
  //double sq2 = 7.071067811865475e-1;            // = 1/sqrt(2)
  //double sqpi = 1.772453850905516;              // = sqrt(pi)

  double z, e, ylk, simy, lbound, u, d, sim;
  double r = 0;
  int i, ka, kb, k;
  
  // Scaling
  if(mu!=0 || sigma!=1)
  {
    a=(a-mu)/sigma;
    b=(b-mu)/sigma;
  }
  
  //-----------------------------
  
  // Check if a < b
  if(a>=b)
  {
    //std::cerr<<"*** B must be greater than A ! ***"<<std::endl;
    //exit(1);
    b = mu;//std::abs(b);
  }

  // Check if |a| < |b|
  else if(fabs(a)>fabs(b))
    r = -rtnorm(-b,-a);  // Pair (r,p)

  // If a in the right tail (a > xmax), use rejection algorithm with a truncated exponential proposal  
  else if(a>xmax)
    r = rtexp(a,b);

  // If a in the left tail (a < xmin), use rejection algorithm with a Gaussian proposal
  else if(a<xmin)
  {
    while(!stop)
    {
      r = rnorm(1)[0];
      stop = (r>=a) && (r<=b);
    }
  }

  // In other cases (xmin < a < xmax), use Chopin's algorithm
  else
  {
    // Compute ka
    i = I0 + floor(a*INVH);
    ka = Rtnorm::ncell[i];

    // Compute kb
    (b>=xmax) ?
    kb = N :
    (
      i = I0 + floor(b*INVH),
      kb = Rtnorm::ncell[i]
    );

    // If |b-a| is small, use rejection algorithm with a truncated exponential proposal
    if(abs(kb-ka) < kmin)
    {
      r = rtexp(a,b);
      stop = true;  
    }
    
    while(!stop)
    {
      // Sample integer between ka and kb
      k = floor(runif(1)[0]*(kb-ka+1)) + ka;
    
      if(k == N)
      {    
        // Right tail
        lbound = Rtnorm::x[xsize-1];
        z = -log(runif(1)[0]);
        e = -log(runif(1)[0]);
        z = z / lbound;

        if ((pow(z,2) <= 2*e) && (z < b-lbound))
        {
          // Accept this proposition, otherwise reject
          r = lbound + z;
          stop = true;
        }
      }

      else if ((k <= ka+1) || (k>=kb-1 && b<xmax))
      {
          
        // Two leftmost and rightmost regions
        sim = Rtnorm::x[k] + (Rtnorm::x[k+1]-Rtnorm::x[k]) * runif(1)[0];

        if ((sim >= a) && (sim <= b))
        {
          // Accept this proposition, otherwise reject
          simy = Rtnorm::yu[k]*runif(1)[0];
          if ( (simy<yl(k)) || (sim * sim + 2*log(simy) + ALPHA) < 0 )
          {
            r = sim;
            stop = true;
          }
        }        
      }

      else // All the other boxes
      {    
        u = runif(1)[0];
        simy = Rtnorm::yu[k] * u;
        d = Rtnorm::x[k+1] - Rtnorm::x[k];
        ylk = yl(k);
        if(simy < ylk)  // That's what happens most of the time 
        {  
          r = Rtnorm::x[k] + u*d*Rtnorm::yu[k]/ylk;  
          stop = true;
        }
        else
        {
          sim = Rtnorm::x[k] + d * runif(1)[0];

          // Otherwise, check you're below the pdf curve
          if((sim * sim + 2*log(simy) + ALPHA) < 0)
          {
            r = sim;
            stop = true;
          }
        }
        
      }
    }  
  }
  
  //-----------------------------

  // Scaling
  if(mu!=0 || sigma!=1)
    r = r*sigma + mu;
  
  /* Compute the probability
  Z = sqpi *sq2 * sigma * ( erf(b*sq2) - erf(a*sq2) );
  p = exp(-pow((r-mu)/sigma,2)/2) / Z; */

  return r;  
}
