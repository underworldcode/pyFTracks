#include <math.h>
#include "ketcham1999.h"
#include <stdio.h>

#define	ETCH_PIT_LENGTH	0
#define CL_PFU 1
#define OH_PFU 2
#define	CL_WT_PCT 3

#define MIN_OBS_RCMOD  0.55

void ketch99_reduced_lengths(double *time, double *temperature, int numTTNodes,
                             double *redLength,  double kinPar, int kinParType,
                             int *firstTTNode, int etchant)
{
  int     node, nodeB;
  double  equivTime;
  double  timeInt,x1,x2,x3;
  double  totAnnealLen;
  double  equivTotAnnLen;
  double  rmr0,k;
  double  calc;
  double  tempCalc;

  typedef struct {
      double c0, c1, c2, c3, a, b;
  } annealModel;

  /* Fanning Curvilinear Model lcMod FC, See Ketcham 1999, Table 5e */
  annealModel modKetch99 = {-19.844, 0.38951, -51.253, -7.6423, -0.12327, -11.988};

  rmr0 = 0;

  /* Calculate the rmr0-k values for the kinetic parameter given */
  switch (kinParType) {

    /* See Ketcham et al 1999, Figure 7b */
    case ETCH_PIT_LENGTH:
      if (kinPar <= 1.75) rmr0 = 0.84;
      else if (kinPar >= 4.58) rmr0 = 0.0;
      else rmr0 = 1.0 - exp(0.647 * (kinPar - 1.75) - 1.834);
      break;

    case CL_WT_PCT:
      /* Just convert the kinetic parameter to Cl apfu
         Note that this invalidates kinPar for the rest of the routine */
      kinPar = kinPar * 0.2978;
      /* See Ketcham, et al, 1999, Figure 7a */
      calc = fabs(kinPar - 1.0);
      if (calc <= 0.130) rmr0 = 0.0;
      else rmr0 = 1.0 - exp(2.107 * (1.0 - calc) - 1.834);
      break;
    
    case CL_PFU:
      /* See Ketcham, et al, 1999, Figure 7a */
      calc = fabs(kinPar - 1.0);
      if (calc <= 0.130) rmr0 = 0.0;
      else rmr0 = 1.0 - exp(2.107 * (1.0 - calc) - 1.834);
      break;

    /* See Ketcham, et al, 1999, Figure 7c */
    case OH_PFU:
      calc = fabs(kinPar - 1.0);
      rmr0 = 0.84 * (1.0 - pow(1.0 - calc, 4.5));
      break;
  }

  k = 1.0 - rmr0;
  
  totAnnealLen = MIN_OBS_RCMOD;
  equivTotAnnLen = pow(totAnnealLen,1.0/k)*(1.0-rmr0)+rmr0;

  equivTime = 0.0;
  tempCalc = log(1.0 / ((temperature[numTTNodes - 2] + temperature[numTTNodes - 1]) / 2.0));

  for (node = numTTNodes-2; node >= 0; node--) {
    timeInt = time[node] - time[node + 1] + equivTime;
    x1 = (log(timeInt) - modKetch99.c2) / (tempCalc - modKetch99.c3);
    x2 = 1.0 + modKetch99.a * (modKetch99.c0 + modKetch99.c1 * x1);
    redLength[node] = pow(x2, 1.0 / modKetch99.a);
    x3 = 1.0 - modKetch99.b * redLength[node];
    redLength[node] = (x3 < 0) ? 0.0 : pow(x3, 1.0 / modKetch99.b);

    if (redLength[node] < equivTotAnnLen) redLength[node] = 0.0;
   /* Check to see if we've reached the end of the length distribution
   If so, we then do the kinetic conversion. */
    if ((redLength[node] == 0.0) || (node == 0)) {
        *firstTTNode = (node ? node+1 : node);
      
      for (nodeB = *firstTTNode; nodeB < numTTNodes-1; nodeB++) {
          /* Note sure why this should happen */
          if (redLength[nodeB] <= rmr0) {
            redLength[nodeB] = 0.0;
            *firstTTNode = nodeB;
          }
          else {
          /* This is equation 8 from Ketcham et al, 1999 */
            redLength[nodeB] = pow((redLength[nodeB] - rmr0)/(1.0 - rmr0),k);
            if (redLength[nodeB] < totAnnealLen) {
              redLength[nodeB] = 0.0;
              *firstTTNode = nodeB;
            }
          }
      }
     return;
    }

    /* Update tiq for this time step */
    if (redLength[node] < 0.999) {
      
      tempCalc = log(1.0 / ((temperature[node-1] + temperature[node]) / 2.0));
      equivTime = pow((1.0 - pow(redLength[node], modKetch99.b)) / modKetch99.b, modKetch99.a);
      equivTime = ((equivTime - 1.0) / modKetch99.a - modKetch99.c0) / modKetch99.c1;
      equivTime = exp(equivTime * (tempCalc - modKetch99.c3) + modKetch99.c2);
    
    }
  }  
}
