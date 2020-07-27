#include <math.h>
#include "ketcham2007.h"
#include <stdio.h>

#define MIN_OBS_RCMOD  0.55

void ketch07_reduced_lengths(double *time, double *temperature, int numTTNodes,
                             double *redLength, double rmr0,
                             int *firstTTNode)
{
    int     node, nodeB;
    double  equivTime;
    double  timeInt, x1, x2;
    double  totAnnealLen;
    double  equivTotAnnLen;
    double  k;
    double  calc;
    double  tempCalc;
  
    typedef struct {
         double c0, c1, c2, c3, a;
    } annealModel;

    /* Fanning Curvilinear Model lcMod FC, See Ketcham 2007, Table 5c */
    annealModel modKetch07 = {0.39528, 0.01073, -65.12969, -7.91715, 0.04672};

    k = 1.04 - rmr0;
  
    totAnnealLen = MIN_OBS_RCMOD;
    equivTotAnnLen = pow(totAnnealLen, 1.0 / k) * (1.0 - rmr0) + rmr0;

    equivTime = 0.0;
    tempCalc = log(1.0 / ((temperature[numTTNodes - 2] + temperature[numTTNodes - 1]) / 2.0));
    for (node = numTTNodes-2; node >= 0; node--) {
        timeInt = time[node] - time[node + 1] + equivTime;
        x1 = (log(timeInt) - modKetch07.c2) / (tempCalc - modKetch07.c3);
        x2 = pow(modKetch07.c0 + modKetch07.c1 * x1, 1.0 / modKetch07.a) + 1.0;
        redLength[node] = 1.0 / x2;

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
            tempCalc = log(1.0 / ((temperature[node - 1] + temperature[node]) / 2.0));
            equivTime = pow(1.0 / redLength[node] - 1.0, modKetch07.a);
            equivTime = (equivTime - modKetch07.c0) / modKetch07.c1;
            equivTime = exp(equivTime * (tempCalc - modKetch07.c3) + modKetch07.c2);
        }
 
    }
}
