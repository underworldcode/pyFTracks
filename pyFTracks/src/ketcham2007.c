#include <math.h>
#include "ketcham2007.h"
#include <stdio.h>

#define	ETCH_PIT_LENGTH	0
#define CL_PFU 1
#define OH_PFU 2
#define	CL_WT_PCT 3
#define UNIT_PARAM_A 4
#define WEAK_ETCHANT 0

#define MIN_OBS_RCMOD  0.55

void ketch07_reduced_lengths(double *time, double *temperature, int numTTNodes,
                             double *redLength, double kinPar, int kinParType,
                             int *firstTTNode, int etchant)
{
    int     node, nodeB;
    double  equivTime;
    double  timeInt, x1, x2;
    double  totAnnealLen;
    double  equivTotAnnLen;
    double  rmr0, k;
    double  calc;
    double  tempCalc;
  
    typedef struct {
         double c0, c1, c2, c3, a;
    } annealModel;

    /* Fanning Curvilinear Model lcMod FC, See Ketcham 2007, Table 5c */
    annealModel modKetch07 = {0.39528, 0.01073, -65.12969, -7.91715, 0.04672};

    rmr0 = 0;

    switch (kinParType) {

        case ETCH_PIT_LENGTH:
            /* Here depends on the etchant (5.5 or 5.0 HNO3) */
            /* This is based on the relation between the fitted rmr0 values and
             * the Dpar etched using a 5.5M etchant as published in Ketcham et al, 2007,
             * Figure 6b */
            /* We use the linear conversion defined in Ketcham et al 2007 to
             * make sure that we are using 5.5M DPar */
            if (etchant == WEAK_ETCHANT) kinPar = 0.9231 * kinPar + 0.2515;
            // The following is for Durango like apatites
            if (kinPar <= 1.75) rmr0 = 0.84; //
            // for apatites with DPars larger than the ones measured for the B2 apatite.
            else if (kinPar >= 4.58) rmr0 = 0.0;
            // Fit from Ketcham et al, 2007
            else rmr0 = 0.84 * pow((4.58 - kinPar) / 2.98, 0.21);
            break;

        case CL_WT_PCT:
            /* Convert %wt to APFU */
            kinPar = kinPar * 0.2978;
            /* Relation between fitted rmr0 value from the fanning curvilinear model and
             * Cl content is taken from Ketcham et al 2007 Figure 6a*/
            calc = fabs(kinPar - 1.0);
            if (calc <= 0.130) rmr0 = 0.0;
            else rmr0 = 0.83 * pow((calc - 0.13) / 0.87, 0.23);
            break;
        
        case CL_PFU:
            /* Relation between fitted rmr0 value from the fanning curvilinear model and
             * Cl content is taken from Ketcham et al 2007 Figure 6a*/
            calc = fabs(kinPar - 1.0);
            if (calc <= 0.130) rmr0 = 0.0;
            else rmr0 = 0.83 * pow((calc - 0.13) / 0.87, 0.23);
            break;

        case UNIT_PARAM_A:
            /* This is based on the relation between the fitted rmr0 values and
             * the cell parameter A as published in Ketcham et al, 2007,
             * Figure 6c */
            // For apatites with A > A(B2 apatite)
            if (kinPar >= 9.51) rmr0 = 0.0;
            // Fit from Ketcham et al, 2007
            else rmr0 = 0.84 * pow((9.509 - kinPar) / 0.162, 0.175);
            break;

    }

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
