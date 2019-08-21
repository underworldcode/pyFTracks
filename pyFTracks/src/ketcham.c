#include <math.h>
#include "ketcham.h"
#include <stdio.h>

#define	ETCH_PIT_LENGTH	0
#define CL_PFU 1
#define OH_PFU 2
#define	CL_WT_PCT 3
#define UNIT_PARAM_A 4
#define WEAK_ETCHANT 0

#define MIN_OBS_RCMOD  0.55

/* Useful constants */
#define	SECS_PER_MA		3.1556736e13
#define U238SEC 4.91575e-18
#define SQRT2PI	2.50662827463
#define	PDF_NUMSD 4



double calculate_annealing_temperature(double abs_gradient){
       return 377.67 * pow(abs_gradient, 0.019837);
}

/*  calculate_reduced_stddev
	 Calculates the reduced standard deviation of a track population length
	 from the reduced mean length.  Based on Carlson and Donelick
 */
double  calculate_reduced_stddev(double redLength,int doProject){
  if (doProject) return(0.1081 - 0.1642 * redLength + 0.1052 * redLength * redLength);
  else return(0.4572 - 0.8815 * redLength + 0.4947 * redLength * redLength);

//double  calculate_reduced_stddev(double redLength,int doProject){
//  if (doProject) return(2.312 - 0.2442 * redLength + 0.008452 * redLength * redLength);
//  else return(7.464 - 0.8733 * redLength + 0.02858 * redLength * redLength);
}



/*  ketcham_age_correction
    Does the conversion from length to density for the Ketcham et al., 1999 model.
    The routine is placed "way up here" because it will also be used to estimate
    bias for population summing.

    Assumes we're passing in a c-axis-projected length
 */

/* The following text is taken from Forward Inverse and Modeling of LT Thermoch Data
 * Low-T Thermochronology: Techniques, Interpretations and Applications (ed. Reiners an Ehlers)
 *
 * The observational bias quantifies the relative probability of observation among different
 * fission-track populations calculated by the model. Highly annealed populations are less
 * likely to be detected and measured than less-annealed populations for 2 primary reasons.
 * - Shorter track are less frequently impinged and thus etched
 * - At advanced stage of annealing some proportion of tracks at high angles to the c-axis
 *   may be lost altogether, even though lower-angle tracks remain long
 * Thus the number of detectable tracks in the more annealed population diminishes, at a rate
 * dispropportionate to measured mean length (Ketcham 2003b). These 2 factors can be approximated
 * in a general way by using an empirical function that relates measured fission-track length to
 * fission-track density (e,g. Green 1998). The following is taken from Ketcham et al 2000 */

double ketcham_age_correction(double cparlen){
  if (cparlen > 0.765) return(1.600 * cparlen - 0.599);
  if (cparlen >= MIN_OBS_RCMOD) return(9.205 * cparlen * cparlen - 9.157 * cparlen + 2.269);
  return(0.0);
}

int refine_history(double *time, double *temperature, int npoints,
                   double max_temp_per_step, double max_temp_step_near_ta,
                   double *new_time, double *new_temperature, int *new_npoints){

  double default_timestep;
  double alternative_timestep;
  double gradient, abs_gradient;
  double temperature_interval;
  double end_temperature;
  double fact;
  double temp_per_step;
  double current_default_timestep;
  double Ta_near;
  double max_temperature;
  double timestep;
  double time_interval;

  int seg;

  *new_npoints = 1;

  alternative_timestep=0.0;

  new_temperature[0] = temperature[npoints - 1];
  new_time[0] = time[npoints - 1];

  default_timestep = time[npoints - 1] * 1.0 / 100.0;

  for (seg = npoints-1; seg > 0; seg--) {
      temperature_interval = temperature[seg] - temperature[seg-1]; 
      time_interval = time[seg] - time[seg-1];
      gradient = temperature_interval / time_interval;
      abs_gradient = fabs(gradient);
      end_temperature = temperature[seg-1];
      fact = gradient > 0 ? 0 : -1;

      temp_per_step = abs_gradient * default_timestep;

      if (temp_per_step <= max_temp_per_step)
          current_default_timestep = default_timestep;
      else {
          current_default_timestep = max_temp_per_step / abs_gradient;
      }
          
      if (abs_gradient < 0.1)
          Ta_near = 1000.; 
      else {
          Ta_near = calculate_annealing_temperature(abs_gradient) + 10.;
	  alternative_timestep = max_temp_step_near_ta / abs_gradient;
      }
      
      while (new_time[*new_npoints - 1] > time[seg-1]){
          
          max_temperature = new_temperature[*new_npoints - 1] + default_timestep * gradient * fact;
          if ((gradient < 0) && (max_temperature > end_temperature)) max_temperature = end_temperature;
          
          timestep = current_default_timestep;

          if (max_temperature > Ta_near)
              if (alternative_timestep < default_timestep) timestep = alternative_timestep;

          if (timestep + 0.001 > new_time[*new_npoints - 1] - time[seg - 1]){
              new_time[*new_npoints] = time[seg - 1];
	      new_temperature[*new_npoints] = end_temperature;
          }
          else {
              new_time[*new_npoints] = new_time[*new_npoints - 1] - timestep;
              new_temperature[*new_npoints] = new_temperature[*new_npoints - 1] - gradient * timestep;
          }
          *new_npoints = *new_npoints + 1;

      } 

 }

  return 1;
}

void ketch99_reduced_lengths(double *time, double *temperature, int numTTNodes,
                             double *redLength,  double kinPar, int kinParType,
                             int *firstTTNode)
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

void ketcham_sum_population(int num_points_pdf, int numTTNodes, int firstTTNode, int doProject,
                    int usedCf, double *time, double *temperature, double *pdfAxis,
                    double *pdf, double *cdf, double  initLength, double min_length, double  *redLength)
{
  int i,j;
  double weight, rLen, rStDev, obsBias, rmLen, calc, z;
  double wt1,wt2;

/* Sum curves for pdf */
  for (i=0; i < num_points_pdf; i++) pdf[i] = 0.0;
  for (i=0; i < num_points_pdf; i++) pdfAxis[i] = (double)(i * 1.0 + 0.5) * 20.0 / num_points_pdf;

  wt1 = exp(U238SEC * time[firstTTNode]) / U238SEC;
  
  for (j = firstTTNode; j < numTTNodes - 1; j++) {

    wt2 = exp(U238SEC * time[j+1]) / U238SEC;
    weight = wt1 - wt2;
    wt1 = wt2;
    rmLen = usedCf ? 1.396 * redLength[j] - 0.4017 : -1.499 * redLength[j] * redLength[j] + 4.150 * redLength[j] - 1.656;
    rLen = doProject ? redLength[j] : rmLen;
    rStDev = calculate_reduced_stddev(rLen, doProject);
    obsBias = ketcham_age_correction(redLength[j]);
    calc = weight * obsBias / (rStDev * SQRT2PI);
    if (rLen > 0) {
      for (i = 0; i < num_points_pdf; i++) {
        if (pdfAxis[i] >= min_length) {
          z = (rLen - pdfAxis[i] / initLength) / rStDev;
          if (z <= PDF_NUMSD) pdf[i] += calc * exp(-(z * z) / 2.0);
        }
      }
    }
  }

  /* Calculate cdfs. */
  cdf[0] = pdf[0];
  for (i=1; i < num_points_pdf; i++)
    cdf[i] = cdf[i - 1]+((pdf[i] + pdf[i - 1]) / 2.0)*(pdfAxis[i] - pdfAxis[i - 1]);

  /* Normalize */
  if (cdf[num_points_pdf-1] > 0.0)  /* Some non-zero lengths */
    for (i=0; i < num_points_pdf; i++)  {
      pdf[i] = pdf[i] / cdf[num_points_pdf - 1];
      cdf[i] = cdf[i] / cdf[num_points_pdf - 1];
    }
}


void ketcham_calculate_model_age(double *time, double *temperature, double  *redLength,
                                 int numTTNodes, int firstNode, double  *oldestModelAge,
		                         double  *ftModelAge, double stdLengthReduction,
		                         double  *redDensity){
  int     node;
  double  midLength;

  *redDensity = 0.0;
  *oldestModelAge = time[firstNode] / SECS_PER_MA;

  /* Correct each time interval for length reduction */
  for (*ftModelAge = 0.0, node = firstNode; node < numTTNodes - 2; node++) {
      
      midLength = (redLength[node] + redLength[node + 1]) / 2.0;
      *ftModelAge += ketcham_age_correction(midLength) * (time[node] - time[node + 1]);
      *redDensity += ketcham_age_correction(midLength);
  }

  *ftModelAge += ketcham_age_correction(redLength[numTTNodes - 2]) * (time[node] - time[node + 1]);
  *redDensity += ketcham_age_correction(redLength[numTTNodes - 2]); 
  *redDensity /= stdLengthReduction * (numTTNodes-2);
  
  /* Account for length reduction in length standard, convert to Ma */
  *ftModelAge /= (stdLengthReduction * SECS_PER_MA);
}
