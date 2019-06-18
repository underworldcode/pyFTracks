#include <math.h>
#include "ketcham.h"
#include <stdio.h>

#define NUM_KINETIC_TYPES 4
#define	ETCH_PIT_LENGTH	0
#define CL_PFU 1
#define OH_PFU 2
#define	CL_WT_PCT 3

#define MIN_OBS_RCMOD  0.55

/* Useful constants */
#define PI 3.1415926535
#define	SECS_PER_MA		3.1556736e13
#define	KELVINS_AT_0C	273.15
#define	U238YR	1.55125e-10
#define	U238MA	1.55125e-4
#define U238SEC 4.91575e-18
#define SQRT2PI	2.50662827463

#define	PDF_NUMSD 4

typedef struct {
    double c0, c1, c2, c3, a, b, lMin;
} annealModel;

annealModel modKetchamEtAl = {-19.844,0.38951,-51.253,-7.6423,-0.12327,-11.988,0.0};

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
}



/*  ketcham_age_correction
    Does the conversion from length to density for the Ketcham et al., 1999 model.
    The routine is placed "way up here" because it will also be used to estimate
    bias for population summing.

    Assumes we're passing in a c-axis-projected length
 */
double ketcham_age_correction(double cparlen){
  if (cparlen > 0.757) return(1.600 * cparlen - 0.599);
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

void ketcham_calculate_model_length(double *time, double *temperature, int numTTNodes,
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

  rmr0 = 0;

  /* Calculate the rmr0-k values for the kinetic parameter given */
  switch (kinParType) {

    case ETCH_PIT_LENGTH:
  
      if (kinPar <= 1.75) rmr0 = 0.84;
      else if (kinPar >= 4.58) rmr0 = 0.0;
      else rmr0 = 1.0 - exp(0.647 * (kinPar - 1.75) - 1.834);
      break;

    case CL_WT_PCT:
      /* Just convert the kinetic parameter to Cl apfu
         Note that this invalidates kinPar for the rest of the routine */
      kinPar = kinPar * 0.2978;
    
    case CL_PFU:

      calc = fabs(kinPar - 1.0);
      if (calc <= 0.130) rmr0 = 0.0;
      else rmr0 = 1.0-exp(2.107 * (1.0 - calc) - 1.834);
      break;
    
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
    x1 = (log(timeInt) - modKetchamEtAl.c2) / (tempCalc - modKetchamEtAl.c3);
    x2 = 1.0 + modKetchamEtAl.a * (modKetchamEtAl.c0 + modKetchamEtAl.c1 * x1);
    redLength[node] = pow(x2, 1.0 / modKetchamEtAl.a);
    x3 = 1.0 - modKetchamEtAl.b * redLength[node];
    redLength[node] = (x3 < 0) ? 0.0 : pow(x3, 1.0 / modKetchamEtAl.b);

    if (redLength[node] < equivTotAnnLen) redLength[node] = 0.0;
   /* Check to see if we've reached the end of the length distribution
   If so, we then do the kinetic conversion. */
    if ((redLength[node] == 0.0) || (node == 0)) {
        *firstTTNode = (node ? node+1 : node);
      
      for (nodeB = *firstTTNode; nodeB < numTTNodes-1; nodeB++) {
          if (redLength[nodeB] <= rmr0) {
            redLength[nodeB] = 0.0;
            *firstTTNode = nodeB;
          }
          else {
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
      equivTime = pow((1.0 - pow(redLength[node], modKetchamEtAl.b)) / modKetchamEtAl.b, modKetchamEtAl.a);
      equivTime = ((equivTime - 1.0) / modKetchamEtAl.a - modKetchamEtAl.c0) / modKetchamEtAl.c1;
      equivTime = exp(equivTime * (tempCalc - modKetchamEtAl.c3) + modKetchamEtAl.c2);
    
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
