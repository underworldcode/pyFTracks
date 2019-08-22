#include <math.h>
#include "utilities.h"
#include <stdio.h>

#define MIN_OBS_RCMOD  0.55

#define	SECS_PER_MA		3.1556736e13
#define U238SEC 4.91575e-18
#define SQRT2PI	2.50662827463
#define	PDF_NUMSD 4


/************************************************************************************************
 * NAME: calculate_annealing_temperature
 * DESCRIPTION: Calculate the annealing temperature based on absolute temperature gradient
 * The total annealing temperature (TA) for F-apatite
 *  for a given heating or cooling rate (R) is given by the equation:
 *  
 *                      Ta = 377.67 * R**0.019837
 *
 */
double calculate_annealing_temperature(double abs_gradient)
{
       return 377.67 * pow(abs_gradient, 0.019837);
}


/************************************************************************************************  
 *
 *  NAME: calculate_reduced_stddev
 *  DESCRIPTION: Calculates the reduced standard deviation of a track population length
 *  from the reduced mean length.  Based on Carlson and Donelick
 */
double  calculate_reduced_stddev(double redLength,int doProject)
{
    if (doProject){
        return(0.1081 - 0.1642 * redLength + 0.1052 * redLength * redLength);
    }
    else{
        return(0.4572 - 0.8815 * redLength + 0.4947 * redLength * redLength);
    } 
//double  calculate_reduced_stddev(double redLength,int doProject){
//  if (doProject) return(2.312 - 0.2442 * redLength + 0.008452 * redLength * redLength);
//  else return(7.464 - 0.8733 * redLength + 0.02858 * redLength * redLength);
}


/*************************************************************************************************
 *  NAME: correct_observational_bias
 *
 *  DESCRIPTION: Does the conversion from length to density for the Ketcham et al., 1999 model.
 *  
 *  The routine is also used to estimate bias for population summing.
 *
 *  Assumes we're passing in a c-axis-projected length
 *
 *  The following text is taken from Forward Inverse and Modeling of LT Thermoch Data
 *  Low-T Thermochronology: Techniques, Interpretations and Applications (ed. Reiners an Ehlers)
 *
 *  The observational bias quantifies the relative probability of observation among different
 *  fission-track populations calculated by the model. Highly annealed populations are less
 *  likely to be detected and measured than less-annealed populations for 2 primary reasons.
 *    - Shorter track are less frequently impinged and thus etched
 *    - At advanced stage of annealing some proportion of tracks at high angles to the c-axis
 *      may be lost altogether, even though lower-angle tracks remain long
 * Thus the number of detectable tracks in the more annealed population diminishes, at a rate
 * dispropportionate to measured mean length (Ketcham 2003b). These 2 factors can be approximated
 * in a general way by using an empirical function that relates measured fission-track length to
 * fission-track density (e,g. Green 1998). The following is taken from Ketcham et al 2000 
 */

double correct_observational_bias(double cparlen)

{
    if (cparlen > 0.765) return(1.600 * cparlen - 0.599);
    if (cparlen >= MIN_OBS_RCMOD) return(9.205 * cparlen * cparlen - 9.157 * cparlen + 2.269);
    return(0.0);
}



/************************************************************************************************
* NAME: refine_history
*
* DESCRIPTION: Interpolate Time Temperature path
* Takes the time-temperature path specification and subdivides it for
* calculation in isothermal intervals. 
* 
* Reference:
* 
* Ketcham, R. A. (2005). Forward and Inverse Modeling of Low-Temperature
* Thermochronometry Data. Reviews in Mineralogy and Geochemistry, 58(1),
* 275–314. doi:10.2138/rmg.2005.58.11
*
* It is calibrated to facilitate 0.5% accuracy for end-member F-apatite by
* having a maximum temperature step of 3.5 degrees C when the model temperature
* is within 10C of the total annealing temperature. Before this cutoff the
* maximum temperature step required is 8 C. If the overall model tine steps are
* too large, these more distant requirement may not be meet.

* Quoted text:
* 
* "The more segments a time-temperature path is subdivided into, the more accurate
* the numerical solution will be. Conversely, an excessive number of time steps
* will slow computation down unnecessarily. The optimal time step size to achieve a desired
* solution accuracy was examined in detail by Issler (1996b), who demonstrated that time
* steps should be smaller as the total annealing temperature of apatite is approached.
* For the Ketcham et al. (1999) annealing model for F-apatite, Ketcham et al. (2000) found that 0.5%
* precision is assured if there is no step with greater than a 3.5 ºC change within 10 ºC of
* the F-apatite total annealing temperature.*/ 

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
    obsBias = correct_observational_bias(redLength[j]);
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
      *ftModelAge += correct_observational_bias(midLength) * (time[node] - time[node + 1]);
      *redDensity += correct_observational_bias(midLength);
  }

  *ftModelAge += correct_observational_bias(redLength[numTTNodes - 2]) * (time[node] - time[node + 1]);
  *redDensity += correct_observational_bias(redLength[numTTNodes - 2]); 
  *redDensity /= stdLengthReduction * (numTTNodes-2);
  
  /* Account for length reduction in length standard, convert to Ma */
  *ftModelAge /= (stdLengthReduction * SECS_PER_MA);
}
