

double calculate_annealing_temperature(double abs_gradient);
double ReducedStdev(double redLength,int doProject);
double AgeCorrectionKet(double cparlen);

int refine_history(double *time, double *temperature, int npoints,
                   double max_temp_per_step, double max_temp_step_near_ta,
                   double *new_time, double *new_temperature, int *new_npoints);

void ketcham_sum_population(int numPDFPts, int numTTNodes, int firstTTNode, int doProject,
                            int usedCf, double *time, double *temperature, double *pdfAxis,
			    double *pdf, double *cdf, double  initLength, double min_length,
                            double  *redLength);

void ketcham_calculate_model_age(double *time, double *temperature, double  *redLength,
                                 int numTTNodes, int firstNode, double  *oldestModelAge,
		                 double  *ftModelAge, double stdLengthReduction,
		                 double  *redDensity);
