#include "mex.h"
#include <time.h>


//mex CFLAGS='$CFLAGS -ffast-math' filter.c

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    
    mwSize n_data_samples = mxGetNumberOfElements(prhs[2]);
    mwSize b_length = mxGetNumberOfElements(prhs[0]);
    mwSize a_length = mxGetNumberOfElements(prhs[0]);
    
    double b[8] = {0.123,0.234,0.345,0.456,0.567,0.678,0.789,0.890};

    
    double *x = calloc(n_data_samples,sizeof(double));
        
    for (int i = 0; i <n_data_samples; i++){
        x[i] = i;
    }
        
    //double b[8] = {0.123,0.234,0.345,0.456,0.567,0.678,0.789,0.890};

    
    //Just running memory allocation is 18 ms for 1e7 samples ...
    //Doing 5e7 yields 6 ms ... wtf????

    double *y = calloc(n_data_samples,sizeof(double));


    clock_t clock_begin;
    clock_t clock_end;
    
    clock_begin = clock();
    #pragma omp parallel for
    for (mwSize j = b_length; j < n_data_samples; j++) {
        y[j] = b[0]*x[j] + b[1]*x[j-1] + b[2]*x[j-2] + b[3]*x[j-3] +
                + b[4]*x[j-4] + b[5]*x[j-5] + b[6]*x[j-6] + b[7]*x[j-7];
    }
    clock_end = clock();
    
    double run_time = (double)(clock_end - clock_begin) / CLOCKS_PER_SEC;
    mexPrintf("t1: %g\n",run_time);
    free(x);
    free(y);
}