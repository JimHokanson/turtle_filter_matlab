#include "mex.h"
#include <time.h>
#include <omp.h>

//mex CFLAGS='$CFLAGS -ffast-math' filter.c

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    
    mwSize n_data_samples = mxGetNumberOfElements(prhs[2]);
    mwSize b_length = mxGetNumberOfElements(prhs[0]);
    mwSize a_length = mxGetNumberOfElements(prhs[0]);
    
    double *x = mxGetData(prhs[2]);
    double *b = mxGetData(prhs[0]);
    double *a = mxGetData(prhs[1]);
    
    //Just running memory allocation is 18 ms for 1e7 samples ...
    //Doing 5e7 yields 6 ms ... wtf????
    plhs[0] = mxCreateDoubleMatrix(1,0,mxREAL);

    
    double *y = mxCalloc(n_data_samples,sizeof(double));
    mxSetData(plhs[0],y);
    mxSetN(plhs[0],n_data_samples);

    //clock_t clock_begin;
    //clock_t clock_end;
    
//     double *x0;
//     double *x1;
//     double *x2;
//     double *x3;
//     double *x4;
//     double *x5;
//     double *x6;
//     double *x7;
//     
//     x0 = x + 7;
//     x1 = x + 6;
//     x2 = x + 5;
//     x3 = x + 4;
//     x4 = x + 3;
//     x5 = x + 2;
//     x6 = x + 1;
//     x7 = x;
    
    double b0 = b[0];
    double b1 = b[1];
    double b2 = b[2];
    double b3 = b[3];
    double b4 = b[4];
    double b5 = b[5]; 
    double b6 = b[6]; 
    double b7 = b[7]; 
    
    
// // //     int tid;
// // //     int n_threads;
// // //     mwSize start_I;
// // //     mwSize end_I;
// // // //     mwSize wtf[4];
// // // //     mwSize wtf2[4];
// // //     #pragma omp parallel
// // //     { 
// // //         int tid = omp_get_thread_num();
// // //         int n_threads = omp_get_num_threads();
// // //         if (tid == 0){
// // //             start_I = b_length;
// // //         }else{
// // //             //Note < stopping condition
// // //             start_I = (n_data_samples*tid)/n_threads;
// // //         }   
// // //         
// // //         if (tid == n_threads-1){
// // //             end_I = n_data_samples;
// // //         }else{
// // //             end_I = (n_data_samples*(tid+1))/n_threads;
// // //         } 
// // //         
// // // //         wtf[tid] = start_I;
// // // //         wtf2[tid] = end_I;
// // //         
// // //         double *x0 = x + tid;
// // //         double *x1 = x + start_I - 1;
// // //         double *x2 = x + start_I - 2;
// // //         double *x3 = x + start_I - 3;
// // //         double *x4 = x + start_I - 4;
// // //         double *x5 = x + start_I - 5;
// // //         double *x6 = x + start_I - 6;
// // //         double *x7 = x + start_I - 7;
// // //         
// // //         double *y_out = y + start_I;
// // //         
// // //         //This seems slightly faster but is less amenable to openmp
// // //         for (mwSize j = start_I; j < end_I; j++) {
// // //             *y_out = *x0*b0 + *x1*b1 + *x2*b2 + *x3*b3 + *x4*b4 + *x5*b5 + *x6*b6 +
// // //                     *x7*b7;
// // //             y_out++;
// // //             x0++;
// // //             x1++;
// // //             x2++;
// // //             x3++;
// // //             x4++;
// // //             x5++;
// // //             x6++;
// // //             x7++;
// // //         }
// // //     }
    
//     mexPrintf("s0: %d\n",wtf[0]);
//     mexPrintf("e0: %d\n",wtf2[0]);
//     mexPrintf("s1: %d\n",wtf[1]);
//     mexPrintf("e1: %d\n",wtf2[1]);
//     mexPrintf("s2: %d\n",wtf[2]);
//     mexPrintf("e2: %d\n",wtf2[2]);
//     mexPrintf("s3: %d\n",wtf[3]);
//     mexPrintf("e3: %d\n",wtf2[3]);
//     
    
//     //This seems slightly faster but is less amenable to openmp
//     for (mwSize j = b_length; j < n_data_samples; j++) {
//         *y = *x0*b0 + *x1*b1 + *x2*b2 + *x3*b3 + *x4*b4 + *x5*b5 + *x6*b6 +
//                 *x7*b7;
//         y++;
//         x0++;
//         x1++;
//         x2++;
//         x3++;
//         x4++;
//         x5++;
//         x6++;
//         x7++;
//     }
    
    //TODO: Change y++ and x0++ to y+=4; and x0+=4;
    
    //clock_begin = clock();
    #pragma omp parallel for
    for (mwSize j = b_length; j < n_data_samples; j++) {
        y[j] = b[0]*x[j] + b[1]*x[j-1] + b[2]*x[j-2] + b[3]*x[j-3] +
                + b[4]*x[j-4] + b[5]*x[j-5] + b[6]*x[j-6] + b[7]*x[j-7];
    }
    
//memcpy(y,x,n_data_samples);
//     for (mwSize j = b_length; j < n_data_samples; j++) {
//         y++;
//         x++;
//         *y = (*b)*(*x);
//     }
    //clock_end = clock();
    
    //double run_time = (double)(clock_end - clock_begin) / CLOCKS_PER_SEC;
    //mexPrintf("t1: %g\n",run_time);
}