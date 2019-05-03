#include "mex.h"
#include <time.h>
#include <omp.h>
#include <immintrin.h>

#define N_CHUNKS 100

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


//This may be slightly faster
//Need to work out bugs ...
if (0){
    
    __m256d x0,x1,x2,x3,x4,x5,x6,x7; //0 - 0 back
    __m256d c0 = _mm256_set1_pd(b[0]);
    __m256d c1 = _mm256_set1_pd(b[1]);
    __m256d c2 = _mm256_set1_pd(b[2]);
    __m256d c3 = _mm256_set1_pd(b[3]);
    __m256d c4 = _mm256_set1_pd(b[4]);
    __m256d c5 = _mm256_set1_pd(b[5]);
    __m256d c6 = _mm256_set1_pd(b[6]);
    __m256d c7 = _mm256_set1_pd(b[7]);
    
    __m256d y0, y1, y2, y3, y4, y5, y6, y7;
    
    
    int starts[N_CHUNKS];
    int stops[N_CHUNKS]; 
    
    //For all but the first bit, need to backtrack to fill in gaps
    
    // 17  16  15  14  13  12  11  10  9  8  7  6  5  4  3  2  1  0
    
    int cur_start;
    int cur_end = -1;
    //TODO: We'll have extra values ...
    int step_size = ((n_data_samples-3)/(N_CHUNKS*4))*4;
    for (int i = 0; i < N_CHUNKS; i++){
        cur_start = cur_end + 1;
        cur_end = cur_end + step_size;
        starts[i] = cur_start;
        stops[i] = cur_end;
    }
        
    //mexPrintf("start: %d\n",starts[99]);
    //mexPrintf("stop: %d\n",stops[99]);
    
    #pragma omp parallel for simd
    for (int k = 0; k < 100; k++){
    
        cur_start = starts[k];
        cur_end = stops[k];
        y1 = _mm256_loadu_pd(x+cur_start);     //0:3 => becomes y0 on first run
        y2 = _mm256_loadu_pd(x+cur_start+4);   //4:7 => becomes y1 on first run
        y3 = _mm256_loadu_pd(x+cur_start+8);
        y4 = _mm256_loadu_pd(x+cur_start+12);
        y5 = _mm256_loadu_pd(x+cur_start+16);
        y6 = _mm256_loadu_pd(x+cur_start+20);
        y7 = _mm256_loadu_pd(x+cur_start+24);
    
    
    //TODO: still need to manually execute each of the first seven
//     y1 = _mm256_loadu_pd(x);     //0:3 => becomes y0 on first run
//     y2 = _mm256_loadu_pd(x+4);   //4:7 => becomes y1 on first run
//     y3 = _mm256_loadu_pd(x+8);
//     y4 = _mm256_loadu_pd(x+12);
//     y5 = _mm256_loadu_pd(x+16);
//     y6 = _mm256_loadu_pd(x+20);
//     y7 = _mm256_loadu_pd(x+24);
    
    // 17  16  15  14  13  12  11  10  9  8  7  6  5  4  3  2  1  0 
    //                                                            
    //  - completed numbers need to start at 7
    //  - we start storing when incomplete ...
    //  - go back (filter_length-1)*4
    
    //old 
    //for (int j = 28; j < n_samples; j+=4)
    
        
        
        
        for (int j = cur_start+28; j < cur_end; j+=4) {

            //TODO: Can we avoid the awkward loads????
            //Perhaps store temporary variables ...

            //Can we do the shift in the fma stage????

            y0 = y1;
            y1 = y2;
            y2 = y3;
            y3 = y4;
            y4 = y5;
            y5 = y6;
            y6 = y7;

            x0 = _mm256_loadu_pd(&x[j-28-0]);
            x1 = _mm256_loadu_pd(&x[j-24-1]);
            x2 = _mm256_loadu_pd(&x[j-20-2]);
            x3 = _mm256_loadu_pd(&x[j-16-3]);
            x4 = _mm256_loadu_pd(&x[j-12-4]);
            x5 = _mm256_loadu_pd(&x[j-8-5]);
            x6 = _mm256_loadu_pd(&x[j-4-6]);
            x7 = _mm256_loadu_pd(&x[j-0-7]);

            y0 = _mm256_add_pd(c0,x0);
            _mm256_storeu_pd(&y[j-28], y0);
            y1 = _mm256_fmadd_pd(c1,x1,y1);
            y2 = _mm256_fmadd_pd(c2,x2,y2);
            y3 = _mm256_fmadd_pd(c3,x3,y3);
            y4 = _mm256_fmadd_pd(c4,x4,y4);
            y5 = _mm256_fmadd_pd(c5,x5,y5);
            y6 = _mm256_fmadd_pd(c6,x6,y6);
            y7 = _mm256_fmadd_pd(c7,x7,y7);
        }
    }   
    
}
    
if (0){
  __m256d x0,x1,x2,x3,x4,x5,x6,x7; //0 - 0 back
    __m256d c0 = _mm256_set1_pd(b[0]);
    __m256d c1 = _mm256_set1_pd(b[1]);
    __m256d c2 = _mm256_set1_pd(b[2]);
    __m256d c3 = _mm256_set1_pd(b[3]);
    __m256d c4 = _mm256_set1_pd(b[4]);
    __m256d c5 = _mm256_set1_pd(b[5]);
    __m256d c6 = _mm256_set1_pd(b[6]);
    __m256d c7 = _mm256_set1_pd(b[7]);
    
    __m256d y0, y1, y2, y3, y4, y5, y6, y7;

    int cur_start = 0;
    y1 = _mm256_loadu_pd(x);     //0:3 => becomes y0 on first run
    y2 = _mm256_loadu_pd(x+4);   //4:7 => becomes y1 on first run
    y3 = _mm256_loadu_pd(x+8);
    y4 = _mm256_loadu_pd(x+12);
    y5 = _mm256_loadu_pd(x+16);
    y6 = _mm256_loadu_pd(x+20);
    y7 = _mm256_loadu_pd(x+24);
    
    for (int j = 28; j < n_data_samples; j+=4){
    

            //TODO: Can we avoid the awkward loads????
            //Perhaps store temporary variables ...

            //Can we do the shift in the fma stage????

            y0 = y1;
            y1 = y2;
            y2 = y3;
            y3 = y4;
            y4 = y5;
            y5 = y6;
            y6 = y7;

            x0 = _mm256_loadu_pd(&x[j-28-0]);
            x1 = _mm256_loadu_pd(&x[j-24-1]);
            x2 = _mm256_loadu_pd(&x[j-20-2]);
            x3 = _mm256_loadu_pd(&x[j-16-3]);
            x4 = _mm256_loadu_pd(&x[j-12-4]);
            x5 = _mm256_loadu_pd(&x[j-8-5]);
            x6 = _mm256_loadu_pd(&x[j-4-6]);
            x7 = _mm256_loadu_pd(&x[j-0-7]);

            y0 = _mm256_add_pd(c0,x0);
            _mm256_storeu_pd(&y[j-28], y0);
            y1 = _mm256_fmadd_pd(c1,x1,y1);
            y2 = _mm256_fmadd_pd(c2,x2,y2);
            y3 = _mm256_fmadd_pd(c3,x3,y3);
            y4 = _mm256_fmadd_pd(c4,x4,y4);
            y5 = _mm256_fmadd_pd(c5,x5,y5);
            y6 = _mm256_fmadd_pd(c6,x6,y6);
            y7 = _mm256_fmadd_pd(c7,x7,y7);
        }      
        
    }

    
    //TODO: Change y++ and x0++ to y+=4; and x0+=4;
if (1){
    //Approach 1: Std
    
    
    y[0] = b[0]*x[0];
    y[1] = b[0]*x[1] + b[1]*x[0];
    y[2] = b[0]*x[2] + b[1]*x[1] + b[2]*x[0];
    y[3] = b[0]*x[3] + b[1]*x[2] + b[2]*x[1] + b[3]*x[0];
    y[4] = b[0]*x[4] + b[1]*x[3] + b[2]*x[2] + b[3]*x[1] + b[4]*x[0];
    y[5] = b[0]*x[5] + b[1]*x[4] + b[2]*x[3] + b[3]*x[2] + b[4]*x[1] + b[5]*x[0]; 
    y[6] = b[0]*x[6] + b[1]*x[5] + b[2]*x[4] + b[3]*x[3] + b[4]*x[2] + b[5]*x[1] + b[6]*x[0];
    
    #pragma omp parallel for
    for (mwSize j = b_length-1; j < n_data_samples; j++) {
        y[j] = b[0]*x[j] + b[1]*x[j-1] + b[2]*x[j-2] + b[3]*x[j-3] +
                + b[4]*x[j-4] + b[5]*x[j-5] + b[6]*x[j-6] + b[7]*x[j-7];
    }
}

    
    //double run_time = (double)(clock_end - clock_begin) / CLOCKS_PER_SEC;
    //mexPrintf("t1: %g\n",run_time);
}


//    double b0 = b[0];
//     double b1 = b[1];
//     double b2 = b[2];
//     double b3 = b[3];
//     double b4 = b[4];
//     double b5 = b[5]; 
//     double b6 = b[6]; 
//     double b7 = b[7]; 
//     
    

    

    

//     int tid;
//     int n_threads;
//     #pragma omp parallel
//     { 
//         int tid = omp_get_thread_num();
//         int n_threads = omp_get_num_threads();
//         
//         double *x0 = x + tid + b_length;
//         double *x1 = x + tid + b_length - 1;
//         double *x2 = x + tid + b_length - 2;
//         double *x3 = x + tid + b_length - 3;
//         double *x4 = x + tid + b_length - 4;
//         double *x5 = x + tid + b_length - 5;
//         double *x6 = x + tid + b_length - 6;
//         double *x7 = x + tid + b_length - 7;
//         
//         double *y_out = y + tid + b_length;
//         
//         //This seems slightly faster but is less amenable to openmp
//         for (mwSize j = tid + b_length; j < n_data_samples; j+=n_threads) {
//             *y_out = *x0*b0 + *x1*b1 + *x2*b2 + *x3*b3 + *x4*b4 + *x5*b5 + *x6*b6 +
//                     *x7*b7;
//             y_out++;
//             x0++;
//             x1++;
//             x2++;
//             x3++;
//             x4++;
//             x5++;
//             x6++;
//             x7++;
//         }
//     }

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