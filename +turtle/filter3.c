#include "mex.h"
#include <time.h>
#include <omp.h>
#include <immintrin.h>


//mex CFLAGS='$CFLAGS -ffast-math' filter.c

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    
    mwSize n_data_samples = mxGetNumberOfElements(prhs[2]);
    mwSize b_length = mxGetNumberOfElements(prhs[0]);
    mwSize a_length = mxGetNumberOfElements(prhs[0]);
    
    //double b[8] = {0.123,0.234,0.345,0.456,0.567,0.678,0.789,0.890};
    double c[8] = {0.123,0.234,0.345,0.456,0.567,0.678,0.789,0.890};

    
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

    __m256d x0,x1,x2,x3,x4,x5,x6,x7; //0 - 0 back
    __m256d c0 = _mm256_set1_pd(c[0]);
    __m256d c1 = _mm256_set1_pd(c[1]);
    __m256d c2 = _mm256_set1_pd(c[2]);
    __m256d c3 = _mm256_set1_pd(c[3]);
    __m256d c4 = _mm256_set1_pd(c[4]);
    __m256d c5 = _mm256_set1_pd(c[5]);
    __m256d c6 = _mm256_set1_pd(c[6]);
    __m256d c7 = _mm256_set1_pd(c[6]);
    
    __m256d y0, y1, y2, y3, y4, y5, y6, y7;

                                          
    // 17  16  15  14  13  12  11  10  9  8  7  6  5  4  3  2  1  0  (round 1)
    //                                                   3  2  1  0 <= 0 back
    //                                       6  5  4  3             <= 1 back
    //                         9   8   7  6                         <= 2 back
    // 
    //
    
    // 17  16  15  14  13  12  11  10  9  8  7  6  5  4  3  2  1  0  (round 2)                                     
    //                                       7  6  5  4  <= 0 back
    //                         10  9   8  7
    
    int starts[100];
    int stops[100]; 
    
    //For all but the first bit, need to backtrack to fill in gaps
    
    // 17  16  15  14  13  12  11  10  9  8  7  6  5  4  3  2  1  0
    
    int cur_start;
    int cur_end = -1;
    //TODO: We'll have extra values ...
    int step_size = (n_data_samples/400)*4;
    for (int i = 0; i < 100; i++){
        cur_start = cur_end + 1;
        cur_end = cur_end + step_size;
        starts[i] = cur_start;
        stops[i] = cur_end;
    }
    
    mexPrintf("Step Size: %d\n",step_size);
    
    //sprintf("stop: %d\n",stops[99]);
    
    //omp_set_num_threads(1);
    
    //#pragma omp parallel for simd
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
    
        for (int j = cur_start; j < cur_end; j+=4) {

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
    
    
    clock_end = clock();
    
    double run_time = (double)(clock_end - clock_begin) / CLOCKS_PER_SEC;
    mexPrintf("t1: %g\n",run_time);
    mexPrintf("y: %g\n",y[10000000]);
    free(x);
    free(y);
}