#include "mex.h"
#include <time.h>
#include <omp.h>
#include <immintrin.h>

#define N_CHUNKS 100

//mex CFLAGS='$CFLAGS -ffast-math' filter.c

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    //  turtle.filter(b,a,x)
    
    //  TODO: Add on FIR and IIR switches
    
    int IIR_option = 0;
    if (nrhs == 4){
        //TODO: Check type of 3
        IIR_option = (int)mxGetScalar(prhs[3]);
    }
    
//       if (nrhs < 3 || nrhs > 5) {
//      mexErrMsgIdAndTxt(ERR_ID   "BadNInput",
//                        ERR_HEAD "3 to 5 inputs required.");
//   }
//   if (nlhs > 2) {
//      mexErrMsgIdAndTxt(ERR_ID   "BadNOutput",
//                        ERR_HEAD "2 outputs allowed.");
//   }
//   
//   // Check type of inputs:
//   if (!mxIsDouble(b_in) || !mxIsDouble(a_in)) {
//      mexErrMsgIdAndTxt(ERR_ID   "BadTypeInput1_2",
//                        ERR_HEAD "Filter parameters must be DOUBLES.");
//   }
    
    
    //FIR
    //0 - openmp with SIMD
    //1 - SIMD on its own
    //2 - STD approach (with openmp)
    //3 - TODO: std approach with pointers - no openmp ...
    
    mwSize n_data_samples = mxGetNumberOfElements(prhs[2]);
    mwSize b_length = mxGetNumberOfElements(prhs[0]);
    mwSize a_length = mxGetNumberOfElements(prhs[1]);
    
    double *x = mxGetData(prhs[2]);
    double *b = mxGetData(prhs[0]);
    double *a = mxGetData(prhs[1]);
    
    //Just running memory allocation is 18 ms for 1e7 samples ...
    //Doing 5e7 yields 6 ms ... wtf????
    plhs[0] = mxCreateDoubleMatrix(1,0,mxREAL);

    
    double *y = mxCalloc(n_data_samples,sizeof(double));
    mxSetData(plhs[0],y);
    mxSetN(plhs[0],n_data_samples);


    
//SIMD, with openmp
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

//SIMD, without openmp    
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


//Standard approach - FIR
//-------------------------------------------------------------------------
if (1){
    //Approach 1: Std
 	if (b_length == 1){
        #pragma omp parallel for
        for (mwSize j = b_length-1; j < n_data_samples; j++){
            y[j] = b[0]*x[j];
        }
    }else if (b_length == 2){
        y[0] = b[0]*x[0];
        #pragma omp parallel for
        for (mwSize j = b_length-1; j < n_data_samples; j++){
            y[j] = b[0]*x[j] + b[1]*x[j-1];
        }
    }else if(b_length == 3){    
     	y[0] = b[0]*x[0];
        y[1] = b[0]*x[1] + b[1]*x[0];
        #pragma omp parallel for
        for (mwSize j = b_length-1; j < n_data_samples; j++){
            y[j] = b[0]*x[j] + b[1]*x[j-1] + b[2]*x[j-2];
        }        
    }else if (b_length == 4){    
     	y[0] = b[0]*x[0];
        y[1] = b[0]*x[1] + b[1]*x[0];
        y[2] = b[0]*x[2] + b[1]*x[1] + b[2]*x[0];

        #pragma omp parallel for
        for (mwSize j = b_length-1; j < n_data_samples; j++){
            y[j] = b[0]*x[j] + b[1]*x[j-1] + b[2]*x[j-2] + b[3]*x[j-3];
        }
    }else if (b_length == 5){    
     	y[0] = b[0]*x[0];
        y[1] = b[0]*x[1] + b[1]*x[0];
        y[2] = b[0]*x[2] + b[1]*x[1] + b[2]*x[0];
        y[3] = b[0]*x[3] + b[1]*x[2] + b[2]*x[1] + b[3]*x[0];

        #pragma omp parallel for
        for (mwSize j = b_length-1; j < n_data_samples; j++){
            y[j] = b[0]*x[j] + b[1]*x[j-1] + b[2]*x[j-2] + b[3]*x[j-3] +
                    + b[4]*x[j-4];
        }
    }else if (b_length == 6){
    	y[0] = b[0]*x[0];
        y[1] = b[0]*x[1] + b[1]*x[0];
        y[2] = b[0]*x[2] + b[1]*x[1] + b[2]*x[0];
        y[3] = b[0]*x[3] + b[1]*x[2] + b[2]*x[1] + b[3]*x[0];
        y[4] = b[0]*x[4] + b[1]*x[3] + b[2]*x[2] + b[3]*x[1] + b[4]*x[0];

        #pragma omp parallel for
        for (mwSize j = b_length-1; j < n_data_samples; j++) {
            y[j] = b[0]*x[j] + b[1]*x[j-1] + b[2]*x[j-2] + b[3]*x[j-3] +
                    + b[4]*x[j-4] + b[5]*x[j-5];
        }
    }else if (b_length == 7){
        y[0] = b[0]*x[0];
        y[1] = b[0]*x[1] + b[1]*x[0];
        y[2] = b[0]*x[2] + b[1]*x[1] + b[2]*x[0];
        y[3] = b[0]*x[3] + b[1]*x[2] + b[2]*x[1] + b[3]*x[0];
        y[4] = b[0]*x[4] + b[1]*x[3] + b[2]*x[2] + b[3]*x[1] + b[4]*x[0];
        y[5] = b[0]*x[5] + b[1]*x[4] + b[2]*x[3] + b[3]*x[2] + b[4]*x[1] + b[5]*x[0]; 

        #pragma omp parallel for
        for (mwSize j = b_length-1; j < n_data_samples; j++) {
            y[j] = b[0]*x[j] + b[1]*x[j-1] + b[2]*x[j-2] + b[3]*x[j-3] +
                    + b[4]*x[j-4] + b[5]*x[j-5] + b[6]*x[j-6];
        }
    }else if (b_length == 8){    
    
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
    }else if (b_length == 9){    
    
        y[0] = b[0]*x[0];
        y[1] = b[0]*x[1] + b[1]*x[0];
        y[2] = b[0]*x[2] + b[1]*x[1] + b[2]*x[0];
        y[3] = b[0]*x[3] + b[1]*x[2] + b[2]*x[1] + b[3]*x[0];
        y[4] = b[0]*x[4] + b[1]*x[3] + b[2]*x[2] + b[3]*x[1] + b[4]*x[0];
        y[5] = b[0]*x[5] + b[1]*x[4] + b[2]*x[3] + b[3]*x[2] + b[4]*x[1] + b[5]*x[0]; 
        y[6] = b[0]*x[6] + b[1]*x[5] + b[2]*x[4] + b[3]*x[3] + b[4]*x[2] + b[5]*x[1] + b[6]*x[0];
        y[7] = b[0]*x[7] + b[1]*x[6] + b[2]*x[5] + b[3]*x[4] + b[4]*x[3] + b[5]*x[2] + b[6]*x[1] + b[7]*x[0];
        #pragma omp parallel for
        for (mwSize j = b_length-1; j < n_data_samples; j++) {
            y[j] = b[0]*x[j] + b[1]*x[j-1] + b[2]*x[j-2] + b[3]*x[j-3] +
                    + b[4]*x[j-4] + b[5]*x[j-5] + b[6]*x[j-6] + b[7]*x[j-7] + b[8]*x[j-8];
        }
    }else{        
        //TODO: I need the start bits here ...
        
        double *z = mxCalloc(b_length,sizeof(double));
        double Yi;
        
        for (mwSize j = 0; j < b_length-1; j++){
            y[j] = b[0]*x[j] + z[0];
            for (int i = 1; i < b_length-1; i++) {
               z[i - 1] = b[i]*x[j] + z[i];
            }            
        }
        
        mxFree(z);
        
        //This approach really slows things down ...
        #pragma omp parallel for
        for (mwSize j = b_length-1; j < n_data_samples; j++) {
            //TODO: This could be done with pointers as well 
            for (int i = 0; i < b_length; i++){
                y[j] += b[i]*x[j-i];
            }
        }
    }
}

//=======================================================================
//                              END OF FIR
//=======================================================================

//=======================================================================
//                            START OF FIR
//=======================================================================    
    
//Let's start with a naive approach for IIR portion
//-------------------------------------------------------------------
if (IIR_option == 0){
    if(a_length == 1){
        //Do nothing ...
    }else if(a_length == 2){    
        //1 back for everything ...
        for (mwSize j = a_length-1; j < n_data_samples; j++){
            y[j] -= a[1]*y[j-1];
        }  
    }else if(a_length == 3){    
        y[1] -= a[1]*y[0];
        for (mwSize j = a_length-1; j < n_data_samples; j++){
            y[j] -= a[1]*y[j-1] + a[2]*y[j-2];
        }  
    }else if(a_length == 4){
        y[1] -= a[1]*y[0];
        y[2] -= a[1]*y[1] + a[2]*y[0];
        for (mwSize j = a_length-1; j < n_data_samples; j++){
            y[j] -= a[1]*y[j-1] + a[2]*y[j-2] + a[3]*y[j-3];
        }      
    }else if(a_length == 5){
        y[1] -= a[1]*y[0];
        y[2] -= a[1]*y[1] + a[2]*y[0];
        y[3] -= a[1]*y[2] + a[2]*y[1] + a[3]*y[0];
        for (mwSize j = a_length-1; j < n_data_samples; j++){
            y[j] -= a[1]*y[j-1] + a[2]*y[j-2] + a[3]*y[j-3] + a[4]*y[j-4];
        }      
    }else if(a_length == 6){
        y[1] -= a[1]*y[0];
        y[2] -= a[1]*y[1] + a[2]*y[0];
        y[3] -= a[1]*y[2] + a[2]*y[1] + a[3]*y[0];
        y[4] -= a[1]*y[3] + a[2]*y[2] + a[3]*y[1] + a[4]*y[0];
        for (mwSize j = a_length-1; j < n_data_samples; j++){
            y[j] -= a[1]*y[j-1] + a[2]*y[j-2] + a[3]*y[j-3] + a[4]*y[j-4] + a[5]*y[j-5];
        }      
    }else if(a_length == 7){
        y[1] -= a[1]*y[0];
        y[2] -= a[1]*y[1] + a[2]*y[0];
        y[3] -= a[1]*y[2] + a[2]*y[1] + a[3]*y[0];
        y[4] -= a[1]*y[3] + a[2]*y[2] + a[3]*y[1] + a[4]*y[0];
        y[5] -= a[1]*y[4] + a[2]*y[3] + a[3]*y[2] + a[4]*y[1] + a[5]*y[0];
        for (mwSize j = a_length-1; j < n_data_samples; j++){
            y[j] -= a[1]*y[j-1] + a[2]*y[j-2] + a[3]*y[j-3] + a[4]*y[j-4] + a[5]*y[j-5] + a[6]*y[j-6];
        }      
    }else if(a_length == 8){
        y[1] -= a[1]*y[0];
        y[2] -= a[1]*y[1] + a[2]*y[0];
        y[3] -= a[1]*y[2] + a[2]*y[1] + a[3]*y[0];
        y[4] -= a[1]*y[3] + a[2]*y[2] + a[3]*y[1] + a[4]*y[0];
        y[5] -= a[1]*y[4] + a[2]*y[3] + a[3]*y[2] + a[4]*y[1] + a[5]*y[0];
        y[6] -= a[1]*y[5] + a[2]*y[4] + a[3]*y[3] + a[4]*y[2] + a[5]*y[1] + a[6]*y[0];
        for (mwSize j = a_length-1; j < n_data_samples; j++){
            y[j] -= a[1]*y[j-1] + a[2]*y[j-2] + a[3]*y[j-3] + a[4]*y[j-4] + a[5]*y[j-5] + a[6]*y[j-6] + a[7]*y[j-7];
        }      
    }else if(a_length == 9){
        y[1] -= a[1]*y[0];
        y[2] -= a[1]*y[1] + a[2]*y[0];
        y[3] -= a[1]*y[2] + a[2]*y[1] + a[3]*y[0];
        y[4] -= a[1]*y[3] + a[2]*y[2] + a[3]*y[1] + a[4]*y[0];
        y[5] -= a[1]*y[4] + a[2]*y[3] + a[3]*y[2] + a[4]*y[1] + a[5]*y[0];
        y[6] -= a[1]*y[5] + a[2]*y[4] + a[3]*y[3] + a[4]*y[2] + a[5]*y[1] + a[6]*y[0];
        y[7] -= a[1]*y[6] + a[2]*y[5] + a[3]*y[4] + a[4]*y[3] + a[5]*y[2] + a[6]*y[1] + a[7]*y[0];
        for (mwSize j = a_length-1; j < n_data_samples; j++){
            y[j] -= a[1]*y[j-1] + a[2]*y[j-2] + a[3]*y[j-3] + a[4]*y[j-4] + a[5]*y[j-5] + a[6]*y[j-6] + a[7]*y[j-7] + a[8]*y[j-8];
        }      
    }else{
        //y[0] = b[0]*x[0] + z[0]
        //z[0] = b[1]*x[0] + z[1] - a[1]*y[0]
        //z[1] = b[2]*x[0] + z[2] - a[2]*y[0]
        //z[2] = b[3]*x[0] + z[3] - a[3]*y[0]
        
        //y[1] = b[1]*x[1] + z[0]
        //z[0] = b[1]*x[1] + z[1] - a[1]*y[1]
     	//z[1] = b[2]*x[1] + z[2] - a[2]*y[1]
        //z[2] = b[3]*x[1] + z[3] - a[3]*y[1]
        
        //y[2] = b[1]*x[2] + z[0]
        //z[0] = b[1]*x[2] + z[1] - a[1]*y[2]
     	//z[1] = b[2]*x[2] + z[2] - a[2]*y[2]
        //z[2] = b[3]*x[2] + z[3] - a[3]*y[2]
        
        //y[3] = b[1]*x[3] + z[0]
        //z[0] = b[1]*x[3] + z[1] - a[1]*y[3]
     	//z[1] = b[2]*x[3] + z[2] - a[2]*y[3]
        //z[2] = b[3]*x[3] + z[3] - a[3]*y[3]
        
        //y[3] = b[1]*x[3] + z[0]
        //z[0] = b[1]*x[3] + z[1] - a[1]*y[3]
     	//z[1] = b[2]*x[3] + z[2] - a[2]*y[3]
        //z[2] = b[3]*x[3] + z[3] - a[3]*y[3]
        
        
        //y[1] = X - a1*y0
        
        //y[2] = X - a1*y1 - a2*y0
        
        //y[3] = X - a1*y2 - a2*y1 - a3*y0
        
        double *z = mxCalloc(b_length,sizeof(double));
        for (mwSize j = 0; j < a_length-1; j++){
            y[j] += z[0];
            for (int i = 1; i < a_length-1; i++) {
               z[i - 1] = z[i] - a[i] * y[j];
            }            
            z[a_length-2] = -a[a_length-1] * y[j];
        }
        mxFree(z);
        for (mwSize j = a_length-1; j < n_data_samples; j++){
            for (int i = 1; i < a_length; i++){
                y[j] -= a[i]*y[j-i];
            }
        }     
    }
}

 
//Pointers for IIR
//---------------------------------------------------------
if (IIR_option == 1){
    if(a_length == 1){
        //Do nothing ...
    }else if(a_length == 2){    
        //1 back for everything ...
        double *y0 = &y[1];
        double *y1 = y;
        while (y0 < y + n_data_samples){
            *y0 -= a[1]*(*y1);
            y0++;
            y1++;
        }  
    }else if(a_length == 3){ 
       	double *y0 = &y[2];
        double *y1 = &y[1];
        double *y2 = &y[0];
        
        y[1] -= a[1]*y[0];
        while (y0 < y + n_data_samples){
            *y0 -= a[1]*(*y1) + a[2]*(*y2);
          	y0++;
            y1++;
            y2++;
        }  
    }else if(a_length == 4){
        double *y0 = &y[3];
        double *y1 = &y[2];
        double *y2 = &y[1];
        double *y3 = &y[0];
        y[1] -= a[1]*y[0];
        y[2] -= a[1]*y[1] + a[2]*y[0];
        while (y0 < y + n_data_samples){
            *y0 -= a[1]*(*y1) + a[2]*(*y2) + a[3]*(*y3);
            y0++;
            y1++;
            y2++;
            y3++;
        }      
    }else if(a_length == 5){
        double *y0 = &y[4];
        double *y1 = &y[3];
        double *y2 = &y[2];
        double *y3 = &y[1];
        double *y4 = &y[0];
        y[1] -= a[1]*y[0];
        y[2] -= a[1]*y[1] + a[2]*y[0];
        y[3] -= a[1]*y[2] + a[2]*y[1] + a[3]*y[0];
        while (y0 < y + n_data_samples){
            *y0 -= a[1]*(*y1) + a[2]*(*y2) + a[3]*(*y3) + a[4]*(*y4);
            y0++;
            y1++;
            y2++;
            y3++;
            y4++;
        }      
    }else if(a_length == 6){
        double *y0 = &y[5];
        double *y1 = &y[4];
        double *y2 = &y[3];
        double *y3 = &y[2];
        double *y4 = &y[1];
        double *y5 = &y[0];
        y[1] -= a[1]*y[0];
        y[2] -= a[1]*y[1] + a[2]*y[0];
        y[3] -= a[1]*y[2] + a[2]*y[1] + a[3]*y[0];
        y[4] -= a[1]*y[3] + a[2]*y[2] + a[3]*y[1] + a[4]*y[0];
        while (y0 < y + n_data_samples){
            *y0 -= a[1]*(*y1) + a[2]*(*y2) + a[3]*(*y3) + a[4]*(*y4) + a[5]*(*y5);
          	y0++;
            y1++;
            y2++;
            y3++;
            y4++;
            y5++;
        }      
    }else if(a_length == 7){
        double *y0 = &y[6];
        double *y1 = &y[5];
        double *y2 = &y[4];
        double *y3 = &y[3];
        double *y4 = &y[2];
        double *y5 = &y[1];
        double *y6 = &y[0];
        y[1] -= a[1]*y[0];
        y[2] -= a[1]*y[1] + a[2]*y[0];
        y[3] -= a[1]*y[2] + a[2]*y[1] + a[3]*y[0];
        y[4] -= a[1]*y[3] + a[2]*y[2] + a[3]*y[1] + a[4]*y[0];
        y[5] -= a[1]*y[4] + a[2]*y[3] + a[3]*y[2] + a[4]*y[1] + a[5]*y[0];
        while (y0 < y + n_data_samples){
            *y0 -= a[1]*(*y1) + a[2]*(*y2) + a[3]*(*y3) + a[4]*(*y4) + a[5]*(*y5) + a[6]*(*y6);
          	y0++;
            y1++;
            y2++;
            y3++;
            y4++;
            y5++;
            y6++;
        }
    }else if(a_length == 8){
        double *y0 = &y[7];
        double *y1 = &y[6];
        double *y2 = &y[5];
        double *y3 = &y[4];
        double *y4 = &y[3];
        double *y5 = &y[2];
        double *y6 = &y[1];
        double *y7 = &y[0];
        y[1] -= a[1]*y[0];
        y[2] -= a[1]*y[1] + a[2]*y[0];
        y[3] -= a[1]*y[2] + a[2]*y[1] + a[3]*y[0];
        y[4] -= a[1]*y[3] + a[2]*y[2] + a[3]*y[1] + a[4]*y[0];
        y[5] -= a[1]*y[4] + a[2]*y[3] + a[3]*y[2] + a[4]*y[1] + a[5]*y[0];
        y[6] -= a[1]*y[5] + a[2]*y[4] + a[3]*y[3] + a[4]*y[2] + a[5]*y[1] + a[6]*y[0];
        while (y0 < y + n_data_samples){
            *y0 -= a[1]*(*y1) + a[2]*(*y2) + a[3]*(*y3) + a[4]*(*y4) + a[5]*(*y5) + a[6]*(*y6) + a[7]*(*y7);
          	y0++;
            y1++;
            y2++;
            y3++;
            y4++;
            y5++;
            y6++;
            y7++;
        }
    }else if(a_length == 9){
        double *y0 = &y[8];
        double *y1 = &y[7];
        double *y2 = &y[6];
        double *y3 = &y[5];
        double *y4 = &y[4];
        double *y5 = &y[3];
        double *y6 = &y[2];
        double *y7 = &y[1];
        double *y8 = &y[0];
        y[1] -= a[1]*y[0];
        y[2] -= a[1]*y[1] + a[2]*y[0];
        y[3] -= a[1]*y[2] + a[2]*y[1] + a[3]*y[0];
        y[4] -= a[1]*y[3] + a[2]*y[2] + a[3]*y[1] + a[4]*y[0];
        y[5] -= a[1]*y[4] + a[2]*y[3] + a[3]*y[2] + a[4]*y[1] + a[5]*y[0];
        y[6] -= a[1]*y[5] + a[2]*y[4] + a[3]*y[3] + a[4]*y[2] + a[5]*y[1] + a[6]*y[0];
        y[7] -= a[1]*y[6] + a[2]*y[5] + a[3]*y[4] + a[4]*y[3] + a[5]*y[2] + a[6]*y[1] + a[7]*y[0];
        while (y0 < y + n_data_samples){
            *y0 -= a[1]*(*y1) + a[2]*(*y2) + a[3]*(*y3) + a[4]*(*y4) + a[5]*(*y5) + a[6]*(*y6) + a[7]*(*y7) + a[8]*(*y8);
          	y0++;
            y1++;
            y2++;
            y3++;
            y4++;
            y5++;
            y6++;
            y7++;
            y8++;
        }      
    }else{
        double *z = mxCalloc(b_length,sizeof(double));
        for (mwSize j = 0; j < a_length-1; j++){
            y[j] += z[0];
            for (int i = 1; i < a_length-1; i++) {
               z[i - 1] = z[i] - a[i] * y[j];
            }            
            z[a_length-2] = -a[a_length-1] * y[j];
        }
        mxFree(z);
        for (mwSize j = b_length-1; j < n_data_samples; j++){
            for (int i = 1; i < b_length; i++){
                y[j] -= a[i]*y[j-i];
            }
        }     
    }
}
    
if (IIR_option == 2){
    if(a_length == 1){
        //Do nothing ...
    }else if(a_length == 2){    
        //1 back for everything ...
        double z0 = 0;
        double *y0 = y;
        while (y0 < y + n_data_samples){
            *y0 += z0;
            z0 = a[1]*(*y0);
            y0++;
        }   
    }else if(a_length == 3){ 
        double z0 = 0;
        double z1 = 0;
        double *y0 = y;
        while (y0 < y + n_data_samples){
            *y0 += z0;
            z0 = z1 - a[1]*(*y0);
            z1 = -a[2]*(*y0);
            y0++;
        }  
    }else if(a_length == 4){
        double z0 = 0;
        double z1 = 0;
        double z2 = 0;
        double *y0 = y;
        while (y0 < y + n_data_samples){
            *y0 += z0;
            z0 = z1 - a[1]*(*y0);
            z1 = z2 - a[2]*(*y0);
            z2 = - a[3]*(*y0);
            y0++;
        }     
    }else if(a_length == 5){
        double z0 = 0;
        double z1 = 0;
        double z2 = 0;
        double z3 = 0;
        double *y0 = y;
        while (y0 < y + n_data_samples){
            *y0 += z0;
            z0 = z1 - a[1]*(*y0);
            z1 = z2 - a[2]*(*y0);
            z2 = z3 - a[3]*(*y0);
            z3 = - a[4]*(*y0);
            y0++;
        }      
    }else if(a_length == 6){
        double z0 = 0;
        double z1 = 0;
        double z2 = 0;
        double z3 = 0;
        double z4 = 0;
        double *y0 = y;
        while (y0 < y + n_data_samples){
            *y0 += z0;
            z0 = z1 - a[1]*(*y0);
            z1 = z2 - a[2]*(*y0);
            z2 = z3 - a[3]*(*y0);
            z3 = z4 - a[4]*(*y0);
            z4 = - a[5]*(*y0);
            y0++;
        }      
    }else if(a_length == 7){
        double z0 = 0;
        double z1 = 0;
        double z2 = 0;
        double z3 = 0;
        double z4 = 0;
        double z5 = 0;
        double *y0 = y;
        while (y0 < y + n_data_samples){
            *y0 += z0;
            z0 = z1 - a[1]*(*y0);
            z1 = z2 - a[2]*(*y0);
            z2 = z3 - a[3]*(*y0);
            z3 = z4 - a[4]*(*y0);
            z4 = z5 - a[5]*(*y0);
            z5 = - a[6]*(*y0);
            y0++;
        }  
    }else if(a_length == 8){
        double z0 = 0;
        double z1 = 0;
        double z2 = 0;
        double z3 = 0;
        double z4 = 0;
        double z5 = 0;
        double z6 = 0;
        double *y0 = y;
        while (y0 < y + n_data_samples){
            *y0 += z0;
            z0 = z1 - a[1]*(*y0);
            z1 = z2 - a[2]*(*y0);
            z2 = z3 - a[3]*(*y0);
            z3 = z4 - a[4]*(*y0);
            z4 = z5 - a[5]*(*y0);
            z5 = z6 - a[6]*(*y0);
            z6 = - a[7]*(*y0);
            y0++;
        }  
    }else if(a_length == 9){
        double z0 = 0;
        double z1 = 0;
        double z2 = 0;
        double z3 = 0;
        double z4 = 0;
        double z5 = 0;
        double z6 = 0;
        double z7 = 0;
        double *y0 = y;
        while (y0 < y + n_data_samples){
            *y0 += z0;
            z0 = z1 - a[1]*(*y0);
            z1 = z2 - a[2]*(*y0);
            z2 = z3 - a[3]*(*y0);
            z3 = z4 - a[4]*(*y0);
            z4 = z5 - a[5]*(*y0);
            z5 = z6 - a[6]*(*y0);
            z6 = z7 - a[7]*(*y0);
            z7 = - a[8]*(*y0);
            y0++;
        }     
    }else{
        double *z = mxCalloc(b_length,sizeof(double));
        for (mwSize j = 0; j < a_length-1; j++){
            y[j] += z[0];
            for (int i = 1; i < a_length-1; i++) {
               z[i - 1] = z[i] - a[i] * y[j];
            }            
            z[a_length-2] = -a[a_length-1] * y[j];
        }
        double yi;
        int i;
        int order = a_length-1;
        for (mwSize j = a_length-1; j < n_data_samples; j++){            
            yi += z[0];
            for (i = 1; i < order; i++) {
                z[i - 1] = z[i] - a[i] * yi;
            }
                z[order - 1] = - a[order] * yi;
        
            y[j++] = yi;
        }     
    }
}


//Now onto the IIR part ...
//-------------------------------------------------------------

//Redefine a1 as -a1 to avoid - signs in equations    

//Let's start for a 4th order filter
    
//y0 = a1*yn1 + a2*yn2 + a3*yn3 + F0   //F0 is IIR part
//y1 = a1*y0  + a2*yn1 + a3*yn2 + F1
//y2 = a1*y1  + a2*y0  + a3*yn1 + F2


//Introduce notation
//A1 = [a1 a2 a3][1 0 0] which get multiplied by X = [yn1 yn2 yn3][F0 F1 F2]
    
//y0 = [a1 a2 a3][1 0 0]*X = A1*X  
//y1 = a1*A1 + [0 a2 a3][0 1 0] = [a1*a1 a1*a2 a1*a3][a1 0 0] + [0 a2 a3][0 1 0]
//   => [a1*a1, a1*a2+a2, a1*a3+a3][a1 1 0] = A2*X
//y2 = a1*A2 + a2*A1 + [0 0 a3][0 0 1] 
//   => [a1^3,a1^2*a2+a1*a2,

    
} 
    

    
