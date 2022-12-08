#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "support.h"

void verify(double *A, double *B, double *C,
   int N) {

  const float relativeTolerance = 1e-6;
  unsigned int count = 0;
  double res ;
  int i,j,k;
  for( i = 0; i < N; i++) {
    for( j = 0; j < N; j++) {
      float sum = 0;
       for( k =0; k < N; k++){
        res +=A[i*N+k]*B[j*N+k];
       }
       count++;
       double relativeError =(res - C[i*N+j])/res;
       if( relativeError > relativeTolerance || relativeError  < -relativeTolerance){
        printf("TEst Failesd", count);
        printf("Expected Valur : %f\n  ComputedValue:%f \n\n", res, C[i*N+j]);
        exit(1);
       }
      }
    }
  
  printf("TEST PASSED %u\n\n", count);

}

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}
