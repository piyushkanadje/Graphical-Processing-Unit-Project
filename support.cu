

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "support.h"

void  resultVerify(double *A, double *B, double *C, int N) {

		int i,j,k;
		unsigned int count = 0;
		double res; 
		const float relativeTolerance = 1e-6;
		for(i=0;i<N;i++){
				for(j=0;j<N;j++){
						res=0;
						for(k=0;k<N;k++){
								res+=A[i*N+k]*B[k*N+j];
						}
						count++;
					 //printf("%8.3f ",res);
					double relativeError = (res - C[i*N+j])/res;
					 if(relativeError > relativeTolerance || relativeError < -relativeTolerance){
								printf("\n\nTEST FAILED %u\n",count);
								printf("Expected value:%f \nComputed value:%f\n\n",res,C[i*N+j]);
								exit(1);
					 }
				}
				//printf("\n");
		}
	printf("\nTEST PASSED %u\n", count);

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

