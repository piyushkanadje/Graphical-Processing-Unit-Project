/******************************
Multi GPU Multiplication is done by splitting the comutation into 4 parts as follows:
A * B = C   
| A1 |      |    |    |     C1 | C2
------  *   | B1 | B2 | =   -------
| A2 |      |    |    |     C3 | C4 
  
A1 * B1 = C1
A1 * B2 = c2
A2 * B1 = C3
A2 * B2 = C4
These 4 individual computations take place simultaneously on different GPUs.
******************************/

/******************************/
//Required library includes
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "kernel.cu"
#include "support.h"

/******************************/
//main function
int main(int argc,char *argv[]){


    // Setup Timer for checking execution times
    Timer timer;
    //cudaError_t cuda_ret;

    /******************************/
    startTime(&timer); //"Setting up the problem" starts here

    //create cuda streams
    const int num_streams = 4;
    cudaStream_t streams[num_streams];
    //size of matrix
    //currently this code only works on square matrices
    int N;
    if (argc == 1) {
        N = 1000;
    } else if (argc == 2) {
       N = atoi(argv[1]);
    }
    else {
        printf("\nInvalid input parameters!"
      "\n    Usage: ./multiply        # All matrices are 1000 x 1000"
      "\n    Usage: ./multiply <m>    # All matrices are m x m"
      "\n");
        exit(0);
    }

    /******************************/
    //declare matrices and aother values
    double *h_A,*h_B,*h_C;
    int id,i,j;
    int ndev; //number of GPU devices available
    double r = 0.5; //block division
    double r_1 = (1-r);
    double *h_A1,*h_A2,*h_B1,*h_B2,*h_C1,*h_C2,*h_C3,*h_C4; //host matrices
    double *d_A1,*d_A1_2,*d_A2,*d_A2_2,*d_B1,*d_B1_2,*d_B2,*d_B2_2; //device matrix blocks
    double *d_C1,*d_C2,*d_C3,*d_C4; //device resultant matix

    // Get number of GPU Devices
    cudaGetDeviceCount(&ndev);
    if(ndev==0){
        printf(" GPU DEVICES ARE UNAVAILABLE\n\n");
        exit(-1);
            
    }else{
        printf("\nAvailable Number of GPU: %d\n",ndev);
    }
        
    // Allocate Host variables
    //  fill matrices A & B with random values
    //  fill matrix C with zeroes
    cudaMallocHost(&h_A,N*N*sizeof(double));

    for (unsigned int i=0; i < N*N; i++) { 
        h_A[i] = (rand()%100)/100.00; 
    }

    cudaMallocHost(&h_B,N*N*sizeof(double));
    for (unsigned int i=0; i < N*N; i++) { 
        h_B[i] = (rand()%100)/100.00; 
    }

    cudaMallocHost(&h_C,N*N*sizeof(double));
    for (unsigned int i=0; i < N*N; i++) { 
        h_C[i] = 0; 
    }
    stopTime(&timer); 
    printf("\n Time For Setting Up %f s\n", elapsedTime(timer));
    printf("    A: %d x %d, %d elements\n    B: %d x %d, %d elements\n    C: %d x %d, %d elements\n", 
        N, N, N*N, N, N, N*N, N, N, N*N);
    printf("\nWidth Of Block 1: %d\n",(int)(N*r));
    printf("Width Of Block 2: %d\n",(int)(N*r_1));
    
        
/******************************/
//Allocation Of Kernel 2

    startTime(&timer); //timer starts here
    id=0;
    cudaSetDevice((int)(id%ndev));
    cudaStreamCreateWithFlags(&streams[id],cudaStreamNonBlocking);
    
    cudaMallocManaged(&h_A1,(int)(N*N*r*sizeof(double)));
    cudaMallocManaged(&h_B1,(int)(N*N*r*sizeof(double)));
    cudaMallocManaged(&h_C1,(int)(N*N*r*r*sizeof(double)));
    
    for(int i=0;i<(int)(N*r);i++){
        for(int j=0;j<N;j++){
            h_A1[i*N+j] =  h_A[i*N+j];
        }
    }
    for(int i=0;i<N;i++){
        for(int j=0;j<(N*r);j++){
            h_B1[i*(int)(N*r)+j] =  h_B[i*N+j];
        }
    }

    cudaMalloc((void**)&d_A1,(int)(N*N*r*sizeof(double)));
    cudaMalloc((void**)&d_B1,(int)(N*N*r*sizeof(double)));
    cudaMalloc((void**)&d_C1,(int)(N*N*r*r*sizeof(double)));

    stopTime(&timer);
    printf("\n Allocating the parts of data to Kernel 1....%f s\n", elapsedTime(timer));  //timer ends here


 /******************************/
 //Allocation For Kernel 2...

    startTime(&timer); //timer starts here
    id=1;
    cudaSetDevice((int)(id%ndev));
    cudaStreamCreateWithFlags(&streams[id],cudaStreamNonBlocking);
    
    cudaMallocHost(&h_B2,(int)(N*N*r_1*sizeof(double)));
    cudaMallocHost(&h_C2,(int)(N*N*r*r_1*sizeof(double)));

    
    for(int i=0;i<N;i++){
        for(int j=0;j<(N*r_1);j++){
            h_B2[i*(int)(N*r_1)+j] =  h_B[i*N+(int)(N*r)+j];
        }
    }
     
    cudaMalloc((void**)&d_A1_2,(int)(N*N*r*sizeof(double)));
    cudaMalloc((void**)&d_B2,(int)(N*N*r_1*sizeof(double)));
    cudaMalloc((void**)&d_C2,(int)(N*N*r*r_1*sizeof(double)));
 stopTime(&timer);
    printf("Allocating the parts of data to Kernel 2....%f s\n", elapsedTime(timer));  //timer ends here
        
    /******************************/
    // Allocation for Kernel 3......

    startTime(&timer); //timer starts here
    id=2;
    cudaSetDevice(id%ndev);
    cudaStreamCreateWithFlags(&streams[id],cudaStreamNonBlocking);
    
    cudaMallocHost(&h_A2,(int)(N*N*r_1*sizeof(double)));
    cudaMallocHost(&h_C3,(int)(N*N*r_1*r*sizeof(double)));
    
    for(int i=0;i<(int)(N*r_1);i++){
        for(int j=0;j<N;j++){
            h_A2[i*N+j] =  h_A[(i+(int)(N*r))*N+j];
        }
    }
    
    cudaMalloc((void**)&d_A2,(int)(N*N*r_1*sizeof(double)));
    cudaMalloc((void**)&d_B1_2,(int)(N*N*r*sizeof(double)));
    cudaMalloc((void**)&d_C3,(int)(N*N*r*r_1*sizeof(double))); 
 stopTime(&timer);
    printf("Allocating the parts of data to Kernel 3....%f s\n", elapsedTime(timer));  //timer ends here
        
    /******************************/
    // Allocation for kernel 3......
    startTime(&timer); //timer starts here
    id=3;
    cudaSetDevice(id%ndev);
    //cudaStreamCreate(&streams[id]);
    cudaStreamCreateWithFlags(&streams[id],cudaStreamNonBlocking);

    cudaMallocHost(&h_C4,(int)(N*N*r_1*r_1*sizeof(double)));
    
    cudaMalloc((void**)&d_A2_2,(int)(N*N*r_1*sizeof(double)));
    cudaMalloc((void**)&d_B2_2,(int)(N*N*r_1*sizeof(double)));
    cudaMalloc((void**)&d_C4,(int)(N*N*r_1*r_1*sizeof(double)));
 stopTime(&timer);
    printf("Allocating the parts of data to Kernel 4....%f s\n", elapsedTime(timer));  //timer ends here
        

    /******************************/
    //Grid and block size initialization
    int grid_width = 1+N/32;
    dim3 dimGrid(grid_width,grid_width,1);
    dim3 dimBlock(32,32,1);

    // kernel 1 data copy and execution
    startTime(&timer); //timer starts here
    
    id=0;
    cudaSetDevice(id%ndev);
        
    cudaMemcpyAsync(d_A1,h_A1,(int)(N*N*r*sizeof(double)),cudaMemcpyHostToDevice,streams[id]);
    cudaMemcpyAsync(d_B1,h_B1,(int)(N*N*r*sizeof(double)),cudaMemcpyHostToDevice,streams[id]);

    stopTime(&timer);
    printf("\nCopying data from host to device %d...%f s\n", id+1, elapsedTime(timer));

    startTime(&timer); //timer starts here      
    kernelC1 <<< dimGrid,dimBlock,0,streams[id]>>>(d_A1,d_B1,d_C1,N,r);
     stopTime(&timer);
    printf("Execution Of Kernel%d...%f s\n", id+1, elapsedTime(timer));

    
    startTime(&timer); //timer starts here

    // kernel 2 data copy and execution
    id=1;
    cudaSetDevice(id%ndev);

    cudaMemcpyAsync(d_A1_2,h_A1,(int)(N*N*r*sizeof(double)),cudaMemcpyHostToDevice,streams[id]);
    cudaMemcpyAsync(d_B2,h_B2,(int)(N*N*r_1*sizeof(double)),cudaMemcpyHostToDevice,streams[id]);

    stopTime(&timer);
    printf("\nCopying data from host to device %d...%f s\n", id+1, elapsedTime(timer));
    
    startTime(&timer); //timer starts here
    kernelC2 <<< dimGrid,dimBlock,0,streams[id]>>>(d_A1_2,d_B2,d_C2,N,r);
     stopTime(&timer);
    printf("Execution Of Kernel   %d...%f s\n", id+1, elapsedTime(timer));

    
    startTime(&timer); //timer starts here

    //kernel 3 data copy and execution
    id=2;
    cudaSetDevice(id%ndev);
    
    cudaMemcpyAsync(d_A2,h_A2,(int)(N*N*r_1*sizeof(double)),cudaMemcpyHostToDevice,streams[id]);
    cudaMemcpyAsync(d_B1_2,h_B1,(int)(N*N*r*sizeof(double)),cudaMemcpyHostToDevice,streams[id]);

    stopTime(&timer);
    printf("\nCopying data from host to device %d...%f s\n", id+1, elapsedTime(timer));

    startTime(&timer); //timer starts here
    kernelC3 <<< dimGrid,dimBlock,0,streams[id]>>>(d_A2,d_B1_2,d_C3,N,r);
     stopTime(&timer);
    printf("Execution Of Kernel %d...%f s\n", id+1, elapsedTime(timer));
    

    startTime(&timer); //timer starts here

    //kernel 4 data copy and execution
    id=3;
    cudaSetDevice(id%ndev);
    
    cudaMemcpyAsync(d_A2_2,h_A2,(int)(N*N*r_1*sizeof(double)),cudaMemcpyHostToDevice,streams[id]);
    cudaMemcpyAsync(d_B2_2,h_B2,(int)(N*N*r_1*sizeof(double)),cudaMemcpyHostToDevice,streams[id]);

    stopTime(&timer);
    printf("\nCopying data from host to device %d...%f s\n", id+1, elapsedTime(timer));

    startTime(&timer); //timer starts here
    kernelC4 <<< dimGrid,dimBlock,0,streams[id]>>>(d_A2_2,d_B2_2,d_C4,N,r);
     stopTime(&timer);
    printf("Execution Of Kernel %d...%f s\n", id+1, elapsedTime(timer));


    /******************************/
    // copy data back to host
    startTime(&timer); //timer starts here

    cudaMemcpyAsync(h_C1,d_C1,(int)(N*N*r*r*sizeof(double)),cudaMemcpyDeviceToHost,streams[id]);
   
    cudaMemcpyAsync(h_C2,d_C2,(int)(N*N*r*r_1*sizeof(double)),cudaMemcpyDeviceToHost,streams[id]);
   
    cudaMemcpyAsync(h_C3,d_C3,(int)(N*N*r*r_1*sizeof(double)),cudaMemcpyDeviceToHost,streams[id]);
   
    cudaMemcpyAsync(h_C4,d_C4,(int)(N*N*r_1*r_1*sizeof(double)),cudaMemcpyDeviceToHost,streams[id]);
 stopTime(&timer);
    printf("\nCopying data from devices to host...%f s\n", elapsedTime(timer));
   
    
    //Synchronize in order to process the results of every kernel and stream

    id=0;
    cudaSetDevice(id%ndev);
    cudaStreamSynchronize(streams[id]);
    
    id=1;
    cudaSetDevice(id%ndev);
    cudaStreamSynchronize(streams[id]);

    id=2;
    cudaSetDevice(id%ndev);
    cudaStreamSynchronize(streams[id]);

    id=3;
    cudaSetDevice(id%ndev);
    cudaStreamSynchronize(streams[id]);


    /******************************/
    //build the final Matrix from blocks

    printf("Building The Final Matrix from Block");
    startTime(&timer); //timer starts here

    for(i=0;i<(int)N*r;i++){
        for(j=0;j<(int)N*r;j++){
              h_C[i*N+j] = h_C1[i*(int)(N*r)+j];
        }
    }
  
    
    
    for(i=0;i<(int)N*r;i++){
        for(j=0;j<(int)(N*r_1);j++){
             h_C[i*N+j+(int)(N*r)] = h_C2[i*(int)(N*r_1)+j];
        }
    }
  
    
    for(i=0;i<(int)(N*r_1);i++){
        for(j=0;j<(int)(N*r);j++){
             h_C[(i+(int)(N*r))*N+j] = h_C3[i*(int)(N*r)+j];     
        }
    }
 
    
  
    for(i=0;i<(int)(N*r_1);i++){
        for(j=0;j<(int)(N*r_1);j++){
            h_C[(i+(int)(N*r))*N+j+(int)(N*r)] = h_C4[i*(int)(N*r_1)+j];
        } 
    } 
     stopTime(&timer);
    printf("\nBuilding final matrix from blocks...%f s\n", elapsedTime(timer));

    /******************************/
    // validate results by calculating on cpu
    fflush(stdout);
    startTime(&timer);
    resultVerify(h_A, h_B, h_C, N);
    stopTime(&timer); 
    printf("\nVerifying results...%f s\n", elapsedTime(timer));
    
    /******************************/
    // Free Memory
    startTime(&timer);

    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFreeHost(h_A1);
    cudaFreeHost(h_A2);
    cudaFreeHost(h_B1);
    cudaFreeHost(h_B2);
    cudaFreeHost(h_C1);
    cudaFreeHost(h_C2);
    cudaFreeHost(h_C3);
    cudaFreeHost(h_C4);
    
    id=0;
    cudaSetDevice(id%ndev);
    cudaFree(d_A1);
    cudaFree(d_B1);
    cudaFree(d_C1);
    
    id=1;
    cudaSetDevice(id%ndev);
    cudaFree(d_A1_2);
    cudaFree(d_B2);
    cudaFree(d_C2);
   
    id=2;
    cudaSetDevice(id%ndev);
    cudaFree(d_A2);
    cudaFree(d_B1_2);
    cudaFree(d_C3);

    id=3;
    cudaSetDevice(id%ndev);
    cudaFree(d_A2_2);
    cudaFree(d_B2_2);
    cudaFree(d_C4);
   
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
    cudaStreamDestroy(streams[2]);
    cudaStreamDestroy(streams[3]);

    stopTime(&timer); 
    printf("\nFreeing Host and Device Memory...%f s\n\n", elapsedTime(timer));
    
    // Exit
    return(0);
}
