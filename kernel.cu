#include <stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include <time.h>


//general kernel(not used)
__global__ void matrix_multiplication(double *A,double *B,double *C,int width){
    int idy = blockIdx.y*blockDim.y+threadIdx.y;
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    
    int step;
    double prod_val = 0;
    if((idy>=width)||((idx>=width))) return;
    
    for(step=0;step<width;step++){
        prod_val += A[idy*width+step] * B[step*width+idx];
    }
    
    C[idy*width+idx] = prod_val;
}

// Kernel for the computation of C1 portion
__global__ void kernelC1(double *A,double *B,double *C,int width, double r){
    int idy = blockIdx.y*blockDim.y+threadIdx.y;
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int step;
    double prod_val = 0;
    
    if((idy>=(int)(width*r))||(idx>=(int)(width*r))) return;
    
    for(step=0;step<width;step++){
        prod_val += A[idy*width+step] * B[step*(int)(width*r)+idx];
    }
    
    C[idy*(int)(width*r)+idx] = prod_val;
}

// Kernel for the computation of C2 portion
__global__ void kernelC2(double *A,double *B,double *C,int width, double r){
    int idy = blockIdx.y*blockDim.y+threadIdx.y;
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int step;
    double prod_val = 0;
    
    if((idy>=(int)(width*r))||(idx>=(int)(width*(1-r)))) return;
    
    for(step=0;step<width;step++){
        prod_val += A[idy*width+step] * B[step*(int)(width*(1-r))+idx];
    }
    
    C[idy*(int)(width*(1-r))+idx] = prod_val;
}


// Kernel for the computation of C3 portion
__global__ void kernelC3(double *A,double *B,double *C,int width, double r){
    int idy = blockIdx.y*blockDim.y+threadIdx.y;
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    
    int step;
    double prod_val = 0;
    if((idy>=(int)(width*(1-r)))||(idx>=(int)(width*r))) return;
    
    for(step=0;step<width;step++){
        prod_val += A[idy*width+step] * B[step*(int)(width*r)+idx];
    }
    
    
    C[idy*(int)(width*r)+idx] = prod_val;
}

// // Kernel for the computation of C4 portion
__global__ void kernelC4(double *A,double *B,double *C,int width, double r){
    int idy = blockIdx.y*blockDim.y+threadIdx.y;
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    
    int step;
    double prod_val = 0;
    if((idy>=(int)(width*(1-r)))||(idx>=(int)(width*(1-r)))) return;
    
    for(step=0;step<width;step++){
        prod_val += A[idy*width+step] * B[step*(int)(width*(1-r))+idx];
    }
    C[idy*(int)(width*(1-r))+idx] = prod_val;
}