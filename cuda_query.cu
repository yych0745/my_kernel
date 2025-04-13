#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("Device: %s\n", prop.name);
    printf("Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Shared Memory per Multiprocessor: %zu KB\n", 
           prop.sharedMemPerMultiprocessor / 1024);
    printf("Maximum Threads per Block: %d\n", prop.maxThreadsPerBlock);
    
    return 0;
}