#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "hawkZip_compressor.h"
int testing = 1; //0 for release mode, 1 for testing mode

void hawkZip_compress(float* oriData, unsigned char* cmpData, size_t nbEle, size_t* cmpSize, float errorBound)
{
    // Variables used in compression kernel
    int blockNum = ((nbEle + NUM_THREADS - 1) / NUM_THREADS  + 31) / 32 * NUM_THREADS;
    int* absQuant = (int*)malloc(sizeof(int)*nbEle);
    unsigned int* signFlag = (unsigned int*)malloc(sizeof(unsigned int)*blockNum);
    int* fixedRate = (int*)malloc(sizeof(int)*blockNum);
    unsigned int* threadOfs = (unsigned int*)malloc(sizeof(unsigned int)*NUM_THREADS);
    memset(cmpData, 0, sizeof(float)*nbEle);
    double timerCMP_start, timerCMP_end;

    // Compression kernel computation and time measurement.
    timerCMP_start = omp_get_wtime();
    hawkZip_compress_kernel(oriData, cmpData, absQuant, signFlag, fixedRate, threadOfs, nbEle, cmpSize, errorBound);
    timerCMP_end = omp_get_wtime();
    printf("hawkZip   compression ratio:      %f\n", (float)(sizeof(float)*nbEle) / (float)(sizeof(unsigned char)*(*cmpSize)));
    printf("hawkZip   compression throughput: %f GB/s\n", (nbEle*sizeof(float)/1024.0/1024.0/1024.0)/(timerCMP_end-timerCMP_start));

    // Reallocate memory consumption of cmpData.
    cmpData = (unsigned char*)realloc(cmpData, sizeof(unsigned char)*(*cmpSize));

    free(absQuant);
    free(signFlag);
    free(fixedRate);
    free(threadOfs);
}

void hawkZip_decompress(float* decData, unsigned char* cmpData, size_t nbEle, float errorBound)
{
    // Varaibles used in decompression kernel
    int blockNum = ((nbEle + NUM_THREADS - 1) / NUM_THREADS  + 31) / 32 * NUM_THREADS;
    int* absQuant = (int*)malloc(sizeof(int)*nbEle);
    memset(absQuant, 0, sizeof(int)*nbEle);
    int* fixedRate = (int*)malloc(sizeof(int)*blockNum);
    unsigned int* threadOfs = (unsigned int*)malloc(sizeof(unsigned int)*NUM_THREADS);
    memset(decData, 0, sizeof(float)*nbEle);
    double timerDEC_start, timerDEC_end;

    // Decompression kernel computation and time measurement.
    timerDEC_start = omp_get_wtime();
    hawkZip_decompress_kernel(decData, cmpData, absQuant, fixedRate, threadOfs, nbEle, errorBound);
    timerDEC_end = omp_get_wtime();
    printf("hawkZip decompression throughput: %f GB/s\n", (nbEle*sizeof(float)/1024.0/1024.0/1024.0)/(timerDEC_end-timerDEC_start));

    free(absQuant);
    free(fixedRate);
    free(threadOfs);
}









//Code below is for testing purposes only, will be extempt from final submission

float* hawkZip_compress_Test(float* oriData, unsigned char* cmpData, size_t nbEle, size_t* cmpSize, float errorBound)
{
    // Variables used in compression kernel
    int blockNum = ((nbEle + NUM_THREADS - 1) / NUM_THREADS  + 31) / 32 * NUM_THREADS;
    int* absQuant = (int*)malloc(sizeof(int)*nbEle);
    unsigned int* signFlag = (unsigned int*)malloc(sizeof(unsigned int)*blockNum);
    int* fixedRate = (int*)malloc(sizeof(int)*blockNum);
    unsigned int* threadOfs = (unsigned int*)malloc(sizeof(unsigned int)*NUM_THREADS);
    memset(cmpData, 0, sizeof(float)*nbEle);
    double timerCMP_start, timerCMP_end;

    // Compression kernel computation and time measurement.
    timerCMP_start = omp_get_wtime();
    hawkZip_compress_kernel(oriData, cmpData, absQuant, signFlag, fixedRate, threadOfs, nbEle, cmpSize, errorBound);
    timerCMP_end = omp_get_wtime();
    float* output = (float*)malloc(2 * sizeof(float));
    output[0] = (float)(sizeof(float)*nbEle) / (float)(sizeof(unsigned char)*(*cmpSize));
    output[1] = (nbEle*sizeof(float)/1024.0/1024.0/1024.0)/(timerCMP_end-timerCMP_start);

    if (!testing) {
        printf("hawkZip   compression ratio:      %f\n", output[0]);
        printf("hawkZip   compression throughput: %f GB/s\n", output[1]);
    }

    

    // Reallocate memory consumption of cmpData.
    cmpData = (unsigned char*)realloc(cmpData, sizeof(unsigned char)*(*cmpSize));

    free(absQuant);
    free(signFlag);
    free(fixedRate);
    free(threadOfs);

    return output;
}

float hawkZip_decompress_Test(float* decData, unsigned char* cmpData, size_t nbEle, float errorBound)
{
    // Varaibles used in decompression kernel
    int blockNum = ((nbEle + NUM_THREADS - 1) / NUM_THREADS  + 31) / 32 * NUM_THREADS;
    int* absQuant = (int*)malloc(sizeof(int)*nbEle);
    memset(absQuant, 0, sizeof(int)*nbEle);
    int* fixedRate = (int*)malloc(sizeof(int)*blockNum);
    unsigned int* threadOfs = (unsigned int*)malloc(sizeof(unsigned int)*NUM_THREADS);
    memset(decData, 0, sizeof(float)*nbEle);
    double timerDEC_start, timerDEC_end;

    // Decompression kernel computation and time measurement.
    timerDEC_start = omp_get_wtime();
    hawkZip_decompress_kernel(decData, cmpData, absQuant, fixedRate, threadOfs, nbEle, errorBound);
    timerDEC_end = omp_get_wtime();
    float output = (nbEle*sizeof(float)/1024.0/1024.0/1024.0)/(timerDEC_end-timerDEC_start);

    if (!testing) {
        printf("hawkZip decompression throughput: %f GB/s\n", output);
    }

    free(absQuant);
    free(fixedRate);
    free(threadOfs);

    return output;
}