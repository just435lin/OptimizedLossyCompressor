#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <emmintrin.h>
#include <omp.h>

#define NUM_THREADS 4
#define BLOCK_SIZE 33

//#define COMPRESS_DEBUG_PRINT

#ifdef COMPRESS_DEBUG_PRINT
int chosen_block;

void printbin(int val) {
  for (int i = 0; i < 32; i ++) {
    printf("%d ", (val >> (31 - i)) & 1);
  }
  //printf("\n");
}
#endif

void hawkZip_compress_kernel(float* oriData, unsigned char* cmpData, int* absQuant, unsigned int* signFlag, int* fixedRate, unsigned int* threadOfs, size_t nbEle, size_t* cmpSize, float errorBound)
{
    #ifdef COMPRESS_DEBUG_PRINT
    	printf("ENTER COMPRESS\n");
    #endif
    // Shared variables across threads.
    int chunk_size = (nbEle + NUM_THREADS - 1) / NUM_THREADS;
    omp_set_num_threads(NUM_THREADS);
    int block_num = (chunk_size+BLOCK_SIZE-1)/BLOCK_SIZE;
    
    #ifdef COMPRESS_DEBUG_PRINT
        srand(getpid());
        chosen_block = rand() % block_num;
        printf("INSPECTING BLOCK %d\n", chosen_block);
    #endif
   
    // hawkZip parallel compression begin.
    #pragma omp parallel
    {
        // Divide data chunk for each thread
        int thread_id = omp_get_thread_num();
        int start = thread_id * chunk_size;
        int end = start + chunk_size;
        if(end > nbEle) end = nbEle;
        int start_block = thread_id * block_num;
        int block_start, block_end;
        const float recip_precision = 0.5f/errorBound;
        int sign_ofs;
        unsigned int thread_ofs = 0; 

        #ifdef COMPRESS_DEBUG_PRINT
            if (thread_id == 0)
                printf("ERROR BOUND: %f\n", errorBound);
        #endif


        // Iterate all blocks in current thread.
        for(int i=0; i<block_num; i++)
        {
            // Block initialization.
            block_start = start + i * BLOCK_SIZE;
            block_end = (block_start+BLOCK_SIZE) > end ? end : block_start+BLOCK_SIZE;
            float data_recip;
            int s;
            int curr_quant, prev_quant = 0;
            unsigned int max_quant=0;
            int curr_block = start_block + i;
            unsigned int sign_flag = 0;
            int temp_fixed_rate;

            
            // store the quantized first directly...
            data_recip = (oriData[block_start]) * recip_precision;
            curr_quant = data_recip < 0.0f ? (int)(data_recip - 0.5f) : (int)(data_recip + 0.5f);
            absQuant[block_start] = curr_quant;
            
            
            #ifdef COMPRESS_DEBUG_PRINT
                if (chosen_block == curr_block)
                {
                    printf("first: %f quant: ", oriData[block_start]);
                    printbin(curr_quant);
                    printf("\n");
                }
            #endif
            
            // Prequantization, get absolute value for each data.
            for(int j=0; j < 32 && j + block_start + 1 < block_end; j ++)
            {
                prev_quant = curr_quant;
                int loc = j + block_start + 1;

                // Prequantization.
                data_recip = (oriData[loc]) * recip_precision;
                curr_quant = data_recip < 0.0f ? (int)(data_recip - 0.5f) : (int)(data_recip + 0.5f);
                

                // Get sign data.
                sign_flag |= (curr_quant < 0) << j;


                int dif = abs(curr_quant) - abs(prev_quant);
                int store = (dif<0) ? (abs(dif)<< 1)-1 : (dif << 1);

                // Get absolute quantization code.
                max_quant |= store;
                absQuant[loc] = store;


                #ifdef COMPRESS_DEBUG_PRINT
                    if (chosen_block == curr_block)
                    {
                        printf("%d: orig: %f, absquant: %d, quant: ", j, oriData[loc], absQuant[loc]);
                        printbin(store);
                        printf("\n");
                    }
                #endif
            }

            
            // Record fixed-length encoding rate for each block.
            signFlag[curr_block] = sign_flag;
            temp_fixed_rate = max_quant==0 ? 0 : (32 - __builtin_clz(max_quant));
            fixedRate[curr_block] = temp_fixed_rate;
            cmpData[curr_block] = (unsigned char)temp_fixed_rate;

            #ifdef COMPRESS_DEBUG_PRINT
                if (chosen_block ==curr_block) {
                    printf("max_quant: %d bitwise: ", max_quant);
                    printbin(max_quant);
                    printf("\n");

                    printf("signs: ");
                    printbin(sign_flag);
                    printf("\n");

                    printf("bytes needed: %d\n", ((temp_fixed_rate+1)<<2));
                }
            #endif


            // Inner thread prefix-sum.
            thread_ofs += 4 + (temp_fixed_rate ? ((temp_fixed_rate+1)<<2) : 0);
        }

        // Store thread ofs to global varaible, used for later global prefix-sum.
        threadOfs[thread_id] = thread_ofs;
        #pragma omp barrier

        // Exclusive prefix-sum.
        unsigned int global_ofs = 0;
        for(int i=thread_id; i--;) global_ofs += threadOfs[i];
        unsigned int cmp_byte_ofs = global_ofs + block_num * NUM_THREADS;


        // Fixed-length encoding and store data to compressed data.
        for(int i= 0; i< block_num-1; i++)
        {
            // Block initialization.
            block_start = start + i * BLOCK_SIZE;
            block_end = block_start + BLOCK_SIZE;
            int curr_block = start_block + i;
            int temp_fixed_rate = fixedRate[curr_block];
	
            // directly store first floating point of block...
            memcpy(&cmpData[cmp_byte_ofs], &absQuant[block_start], 4);
            cmp_byte_ofs += 4;


            // Operation for each block, if zero block then do nothing.
            if(temp_fixed_rate)
            {
                
                // Retrieve sign information for one block.
                memcpy(&cmpData[cmp_byte_ofs], &signFlag[curr_block], 4);
                cmp_byte_ofs += 4;

                // Retrieve quant data for one block.

                int data;
                int loc;
                int o;
                for (int bit = temp_fixed_rate; bit --;) {
                    data = 0;
                    loc = block_end;
                    o = 32;
                    do {
                        data |= 
                           (((absQuant[--loc] >> bit) & 1) << (--o))
                         | (((absQuant[--loc] >> bit) & 1) << (--o))
                         | (((absQuant[--loc] >> bit) & 1) << (--o))
                         | (((absQuant[--loc] >> bit) & 1) << (--o));
                    } while (o);
                    memcpy(&cmpData[cmp_byte_ofs + (bit << 2)], &data,  4);

                    #ifdef COMPRESS_DEBUG_PRINT
                        if (chosen_block == curr_block) 
                        {
                            printf("%d-bits: ",bit);
                            printbin(data);
                            printf("\n");
                        }
                    #endif
                }
                cmp_byte_ofs += temp_fixed_rate << 2;
            }
        }
     
        // the last block
        {
            // Block initialization.
            block_start = start + (block_num-1) * BLOCK_SIZE;
            block_end = (block_start + BLOCK_SIZE > end) ? end : block_start + BLOCK_SIZE;
            int curr_block = start_block + (block_num-1);
            int temp_fixed_rate = fixedRate[curr_block];
	
            // directly store first floating point of block...
            memcpy(&cmpData[cmp_byte_ofs], &absQuant[block_start], 4);
            cmp_byte_ofs += 4;


            // Operation for each block, if zero block then do nothing.
            if(temp_fixed_rate)
            {
                
                // Retrieve sign information for one block.
                memcpy(&cmpData[cmp_byte_ofs], &signFlag[curr_block], 4);
                cmp_byte_ofs += 4;

                // Retrieve quant data for one block.

                int data;
                int loc;
                int o;
                for (int bit = temp_fixed_rate; bit --;) {
                    data = 0;
                    loc = block_end;
                    o = block_end - block_start - 1;
                    do {
                        data |= 
                           (((absQuant[--loc] >> bit) & 1) << (--o));
                    } while (o);
                    memcpy(&cmpData[cmp_byte_ofs + (bit << 2)], &data,  4);

                    #ifdef COMPRESS_DEBUG_PRINT
                        if (chosen_block == curr_block) 
                        {
                            printf("%d-bits: ",bit);
                            printbin(data);
                            printf("\n");
                        }
                    #endif
                }
                cmp_byte_ofs += temp_fixed_rate << 2;
            }
        }
     

        // Return the compression data length.
        if(thread_id == NUM_THREADS - 1)
        {
            unsigned int cmpBlockInBytes = 0;
            for(int i=0; i<=thread_id; i++) cmpBlockInBytes += threadOfs[i];
            *cmpSize = (size_t)(cmpBlockInBytes + block_num * NUM_THREADS);
        }
    }
}

void hawkZip_decompress_kernel(float* decData, unsigned char* cmpData, int* absQuant, int* fixedRate, unsigned int* threadOfs, size_t nbEle, float errorBound)
{
    // Shared variables across threads.
    int chunk_size = (nbEle + NUM_THREADS - 1) / NUM_THREADS;
    omp_set_num_threads(NUM_THREADS);
    
    // hawkZip parallel decompression begin.
    #pragma omp parallel
    {
        // Divide data chunk for each thread
        int thread_id = omp_get_thread_num();
        int start = thread_id * chunk_size;
        int end = start + chunk_size;
        if(end > nbEle) end = nbEle;
        int block_num = (chunk_size+BLOCK_SIZE-1)/BLOCK_SIZE;
        int block_start, block_end;
        int start_block = thread_id * block_num;
        unsigned int thread_ofs = 0;


        // Iterate all blocks in current thread.
        for(int i=0; i<block_num; i++)
        {
            // Retrieve fixed-rate for each block in the compressed data.
            int curr_block = start_block + i;
            int temp_fixed_rate = (int)cmpData[curr_block];
            fixedRate[curr_block] = temp_fixed_rate;

            // Inner thread prefix-sum.
            thread_ofs += 4 + (temp_fixed_rate ? (((1+temp_fixed_rate)*32) >> 3) : 0);
        }

        // Store thread ofs to global varaible, used for later global prefix-sum.
        threadOfs[thread_id] = thread_ofs;
        #pragma omp barrier


        // Exclusive prefix-sum.
        unsigned int global_ofs = 0;
        for(int i=0; i<thread_id; i++) global_ofs += threadOfs[i];
        unsigned int cmp_byte_ofs = global_ofs + block_num * NUM_THREADS;

        // Restore decompressed data.
        for(int i=0; i<block_num-1;i++)
        {
            // Block initialization.
            block_start = start + i * BLOCK_SIZE;
            block_end = block_start+BLOCK_SIZE;
            int curr_block = start_block + i;
            int temp_fixed_rate = fixedRate[curr_block];
            unsigned int sign_flag = 0;
            int sign_ofs;

            
            
            memcpy((char*)&absQuant[block_start], &cmpData[cmp_byte_ofs], 4);
            cmp_byte_ofs += 4;
            decData[block_start] = absQuant[block_start] * errorBound * 2;


            #ifdef COMPRESS_DEBUG_PRINT
                if (chosen_block == curr_block) 
                {
                    printf("First item: %f\n", *(float*)&decData[block_start]);
                }
            #endif

            // Operation for each block, if zero block then do nothing.
            if(temp_fixed_rate)
            {
                memcpy((char*)&sign_flag, &cmpData[cmp_byte_ofs],4);
                cmp_byte_ofs += 4;


                #ifdef COMPRESS_DEBUG_PRINT
                    if (chosen_block == curr_block) 
                    {
                        printf("Sign flag: ");
                        printbin(sign_flag);
                        printf("\n");
                    }
                #endif
                
                int bitset;
                int o;
                int loc;
                for (int bit = temp_fixed_rate; bit --;) {
                    memcpy(&bitset, &cmpData[cmp_byte_ofs + (bit << 2)], 4);
                    o = 32;
                    loc = block_end;
                    do {
                        absQuant[--loc] |= ((bitset >> (--o)) & 1) << bit;
                        absQuant[--loc] |= ((bitset >> (--o)) & 1) << bit;
                        absQuant[--loc] |= ((bitset >> (--o)) & 1) << bit;
                        absQuant[--loc] |= ((bitset >> (--o)) & 1) << bit;
                    } while (o);
                 } 
                 cmp_byte_ofs += 4 * temp_fixed_rate;
                

                {
                    // De-quantize and store data back to decompression data.
                    int currQuant = 0;
                    int prevQuant = absQuant[block_start];
                    int o = 0;
                    int loc = block_start+1;
                    do
                    {
                        currQuant = absQuant[loc];
                        if(currQuant & 1) {
                             // if odd, then it was < 0 before...
                             currQuant = -((1+currQuant) >> 1);
                        }
                        else {
                             currQuant >>= 1;
                        }
                        currQuant += abs(prevQuant);

                        if(sign_flag & (1 << o))
                            currQuant = -currQuant;
                        

                        decData[loc] = currQuant * errorBound * 2;
                        #ifdef COMPRESS_DEBUG_PRINT
                            if (chosen_block == curr_block) 
                            {
                                printf("%d: Recovered value: %f, absQuant: %d, bin: ",o, decData[loc], currQuant);
                                printbin(absQuant[loc]);
                                printf("\n");
                            }
                        #endif 
                        o++;
                        loc++;
                        prevQuant = currQuant;

                    } while (o < 32);
                }
            }
            else {
                for (int j = block_start + 1; j < block_end; j++) {
                    decData[j] = decData[block_start];
                }
            }
        }
        
        // the last block
        {
            // Block initialization.
            block_start = start + (block_num-1) * BLOCK_SIZE;
            block_end = (block_start+BLOCK_SIZE) > end ? end : block_start+BLOCK_SIZE;
            int curr_block = start_block + (block_num-1);
            int temp_fixed_rate = fixedRate[curr_block];
            unsigned int sign_flag = 0;
            int sign_ofs;

            
            
            memcpy((char*)&absQuant[block_start], &cmpData[cmp_byte_ofs], 4);
            cmp_byte_ofs += 4;
            decData[block_start] = absQuant[block_start] * errorBound * 2;


            #ifdef COMPRESS_DEBUG_PRINT
                if (chosen_block == curr_block) 
                {
                    printf("First item: %f\n", *(float*)&decData[block_start]);
                }
            #endif

            // Operation for each block, if zero block then do nothing.
            if(temp_fixed_rate)
            {
                memcpy((char*)&sign_flag, &cmpData[cmp_byte_ofs],4);
                cmp_byte_ofs += 4;


                #ifdef COMPRESS_DEBUG_PRINT
                    if (chosen_block == curr_block) 
                    {
                        printf("Sign flag: ");
                        printbin(sign_flag);
                        printf("\n");
                    }
                #endif
                
                int bitset;
                int o;
                int loc;
                for (int bit = temp_fixed_rate; bit --;) {
                    memcpy(&bitset, &cmpData[cmp_byte_ofs + (bit << 2)], 4);
                    o = block_end - block_start - 1;
                    loc = block_end;
                    do {
                        absQuant[--loc] |= ((bitset >> (--o)) & 1) << bit;
                    } while (o);
                 } 
                 cmp_byte_ofs += 4 * temp_fixed_rate;
                

                {
                    // De-quantize and store data back to decompression data.
                    int currQuant = 0;
                    int prevQuant = absQuant[block_start];
                    int o = 0;
                    int loc = block_start+1;
                    do
                    {
                        currQuant = absQuant[loc];
                        if(currQuant & 1) {
                             // if odd, then it was < 0 before...
                             currQuant = -((1+currQuant) >> 1);
                        }
                        else {
                             currQuant >>= 1;
                        }
                        currQuant += abs(prevQuant);

                        if(sign_flag & (1 << o))
                            currQuant = -currQuant;
                        

                        decData[loc] = currQuant * errorBound * 2;
                        #ifdef COMPRESS_DEBUG_PRINT
                            if (chosen_block == curr_block) 
                            {
                                printf("%d: Recovered value: %f, absQuant: %d, bin: ",o, decData[loc], currQuant);
                                printbin(absQuant[loc]);
                                printf("\n");
                            }
                        #endif 
                        o++;
                        loc++;
                        prevQuant = currQuant;

                    } while (o < block_end - block_start - 1);
                }
            }
            else {
                for (int j = block_start + 1; j < block_end; j++) {
                    decData[j] = decData[block_start];
                }
            }
        }


    }
}