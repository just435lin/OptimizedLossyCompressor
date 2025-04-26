#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tmmintrin.h>
#include <immintrin.h> 
#include <omp.h>

#define NUM_THREADS 32
#define BLOCK_SIZE 33



void hawkZip_compress_kernel(float* oriData, unsigned char* cmpData, int* absQuant, unsigned int* signFlag, int* fixedRate, unsigned int* threadOfs, size_t nbEle, size_t* cmpSize, float errorBound)
{
    // Shared variables across threads.
    int chunk_size = (nbEle + NUM_THREADS - 1) / NUM_THREADS;
    omp_set_num_threads(NUM_THREADS);
    int block_num = (chunk_size+BLOCK_SIZE-1)/BLOCK_SIZE;
    
    
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
            
            // Prequantization, get absolute value for each data.
            for(int j= block_start + 1; j  < block_end; j ++)
            {
                prev_quant = curr_quant;

                // Prequantization.
                data_recip = (oriData[j]) * recip_precision;
                curr_quant = data_recip < 0.0f ? (int)(data_recip - 0.5f) : (int)(data_recip + 0.5f);

                // Get sign data.
                sign_flag |= (curr_quant < 0) << (j - (block_start + 1));
                int dif = abs(curr_quant) - abs(prev_quant);
                int store = (dif << 1) ^ (dif >> 31);

                // Get absolute quantization code.
                max_quant |= store;
                absQuant[j] = store;
            }

            char all_signs = (sign_flag == 0 || sign_flag == 0xFFFFFFFF) & 1;            

            // Record fixed-length encoding rate for each block.
            signFlag[curr_block] = sign_flag;
            temp_fixed_rate = max_quant==0 ? 0 : (32 - __builtin_clz(max_quant));
            
            // Inner thread prefix-sum.
            thread_ofs += ((1+temp_fixed_rate+!all_signs)<<2);

            temp_fixed_rate |= (all_signs << 5) | ((all_signs & sign_flag) << 6);
            fixedRate[curr_block] = temp_fixed_rate;
            cmpData[curr_block] = (unsigned char)temp_fixed_rate;


        }

        // Store thread ofs to global varaible, used for later global prefix-sum.
        threadOfs[thread_id] = thread_ofs;
        #pragma omp barrier

        // Exclusive prefix-sum.
        unsigned int global_ofs = 0;
        for(int i=thread_id; i--;) global_ofs += threadOfs[i];
        unsigned int cmp_byte_ofs = global_ofs + block_num * NUM_THREADS;
        __m128i mask = _mm_set1_epi32(0x00000001);

        // Fixed-length encoding and store data to compressed data.
        for(int i= 0; i< block_num-1; i++)
        {
            // Block initialization.
            block_start = start + i * BLOCK_SIZE;
            block_end = block_start + BLOCK_SIZE;
            int curr_block = start_block + i;
            int temp_fixed_rate = fixedRate[curr_block];
            int all_signs = temp_fixed_rate & (1 << 5);
            temp_fixed_rate &= 0x1F;
	
            // directly store first floating point of block...
            memcpy(&cmpData[cmp_byte_ofs], &absQuant[block_start], 4);
            cmp_byte_ofs += 4;


            // Operation for each block, if zero block then do nothing.
            if(temp_fixed_rate)
            {
                if (!all_signs) {
                    // Retrieve sign information for one block.
                    memcpy(&cmpData[cmp_byte_ofs], &signFlag[curr_block], 4);
                    cmp_byte_ofs += 4;
                }

                // Retrieve quant data for one block.

                int data;
                int loc;
                for (int bit = temp_fixed_rate; bit --;) {
                    data = 0;
                    loc = block_end;
                    for(int i = 0; i <2; i++){
                    __m128i absQuantVec0 = _mm_loadu_si128((__m128i *)&absQuant[loc-4]);
                    __m128i absQuantVec1 = _mm_loadu_si128((__m128i *)&absQuant[loc-8]);
                    __m128i absQuantVec2 = _mm_loadu_si128((__m128i *)&absQuant[loc-12]);
                    __m128i absQuantVec3 = _mm_loadu_si128((__m128i *)&absQuant[loc-16]);
                    loc -= 16;

                    absQuantVec0 = _mm_and_si128(_mm_srli_epi32(absQuantVec0, bit), mask);  
                    absQuantVec1 = _mm_and_si128(_mm_srli_epi32(absQuantVec1, bit), mask);
                    absQuantVec2 = _mm_and_si128(_mm_srli_epi32(absQuantVec2, bit), mask);
                    absQuantVec3 = _mm_and_si128(_mm_srli_epi32(absQuantVec3, bit), mask);
                    data |=   _mm_extract_epi32(absQuantVec0, 3) << (31 - 16 * i)
                         |    _mm_extract_epi32(absQuantVec0, 2) << (30 - 16 * i)
                         |    _mm_extract_epi32(absQuantVec0, 1) << (29 - 16 * i)
                         |    _mm_extract_epi32(absQuantVec0, 0) << (28 - 16 * i) 
                         |    _mm_extract_epi32(absQuantVec1, 3) << (27 - 16 * i)
                         |    _mm_extract_epi32(absQuantVec1, 2) << (26 - 16 * i)
                         |    _mm_extract_epi32(absQuantVec1, 1) << (25 - 16 * i)
                         |    _mm_extract_epi32(absQuantVec1, 0) << (24 - 16 * i) 
                         |    _mm_extract_epi32(absQuantVec2, 3) << (23 - 16 * i)
                         |    _mm_extract_epi32(absQuantVec2, 2) << (22 - 16 * i)
                         |    _mm_extract_epi32(absQuantVec2, 1) << (21 - 16 * i)
                         |    _mm_extract_epi32(absQuantVec2, 0) << (20 - 16 * i) 
                         |    _mm_extract_epi32(absQuantVec3, 3) << (19 - 16 * i)
                         |    _mm_extract_epi32(absQuantVec3, 2) << (18 - 16 * i)
                         |    _mm_extract_epi32(absQuantVec3, 1) << (17 - 16 * i)
                         |    _mm_extract_epi32(absQuantVec3, 0) << (16 - 16 * i);
                    } 
                    memcpy(&cmpData[cmp_byte_ofs + (bit << 2)], &data,  4);

                  
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
            int all_signs = temp_fixed_rate & (1 << 5);
            temp_fixed_rate &= 0x1F;
	
            // directly store first floating point of block...
            memcpy(&cmpData[cmp_byte_ofs], &absQuant[block_start], 4);
            cmp_byte_ofs += 4;


            // Operation for each block, if zero block then do nothing.
            if(temp_fixed_rate)
            {
                
                if (!all_signs) {
                    // Retrieve sign information for one block.
                    memcpy(&cmpData[cmp_byte_ofs], &signFlag[curr_block], 4);
                    cmp_byte_ofs += 4;
                }

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
            char all_signs = temp_fixed_rate & (1 << 5);
            temp_fixed_rate &= 0x1F;
            fixedRate[curr_block] = temp_fixed_rate;

            // Inner thread prefix-sum.
            thread_ofs += (1 + temp_fixed_rate+!all_signs)<<2;

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
            char all_signs = cmpData[curr_block] & (3 << 5);
            unsigned int sign_flag = 0;
            int sign_ofs;
            
            memcpy((char*)&absQuant[block_start], &cmpData[cmp_byte_ofs], 4);
            cmp_byte_ofs += 4;
            decData[block_start] = absQuant[block_start] * errorBound * 2;

            // Operation for each block, if zero block then do nothing.
            if(temp_fixed_rate)
            {
                if(all_signs) {
                    sign_flag = (all_signs & (1 << 6)) ? 0xFFFFFFFF : 0;
                }
                else {
                    memcpy((char*)&sign_flag, &cmpData[cmp_byte_ofs],4);
                    cmp_byte_ofs += 4;
                }
               
                int bitset;
                __m128i mask = _mm_set1_epi32(0x00000001);
                __m128i absQuant00 = _mm_set1_epi32(0);
                __m128i absQuant01 = _mm_set1_epi32(0);
                __m128i absQuant10 = _mm_set1_epi32(0);
                __m128i absQuant11 = _mm_set1_epi32(0);
                __m128i absQuant20 = _mm_set1_epi32(0);
                __m128i absQuant21 = _mm_set1_epi32(0);
                __m128i absQuant30 = _mm_set1_epi32(0);
                __m128i absQuant31 = _mm_set1_epi32(0);
                for (int bit = temp_fixed_rate; bit --;) {
                    memcpy(&bitset, &cmpData[cmp_byte_ofs + (bit << 2)], 4);
                    absQuant00 = _mm_or_si128(absQuant00, _mm_slli_epi32(
                                    _mm_and_si128( 
                                        _mm_set_epi32(bitset >> 31, bitset >> 30, bitset >> 29, bitset >> 28),
                                        mask),
                                    bit));    
                    absQuant01 = _mm_or_si128(absQuant01, _mm_slli_epi32(
                                    _mm_and_si128( 
                                        _mm_set_epi32(bitset >> 27, bitset >> 26, bitset >> 25, bitset >> 24),
                                        mask),
                                    bit));    
                    absQuant10 = _mm_or_si128(absQuant10, _mm_slli_epi32(
                                    _mm_and_si128( 
                                        _mm_set_epi32(bitset >> 23, bitset >> 22, bitset >> 21, bitset >> 20),
                                        mask),
                                    bit));    
                    absQuant11 = _mm_or_si128(absQuant11, _mm_slli_epi32(
                                    _mm_and_si128( 
                                        _mm_set_epi32(bitset >> 19, bitset >> 18, bitset >> 17, bitset >> 16),
                                        mask),
                                    bit));    
                    absQuant20 = _mm_or_si128(absQuant20, _mm_slli_epi32(
                                    _mm_and_si128( 
                                        _mm_set_epi32(bitset >> 15, bitset >> 14, bitset >> 13, bitset >> 12),
                                        mask),
                                    bit));    
                    absQuant21 = _mm_or_si128(absQuant21, _mm_slli_epi32(
                                    _mm_and_si128( 
                                        _mm_set_epi32(bitset >> 11, bitset >> 10, bitset >> 9, bitset >> 8),
                                        mask),
                                    bit));    
                    absQuant30 = _mm_or_si128(absQuant30, _mm_slli_epi32(
                                    _mm_and_si128( 
                                        _mm_set_epi32(bitset >> 7, bitset >> 6, bitset >> 5, bitset >> 4),
                                        mask),
                                    bit));    
                    absQuant31 = _mm_or_si128(absQuant31, _mm_slli_epi32(
                                    _mm_and_si128( 
                                        _mm_set_epi32(bitset >> 3, bitset >> 2, bitset >> 1, bitset >> 0),
                                        mask),
                                    bit));    


                    _mm_storeu_si128((__m128i *)&absQuant[block_end-4], absQuant00);
                    _mm_storeu_si128((__m128i *)&absQuant[block_end-8], absQuant01);
                    _mm_storeu_si128((__m128i *)&absQuant[block_end-12], absQuant10);
                    _mm_storeu_si128((__m128i *)&absQuant[block_end-16], absQuant11);
                    _mm_storeu_si128((__m128i *)&absQuant[block_end-20], absQuant20);
                    _mm_storeu_si128((__m128i *)&absQuant[block_end-24], absQuant21);
                    _mm_storeu_si128((__m128i *)&absQuant[block_end-28], absQuant30);
                    _mm_storeu_si128((__m128i *)&absQuant[block_end-32], absQuant31);

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
            char all_signs = cmpData[curr_block] & (3 << 5);
            unsigned int sign_flag = 0;
            int sign_ofs;
            
            memcpy((char*)&absQuant[block_start], &cmpData[cmp_byte_ofs], 4);
            cmp_byte_ofs += 4;
            decData[block_start] = absQuant[block_start] * errorBound * 2;

            // Operation for each block, if zero block then do nothing.
            if(temp_fixed_rate)
            {
                if(all_signs) {
                    sign_flag = (all_signs & (1 << 6)) ? 0xFFFFFFFF : 0;
                }
                else {
                    memcpy((char*)&sign_flag, &cmpData[cmp_byte_ofs],4);
                    cmp_byte_ofs += 4;
                }

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