#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tmmintrin.h>
#include <immintrin.h> 
#include <omp.h>
#include <time.h>

#define NUM_THREADS 1

int max(int a, int b) {
    return (a > b) ? a : b;
}


void hawkZip_compress_kernel(float* oriData, unsigned char* cmpData, int* absQuant, unsigned int* signFlag, int* fixedRate, unsigned int* threadOfs, size_t nbEle, size_t* cmpSize, float errorBound)
{
   
    double t_start,t_quantize, t_prefix, t_encoding, t_total_start, t_end;
    t_total_start = omp_get_wtime();
    int chunk_size = (nbEle + NUM_THREADS - 1) / NUM_THREADS;
    omp_set_num_threads(NUM_THREADS);
    // hawkZip parallel compression begin.
    #pragma omp parallel
    {
        t_start = omp_get_wtime();
        // Divide data chunk for each thread
        int thread_id = omp_get_thread_num();
        int start = thread_id * chunk_size;
        int end = start + chunk_size;
        if(end > nbEle) end = nbEle;
        int block_num = (chunk_size+31)/32;
        int start_block = thread_id * block_num;
        int block_start, block_end;
        const float recip_precision = 0.5f/errorBound;
        int sign_ofs, sign_ofs1, sign_ofs2, sign_ofs3;
        unsigned int thread_ofs = 0; 

        // Iterate all blocks in current thread.
        for(int i=0; i<block_num; i++)
        {
            // Block initialization.
        
            block_start = start + i * 32;
            block_end = (block_start+32) > end ? end : block_start+32;
            float data_recip, data_recip1, data_recip2, data_recip3;
            int s;
            int max_quant=0;
            int curr_block = start_block + i;
            unsigned int sign_flag = 0;
            int temp_fixed_rate;
            int j = block_start;
            int parallel_iters = 0;
            int sequential_iters = 0;
            __m128 data_recip_vec;
            __m128i s_vals, curr_quant_vals,bools_temp,sign_mask, abs_vals;

            //Prequantization, get absolute value for each data.
            for( ; j<=block_end-4; j+=4){
                // Prequantization.
                
                data_recip_vec = _mm_mul_ps( _mm_load_ps(&oriData[j]),  _mm_set1_ps(recip_precision));
                s_vals = _mm_srli_epi32(
                                    _mm_castps_si128( _mm_cmplt_ps(data_recip_vec, _mm_set1_ps(-0.5f))),
                                    31);
                curr_quant_vals = _mm_sub_epi32(
                                                _mm_cvttps_epi32(_mm_add_ps(data_recip_vec, _mm_set1_ps(0.5f))), 
                                                s_vals
                                            );
                bools_temp = _mm_and_si128(_mm_cmplt_epi32( curr_quant_vals, _mm_set1_epi32(0)),
                                              _mm_set_epi32(
                                                (1 << (31 - ((j + 3) % 32))),  
                                                (1 << (31 - ((j + 2) % 32))),
                                                (1 << (31 - ((j + 1) % 32))),
                                                (1 << (31 - (j % 32)))
                ));

                
                
                sign_mask = _mm_srai_epi32(curr_quant_vals, 31); // Extract sign bits
                abs_vals = _mm_sub_epi32(_mm_xor_si128(curr_quant_vals, sign_mask), sign_mask);
                _mm_store_si128((__m128i*)&absQuant[j], abs_vals);

                //only need leading zeros, not the true max. just or together to get all the leading zeros
                max_quant |= absQuant[j] | absQuant[j+1] | absQuant[j+3] | absQuant[j+2];
                
                sign_flag |= (_mm_extract_epi32(bools_temp, 0) | _mm_extract_epi32(bools_temp, 1)
                          | _mm_extract_epi32(bools_temp, 2)|_mm_extract_epi32(bools_temp, 3));  

                parallel_iters++;
            }

            // Record fixed-length encoding rate for each block.
            //TODO: Not sure how to make this faster
            signFlag[curr_block] = sign_flag;
            temp_fixed_rate = max_quant==0 ? 0 : sizeof(int) * 8 - __builtin_clz(max_quant);
            fixedRate[curr_block] = temp_fixed_rate;
            cmpData[curr_block] = (unsigned char)temp_fixed_rate;

            // Inner thread prefix-sum.
            thread_ofs += temp_fixed_rate ? (32+temp_fixed_rate*32)/8 : 0; 
        }

        // Store thread ofs to global varaible, used for later global prefix-sum.
        threadOfs[thread_id] = thread_ofs;
        #pragma omp barrier
        t_quantize = omp_get_wtime()- t_start;
        t_start = omp_get_wtime();

        // Exclusive prefix-sum.
        unsigned int global_ofs = 0;
        for(int i=0; i<thread_id; i++) global_ofs += threadOfs[i];
        unsigned int cmp_byte_ofs = global_ofs + block_num * NUM_THREADS;

        t_prefix = omp_get_wtime() - t_start;
        t_start = omp_get_wtime();
        // Fixed-length encoding and store data to compressed data.
        
        
        for(int i=0; i<block_num; i++)
        {
            // Block initialization.
            block_start = start + i * 32;
            block_end = (block_start+32) > end ? end : block_start+32;
            int curr_block = start_block + i;
            int temp_fixed_rate = fixedRate[curr_block];
            unsigned int sign_flag = signFlag[curr_block];
            // Operation for each block, if zero block then do nothing.
            if(temp_fixed_rate)
            {
                // Retrieve sign information for one block.
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag >> 24);
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag >> 16);
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag >> 8);
                cmpData[cmp_byte_ofs++] = 0xff & sign_flag;
                __m128i mask = _mm_set1_epi32(1);

                //TODO potentially invert these two loops so that the acess pattern (of absQuat) is more predictable
                for(int j=0; j<temp_fixed_rate; j++){
                    
                    __m128i block= _mm_loadu_si128((__m128i *)&absQuant[block_start]);
                    __m128i result = _mm_srli_epi32( _mm_and_si128(block, mask), j);
                    __m128i block1= _mm_loadu_si128((__m128i *)&absQuant[block_start + 4]);  
                    __m128i result1 = _mm_srli_epi32( _mm_and_si128(block1, mask), j);
                    __m128i block2= _mm_loadu_si128((__m128i *)&absQuant[block_start + 8]);
                    __m128i result2 = _mm_srli_epi32( _mm_and_si128(block2, mask), j);
                    __m128i block3= _mm_loadu_si128((__m128i *)&absQuant[block_start + 12]);  
                    __m128i result3 = _mm_srli_epi32( _mm_and_si128(block3, mask), j);
                    __m128i block4= _mm_loadu_si128((__m128i *)&absQuant[block_start + 16]);
                    __m128i result4 = _mm_srli_epi32( _mm_and_si128(block4, mask), j);
                    __m128i block5= _mm_loadu_si128((__m128i *)&absQuant[block_start + 20]);  
                    __m128i result5 = _mm_srli_epi32( _mm_and_si128(block5, mask), j);
                    __m128i block6= _mm_loadu_si128((__m128i *)&absQuant[block_start + 24]);
                    __m128i result6 = _mm_srli_epi32( _mm_and_si128(block6, mask), j);
                    __m128i block7= _mm_loadu_si128((__m128i *)&absQuant[block_start + 28]);  
                    __m128i result7 = _mm_srli_epi32( _mm_and_si128(block7, mask), j);



                    cmpData[cmp_byte_ofs++] |= _mm_extract_epi32 (result, 0) << 7
                                | _mm_extract_epi32 (result, 1) << 6
                                | _mm_extract_epi32 (result, 2) << 5
                                | _mm_extract_epi32 (result, 3) << 4
                                | _mm_extract_epi32 (result1, 0) << 3
                                | _mm_extract_epi32 (result1, 1) << 2
                                | _mm_extract_epi32 (result1, 2) << 1
                                | _mm_extract_epi32 (result1, 3) << 0;
                    cmpData[cmp_byte_ofs++] |= _mm_extract_epi32 (result2, 0) << 7
                                | _mm_extract_epi32 (result2, 1) << 6
                                | _mm_extract_epi32 (result2, 2) << 5
                                | _mm_extract_epi32 (result2, 3) << 4
                                | _mm_extract_epi32 (result3, 0) << 3
                                | _mm_extract_epi32 (result3, 1) << 2
                                | _mm_extract_epi32 (result3, 2) << 1
                                | _mm_extract_epi32 (result3, 3) << 0;
                    cmpData[cmp_byte_ofs++] |= _mm_extract_epi32 (result4, 0) << 7
                                | _mm_extract_epi32 (result4, 1) << 6
                                | _mm_extract_epi32 (result4, 2) << 5
                                | _mm_extract_epi32 (result4, 3) << 4
                                | _mm_extract_epi32 (result5, 0) << 3
                                | _mm_extract_epi32 (result5, 1) << 2
                                | _mm_extract_epi32 (result5, 2) << 1
                                | _mm_extract_epi32 (result5, 3) << 0;
                    cmpData[cmp_byte_ofs++] |= _mm_extract_epi32 (result6, 0) << 7
                                | _mm_extract_epi32 (result6, 1) << 6
                                | _mm_extract_epi32 (result6, 2) << 5
                                | _mm_extract_epi32 (result6, 3) << 4
                                | _mm_extract_epi32 (result7, 0) << 3
                                | _mm_extract_epi32 (result7, 1) << 2
                                | _mm_extract_epi32 (result7, 2) << 1
                                | _mm_extract_epi32 (result7, 3) << 0;
                    mask = _mm_slli_epi32(mask, 1);
                }
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
    #pragma omp barrier
    t_encoding = omp_get_wtime() - t_start;
    t_end = omp_get_wtime();
    double total_time = (t_end - t_total_start) * 1000;
    double quantize_time = ( (t_quantize)) * 1000;
    double encoding_time = ((t_encoding)) * 1000;
    double prefix_time = ((t_prefix)) * 1000;
    

    printf("quantize_time=%fms, prefix time=%fms, encoding time=%fms, ratio = %.2f:%.2f:%.2f, total=%fms\n", 
        quantize_time, prefix_time, encoding_time, quantize_time/total_time, prefix_time/total_time, encoding_time/total_time, total_time);
}

void hawkZip_decompress_kernel(float* decData, unsigned char* cmpData, int* absQuant, int* fixedRate, unsigned int* threadOfs, size_t nbEle, float errorBound)
{
    // Shared variables across threads.
    int chunk_size = (nbEle + NUM_THREADS - 1) / NUM_THREADS;
    omp_set_num_threads(NUM_THREADS);
    double t_start,t_read_rates, t_sign_read, t_quant_read, t_total_start, t_end;
    t_total_start = omp_get_wtime();
    // hawkZip parallel decompression begin.
    #pragma omp parallel
    {
        // Divide data chunk for each thread
        int thread_id = omp_get_thread_num();
        int start = thread_id * chunk_size;
        int end = start + chunk_size;
        if(end > nbEle) end = nbEle;
        int block_num = (chunk_size+31)/32;
        int block_start, block_end;
        int start_block = thread_id * block_num;
        unsigned int thread_ofs = 0;
        long paralell_iters, sequential_iters = 0;
        t_start = omp_get_wtime();
        int i  = start_block;
        
        // Iterate all blocks in current thread.

        // for(; i<start + block_num && (i % 16 != 0) ; i++){
        //     // Retrieve fixed-rate for each block in the compressed data.
        //     int temp_fixed_rate = (int)cmpData[i];
        //     fixedRate[i] = temp_fixed_rate;

        //     // Inner thread prefix-sum.
        //     thread_ofs += temp_fixed_rate ? (32+temp_fixed_rate*32)/8 : 0;
        //     sequential_iters++;
        // }

        __m128i *vcmpData = (__m128i *) cmpData;

        __m128i zero_vec = _mm_setzero_si128();
        __m128i four = _mm_set1_epi16(4);

        for (; i < block_num - 16; i += 16) {
            // Load 16 bytes of compressed data
            __m128i temp_fixed_rates = _mm_loadu_si128(&vcmpData[i / 16]);

            // Step 1: Widen 8-bit integers to 16-bit integers
            __m128i lower_16 = _mm_unpacklo_epi8(temp_fixed_rates, zero_vec);
            __m128i upper_16 = _mm_unpackhi_epi8(temp_fixed_rates, zero_vec);

            // Step 2: Widen 16-bit integers to 32-bit integers
            __m128i lower_32_1 = _mm_unpacklo_epi16(lower_16, zero_vec);
            __m128i lower_32_2 = _mm_unpackhi_epi16(lower_16, zero_vec);
            __m128i upper_32_1 = _mm_unpacklo_epi16(upper_16, zero_vec);
            __m128i upper_32_2 = _mm_unpackhi_epi16(upper_16, zero_vec);

            // Store widened values into fixedRate
            _mm_storeu_si128((__m128i *)(&(fixedRate[i])), lower_32_1);
            _mm_storeu_si128((__m128i *)(&(fixedRate[i + 4])), lower_32_2);
            _mm_storeu_si128((__m128i *)(&(fixedRate[i + 8])), upper_32_1);
            _mm_storeu_si128((__m128i *)(&(fixedRate[i + 12])), upper_32_2);

            // Compute activators for prefix-sum
            __m128i activator = _mm_cmpeq_epi8(temp_fixed_rates, zero_vec);
            __m128i uactivator = _mm_srli_epi16(_mm_unpacklo_epi8(activator, zero_vec), 7);
            __m128i hactivator = _mm_srli_epi16(_mm_unpackhi_epi8(activator, zero_vec), 7);

            __m128i uresult = _mm_mullo_epi16(
                _mm_add_epi16(four, _mm_mullo_epi16(four, lower_16)),
                uactivator);
            __m128i hresult = _mm_mullo_epi16(
                _mm_add_epi16(four, _mm_mullo_epi16(four, upper_16)),
                hactivator);

            // Sum the results
            __m128i terms = _mm_add_epi16(uresult, hresult);
            thread_ofs += _mm_extract_epi16(terms, 0) +
                        _mm_extract_epi16(terms, 1) +
                        _mm_extract_epi16(terms, 2) +
                        _mm_extract_epi16(terms, 3);

            paralell_iters++;
        }
        for(; i<start + block_num  ; i++){
            // Retrieve fixed-rate for each block in the compressed data.
            int temp_fixed_rate = (int)cmpData[i];
            fixedRate[i] = temp_fixed_rate;

            // Inner thread prefix-sum.
            thread_ofs += temp_fixed_rate ? (32+temp_fixed_rate*32)/8 : 0;
            sequential_iters++;
        }

        printf("sequentially loaded elements: %ld, vectorized loaded elements:%ld\n", sequential_iters, paralell_iters*16);
        

        // Store thread ofs to global varaible, used for later global prefix-sum.
        threadOfs[thread_id] = thread_ofs;
        #pragma omp barrier
        t_read_rates = omp_get_wtime() - t_start;
        // Exclusive prefix-sum.
        unsigned int global_ofs = 0;
        for(int i=0; i<thread_id; i++) global_ofs += threadOfs[i];
        unsigned int cmp_byte_ofs = global_ofs + block_num * NUM_THREADS;

        t_start = omp_get_wtime();
        // Restore decompressed data.
        for(int i=0; i<block_num; i++)
        {
            // Block initialization.
            block_start = start + i * 32;
            block_end = (block_start+32) > end ? end : block_start+32;
            int curr_block = start_block + i;
            int temp_fixed_rate = fixedRate[curr_block];
            unsigned int sign_flag = 0;
            int sign_ofs;

            // Operation for each block, if zero block then do nothing.
            if(temp_fixed_rate)
            {
                // Retrieve sign information for one block.
                sign_flag = (0xff000000 & (cmpData[cmp_byte_ofs++] << 24)) |
                            (0x00ff0000 & (cmpData[cmp_byte_ofs++] << 16)) |
                            (0x0000ff00 & (cmpData[cmp_byte_ofs++] << 8))  |
                            (0x000000ff & cmpData[cmp_byte_ofs++]);

                // Retrieve quant data for one block.
                unsigned char tmp_char0, tmp_char1, tmp_char2, tmp_char3;
                for(int j=0; j<temp_fixed_rate; j++)
                {
                    // Initialization.
                    tmp_char0 = cmpData[cmp_byte_ofs++];
                    tmp_char1 = cmpData[cmp_byte_ofs++];
                    tmp_char2 = cmpData[cmp_byte_ofs++];
                    tmp_char3 = cmpData[cmp_byte_ofs++];

                    // Get ith bit in 0~7 abs quant from global memory.
                    for(int k=block_start; k<block_start+8; k++)
                        absQuant[k] |= ((tmp_char0 >> (7+block_start-k)) & 0x00000001) << j;

                    // Get ith bit in 8~15 abs quant from global memory.
                    for(int k=block_start+8; k<block_start+16; k++)
                        absQuant[k] |= ((tmp_char1 >> (15+block_start-k)) & 0x00000001) << j;

                    // Get ith bit in 16-23 abs quant from global memory.
                    for(int k=block_start+16; k<block_start+24; k++)
                        absQuant[k] |= ((tmp_char2 >> (23+block_start-k)) & 0x00000001) << j;

                    // Get ith bit in 24-31 abs quant from global memory.
                    for(int k=block_start+24; k<block_end; k++)
                        absQuant[k] |= ((tmp_char3 >> (31+block_start-k)) & 0x00000001) << j;
                }
                t_quant_read = omp_get_wtime() - t_start;
                t_start = omp_get_wtime();
                // De-quantize and store data back to decompression data.
                int currQuant;
                for(int i=block_start; i<block_end; i++)
                {
                    sign_ofs = i % 32;
                    if(sign_flag & (1 << (31 - sign_ofs)))
                        currQuant = absQuant[i] * -1;
                    else
                        currQuant = absQuant[i];
                    decData[i] = currQuant * errorBound * 2;
                }
                t_sign_read = omp_get_wtime() - t_start;
            }
        }
    }
    t_end = omp_get_wtime();
    double total_time = (t_end - t_total_start) * 1000;
    double quantize_time = ( (t_quant_read)) * 1000;
    double rates_time = ((t_read_rates)) * 1000;
    double sign_time = ((t_sign_read)) * 1000;
    

    printf("rates time=%fms, quantize time=%fms, sign time=%fms, ratio = %.2f:%.2f:%.2f, total=%fms\n", 
        rates_time, quantize_time, sign_time, rates_time/total_time, quantize_time/total_time, sign_time/total_time, total_time);
}


//Prequantization, get absolute value for each data.
            //8-wide vectorization 
            // for( ; j<=block_end-4; j+=8)
            // {
               
            //     // Prequantization.
                
            //     data_recip_vec = _mm256_mul_ps( _mm256_load_ps(&oriData[j]),  _mm256_set1_ps(recip_precision));
            //     s_vals = _mm256_srli_epi32(
            //                         _mm256_castps_si256( _mm256_cmplt_ps(data_recip_vec, _mm256_set1_ps(-0.5f))),
            //                         31);
            //     curr_quant_vals = _mm256_sub_epi32(
            //                                     _mm256_cvttps_epi32(_mm256_add_ps(data_recip_vec, _mm256_set1_ps(0.5f))), 
            //                                     s_vals
            //                                 );
            //     bools_temp = _mm256_and_si256(_mm256_cmpgt_epi32( curr_quant_vals, _mm256_set1_epi32(0)),
            //                                   _mm256_set_epi32(
            //                                     (1 << (31 - ((j + 3) % 32))),  
            //                                     (1 << (31 - ((j + 2) % 32))),
            //                                     (1 << (31 - ((j + 1) % 32))),
            //                                     (1 << (31 - (j % 32)))
                                             
            //     ));
            //     int bool_vals_temp[4];
            //     _mm_store_si128((__m128i*)bool_vals_temp, bools_temp);
            //     //printf("bool_vals = %d %d %d %d\n", bool_vals[0] << (31 - j % 32) , bool_vals[1]<< (31 - ((j+1) % 32)), bool_vals[2]<< (31 - ((j+2) % 32)), bool_vals[3]<< (31 - ((j + 3) % 32)));
            //     //printf("bool_vals_temp = %d %d %d %d\n", bool_vals_temp[0], bool_vals_temp[1], bool_vals_temp[2], bool_vals_temp[3]);
            //     sign_flag |= (bool_vals_temp[0] | bool_vals_temp[1]| bool_vals_temp[2]| bool_vals_temp[3]);  
                
            //     sign_mask = _mm_srai_epi32(curr_quant_vals, 31); // Extract sign bits
            //     abs_vals = _mm_sub_epi32(_mm_xor_si128(curr_quant_vals, sign_mask), sign_mask);
            //     _mm_store_si128((__m128i*)&absQuant[j], abs_vals);
                
            //     // int max01 = absQuant[j] > absQuant[j+1] ? absQuant[j] : absQuant[j+1]; // use or to find the maximum quant
            //     // int max23 = absQuant[j+2] > absQuant[j+3] ? absQuant[j+2] : absQuant[j+3]; // use or to find the maximum quant
            //     // max_quant = max01 > max23 ? max01 : max23; // use or to find the maximum quant
            //     max_quant = max_quant > absQuant[j] ? max_quant : absQuant[j]; // use or to find the maximum quant
            //     max_quant = max_quant > absQuant[j+1] ? max_quant : absQuant[j+1]; // use or to find the maximum quant
            //     max_quant = max_quant > absQuant[j+2] ? max_quant : absQuant[j+2]; // use or to find the maximum quant
            //     max_quant = max_quant > absQuant[j+3] ? max_quant : absQuant[j+3]; // use or to find the maximum quant               
            //     parallel_iters++;
            // }

// tmp_char[0] |= (((absQuant[block_start] & mask) >> j) << (7));
                    // tmp_char[0] |= (((absQuant[block_start+1] & mask) >> j) << (6));
                    // tmp_char[0] |= (((absQuant[block_start+2] & mask) >> j) << (5));
                    // tmp_char[0] |= (((absQuant[block_start+3] & mask) >> j) << (4));
                    // tmp_char[0] |= (((absQuant[block_start+4] & mask) >> j) << (3));
                    // tmp_char[0] |= (((absQuant[block_start+5] & mask) >> j) << (2));
                    // tmp_char[0] |= (((absQuant[block_start+6] & mask) >> j) << (1));
                    // tmp_char[0] |= (((absQuant[block_start+7] & mask) >> j) << (0));

                    // tmp_char[1] |= (((absQuant[block_start+8] & mask) >> j) << (7));
                    // tmp_char[1] |= (((absQuant[block_start+8 + 1] & mask) >> j) << (6));
                    // tmp_char[1] |= (((absQuant[block_start+8 + 2] & mask) >> j) << (5));
                    // tmp_char[1] |= (((absQuant[block_start+8 + 3] & mask) >> j) << (4));
                    // tmp_char[1] |= (((absQuant[block_start+8 + 4] & mask) >> j) << (3));
                    // tmp_char[1] |= (((absQuant[block_start+8 + 5] & mask) >> j) << (2));
                    // tmp_char[1] |= (((absQuant[block_start+8 + 6] & mask) >> j) << (1));
                    // tmp_char[1] |= (((absQuant[block_start+8 + 7] & mask) >> j) << (0));

                    // tmp_char[2] |= (((absQuant[block_start+16] & mask) >> j) << (7));
                    // tmp_char[2] |= (((absQuant[block_start+16 + 1] & mask) >> j) << (6));
                    // tmp_char[2] |= (((absQuant[block_start+16 + 2] & mask) >> j) << (5));
                    // tmp_char[2] |= (((absQuant[block_start+16 + 3] & mask) >> j) << (4));
                    // tmp_char[2] |= (((absQuant[block_start+16 + 4] & mask) >> j) << (3));
                    // tmp_char[2] |= (((absQuant[block_start+16 + 5] & mask) >> j) << (2));
                    // tmp_char[2] |= (((absQuant[block_start+16 + 6] & mask) >> j) << (1));
                    // tmp_char[2] |= (((absQuant[block_start+16 + 7] & mask) >> j) << (0));

                    // tmp_char[3] |= (((absQuant[block_start+24] & mask) >> j) << (7));
                    // tmp_char[3] |= (((absQuant[block_start+24 + 1] & mask) >> j) << (6));
                    // tmp_char[3] |= (((absQuant[block_start+24 + 2] & mask) >> j) << (5));
                    // tmp_char[3] |= (((absQuant[block_start+24 + 3] & mask) >> j) << (4));
                    // tmp_char[3] |= (((absQuant[block_start+24 + 4] & mask) >> j) << (3));
                    // tmp_char[3] |= (((absQuant[block_start+24 + 5] & mask) >> j) << (2));
                    // tmp_char[3] |= (((absQuant[block_start+24 + 6] & mask) >> j) << (1));
                    // tmp_char[3] |= (((absQuant[block_start+24 + 7] & mask) >> j) << (0));

                    // for( ; j<block_end; j++)
            // {
            //     // Prequantization.

            //     data_recip = oriData[j] * recip_precision; //unroll and use SIMD for this operation
            //     //printf("original data = 0x%08x\n", *((u_int32_t*)&oriData[j]));

            //     s = data_recip >= -0.5f ? 0 : 1;
            //     curr_quant = (int)(data_recip + 0.5f) - s;

            //     // Get sign data.
            //     sign_ofs = j % 32; // should inline this variable, no reason for it to be a declared variable
            //     sign_flag |= (curr_quant < 0) << (31 - sign_ofs); //critical region, 

            //     // Get absolute quantization code.
            //     max_quant = max_quant > abs(curr_quant) ? max_quant : abs(curr_quant); // use or to find the maximum quant


            //     absQuant[j] = abs(curr_quant); //unroll and use SIMD for this operation
            //     sequential_iters++;
            // }
            //printf("Parallel iters=%d, sequential iters=%d\n", parallel_iters, sequential_iters);

            // for(int k=block_start; k<block_start+8; k+=1){
                        
                    //     offset = 7-k + block_start;
                        
                        
                    //     tmp_char[0] |= (((absQuant[k] & mask) >> j) << (offset));

                    //     tmp_char[1] |= (((absQuant[k+8] & mask) >> j) << (offset));
                    
                    //     tmp_char[2] |= (((absQuant[k+16] & mask) >> j) << (offset));
                   
                    //     tmp_char[3] |= (((absQuant[k+24] & mask) >> j) << (offset));



                    //     //printf("k=%d: ", k);
                    //     //printBits(tmp_char, 0, 4);
                    // }