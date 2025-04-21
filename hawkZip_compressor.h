#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <emmintrin.h>
#include <omp.h>

#define NUM_THREADS 4

#define BLOCKSIZE 33

void hawkZip_compress_kernel(
    float* oriData,            // a pointer to the memory containing floats to compress (accessed ber f32)
    unsigned char* cmpData,    // a pointer to the memory to write compressed data to   (accessed per-byte)
    int* absQuant,             // array of per fp quantization
    unsigned int* signFlag,    // array of per-block sign flags
    int* fixedRate,            // array of each block's size.
    unsigned int* threadOfs,   // available space for an array of how much compressed memory is needed per thread. (written to by method)
    size_t nbEle,              // size of original memory as interpreted as 32f.
    size_t* cmpSize,           // available space for size of compressed memory (written to by method, not a parameter!)
    float errorBound           // maximum error allowed
    )
{
    // Shared variables across threads.
    int chunk_size = (nbEle + NUM_THREADS - 1) / NUM_THREADS;
    int block_num = (chunk_size+BLOCKSIZE-1)/BLOCKSIZE;            // the number of blocks per chunk ( could be moved outside parallel? )
    omp_set_num_threads(NUM_THREADS);
    
    // hawkZip parallel compression begin.
    #pragma omp parallel
    {
        // Divides the original data into per-thread 'chunks'
        int thread_id = omp_get_thread_num();             // this thread's id
        int chunk_start = thread_id * chunk_size;         // where in the original memory the chunk begins
        int chunk_end = chunk_start + chunk_size;         // where in the original memory the chunk ends
        if (chunk_end > nbEle) chunk_end = nbEle;         //  > prevent the chunk from exceeding the bounds of the original data
        
        // Divides the per-thread 'chunks' into 'blocks' of 32 floating-point numbers
        int start_block = thread_id * block_num;       // the first block in memory
        int block_start, block_end;                    // where the current block starts and ends
        
        const float recip_precision = 0.5f/errorBound; //
        unsigned int thread_ofs = 0;
        

        // Iterate all blocks in current thread.
        for(int i=0; i<block_num; i++)
        {
            // Block initialization.
            block_start = chunk_start + i * BLOCKSIZE;                               // 
            block_end = (block_start+BLOCKSIZE) > chunk_end ? chunk_end : block_start+BLOCKSIZE;
            
            float data_recip;
            int s;
            int curr_quant, max_quant = 0;     // the data's current quantization
            int curr_block = start_block + i;  // the current block (across all memory)
            unsigned int sign_flag = 0;
            int temp_fixed_rate;
            
            float oriStart = oriData[block_start];
            data_recip = oriStart * recip_precision;
            curr_quant = (int)(data_recip + (data_recip < 0.0f ? -0.5f : 0.5f));
            absQuant[block_start] = abs(curr_quant);            
            int firstQuant = absQuant[block_start];
            // Over every value in the block.
            // Prequantization, get absolute value for each data.
            int loc = block_start + 1;
            int j = 0;
            do 
            {
                // Prequantization.
                data_recip = oriData[loc] * recip_precision; // divide the data by errorBound ( and by 2?)
                curr_quant = (int)(data_recip + (data_recip < 0.0f ? -0.5f : 0.5f)); // Round to nearest integer by truncating
                
                // Get sign data.
                sign_flag |= (curr_quant < 0) << (31 - j);  // marking the sign of the quantization as a bitflag: 1 => negative, 0 => positive
                
                int temp = abs(curr_quant);

                curr_quant = (firstQuant - (abs(curr_quant)));

                int temp2 = curr_quant;
                
                if (curr_quant!=0) {
                    curr_quant = (curr_quant<0) ? curr_quant*(-2)-1 : (curr_quant*2);        //zig zag maps -1=1 1=2 -2=2 2=4 etc
                }
                
                if (loc==25636) {
                    printf("Compressed Quant: %d-- %d - %d = %d\n",curr_quant,firstQuant,temp,temp2);
                }

                absQuant[loc] = curr_quant;                                         // write down the data
                //printf("Quant: %d\n",curr_quant);
                
                // Get absolute quantization code.
                max_quant |= curr_quant; // max_quant > curr_quant ? max_quant : curr_quant; // increase max_quant to keep quantization consistent within the block
                
                j++;
                loc++;
            } while (loc < block_end);

            // Record fixed-length encoding rate for each block.
            signFlag[curr_block] = sign_flag;                                                 // the signs of each fp, represented as a bitfield.
            temp_fixed_rate = max_quant==0 ? 0 : sizeof(int) * 8 - __builtin_clz(max_quant);  // # of bits needed per floating-point in block
            fixedRate[curr_block] = temp_fixed_rate;                                          // store that in global
            cmpData[curr_block] = (unsigned char)temp_fixed_rate;                             // and store it in the compressed data

            // Inner thread prefix-sum.
            thread_ofs += 4 + (temp_fixed_rate ? (32+temp_fixed_rate*32)/8 : 0);                    // # of bytes of memory this block needs in compressed      
        }

        // Store thread ofs to global variable, used for prefix-sum.
        threadOfs[thread_id] = thread_ofs;
        #pragma omp barrier

        // Exclusive prefix-sum.
        unsigned int global_ofs = 0;
        for(int i=0; i<thread_id; i++) global_ofs += threadOfs[i];
        unsigned int cmp_byte_ofs = global_ofs + block_num * NUM_THREADS;          // starting location in compressed memory


        block_start = chunk_start;
        block_end = chunk_start + BLOCKSIZE;
        // Fixed-length encoding and store data to compressed data.
        for(int i=0; i<block_num; i++)
        {
            int curr_block = start_block + i;                    // the current block;
            int temp_fixed_rate = fixedRate[curr_block];         // # of bits needed per fp
            unsigned int sign_flag;
            
            // Commit the first quant regardless
            //*(int*)(&(cmpData[cmp_byte_ofs])) = absQuant[block_start];
			memcpy(&(cmpData[cmp_byte_ofs]), &absQuant[block_start], 4);
            cmp_byte_ofs += 4;
            
            // Operation for each block, if zero block then do nothing.
            if(temp_fixed_rate)
            {
                
                sign_flag = signFlag[curr_block];
                
                // Commit sign information for one block.
				memcpy(&(cmpData[cmp_byte_ofs]), &sign_flag, 4);
                cmp_byte_ofs += 4;
                

                // Commit quant data for one block.
                // unsigned char tmp_char0, tmp_char1, tmp_char2, tmp_char3;
                unsigned int tmp_dat;
                int mask = 1;
                for(int j=0; j<temp_fixed_rate; j++)
                {
                    // Initialization.
                    tmp_dat = 0;

                    int loc = block_start + 1;
                    int fp = 0;
                    do {
                        tmp_dat = (((absQuant[loc     ] & mask) >> j) << (31-fp))
                               |  (((absQuant[loc +  8] & mask) >> j) << (23-fp))
                               |  (((absQuant[loc + 16] & mask) >> j) << (15-fp))
                               |  (((absQuant[loc + 24] & mask) >> j) << ( 7-fp));
                        
                        loc ++;
                        fp ++;
                    } while (fp < 8);
                    
                    // store data
					memcpy(&(cmpData[cmp_byte_ofs]), &tmp_dat, 4);
                    cmp_byte_ofs += 4;
                    
                    mask <<= 1;
                }
            }
        
            // Block initialization.
            block_start += BLOCKSIZE; // the start of this block
            block_end = (block_start+BLOCKSIZE) > chunk_end ? chunk_end : block_start+BLOCKSIZE; // the end of this block, corrected to not overflow out of bounds
                
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
	//printf("ENTERED DECOMPRESS\n");
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
        int block_num = (chunk_size+BLOCKSIZE-1)/BLOCKSIZE;
        int block_start, block_end;
        int start_block = thread_id * block_num;
        unsigned int thread_ofs = 0;

		
		//printf("ENTERED_PARALLEL %d\n", thread_id);

        // Iterate all blocks in current thread.
        for(int i=0; i<block_num; i++)
        {
            // Retrieve fixed-rate for each block in the compressed data.
            int curr_block = start_block + i;
            int temp_fixed_rate = (int)cmpData[curr_block];
            fixedRate[curr_block] = temp_fixed_rate;

            // Inner thread prefix-sum.
            thread_ofs += 4 + (temp_fixed_rate ? (32+temp_fixed_rate*32)/8 : 0);
        }
		
		//printf("FINISHED_SIZING %d\n", thread_id);

        // Store thread ofs to global varaible, used for later global prefix-sum.
        threadOfs[thread_id] = thread_ofs;
        #pragma omp barrier

        // Exclusive prefix-sum.
        unsigned int global_ofs = 0;
        for(int i=0; i<thread_id; i++) global_ofs += threadOfs[i];
        unsigned int cmp_byte_ofs = global_ofs + block_num * NUM_THREADS;

		//printf("FINISHED_SUM %d\n", thread_id);
		
        // Restore decompressed data.
        for(int i=0; i<block_num; i++)
        {
            // Block initialization.
            block_start = start + i * BLOCKSIZE;
            block_end = (block_start+BLOCKSIZE) > end ? end : block_start+BLOCKSIZE;
            int curr_block = start_block + i;
            int temp_fixed_rate = fixedRate[curr_block];
            unsigned int sign_flag = 0;
            int sign_ofs;
            
			
			//printf("START_WRITE %d\n", thread_id);
			int firstQuant = 0;
            // Retrieve first value from block
			{
				int write;
                
				memcpy(&write, &(cmpData[cmp_byte_ofs]), 4);
                firstQuant = write;
				decData[block_start] = (float)write * errorBound * 2;
				cmp_byte_ofs += 4;	
			}

            // Operation for each block, if zero block then do nothing.
            if(temp_fixed_rate)
            {
				
				//printf("EXTRACT ALL BITS %d\n", thread_id);
                // Retrieve sign information for one block.
                //sign_flag = *(unsigned int*)(&(cmpData[cmp_byte_ofs]));
                memcpy(&sign_flag, &(cmpData[cmp_byte_ofs]), 4);
                cmp_byte_ofs += 4;
                
				//printf("EXTRACTED SIGN BITS %d\n", thread_id);

                // Retrieve quant data for one block.
                //unsigned char tmp_char0, tmp_char1, tmp_char2, tmp_char3;
                unsigned int tmp_dat;
                
                for(int j=0; j<temp_fixed_rate; j++)
                {
					
					//printf("ABOUT TO GET DATA BITS %d\n", thread_id);
                    // Initialization.
                    //tmp_dat = *(unsigned int*)(&(cmpData[cmp_byte_ofs]));
					memcpy(&tmp_dat, &(cmpData[cmp_byte_ofs]), 4);
                    cmp_byte_ofs += 4;
					
					//printf("EXTRACTED DATA BITS %d\n", thread_id);
					
                    
                    int loc = 1 + block_start;
                    int fp = 0;
                    do {
                        absQuant[loc    ] |= ((tmp_dat >> (7-fp))  & 1) << j;
                        absQuant[loc + 8] |= ((tmp_dat >> (15-fp)) & 1) << j;
                        absQuant[loc +16] |= ((tmp_dat >> (23-fp)) & 1) << j;
						absQuant[loc +24] |= ((tmp_dat >> (31-fp)) & 1) << j;
                        loc ++;
                        fp ++;
                    } while (fp < 8);
					
					//printf("EXTRACTED ALL DATA BITS %d\n", thread_id);
                }

                // De-quantize and store data back to decompression data.
                int currQuant;
                //printf("Startquant: %d\n",absQuant[block_start]);
                for(int i=0; i<32; i++)
                {
                    int unZigZag = absQuant[i + 1 + block_start];

                    int temp = unZigZag;

                    if (unZigZag!=0) {
                        unZigZag = unZigZag%2 ? (unZigZag+1)/2 : (unZigZag/2);        //maps to original value, 2=>1  3=>-2
                    }
                    
                    int temp2 = unZigZag;

                    //printf("First: %d Curr: %d\n",firstQuant,unZigZag);

                    if(sign_flag & (1 << (31 - i)))
                        currQuant = (firstQuant + unZigZag) * -1;
                    else
                        currQuant = firstQuant + unZigZag;
                    
                    //printf("Decompress: %d - %d = %d\n", absQuant[i + 1 + block_start],absQuant[block_start],currQuant);

                    if ((i+1+block_start)==25636) {
                        printf("Decompressed Quant: %d-- %d - %d = %d\n",temp,firstQuant,currQuant,temp2);
                    }

                    decData[i + 1 + block_start] = (currQuant * errorBound * 2);
                }
            }
            else 
            {
				//printf("COPY START VALUE %d\n", thread_id);
                for(int i = 1 + block_start; i < block_end; i++) {
                    decData[i] = decData[block_start];
                }
            }
			//printf("FINISHED_WRITE %d %d\n", thread_id, i);
        }
		
		
	}
	
	//printf("EXITED DECOMPRESS\n");
}