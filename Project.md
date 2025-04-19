Justin
What I did
in hawkZip_compressor.h compress_kernel recip_precision is the same accross threads, so only need to be computed once, move it outside parallel region