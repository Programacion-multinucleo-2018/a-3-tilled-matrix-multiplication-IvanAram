#pragma once
#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number){
	if(err!=cudaSuccess){
		fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CALL(call,msg) _safe_cuda_call(call,msg,__FILE__,__LINE__)

#endif
