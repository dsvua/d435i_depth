/*
 * safe_call.hpp
 *
 *  Created on: 1 Sep 2014
 *      Author: thomas
 */

#ifndef SAFE_CALL_H_2_
#define SAFE_CALL_H_2_

#include <cuda_runtime_api.h>
#include <cstdlib>
#include "init.h"

#if defined(__GNUC__)
    #define cudaSafeCall(expr)  ___cudaSafeCall(expr, __FILE__, __LINE__, __func__)
#else /* defined(__CUDACC__) || defined(__MSVC__) */
    #define cudaSafeCall(expr)  ___cudaSafeCall(expr, __FILE__, __LINE__)
#endif

static inline void ___cudaSafeCall(cudaError_t err2, const char *file, const int line, const char *func = "")
{
    if (cudaSuccess != err2)
        error2(cudaGetErrorString(err2), file, line, func);
}

#endif /* SAFE_CALL_HPP_2_ */
