#ifndef CUTIL_MATH_H
#define CUTIL_MATH_H


////////////////////////////////////////////////////////////////////////////////
// missing functions from helper_math.h
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ int sign(float val) {
	return (float(0) < val) - (val < float(0));
}

inline __host__ __device__ int4 sign(const float4& v) {
	return make_int4(sign(v.x), sign(v.y), sign(v.z), sign(v.w));
}

inline __host__ __device__ int3 sign(const float3& v) {
	return make_int3(sign(v.x), sign(v.y), sign(v.z));
}

inline __host__ __device__ int2 sign(const float2& v) {
	return make_int2(sign(v.x), sign(v.y));
}
#endif