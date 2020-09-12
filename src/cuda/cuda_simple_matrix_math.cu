
#include <iostream>
#include "helper_cuda.h"
#include "helper_math.h"
#include "cuda_simple_matrix_math.h"
#include <cuda_runtime.h>

#ifndef MINF
#define MINF __int_as_float((int)0xff800000)
#endif

#ifndef PINF
#define PINF __int_as_float((int)0x7f800000)
#endif

#ifndef INF
#define INF __int_as_float((int)0x7f800000)
#endif

//////////////////////////////
// float2x2
//////////////////////////////

// class float2x2
inline __device__ __host__ float2x2::float2x2(const float values[4])
{
    m11 = values[0];	m12 = values[1];
    m21 = values[2];	m22 = values[3];
}

// inline __device__ __host__ float2x2::float2x2(const float2x2& other)
// {
//     m11 = other.m11;	m12 = other.m12;
//     m21 = other.m21; 	m22 = other.m22;
// }

inline __device__ __host__ void float2x2::setZero()
{
    m11 = 0.0f;	m12 = 0.0f;
    m21 = 0.0f; m22 = 0.0f;
}

inline __device__ __host__ float2x2 float2x2::getIdentity()
{
    float2x2 res;
    res.setZero();
    res.m11 = res.m22 = 1.0f;
    return res;
}

inline __device__ __host__ float2x2& float2x2::operator=(const float2x2& other)
{
    m11 = other.m11;	m12 = other.m12;
    m21 = other.m21;	m22 = other.m22;
    return *this;
}

inline __device__ __host__ float2x2 float2x2::getInverse()
{
    float2x2 res;
    res.m11 =  m22; res.m12 = -m12;
    res.m21 = -m21; res.m22 =  m11;

    return res*(1.0f/det());
}

inline __device__ __host__ float float2x2::det()
{
    return m11*m22-m21*m12;
}

inline __device__ __host__ float2 float2x2::operator*(const float2& v) const
{
    return make_float2(m11*v.x + m12*v.y, m21*v.x + m22*v.y);
}

//! matrix scalar multiplication
inline __device__ __host__ float2x2 float2x2::operator*(const float t) const
{
    float2x2 res;
    res.m11 = m11 * t;	res.m12 = m12 * t;
    res.m21 = m21 * t;	res.m22 = m22 * t;
    return res;
}

//! matrix matrix multiplication
inline __device__ __host__ float2x2 float2x2::operator*(const float2x2& other) const
{
    float2x2 res;
    res.m11 = m11 * other.m11 + m12 * other.m21;
    res.m12 = m11 * other.m12 + m12 * other.m22;
    res.m21 = m21 * other.m11 + m22 * other.m21;
    res.m22 = m21 * other.m12 + m22 * other.m22;
    return res;
}

//! matrix matrix addition
inline __device__ __host__ float2x2 float2x2::operator+(const float2x2& other) const
{
    float2x2 res;
    res.m11 = m11 + other.m11;
    res.m12 = m12 + other.m12;
    res.m21 = m21 + other.m21;
    res.m22 = m22 + other.m22;
    return res;
}

inline __device__ __host__ float& float2x2::operator()(int i, int j)
{
    return entries2[i][j];
}

inline __device__ __host__ float float2x2::operator()(int i, int j) const
{
    return entries2[i][j];
}

inline __device__ __host__ const float* float2x2::ptr() const {
    return entries;
}
inline __device__ __host__ float* float2x2::ptr() {
    return entries;
}

//////////////////////////////
// float2x3
//////////////////////////////

inline __device__ __host__ float2x3::float2x3(const float values[6])
{
    m11 = values[0];	m12 = values[1];	m13 = values[2];
    m21 = values[3];	m22 = values[4];	m23 = values[5];
}

// inline __device__ __host__ float2x3::float2x3(const float2x3& other)
// {
//     m11 = other.m11;	m12 = other.m12;	m13 = other.m13;
//     m21 = other.m21; 	m22 = other.m22;	m23 = other.m23;
// }

inline __device__ __host__ float2x3& float2x3::operator=(const float2x3 &other)
{
    m11 = other.m11;	m12 = other.m12; m13 = other.m13;
    m21 = other.m21;	m22 = other.m22; m23 = other.m23;
    return *this;
}

inline __device__ __host__ float2 float2x3::operator*(const float3 &v) const
{
    return make_float2(m11*v.x + m12*v.y + m13*v.z, m21*v.x + m22*v.y + m23*v.z);
}

//! matrix scalar multiplication
inline __device__ __host__ float2x3 float2x3::operator*(const float t) const
{
    float2x3 res;
    res.m11 = m11 * t;	res.m12 = m12 * t;	res.m13 = m13 * t;
    res.m21 = m21 * t;	res.m22 = m22 * t;	res.m23 = m23 * t;
    return res;
}

//! matrix scalar division
inline __device__ __host__ float2x3 float2x3::operator/(const float t) const
{
    float2x3 res;
    res.m11 = m11 / t;	res.m12 = m12 / t;	res.m13 = m13 / t;
    res.m21 = m21 / t;	res.m22 = m22 / t;	res.m23 = m23 / t;
    return res;
}

inline __device__ __host__ float& float2x3::operator()(int i, int j)
{
    return entries2[i][j];
}

inline __device__ __host__ float float2x3::operator()(int i, int j) const
{
    return entries2[i][j];
}

inline __device__ __host__ const float* float2x3::ptr() const {
    return entries;
}

inline __device__ __host__ float* float2x3::ptr() {
    return entries;
}

//////////////////////////////
// float3x2
//////////////////////////////

inline __device__ __host__ float3x2::float3x2(const float values[6])
{
    m11 = values[0];	m12 = values[1];
    m21 = values[2];	m22 = values[3];
    m31 = values[4];	m32 = values[5];
}

// inline __device__ __host__ float3x2& operator=(const float3x2& other)
// {
//     m11 = other.m11;	m12 = other.m12;
//     m21 = other.m21;	m22 = other.m22;
//     m31 = other.m31;	m32 = other.m32;
//     return *this;
// }

inline __device__ __host__ float3 float3x2::operator*(const float2& v) const
{
    return make_float3(m11*v.x + m12*v.y, m21*v.x + m22*v.y, m31*v.x + m32*v.y);
}

inline __device__ __host__ float3x2 float3x2::operator*(const float t) const
{
    float3x2 res;
    res.m11 = m11 * t;	res.m12 = m12 * t;
    res.m21 = m21 * t;	res.m22 = m22 * t;
    res.m31 = m31 * t;	res.m32 = m32 * t;
    return res;
}

inline __device__ __host__ float& float3x2::operator()(int i, int j)
{
    return entries2[i][j];
}

inline __device__ __host__ float float3x2::operator()(int i, int j) const
{
    return entries2[i][j];
}

inline __device__ __host__ float2x3 float3x2::getTranspose()
{
    float2x3 res;
    res.m11 = m11; res.m12 = m21; res.m13 = m31;
    res.m21 = m12; res.m22 = m22; res.m23 = m32;
    return res;
}

inline __device__ __host__ const float* float3x2::ptr() const {
    return entries;
}

inline __device__ __host__ float* float3x2::ptr() {
    return entries;
}

inline __device__ __host__ float2x2 matMul(const float2x3& m0, const float3x2& m1)
{
	float2x2 res;
	res.m11 = m0.m11*m1.m11+m0.m12*m1.m21+m0.m13*m1.m31;
	res.m12 = m0.m11*m1.m12+m0.m12*m1.m22+m0.m13*m1.m32;
	res.m21 = m0.m21*m1.m11+m0.m22*m1.m21+m0.m23*m1.m31;
	res.m22 = m0.m21*m1.m12+m0.m22*m1.m22+m0.m23*m1.m32;
	return res;
}

//////////////////////////////
// float3x3
//////////////////////////////

inline __device__ __host__ float3x3::float3x3(const float values[9]) {
    m11 = values[0];	m12 = values[1];	m13 = values[2];
    m21 = values[3];	m22 = values[4];	m23 = values[5];
    m31 = values[6];	m32 = values[7];	m33 = values[8];
}

// inline __device__ __host__ float3x3(const float3x3& other) {
//     m11 = other.m11;	m12 = other.m12;	m13 = other.m13;
//     m21 = other.m21;	m22 = other.m22;	m23 = other.m23;
//     m31 = other.m31;	m32 = other.m32;	m33 = other.m33;
// }

inline __device__ __host__ float3x3::float3x3(const float2x2& other) {
    m11 = other.m11;	m12 = other.m12;	m13 = 0.0;
    m21 = other.m21;	m22 = other.m22;	m23 = 0.0;
    m31 = 0.0;			m32 = 0.0;			m33 = 0.0;
}

inline __device__ __host__ float3x3& float3x3::operator=(const float3x3 &other) {
    m11 = other.m11;	m12 = other.m12;	m13 = other.m13;
    m21 = other.m21;	m22 = other.m22;	m23 = other.m23;
    m31 = other.m31;	m32 = other.m32;	m33 = other.m33;
    return *this;
}

inline __device__ __host__ float& float3x3::operator()(int i, int j) {
    return entries2[i][j];
}

inline __device__ __host__ float float3x3::operator()(int i, int j) const {
    return entries2[i][j];
}

inline __device__ __host__  void float3x3::swap(float& v0, float& v1) {
    float tmp = v0;
    v0 = v1;
    v1 = tmp;
}

inline __device__ __host__ void float3x3::transpose() {
    swap(m12, m21);
    swap(m13, m31);
    swap(m23, m32);
}

inline __device__ __host__ float3x3 float3x3::getTranspose() const {
    float3x3 ret = *this;
    ret.transpose();
    return ret;
}

//! inverts the matrix
inline __device__ __host__ void float3x3::invert() {
    *this = getInverse();
}

//! computes the inverse of the matrix; the result is returned
inline __device__ __host__ float3x3 float3x3::getInverse() const {
    float3x3 res;
    res.entries[0] = entries[4]*entries[8] - entries[5]*entries[7];
    res.entries[1] = -entries[1]*entries[8] + entries[2]*entries[7];
    res.entries[2] = entries[1]*entries[5] - entries[2]*entries[4];

    res.entries[3] = -entries[3]*entries[8] + entries[5]*entries[6];
    res.entries[4] = entries[0]*entries[8] - entries[2]*entries[6];
    res.entries[5] = -entries[0]*entries[5] + entries[2]*entries[3];

    res.entries[6] = entries[3]*entries[7] - entries[4]*entries[6];
    res.entries[7] = -entries[0]*entries[7] + entries[1]*entries[6];
    res.entries[8] = entries[0]*entries[4] - entries[1]*entries[3];
    float nom = 1.0f/det();
    return res * nom;
}

inline __device__ __host__ void float3x3::setZero(float value) {
    m11 = m12 = m13 = value;
    m21 = m22 = m23 = value;
    m31 = m32 = m33 = value;
}

inline __device__ __host__ float float3x3::det() const {
    return
        + m11*m22*m33
        + m12*m23*m31
        + m13*m21*m32
        - m31*m22*m13
        - m32*m23*m11
        - m33*m21*m12;
}

inline __device__ __host__ float3 float3x3::getRow(unsigned int i) {
    return make_float3(entries[3*i+0], entries[3*i+1], entries[3*i+2]);
}

inline __device__ __host__ void float3x3::setRow(unsigned int i, float3& r) {
    entries[3*i+0] = r.x;
    entries[3*i+1] = r.y;
    entries[3*i+2] = r.z;
}

inline __device__ __host__ void float3x3::normalizeRows()
{
    //#pragma unroll 3
    for(unsigned int i = 0; i<3; i++)
    {
        float3 r = getRow(i);
        r/=length(r);
        setRow(i, r);
    }
}

//! computes the product of two matrices (result stored in this)
inline __device__ __host__ void float3x3::mult(const float3x3 &other) {
    float3x3 res;
    res.m11 = m11 * other.m11 + m12 * other.m21 + m13 * other.m31;
    res.m12 = m11 * other.m12 + m12 * other.m22 + m13 * other.m32;
    res.m13 = m11 * other.m13 + m12 * other.m23 + m13 * other.m33;

    res.m21 = m21 * other.m11 + m22 * other.m21 + m23 * other.m31;
    res.m22 = m21 * other.m12 + m22 * other.m22 + m23 * other.m32;
    res.m23 = m21 * other.m13 + m22 * other.m23 + m23 * other.m33;

    res.m31 = m21 * other.m11 + m32 * other.m21 + m33 * other.m31;
    res.m32 = m21 * other.m12 + m32 * other.m22 + m33 * other.m32;
    res.m33 = m21 * other.m13 + m32 * other.m23 + m33 * other.m33;
    *this = res;
}

//! computes the sum of two matrices (result stored in this)
inline __device__ __host__ void float3x3::add(const float3x3 &other) {
    m11 += other.m11;	m12 += other.m12;	m13 += other.m13;
    m21 += other.m21;	m22 += other.m22;	m23 += other.m23;
    m31 += other.m31;	m32 += other.m32;	m33 += other.m33;
}

//! standard matrix matrix multiplication
inline __device__ __host__ float3x3 float3x3::operator*(const float3x3 &other) const {
    float3x3 res;
    res.m11 = m11 * other.m11 + m12 * other.m21 + m13 * other.m31;
    res.m12 = m11 * other.m12 + m12 * other.m22 + m13 * other.m32;
    res.m13 = m11 * other.m13 + m12 * other.m23 + m13 * other.m33;

    res.m21 = m21 * other.m11 + m22 * other.m21 + m23 * other.m31;
    res.m22 = m21 * other.m12 + m22 * other.m22 + m23 * other.m32;
    res.m23 = m21 * other.m13 + m22 * other.m23 + m23 * other.m33;

    res.m31 = m31 * other.m11 + m32 * other.m21 + m33 * other.m31;
    res.m32 = m31 * other.m12 + m32 * other.m22 + m33 * other.m32;
    res.m33 = m31 * other.m13 + m32 * other.m23 + m33 * other.m33;
    return res;
}

//! standard matrix matrix multiplication
inline __device__ __host__ float3x2 float3x3::operator*(const float3x2 &other) const {
    float3x2 res;
    res.m11 = m11 * other.m11 + m12 * other.m21 + m13 * other.m31;
    res.m12 = m11 * other.m12 + m12 * other.m22 + m13 * other.m32;

    res.m21 = m21 * other.m11 + m22 * other.m21 + m23 * other.m31;
    res.m22 = m21 * other.m12 + m22 * other.m22 + m23 * other.m32;

    res.m31 = m31 * other.m11 + m32 * other.m21 + m33 * other.m31;
    res.m32 = m31 * other.m12 + m32 * other.m22 + m33 * other.m32;
    return res;
}

inline __device__ __host__ float3 float3x3::operator*(const float3 &v) const {
    return make_float3(
        m11*v.x + m12*v.y + m13*v.z,
        m21*v.x + m22*v.y + m23*v.z,
        m31*v.x + m32*v.y + m33*v.z
        );
}

inline __device__ __host__ float3x3 float3x3::operator*(const float t) const {
    float3x3 res;
    res.m11 = m11 * t;		res.m12 = m12 * t;		res.m13 = m13 * t;
    res.m21 = m21 * t;		res.m22 = m22 * t;		res.m23 = m23 * t;
    res.m31 = m31 * t;		res.m32 = m32 * t;		res.m33 = m33 * t;
    return res;
}


inline __device__ __host__ float3x3 float3x3::operator+(const float3x3 &other) const {
    float3x3 res;
    res.m11 = m11 + other.m11;	res.m12 = m12 + other.m12;	res.m13 = m13 + other.m13;
    res.m21 = m21 + other.m21;	res.m22 = m22 + other.m22;	res.m23 = m23 + other.m23;
    res.m31 = m31 + other.m31;	res.m32 = m32 + other.m32;	res.m33 = m33 + other.m33;
    return res;
}

inline __device__ __host__ float3x3 float3x3::operator-(const float3x3 &other) const {
    float3x3 res;
    res.m11 = m11 - other.m11;	res.m12 = m12 - other.m12;	res.m13 = m13 - other.m13;
    res.m21 = m21 - other.m21;	res.m22 = m22 - other.m22;	res.m23 = m23 - other.m23;
    res.m31 = m31 - other.m31;	res.m32 = m32 - other.m32;	res.m33 = m33 - other.m33;
    return res;
}

inline __device__ __host__ float3x3 float3x3::getIdentity() {
    float3x3 res;
    res.setZero();
    res.m11 = res.m22 = res.m33 = 1.0f;
    return res;
}

inline __device__ __host__ float3x3 float3x3::getZeroMatrix() {
    float3x3 res;
    res.setZero();
    return res;
}

inline __device__ __host__ float3x3 float3x3::getDiagonalMatrix(float diag) {
    float3x3 res;
    res.m11 = diag;		res.m12 = 0.0f;		res.m13 = 0.0f;
    res.m21 = 0.0f;		res.m22 = diag;		res.m23 = 0.0f;
    res.m31 = 0.0f;		res.m32 = 0.0f;		res.m33 = diag;
    return res;
}

inline __device__ __host__  float3x3 float3x3::tensorProduct(const float3 &v, const float3 &vt) {
    float3x3 res;
    res.m11 = v.x * vt.x;	res.m12 = v.x * vt.y;	res.m13 = v.x * vt.z;
    res.m21 = v.y * vt.x;	res.m22 = v.y * vt.y;	res.m23 = v.y * vt.z;
    res.m31 = v.z * vt.x;	res.m32 = v.z * vt.y;	res.m33 = v.z * vt.z;
    return res;
}

inline __device__ __host__ const float* float3x3::ptr() const {
    return entries;
}

inline __device__ __host__ float* float3x3::ptr() {
    return entries;
}

inline __device__ __host__ float2x3 matMul(const float2x3& m0, const float3x3& m1)
{
	float2x3 res;
	res.m11 = m0.m11*m1.m11+m0.m12*m1.m21+m0.m13*m1.m31;
	res.m12 = m0.m11*m1.m12+m0.m12*m1.m22+m0.m13*m1.m32;
	res.m13 = m0.m11*m1.m13+m0.m12*m1.m23+m0.m13*m1.m33;

	res.m21 = m0.m21*m1.m11+m0.m22*m1.m21+m0.m23*m1.m31;
	res.m22 = m0.m21*m1.m12+m0.m22*m1.m22+m0.m23*m1.m32;
	res.m23 = m0.m21*m1.m13+m0.m22*m1.m23+m0.m23*m1.m33;
	return res;
}

// (1x2) row matrix as float2
inline __device__ __host__ float3 matMul(const float2& m0, const float2x3& m1)
{
	float3 res;
	res.x = m0.x*m1.m11+m0.y*m1.m21;
	res.y = m0.x*m1.m12+m0.y*m1.m22;
	res.z = m0.x*m1.m13+m0.y*m1.m23;

	return res;
}

//////////////////////////////
// float3x4
//////////////////////////////

inline __device__ __host__ float3x4::float3x4(const float values[12]) {
    m11 = values[0];	m12 = values[1];	m13 = values[2];	m14 = values[3];
    m21 = values[4];	m22 = values[5];	m23 = values[6];	m24 = values[7];
    m31 = values[8];	m32 = values[9];	m33 = values[10];	m34 = values[11];
}

// inline __device__ __host__ float3x4::float3x4(const float3x4& other) {
//     m11 = other.m11;	m12 = other.m12;	m13 = other.m13;	m14 = other.m14;
//     m21 = other.m21;	m22 = other.m22;	m23 = other.m23;	m24 = other.m24;
//     m31 = other.m31;	m32 = other.m32;	m33 = other.m33;	m34 = other.m34;
// }

inline __device__ __host__ float3x4::float3x4(const float3x3& other) {
    m11 = other.m11;	m12 = other.m12;	m13 = other.m13;	m14 = 0.0f;
    m21 = other.m21;	m22 = other.m22;	m23 = other.m23;	m24 = 0.0f;
    m31 = other.m31;	m32 = other.m32;	m33 = other.m33;	m34 = 0.0f;
}

inline __device__ __host__ float3x4 float3x4::operator=(const float3x4 &other) {
    m11 = other.m11;	m12 = other.m12;	m13 = other.m13;	m14 = other.m14;
    m21 = other.m21;	m22 = other.m22;	m23 = other.m23;	m24 = other.m24;
    m31 = other.m31;	m32 = other.m32;	m33 = other.m33;	m34 = other.m34;
    return *this;
}

inline __device__ __host__ float3x4& float3x4::operator=(const float3x3 &other) {
    m11 = other.m11;	m12 = other.m12;	m13 = other.m13;	m14 = 0.0f;
    m21 = other.m21;	m22 = other.m22;	m23 = other.m23;	m24 = 0.0f;
    m31 = other.m31;	m32 = other.m32;	m33 = other.m33;	m34 = 0.0f;
    return *this;
}

//! assumes the last line of the matrix implicitly to be (0,0,0,1)
inline __device__ __host__ float4 float3x4::operator*(const float4 &v) const {
    return make_float4(
        m11*v.x + m12*v.y + m13*v.z + m14*v.w,
        m21*v.x + m22*v.y + m23*v.z + m24*v.w,
        m31*v.x + m32*v.y + m33*v.z + m34*v.w,
        v.w
        );
}

//! assumes an implicit 1 in w component of the input vector
inline __device__ __host__ float3 float3x4::operator*(const float3 &v) const {
    return make_float3(
        m11*v.x + m12*v.y + m13*v.z + m14,
        m21*v.x + m22*v.y + m23*v.z + m24,
        m31*v.x + m32*v.y + m33*v.z + m34
        );
}

//! matrix scalar multiplication
inline __device__ __host__ float3x4 float3x4::operator*(const float t) const {
    float3x4 res;
    res.m11 = m11 * t;		res.m12 = m12 * t;		res.m13 = m13 * t;		res.m14 = m14 * t;
    res.m21 = m21 * t;		res.m22 = m22 * t;		res.m23 = m23 * t;		res.m24 = m24 * t;
    res.m31 = m31 * t;		res.m32 = m32 * t;		res.m33 = m33 * t;		res.m34 = m34 * t;
    return res;
}
inline __device__ __host__ float3x4& float3x4::operator*=(const float t) {
    *this = *this * t;
    return *this;
}

//! matrix scalar division
inline __device__ __host__ float3x4 float3x4::operator/(const float t) const {
    float3x4 res;
    res.m11 = m11 / t;		res.m12 = m12 / t;		res.m13 = m13 / t;		res.m14 = m14 / t;
    res.m21 = m21 / t;		res.m22 = m22 / t;		res.m23 = m23 / t;		res.m24 = m24 / t;
    res.m31 = m31 / t;		res.m32 = m32 / t;		res.m33 = m33 / t;		res.m34 = m34 / t;
    return res;
}
inline __device__ __host__ float3x4& float3x4::operator/=(const float t) {
    *this = *this / t;
    return *this;
}

//! assumes the last line of the matrix implicitly to be (0,0,0,1)
inline __device__ __host__ float3x4 float3x4::operator*(const float3x4 &other) const {
    float3x4 res;
    res.m11 = m11*other.m11 + m12*other.m21 + m13*other.m31;  
    res.m12 = m11*other.m12 + m12*other.m22 + m13*other.m32;  
    res.m13 = m11*other.m13 + m12*other.m23 + m13*other.m33; 
    res.m14 = m11*other.m14 + m12*other.m24 + m13*other.m34 + m14;
    
    res.m21 = m21*other.m11 + m22*other.m21 + m23*other.m31;  
    res.m22 = m21*other.m12 + m22*other.m22 + m23*other.m32;  
    res.m23 = m21*other.m13 + m22*other.m23 + m23*other.m33; 
    res.m24 = m21*other.m14 + m22*other.m24 + m23*other.m34 + m24;

    res.m31 = m31*other.m11 + m32*other.m21 + m33*other.m31;  
    res.m32 = m31*other.m12 + m32*other.m22 + m33*other.m32;  
    res.m33 = m31*other.m13 + m32*other.m23 + m33*other.m33; 
    res.m34 = m31*other.m14 + m32*other.m24 + m33*other.m34 + m34;

    return res;
}

//! assumes the last line of the matrix implicitly to be (0,0,0,1); and a (0,0,0) translation of other
inline __device__ __host__ float3x4 float3x4::operator*(const float3x3 &other) const {
    float3x4 res;
    res.m11 = m11*other.m11 + m12*other.m21 + m13*other.m31;  
    res.m12 = m11*other.m12 + m12*other.m22 + m13*other.m32;  
    res.m13 = m11*other.m13 + m12*other.m23 + m13*other.m33; 
    res.m14 = m14;

    res.m21 = m21*other.m11 + m22*other.m21 + m23*other.m31;  
    res.m22 = m21*other.m12 + m22*other.m22 + m23*other.m32;  
    res.m23 = m21*other.m13 + m22*other.m23 + m23*other.m33; 
    res.m24 = m24;

    res.m31 = m31*other.m11 + m32*other.m21 + m33*other.m31;  
    res.m32 = m31*other.m12 + m32*other.m22 + m33*other.m32;  
    res.m33 = m31*other.m13 + m32*other.m23 + m33*other.m33; 
    res.m34 = m34;

    return res;
}

inline __device__ __host__ float& float3x4::operator()(int i, int j) {
    return entries2[i][j];
}

inline __device__ __host__ float float3x4::operator()(int i, int j) const {
    return entries2[i][j];
}

//! returns the translation part of the matrix
inline __device__ __host__ float3 float3x4::getTranslation() {
    return make_float3(m14, m24, m34);
}

//! sets only the translation part of the matrix (other values remain unchanged)
inline __device__ __host__ void float3x4::setTranslation(const float3 &t) {
    m14 = t.x;
    m24 = t.y;
    m34 = t.z;
}

//! returns the 3x3 part of the matrix
inline __device__ __host__ float3x3 float3x4::getFloat3x3() {
    float3x3 ret;
    ret.m11 = m11;	ret.m12 = m12;	ret.m13 = m13;
    ret.m21 = m21;	ret.m22 = m22;	ret.m23 = m23;
    ret.m31 = m31;	ret.m32 = m32;	ret.m33 = m33;
    return ret;
}

//! sets the 3x3 part of the matrix (other values remain unchanged)
inline __device__ __host__ void float3x4::setFloat3x3(const float3x3 &other) {
    m11 = other.m11;	m12 = other.m12;	m13 = other.m13;
    m21 = other.m21;	m22 = other.m22;	m23 = other.m23;
    m31 = other.m31;	m32 = other.m32;	m33 = other.m33;
}

//! inverts the matrix
inline __device__ __host__ void float3x4::inverse() {
    *this = getInverse();
}

//! computes the inverse of the matrix
inline __device__ __host__ float3x4 float3x4::getInverse() {
    float3x3 A = getFloat3x3();
    A.invert();
    float3 t = getTranslation();
    t = A*t;

    float3x4 ret;
    ret.setFloat3x3(A);
    ret.setTranslation(make_float3(-t.x, -t.y, -t.z));	//float3 doesn't have unary '-'... thank you cuda
    return ret;
}

//! prints the matrix; only host	
__host__ void float3x4::print() {
    std::cout <<
        m11 << " " << m12 << " " << m13 << " " << m14 << std::endl <<
        m21 << " " << m22 << " " << m23 << " " << m24 << std::endl <<
        m31 << " " << m32 << " " << m33 << " " << m34 << std::endl <<
        std::endl;
}

inline __device__ __host__ const float* float3x4::ptr() const {
    return entries;
}

inline __device__ __host__ float* float3x4::ptr() {
    return entries;
}

//////////////////////////////
// float4x4
//////////////////////////////

inline __device__ __host__ float4x4::float4x4(const float values[16]) {
    m11 = values[0];	m12 = values[1];	m13 = values[2];	m14 = values[3];
    m21 = values[4];	m22 = values[5];	m23 = values[6];	m24 = values[7];
    m31 = values[8];	m32 = values[9];	m33 = values[10];	m34 = values[11];
    m41 = values[12];	m42 = values[13];	m43 = values[14];	m44 = values[15];
}

//implicitly assumes last line to (0,0,0,1)
inline __device__ __host__ float4x4::float4x4(const float3x4& other) {
    m11 = other.m11;	m12 = other.m12;	m13 = other.m13;	m14 = other.m14;
    m21 = other.m21;	m22 = other.m22;	m23 = other.m23;	m24 = other.m24;
    m31 = other.m31;	m32 = other.m32;	m33 = other.m33;	m34 = other.m34;
    m41 = 0.0f;			m42 = 0.0f;			m43 = 0.0f;			m44 = 1.0f;
}

inline __device__ __host__ float4x4::float4x4(const float3x3& other) {
    m11 = other.m11;	m12 = other.m12;	m13 = other.m13;	m14 = 0.0f;
    m21 = other.m21;	m22 = other.m22;	m23 = other.m23;	m24 = 0.0f;
    m31 = other.m31;	m32 = other.m32;	m33 = other.m33;	m34 = 0.0f;
    m41 = 0.0f;			m42 = 0.0f;			m43 = 0.0f;			m44 = 1.0f;
}

// inline __device__ __host__ float4x4 float4x4::operator=(const float4x4 &other) {
//     m11 = other.m11;	m12 = other.m12;	m13 = other.m13;	m14 = other.m14;
//     m21 = other.m21;	m22 = other.m22;	m23 = other.m23;	m24 = other.m24;
//     m31 = other.m31;	m32 = other.m32;	m33 = other.m33;	m34 = other.m34;
//     m41 = other.m41;	m42 = other.m42;	m43 = other.m43;	m44 = other.m44;
//     return *this;
// }

// inline __device__ __host__ float4x4 float4x4::operator=(const float3x4 &other) {
//     m11 = other.m11;	m12 = other.m12;	m13 = other.m13;	m14 = other.m14;
//     m21 = other.m21;	m22 = other.m22;	m23 = other.m23;	m24 = other.m24;
//     m31 = other.m31;	m32 = other.m32;	m33 = other.m33;	m34 = other.m34;
//     m41 = 0.0f;			m42 = 0.0f;			m43 = 0.0f;			m44 = 1.0f;
//     return *this;
// }

// inline __device__ __host__ float4x4& float4x4::operator=(const float3x3 &other) {
//     m11 = other.m11;	m12 = other.m12;	m13 = other.m13;	m14 = 0.0f;
//     m21 = other.m21;	m22 = other.m22;	m23 = other.m23;	m24 = 0.0f;
//     m31 = other.m31;	m32 = other.m32;	m33 = other.m33;	m34 = 0.0f;
//     m41 = 0.0f;			m42 = 0.0f;			m43 = 0.0f;			m44 = 1.0f;
//     return *this;
// }

//! not tested
inline __device__ __host__ float4x4 float4x4::operator*(const float4x4 &other) const {
    float4x4 res;
    res.m11 = m11*other.m11 + m12*other.m21 + m13*other.m31 + m14*other.m41;  
    res.m12 = m11*other.m12 + m12*other.m22 + m13*other.m32 + m14*other.m42;  
    res.m13 = m11*other.m13 + m12*other.m23 + m13*other.m33 + m14*other.m43; 
    res.m14 = m11*other.m14 + m12*other.m24 + m13*other.m34 + m14*other.m44;

    res.m21 = m21*other.m11 + m22*other.m21 + m23*other.m31 + m24*other.m41;  
    res.m22 = m21*other.m12 + m22*other.m22 + m23*other.m32 + m24*other.m42;  
    res.m23 = m21*other.m13 + m22*other.m23 + m23*other.m33 + m24*other.m43; 
    res.m24 = m21*other.m14 + m22*other.m24 + m23*other.m34 + m24*other.m44;

    res.m31 = m31*other.m11 + m32*other.m21 + m33*other.m31 + m34*other.m41;  
    res.m32 = m31*other.m12 + m32*other.m22 + m33*other.m32 + m34*other.m42;  
    res.m33 = m31*other.m13 + m32*other.m23 + m33*other.m33 + m34*other.m43; 
    res.m34 = m31*other.m14 + m32*other.m24 + m33*other.m34 + m34*other.m44;

    res.m41 = m41*other.m11 + m42*other.m21 + m43*other.m31 + m44*other.m41;  
    res.m42 = m41*other.m12 + m42*other.m22 + m43*other.m32 + m44*other.m42;  
    res.m43 = m41*other.m13 + m42*other.m23 + m43*other.m33 + m44*other.m43; 
    res.m44 = m41*other.m14 + m42*other.m24 + m43*other.m34 + m44*other.m44;

    return res;
}

// untested
inline __device__ __host__ float4 float4x4::operator*(const float4& v) const
{
    return make_float4(
        m11*v.x + m12*v.y + m13*v.z + m14*v.w,
        m21*v.x + m22*v.y + m23*v.z + m24*v.w,
        m31*v.x + m32*v.y + m33*v.z + m34*v.w,
        m41*v.x + m42*v.y + m43*v.z + m44*v.w
        );
}


inline __device__ __host__ float& float4x4::operator()(int i, int j) {
    return entries2[i][j];
}

inline __device__ __host__ float float4x4::operator()(int i, int j) const {
    return entries2[i][j];
}


inline __device__ __host__  void float4x4::swap(float& v0, float& v1) {
    float tmp = v0;
    v0 = v1;
    v1 = tmp;
}

inline __device__ __host__ void float4x4::transpose() {
    swap(m12, m21);
    swap(m13, m31);
    swap(m23, m32);
    swap(m41, m14);
    swap(m42, m24);
    swap(m43, m34);
}

inline __device__ __host__ float4x4 float4x4::getTranspose() const {
    float4x4 ret = *this;
    ret.transpose();
    return ret;
}

inline __device__ __host__ void float4x4::invert() {
    *this = getInverse();
}



//! returns the 3x3 part of the matrix
inline __device__ __host__ float3x3 float4x4::getFloat3x3() {
    float3x3 ret;
    ret.m11 = m11;	ret.m12 = m12;	ret.m13 = m13;
    ret.m21 = m21;	ret.m22 = m22;	ret.m23 = m23;
    ret.m31 = m31;	ret.m32 = m32;	ret.m33 = m33;
    return ret;
}

//! sets the 3x3 part of the matrix (other values remain unchanged)
inline __device__ __host__ void float4x4::setFloat3x3(const float3x3 &other) {
    m11 = other.m11;	m12 = other.m12;	m13 = other.m13;
    m21 = other.m21;	m22 = other.m22;	m23 = other.m23;
    m31 = other.m31;	m32 = other.m32;	m33 = other.m33;
}

//! sets the 4x4 part of the matrix to identity
inline __device__ __host__ void float4x4::setValue(float v)
{
    m11 = v;	m12 = v;	m13 = v;	m14 = v;
    m21 = v;	m22 = v;	m23 = v;	m24 = v;
    m31 = v;	m32 = v;	m33 = v;	m34 = v;
    m41 = v;	m42 = v;	m43 = v;	m44 = v;
}

//! returns the 3x4 part of the matrix
inline __device__ __host__ float3x4 float4x4::getFloat3x4() {
    float3x4 ret;
    ret.m11 = m11;	ret.m12 = m12;	ret.m13 = m13;	ret.m14 = m14;
    ret.m21 = m21;	ret.m22 = m22;	ret.m23 = m23;	ret.m24 = m24;
    ret.m31 = m31;	ret.m32 = m32;	ret.m33 = m33;	ret.m34 = m34;
    return ret;
}

//! sets the 3x4 part of the matrix (other values remain unchanged)
inline __device__ __host__ void float4x4::setFloat3x4(const float3x4 &other) {
    m11 = other.m11;	m12 = other.m12;	m13 = other.m13;	m14 = other.m14;
    m21 = other.m21;	m22 = other.m22;	m23 = other.m23;	m24 = other.m24;
    m31 = other.m31;	m32 = other.m32;	m33 = other.m33;	m34 = other.m34;
}

inline __device__ __host__ const float* float4x4::ptr() const {
    return entries;
}

inline __device__ __host__ float* float4x4::ptr() {
    return entries;
}



