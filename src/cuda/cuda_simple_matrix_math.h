#pragma once

#ifndef _CUDA_SIMPLE_MATRIX_UTIL_
#define _CUDA_SIMPLE_MATRIX_UTIL_

#include <iostream>
#include <cuda_runtime.h>
#include "helper_math.h"

#ifndef cudaAssert
#define cudaAssert(condition) if (!(condition)) { printf("ASSERT: %s %s\n", #condition, __FILE__); }
#endif

//////////////////////////////
// float2x2
//////////////////////////////

class float2x2
{
public:

	inline __device__ __host__ float2x2()
	{
	}

	inline __device__ __host__ float2x2(const float values[4]);

	inline __device__ __host__ float2x2(const float2x2& other)
	{
		m11 = other.m11;	m12 = other.m12;
		m21 = other.m21; 	m22 = other.m22;
	}

	inline __device__ __host__ void setZero();

	static inline __device__ __host__ float2x2 getIdentity();

	inline __device__ __host__ float2x2& operator=(const float2x2& other);

	inline __device__ __host__ float2x2 getInverse();

	inline __device__ __host__ float det();

	inline __device__ __host__ float2 operator*(const float2& v) const;

	//! matrix scalar multiplication
	inline __device__ __host__ float2x2 operator*(const float t) const;

	//! matrix matrix multiplication
	inline __device__ __host__ float2x2 operator*(const float2x2& other) const;

	//! matrix matrix addition
	inline __device__ __host__ float2x2 operator+(const float2x2& other) const;
	
	inline __device__ __host__ float& operator()(int i, int j);

	inline __device__ __host__ float operator()(int i, int j) const;

	inline __device__ __host__ const float* ptr() const;

	inline __device__ __host__ float* ptr();

	union
	{
		struct
		{
			float m11; float m12;
			float m21; float m22;
		};

		float entries[4];
		float entries2[2][2];
	};
};

//////////////////////////////
// float2x3
//////////////////////////////

class float2x3
{
public:

	union
	{
		struct
		{
			float m11; float m12; float m13;
			float m21; float m22; float m23;
		};

		float entries[6];
		float entries2[3][2];
	};

	inline __device__ __host__ float2x3()
	{
	}

	inline __device__ __host__ float2x3(const float values[6]);

	inline __device__ __host__ float2x3(const float2x3& other)
	{
		m11 = other.m11;	m12 = other.m12;	m13 = other.m13;
		m21 = other.m21; 	m22 = other.m22;	m23 = other.m23;
	}

	inline __device__ __host__ float2x3& operator=(const float2x3 &other);

	inline __device__ __host__ float2 operator*(const float3 &v) const;

	//! matrix scalar multiplication
	inline __device__ __host__ float2x3 operator*(const float t) const;

	//! matrix scalar division
	inline __device__ __host__ float2x3 operator/(const float t) const;
	
	inline __device__ __host__ float& operator()(int i, int j);

	inline __device__ __host__ float operator()(int i, int j) const;

	inline __device__ __host__ const float* ptr() const;

	inline __device__ __host__ float* ptr();

};

//////////////////////////////
// float3x2
//////////////////////////////

class float3x2
{
public:

	inline __device__ __host__ float3x2()
	{
	}

	inline __device__ __host__ float3x2(const float values[6]);

	inline __device__ __host__ float3x2& operator=(const float3x2& other)
	{
		m11 = other.m11;	m12 = other.m12;
		m21 = other.m21;	m22 = other.m22;
		m31 = other.m31;	m32 = other.m32;
		return *this;
	}

	inline __device__ __host__ float3 operator*(const float2& v) const;

	inline __device__ __host__ float3x2 operator*(const float t) const;

	inline __device__ __host__ float& operator()(int i, int j);

	inline __device__ __host__ float operator()(int i, int j) const;

	inline __device__ __host__ float2x3 getTranspose();

	inline __device__ __host__ const float* ptr() const;

	inline __device__ __host__ float* ptr();

	union
	{
		struct
		{
			float m11; float m12;
			float m21; float m22;
			float m31; float m32;
		};

		float entries[6];
		float entries2[3][2];
	};
};

inline __device__ __host__ float2x2 matMul(const float2x3& m0, const float3x2& m1);

//////////////////////////////
// float3x3
//////////////////////////////

class float3x3 {
public:
	inline __device__ __host__ float3x3() {}

	inline __device__ __host__ float3x3(const float values[9]);

	inline __device__ __host__ float3x3(const float3x3& other) {
		m11 = other.m11;	m12 = other.m12;	m13 = other.m13;
		m21 = other.m21;	m22 = other.m22;	m23 = other.m23;
		m31 = other.m31;	m32 = other.m32;	m33 = other.m33;
	}

	explicit inline __device__ __host__ float3x3(const float2x2& other);

	inline __device__ __host__ float3x3& operator=(const float3x3 &other);

	inline __device__ __host__ float& operator()(int i, int j);

	inline __device__ __host__ float operator()(int i, int j) const;

	static inline __device__ __host__  void swap(float& v0, float& v1);

	inline __device__ __host__ void transpose();

	inline __device__ __host__ float3x3 getTranspose() const;

	//! inverts the matrix
	inline __device__ __host__ void invert();

	//! computes the inverse of the matrix; the result is returned
	inline __device__ __host__ float3x3 getInverse() const;

	inline __device__ __host__ void setZero(float value = 0.0f);

	inline __device__ __host__ float det() const;

	inline __device__ __host__ float3 getRow(unsigned int i);

	inline __device__ __host__ void setRow(unsigned int i, float3& r);

	inline __device__ __host__ void normalizeRows();

	//! computes the product of two matrices (result stored in this)
	inline __device__ __host__ void mult(const float3x3 &other);

	//! computes the sum of two matrices (result stored in this)
	inline __device__ __host__ void add(const float3x3 &other);

	//! standard matrix matrix multiplication
	inline __device__ __host__ float3x3 operator*(const float3x3 &other) const;

	//! standard matrix matrix multiplication
	inline __device__ __host__ float3x2 operator*(const float3x2 &other) const;

	inline __device__ __host__ float3 operator*(const float3 &v) const;

	inline __device__ __host__ float3x3 operator*(const float t) const;

	inline __device__ __host__ float3x3 operator+(const float3x3 &other) const;

	inline __device__ __host__ float3x3 operator-(const float3x3 &other) const;

	static inline __device__ __host__ float3x3 getIdentity();

	static inline __device__ __host__ float3x3 getZeroMatrix();

	static inline __device__ __host__ float3x3 getDiagonalMatrix(float diag = 1.0f);

	static inline __device__ __host__  float3x3 tensorProduct(const float3 &v, const float3 &vt);

	inline __device__ __host__ const float* ptr() const;

	inline __device__ __host__ float* ptr();

	union {
		struct {
			float m11; float m12; float m13;
			float m21; float m22; float m23;
			float m31; float m32; float m33;
		};
		float entries[9];
		float entries2[3][3];
	};
};

inline __device__ __host__ float2x3 matMul(const float2x3& m0, const float3x3& m1);

// (1x2) row matrix as float2
inline __device__ __host__ float3 matMul(const float2& m0, const float2x3& m1);

//////////////////////////////
// float3x4
//////////////////////////////

class float3x4 {
public:
	inline __device__ __host__ float3x4() {}

	inline __device__ __host__ float3x4(const float values[12]);

	inline __device__ __host__ float3x4(const float3x4& other) {
		m11 = other.m11;	m12 = other.m12;	m13 = other.m13;	m14 = other.m14;
		m21 = other.m21;	m22 = other.m22;	m23 = other.m23;	m24 = other.m24;
		m31 = other.m31;	m32 = other.m32;	m33 = other.m33;	m34 = other.m34;
	}

	inline __device__ __host__ float3x4(const float3x3& other);

	inline __device__ __host__ float3x4 operator=(const float3x4 &other);

	inline __device__ __host__ float3x4& operator=(const float3x3 &other);

	//! assumes the last line of the matrix implicitly to be (0,0,0,1)
	inline __device__ __host__ float4 operator*(const float4 &v) const;

	//! assumes an implicit 1 in w component of the input vector
	inline __device__ __host__ float3 operator*(const float3 &v) const;

	//! matrix scalar multiplication
	inline __device__ __host__ float3x4 operator*(const float t) const;

	inline __device__ __host__ float3x4& operator*=(const float t);

	//! matrix scalar division
	inline __device__ __host__ float3x4 operator/(const float t) const;

	inline __device__ __host__ float3x4& operator/=(const float t);

	//! assumes the last line of the matrix implicitly to be (0,0,0,1)
	inline __device__ __host__ float3x4 operator*(const float3x4 &other) const;

	//! assumes the last line of the matrix implicitly to be (0,0,0,1); and a (0,0,0) translation of other
	inline __device__ __host__ float3x4 operator*(const float3x3 &other) const;

	inline __device__ __host__ float& operator()(int i, int j);

	inline __device__ __host__ float operator()(int i, int j) const;

	//! returns the translation part of the matrix
	inline __device__ __host__ float3 getTranslation();

	//! sets only the translation part of the matrix (other values remain unchanged)
	inline __device__ __host__ void setTranslation(const float3 &t);

	//! returns the 3x3 part of the matrix
	inline __device__ __host__ float3x3 getFloat3x3();

	//! sets the 3x3 part of the matrix (other values remain unchanged)
	inline __device__ __host__ void setFloat3x3(const float3x3 &other);

	//! inverts the matrix
	inline __device__ __host__ void inverse();

	//! computes the inverse of the matrix
	inline __device__ __host__ float3x4 getInverse();

	//! prints the matrix; only host	
	__host__ void print();

	inline __device__ __host__ const float* ptr() const;

	inline __device__ __host__ float* ptr();

	union {
		struct {
			float m11; float m12; float m13; float m14;
			float m21; float m22; float m23; float m24;
			float m31; float m32; float m33; float m34;
		};
		float entries[9];
		float entries2[3][4];
	};
};


//////////////////////////////
// float4x4
//////////////////////////////

class float4x4 {
public:
	inline __device__ __host__ float4x4() {}

	inline __device__ __host__ float4x4(const float values[16]);

	inline __device__ __host__ float4x4(const float4x4& other) {
		m11 = other.m11;	m12 = other.m12;	m13 = other.m13;	m14 = other.m14;
		m21 = other.m21;	m22 = other.m22;	m23 = other.m23;	m24 = other.m24;
		m31 = other.m31;	m32 = other.m32;	m33 = other.m33;	m34 = other.m34;
		m41 = other.m41;	m42 = other.m42;	m43 = other.m43;	m44 = other.m44;
	}

	//implicitly assumes last line to (0,0,0,1)
	inline __device__ __host__ float4x4(const float3x4& other);

	inline __device__ __host__ float4x4(const float3x3& other);

	inline __device__ __host__ float4x4 operator=(const float4x4 &other);

	inline __device__ __host__ float4x4 operator=(const float3x4 &other);

	inline __device__ __host__ float4x4& operator=(const float3x3 &other);


	//! not tested
	inline __device__ __host__ float4x4 operator*(const float4x4 &other) const;

	// untested
	inline __device__ __host__ float4 operator*(const float4& v) const;

	// untested
	//implicitly assumes w to be 1
	inline __device__ __host__ float3 operator*(const float3& v) const;
	
	inline __device__ __host__ float& operator()(int i, int j);

	inline __device__ __host__ float operator()(int i, int j) const;


	static inline __device__ __host__  void swap(float& v0, float& v1);

	inline __device__ __host__ void transpose();

	inline __device__ __host__ float4x4 getTranspose() const;

	inline __device__ __host__ void invert();

	//! return the inverse matrix; but does not change the current matrix
	inline __device__ __host__ float4x4 getInverse() const;

	//! returns the 3x3 part of the matrix
	inline __device__ __host__ float3x3 getFloat3x3();

	//! sets the 3x3 part of the matrix (other values remain unchanged)
	inline __device__ __host__ void setFloat3x3(const float3x3 &other);

	//! sets the 4x4 part of the matrix to identity
	inline __device__ __host__ void setIdentity();

	//! sets the 4x4 part of the matrix to identity
	inline __device__ __host__ void setValue(float v);

	//! returns the 3x4 part of the matrix
	inline __device__ __host__ float3x4 getFloat3x4();

	//! sets the 3x4 part of the matrix (other values remain unchanged)
	inline __device__ __host__ void setFloat3x4(const float3x4 &other);

	inline __device__ __host__ const float* ptr() const;

	inline __device__ __host__ float* ptr();

	union {
		struct {
			float m11; float m12; float m13; float m14;
			float m21; float m22; float m23; float m24;
			float m31; float m32; float m33; float m34;
			float m41; float m42; float m43; float m44;
		};
		float entries[16];
		float entries2[4][4];
	};
};



//////////////////////////////
// matNxM
//////////////////////////////

template<unsigned int N, unsigned int M>
class matNxM
{
	public:

		//////////////////////////////
		// Initialization
		//////////////////////////////
		inline __device__ __host__ matNxM() {}

		inline __device__ __host__ matNxM(float* values)
		{
			for(unsigned int i = 0; i<N*M; i++) entries[i] = values[i];
		}

		inline __device__ __host__ matNxM(const float* values)
		{
			for(unsigned int i = 0; i<N*M; i++) entries[i] = values[i];
		}

		inline __device__ __host__ matNxM(const matNxM& other)
		{
			(*this) = other;
		}

		inline __device__ __host__ matNxM<N,M>& operator=(const matNxM<N,M>& other)
		{
			for(unsigned int i = 0; i<N*M; i++) entries[i] = other.entries[i];
			return *this;
		}
		
		inline __device__ __host__ void setZero()
		{
			for(unsigned int i = 0; i<N*M; i++) entries[i] = 0.0f;
		}

		inline __device__ __host__ void setIdentity()
		{
			setZero();
			for(unsigned int i = 0; i<min(N, M); i++) entries2D[i][i] = 1.0f;
		}

		static inline __device__ __host__ matNxM<N, M> getIdentity()
		{
			matNxM<N, M> R; R.setIdentity();
			return R;
		}

		//////////////////////////////
		// Conversion
		//////////////////////////////

		// declare generic constructors for compile time checking of matrix size
		template<class B>
		explicit inline __device__ __host__  matNxM(const B& other);

		template<class B>
		explicit inline __device__ __host__  matNxM(const B& other0, const B& other1);

		// declare generic casts for compile time checking of matrix size
		inline __device__ __host__ operator float();
		inline __device__ __host__ operator float2();
		inline __device__ __host__ operator float3();
		inline __device__ __host__ operator float4();

		inline __device__ __host__ operator float2x2();
		inline __device__ __host__ operator float3x3();
		inline __device__ __host__ operator float4x4();

		//////////////////////////////
		// Matrix - Matrix Multiplication
		//////////////////////////////
		template<unsigned int NOther, unsigned int MOther>
		inline __device__ __host__ matNxM<N,MOther> operator*(const matNxM<NOther,MOther>& other) const
		{
			cudaAssert(M == NOther);
			matNxM<N,MOther> res;

			for(unsigned int i = 0; i<N; i++)
			{
				for(unsigned int j = 0; j<MOther; j++)
				{
					float sum = 0.0f;
					for(unsigned int k = 0; k<M; k++)
					{
						sum += (*this)(i, k)*other(k, j);
					}

					res(i, j) = sum;
				}
			}

			return res;
		}

		//////////////////////////////
		// Matrix - Inversion
		//////////////////////////////

		inline __device__ __host__ float det() const;
		inline __device__ __host__  matNxM<N, M> getInverse() const;

		//////////////////////////////
		// Matrix - Transpose
		//////////////////////////////
		inline __device__ __host__ matNxM<M,N> getTranspose() const
		{
			matNxM<M,N> res;

			for(unsigned int i = 0; i<M; i++)
			{
				for(unsigned int j = 0; j<N; j++)
				{
					res(i, j) = (*this)(j, i);
				}
			}

			return res;
		}

		inline __device__ void printCUDA() const
		{
			for(unsigned int i = 0; i<N; i++)
			{
				for(unsigned int j = 0; j<M; j++)
				{
					printf("%f ", (*this)(i, j));
				}
				printf("\n");
			}
		}

		// inline __device__ bool checkMINF() const;

		// inline __device__ bool checkINF() const;

		inline __device__ bool checkQNAN() const
		{
			for(unsigned int i = 0; i<N; i++)
			{
				for(unsigned int j = 0; j<M; j++)
				{
					if((*this)(i, j) != (*this)(i, j)) return true;
				}
			}

			return false;
		}

		//////////////////////////////
		// Matrix - Matrix Addition
		//////////////////////////////
		inline __device__ __host__ matNxM<N,M> operator+(const matNxM<N,M>& other) const
		{
			matNxM<N,M> res = (*this);
			res+=other;
			return res;
		}

		inline __device__ __host__ matNxM<N,M>& operator+=(const matNxM<N,M>& other)
		{
			for(unsigned int i = 0; i<N*M; i++) entries[i] += other.entries[i];
			return (*this);
		}

		//////////////////////////////
		// Matrix - Negation
		//////////////////////////////
		inline __device__ __host__ matNxM<N,M> operator-() const
		{
			matNxM<N,M> res = (*this)*(-1.0f);
			return res;
		}

		//////////////////////////////
		// Matrix - Matrix Subtraction
		//////////////////////////////
		inline __device__ __host__ matNxM<N,M> operator-(const matNxM<N,M>& other) const
		{
			matNxM<N,M> res = (*this);
			res-=other;
			return res;
		}

		inline __device__ __host__ matNxM<N,M>& operator-=(const matNxM<N,M>& other)
		{
			for(unsigned int i = 0; i<N*M; i++) entries[i] -= other.entries[i];
			return (*this);
		}

		//////////////////////////////
		// Matrix - Scalar Multiplication
		//////////////////////////////
		inline __device__ __host__ matNxM<N,M> operator*(const float t) const
		{
			matNxM<N,M> res = (*this);
			res*=t;
			return res;
		}

		inline __device__ __host__ matNxM<N, M>& operator*=(const float t)
		{
			for(unsigned int i = 0; i<N*M; i++) entries[i] *= t;
			return (*this);
		}

		//////////////////////////////
		// Matrix - Scalar Division
		//////////////////////////////
		inline __device__ __host__ matNxM<N, M> operator/(const float t) const
		{
			matNxM<N, M> res = (*this);
			res/=t;
			return res;
		}

		inline __device__ __host__ matNxM<N, M>& operator/=(const float t)
		{
			for(unsigned int i = 0; i<N*M; i++) entries[i] /= t;
			return (*this);
		}

		//////////////////////////
		// Element Access
		//////////////////////////
		inline __device__ __host__ unsigned int nRows()
		{
			return N;
		}

		inline __device__ __host__ unsigned int nCols()
		{
			return M;
		}

		inline __device__ __host__ float& operator()(unsigned int i, unsigned int j)
		{
			cudaAssert(i<N && j<M);
			return entries2D[i][j];
		}

		inline __device__ __host__ float operator()(unsigned int i, unsigned int j) const
		{
			cudaAssert(i<N && j<M);
			return entries2D[i][j];
		}

		inline __device__ __host__ float& operator()(unsigned int i)
		{
			cudaAssert(i<N*M);
			return entries[i];
		}

		inline __device__ __host__ float operator()(unsigned int i) const
		{
			cudaAssert(i<N*M);
			return entries[i];
		}

		template<unsigned int NOther, unsigned int MOther>
		inline __device__ __host__ void getBlock(unsigned int xStart, unsigned int yStart, matNxM<NOther, MOther>& res) const
		{
			cudaAssert(xStart+NOther <= N && yStart+MOther <= M);
			
			for(unsigned int i = 0; i<NOther; i++)
			{
				for(unsigned int j = 0; j<MOther; j++)
				{
					res(i, j) = (*this)(xStart+i, yStart+j);
				}
			}
		}

		template<unsigned int NOther, unsigned int MOther>
		inline __device__ __host__ void setBlock(matNxM<NOther, MOther>& input, unsigned int xStart, unsigned int yStart)
		{
			cudaAssert(xStart+NOther <= N && yStart+MOther <= M);
			
			for(unsigned int i = 0; i<NOther; i++)
			{
				for(unsigned int j = 0; j<MOther; j++)
				{
					(*this)(xStart+i, yStart+j) = input(i, j);
				}
			}
		}

		inline __device__ __host__ const float* ptr() const {
			return entries;
		}
		inline __device__ __host__ float* ptr() {
			return entries;
		}

		// Operators

		inline __device__ __host__ float norm1DSquared() const
		{
			cudaAssert(M==1 || N==1);

			float sum = 0.0f;
			for(unsigned int i = 0; i<max(N, M); i++) sum += entries[i]*entries[i];

			return sum;
		}

		inline __device__ __host__ float norm1D() const
		{
			return sqrt(norm1DSquared());
		}

	private:

		union
		{
			float entries[N*M];
			float entries2D[N][M];
		};
};

//////////////////////////////
// Scalar - Matrix Multiplication
//////////////////////////////
template<unsigned int N, unsigned int M>
inline __device__ __host__ matNxM<N,M> operator*(const float t, const matNxM<N, M>& mat)
{
	matNxM<N,M> res = mat;
	res*=t;
	return res;
}

//////////////////////////////
// Matrix Inversion
//////////////////////////////

template<>
inline __device__ __host__ float  matNxM<3, 3>::det() const
{
	const float& m11 = entries2D[0][0];
	const float& m12 = entries2D[0][1];
	const float& m13 = entries2D[0][2];

	const float& m21 = entries2D[1][0];
	const float& m22 = entries2D[1][1];
	const float& m23 = entries2D[1][2];

	const float& m31 = entries2D[2][0];
	const float& m32 = entries2D[2][1];
	const float& m33 = entries2D[2][2];

	return m11*m22*m33 + m12*m23*m31 + m13*m21*m32 - m31*m22*m13 - m32*m23*m11 - m33*m21*m12;
}

template<>
inline __device__ __host__ matNxM<3, 3> matNxM<3, 3>::getInverse() const
{
	matNxM<3, 3> res;
	res.entries[0] = entries[4]*entries[8] - entries[5]*entries[7];
	res.entries[1] = -entries[1]*entries[8] + entries[2]*entries[7];
	res.entries[2] = entries[1]*entries[5] - entries[2]*entries[4];

	res.entries[3] = -entries[3]*entries[8] + entries[5]*entries[6];
	res.entries[4] = entries[0]*entries[8] - entries[2]*entries[6];
	res.entries[5] = -entries[0]*entries[5] + entries[2]*entries[3];

	res.entries[6] = entries[3]*entries[7] - entries[4]*entries[6];
	res.entries[7] = -entries[0]*entries[7] + entries[1]*entries[6];
	res.entries[8] = entries[0]*entries[4] - entries[1]*entries[3];
	return res*(1.0f/det());
}

template<>
inline __device__ __host__ float matNxM<2, 2>::det() const
{
	return (*this)(0, 0)*(*this)(1, 1)-(*this)(1, 0)*(*this)(0, 1);
}

template<>
inline __device__ __host__ matNxM<2, 2> matNxM<2, 2>::getInverse() const
{
	matNxM<2, 2> res;
	res(0, 0) =  (*this)(1, 1); res(0, 1) = -(*this)(0, 1);
	res(1, 0) = -(*this)(1, 0); res(1, 1) =  (*this)(0, 0);

	return res*(1.0f/det());
}

//////////////////////////////
// Conversion
//////////////////////////////

// To Matrix from floatNxN
template<>
template<>
inline __device__ __host__  matNxM<1, 1>::matNxM(const float& other)
{
	entries[0] = other;
}

// To Matrix from floatNxN
template<>
template<>
inline __device__ __host__  matNxM<2, 2>::matNxM(const float2x2& other)
{
	for(unsigned int i = 0; i<4; i++) entries[i] = other.entries[i];
}

template<>
template<>
inline __device__ __host__  matNxM<3, 3>::matNxM(const float3x3& other)
{
	for(unsigned int i = 0; i<9; i++) entries[i] = other.entries[i];
}

template<>
template<>
inline __device__ __host__  matNxM<4, 4>::matNxM(const float4x4& other)
{
	for(unsigned int i = 0; i<16; i++) entries[i] = other.entries[i];
}

template<>
template<>
inline __device__ __host__ matNxM<3, 2>::matNxM(const float3& col0, const float3& col1)
{
	entries2D[0][0] = col0.x; entries2D[0][1] = col1.x;
	entries2D[1][0] = col0.y; entries2D[1][1] = col1.y;
	entries2D[2][0] = col0.z; entries2D[2][1] = col1.z;
}

// To floatNxM from Matrix
template<>
inline __device__ __host__ matNxM<4, 4>::operator float4x4()
{
	float4x4 res;
	for(unsigned int i = 0; i<16; i++) res.entries[i] = entries[i];
	return res;
}

template<>
inline __device__ __host__ matNxM<3, 3>::operator float3x3()
{
	float3x3 res;
	for(unsigned int i = 0; i<9; i++) res.entries[i] = entries[i];
	return res;
}

template<>
inline __device__ __host__ matNxM<2, 2>::operator float2x2()
{
	float2x2 res;
	for(unsigned int i = 0; i<4; i++) res.entries[i] = entries[i];
	return res;
}

// To Matrix from floatN
template<>
template<>
inline __device__ __host__ matNxM<2, 1>::matNxM(const float2& other)
{
	entries[0] = other.x;
	entries[1] = other.y;
}

template<>
template<>
inline __device__ __host__ matNxM<3, 1>::matNxM(const float3& other)
{
	entries[0] = other.x;
	entries[1] = other.y;
	entries[2] = other.z;
}

template<>
template<>
inline __device__ __host__ matNxM<4, 1>::matNxM(const float4& other)
{
	entries[0] = other.x;
	entries[1] = other.y;
	entries[2] = other.z;
	entries[3] = other.w;
}

// To floatN from Matrix
template<>
inline __device__ __host__ matNxM<1, 1>::operator float()
{
	return entries[0];
}

template<>
inline __device__ __host__ matNxM<2, 1>::operator float2()
{
	return make_float2(entries[0],  entries[1]);
}

template<>
inline __device__ __host__ matNxM<3, 1>::operator float3()
{
	return make_float3(entries[0], entries[1], entries[2]);
}

template<>
inline __device__ __host__ matNxM<4, 1>::operator float4()
{
	return make_float4(entries[0],  entries[1], entries[2], entries[3]);
}

//////////////////////////////
// Typedefs
//////////////////////////////

typedef matNxM<9, 3> mat9x3;
typedef matNxM<3, 9> mat3x9;

typedef matNxM<9, 1> mat9x1;
typedef matNxM<1, 9> mat1x9;

typedef matNxM<6, 6> mat6x6;

typedef matNxM<6, 1> mat6x1;
typedef matNxM<1, 6> mat1x6;

typedef matNxM<3, 6> mat3x6;
typedef matNxM<6, 3> mat6x3;

typedef matNxM<4, 4> mat4x4;

typedef matNxM<4, 1> mat4x1;
typedef matNxM<1, 4> mat1x4;

typedef matNxM<3, 3> mat3x3;

typedef matNxM<2, 3> mat2x3;
typedef matNxM<3, 2> mat3x2;

typedef matNxM<2, 2> mat2x2;

typedef matNxM<1, 2> mat1x2;
typedef matNxM<2, 1> mat2x1;

typedef matNxM<1, 3> mat1x3;
typedef matNxM<3, 1> mat3x1;

typedef matNxM<1, 1> mat1x1;


#endif
