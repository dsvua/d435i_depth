
__global__ void computeNormalsDevice(float4* d_output, float4* d_input, unsigned int width, unsigned int height);

__global__ void resampleFloat3MapDevice(float4* d_colorMapResampledFloat4, float4* d_colorMapFloat4,
            unsigned int inputWidth, unsigned int inputHeight, unsigned int outputWidth, unsigned int outputHeight);

inline __device__ float3 bilinearInterpolationFloat3(float x, float y, float3* d_input, unsigned int imageWidth,
    unsigned int imageHeight);