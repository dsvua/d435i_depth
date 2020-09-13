#include "icp_kernels.h"

#ifndef MINF
#define MINF __int_as_float(0xff800000)
#endif

#ifndef PINF
#define PINF __int_as_float(0x7f800000)
#endif

#ifndef INF
#define INF __int_as_float(0x7f800000)
#endif

__global__ void computeNormalsDevice(float3* d_output, float3* d_input, unsigned int width, unsigned int height) {
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

	d_output[y*width+x] = make_float3(MINF, MINF, MINF);

	if(x > 0 && x < width-1 && y > 0 && y < height-1)
	{
		const float3 CC = d_input[(y+0)*width+(x+0)];
		const float3 PC = d_input[(y+1)*width+(x+0)];
		const float3 CP = d_input[(y+0)*width+(x+1)];
		const float3 MC = d_input[(y-1)*width+(x+0)];
		const float3 CM = d_input[(y+0)*width+(x-1)];

		if(CC.x != MINF && PC.x != MINF && CP.x != MINF && MC.x != MINF && CM.x != MINF)
		{
			const float3 n = cross(PC-MC, CP-CM);
			const float  l = length(n);

			if(l > 0.0f)
			{
				d_output[y*width+x] = n/-l;
			}
		}
	}
}
