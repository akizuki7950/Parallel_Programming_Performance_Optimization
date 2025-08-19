#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>


//======================
#define DEV_NO_1 0
#define DEV_NO_2 1

const int INF = ((1 << 30) - 1);
cudaDeviceProp prop;
//const unsigned int V = 60010;
int n, m;
size_t mPitch;
static int* Dist;
static int* dDist_dv1, *dDist_dv2;

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    printf("%d %d\n", n, m);

    mPitch = (n * sizeof(int) / 512) * 512;
    if(n % (512 / sizeof(int)))mPitch += 512;

    cudaHostAlloc(&Dist, mPitch * (n + 128), cudaHostAllocMapped | cudaHostAllocPortable);
    for (int i = 0; i < (mPitch / sizeof(int)); ++i) {
        for (int j = 0; j < (n + 128); ++j) {
            if(i < n && j < n){
                if (i == j) {
                    Dist[i * (mPitch / sizeof(int)) + j] = 0;
                } else {
                    Dist[i * (mPitch / sizeof(int)) + j] = INF;
                }
            }
            else {
                Dist[i * (mPitch / sizeof(int)) + j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        //printf("%d %d %d\n", pair[0], pair[1], pair[2]);
        Dist[pair[0] * (mPitch / sizeof(int)) + pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (Dist[i * (mPitch / sizeof(int)) + j] >= INF) Dist[i * (mPitch / sizeof(int)) + j] = INF;
        }
        fwrite(&Dist[i * (mPitch / sizeof(int))], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }

__device__ unsigned addr(unsigned pitch, unsigned i, unsigned j){
    return pitch * i + j;
}

__global__ void calc(
    int* dDist, size_t pitch, int B, int Round,
    int block_start_x, int block_start_y
    //int block_width, int block_height
    ){
    int local_y = block_start_y + blockIdx.y;
    int local_x = block_start_x + blockIdx.x;

    unsigned x = local_x * B + threadIdx.x;
    unsigned y = local_y * B;
    for(int i = 0; i < B; i++, y++){
        for(int k = Round * B; k < (Round + 1) * B; k++){
            int nd = dDist[y * pitch + k] + dDist[k * pitch + x];
            if(nd < dDist[y * pitch + x]){
                dDist[y * pitch + x] = nd;
            }
        }
        //__syncthreads();
    }
}

__global__ void calc_p1(
    int* dDist, size_t pitch, int B, int Round,
    int block_start_x, int block_start_y
    //int block_width, int block_height
    ){
    unsigned local_y = block_start_y + blockIdx.y;
    unsigned local_x = block_start_x + blockIdx.x;

    __shared__ unsigned d[64][64];
    unsigned x = (local_x << 6) + threadIdx.x;
    unsigned y = (local_y << 6) + threadIdx.y;
    #pragma unroll
    for(int i = 0; i < 64; i+=16){
        d[i+threadIdx.y][threadIdx.x] = dDist[addr(pitch, y+i, x)];
    }
    unsigned y_step = pitch << 4;
    unsigned add = addr(pitch, y, x);
    __syncthreads();
    #pragma unroll
    for(unsigned i = 0; i < 64; i+=16, y+=16){
        #pragma unroll 32
        for(unsigned k = 0; k < 64; k+=1){
            unsigned nd = d[i+threadIdx.y][k] + d[k][threadIdx.x];
            dDist[add] = d[i+threadIdx.y][threadIdx.x] = min(d[i+threadIdx.y][threadIdx.x], nd);
            //dDist[addr(pitch, y, x)] = d[i+threadIdx.y][threadIdx.x];
            //if(nd < d[i+threadIdx.y][threadIdx.x]){
            //    dDist[addr(pitch, y, x)] = d[i+threadIdx.y][threadIdx.x] = nd;
            //}
        }
        add += y_step;
    }
    /*
    __syncthreads();
    y = (local_y << 6) + threadIdx.y;
    #pragma unroll
    for(int i = 0; i < 64; i+= 16, y+=16){
        dDist[y*pitch + x] = d[i+threadIdx.y][threadIdx.x];
    }
    */
}

__global__ void calc_p2_r(
    int* dDist, size_t pitch, int B, int Round,
    int block_start_x, int block_start_y
    //int block_width, int block_height
    ){
    unsigned local_y = block_start_y + blockIdx.y;
    unsigned local_x = block_start_x + blockIdx.x;

    //if(local_y == Round && local_x == Round)return;

    __shared__ unsigned d_gray[64][64];
    __shared__ unsigned d_me[64][64];
    unsigned x = (local_x << 6) + threadIdx.x;
    unsigned y = (local_y << 6) + threadIdx.y;
    unsigned xr = (Round << 6) + threadIdx.x;
    unsigned yr = (Round << 6) + threadIdx.y;
    #pragma unroll
    for(unsigned i = 0; i < 64; i+=16){
        d_gray[i+threadIdx.y][threadIdx.x] = dDist[(yr+i) * pitch + xr];
        d_me[i+threadIdx.y][threadIdx.x] = dDist[(y+i) * pitch + x];
    }
    unsigned y_step = pitch << 4;
    unsigned add = addr(pitch, y, x);
    __syncthreads();
    #pragma unroll
    for(unsigned i = 0; i < 64; i+=16, y+=16){
        #pragma unroll 32
        for(unsigned k = 0; k < 64; k+=1){
            unsigned nd = d_gray[i+threadIdx.y][k] + d_me[k][threadIdx.x];
            //int add = addr(pitch, y, x);
            dDist[add] = d_me[i+threadIdx.y][threadIdx.x] = min(d_me[i+threadIdx.y][threadIdx.x], nd);
            //dDist[add] = d_me[i+threadIdx.y][threadIdx.x];
            //if(nd < dDist[add]){
            //    dDist[add] = d_me[i+threadIdx.y][threadIdx.x] = nd;
            //}
        }
        add += y_step;
    }

    /*
    __syncthreads();
    y = (local_y << 6) + threadIdx.y;
    #pragma unroll
    for(int i = 0; i < 64; i+= 16, y+=16){
        dDist[y*pitch + x] = d_me[i+threadIdx.y][threadIdx.x];
    }
    */
}

__global__ void calc_p2_c(
    int* dDist, size_t pitch, int B, int Round,
    int block_start_x, int block_start_y
    //int block_width, int block_height
    ){
    unsigned local_y = block_start_y + blockIdx.y;
    unsigned local_x = block_start_x + blockIdx.x;

    //if(local_y == Round && local_x == Round)return;

    __shared__ unsigned d_gray[64][64];
    __shared__ unsigned d_me[64][64];
    unsigned x = (local_x << 6) + threadIdx.x;
    unsigned y = (local_y << 6) + threadIdx.y;
    unsigned xr = (Round << 6) + threadIdx.x;
    unsigned yr = (Round << 6) + threadIdx.y;
    #pragma unroll
    for(unsigned i = 0; i < 64; i+=16){
        d_gray[i+threadIdx.y][threadIdx.x] = dDist[(yr+i) * pitch + xr];
        d_me[i+threadIdx.y][threadIdx.x] = dDist[(y+i) * pitch + x];
    }
    unsigned y_step = pitch << 4;
    unsigned add = addr(pitch, y, x);
    __syncthreads();
    #pragma unroll
    for(unsigned i = 0; i < 64; i+=16, y+=16){
        #pragma unroll 32
        for(unsigned k = 0; k < 64; k+=1){
            unsigned nd = d_me[i+threadIdx.y][k] + d_gray[k][threadIdx.x];
            //int add = addr(pitch, y, x);
            dDist[add] = d_me[i+threadIdx.y][threadIdx.x] = min(d_me[i+threadIdx.y][threadIdx.x], nd);
            //dDist[add] = d_me[i+threadIdx.y][threadIdx.x];
            //if(nd < dDist[add]){
            //    dDist[add] = d_me[i+threadIdx.y][threadIdx.x] = nd;
            //}
        }
        add += y_step;
    }
    /*
    __syncthreads();
    y = (local_y << 6) + threadIdx.y;
    #pragma unroll
    for(int i = 0; i < 64; i+= 16, y+=16){
        dDist[y*pitch + x] = d_me[i+threadIdx.y][threadIdx.x];
    }
    */
}

__global__ void calc_p3(
    int* dDist, size_t pitch, int B, int Round,
    int block_start_x, int block_start_y
    //int block_width, int block_height
    ){
    unsigned local_y = block_start_y + blockIdx.y;
    unsigned local_x = block_start_x + blockIdx.x;

    //if(local_y == Round || local_x == Round)return;

    __shared__ unsigned d_blue_r[64][64];
    __shared__ unsigned d_blue_c[64][64];
    //__shared__ int d_me[64][64];
    unsigned x = (local_x << 6) + threadIdx.x;
    unsigned y = (local_y << 6) + threadIdx.y;
    unsigned xr = (Round << 6) + threadIdx.x;
    unsigned yr = (Round << 6) + threadIdx.y;

    #pragma unroll
    for(unsigned i = 0; i < 64; i+=16){
        d_blue_r[i+threadIdx.y][threadIdx.x] = dDist[addr(pitch, yr+i, x)];
        d_blue_c[i+threadIdx.y][threadIdx.x] = dDist[addr(pitch, y+i, xr)];
        //d_me[i+threadIdx.y][threadIdx.x] = dDist[(y+i) * pitch + x];
    }
    unsigned y_step = pitch << 4;
    unsigned add = addr(pitch, y, x);
    __syncthreads();

    #pragma unroll
    for(unsigned i = 0; i < 64; i+=16, y+=16){
        #pragma unroll 32
        for(unsigned k = 0; k < 64; k+=1){
            unsigned nd = d_blue_c[i+threadIdx.y][k] + d_blue_r[k][threadIdx.x];
            dDist[add] = min(dDist[add], nd);
            //if(nd < dDist[add]){
            //    dDist[add] = nd;
            //}
        }
        add += y_step;
    }

    /*
    __syncthreads();
    y = (local_y << 6) + threadIdx.y;
    #pragma unroll
    for(int i = 0; i < 64; i+= 16, y+=16){
        dDist[y*pitch + x] = d_me[i+threadIdx.y][threadIdx.x];
    }
    */
}


int main(int argc, char* argv[]) {
    cudaGetDeviceProperties(&prop, DEV_NO_1);
    printf("maxThreadsPerBlock = %d, sharedMemPerBlock = %d, maxGridSize = %d %d %d\n", prop.maxThreadsPerBlock, prop.sharedMemPerBlock, prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    cudaGetDeviceProperties(&prop, DEV_NO_2);
    printf("maxThreadsPerBlock = %d, sharedMemPerBlock = %d, maxGridSize = %d %d %d\n", prop.maxThreadsPerBlock, prop.sharedMemPerBlock, prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    
    input(argv[1]);

    cudaSetDevice(DEV_NO_1);
    cudaHostGetDevicePointer(&dDist_dv1, Dist, 0);

    cudaSetDevice(DEV_NO_2);
    cudaHostGetDevicePointer(&dDist_dv2, Dist, 0);

    /*
    cudaSetDevice(DEV_NO_1);
    cudaMallocPitch(&dDist_dv1, &pitch_dv1, n * sizeof(int), n + 128);
    cudaMemcpy2D(dDist_dv1, pitch_dv1, Dist, (n + 128) * sizeof(int), pitch_dv1, n + 128, cudaMemcpyHostToDevice);

    cudaSetDevice(DEV_NO_2);
    cudaMallocPitch(&dDist_dv2, &pitch_dv2, n * sizeof(int), n + 128);
    cudaMemcpy2D(dDist_dv2, pitch_dv2, Dist, (n + 128) * sizeof(int), pitch_dv2, n + 128, cudaMemcpyHostToDevice);
    */

    int B = 64;
    int round = ceil(n, B);
    
    dim3 dim_p1(1, 1);
    dim3 dim_p2_r(round, 1);
    dim3 dim_p2_c(1, round);
    dim3 dim_p3_dv1(round, round >> 1);
    dim3 dim_p3_dv2(round, round - (round >> 1));
    dim3 dim_p3_ori(round, round);
    dim3 tDim(64, 16);
    //cudaMemcpy3DPeerParms p = { 0 };
    //cudaError_t error;
    for(int r = 0; r < round; r++){
        cudaSetDevice(DEV_NO_1);
        calc_p1<<<dim_p1, tDim>>>(dDist_dv1, mPitch / sizeof(int), B, r, r, r);
        cudaDeviceSynchronize();

        cudaSetDevice(DEV_NO_1);
        calc_p2_r<<<dim_p2_r, tDim>>>(dDist_dv1, mPitch / sizeof(int), B, r, 0, r);
        cudaSetDevice(DEV_NO_2);
        calc_p2_c<<<dim_p2_c, tDim>>>(dDist_dv2, mPitch / sizeof(int), B, r, r, 0);

        cudaSetDevice(DEV_NO_2);
        cudaDeviceSynchronize();
        cudaSetDevice(DEV_NO_1);
        cudaDeviceSynchronize();
        
        cudaSetDevice(DEV_NO_1);
        calc_p3<<<dim_p3_dv1, tDim>>>(dDist_dv1, mPitch / sizeof(int), B, r, 0, 0);
        cudaSetDevice(DEV_NO_2);
        calc_p3<<<dim_p3_dv2, tDim>>>(dDist_dv2, mPitch / sizeof(int), B, r, 0, round >> 1);

        cudaSetDevice(DEV_NO_2);
        cudaDeviceSynchronize();
        cudaSetDevice(DEV_NO_1);
        cudaDeviceSynchronize();
    }
    //cudaSetDevice(DEV_NO_1);
    //cudaMemcpy2D(Dist, (n + 128) * sizeof(int), dDist_dv1, pitch_dv1, n * sizeof(int), n, cudaMemcpyDeviceToHost);
    /*
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            printf("%d ", Dist[i * mPitch / sizeof(int) + j]);
        }
        printf("\n");
    }
    */
    // block_FW(B);
    output(argv[2]);
    return 0;
}