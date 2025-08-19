#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <time.h>


//======================
#define DEV_NO 0

const int INF = ((1 << 30) - 1);
cudaDeviceProp prop;
//const unsigned int V = 60010;
int n, m;
static int* Dist;
static int* dDist;

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    //printf("%d %d\n", n, m);
    cudaHostAlloc(&Dist, (n + 128) * (n + 128) * sizeof(int), cudaHostAllocMapped);

    int y_addr = 0;
    int y_step = n + 128;
    for (int i = 0; i < (n + 128); ++i, y_addr += y_step) {
        for (int j = 0; j < (n + 128); ++j) {
            if(i < n && j < n){
                if (i == j) {
                    Dist[y_addr + j] = 0;
                } else {
                    Dist[y_addr + j] = INF;
                }
            }
            else {
                Dist[y_addr + j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0] * y_step + pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    int y_addr = 0;
    int y_step = n + 128;
    for (int i = 0; i < n; ++i, y_addr += y_step) {
        //for (int j = 0; j < n; ++j) {
        //    if (Dist[y_addr + j] >= INF) Dist[y_addr + j] = INF;
        //}
        fwrite(&Dist[y_addr], sizeof(int), n, outfile);
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
    //cudaGetDeviceProperties(&prop, DEV_NO);
    //printf("maxThreadsPerBlock = %d, sharedMemPerBlock = %d, maxGridSize = %d %d %d\n", prop.maxThreadsPerBlock, prop.sharedMemPerBlock, prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    double cputime = 0, memtime = 0, IOtime = 0;

    clock_t begin = clock();
    input(argv[1]);
    clock_t end = clock();
    IOtime += end - begin;

    size_t pitch;
    cudaSetDevice(DEV_NO);
    cudaMallocPitch(&dDist, &pitch, n * sizeof(int), n + 128);
    begin = clock();
    cudaMemcpy2D(dDist, pitch, Dist, (n + 128) * sizeof(int), pitch, n + 128, cudaMemcpyHostToDevice);
    end = clock();
    memtime += end - begin;
    int B = 64;
    int round = ceil(n, B);
    
    dim3 dim_p1(1, 1);
    dim3 dim_p2_r(round, 1);
    dim3 dim_p2_c(1, round);
    dim3 dim_p3(round, round);
    dim3 tDim(64, 16);
    begin = clock();
    for(int r = 0; r < round; r++){
        calc_p1<<<dim_p1, tDim>>>(dDist, pitch / sizeof(int), B, r, r, r);
        cudaDeviceSynchronize();

        calc_p2_r<<<dim_p2_r, tDim>>>(dDist, pitch / sizeof(int), B, r, 0, r);
        calc_p2_c<<<dim_p2_c, tDim>>>(dDist, pitch / sizeof(int), B, r, r, 0);
        cudaDeviceSynchronize();
        
        calc_p3<<<dim_p3, tDim>>>(dDist, pitch / sizeof(int), B, r, 0, 0);
        cudaDeviceSynchronize();
    }
    end = clock();
    cputime += end - begin;

    begin = clock();
    cudaMemcpy2D(Dist, (n + 128) * sizeof(int), dDist, pitch, n * sizeof(int), n, cudaMemcpyDeviceToHost);
     end = clock();
    memtime += end - begin;
    // block_FW(B);

    begin = clock();
    output(argv[2]);
    end = clock();
    IOtime += end - begin;

    //clock_t end = clock();
    printf("Elapsed time: %lf %lf %lf\n", cputime / CLOCKS_PER_SEC, memtime / CLOCKS_PER_SEC, IOtime / CLOCKS_PER_SEC);
    return 0;
}