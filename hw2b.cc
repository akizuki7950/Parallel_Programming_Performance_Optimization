#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <mpi.h>
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
//#include <pmmintrin.h>
#include <nmmintrin.h>
#include <pthread.h>

#define DATA_TAG 1
#define TERMINATE_TAG 1 << 1
#define RESULT_TAG 1 << 2
#define PERMIT_TAG 1 << 3

int iters;
double left;
double right;
double lower;
double upper;
int width;
int height;
int p_task_idx = 0;
pthread_mutex_t idx_lock;
pthread_cond_t notify;
int* f_image;
bool done;
bool start;

typedef struct{
    void (*function)(void*);
    void *argument;
}threadpool_task_t;

typedef struct{
    int width;
    int height;
    int row_block_size;
    int mpi_size;
    int iters;
    const char* filename;
}master_thread_args;

struct threadpool_t {
    threadpool_task_t *queue;
    int thread_count;
    int queue_size;
    int head;
    int tail;
    int count;
    int shutdown;
    int started;
    int done;
};

typedef struct{
    int start;
    int end;
    int write;
    int* img;
}task_params_t;

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

void CalcMandelbrot(void* args){
    
    task_params_t* param = (task_params_t*)args;
    int start = param->start;
    int end = param->end;
    int* image = param->img;

    int size = ((end - start) >> 1) << 1;
    int k = 0;
    int i, j;
    j = start / width;
    i = start % width;
    __m128i i_v, j_v, w_v_cmp, w_v;
    i_v = _mm_set_epi32(i + 1, i, i + 1, i);
    j_v = _mm_set1_epi32(j);
    w_v_cmp = _mm_set1_epi32(width-1);
    w_v = _mm_set1_epi32(width);

    __m128d lower_v, left_v, i_step_v, j_step_v;
    lower_v = _mm_set1_pd(lower);
    left_v = _mm_set1_pd(left);
    j_step_v = _mm_set1_pd(((upper - lower) / height));
    i_step_v = _mm_set1_pd(((right - left) / width));

    __m128i add_re, add_ite, add_cry;
    add_re = _mm_set1_epi64x(1);
    add_ite = _mm_set1_epi32(2);
    add_cry = _mm_set1_epi32(1);

    __m128d mul2 = _mm_set1_pd((double)2.0);
    __m128d cmp = _mm_set1_pd((double)4.0);

    for (k = start; k < start + size; k += 2) {

        __m128d x_v, y_v, x0_v, y0_v, temp_v, temp1_v, temp2_v, temp3_v, mask_val, len_sq;
        __m128i repeats_v, mask_ite, tempi1_v, tempi2_v;

        repeats_v = _mm_set1_epi64x(0);

        __m128i carry_mask = _mm_cmpgt_epi32(i_v, w_v_cmp);
        __m128i c_j_v = _mm_add_epi32(add_cry, j_v);
        __m128i c_i_v = _mm_sub_epi32(i_v, w_v);

        i_v = _mm_blendv_epi8(i_v, c_i_v, carry_mask);
        j_v = _mm_blendv_epi8(j_v, c_j_v, carry_mask);

        y0_v = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(j_v), j_step_v), lower_v);
        x0_v = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(i_v), i_step_v), left_v);

        x_v = _mm_set1_pd((double)0.0);
        y_v = _mm_set1_pd((double)0.0);

        i_v = _mm_add_epi32(i_v, add_ite);

        mask_val = _mm_cmplt_pd(mul2, cmp);
        
        __m128d last_x_squared, last_y_squared, new_x_v, new_y_v, x_diff_v, y_diff_v;
        last_x_squared = _mm_set1_pd((double)0.0);
        last_y_squared = _mm_set1_pd((double)0.0);

        int r = 0;
        while (r < iters) {
            mask_ite = _mm_castpd_si128(mask_val);
            if(_mm_test_all_zeros(mask_ite, mask_ite))break;
            
            // temp = x * x - y * y + x0
            temp3_v = _mm_sub_pd(last_x_squared, last_y_squared);
            temp_v = _mm_add_pd(temp3_v, x0_v); // temp

            // y = 2 * x * y + y0
            temp2_v = _mm_mul_pd(_mm_add_pd(x_v, x_v), y_v);
            temp3_v = _mm_add_pd(temp2_v, y0_v);

            // masked x, y replacements
            x_v = _mm_blendv_pd(x_v, temp_v, mask_val);
            y_v = _mm_blendv_pd(y_v, temp3_v, mask_val);

            // masked repeats increments
            tempi1_v = _mm_add_epi64(repeats_v, add_re);
            mask_ite = _mm_castpd_si128(mask_val);
            repeats_v = _mm_blendv_epi8(repeats_v, tempi1_v, mask_ite);
            r++;

            last_x_squared = _mm_mul_pd(x_v, x_v);
            last_y_squared = _mm_mul_pd(y_v, y_v);
            // length_squared = x * x + y * y
            len_sq = _mm_add_pd(last_x_squared, last_y_squared);
            mask_val = _mm_cmplt_pd(len_sq, cmp);

        }
        repeats_v = _mm_shuffle_epi32(repeats_v, _MM_SHUFFLE(3, 1, 2, 0));
        _mm_storel_epi64((__m128i*)&(image[param->write + k - start]), repeats_v);
    }

    for (; k < end; k++) {
        int j = k / width;
        int i = k % width;

        double y0 = j * ((upper - lower) / height) + lower;
        double x0 = i * ((right - left) / width) + left;

        int repeats = 0;
        double x = 0;
        double y = 0;
        double length_squared = 0;
        while (repeats < iters && length_squared < 4) {
            double temp = x * x - y * y + x0;
            y = 2 * x * y + y0;
            x = temp;
            length_squared = x * x + y * y;
            ++repeats;
        }
        image[param->write + k - start] = repeats;
    }
}

void* MasterThread(void* args){
    master_thread_args* params = (master_thread_args*)args;
    int width = params->width;
    int height = params->height;
    int row_block_size = params->row_block_size;
    int mpi_size = params->mpi_size;
    int iters = params->iters;
    const char* filename = params->filename;

    MPI_Status status, permit_status;
    f_image = (int*)malloc(width * height * sizeof(int));
    int* buf = (int*)malloc((row_block_size << 1) * sizeof(int));
    int p_num_tasks = (width * height) / row_block_size;
    int proc_to_task[mpi_size];
    assert(f_image);

    int count = 0;
    for(int k = 1; k < mpi_size; k++){
        pthread_mutex_lock(&idx_lock);
        MPI_Send(&p_task_idx, 1, MPI_INT, k, DATA_TAG, MPI_COMM_WORLD);
        proc_to_task[k] = p_task_idx;
        count++;
        p_task_idx++;
        pthread_mutex_unlock(&idx_lock);
    }
    pthread_mutex_lock(&idx_lock);
    start = true;
    pthread_cond_signal(&notify);
    pthread_mutex_unlock(&idx_lock);
    do {
        MPI_Recv(buf, (row_block_size << 1), MPI_INT, MPI_ANY_SOURCE, RESULT_TAG, MPI_COMM_WORLD, &status);
        int size_cpy = row_block_size;
        if(proc_to_task[status.MPI_SOURCE] == p_num_tasks - 1){
            size_cpy = width * height - proc_to_task[status.MPI_SOURCE] * row_block_size;
        }
        memcpy((void*)(f_image + proc_to_task[status.MPI_SOURCE] * row_block_size), (void*)buf, size_cpy * sizeof(int));
        count--;
        pthread_mutex_lock(&idx_lock);
        if(p_task_idx < p_num_tasks){
            MPI_Send(&p_task_idx, 1, MPI_INT, status.MPI_SOURCE, DATA_TAG, MPI_COMM_WORLD);
            proc_to_task[status.MPI_SOURCE] = p_task_idx;
            count++;
            p_task_idx++;
        }
        else {
            MPI_Send(&p_task_idx, 1, MPI_INT, status.MPI_SOURCE, TERMINATE_TAG, MPI_COMM_WORLD);
        }
        pthread_mutex_unlock(&idx_lock);
    } while(count > 0);

    while(!done){
        pthread_cond_wait(&notify, &idx_lock);
    }

    //double s, t;
    //s = MPI_Wtime();
    write_png(filename, iters, width, height, f_image);
    //t = MPI_Wtime();
    //printf("write png time: %lf\n", t - s);
    free(f_image);
    pthread_exit(NULL);
}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    unsigned long long ncpus = CPU_COUNT(&cpu_set);
	unsigned long long int num_threads = ncpus;

    int mpi_rank, mpi_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    MPI_Status status, permit_status;

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);
    int row_block_size = width << 2; //1 << 10;
    int p_num_tasks = (width * height) / row_block_size;
    if(p_num_tasks == 0){
        p_num_tasks = 1;
    }
    pthread_t master_thread;
    master_thread_args m_args;

    //double s, t;
    //s = MPI_Wtime();

    //double s_b, t_b;
    //double cpu_time = 0;

    if (mpi_rank == 0){
        m_args.width = width;
        m_args.height = height;
        m_args.row_block_size = row_block_size;
        m_args.mpi_size = mpi_size;
        m_args.filename = filename;
        m_args.iters = iters;
        pthread_create(&master_thread, NULL, MasterThread, (void*)&m_args);
    }

    // Slave
    int task_idx;
    int thread_block_size = 1 << 4;
    int t_num_tasks = row_block_size / thread_block_size;
    int* buf = (int*)malloc((row_block_size << 1) * sizeof(int));

    done = start = false;

    task_params_t task_params[2 * t_num_tasks];

    if(mpi_rank){
        MPI_Recv(&task_idx, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        while(status.MPI_TAG == DATA_TAG){
            if(task_idx == p_num_tasks - 1){
                t_num_tasks = (width * height - task_idx * row_block_size) / thread_block_size;
            }
            //s_b = MPI_Wtime();
            #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
            for(int i = 0; i < t_num_tasks; i++){
                task_params[i].start = task_idx * row_block_size + i * thread_block_size;
                task_params[i].end = task_params[i].start + thread_block_size;
                task_params[i].write = i * thread_block_size;
                if (i == t_num_tasks - 1){
                    if(task_idx == p_num_tasks - 1){
                        task_params[i].end = width * height;
                    }
                    else {
                        task_params[i].end = (task_idx + 1) * row_block_size;
                    }
                }
                task_params[i].img = buf;
                CalcMandelbrot((void*)&task_params[i]);
            }
            int size_send = task_params[t_num_tasks-1].end - task_params[0].start;
            //t_b = MPI_Wtime();
            //cpu_time += t_b - s_b;

            MPI_Send(buf, size_send, MPI_INT, 0, RESULT_TAG, MPI_COMM_WORLD);
            MPI_Recv(&task_idx, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        }
    }
    else {
        bool bk = false;
        while(1){
            pthread_mutex_lock(&idx_lock);
            while(!start){
                pthread_cond_wait(&notify, &idx_lock);
            }
            if(p_task_idx < p_num_tasks){
                task_idx = p_task_idx;
                p_task_idx++;
            }
            else {
                bk = true;
            }
            pthread_mutex_unlock(&idx_lock);
            if(bk)break;
            int* task_write = f_image + task_idx * row_block_size;

            //s_b = MPI_Wtime();
            #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
            for(int i = 0; i < t_num_tasks; i++){
                task_params[i].start = task_idx * row_block_size + i * thread_block_size;
                task_params[i].end = task_params[i].start + thread_block_size;
                task_params[i].write = i * thread_block_size;
                if (i == t_num_tasks - 1){
                    if(task_idx == p_num_tasks - 1){
                        task_params[i].end = width * height;
                    }
                    else {
                        task_params[i].end = (task_idx + 1) * row_block_size;
                    }
                }
                task_params[i].img = task_write;
                CalcMandelbrot((void*)&task_params[i]);
            }
            //t_b = MPI_Wtime();
            //cpu_time += t_b - s_b;
            //int size_send = row_block_size;
            //if(task_idx == p_num_tasks - 1){
            //    size_send = (width * height) - task_idx * row_block_size;
            //}
            //memcpy((void*)(f_image + task_idx * row_block_size), (void*)buf, size_send * sizeof(int));
        }
        done = true;
        pthread_cond_signal(&notify);
    }

    if(mpi_rank == 0){
        pthread_join(master_thread, NULL);
    }
    //t = MPI_Wtime();

    //double all_avg, avg;
    //avg = t - s;
    //MPI_Reduce(&avg, &all_avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    //if(mpi_rank == 0)
    //    printf("total time: %lf\n", all_avg / mpi_size);
    //printf("rank: %d, cpu time: %lf\n", mpi_rank, cpu_time);
}
