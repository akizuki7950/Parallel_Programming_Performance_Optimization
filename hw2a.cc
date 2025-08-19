#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
//#include <pmmintrin.h>
#include <nmmintrin.h>
#include<time.h> 

int iters;
double left;
double right;
double lower;
double upper;
int width;
int height;

typedef struct{
    void (*function)(void*);
    void *argument;
}threadpool_task_t;

struct threadpool_t {
    pthread_mutex_t lock;
    pthread_cond_t notify;
    pthread_t *threads;
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

static void *threadpool_thread(void *threadpool){
    threadpool_t *pool = (threadpool_t*)threadpool;
    threadpool_task_t task;

    for(;;){
        pthread_mutex_lock(&(pool->lock));

        if(pool->count == 0){
            break;
        }

        task.function = pool->queue[pool->head].function;
        task.argument = pool->queue[pool->head].argument;
        pool->head += 1;
        pool->head = (pool->head == pool->queue_size) ? 0 : pool->head;
        pool->count -= 1;
        //printf("%d %d\n", pool->count, pool->done);

        pthread_mutex_unlock(&(pool->lock));

        (*(task.function))(task.argument);
    }

    pthread_exit(NULL);
}


void initThreadPool(threadpool_t* pool, int thread_count, int queue_size){
    pool->thread_count = thread_count;
    pool->queue_size = queue_size;
    pool->threads = (pthread_t*)malloc(sizeof(pthread_t) * thread_count);
    pool->queue = (threadpool_task_t*)malloc(sizeof(threadpool_task_t) * queue_size);
    pool->head = 0;
    pool->tail = 0;
    pool->count = 0;
    pool->shutdown = 0;
    pool->started = 0;
    pool->done = 0;
}

void calcMandelbrot(void* args){
    task_params_t* param = (task_params_t*)args;
    int start = param->start;
    int end = param->end;
    int* image = param->img;

    //printf("work: %d %d\n", start, end);
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

        /*
        double y0[2], x0[2];
        for(int l = 0; l < 2; l++){
            if(i >= width){
                i = 0;
                j++;
            }
            y0[l] = j * ((upper - lower) / height) + lower;
            x0[l] = i * ((right - left) / width) + left;
            i++;
        }
        */
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

            // replace multiplication with addition
            //x_diff_v = _mm_sub_pd(new_x_v, x_v);
            //y_diff_v = _mm_sub_pd(new_y_v, y_v);

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
        _mm_storel_epi64((__m128i*)&(image[k]), repeats_v);
        //if(image[k] != image[k+1])
        //    printf("different %d %d %d\n", r, image[k], image[k+1]);
    }

    for (; k < end; k++) {
        //printf("k: %d\n", k);
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
        image[k] = repeats;
    }
    
}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    //printf("%d cpus available\n", CPU_COUNT(&cpu_set));
    unsigned long long ncpus = CPU_COUNT(&cpu_set);
	unsigned long long int num_threads = ncpus;

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
    int blockSize = 1 << 9;

    int num_tasks = (width * height) / blockSize;
    //printf("num tasks: %d\n", num_tasks);
    threadpool_t threadpool;
    task_params_t task_params[num_tasks];
    initThreadPool(&threadpool, num_threads, num_tasks);

    //struct timespec start, finish;
    //double elapsed;
    //clock_gettime(CLOCK_MONOTONIC, &start);
    
    /* allocate memory for image */
    int* image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    // Initialize tasks
    for (int i = 0; i < num_tasks; i++){
        task_params[i].start = i * blockSize;
        task_params[i].end = task_params[i].start + blockSize;
        if (i == num_tasks - 1){
            task_params[i].end = width * height;
        }
        //printf("e: %d %d %d\n", i, task_params[i].start, task_params[i].end);
        task_params[i].img = image;

        threadpool.queue[i].function = calcMandelbrot;
        threadpool.queue[i].argument = (void*)&task_params[i];
    }
    threadpool.count = num_tasks;

    for (int i = 0; i < num_threads; i++){
        pthread_create(&(threadpool.threads[i]), NULL, threadpool_thread, (void*)&threadpool);
    }
    //printf("tasks: %d all done\n", threadpool.done);
    //pthread_cond_broadcast(&(threadpool.notify));

    for (int i = 0; i < num_threads; i++){
       pthread_join(threadpool.threads[i], NULL);
	}

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);

    //clock_gettime(CLOCK_MONOTONIC, &finish);
    //elapsed = (finish.tv_sec - start.tv_sec);
    //elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    //printf("total time: %lf\n", elapsed);
}
