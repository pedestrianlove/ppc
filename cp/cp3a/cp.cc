#include <stdlib.h>
#include <math.h>
#include <omp.h>

constexpr int LANE = 8;

/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
void correlate(int ny, int nx, const float *data, float *result) {

    /* Allocate temporary arrays for per-row mean and “sum of squared deviations.” */
    float* n_data = (float*) malloc(sizeof(float) * (size_t)ny * nx);
    double* mean    = (double*)malloc(ny * sizeof(double));

    /* 1) Compute mean and the sum of (value - mean)^2 for each row. */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int WORLD_SIZE = omp_get_num_threads();
        int i = tid;
        for (; i < ny-(LANE*WORLD_SIZE-1); i+=LANE*WORLD_SIZE) {
            double sum[LANE] = {0.0};
            for (int k = 0; k < nx; ++k) {
                for (int c = 0; c < LANE; ++c) {
                    sum[c] += data[(i + c) * nx + k];
                }
            }
            for (int c = 0; c < LANE; ++c) {
                mean[i + c] = sum[c] / nx;
            }

            double varsum[LANE] = {0.0};
            for (int k = 0; k < nx; ++k) {
                for (int c = 0; c < LANE; ++c) {
                    double diff = data[(i+c)*nx + k] - mean[i+c];
                    varsum[c] = diff * diff;
                    n_data[(i+c)*nx + k] = diff;  /* Normalize the data on-the-fly */
                }
            }
            double stddev[LANE];
            for (int c = 0; c < LANE; ++c) {
                stddev[c] = sqrt(varsum[c]);
            }
            for (int k = 0; k < nx; ++k) {
                for (int c = 0; c < LANE; ++c) {
                    n_data[(i+c)*nx + k] /= stddev[c];
                }
            }
        }

        for (; i < ny; i+=WORLD_SIZE) {
            const float* row_ptr = data + (size_t)i * nx;
            double sum = 0.0;
            for (int k = 0; k < nx; ++k) {
                sum += row_ptr[k];
            }
            mean[i] = sum / nx;

            double varsum = 0.0;
            #pragma omp simd reduction(+:varsum)
            for (int k = 0; k < nx; ++k) {
                double diff = row_ptr[k] - mean[i];
                varsum = fma(diff, diff, varsum);
                n_data[i*nx + k] = diff;  /* Normalize the data on-the-fly */
            }

            double stddev = sqrt(varsum);

            for (int k = 0; k < nx; ++k) {
                n_data[i*nx + k] /= stddev;
            }
        }
    }
    free(mean);

    /* 2) For each pair (i, j) with j <= i, compute the covariance and then the correlation. */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ny; ++i) {
        int j;
        for (j = 0; j <= i-(LANE-1); j+=LANE) {
            double cov[LANE] = {0.0};
            for (int k = 0; k < nx; ++k) {
                const double diff_ik = n_data[i*nx + k];
                for (int c = 0; c < LANE; ++c) {
                    const double diff_jkc = n_data[(j+c)*nx + k];
                    cov[c] += diff_ik*diff_jkc;
                }
            }

            for (int c = 0; c < LANE; ++c) {
                result[i + (j+c) * (size_t)ny] = cov[c];
            }
        }

        for (; j <= i; ++j) {
            double cov = 0.0;
            #pragma omp simd reduction(+:cov)
            for (int k = 0; k < nx; ++k) {
                cov = fma((n_data[i*nx + k]), (n_data[j*nx + k]), cov);
            }

            result[i + j * (size_t)ny] = cov;
        }
    }
}
