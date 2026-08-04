#include <stdlib.h>
#include <math.h>
#include <omp.h>

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
    double* mean    = (double*)malloc(ny * sizeof(double));
    double* stdterm = (double*)malloc(ny * sizeof(double));

    #pragma omp parallel
    {

        /* 1) Compute mean and the sum of (value - mean)^2 for each row. */
        #pragma omp for
        for (int i = 0; i < ny; ++i) {
            const float* row_ptr = data + (size_t)i * nx;
            double sum = 0.0;
            double sum_sq = 0.0;

            #pragma omp simd reduction(+: sum, sum_sq)
            for (int k = 0; k < nx; ++k) {
                double val = row_ptr[k];
                sum += val;
                sum_sq += val * val;
            }

            double m = sum / nx;
            mean[i] = m;
            stdterm[i] = sqrt(sum_sq - m * sum);
        }

        /* 2) For each pair (i, j) with j <= i, compute the covariance and then the correlation. */
        #pragma omp for collapse(2)
        for (int i = 0; i < ny; ++i) {
            for (int j = 0; j <= i; ++j) {
                const float* row_i = data + (size_t)i * nx;
                const float* row_j = data + (size_t)j * nx;
                double cov = 0.0;
                #pragma omp simd reduction(+:cov)
                for (int k = 0; k < nx; ++k) {
                    cov += (row_i[k] - mean[i]) * (row_j[k] - mean[j]);
                }

                result[i + j * (size_t)ny] = (float)(cov / (stdterm[i] * stdterm[j]));
            }
        }
    }

    free(mean);
    free(stdterm);
}
