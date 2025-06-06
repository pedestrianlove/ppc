#include <stdlib.h>
#include <math.h>

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

    /* 1) Compute mean and the sum of (value - mean)^2 for each row. */
    for (int i = 0; i < ny; ++i) {
        const float* row_ptr = data + (size_t)i * nx;
        double sum = 0.0;
        for (int k = 0; k < nx; ++k) {
            sum += row_ptr[k];
        }
        mean[i] = sum / nx;

        double varsum = 0.0;
        for (int k = 0; k < nx; ++k) {
            double diff = row_ptr[k] - mean[i];
            varsum += diff * diff;
        }
        /* stdterm[i] holds Σ (a_i,k – mean[i])^2 */
        stdterm[i] = varsum;
    }

    /* 2) For each pair (i, j) with j <= i, compute the covariance and then the correlation. */
    for (int i = 0; i < ny; ++i) {
        const float* row_i = data + (size_t)i * nx;
        for (int j = 0; j <= i; ++j) {
            const float* row_j = data + (size_t)j * nx;
            double cov = 0.0;
            for (int k = 0; k < nx; ++k) {
                cov += (row_i[k] - mean[i]) * (row_j[k] - mean[j]);
            }

            float corr_val;
            if (stdterm[i] > 0.0 && stdterm[j] > 0.0) {
                corr_val = (float)(cov / (sqrt(stdterm[i]) * sqrt(stdterm[j])));
            } else {
                /* If either row has zero variance, define correlation as 0. */
                corr_val = 0.0f;
            }
            result[i + j * (size_t)ny] = corr_val;
        }
    }

    free(mean);
    free(stdterm);
}
