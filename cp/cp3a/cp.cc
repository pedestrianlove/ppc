#include <stdlib.h>
#include <math.h>

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
    double* mean    = (double*)malloc(ny * sizeof(double));
    double* stdterm = (double*)malloc(ny * sizeof(double));

    /* 1) Compute mean and the sum of (value - mean)^2 for each row. */
    int i = 0;
    for (; i < ny-(LANE-1); i+=LANE) {
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
                varsum[c] = fma(diff, diff, varsum[c]);
            }
        }
        /* stdterm[i] holds Σ (a_i,k – mean[i])^2 */
        for (int c = 0; c < LANE; ++c) {
            stdterm[i+c] = sqrt(varsum[c]);
        }
    }
    for (; i < ny; ++i) {
        const float* row_ptr = data + (size_t)i * nx;
        double sum = 0.0;
        for (int k = 0; k < nx; ++k) {
            sum += row_ptr[k];
        }
        mean[i] = sum / nx;

        double varsum = 0.0;
        for (int k = 0; k < nx; ++k) {
            double diff = row_ptr[k] - mean[i];
            varsum += diff*diff;
        }
        /* stdterm[i] holds Σ (a_i,k – mean[i])^2 */
        stdterm[i] = sqrt(varsum);
    }

    /* 2) For each pair (i, j) with j <= i, compute the covariance and then the correlation. */
    for (int i = 0; i < ny; ++i) {
        int j;
        for (j = 0; j <= i-(LANE-1); j+=LANE) {
            double cov[LANE] = {0.0};
            for (int k = 0; k < nx; ++k) {
                for (int c = 0; c < LANE; ++c) {
                    cov[c] = fma((data[i*nx + k] - mean[i]), (data[(j+c)*nx + k] - mean[j+c]), cov[c] );
                }
            }

            for (int c = 0; c < LANE; ++c) {
                result[i + (j+c) * (size_t)ny] = (float)(cov[c] / (stdterm[i] * stdterm[j+c]));
            }
        }

        for (; j <= i; ++j) {
            double cov = 0.0;
            for (int k = 0; k < nx; ++k) {
                cov = fma((data[i*nx + k] - mean[i]), (data[j*nx + k] - mean[j]), cov);
            }

            result[i + j * (size_t)ny] = (float)(cov / (stdterm[i] * stdterm[j]));
        }
    }

    free(mean);
    free(stdterm);
}
