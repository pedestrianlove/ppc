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
    double* n_data = (double*)malloc(ny*nx * sizeof(double));

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
        double std[LANE] = {0.0};
        for (int k = 0; k < nx; ++k) {
            for (int c = 0; c < LANE; ++c) {
                double diff = data[(i+c)*nx + k] - mean[i+c];
                varsum[c] += diff*diff;
                n_data[(i+c)*nx + k] = diff;
            }
        }
        for (int c = 0; c < LANE; ++c) {
            std[c] = sqrt(varsum[c]);
        }
        for (int k = 0; k < nx; ++k) {
            for (int c = 0; c < LANE; ++c) {
                n_data[(i+c)*nx + k] /= std[c];
            }
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
            n_data[i*nx + k] = diff;
        }
        /* stdterm[i] holds Σ (a_i,k – mean[i])^2 */
        double std = sqrt(varsum);
        for (int k = 0; k < nx; ++k) {
            n_data[i*nx + k] /= std;
        }
    }

    /* 2) For each pair (i, j) with j <= i, compute the covariance and then the correlation. */
    for (int i = 0; i < ny; ++i) {
        int j;
        for (j = 0; j <= i-(LANE-1); j+=LANE) {
            double cov[LANE] = {0.0};
            for (int k = 0; k < nx; ++k) {
                for (int c = 0; c < LANE; ++c) {
                    cov[c] = fma(n_data[i*nx + k], n_data[(j+c)*nx + k], cov[c] );
                }
            }

            for (int c = 0; c < LANE; ++c) {
                result[i + (j+c) * (size_t)ny] = (float)(cov[c]);
            }
        }

        for (; j <= i; ++j) {
            double cov = 0.0;
            for (int k = 0; k < nx; ++k) {
                cov = fma(n_data[i*nx + k], n_data[j*nx + k] , cov);
            }

            result[i + j * (size_t)ny] = (float)(cov);
        }
    }

    free(mean);
    free(n_data);
}
