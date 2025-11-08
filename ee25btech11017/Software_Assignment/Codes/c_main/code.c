#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double dot(const double *a, const double *b, int n) {
    double s = 0;
    for (int i = 0; i < n; i++) {
        s += a[i] * b[i];
    }
    return s;
}

double norm(const double *x, int n) {
    return sqrt(dot(x, x, n));
}

void Ax(const double *A, const double *x, double *y, int r, int c) {
    for (int i = 0; i < r; i++) {
        double s = 0;
        for (int j = 0; j < c; j++) {
            s += A[i * c + j] * x[j];
        }
        y[i] = s;
    }
}

void ATx(const double *A, const double *x, double *y, int r, int c) {
    for (int j = 0; j < c; j++) {
        double s = 0;
        for (int i = 0; i < r; i++) {
            s += A[i * c + j] * x[i];
        }
        y[j] = s;
    }
}

void top_svd(const double *A, int r, int c, double *u, double *v, double *s) {
    for (int j = 0; j < c; j++) {
        v[j] = 1.0;
    }
    double *Av = (double *)malloc(sizeof(double) * r);
    double *ATu = (double *)malloc(sizeof(double) * c);
    for (int t = 0; t < 15; t++) {
        Ax(A, v, Av, r, c);
        double n1 = fmax(1e-12, norm(Av, r));
        for (int i = 0; i < r; i++) {
            u[i] = Av[i] / n1;
        }
        ATx(A, u, ATu, r, c);
        double n2 = fmax(1e-12, norm(ATu, c));
        for (int j = 0; j < c; j++) {
            v[j] = ATu[j] / n2;
        }
    }
    Ax(A, v, Av, r, c);
    *s = norm(Av, r);
    free(Av);
    free(ATu);
}

// Function to compute Frobenius norm of a matrix
double frob_norm(const double *X, int h, int w) {
    size_t N = (size_t)h * (size_t)w;
    double sum = 0.0;
    for (size_t i = 0; i < N; i++) {
        sum += X[i] * X[i];
    }
    return sqrt(sum);
}

// Function to compute Frobenius error ||A - Ak||_F
double frob_error(const double *A, const double *Ak, int h, int w) {
    size_t N = (size_t)h * (size_t)w;
    double sum = 0.0;
    for (size_t i = 0; i < N; i++) {
        double diff = A[i] - Ak[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}


int main(void) {
    const char *infile = "greyscale.jpg";
    const char *outfile = "greyscale_k100.jpg";
    const int k = 100;

    FILE *fp = fopen(infile, "rb");
    if (!fp) {
        printf("open fail\n");
        return 1;
    }
    //  safer file read 
fseek(fp, 0, SEEK_END);
long sz = ftell(fp);
fseek(fp, 0, SEEK_SET);

if (sz <= 0) {
    fclose(fp);
    printf("bad size\n");
    return 1;
}

unsigned char *buf = (unsigned char *)malloc((size_t)sz);
if (!buf) {
    fclose(fp);
    printf("alloc fail (buf)\n");
    return 1;
}

size_t rd = fread(buf, 1, (size_t)sz, fp);
fclose(fp);
if (rd != (size_t)sz) {
    free(buf);
    printf("read fail\n");
    return 1;
}

   int w, h, c_in;

// Load the image as a single-channel greyscale directly (last argument = 1)
unsigned char *img = stbi_load_from_memory(buf, (int)sz, &w, &h, &c_in, 1);
free(buf);

if (!img) {
    printf("decode fail\n");
    return 2;
}
if (k < 1 || k > (h < w ? h : w)) {
    printf("bad k\n");
    stbi_image_free(img);
    return 3;
}

// Allocate matrix A for greyscale Pixels
double *A = (double *)malloc((size_t)h * (size_t)w * sizeof(double));
if (!A) {
    printf("allocation failed for A\n");
    stbi_image_free(img);
    return 4;
}

// Copy pixel data (each pixel already greyscale)
for (size_t t = 0, N = (size_t)h * (size_t)w; t < N; t++) {
    A[t] = (double)img[t]; 
}

    stbi_image_free(img);

    double *M = (double *)malloc((size_t)h * (size_t)w * sizeof(double));
    for (size_t t = 0; t < (size_t)h * (size_t)w; t++) {
        M[t] = A[t];
    }

    double **U = (double **)malloc(k * sizeof(double *));
    double **V = (double **)malloc(k * sizeof(double *));
    double *S = (double *)malloc(k * sizeof(double));
    for (int t = 0; t < k; t++) {
        U[t] = (double *)malloc(h * sizeof(double));
        V[t] = (double *)malloc(w * sizeof(double));
    }

    for (int t = 0; t < k; t++) {
        top_svd(M, h, w, U[t], V[t], &S[t]);
        for (int i = 0; i < h; i++) {
            double ui = U[t][i] * S[t];
            for (int j = 0; j < w; j++) {
                M[i * w + j] -= ui * V[t][j];
            }
        }
    }

  
    double *R = (double *)calloc((size_t)h * (size_t)w, sizeof(double));
    for (int t = 0; t < k; t++) {
        for (int i = 0; i < h; i++) {
            double ui = U[t][i] * S[t];
            for (int j = 0; j < w; j++) {
                R[i * w + j] += ui * V[t][j];
            }
        }
    }

  // Compute Frobenius error between original and reconstructed image 
double errorF = frob_error(A, R, h, w);
double normA = frob_norm(A, h, w);
double relative_error = (errorF / normA)*100;
printf("Frobenius Error (||A - Ak||_F): %.4f\n", errorF);
printf("Relative Error: %.4f\n", relative_error);


    unsigned char *out = (unsigned char *)malloc((size_t)h * (size_t)w);
    for (size_t q = 0; q < (size_t)h * (size_t)w; q++) {
        double v = R[q];
        if (v < 0) {
            v = 0;
        }
        if (v > 255) {
            v = 255;
        }
        out[q] = (unsigned char)(v + 0.5);
    }

    stbi_write_jpg(outfile, w, h, 1, out, 90);

    for (int t = 0; t < k; t++) {
        free(U[t]);
    }
    for (int t = 0; t < k; t++) {
        free(V[t]);
    }
    free(U);
    free(V);
    free(S);
    free(M);
    free(A);
    free(R);
    free(out);

    return 0;
}
