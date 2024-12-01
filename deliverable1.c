/*
 *    Nicolò Bellè
 *    238178
 *    DELIVERABLE 1
 *    nicolo.belle@studenti.unitn.it
 */

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <ctype.h>
#include <string.h>
#include <errno.h>
#include <limits.h>

#define BLOCK_SIZE 64
#define N_RUNS 10
#ifdef _OPENMP
#include <omp.h>
double wt_symSequential, wt_symImplicit, wt_symOmp, wt_sequentialTransposition, wt_implicitTransposition, wt_ompTransposition;
#endif

//-->init n with scanf: matrix size
int init_matrixSize();

//-->check_input_(atoi)
bool check_invalid_input_atoi(int argc, char const *argv);

//-->init_random_float_matrix
float **init_random_float_matrix(const int r, const int c);

//-->printMatrix (delete internal comment to print all matrix)
void printMatrix(float **printMatrix, const int r, const int c);

//-->checkSymmetrical
bool checkSym(float **M, const int r, const int c);
bool checkSymImplicit(float **M, const int r, const int c);
bool checkSymOMP(float **M, const int r, const int c);

//-->transpose
float **matTranspose(float **Mc, const int r, const int c);
float **matTransposeImplicit(float **Mc, const int r, const int c);
float **matTransposeOMP(float **Mc, const int r, const int c);

//-->checkEqualMatrix
bool checkEqualMatrix(float **Mc, float **Tc, const int r, const int c);

int main(int argc, char const *argv[])
{
    srand(time(NULL));

    //----------start_input_size_matrix_n---------- (scanf)
    // int n = init_matrixSize(); //if uncomment line 'scanf' --> comment lines 'atoi'
    //----------           end           ----------

    //----------start_input_size_matrix_n---------- (atoi)
    if (check_invalid_input_atoi(argc, argv[1])) // if uncomment lines 'atoi' --> comment line 'scanf'
    {
        return 0;
    }
    int n = atoi(argv[1]);
    //----------           end           ----------

    //-->print_size_matrix_n
    printf("\n-------------------------------- MATRIX SIZE (n) = %d --------------------------------\n", n);

    //-->declarationMatrix
    float **T, **T_iP, **T_ompP;

    //-->init_random_float_matrix
    float **M = init_random_float_matrix(n, n);

    //-->checkSymmetric
    checkSym(M, n, n);
    checkSymImplicit(M, n, n);
    printf("CHECK SYMMETRIC (omp): ");
    if (checkSymOMP(M, n, n))
    {
        printf("SI\n");
    }
    else
    {
        printf("NO\n");
    }
    //-->print_wall_clock_time_symmetrical_check
#ifdef _OPENMP
    printf("\n-->WALL CLOCK TIME SYMMETRICAL CHECK:\n");
    printf("\t-SEQUENTIAL: wall clock time SYMMETRICAL check = %12.4g sec\n", wt_symSequential);
    printf("\t-IMPLICIT: wall clock time SYMMETRICAL check =   %12.4g sec\n", wt_symImplicit);
#endif

    //-->header_terminal
    printf("\t-OMP: wall clock time SYMMETRICAL check:");
    printf("\n\n\t N THREADS  |     AVG TIME     \n");
    //-->start_run_omp_symmetrical_check
    for (int num_threads = 1; num_threads <= 64; num_threads *= 2)
    {
        omp_set_num_threads(num_threads);
        checkSymOMP(M, n, n);
#ifdef _OPENMP
        printf("\t%11d |  %6.4g sec \n", num_threads, wt_symOmp);
#endif
    }

    //-->init_file_sequential_implicit
    FILE *file_si = fopen("deliverable1_imp_seq.csv", "a");
    if (file_si == NULL)
    {
        perror("ERROR: failed to open file\n");
        return 1;
    }
    //-->header_file_sequential_implicit
    if (n == 16)
    {
        fprintf(file_si, "SIZE_n;AVG_TIME_SEQ;AVG_TIME_IMP;\n");
    }

    //-->run_sequential_and_implicit_transposition
    double total_time_parallel_sequential = 0.0;
    double total_time_parallel_implicit = 0.0;
    for (int r = 0; r < N_RUNS; r++)
    {
        T = matTranspose(M, n, n);
        T_iP = matTransposeImplicit(M, n, n);
#ifdef _OPENMP
        total_time_parallel_sequential += wt_sequentialTransposition;
        total_time_parallel_implicit += wt_implicitTransposition;
#endif
    }
    //-->compute_avg_time_sequential_and_implicit_transposition
    double avg_time_parallel_sequential = total_time_parallel_sequential / N_RUNS;
    double avg_time_parallel_implicit = total_time_parallel_implicit / N_RUNS;

    //-->bandwidth_sequential_and_implicit_transposition
    double total_dataTransfered_bandwidth = 2.0 * n * n * sizeof(float);
    double bandwidth_sequential = total_dataTransfered_bandwidth / (avg_time_parallel_sequential * 1e9);
    double bandwidth_implicit = total_dataTransfered_bandwidth / (avg_time_parallel_implicit * 1e9);

    printf("\n-->STATS MATRICES TRANSPOSITION:\n");
    //-->print_stats_sequential
    printf("\t-SEQUENTIAL:");
    printf("\n\t average wall clock time TRANSPOSED MATRIX: %7.9f sec\n", avg_time_parallel_sequential);
    printf("\t effective BANDWIDTH: %3.3f GB/s\n", bandwidth_sequential);
    //-->print_stats_implicit
    printf("\n\t-IMPLICIT:");
    printf("\n\t average wall clock time TRANSPOSED MATRIX: %7.9f sec\n", avg_time_parallel_implicit);
    printf("\t effective BANDWIDTH: %3.3f GB/s\n", bandwidth_implicit);

    fprintf(file_si, "%11d;%7.9f;%7.9f\n", n, avg_time_parallel_sequential, avg_time_parallel_implicit);

    //-->close_csv
    fclose(file_si);

    //-->init_file_omp
    FILE *file = fopen("deliverable1_omp.csv", "a");
    if (file == NULL)
    {
        perror("ERROR: failed to open file\n");
        return 1;
    }
    //-->header_file
    fprintf(file, "\nRun: n = %d\n", n);
    fprintf(file, "NTHREADS;AVG_TIME;SPEEDUP;EFFICIENCY;BANDWIDTH\n");

    //-->print_stats_omp
    printf("\n\t-OMP:");
    printf("\n\t average wall clock time, speed up and efficiency TRANSPOSED MATRIX per threads:\n");

    //-->header_terminal
    printf("\n\t N THREADS  |     AVG TIME     |  SPEED UP  |  EFFICIENCY  |   BANDWIDTH\n");

    //-->run_omp
    for (int num_threads = 1; num_threads <= 64; num_threads *= 2)
    {
        omp_set_num_threads(num_threads);
        double total_time_parallel_omp = 0.0;
        for (int r = 0; r < N_RUNS; r++)
        {
            T_ompP = matTransposeOMP(M, n, n);
#ifdef _OPENMP
            total_time_parallel_omp += wt_ompTransposition;
#endif
        }
        //-->compute_stats_omp: avg time, speed up, efficency and bandwidth
        double avg_time_parallel_omp = total_time_parallel_omp / N_RUNS;
        double avg_speedup_omp = avg_time_parallel_sequential / avg_time_parallel_omp;
        double avg_efficiency_omp = avg_speedup_omp / num_threads;
        double bandwidth_omp = total_dataTransfered_bandwidth / (avg_time_parallel_omp * 1e9);
        //-->print_stats_terminal
        printf("\t%11d |  %7.9f sec | %10.2f | %11.2f%% | %3.3f\n", num_threads, avg_time_parallel_omp, avg_speedup_omp, avg_efficiency_omp * 100, bandwidth_omp);
        //-->print_stats_file_csv
        fprintf(file, "%11d;%7.9f;%6.2f;%11.2f%%;%3.3f\n", num_threads, avg_time_parallel_omp, avg_speedup_omp, avg_efficiency_omp * 100, bandwidth_omp);
    }

    //-->close_csv
    fclose(file);

    // checkEqualMatrix
    printf("\nExample to test the function: main matrix compared to transposed matrix (must be NOT EQUAL)");
    checkEqualMatrix(M, T, n, n);
    printf("\n");
    printf("Example to test the function: transposed matrix sequential compared to transposed matrix implicit (muste be EQUAL)");
    checkEqualMatrix(T, T_iP, n, n);
    printf("\n");

    // deallocate all matrix
    for (int i = 0; i < n; i++)
    {
        free(M[i]);
    }
    free(M);
    for (int i = 0; i < n; i++)
    {
        free(T[i]);
    }
    free(T);
    for (int i = 0; i < n; i++)
    {
        free(T_iP[i]);
    }
    free(T_iP);
    for (int i = 0; i < n; i++)
    {
        free(T_ompP[i]);
    }
    free(T_ompP);

    printf("\nAll matrices DEALLOCATED\n\n");

    return 0;
} //-->end

//-->init n with scanf: matrix size
int init_matrixSize()
{
    long int n;
    bool check_returnScanf, check_nPowerOfTwo, check_outOfRange;
    do
    {
        printf("\nINSERT n: ");
        fflush(stdin);
        check_returnScanf = scanf("%ld", &n);
        if (check_returnScanf != true || n <= 0)
        {
            printf("\nINVALID INPUT: 'n' must be an integer greater than 0!\n");
            check_returnScanf = false;
        }
        if (n < INT_MIN || n > INT_MAX)
        {
            printf("\nINVALID INPUT: 'n' out of integer range!\n");
            check_outOfRange = true;
        }
        else
        {
            check_outOfRange = false;
        }
        if ((n > 0) && ((n & (n - 1)) == 0))
        {
            check_nPowerOfTwo = true;
        }
        else
        {
            check_nPowerOfTwo = false;
        }
        if (check_nPowerOfTwo != true && check_returnScanf)
        {
            printf("\nINVALID INPUT: 'n' must be a power of two!\n");
        }
    } while (check_returnScanf != true || n <= 0 || check_nPowerOfTwo != true || check_outOfRange == true);
    return (int)n;
}

//-->check_input_(atoi)
bool check_invalid_input_atoi(int argc, char const *argv)
{

    bool is_in_argv = true, is_numeric = true, check_nPowerOfTwo;
    if (argc < 2)
    {
        printf("\nINVALID INPUT: insert size 'n'!\n\n");
        return true;
    }
    for (int i = 0; argv[i] != '\0'; i++)
    {
        if (!isdigit(argv[i]))
        {
            printf("\nINVALID INPUT: 'n' must be an integer greater than 0!\n\n");
            return true;
        }
    }
    char *endptr;
    errno = 0;
    long int n = strtol(argv, &endptr, 10);
    if (errno == ERANGE || n < INT_MIN || n > INT_MAX)
    {
        printf("\nINVALID INPUT: 'n' out of integer range!\n\n");
        return 1;
    }
    if (n <= 0)
    {
        printf("\nINVALID INPUT: 'n' must be an integer greater than 0!\n\n");
        return true;
    }
    if ((n > 0) && ((n & (n - 1)) == 0))
    {
        return false;
    }
    else
    {
        printf("\nINVALID INPUT: 'n' must be a power of two!\n\n");
        return true;
    }
}

//-->init_random_float_matrix
float **init_random_float_matrix(const int r, const int c)
{
    float **Mc = (float **)malloc(r * sizeof(float *));
    if (Mc == NULL)
    {
        printf("\nERROR: Memory allocation failed! (init matrix)\n");
        return NULL;
    }
    for (int i = 0; i < c; i++)
    {
        Mc[i] = (float *)malloc(c * sizeof(float));
        if (Mc[i] == NULL)
        {
            printf("\nERROR: Memory allocation failed for row %d! (init matrix)\n", i);
            return NULL;
        }
    }
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            Mc[i][j] = ((float)rand() / RAND_MAX) * 10.0;
        }
    }

    printf("\nNEW MATRIX: created\n");
    printMatrix(Mc, r, c);
    return Mc;
}

//-->printMatrix (delete internal comment to print matrix)
void printMatrix(float **printMatrix, const int r, const int c)
{
    /*
    if(printMatrix == NULL){
        printf("\nERROR: unable to print, matrix null!\n");
        return;
    }
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            printf("%0.3f ", printMatrix[i][j]);
        }
        printf("\n");
    }
    */
}

//-->checkSymmetrical_sequential
bool checkSym(float **M, const int r, const int c)
{
    if (M == NULL)
    {
        printf("\nERROR: unable to check symmetric, matrix null! (sequential)\n");
        return false;
    }
    bool check = true;
    printf("\nCHECK SYMMETRIC (sequential): ");
#ifdef _OPENMP
    double wt1_sym, wt2_sym;
    wt1_sym = omp_get_wtime();
#endif
    for (int i = 0; i < r; i++)
    {
        for (int j = i + 1; j < c; j++)
        {
            if (M[i][j] != M[j][i])
                check = false;
        }
    }
#ifdef _OPENMP
    wt2_sym = omp_get_wtime();
    wt_symSequential = wt2_sym - wt1_sym;
#endif
    if (check)
    {
        printf("SI\n");
    }
    else
    {
        printf("NO\n");
    }
    return check;
}

//-->checkSymmetrical_implicit
bool checkSymImplicit(float **M, const int r, const int c)
{
    if (M == NULL)
    {
        printf("ERROR: unable to check symmetric, matrix null! (implicit)\n");
        return false;
    }
    bool check = true;
    printf("CHECK SYMMETRIC (implicit): ");
#ifdef _OPENMP
    double wt1_symI, wt2_symI;
    wt1_symI = omp_get_wtime();
#endif
    for (int i = 0; i < r; i++)
    {
        #pragma simd
        #pragma unroll(4)
        for (int j = i + 1; j < c; j++)
        {
            if (M[i][j] != M[j][i])
                check = false;
        }
    }
#ifdef _OPENMP
    wt2_symI = omp_get_wtime();
    wt_symImplicit = wt2_symI - wt1_symI;
#endif
    if (check)
    {
        printf("SI\n");
    }
    else
    {
        printf("NO\n");
    }
    return check;
}

//-->checkSymmetrical_omp
bool checkSymOMP(float **M, const int r, const int c)
{
    if (M == NULL)
    {
        printf("ERROR: unable to check symmetric, matrix null! (omp)\n");
        return false;
    }
    bool check = true;

#ifdef _OPENMP
    double wt1_symOMP, wt2_symOMP;
    wt1_symOMP = omp_get_wtime();
#endif
#pragma omp parallel for shared(check)
    for (int i = 0; i < r; i++)
    {
#pragma omp flush(check)
        for (int j = i + 1; j < c; j++)
        {
            if (M[i][j] != M[j][i])
            {
                check = false;
            }
        }
    }
#ifdef _OPENMP
    wt2_symOMP = omp_get_wtime();
    wt_symOmp = wt2_symOMP - wt1_symOMP;
#endif
    return check;
}

//-->transpose_sequential
float **matTranspose(float **Mc, const int r, const int c)
{
    if (Mc == NULL)
    {
        printf("\nERROR: unable to transpose, matrix null! (sequential)\n");
        return NULL;
    }
    float **Tc = (float **)malloc(r * sizeof(float *));
    if (Tc == NULL)
    {
        printf("\nERROR: Memory allocation failed! (sequential)\n");
        return NULL;
    }
    for (int i = 0; i < c; i++)
    {
        Tc[i] = (float *)malloc(c * sizeof(float));
        if (Tc[i] == NULL)
        {
            printf("\nERROR: Memory allocation failed for row %d! (sequential)\n", i);
            return NULL;
        }
    }
#ifdef _OPENMP
    double wt1, wt2;
    wt1 = omp_get_wtime();
#endif
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            Tc[j][i] = Mc[i][j];
        }
    }
#ifdef _OPENMP
    wt2 = omp_get_wtime();
    wt_sequentialTransposition = wt2 - wt1;
#endif
    // printf("\nTRANSPOSED MATRIX (sequential): done\n");
    printMatrix(Tc, r, c);
    return Tc;
}

//-->transpose_implicit
float **matTransposeImplicit(float **Mc, const int r, const int c)
{
    if (Mc == NULL)
    {
        printf("ERROR: unable to transpose, matrix null! (implicit)\n");
        return NULL;
    }
    float **T_iPc = (float **)malloc(r * sizeof(float *));
    if (T_iPc == NULL)
    {
        printf("ERROR: Memory allocation failed! (implicit)\n");
        return NULL;
    }
    for (int i = 0; i < c; i++)
    {
        T_iPc[i] = (float *)malloc(c * sizeof(float));
        if (T_iPc[i] == NULL)
        {
            printf("ERROR: Memory allocation failed for row %d! (implicit)\n", i);
            return NULL;
        }
    }
#ifdef _OPENMP
    double wtTIMP1, wtTIMP2;
    wtTIMP1 = omp_get_wtime();
#endif
    for (int i = 0; i < r; i += BLOCK_SIZE)
    {
        for (int j = 0; j < c; j += BLOCK_SIZE)
        {
            for (int bi = i; bi < i + BLOCK_SIZE && bi < r; bi++)
            {
                for (int bj = j; bj < j + BLOCK_SIZE && bj < c; bj++)
                {
                    T_iPc[bj][bi] = Mc[bi][bj];
                }
            }
        }
    }
#ifdef _OPENMP
    wtTIMP2 = omp_get_wtime();
    wt_implicitTransposition = wtTIMP2 - wtTIMP1;
#endif
    // printf("TRANSPOSED MATRIX (implicit): done\n");
    printMatrix(T_iPc, r, c);
    return T_iPc;
}

//-->transpose_omp
float **matTransposeOMP(float **Mc, const int r, const int c)
{
    if (Mc == NULL)
    {
        printf("ERROR: unable to transpose, matrix null! (omp)\n");
        return NULL;
    }
    float **T_ompPc = (float **)malloc(r * sizeof(float *));
    if (T_ompPc == NULL)
    {
        printf("ERROR: Memory allocation failed! (omp)\n");
        return NULL;
    }
    for (int i = 0; i < c; i++)
    {
        T_ompPc[i] = (float *)malloc(c * sizeof(float));
        if (T_ompPc[i] == NULL)
        {
            printf("ERROR: Memory allocation failed for row %d! (omp)\n", i);
            return NULL;
        }
    }
#ifdef _OPENMP
    double wtTOMP1, wtTOMP2;
    wtTOMP1 = omp_get_wtime();
#endif
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < r; i += BLOCK_SIZE)
    {
        for (int j = 0; j < c; j += BLOCK_SIZE)
        {
            for (int bi = i; bi < i + BLOCK_SIZE && bi < r; bi++)
            {
                for (int bj = j; bj < j + BLOCK_SIZE && bj < c; bj++)
                {
                    T_ompPc[bj][bi] = Mc[bi][bj];
                }
            }
        }
    }

#ifdef _OPENMP
    wtTOMP2 = omp_get_wtime();
    wt_ompTransposition = wtTOMP2 - wtTOMP1;
#endif
    // printf("TRANSPOSED MATRIX (omp): done\n");
    printMatrix(T_ompPc, r, c);
    return T_ompPc;
}

//-->checkEqualMatrix
bool checkEqualMatrix(float **Mc, float **Tc, const int r, const int c)
{
    if (Mc == NULL || Tc == NULL)
    {
        printf("ERROR: unable to check symmetric, matrix null! (implicit)\n");
        return false;
    }
    printf("\n-->CHECK EQUAL: ");
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            if (Mc[i][j] != Tc[i][j])
            {
                printf("NO\n");
                return false;
            }
        }
    }
    printf("SI\n");
    return true;
}
