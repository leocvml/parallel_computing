#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define rank 512
void T(double *A , double *A_T , int n){

	int i , j ;
	for( i = 0 ; i < n ;i++)
	{
		for( j = 0 ; j < n ; j++)
		{
			A_T[j*n+i] = A[i*n+j];		
		}
	}

}


void mul(double *A, double *B, double *C, int n) 
{   
    int i, j, k;
    for (i = 0; i < n; i++) { 
        for (j = 0; j < n; j++) {
            double temp  = 0;
            for (k = 0; k < n; k++) {
                temp += A[i*n+k]*B[k*n+j];
            } 
            C[i*n+j ] = temp;
        }
    }
}

void mul_omp(double *A, double *B, double *C, int n) 
{   
    #pragma omp parallel
    {
        int i, j, k;
        #pragma omp for
        for (i = 0; i < n; i++) { 
            for (j = 0; j < n; j++) {
                double temp  = 0;
                for (k = 0; k < n; k++) {
                    temp += A[i*n+k]*B[k*n+j];
                } 
                C[i*n+j ] = temp;
            }
        }

    }
}

void mul_T(double *A, double *B, double *C, int n) 
{   
    int i, j, k;
    double *BT;
    BT = (double*)malloc(sizeof(double)*n*n);
    T(B,BT, n);
    for (i = 0; i < n; i++) { 
        for (j = 0; j < n; j++) {
            double temp  = 0;
            for (k = 0; k < n; k++) {
                temp += A[i*n+k]*BT[j*n+k];
            } 
            C[i*n+j ] = temp;
        }
    }
    free(BT);
}

void mul_T_omp(double *A, double *B, double *C, int n) 
{   
    double *BT;
    BT = (double*)malloc(sizeof(double)*n*n);
    T(B,BT, n);
    #pragma omp parallel
    {
        int i, j, k;
        #pragma omp for
        for (i = 0; i < n; i++) { 
            for (j = 0; j < n; j++) {
                double temp  = 0;
                for (k = 0; k < n; k++) {
                    temp += A[i*n+k]*BT[j*n+k];
                } 
                C[i*n+j ] = temp;
            }
        }

    }
    free(BT);
}

int main() {
    int i, n;
    double *A, *B, *C, *BT, execute_time;
	
    n=512;
    A = (double*)malloc(sizeof(double)*n*n);
    B = (double*)malloc(sizeof(double)*n*n);
    C = (double*)malloc(sizeof(double)*n*n);
    BT = (double*)malloc(sizeof(double)*n*n);
    for(i=0; i<n*n; i++) { A[i] = i; B[i] = i;} //A[i] = rand()/RAND_MAX; B[i] = rand()/RAND_MAX
	
	
	

	
	
	for(i = 0 ; i< 10; i++)
	printf(":%lf \n",B[i]);
	
	T(B,BT,n);

	for (i = 0 ; i < 100 ; i+=10)
	{	
		printf("BT %d:%lf  ",i,BT[i]);
	}
	printf("\n");
	




    execute_time = omp_get_wtime();
    mul(A,B,C, n);
    execute_time = omp_get_wtime() - execute_time;
    printf("mul:%lf\n", execute_time);

    execute_time = omp_get_wtime();
    mul_omp(A,B,C, n);
    execute_time = omp_get_wtime() - execute_time;
    printf("mul_omp%lf\n", execute_time);

    execute_time = omp_get_wtime();
    mul_T(A,B,C, n);
    execute_time = omp_get_wtime() - execute_time;
    printf("mul_T:%lf\n", execute_time);

    execute_time = omp_get_wtime();
    mul_T_omp(A,B,C, n);
    execute_time = omp_get_wtime() - execute_time;
    printf("mul_T_omp%lf\n", execute_time);

    return 0;

}