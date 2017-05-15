#include <omp.h>
#include <stdlib.h>
#include <stdio.h>

int facobi(int num){
		if (( 0 == num) || 1 ==num)
			return 1;	
	
	int f1, f2;;
	#pragma omp task shared(f1)
	f1 = facobi(num-1);
	#pragma omp task shared(f2)
	f2 = facobi(num-2);
	#pragma omp taskwait
	return f1 + f2;

}

int main(){

	int r ;

	#pragma omp parallel shared(r)
	{
		#pragma omp single
		r = facobi(5);
	}
		printf("%d\n" , r);
return 0;

}
