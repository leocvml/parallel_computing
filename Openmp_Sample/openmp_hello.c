#include <omp.h>
#include <stdlib.h>
#include <stdio.h>

int main(){

	#pragma omp parallel num_threads(3) 
	{
		int i ;
		for( i = 0 ; i<3; i++)
		printf("helloworld , i am %d , iter%d \n" , omp_get_thread_num() , i );
	}
	

	
	{
		int i ;
		#pragma omp parallel for num_threads(3)
		for(i = 0 ; i < 3; i++)
		printf("hello i am %d , iter %d \n" , omp_get_thread_num() , i);

	}
	return 0;





}
