#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(){

	int x = 3;
	int y = 4;

	omp_lock_t lock;
	omp_init_lock(&lock);

	#pragma omp parallel num_threads(3) shared(x,y)
	{
		omp_set_lock(&lock);

		x += omp_get_thread_num();
		y += omp_get_thread_num();

		omp_unset_lock(&lock);
	}

	printf("x = %d , y = %d\n" , x ,y );
	omp_destroy_lock(&lock);

	return 0;


}
