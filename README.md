# parallel_computing

## openmp  
sudo apt-get install gcc-multilib

gcc anyname.c -o anyname.out  
export OMP_NUM_THREADS=4  
gcc -fopenmp anyname.c -o anyname.out  
./anyname.out  

