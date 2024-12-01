# IntroPARCO-2024-H1

### Tools and Libraries
- **gcc** 9.1.0 version
- **omp.h** (and a compiler that supports OpenMP)

## Running the program with ```deliverable1.pbs```
To reproduce the project you need to have in the same directory these two files that you can find in this repository:
- `deliverable1.pbs`
- `deliverable1.c`
  
IMPORTANT: change the path within the file ```deliverable1.pbs``` by setting the correct one where these two files are located

To run deliverable1.pbs, use the following command:
```bash
qsub deliverable1.pbs
```
Within this file there are several controls (in case of error will be displayed the motivation in the file ```deliverable1.o```):
- if ```delivereble1.c``` is not found in the same directory
- if ```delivereble1``` is not created after compiling ```deliverable1.c``` due to errors reported in the ```deliverable1.e```

Also, within the ```deliverable1.pbs``` file, it is specified that the files ```deliverable1_omp.csv``` and ```deliverable1_imp_seq.csv``` created on the ```deliverable1.c``` are deleted at each execution, being they in append for need.

## Running the program individually
Compile the file ```deliverable1.c``` with this command:

```bash
gcc -o deliverable1 deliverable1.c -fopenmp
```

This will create an executable named ```deliverable1```.
Now, you can use this command by replacing ```size``` with the size of the matrix:

```bash
./deliverable1 size
```

## Example of output 
Here is an example of the output generated by the program:
```bash

-------------------------------- MATRIX SIZE (n) = 4096 --------------------------------

NEW MATRIX: created

CHECK SYMMETRIC (sequential): NO
CHECK SYMMETRIC (implicit): NO
CHECK SYMMETRIC (omp): NO

-->WALL CLOCK TIME SYMMETRICAL CHECK:
	wall clock time SYMMETRICAL check (sequential) =      0.05936 sec
	wall clock time SYMMETRICAL check (implicit) =        0.05771 sec
	wall clock time SYMMETRICAL check (omp) =             0.04013 sec

-->STATS MATRICES TRANSPOSITION:
	-SEQUENTIAL:
	 average wall clock time TRANSPOSED MATRIX: 0.129468390 sec
	 effective BANDWIDTH: 1.037 GB/s

	-IMPLICIT:
	 average wall clock time TRANSPOSED MATRIX: 0.097037317 sec
	 effective BANDWIDTH: 1.383 GB/s

	-OMP:
	 average wall clock time, speed up and efficiency TRANSPOSED MATRIX per threads:

	 N THREADS  |     AVG TIME     |  SPEED UP  |  EFFICIENCY  |   BANDWIDTH
	          1 |  0.088833956 sec |       1.46 |      145.74% | 1.511
	          2 |  0.068224338 sec |       1.90 |       94.88% | 1.967
	          4 |  0.036467781 sec |       3.55 |       88.76% | 3.680
	          8 |  0.022005448 sec |       5.88 |       73.54% | 6.099
	         16 |  0.014156166 sec |       9.15 |       57.16% | 9.481
	         32 |  0.010073677 sec |      12.85 |       40.16% | 13.324
	         64 |  0.009200799 sec |      14.07 |       21.99% | 14.588

All matrices DEALLOCATED

```
