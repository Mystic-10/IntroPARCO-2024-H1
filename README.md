# IntroPARCO-2024-H1

Nicolò Bellè

unitn mail: nicolo.belle@studenti.unitn.it - ID: 238178

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

To change the size of the matrix you can modify the first parameter of the following command within the file `delivereble1.pbs`:
```bash
./deliverable1 size     # Replace size
```

## Running the program individually
You need to have ```deliverable1.c``` in the directory.

Compile the file ```deliverable1.c``` with this command:
```bash
gcc -o deliverable1 deliverable1.c -fopenmp -O2
```

This will create an executable named ```deliverable1```.

Now, you can use the following command by replacing ```size``` with the size of the matrix:
```bash
./deliverable1 size     # Replace size
```

IMPORTANT: remember to remove ```deliverable1_omp.csv``` and ```deliverable1_imp_seq.csv``` becouse they are opened in append.
You can do this by using this command:
```bash
rm deliverable1_omp.csv deliverable1_imp_seq.csv
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
        -SEQUENTIAL: wall clock time SYMMETRICAL check =      0.04416 sec
        -IMPLICIT: wall clock time SYMMETRICAL check =        0.04114 sec
        -OMP: wall clock time SYMMETRICAL check:

         N THREADS  |     AVG TIME
                  1 |  0.04967 sec
                  2 |  0.04151 sec
                  4 |  0.02301 sec
                  8 |  0.01631 sec
                 16 |  0.01021 sec
                 32 |  0.007373 sec
                 64 |  0.002785 sec

-->STATS MATRICES TRANSPOSITION:
        -SEQUENTIAL:
         average wall clock time TRANSPOSED MATRIX: 0.146035025 sec
         effective BANDWIDTH: 0.919 GB/s

        -IMPLICIT:
         average wall clock time TRANSPOSED MATRIX: 0.116103744 sec
         effective BANDWIDTH: 1.156 GB/s

        -OMP:
         average wall clock time, speed up and efficiency TRANSPOSED MATRIX per threads:

         N THREADS  |     AVG TIME     |  SPEED UP  |  EFFICIENCY  |   BANDWIDTH
                  1 |  0.112915087 sec |       1.29 |      129.33% | 1.189
                  2 |  0.071479276 sec |       2.04 |      102.15% | 1.878
                  4 |  0.039301236 sec |       3.72 |       92.89% | 3.415
                  8 |  0.023619071 sec |       6.18 |       77.29% | 5.683
                 16 |  0.015217980 sec |       9.60 |       59.98% | 8.820
                 32 |  0.010164032 sec |      14.37 |       44.90% | 13.205
                 64 |  0.008088565 sec |      18.05 |       28.21% | 16.594

All matrices DEALLOCATED

```
