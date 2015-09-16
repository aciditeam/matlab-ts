% This make.m is used under Windows

mex -Dchar16_t=uint16_t -O -largeArrayDims -c ../blas/daxpy.c ../blas/ddot.c ../blas/dnrm2.c ../blas/dscal.c -outdir ../blas
mex -Dchar16_t=uint16_t -O -largeArrayDims -c ../linear.cpp
mex -Dchar16_t=uint16_t -O -largeArrayDims -c ../tron.cpp
mex -Dchar16_t=uint16_t -O -largeArrayDims -c linear_model_matlab.c -I../
mex -Dchar16_t=uint16_t -O -largeArrayDims train.c -I../ tron.o linear.o linear_model_matlab.o ../blas/daxpy.o ../blas/ddot.o ../blas/dnrm2.o ../blas/dscal.o
mex -Dchar16_t=uint16_t -O -largeArrayDims predict.c -I../ tron.o linear.o linear_model_matlab.o ../blas/daxpy.o ../blas/ddot.o ../blas/dnrm2.o ../blas/dscal.o
mex -Dchar16_t=uint16_t -O -largeArrayDims libsvmread.c
mex -Dchar16_t=uint16_t -O -largeArrayDims libsvmwrite.c
