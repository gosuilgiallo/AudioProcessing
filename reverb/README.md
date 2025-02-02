# Reverb in C++ #
This folder contains a convolution reverb implemented in C++.

## How to run? ##
### To compile ###
```
g++ reverb.cpp -o reverb -lsndfile -lfftw3f -march=native -O3
```

### To run ###
```
./reverb
```

## Thoughts ##
### Parallel computation ###
Previous experiments were tried with **SSE2** functions and different **omp** directives. The significative changes respect to the scalar implementation is only on the convolution loop, but the result can depend on the CPU.

### The samples ###
The samples provided in this folder are written, produced and recorded by me and free to use. However, these samples are just for example. You may want to add your personal WAV files (songs, loops or impulse reponses). The example samples are 16 bit-depth and 441000 Hz sample rate.
