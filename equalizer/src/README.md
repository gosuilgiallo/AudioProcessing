# Source files #
Here the C++ source files. More information about the project on parent folder.
## Updates ##
### The sample rate ###
So far, the implementations were only for audio formats that are aligned to the CD standard, so a frequency rate of 44100 Hz and a bit depth of 16. However, in modern DAWs (and other scenarios) this parameters could change, 
so my goal is to build a free software that is independent from these constraints.
The first change concerns the sample rate: not hard but there are some tricks to watch out for. First, sample rate is crucial to compute the band limits, so the bounds are normalized to Nyquist freqeuency (FREQ_LIMIT/22050).
Second, sample rate is important to read/write the `wav` files, but this is easily solvable using the `sndfile` library, that gives us the file metadata.
