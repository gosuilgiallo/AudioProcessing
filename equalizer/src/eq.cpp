/*
    This equalizer divides the frequency spectrum into three bands, and for each one applies a gain.
    The bands are defined using normalized frequencies (0-1) where 1 corresponds to the Nyquist frequency.

    COMPILE:
    g++ -o eq_omp eq_omp.cpp -lsndfile -lfftw3f -fopenmp 
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <sndfile.h>
#include <fftw3.h>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <omp.h>

// Constants for the bands (normalized frequencies 0-1)
const float LOW_BAND_LIMIT = 0.0136f;   // ~300 Hz @ 44.1kHz
const float MID_BAND_LIMIT = 0.136f;    // ~3000 Hz @ 44.1kHz
const float LOW_GAIN = -60.0f;
const float MID_GAIN = 2.0f;
const float HIGH_GAIN = -3.0f;

void parallelEQ_omp(float* real, float* imag, int numSamples) {
    // Convert normalized frequencies to bin indices
    int lowBandEnd = static_cast<int>(LOW_BAND_LIMIT * numSamples);
    int midBandEnd = static_cast<int>(MID_BAND_LIMIT * numSamples);

    // gain
    __m128 lowGain = _mm_set1_ps(std::pow(10.0f, LOW_GAIN / 20.0f));  
    __m128 midGain = _mm_set1_ps(std::pow(10.0f, MID_GAIN / 20.0f));   
    __m128 highGain = _mm_set1_ps(std::pow(10.0f, HIGH_GAIN / 20.0f)); 

    // Process low frequencies
    #pragma omp parallel for 
    for (int i = 0; i < lowBandEnd; i += 4) {
        _mm_store_ps(&real[i], _mm_mul_ps(lowGain, _mm_load_ps(&real[i])));
        _mm_store_ps(&imag[i], _mm_mul_ps(lowGain, _mm_load_ps(&imag[i])));
    }

    // Process mid frequencies
    #pragma omp parallel for 
    for (int i = lowBandEnd/4; i < midBandEnd/4; i++) {
        _mm_store_ps(&real[4*i], _mm_mul_ps(midGain, _mm_load_ps(&real[4*i])));
        _mm_store_ps(&imag[4*i], _mm_mul_ps(midGain, _mm_load_ps(&imag[4*i])));
    }

    // Process high frequencies
    #pragma omp parallel for 
    for (int i = midBandEnd/4; i < numSamples/4; i++) {
        _mm_store_ps(&real[4*i], _mm_mul_ps(highGain, _mm_load_ps(&real[4*i])));
        _mm_store_ps(&imag[4*i], _mm_mul_ps(highGain, _mm_load_ps(&imag[4*i])));
    }
}

int main(int argc, char* argv[]) {
    clock_t start, end;
    start = clock();

    const char* inputFile = "../samples/fullSong.wav";
    const char* outputFile = "../samples/fullSong_omp.wav";
    SF_INFO sfInfo;
    SNDFILE* inFile = sf_open(inputFile, SFM_READ, &sfInfo);

    if (!inFile) {
        std::cerr << "[ERR] in file " << inputFile << std::endl;
        return 1;
    }

    int numSamples = sfInfo.frames * sfInfo.channels;

    float* real = (float*)std::aligned_alloc(64, sizeof(float) * numSamples);
    float* imag = (float*)std::aligned_alloc(64, sizeof(float) * numSamples);
    std::vector<short> buffer(numSamples);

    sf_read_short(inFile, buffer.data(), numSamples);
    sf_close(inFile);

    #pragma omp parallel for
    for (int i = 0; i < numSamples; ++i) {
        real[i] = static_cast<float>(buffer[i]);
    }
    std::memset(imag, 0, sizeof(float) * numSamples);

    // FFT & IFFT
    fftwf_complex* fftData = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * numSamples);
    fftwf_complex* ifftData = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * numSamples);
    fftwf_plan forwardPlan = fftwf_plan_dft_r2c_1d(numSamples, real, fftData, FFTW_ESTIMATE);
    fftwf_plan inversePlan = fftwf_plan_dft_c2r_1d(numSamples, fftData, real, FFTW_ESTIMATE);

    // FFT
    fftwf_execute(forwardPlan);

    // copy data in two arrays
    #pragma omp parallel for
    for (int i = 0; i < numSamples / 2 + 1; ++i) {
        real[i] = fftData[i][0];
        imag[i] = fftData[i][1];
    }

    // applies EQ
    parallelEQ_omp(real, imag, numSamples / 2);

    // copy data to prepare the Inverse FFT
    #pragma omp parallel for
    for (int i = 0; i < numSamples / 2 + 1; ++i) {
        fftData[i][0] = real[i];
        fftData[i][1] = imag[i];
    }

    // IFFT
    fftwf_execute(inversePlan);

    // normalization
    double normalFactor = 1.0 / numSamples;
    #pragma omp parallel for
    for (int i = 0; i < numSamples; ++i) {
        real[i] *= normalFactor;
    }

    // converts to short to save on out file
    #pragma omp parallel for
    for (int i = 0; i < numSamples; ++i) {
        buffer[i] = static_cast<short>(std::round(real[i]));
    }

    // saves equalized audio file
    SNDFILE* outFile = sf_open(outputFile, SFM_WRITE, &sfInfo);
    if (!outFile) {
        std::cerr << "[ERR] in file " << outputFile << std::endl;
        return 1;
    }

    sf_write_short(outFile, buffer.data(), numSamples);
    sf_close(outFile);

    // free
    fftwf_destroy_plan(forwardPlan);
    fftwf_destroy_plan(inversePlan);
    fftwf_free(fftData);
    fftwf_free(ifftData);
    std::free(real);
    std::free(imag);

    end = clock();
    printf("\tElapsed time: %.3fs\n", (double)(end-start) / CLOCKS_PER_SEC);
    
    return 0;
}
