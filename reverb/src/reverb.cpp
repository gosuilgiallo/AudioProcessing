#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <sndfile.h>
#include <fftw3.h>
#include <immintrin.h>

/*
    to compile:
    g++ reverb.cpp -o reverb -lsndfile -lfftw3f -march=native -O3
*/

// FFT function
std::vector<std::complex<float>> fft(const std::vector<float>& input) {
    int N = input.size();
    fftwf_complex *out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(N, const_cast<float*>(input.data()), out, FFTW_ESTIMATE);
    
    fftwf_execute(plan);
    
    std::vector<std::complex<float>> result(N/2 + 1);
    for (int i = 0; i < N/2 + 1; ++i) {
        result[i] = std::complex<float>(out[i][0], out[i][1]);
    }
    
    fftwf_destroy_plan(plan);
    fftwf_free(out);
    return result;
}

// Inverse FFT function
std::vector<float> ifft(const std::vector<std::complex<float>>& input, int originalSize) {
    int N = originalSize;
    fftwf_complex *in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);
    float *out = (float*) fftwf_malloc(sizeof(float) * N);
    
    // Convert complex input back to fftwf_complex
    for (int i = 0; i < N/2 + 1; ++i) {
        in[i][0] = input[i].real();
        in[i][1] = input[i].imag();
    }
    
    fftwf_plan plan = fftwf_plan_dft_c2r_1d(N, in, out, FFTW_ESTIMATE);
    fftwf_execute(plan);
    
    std::vector<float> result(N);
    for (int i = 0; i < N; ++i) {
        result[i] = out[i] / N;  // Normalize
    }
    
    fftwf_destroy_plan(plan);
    fftwf_free(in);
    fftwf_free(out);
    return result;
}

// Normalize audio function
void normalizeAudio(std::vector<float>& audio) {
    float maxVal = 0;
    for (float sample : audio) {
        maxVal = std::max(maxVal, std::abs(sample));
    }
    
    if (maxVal > 0) {
        for (float& sample : audio) {
            sample /= maxVal;
        }
    }
}

unsigned long long parallelConvolution(std::vector<std::complex<float>>& inputFFT, std::vector<std::complex<float>>& irFFT, std::vector<std::complex<float>>& resultFFT){
    unsigned long long start = __rdtsc();

    #pragma omp parallel for
    for (size_t i = 0; i < inputFFT.size(); ++i) {
        resultFFT[i] = inputFFT[i] * irFFT[i];
    }
    
    return __rdtsc() - start;
}

// Convolve function 
std::vector<int16_t> convolve(const std::vector<int16_t>& input, 
                              const std::vector<int16_t>& impulseResponse) {
    // float conversion
    std::vector<float> inputFloat(input.begin(), input.end());
    std::vector<float> irFloat(impulseResponse.begin(), impulseResponse.end());

    // Zero-padding for efficiency
    int fftSize = 1 << static_cast<int>(std::ceil(std::log2(input.size() + impulseResponse.size() - 1)));
    inputFloat.resize(fftSize, 0.0f);
    irFloat.resize(fftSize, 0.0f);

    auto inputFFT = fft(inputFloat);
    auto irFFT = fft(irFloat);
    
    // convolution
    std::vector<std::complex<float>> resultFFT(inputFFT.size());
    unsigned long long parallelTime = parallelConvolution(inputFFT, irFFT, resultFFT);

    auto convolutionResult = ifft(resultFFT, fftSize);

    normalizeAudio(convolutionResult);

    // conversion in int16
    std::vector<int16_t> output(convolutionResult.size());
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] = static_cast<int16_t>(
            std::max(-32768.0f, std::min(32767.0f, convolutionResult[i] * 32767.0f))
        );
    }

    return output;
}


// File reading function 
std::vector<int16_t> readWavFile(const std::string& filename) {
    SF_INFO sfInfo{};
    sfInfo.format = 0;
    SNDFILE* file = sf_open(filename.c_str(), SFM_READ, &sfInfo);
    
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return {};
    }
    
    std::vector<int16_t> audioData(sfInfo.frames * sfInfo.channels);
    sf_count_t readCount = sf_read_short(file, audioData.data(), audioData.size());
    
    sf_close(file);
    return audioData;
}

// File writing function 
void writeWavFile(const std::string& filename, const std::vector<int16_t>& audioData) {
    SF_INFO sfInfo{};
    sfInfo.samplerate = 44100;
    sfInfo.channels = 2;
    sfInfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
    
    SNDFILE* file = sf_open(filename.c_str(), SFM_WRITE, &sfInfo);
    
    if (!file) {
        std::cerr << "Error creating output file: " << filename << std::endl;
        return;
    }
    
    sf_write_short(file, audioData.data(), audioData.size());
    sf_close(file);
}

int main() {
    auto guitarAudio = readWavFile("../samples/guitar.wav");
    auto impulseResponse = readWavFile("../samples/IR_early.wav");
    
    auto reverbedAudio = convolve(guitarAudio, impulseResponse);
    
    writeWavFile("../samples/guitar_reverb.wav", reverbedAudio);
    
    return 0;
}
