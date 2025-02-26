// reverb.cpp
#include "reverb.h"

namespace audio {

// Implementazione di AudioBuffer
AudioBuffer::AudioBuffer(size_t size, size_t channels) 
    : samples_(size, 0), channels_(channels) {}

AudioBuffer::AudioBuffer(const std::vector<int16_t>& samples, size_t channels)
    : samples_(samples), channels_(channels) {}

std::vector<float> AudioBuffer::getFloatSamples() const {
    std::vector<float> result(samples_.size());
    for (size_t i = 0; i < samples_.size(); ++i) {
        result[i] = static_cast<float>(samples_[i]) / 32768.0f;
    }
    return result;
}

void AudioBuffer::resize(size_t size, int16_t value) {
    samples_.resize(size, value);
}

void AudioBuffer::normalize() {
    int16_t maxVal = 0;
    for (int16_t sample : samples_) {
        maxVal = std::max(maxVal, static_cast<int16_t>(std::abs(sample)));
    }
    
    if (maxVal > 0) {
        float scale = 32767.0f / maxVal;
        for (auto& sample : samples_) {
            sample = static_cast<int16_t>(sample * scale);
        }
    }
}

// Implementazione di AudioFileIO
AudioBuffer AudioFileIO::readWavFile(const std::string& filename) {
    SF_INFO sfInfo{};
    sfInfo.format = 0;
    SNDFILE* file = sf_open(filename.c_str(), SFM_READ, &sfInfo);
    
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return AudioBuffer();
    }
    
    std::vector<int16_t> audioData(sfInfo.frames * sfInfo.channels);
    sf_read_short(file, audioData.data(), audioData.size());
    
    sf_close(file);
    return AudioBuffer(audioData, sfInfo.channels);
}

void AudioFileIO::writeWavFile(const std::string& filename, 
                               const AudioBuffer& buffer,
                               int sampleRate, 
                               int channels) {
    SF_INFO sfInfo{};
    sfInfo.samplerate = sampleRate;
    sfInfo.channels = channels;
    sfInfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
    
    SNDFILE* file = sf_open(filename.c_str(), SFM_WRITE, &sfInfo);
    
    if (!file) {
        std::cerr << "Error creating output file: " << filename << std::endl;
        return;
    }
    
    auto samples = buffer.getSamples();
    sf_write_short(file, samples.data(), samples.size());
    sf_close(file);
}

// Implementazione di FFTProcessor::FFTWMemory
FFTProcessor::FFTWMemory::FFTWMemory(size_t size) {
    complexBuffer_ = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * size);
    realBuffer_ = (float*) fftwf_malloc(sizeof(float) * size);
}

FFTProcessor::FFTWMemory::~FFTWMemory() {
    if (complexBuffer_) fftwf_free(complexBuffer_);
    if (realBuffer_) fftwf_free(realBuffer_);
}

// Implementazione di FFTProcessor
std::vector<std::complex<float>> FFTProcessor::forwardFFT(const std::vector<float>& input) {
    int N = input.size();
    FFTWMemory memory(N);
    
    // Copia i dati di input nel buffer reale
    std::copy(input.begin(), input.end(), memory.getRealBuffer());
    
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(N, memory.getRealBuffer(), 
                                             memory.getComplexBuffer(), FFTW_ESTIMATE);
    fftwf_execute(plan);
    
    std::vector<std::complex<float>> result(N/2 + 1);
    for (int i = 0; i < N/2 + 1; ++i) {
        result[i] = std::complex<float>(memory.getComplexBuffer()[i][0], 
                                         memory.getComplexBuffer()[i][1]);
    }
    
    fftwf_destroy_plan(plan);
    return result;
}

std::vector<float> FFTProcessor::inverseFFT(const std::vector<std::complex<float>>& input, 
                                           int originalSize) {
    int N = originalSize;
    FFTWMemory memory(N);
    
    // Copia i dati di input nel buffer complesso
    for (int i = 0; i < N/2 + 1; ++i) {
        memory.getComplexBuffer()[i][0] = input[i].real();
        memory.getComplexBuffer()[i][1] = input[i].imag();
    }
    
    fftwf_plan plan = fftwf_plan_dft_c2r_1d(N, memory.getComplexBuffer(), 
                                            memory.getRealBuffer(), FFTW_ESTIMATE);
    fftwf_execute(plan);
    
    std::vector<float> result(N);
    for (int i = 0; i < N; ++i) {
        result[i] = memory.getRealBuffer()[i] / N;  // Normalizzazione
    }
    
    fftwf_destroy_plan(plan);
    return result;
}

// Implementazione di ReverbEffect
ReverbEffect::ReverbEffect(const AudioBuffer& impulseResponse)
    : impulseResponse_(impulseResponse) {}

AudioBuffer ReverbEffect::process(const AudioBuffer& input) {
    // Converte in float per il processing
    std::vector<float> inputFloat = input.getFloatSamples();
    std::vector<float> irFloat = impulseResponse_.getFloatSamples();
    
    // Calcola dimensione FFT (potenza di 2)
    int fftSize = 1 << static_cast<int>(std::ceil(std::log2(
        inputFloat.size() + irFloat.size() - 1)));
    
    // Padding
    inputFloat.resize(fftSize, 0.0f);
    irFloat.resize(fftSize, 0.0f);
    
    // FFT
    auto inputFFT = FFTProcessor::forwardFFT(inputFloat);
    auto irFFT = FFTProcessor::forwardFFT(irFloat);
    
    // Convoluzione nel dominio della frequenza
    auto resultFFT = convolveSpectra(inputFFT, irFFT);
    
    // IFFT
    auto convolutionResult = FFTProcessor::inverseFFT(resultFFT, fftSize);
    
    // Normalizza
    float maxVal = 0.0f;
    for (float sample : convolutionResult) {
        maxVal = std::max(maxVal, std::abs(sample));
    }
    
    if (maxVal > 0.0f) {
        for (auto& sample : convolutionResult) {
            sample /= maxVal;
        }
    }
    
    // Converti in int16_t
    std::vector<int16_t> output(convolutionResult.size());
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] = static_cast<int16_t>(
            std::max(-32768.0f, std::min(32767.0f, convolutionResult[i] * 32767.0f))
        );
    }
    
    return AudioBuffer(output, input.getChannels());
}

std::vector<std::complex<float>> ReverbEffect::convolveSpectra(
    const std::vector<std::complex<float>>& inputFFT,
    const std::vector<std::complex<float>>& irFFT) {
    
    std::vector<std::complex<float>> resultFFT(inputFFT.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < inputFFT.size(); ++i) {
        resultFFT[i] = inputFFT[i] * irFFT[i];
    }
    
    return resultFFT;
}

} // namespace audio
