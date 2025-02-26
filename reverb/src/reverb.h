// reverb.h
#pragma once

#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <string>
#include <memory>
#include <sndfile.h>
#include <fftw3.h>
#include <immintrin.h>

namespace audio {

// Classe per rappresentare i campioni audio
class AudioBuffer {
public:
    // Costruttori
    AudioBuffer() = default;
    AudioBuffer(size_t size, size_t channels = 1);
    AudioBuffer(const std::vector<int16_t>& samples, size_t channels = 1);
    
    // Getters e setters
    size_t getSize() const { return samples_.size(); }
    size_t getChannels() const { return channels_; }
    int16_t getSample(size_t index) const { return samples_[index]; }
    void setSample(size_t index, int16_t value) { samples_[index] = value; }
    const std::vector<int16_t>& getSamples() const { return samples_; }
    std::vector<float> getFloatSamples() const;
    
    // Metodi di utilità
    void resize(size_t size, int16_t value = 0);
    void normalize();
    
private:
    std::vector<int16_t> samples_;
    size_t channels_ = 1;
    int sampleRate_ = 44100;
};

// Classe per la gestione dei file audio
class AudioFileIO {
public:
    static AudioBuffer readWavFile(const std::string& filename);
    static void writeWavFile(const std::string& filename, const AudioBuffer& buffer, 
                            int sampleRate = 44100, int channels = 2);
};

// Classe base per gli effetti audio
class AudioEffect {
public:
    virtual ~AudioEffect() = default;
    virtual AudioBuffer process(const AudioBuffer& input) = 0;
};

// Classe per la trasformata di Fourier
class FFTProcessor {
public:
    // Metodi per FFT e IFFT
    static std::vector<std::complex<float>> forwardFFT(const std::vector<float>& input);
    static std::vector<float> inverseFFT(const std::vector<std::complex<float>>& input, int originalSize);
    
private:
    // Classe helper per gestire la memoria FFTW con RAII
    class FFTWMemory {
    public:
        FFTWMemory(size_t size);
        ~FFTWMemory();
        
        fftwf_complex* getComplexBuffer() { return complexBuffer_; }
        float* getRealBuffer() { return realBuffer_; }
        
    private:
        fftwf_complex* complexBuffer_ = nullptr;
        float* realBuffer_ = nullptr;
    };
};

// Implementazione dell'effetto di riverbero
class ReverbEffect : public AudioEffect {
public:
    explicit ReverbEffect(const AudioBuffer& impulseResponse);
    
    // Override del metodo di processo
    AudioBuffer process(const AudioBuffer& input) override;
    
    // Metodi di configurazione
    void setDryWetMix(float mix) { dryWetMix_ = std::clamp(mix, 0.0f, 1.0f); }
    float getDryWetMix() const { return dryWetMix_; }
    
private:
    AudioBuffer impulseResponse_;
    float dryWetMix_ = 0.8f;  // 0.0 = solo dry, 1.0 = solo wet
    
    // Metodi interni
    std::vector<std::complex<float>> convolveSpectra(
        const std::vector<std::complex<float>>& inputFFT,
        const std::vector<std::complex<float>>& irFFT);
};

} // namespace audio
