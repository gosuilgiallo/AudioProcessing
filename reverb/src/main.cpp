// main.cpp
#include "reverb.h"

int main() {
    using namespace audio;
    
    // Leggi file audio
    auto guitarAudio = AudioFileIO::readWavFile("../samples/guitar.wav");
    auto impulseResponse = AudioFileIO::readWavFile("../samples/IR_early.wav");
    
    // Crea effetto di riverbero
    ReverbEffect reverb(impulseResponse);
    reverb.setDryWetMix(0.8f);  // 80% wet, 20% dry
    
    // Applica l'effetto
    auto reverbedAudio = reverb.process(guitarAudio);
    
    // Salva il risultato
    AudioFileIO::writeWavFile("../samples/guitar_reverb.wav", reverbedAudio);
    
    return 0;
}
