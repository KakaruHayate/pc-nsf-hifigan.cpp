// SPDX-License-Identifier: MPL-2.0
// Library-only timing/parity probe: no ORT dependency or CLI environment overrides.
#include <pc_nsf_hifigan/hifigan.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    try {
        if (argc < 2 || argc > 6) {
            std::cerr << "Usage: vocoder_probe MODEL [FRAMES=512] [RUNS=3] [OUTPUT.f32|-] [constant|varying]\n"
                         "Backend selection uses PCNSF_BACKEND; timings exclude model loading.\n";
            return 2;
        }
        const int frames=argc>2 ? std::stoi(argv[2]) : 512;
        const int runs=argc>3 ? std::stoi(argv[3]) : 3;
        const std::string pattern=argc>5 ? argv[5] : "constant";
        if(frames<=0 || runs<=0 || (pattern!="constant" && pattern!="varying"))
            throw std::invalid_argument("positive frames/runs and constant|varying pattern required");
        pc_nsf_hifigan::HifiganModel model(argv[1],9,"F32");
        std::vector<float> mel(static_cast<size_t>(frames)*model.num_mels,-6.0f);
        std::vector<float> f0(frames,220.0f), wav;
        if(pattern=="varying") {
            for(int t=0;t<frames;++t) {
                f0[t]=(t%97<9) ? 0.0f : 220.0f+100.0f*std::sin(t*0.047f);
                for(int c=0;c<model.num_mels;++c)
                    mel[static_cast<size_t>(t)*model.num_mels+c]=-6.0f+2.0f*std::sin(c*0.13f+t*0.021f);
            }
        }
        std::cout << "backend=" << ggml_backend_name(model.gguf->backend)
                  << " frames=" << frames << " pattern=" << pattern << std::endl;
        for(int run=0;run<runs;++run) {
            const auto start=std::chrono::steady_clock::now();
            pc_nsf_hifigan::hifigan_run(model,mel.data(),f0.data(),frames,wav);
            const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            if(wav.size()!=static_cast<size_t>(frames)*model.hop_size ||
               !std::all_of(wav.begin(),wav.end(),[](float v){return std::isfinite(v);}))
                throw std::runtime_error("incorrect output length or nonfinite PCM");
            std::cout << "run=" << run << " ms=" << std::fixed << std::setprecision(3)
                      << ms << " samples=" << wav.size() << std::endl;
        }
        if(argc>4 && std::string(argv[4])!="-") {
            std::ofstream output(argv[4],std::ios::binary);
            output.write(reinterpret_cast<const char*>(wav.data()),static_cast<std::streamsize>(wav.size()*sizeof(float)));
            if(!output) throw std::runtime_error("PCM dump failed");
        }
    } catch(const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
