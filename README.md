Files for openvinotoolkit/openvino#38640.

emulate.cpp replays the blocked GEMM launches of the oneDNN jit gemm driver and checks how the kernel reads the 2D zero points / scales in each launch, old logic vs fixed logic.

Build (from a configured oneDNN GPU build dir):

    awk '/^namespace \{$/{f=1} f{print} f&&/^\} \/\/ namespace$/{exit}' <onednn>/src/gpu/intel/gemm/jit.cpp > quant_helpers.inc
    c++ -std=c++17 -O2 -DGEMMSTONE_CONFIG -DNGEN_CONFIG -I<build>/include -I<onednn>/include -I<onednn>/src \
        -I<onednn>/src/gpu/intel/jit/config -I<onednn>/third_party/ngen -I<onednn>/src/gpu/intel/gemm/jit/include \
        -I. emulate.cpp -o emulate && ./emulate
