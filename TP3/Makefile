bin := reduce

inc  := -I./include -I/usr/local/cuda/include
libs := -lcuda -lcudart
common_flags := -std=c++11 -O3 $(inc)

cppfiles   := $(wildcard src/*.cpp)
cufiles    := $(wildcard src/*.cu)
cpp_ofiles := $(patsubst src/%.cpp,obj/%.o,$(cppfiles))
cu_ofiles  := $(patsubst src/%.cu,obj/%.cu.o,$(cufiles))

cxx := g++
cxx_flags := -Wall $(common_flags)

gpu_arch := compute_60
gpu_code := sm_60
nvcc := nvcc
nvcc_flags := $(strip \
	--gpu-architecture=$(gpu_arch) \
	--gpu-code=$(gpu_code) \
	-ccbin $(cxx) \
	-DGPU_ARCH=\"$(gpu_arch)\" -DGPU_CODE=\"$(gpu_code)\" \
	$(common_flags) \
)

.PHONY: all clean re mrproper

all: $(bin)
	
$(bin): $(cu_ofiles) $(cpp_ofiles)
	$(nvcc) $(nvcc_flags) -o $(bin) $^ $(libs)

obj/%.cu.o: src/%.cu
	$(nvcc) $(nvcc_flags) -c $< -o $@

obj/%.o: src/%.cpp
	$(cxx) $(cxx_flags) -c $< -o $@

clean:
	rm -f $(bin) obj/*.o

re: clean all

mroprer: re
