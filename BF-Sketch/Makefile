EXEC += main_freq hitter throughput
all: $(EXEC)

CFLAGS = -w -std=c++11 -O2
HEADER += util.h hash.h datatypes.hpp adaptor.hpp
SRC += hash.c adaptor.cpp
SKETCHHEADER += bloom_filter.hpp CMSketch.hpp
SKETCHSRC += CMSketch.cpp
LIBS= -lpcap 

throughput: throughput.cpp $(SRC) $(HEADER) $(SKETCHHEADER)
	g++ $(CFLAGS) $(INCLUDES) -o $@ $< $(SRC) $(SKETCHSRC) $(LIBS) 

clean:
	rm -rf $(EXEC)
	rm -rf *log*
	rm -rf *out*
