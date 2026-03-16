CXX ?= g++
CXXFLAGS ?= -O3 -march=native -fopenmp -std=c++17 -Wall -Wextra

heston_mc: heston_mc.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f heston_mc
