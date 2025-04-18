# Makefile
CXX = g++
CXXFLAGS = -O2 -std=c++11
SRC = heat_cpu.cpp
OBJ = $(SRC:.cpp=.o)
BIN = heat_cpu

all: $(BIN)

$(BIN): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f $(OBJ) $(BIN)