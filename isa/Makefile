CXX = g++
CXXFLAGS= -std=c++11
LDLIBS= -lpcap
EXECUTABLE= dns-export
REMOVE= rm -f
SRC= $(wildcard *.cc)
OBJ= $(SRC:%.cc=%.o)

.PHONY: all clean

all: $(EXECUTABLE)

#linking
$(EXECUTABLE): $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) $(LDLIBS) -o $(EXECUTABLE)

#compilation
dns-sniffer.o: dns-sniffer.cc dissect.h syslog.h dns-stats.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
base64.o: base64.cc base64.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
dissect.o: dissect.cc dissect.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
dns-stats.o: dns-stats.cc dns-stats.h base64.h dissect.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
syslog.o: syslog.cc syslog.h dns-stats.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	$(REMOVE) *.o $(EXECUTABLE)
