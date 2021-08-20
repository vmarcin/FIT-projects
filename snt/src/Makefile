CC  	 	 = gcc
CXX 	 	 = g++
INCLUDESPATH = -I/usr/local/include # path to gecode headers
LIBSPATH     = -L/usr/local/lib/ 	#path to gecode libraries
LIBS         = -lgecodedriver -lgecodegist -lgecodesearch -lgecodeminimodel -lgecodeset -lgecodeint -lgecodekernel -lgecodesupport

all: sgp latin

# linking
sgp: sgp.o
	$(CXX) -o $@ $(LIBSPATH) $< $(LIBS)
latin: latinsquare.o
	$(CXX) -o $@ $(LIBSPATH) $< $(LIBS)

# compilation
sgp.o: sgp.cpp
	$(CXX) $(INCLUDESPATH) -c $< -o $@
latinsquare.o: latinsquare.cpp
	$(CXX) $(INCLUDESPATH) -c $< -o $@

run:
	./sgp -g $(g) -s $(s) -w $(w) -mode solution
gist:
	./sgp -g $(g) -s $(s) -w $(w) -mode gist
square:
	./latin -g $(g) -s $(s) -w $(w) -mode solution

benchmark:
	./sgp -g 5 -s 3 -w 7 -mode solution
	./sgp -g 5 -s 5 -w 6 -mode solution
	./sgp -g 6 -s 6 -w 7 -mode solution
	./sgp -g 7 -s 7 -w 8 -mode solution
	./sgp -g 8 -s 8 -w 9 -mode solution
	./sgp -g 11 -s 11 -w 12 -mode solution
	./sgp -g 13 -s 13 -w 14 -mode solution
	./sgp -g 17 -s 17 -w 18 -mode solution

clean:
	rm *.o sgp latin
