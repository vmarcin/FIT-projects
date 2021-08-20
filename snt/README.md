## Prerequisities

**Gecode** requires at least version 4.2 of `gcc`.
```
$ wget https://github.com/Gecode/gecode/archive/release-6.2.0.zip
$ unzip release-6.2.0.zip
```
After unpacking the sources, you need to run the `configure` script in the toplevel directory.
```
$ ./configure --disable-examples
```
**Compiling the sources.** After successful configuration, simply invoking. If installation directory differs from `/usr/local/` you need to update the path in `makefile`
```
$ make
```
in the toplevel Gecode directory will compile the whole library. On Linux (and similar operating systems), Gecode is installed as a set of shared libraries. The default installation directory is `/usr/local`, which means that the header files can be found in `/usr/local/include` and the libraries in `/usr/local/lib`. 

For more information see https://www.gecode.org/doc-latest/MPG.pdf.

## SGP (Social Golfer Problem)
If Gecode is installed elsewhere than in the `/usr/local` directory, then it is necessary to modify the path in the `Makefile` before compiling. 

**Compiling the sources and run.**
In order to use specific constraints for your instance, you need to edit the source code by calling constraints you want to use.
For more information see `doc.pdf`.
```
path/src$ make
path/src$ ./sgp -g [groups] -s [group_size] -w [weeks] -mode solution
```
Other options
```
path/src$ ./sgp --help
```

## Latin Square
To get a GS1 submatrix for instances in form of `prime-prime-(prime+1)` use
```
path/golf$ ./latin -g [groups] -s [group_size] -w [weeks] -mode solution
```
Solution in form of `C++` array can be found in the `square.txt` file a you have to copy it to the `sgp.cpp` file ( Con(13). ).