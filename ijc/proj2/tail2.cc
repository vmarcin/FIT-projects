/*
 * tail2.cc
 * Riesenie IJC-DU2, priklad 1), 22.4.2017
 * Autor: Vladimír Marcin FIT 1-BIT (xmarci10)
 * Prelozene: g++ (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609
 */
#include <iostream>
#include <queue>
#include <string>
#include <fstream>

char *argumentsTest(int argc, char *argv[], long *n);

using namespace std;

int main(int argc, char *argv[])
{
	ios::sync_with_stdio(false);
	
	long n;
	char *file = argumentsTest(argc,argv,&n);
	string buffer;
	queue<string> bufferRows;

	istream	*in = &cin;
	ifstream fp;
	
	if(file != NULL)
	{
   	fp.open(file);
  	if(!fp.is_open())
  	{
    	cerr << "CHYBA: Nepodarilo sa otvorit subor!\n";
    	exit(1);
  	}
		in = &fp;
	}
	//cyklus na nacitanie riadkov
	while(getline(*in,buffer))
	{
		bufferRows.push(buffer);
		if(bufferRows.size() > (unsigned long)n)
			bufferRows.pop();
	}
	//cyklus na vypis riadkov
	while(!bufferRows.empty())
	{
		cout << bufferRows.front() << endl;
		bufferRows.pop();
	}
	if (fp.is_open())
    fp.close();
}

char *argumentsTest(int argc, char *argv[], long *n)
{
	char *ptr = NULL;
	if(argc > 4)
	{
		cerr << "CHYBA: Zly pocet argumentov!\n";
		exit(1);
	}
	switch(argc)
	{
		case 1:
			*n = 10;
			break;

		case 2:
			if(((string)(argv[1])).compare("-n") == 0)
			{
				cerr << "CHYBA: Nastavenie 'n' vyzaduje argument!\n";
				exit(1);
			}
			*n = 10;
			return argv[1];
			break;

		case 3:
			if(((string)(argv[1])).compare("-n") == 0)
			{
				*n = strtol(argv[2],&ptr,10);
				if(*ptr != 0 || *n < 0)
				{
					cerr << "CHYBA: Chybny pocet riadkov!\n";
					exit(1);
				}
			}
			else
			{
				cerr << "CHYBA: Nespravne spustenie!\n";
				exit(1);		
			}
			break;

		case 4:
			if(((string)(argv[1])).compare("-n") == 0)
			{
				*n = strtol(argv[2],&ptr,10);
				if(*ptr != 0 || *n < 0)
				{
					cerr << "CHYBA: Chybny pocet riadkov!\n";
					exit(1);
				}
				return argv[3];
			}			
			if(((string)(argv[2])).compare("-n") == 0)
			{
				*n = strtol(argv[3],&ptr,10);
				if(*ptr != 0 || *n < 0)
				{
					cerr << "CHYBA: Chybny pocet riadkov!\n";
					exit(1);
				}
				return argv[1];
			}
			else
			{
				cerr << "CHYBA: Nespravne spustenie!\n";
				exit(1);
			}
			break;
	}
	return NULL;
}
