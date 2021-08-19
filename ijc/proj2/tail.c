/*
 * tail.c
 * Riesenie IJC-DU2, priklad 1), 22.4.2017
 * Autor: Vladimír Marcin FIT 1-BIT (xmarci10)
 * Prelozene: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdbool.h>

#define MAX_ROW 1024	//maximalny pocet znakov v jednom riadku

/*
 * Funkcia sluzi na osetreni argumentov programu. Funkcia
 * vracia nazov suboru z ktoreho sa ma citat 
 * v pripade ze nazov nebol zadany vrati NULL.
 */
char *argumentsTest(int argc, char *argv[], long *n);
/*
 * Funkcia ktora nacita jeden riadok zo suboru
 * do pola 'buff' a vrati pocet precitanych znakov
 * v pripade konca suboru vrati EOF
 */
int getLine(FILE *fp,char *buff, bool *nonStandardFile);
/*
 * Funckia alokuje miesto pre buffer na ulozenie 'n'
 * riadkov a vrati ukazatel na alokovane miesto
 */
char **allocBuffer(long n, FILE *fp);
/*
 * Funkcia uvolni alokovane miesto pre buffer o 
 * velkosti 'n'
 */
void freeBuffer(long n, char **buffer);

int main(int argc, char *argv[])
{
	bool nonStandardFile = false; 		//premenna indikujuca ze ide o nestandardny subor (Ctrl-D sa nenachadza na zaciatku riadku)
	long n;														//pocet kolko poslednych riadkov sa ma vypisat
	FILE *fp = stdin;
	char buffer[MAX_ROW+2]; 					//buffer pre nacitane slovo (velkost MAX_ROW+2 kvoli ulozeniu '\n' a '\0')
	char *file = argumentsTest(argc,argv,&n);
	
	if(file != NULL)
		fp = fopen(file,"r");
	if(fp == NULL)
	{
		fprintf(stderr,"CHYBA: Nepodarilo sa otvorit subor '%s'!\n",file);
		return 1;
	}
	//ak je pocet riadkov ktore mame vypisat 0 precitame znaky a ukoncime program	
	char c;
	if(n == 0)
	{
		while((c=getc(fp)) != EOF);
		return 0;
	}
	
	char **bufferRows = allocBuffer(n,fp); 	//cyklicky sa prepisujuci buffer na ukladanie jednotilivych riadkov
	long lineCounter = 0;										//pocitadlo sluzi na indexovanie pola bufferRows pri nacitavani
	int lenghtOfLine;
	bool err = false;												//priznak indikujuci ze niektory riadok v subore prekrocil maximalny povoleny limit
	
	/*
 	 * Cyklus v ktorom nacitame riadku do pola bufferRows 
 	 */
	while((lenghtOfLine = getLine(fp,buffer,&nonStandardFile)) != EOF)
	{
		if(lenghtOfLine > MAX_ROW)
			err = true;
		strcpy(bufferRows[lineCounter],buffer);
		lineCounter+=1;

		/*
		 * ak plati podmienka pole je zaplnene
		 * a budeme prepisovat znova od zaciatku
		 */
		if(lineCounter == n)
			lineCounter = 0;
	}
	/*
	 * ak plati(subor je nestandardny) musime este nacitat posledny riadok
	 */
	if(nonStandardFile)
	{
		strcpy(bufferRows[lineCounter],buffer);
		lineCounter+=1;

		if(lineCounter == n)
			lineCounter = 0;
	}
			
	if(err)
		fprintf(stderr,"VAROVANIE: Vo vstupnom subore sa nachadazali riadky dlhsie ako je max limit a boli skratene!\n");

	/*
   * vypis poslednych 'n' riadkov
   */
	for(int i = 0; i < n; i++,lineCounter++)
	{
		if(lineCounter == n) lineCounter = 0;
		if(bufferRows[lineCounter][0] == 0)
			continue;
		printf("%s", bufferRows[lineCounter]);
	}
	
	freeBuffer(n,bufferRows);
	
	if (fp != stdin) fclose(fp);
	return 0;
}

char *argumentsTest(int argc, char *argv[], long *n)
{
	char *ptr = NULL;
	if(argc > 4)
	{
		fprintf(stderr,"CHYBA: Zly pocet argumentov!\n");
		exit(1);
	}
	switch(argc)
	{
		case 1:
			*n = 10;
			break;

		case 2:
			if((strcmp(argv[1],"-n")) == 0)
			{
				fprintf(stderr,"CHYBA: Nastavenie 'n' vyzaduje argument!\n");
				exit(1);
			}
			*n = 10;
			return argv[1];
			break;

		case 3:
			if((strcmp(argv[1],"-n")) == 0)
			{
				*n = strtol(argv[2],&ptr,10);
				if(*ptr != 0 || *n < 0)
				{
					fprintf(stderr,"CHYBA: Chybny pocet riadkov!\n");
					exit(1);
				}
			}
			else
			{
				fprintf(stderr,"CHYBA: Nespravne spustenie!\n");
				exit(1);		
			}
			break;

		case 4:
			if((strcmp(argv[1],"-n")) == 0)
			{
				*n = strtol(argv[2],&ptr,10);
				if(*ptr != 0 || *n < 0)
				{
					fprintf(stderr,"CHYBA: Chybny pocet riadkov!\n");
					exit(1);
				}
				return argv[3];
			}			
			if((strcmp(argv[2],"-n")) == 0)
			{
				*n = strtol(argv[3],&ptr,10);
				if(*ptr != 0 || *n < 0)
				{
					fprintf(stderr,"CHYBA: Chybny pocet riadkov!\n");
					exit(1);
				}
				return argv[1];
			}
			else
			{
				fprintf(stderr,"CHYBA: Nespravne spustenie!\n");
				exit(1);
			}
			break;
	}
	return NULL;
}

int getLine(FILE *fp,char *buff, bool *nonStandardFile)
{
	char c;
	int i;
	buff[0] = 0; //nastavime prvy prvok pola 'buff' na 'null' ak sa prepise znamena ze v poli sa nachadza nacitane slovo
	c = getc(fp);
	if(c == '\n')
	{
		buff[0] = c;
		buff[1] = '\0';
		return 0;
	}
	for(i = 0; c != EOF && c != '\n'; i++, c = getc(fp))
	{
		if(i < MAX_ROW)
		{
			buff[i] = c;
			buff[i+1] = '\n';
			buff[i+2] = '\0';
		}
	}
	if(c == '\n')
		return i;
	if(c == EOF && buff[0] != 0)	//ak 'buff' je prepisany ide o nestandardny subor
		*nonStandardFile = true;
	return EOF;
}

char **allocBuffer(long n, FILE *fp)
{
	char **bufferRows = (char **)malloc(n*sizeof(char *));
	if(bufferRows == NULL)
	{
		if(fp != stdin) fclose(fp);
		fprintf(stderr,"CHYBA: Chyba pri alokacii pamate!");
		exit(1);
	}

	for(int i = 0; i < n; i++)
	{
		bufferRows[i] = (char *)calloc((MAX_ROW+2),sizeof(char));
		if(bufferRows[i] == NULL)
		{
			if(fp != stdin) fclose(fp);
			freeBuffer(i,bufferRows);	
			fprintf(stderr,"CHYBA: Chyba pri alokacii pamate!");
			exit(1);
		}
	}
	return bufferRows;
}

void freeBuffer(long n,char **buffer)
{
	for(int i = 0; i < n; i++)
		free(buffer[i]);
	free(buffer);
}
