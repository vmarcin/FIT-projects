/*
 * @author	Marcin Vladimir
 * @email	xmarci10@stud.fit.vutbr.cz
 * @version	1.1, 06/11/16
 */
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
/*
 * tato funkcia sluzi na spustenie programu s parametrami -s a -n 
 * funkciu volame aj pri spusteni s jednym s tychto parametrov
 * popripade pri spusteni programu bez argumentov 
 * Funkcia ocakava dva parametre typu int
 */
void basic_arguments(int s, int n)
{
	int bytePositionInRow = 0;
	int numberOfRepetitions = 0;
	int actualChar;			//prave spracovavany znak
	int actualBytePosition = 0;	//pozicia bajtu v subore
	int r;				//pomocna premenna
	int b[16];			//pole obsahujuce seriu 16tich znakov aktualneho riadku
	int firstByte = s;		//adresa prveho spracovavaneho bajtu
	
       /*
	* naplenie pola medzerami pre pripad ze riadok bude mat menej ako 16 bajtov
	*/
	for(r = 0; r < 16; r++)		
		b[r] = ' ';
	while((actualChar = getchar()) != EOF)
	{	
		/*
		 * ignoruje znaky, ktore maju mensiu adresu ako je pozadovana adresa
		 */
		if(s <= actualBytePosition)
		{
			if(firstByte == s)
			{
                        	printf("%08x  ",firstByte);
				firstByte++;
			}
                	else if(bytePositionInRow == 0)
                        	printf("%08x  ",actualBytePosition);	//vypise adresu prveho bajtu v riadku v hexa podobe na 8 des. miest
			printf("%02x ",actualChar);			//vypise hexa hodnotu daneho znaku na dve desatinne miesta
			b[bytePositionInRow] = actualChar;			
			bytePositionInRow ++;
			numberOfRepetitions ++;
			//j = 0;	
		}
		actualBytePosition++;
		/*
		 * ak sa bajt nachadza na konci riadku vypise tlacitelnu podobu danych bajtov
		 */
		if(bytePositionInRow == 16)
		{	
			printf(" |");
			for(r = 0; r < 16; r++)
			{
				/*
				 * ak je dany zank tlacitelny vytlaci sa, v opacnom pripade sa vytlaci "."
				 */
				if(isprint(b[r]))	
					putchar(b[r]);
				else				
					printf(".");
			}
			printf("|");
			bytePositionInRow = 0;
			for(r = 0; r < 16; r++)
				b[r] = ' ';
			printf("\n");
		}
		else if(bytePositionInRow == 8)
			printf(" ");
		if(numberOfRepetitions == n)
			break;	
	}
	/*
	 * ak po ukonceni cyklu niesme na konci riadku doplnime riadok medzerami a vypiseme
	 * tlacitelnu podobu zankov v danom riadku 
	 */
	if(bytePositionInRow != 0)
	{
		for(r = 0; r < (16 - bytePositionInRow) * 3; r++)
			printf(" ");
		if(bytePositionInRow < 8)
			printf(" "); 
		printf(" |");
		for(r = 0; r < 16; r++)
		{
			if(isprint(b[r]))
				putchar(b[r]);
			else
				printf(".");	
		}
		printf("|\n");
	}
}
void argument_x()
{
	int actualChar;	//aktualne spracovavany znak 
	
	/*
	 * kym sa nedostane na koniec suboru vypisuje vsetky znaky zo vstupu
	 *  v hexa podobe na dve desatinne miesta
	 */
	while((actualChar = getchar()) != EOF)
		printf("%02x",actualChar);
}
void argument_S(int s)
{
	int n[s];
	int actualChar;
	int count = 0;	
	int p = 0;	//priznak hovoriaci o tom ci dani retazec ma pokracovanie
	int j;

	for(j = 0; j < s; j++)
		n[j] = 0;
	while((actualChar = getchar()) != EOF)
	{
		/*
		 * ak dany znak je tlacitelny alebo prazdny bude sa dalej spracuvavat
		 */
		if(isprint(actualChar) || isblank(actualChar))
		{
			/*
			 * ak pole este nieje naplnene znak sa ulozi do pola
			 */
			if(n[s-1] == 0)
			{
				n[count] = actualChar;
				count++;
			}	
			/*
			 * ak je pole plne vypise sa a priznak sa nastavi na hodnotu '1'
			 */
			if(n[s-1] != 0  && p == 0)
			{
				for(j = 0; j < s; j++)
					putchar(n[j]);
				p = 1;
			}
			/*
			 * ak je pole plne a priznak sa rovna '1' co znamena ze nas retazec pokracuje 
			 * vypise sa kazdy dalsi znak daneho retazca
			 */
			else if(p == 1)
				putchar(actualChar);
		}
		/*
		 * ak narazime na koniec retazca priznak nastavime na hodnotu '0' a vynulujeme pole
		 */
		else
                {
			if(n[s-1] != 0)
				printf("\n");
                        for(j = 0; j < s; j++)
                                n[j] = 0;
                        count = 0;
                        p = 0;
                }
	}			
}
void argument_r()
{
	int actualChar;
	int b[2];
	int count = 0;

	while((actualChar = getchar()) != EOF)
	{
		/*
		 * ignoruje vsetky biele znaky zo vstupu
		 */
		if(!(isspace(actualChar)))
		{
			/*
			 * ak je dany znak cislica
			 * alebo pismeno z intervalu <a;f> || <A;F>
			 * ulozime tento znak do pola na spracovanie
			 */
			if(actualChar <= '9' && actualChar >= '0')
			{
				b[count] = actualChar;
				count++;
			}
			else if((actualChar <= 'f' && actualChar >= 'a') || (actualChar <= 'F' && actualChar >= 'A'))
			{
				b[count] = actualChar;
				/*
				 * ak je pismeno z intervalu <A;F> zmenime ho na male
				 */
				if(!islower(b[count]))
					b[count] = tolower(b[count]);
				count++;
			}
			else
			{
				fprintf(stderr,"\nUnexpected input data!\n");
				exit(1);
			}
		}
		/*
		 * ak je pole plne spracujeme ho a vypiseme prislusny znak
		 */
		if(count == 2)
		{
			/*
			 * pri spracovavani znakov najprv od daneho prvku pola odcitame zank z ASCII tabulky
			 * aby sme ziskali desiatkovu hodnotu daneho znaku a nasledne hodnotu daneho bajtu	
			 * prevedieme do hexadecimalnej sustavy
			 */
			if(b[0] > '9' && b[1] > '9')
				putchar(((b[0] - 'W') * 16) + (b[1] - 'W')); 
			else if(b[1] > '9' && b[0] <= '9')
				putchar(((b[0] - '0') * 16) + (b[1] - 'W'));
			else if(b[0] > '9' && b[1] <= '9')
				putchar(((b[0] - 'W') * 16) + (b[1] - '0'));
			else
				putchar(((b[0] - '0') * 16) + (b[1] - '0'));

			count = 0;	
		}
	}
	if(count == 1)
	{
		if(b[0] > '9')
			putchar((b[0] - 'W'));
		else
			putchar((b[0] - '0'));
	}		
}
/*
 * tato funkcia sluzi na porovanie dvoch retazcov
 * ako parametre ocakava dve polia typu char
 * navratova hodnota je '1' ak sa retazce rovanju a '0' ak sa nerovnaju
 */
int compare_arguments(char a1[], char a2[])
{
	int i;
	int j = 0;
	/*
	 * cyklus sa zopakuje tolko krat kolko prvkov ma pole a1[] (cize prvy retazec)
	 * a v cykle porovname jednotlive znaky dvoch retazcov
	 */
	for(i = 0; a1[i] != '\0'; i++)
	{
		if(a1[i] == a2[i])
			j++;
	}	
	if(i == j)
		return 1;
	else
		return 0;
}
/*
 * tato funkcia sluzi na osetrovanie ciselnych argumentov
 * ako parameter sa ockava pole typu char
 * ak je hodnota argumentu spravna vrati jeho hodnotu
 */
int value_of_argument(char arg[])
{
	int i;
	int argument = 0;
	/*
	 * cyklus prejde celym danym polom a postupne vypocitava hodnotu argumentu
	 */
	for(i = 0; arg[i] != '\0'; i++)
	{
		/*
		 * ak je hodnota cislica spracuvava sa ak nie program sa ukonci,
		 * pretoze narazil na neocakavany znak
		 */
		if(arg[i] <= '9' && arg[i] >= '0')
			argument = (arg[i] - '0') + argument * 10;
		else	
		{
			fprintf(stderr,"Wrong value(s) of parameter(s)!\n");
			exit(1);
		}
	}
	return argument;
}
int main(int argc, char *argv[])
{
	int arg1;
	int arg2;
	
	/*
	 * ak je program spusteny bez parametrov zavola sa funkcia basic_arguments s parametrami (0,-1)
	 */
	if(argc == 1)
		basic_arguments(0,-1);
	/*
	 * ak je program spusteny s jednym argumentom zavolame funkciu compare_arguments a na zaklade vratenej hodnoty
	 * zavolame pozadovanu funkciu
	 * ak spustime program s neznamym argumentom vypise sa chybova sprava
	 */
	else if(argc == 2)
	{
		if(compare_arguments(argv[1],"-x"))
                {
			argument_x();
			printf("\n");
		}
        	else if(compare_arguments(argv[1],"-r"))
                	argument_r();
		else
			fprintf(stderr,"Wrong value(s) of parameter(s)!\n");
	}
	/*
	 * ak je program spusteny s dvomi argumentami pomocou funkcie compare_arguemnts do ktorej posleme prvy
	 * argument zistime, ktoru fuknciu mame zavolat a pomocou funkcie value_of_argument do ktorej posleme
	 * druhy argument zistime hodnotu parametra pre volanu funkciu
	 */
	else if(argc == 3)
	{
		if(compare_arguments(argv[1],"-S"))
		{
			arg1 = value_of_argument(argv[2]);
			if(arg1 < 200)
				argument_S(arg1);
			else
				fprintf(stderr,"Wrong value(s) of parameter(s)!\n");
		}
		else if(compare_arguments(argv[1],"-s"))
		{
                        arg1 = value_of_argument(argv[2]); 
			basic_arguments(arg1,-1);
		}
		else if(compare_arguments(argv[1],"-n"))
		{
			arg1 = value_of_argument(argv[2]);
			if(arg1 > 0)
				basic_arguments(0,arg1);
			else
				fprintf(stderr,"Wrong value(s) of parameter(s)!\n");
		}
		else
			fprintf(stderr,"Wrong value(s) of parameter(s)!\n");
	}
	/*
	 * ak je program spusteny so styrmi argumentami pomocou funkcii compare_arguments a value_of_argument
	 * zistime aka funkcia a s akou hodnotou parametra sa ma spustit
	 */
	else if(argc == 5)
	{
		if((compare_arguments(argv[1],"-s") && compare_arguments(argv[3],"-n")) || 
			(compare_arguments(argv[3],"-s") && compare_arguments(argv[1],"-n"))) 
		{
			arg1 = value_of_argument(argv[2]);
			arg2 = value_of_argument(argv[4]);

			if(argv[1][1] == 'n' && arg1 > 0)
				basic_arguments(arg2,arg1);
			else if(argv[3][1] == 'n' && arg2 > 0)
				basic_arguments(arg1,arg2);
			else
				fprintf(stderr,"Wrong value(s) of parameter(s)!\n");
		}
		else
			fprintf(stderr,"Wrong value(s) of parameter(s)!\n");
	}	
	else		
		fprintf(stderr,"Wrong value(s) of parameter(s)!\n");		
	return 0;	
}
