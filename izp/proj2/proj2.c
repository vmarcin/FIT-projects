/*
 * @author      Marcin Vladimir
 * @email       xmarci10@stud.fit.vutbr.cz
 * @version     1.1, 27/11/16
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
/*
 * Funkcia taylor_log sluzi na vypocet prirodzeneho logaritmu pomocou
 * Taylorovho polynomu
 * Funkcia ocakava dva parametre:
 * prvy je typu double a urcuje cislo z ktoreho chceme vypocitat logaritmus
 * druhy je typu int a urcuje pocet iteracii pre dosiahnutie vysledku
 * Funkcia vracia vysledok logaritmu typu double
 */
double taylor_log(double x, unsigned int n)
{
	/*
 	 * podmienky pre predom definovane pripady
	 * pre funkciu log
 	 */
	if(x == 0)
		return -INFINITY;
	if(x < 0)
		return NAN;
	if(isinf(x))
		return INFINITY;
	if(isinf(x) == -1)
		return NAN;
	if(isnan(x))
		return NAN;

	unsigned int i;
	double taylor = 0.0;
	double base = 1-x;
	double nSquared;     // n-ta mocnina zakladu ktora sa priebezne pripocitava oddIntvysledku

	if(x >= 1)
		base = (x-1)/x;
	
	nSquared = base;
	for(i = 2; i <= n; i++)
	{
		nSquared *= base;
		taylor += (nSquared/i);
	}
	/*
	 * ak je zaklad logaritmu mensi ako jedna vraciam zaporny vysledok
	 */
	if(x < 1)	
		return -(taylor + base);
	else
		return taylor + base;
}
/*
 * Funkcia cfrac_log sluzi na vypocet prirodzeneho logaritmu 
 * pomocou zretazenych zlomkov
 * Funkcia ockava dva paramtere:
 * prvy typu double urcujuci cislo z ktoreho sa pocita logaritmus
 * druhy typu int urcuje rozvoj zretazeneho zlomku
 * Funkcia vracia hodnotu double a urcuje hodnotu logaritmu
 */
double cfrac_log(double x, unsigned int n)
{
	if(x == 0)
		return -INFINITY;
	if(x < 0)
		return NAN;
	if(isinf(x))
		return INFINITY;
	if(isinf(x) == -1)
		return NAN;
	if(isnan(x))
		return NAN;

	unsigned int i;
	double z = (x-1)/(x+1);
	double zz = z*z;
	double cf = 1.0;
	
	for(i = n; i >= 1; i--)
	{
		cf = ((i*i)*zz)/ ((2*i+1) - cf);
	}
	return 2*z/(1-cf);
}
/*
 * Funkcia my_fmod sluzi na vypocet zvysku po deleni desatinnych cisel
 * ocakava dva parametre typu double
 * kde prvy je delenec a druhy delitel
 * vracia hodnotu typu double co je zvysok po deleni parametrov
 */
double my_fmod(double a, double b)
{
	return a-b*((long int)(a/b));
}
/*
 * Funkcia sluzi na vypcet exponencialnej funkcie s obecnym zakldom
 * pomocou taylorovho polynomu
 * funkcia ockava tri parametre:
 * double x urcuje prirodzeny zaklad
 * double y urcuje mocninu	
 * unsigned int n urcuje pocet clenov polynomu
 * vracia hodnotu double urcujucu vysledok exp. funkcie
 */
double taylor_pow(double x, double y, unsigned int n)
{
	/*
	 * podmienky pre predom definove pripady
	 * pri vypocte exp. funkcie
 	 */
	int oddInt = 0;
	if(my_fmod(y,1) == 0 && my_fmod(y,2) != 0)
		oddInt = 1;

	if(x == 1.0)
		return 1.0;
	if(y == 0)
		return 1.0;
	if(isnan(x) || isnan(y))	
		return NAN;
	if(x == -1.0 && isinf(y))
		return 1.0;
	if(fabs(x) < 1.0 && isinf(y) == -1)
		return INFINITY;
	if(fabs(x) > 1.0 && isinf(y) == -1)
		return 0;
	if(fabs(x) < 1.0 && isinf(y) == 1)
		return 0;
	if(fabs(x) > 1.0 && isinf(y) == 1)
		return INFINITY;
	if(isinf(x) == 1 && y < 0)
		return 0;
	if(isinf(x) == 1 && y > 0)
		return INFINITY;
	if((x == 0 || x == -0.0) && !oddInt  && y > 0)
		return 0;
	if((x == 0 || x == -0.0) && oddInt&& y > 0)	
		return 0;
	if(isinf(x) == -1 && oddInt&& y < 0)
		return -0.0;
	if(isinf(x) == -1 && !oddInt && y < 0)
		return 0;
	if(isinf(x) == -1 && oddInt&& y > 0)
		return -INFINITY;
	if(isinf(x) == -1 && !oddInt && y > 0)
		return INFINITY;
	if(x == 0 && oddInt&& y < 0)
		return INFINITY;
	if(x == -0.0 && oddInt&& y < 0)
		return -INFINITY;
	if((x == 0 || x == -0.0) && !oddInt && y < 0)
		return INFINITY;
		
	if(x <= 0)
		return NAN;

	unsigned int i;
        double powtay = 1.0;
        double lnX = taylor_log(x,n);
        double t = 1.0;
	/*
	 * pomocne premenne pre pripad zaporneho exponenta
	 * ale hodnoty logaritmu
	 */
	double py = y;
	double plnX = lnX;
	/*
	 * ak je jedna z hodnot mocnina alebo hodnota logaritmu zaporna
	 * prevediem na kladnu a pocitam s kladnou hodnotou,
	 * z dovodu umocnovania danych hodnot
 	 * (raz by sa dany clen postupnosti pripocital a inokedy odpocital)
	 */
	if(y < 0 && lnX > 0)
		py = -y;
	if(y > 0 && lnX < 0)
		plnX = -lnX;

        for(i = 1; i <= n; i++)
        {
                t = t * ((py * plnX)/i);
                powtay += t;
        }
	/*
	 * ak sme zmenili znamienko pri mocnine alebo logaritme
	 * musime vratit obratenu hodnotu vysledku
	 */
	if((y < 0 && lnX > 0) || (y > 0 && lnX < 0))
		return 1/powtay;
        return powtay;
}
/*
 * vid taylor_pow
 * (funkcia sa lisi iba sposobom vypoctu prirodzeneho logaritmu
 *  v tomto pripade sa pocita pomocou zretazeneho zlomku)
 */
double taylorcf_pow(double x, double y, unsigned int n)
{
	int oddInt= 0;
	if(my_fmod(y,1) == 0 && my_fmod(y,2) != 0)
		oddInt = 1;
	
	if(x == 1.0)
		return 1.0;
	if(y == 0)
		return 1.0;
	if(isnan(x) || isnan(y))	
		return NAN;
	if(x == -1.0 && isinf(y))
		return 1.0;
	if(fabs(x) < 1.0 && isinf(y) == -1)
		return INFINITY;
	if(fabs(x) > 1.0 && isinf(y) == -1)
		return 0;
	if(fabs(x) < 1.0 && isinf(y) == 1)
		return 0;
	if(fabs(x) > 1.0 && isinf(y) == 1)
		return INFINITY;
	if(isinf(x) == 1 && y < 0)
		return 0;
	if(isinf(x) == 1 && y > 0)
		return INFINITY;
	if((x == 0 || x == -0) && !oddInt  && y > 0)
		return 0;
	if((x == 0 || x == -0) && oddInt && y > 0)	
		return 0;
	if(isinf(x) == -1 && oddInt && y < 0)
		return -0.0;
	if(isinf(x) == -1 && !oddInt && y < 0)
		return 0;
	if(isinf(x) == -1 && oddInt && y > 0)
		return -INFINITY;
	if(isinf(x) == -1 && !oddInt && y > 0)
		return INFINITY;
	if(x == 0 && oddInt && y < 0)
		return INFINITY;
	if(x == -0 && oddInt && y < 0)
		return -INFINITY;
	if((x == 0 || x == -0) && !oddInt && y < 0)
		return INFINITY;

	if(x <= 0)
		return NAN;
	unsigned int i;
	double powcf = 1.0;
	double lnX = cfrac_log(x,n);
	double t = 1.0;
	double py = y;
	double plnX = lnX;

	if(y < 0 && lnX > 0)
		py = -y;
	if(y > 0 && lnX < 0)
		plnX = -lnX;
	for(i = 1; i <= n; i++)
	{
		t = t * ((py * plnX)/i);
		powcf += t;
	}
	if((y < 0 && lnX > 0) || (y > 0 && lnX < 0))
		return 1/powcf;
	return powcf;
}
/*
 * Funkcia sluzi na overenie spravnosti argumentu typu int
 * ockava retazec ktory prevedie na int
 * a ukazatel na unsigned int kde ulozi hodnotu arguemntu
 * vracia hodnotu 0 v pripade ze arguemnt vyhovuje podminkam
 * a hodnotu jedna ak arguemnt nevyhovuje
 */
int check_int_argument(char arg[], unsigned int *p)
{
	char *ptr;
	*p = (unsigned int)strtoul(arg,&ptr,10);
	if(*p <= 0 || *ptr != 0 || arg[0] == '-')
		return 1;
	return 0;
}
/*
 * vid funkcia check_int_aruemnt
 * funkcia sa lisi iba v type arguemntu
 */
int check_double_argument(char arg[], double *p)
{
	char *ptr;
	*p = strtod(arg,&ptr);
	if(*ptr != 0)
		return 1;
	return 0;
}
/*
 * Funkcia sluziaca na vypis pri spusteni programu s argumentom --log
 */
void logX(double x,unsigned int n)
{
	printf("       log(%g) = %.12g\n",x,log(x));
	printf(" cfrac_log(%g) = %.12g\n",x,cfrac_log(x,n));	
	printf("taylor_log(%g) = %.12g\n",x,taylor_log(x,n));
}
/*
 * Funkcia sluziaca na vypis pri spusteni programu s arguemtom --pow
 */
void powXY(double x,double y,unsigned int n)
{
	printf("         pow(%g,%g) = %.12g\n",x,y,pow(x,y));
	printf("  taylor_pow(%g,%g) = %.12g\n",x,y,taylor_pow(x,y,n));
	printf("taylorcf_pow(%g,%g) = %.12g\n",x,y,taylorcf_pow(x,y,n));		
}
int main(int argc, char *argv[])
{
	unsigned int n;
	double x,y;

	/*
	 * ak je program spusteny so 4mi argumentmi overim ich spravnost a na zaklade vratenych hodnot
	 * funkcii volam funkciu na vypocet logaritmu pripadne ukoncim program s chybovym hlasenim
	 */	
	if(argc == 4)
	{
		if(!strcmp(argv[1],"--log"))
		{
			if(check_double_argument(argv[2],&x) || check_int_argument(argv[3],&n))
			{
				fprintf(stderr,"Wrong value(s) of argument(s)!\n");	
				return 1;
			}	
			else
				logX(x,n);
		}
		else
		{
			fprintf(stderr,"Wrong value(s) of argument(s)!\n");
			return 1;
		}
	}
	/*
	 * ak je program spusteny s 5timi argumentami overim ich spravnost a na zaklade vratenych hodnot
         * funkcii volam funkciu na vypocet exp. funkcie  pripadne ukoncim program s chybovym hlasenim
	 */
	else if(argc == 5)
	{
		if(!strcmp(argv[1],"--pow"))
		{
			if(check_double_argument(argv[2],&x) || check_double_argument(argv[3],&y) || check_int_argument(argv[4],&n))
			{
				fprintf(stderr,"Wrong value(s) of argument(s)!\n");
				return 1;
			}
			else
				powXY(x,y,n);		
		}
		else
		{
			fprintf(stderr,"Wrong value(s) of argument(s)!\n");
			return 1;
		}
	}	
	else
	{
		fprintf(stderr,"Wrong value(s) of argument(s)!\n");
		return 1;
	}
		return 0;
}
