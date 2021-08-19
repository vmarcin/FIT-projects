/**
 * Kostra programu pro 3. projekt IZP 2015/16
 *
 * Jednoducha shlukova analyza
 * Complete linkage
 * http://is.muni.cz/th/172767/fi_b/5739129/web/web/clsrov.html
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <ctype.h>
#include <math.h> // sqrtf
#include <limits.h> // INT_MAX

/*****************************************************************
 * Ladici makra. Vypnout jejich efekt lze definici makra
 * NDEBUG, napr.:
 *   a) pri prekladu argumentem prekladaci -DNDEBUG
 *   b) v souboru (na radek pred #include <assert.h>
 *      #define NDEBUG
 */
#ifdef NDEBUG
#define debug(s)
#define dfmt(s, ...)
#define dint(i)
#define dfloat(f)
#else

// vypise ladici retezec
#define debug(s) printf("- %s\n", s)

// vypise formatovany ladici vystup - pouziti podobne jako printf
#define dfmt(s, ...) printf(" - "__FILE__":%u: "s"\n",__LINE__,__VA_ARGS__)

// vypise ladici informaci o promenne - pouziti dint(identifikator_promenne)
#define dint(i) printf(" - " __FILE__ ":%u: " #i " = %d\n", __LINE__, i)

// vypise ladici informaci o promenne typu float - pouziti
// dfloat(identifikator_promenne)
#define dfloat(f) printf(" - " __FILE__ ":%u: " #f " = %g\n", __LINE__, f)

#endif

/*****************************************************************
 * Deklarace potrebnych datovych typu:
 *
 * TYTO DEKLARACE NEMENTE
 *
 *   struct obj_t - struktura objektu: identifikator a souradnice
 *   struct cluster_t - shluk objektu:
 *      pocet objektu ve shluku,
 *      kapacita shluku (pocet objektu, pro ktere je rezervovano
 *          misto v poli),
 *      ukazatel na pole shluku.
 */

struct obj_t {
    int id;
    float x;
    float y;
};

struct cluster_t {
    int size;
    int capacity;
    struct obj_t *obj;
};

/*****************************************************************
 * Deklarace potrebnych funkci.
 *
 * PROTOTYPY FUNKCI NEMENTE
 *
 * IMPLEMENTUJTE POUZE FUNKCE NA MISTECH OZNACENYCH 'TODO'
 *
 */

/*
 Inicializace shluku 'c'. Alokuje pamet pro cap objektu (kapacitu).
 Ukazatel NULL u pole objektu znamena kapacitu 0.
*/
void init_cluster(struct cluster_t *c, int cap)
{
    assert(c != NULL);
    assert(cap >= 0);

    // TODO
	c->capacity = cap;
	c->obj = malloc(sizeof(struct obj_t) * cap); 	
	if(c->obj == NULL)
		c->capacity = 0;
}

/*
 Odstraneni vsech objektu shluku a inicializace na prazdny shluk.
 */
void clear_cluster(struct cluster_t *c)
{
    // TODO
    free(c->obj);
	c->size = 0;
}

/// Chunk of cluster objects. Value recommended for reallocation.
const int CLUSTER_CHUNK = 10;

/*
 Zmena kapacity shluku 'c' na kapacitu 'new_cap'.
 */
struct cluster_t *resize_cluster(struct cluster_t *c, int new_cap)
{
    // TUTO FUNKCI NEMENTE
    assert(c);
    assert(c->capacity >= 0);
    assert(new_cap >= 0);

    if (c->capacity >= new_cap)
        return c;

    size_t size = sizeof(struct obj_t) * new_cap;

    void *arr = realloc(c->obj, size);
    if (arr == NULL)
        return NULL;

    c->obj = arr;
    c->capacity = new_cap;
    return c;
}

/*
 Prida objekt 'obj' na konec shluku 'c'. Rozsiri shluk, pokud se do nej objekt
 nevejde.
 */
void append_cluster(struct cluster_t *c, struct obj_t obj)
{
    // TODO
	if(c->capacity <= c->size)
		c = resize_cluster(c,(c->capacity) + CLUSTER_CHUNK);
	if(c != NULL)
	{
		c->obj[c->size] = obj;
		c->size += 1;
	}
}

/*
 Seradi objekty ve shluku 'c' vzestupne podle jejich identifikacniho cisla.
 */
void sort_cluster(struct cluster_t *c);

/*
 Do shluku 'c1' prida objekty 'c2'. Shluk 'c1' bude v pripade nutnosti rozsiren.
 Objekty ve shluku 'c1' budou serazny vzestupne podle identifikacniho cisla.
 Shluk 'c2' bude nezmenen.
 */
void merge_clusters(struct cluster_t *c1, struct cluster_t *c2)
{
    assert(c1 != NULL);
    assert(c2 != NULL);

    // TODO
	for(int i = 0; i < c2->size; i++)
	{
		append_cluster(c1, c2->obj[i]);
	}
	sort_cluster(c1);
}

/**********************************************************************/
/* Prace s polem shluku */

/*
 Odstrani shluk z pole shluku 'carr'. Pole shluku obsahuje 'narr' polozek
 (shluku). Shluk pro odstraneni se nachazi na indexu 'idx'. Funkce vraci novy
 pocet shluku v poli.
*/

int remove_cluster(struct cluster_t *carr, int narr, int idx)
{
    assert(idx < narr);
    assert(narr > 0);
	
    // TODO
	if(idx == (narr-1))
	{
		clear_cluster(&carr[idx]);
	}
	/*
	 * ak idx nie je indexom posledneho prvku pola uvolni dany cluster
	 * vytvori novy cluster do ktoreho vlozi objekty posledneho clustra
	 * a nasledne uvolni posledny cluster
	 */
	else
	{
		clear_cluster(&carr[idx]);	
		init_cluster(&carr[idx], 1);				
		merge_clusters(&carr[idx],&carr[narr-1]);
		clear_cluster(&carr[narr-1]);
	}
	return narr-1;
}

/*
 Pocita Euklidovskou vzdalenost mezi dvema objekty.
 */
float obj_distance(struct obj_t *o1, struct obj_t *o2)
{
    assert(o1 != NULL);
    assert(o2 != NULL);

    // TODO
	return sqrtf(pow(o2->x - o1->x, 2) + pow(o2->y - o1->y,2));	
}

/*
 Pocita vzdalenost dvou shluku.
*/
float cluster_distance(struct cluster_t *c1, struct cluster_t *c2)
{
    assert(c1 != NULL);
    assert(c1->size > 0);
    assert(c2 != NULL);
    assert(c2->size > 0);

    // TODO
	float max = 0;
	float distance;
	int i,j;
	
	/*
	 * vypocita vzdialenost vsetkych objektov v dvoch zhlukoch
	 * a vracia maximalnu vzdialenost 
	 */
	for(i = 0; i < c1->size; i++)
	{
		for(j = 0; j < c2->size; j++)
		{
			distance = obj_distance(&(c1->obj[i]),&(c2->obj[j]));
			if(distance >= max)
				max = distance;
		}
	}
	return max;
}

/*
 Funkce najde dva nejblizsi shluky. V poli shluku 'carr' o velikosti 'narr'
 hleda dva nejblizsi shluky. Nalezene shluky identifikuje jejich indexy v poli
 'carr'. Funkce nalezene shluky (indexy do pole 'carr') uklada do pameti na
 adresu 'c1' resp. 'c2'.
*/
void find_neighbour(struct cluster_t *carr, int narr, int *c1, int *c2)
{
    assert(narr > 0);

    // TODO
	int i, j;
	float min = cluster_distance(&carr[0],&carr[1]);
	*c1 = 0;	
	*c2 = 1;
	float distance;
	for(i = 0; i < narr; i++)
	{
		for(j = i+1; j < narr; j++)
		{
			distance = cluster_distance(&carr[i],&carr[j]);
			if(distance < min)
			{
				*c1 = i;
				*c2 = j;
				min = distance;
			}
		}
	}
}

// pomocna funkce pro razeni shluku
static int obj_sort_compar(const void *a, const void *b)
{
    // TUTO FUNKCI NEMENTE
    const struct obj_t *o1 = a;
    const struct obj_t *o2 = b;
    if (o1->id < o2->id) return -1;
    if (o1->id > o2->id) return 1;
    return 0;
}

/*
 Razeni objektu ve shluku vzestupne podle jejich identifikatoru.
*/
void sort_cluster(struct cluster_t *c)
{
    // TUTO FUNKCI NEMENTE
    qsort(c->obj, c->size, sizeof(struct obj_t), &obj_sort_compar);
}

/*
 Tisk shluku 'c' na stdout.
*/
void print_cluster(struct cluster_t *c)
{
    // TUTO FUNKCI NEMENTE
    for (int i = 0; i < c->size; i++)
    {
        if (i) putchar(' ');
        printf("%d[%g,%g]", c->obj[i].id, c->obj[i].x, c->obj[i].y);
    }
    putchar('\n');
}

/*
 Ze souboru 'filename' nacte objekty. Pro kazdy objekt vytvori shluk a ulozi
 jej do pole shluku. Alokuje prostor pro pole vsech shluku a ukazatel na prvni
 polozku pole (ukalazatel na prvni shluk v alokovanem poli) ulozi do pameti,
 kam se odkazuje parametr 'arr'. Funkce vraci pocet nactenych objektu (shluku).
 V pripade nejake chyby uklada do pameti, kam se odkazuje 'arr', hodnotu NULL.
*/
int load_clusters(char *filename, struct cluster_t **arr)
{
    assert(arr != NULL);

    // TODO
	int count;
	int p = 0;
	int j,i;
	int error = -2;					//identifikuje ci nastala chyba
	FILE *fp = fopen(filename,"r");
	/*
	 * overenie otvorenia suboru
	 */
	if(fp == NULL)
	{
		*arr = NULL;
		return 0;
	}
	/*
	 * precitanie prveho riadku a kontrola
	 * hodnoty count
	 */
	fscanf(fp,"count=%d", &count);
	if(count <= 0)
	{
		*arr = NULL;
		fclose(fp);
		return 0;
	}

	struct cluster_t *clusters;
	float id,x,y;
	int c;
	int id_array[count];
	
	/*
	 * inicializacia pola clustrov a
	 * overenie ci nenastala chyba
	 */
	clusters = malloc(sizeof(struct cluster_t) * count);
	if(clusters == NULL)
	{
		*arr = NULL;
		fclose(fp);
		return -1;
	}
	*arr = clusters;
	
	for(i = 0; i < count; i++)
	{
		/*
		 * inicializacia noveho objektu a
		 * overenie ci nenastala chyba
		 */
		init_cluster(&clusters[i],1);
		if(clusters[i].obj == NULL)
		{
			*arr = NULL;
			fclose(fp);
			error = -1;	
			break;	
		}	
		if((fscanf(fp,"%f %f %f", &id, &x, &y)) == 3)
		{
			/*
			 * overenie ci id a suradnice objektu v subore su cele cisla
			 */
			if(fmod(id,1) != 0 || fmod(x,1) != 0 || fmod(y,1) != 0)
                break;
			/*	
			 * overenie ci zadane suradnice patria do pozadovaneho intervalu
			 * a taktiez spravnost hodnoty id
			 */
			if(x < 0 || x > 1000 || y < 0 || y > 1000 || id > INT_MAX)
				break;
			/*
			 * kontrola ci subor neobsahuje duplicitne id
			 */
			id_array[i] = id;
			for(j = 0; j < i; j++)
			{
				if(id_array[i] == id_array[j])
				{
					error = 0;
					break;
				}
			}
			if(!error)
				break;
			/*
			 * kontrola ze v danom riadku sa nachadza iba jeden cluster
			 * v pripade ak narazime na iny ako blank nastane chyba
			 */
			while((c=getc(fp))!='\n')
            {
                if(c == EOF)
                    break;
                else if(!isblank(c))
                {
            	   	error = 0;
                    break;
                }         
            }
            if(!error)
                 break;
			/*
			 * v pripade uspesneho precitania sa hodnoty so suboru
			 * ulozia do vytvoreneho clustra
			 */
           	p++;
			clusters[i].size = 1;
			clusters[i].obj->id = id;
			clusters[i].obj->x = x;	
			clusters[i].obj->y = y;
		}
		else
			break;
	}
	/*
	 * po ukonceni cyklu uzatvorime subor
	 */
	fclose(fp);
	/*
	 * ak sa hodnota count (pocet clustrov v subore) nerovna
	 * poctu uspesne nacitanych clustrov a zaroven error sa
	 * sa nerovna -1 co znamena chybu pri alokacii miesta
	 * znamena ze chyba nastala pri citani subor a tak error = 0
	 */
	if(p != count && error != -1)
	{
		*arr = NULL;
		error = 0;
	}
	/*
	 * ak sa hodnota error nerovna pociatocnej co znamena ze nastala chyba
	 * uvolni sa doteraz alokovana pamat a vrati hodnotu error
	 */
	if(error != -2)
	{
		int o = p+1;
		if(error == -1)
			o = i;
		for(j = 0; j < o; j++)
			clear_cluster(&clusters[j]);
		free(clusters);
		return error;
	}
	return count;
}

/*
 Tisk pole shluku. Parametr 'carr' je ukazatel na prvni polozku (shluk).
 Tiskne se prvnich 'narr' shluku.
*/
void print_clusters(struct cluster_t *carr, int narr)
{
    printf("Clusters:\n");
    for (int i = 0; i < narr; i++)
    {
        printf("cluster %d: ", i);
        print_cluster(&carr[i]);
    }
}

int main(int argc, char *argv[])
{
    struct cluster_t *clusters;
	// TODO
	
	/*
	 * kontrola poctu argumentov	
	 */
	if(argc == 3 || argc ==2)
	{	
		char *ptr;
		int number_of_clusters = 1; // ak je program spusteni s jednym argumentom druhy ma hodnotu 1
		if(argc == 3)
		{
			number_of_clusters = (int)strtol(argv[2],&ptr,10); //pocet pozadovanych clustrov
			/*
			 * kontrola spravnosti 2heho argumentu
			 */
			if(*ptr!=0 || number_of_clusters <= 0)
			{
				fprintf(stderr,"Wrong value of argument!\n");
				return 1;
			}
		}
		
	
		int array_size = load_clusters(argv[1],&clusters);
		int c1,c2;
		
		/*
		 * ak funkcia load_clusters vrati
		 * 0 (chyba v subore) alebo
		 * 1 (chyba pri alokacii)
		 * program sa ukonci s vypisom chybovej hlasky
		 */
		if(array_size == 0)
		{
			fprintf(stderr,"Check file with input data!\n");
			return 1;
		}
		else if(array_size == -1)
		{
			fprintf(stderr,"Memory error!\n");
			return 1;
		}
		/*
		 * ak sa pocet clustrov nacitanych so suboru
		 * a pocet pozadovanych rovnaju program ich iba vypise
		 */
		if(array_size == number_of_clusters)
			print_clusters(clusters,array_size);	
		/*
	 	 * ak pole clustrov obsahuje viac prvkov ako je pozadovany
		 * pocet clustrov tvorime clustra kym sa tieto hodnoty
		 * nerovnaju
		 */
		else if(array_size > number_of_clusters)
		{
			while(array_size != number_of_clusters)
			{
				find_neighbour(clusters,array_size,&c1,&c2);
				merge_clusters(&clusters[c1],&clusters[c2]);
				array_size = remove_cluster(clusters,array_size,c2);
			}
			print_clusters(clusters,array_size);
		}
		/*
		 * nakoniec uvolnime alokovane miesto
		 */
		for(int i=0; i < array_size; i++)
			clear_cluster(&clusters[i]);
		free(clusters);
		/*
		 * ak je pocet pozadovanych clustrov vacsi ako pole clustrov
		 * ukoncime program s chybovym hlasenim
		 */
		if(array_size < number_of_clusters)
		{
			fprintf(stderr,"Wrong value(s) of argument(s)!\n");
			return 1;
		}
	}
	else
	{
		fprintf(stderr,"Wrong number of arguments!\n");
		return 1;
	}
	return 0;
}
