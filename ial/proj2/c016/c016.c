
/* c016.c: **********************************************************}
{* Téma:  Tabulka s Rozptýlenými Položkami
**                      První implementace: Petr Přikryl, prosinec 1994
**                      Do jazyka C prepsal a upravil: Vaclav Topinka, 2005
**                      Úpravy: Karel Masařík, říjen 2014
**                              Radek Hranický, říjen 2014
**                              Radek Hranický, listopad 2015
**                              Radek Hranický, říjen 2016
**
** Vytvořete abstraktní datový typ
** TRP (Tabulka s Rozptýlenými Položkami = Hash table)
** s explicitně řetězenými synonymy. Tabulka je implementována polem
** lineárních seznamů synonym.
**
** Implementujte následující procedury a funkce.
**
**  HTInit ....... inicializuje tabulku před prvním použitím
**  HTInsert ..... vložení prvku
**  HTSearch ..... zjištění přítomnosti prvku v tabulce
**  HTDelete ..... zrušení prvku
**  HTRead ....... přečtení hodnoty prvku
**  HTClearAll ... zrušení obsahu celé tabulky (inicializace tabulky
**                 poté, co již byla použita)
**
** Definici typů naleznete v souboru c016.h.
**
** Tabulka je reprezentována datovou strukturou typu tHTable,
** která se skládá z ukazatelů na položky, jež obsahují složky
** klíče 'key', obsahu 'data' (pro jednoduchost typu float), a
** ukazatele na další synonymum 'ptrnext'. Při implementaci funkcí
** uvažujte maximální rozměr pole HTSIZE.
**
** U všech procedur využívejte rozptylovou funkci hashCode.  Povšimněte si
** způsobu předávání parametrů a zamyslete se nad tím, zda je možné parametry
** předávat jiným způsobem (hodnotou/odkazem) a v případě, že jsou obě
** možnosti funkčně přípustné, jaké jsou výhody či nevýhody toho či onoho
** způsobu.
**
** V příkladech jsou použity položky, kde klíčem je řetězec, ke kterému
** je přidán obsah - reálné číslo.
*/

#include "c016.h"

int HTSIZE = MAX_HTSIZE;
int solved;

/*          -------
** Rozptylovací funkce - jejím úkolem je zpracovat zadaný klíč a přidělit
** mu index v rozmezí 0..HTSize-1.  V ideálním případě by mělo dojít
** k rovnoměrnému rozptýlení těchto klíčů po celé tabulce.  V rámci
** pokusů se můžete zamyslet nad kvalitou této funkce.  (Funkce nebyla
** volena s ohledem na maximální kvalitu výsledku). }
*/

int hashCode ( tKey key ) {
	int retval = 1;
	int keylen = strlen(key);
	for ( int i=0; i<keylen; i++ )
		retval += key[i];
	return ( retval % HTSIZE );
}

/*
** Inicializace tabulky s explicitně zřetězenými synonymy.  Tato procedura
** se volá pouze před prvním použitím tabulky.
*/

void htInit ( tHTable* ptrht ) {
	//kazda polozka tabulky je nastavena na NULL (neobsahuje ziadnu hodnotu)
	for(unsigned i = 0; i < HTSIZE; (*ptrht)[i++] = NULL);
}

/* TRP s explicitně zřetězenými synonymy.
** Vyhledání prvku v TRP ptrht podle zadaného klíče key.  Pokud je
** daný prvek nalezen, vrací se ukazatel na daný prvek. Pokud prvek nalezen není, 
** vrací se hodnota NULL.
**
*/

tHTItem* htSearch ( tHTable* ptrht, tKey key ) {
	if(!ptrht || !key)
		return NULL;
	unsigned index = hashCode(key);
	tHTItem *tmp = NULL;
	//prechod tabulkou
	for(tmp = (*ptrht)[index]; tmp != NULL; tmp = tmp->ptrnext){
		//ak sme nasli prvok s rovnakym klucom ako aktualny prvok vratime ukazatel na aktualny prvok
		if(!(strcmp(tmp->key,key)))
			return tmp;
	}	
	return NULL;
}

/* 
** TRP s explicitně zřetězenými synonymy.
** Tato procedura vkládá do tabulky ptrht položku s klíčem key a s daty
** data.  Protože jde o vyhledávací tabulku, nemůže být prvek se stejným
** klíčem uložen v tabulce více než jedenkrát.  Pokud se vkládá prvek,
** jehož klíč se již v tabulce nachází, aktualizujte jeho datovou část.
**
** Využijte dříve vytvořenou funkci htSearch.  Při vkládání nového
** prvku do seznamu synonym použijte co nejefektivnější způsob,
** tedy proveďte.vložení prvku na začátek seznamu.
**/

void htInsert ( tHTable* ptrht, tKey key, tData data ) {
	if(!ptrht || !key)
		return;
	//zistime ci sa prvok s danym klucom uz nenachadza v tabulke
	tHTItem *tmp = htSearch(ptrht,key);
	//ak sa prvok s danym klucom nachadza v tabulke aktualizujeme data
	if(tmp){
		tmp->data = data;
		return;
	}
	//inak vytvarame novy prvok
	unsigned index = hashCode(key);
	tHTItem *newItem = (tHTItem*)malloc(sizeof(tHTItem));
	if(!newItem)
		return;
	newItem->key = (char*)malloc((strlen(key)+1));
	if(!newItem->key){
		free(tmp);
		return;
	}
	strcpy(newItem->key,key);
	newItem->data = data;
	//prvok vlozime do zoznamu synonym na zaciatok
	tmp = (*ptrht)[index];
	newItem->ptrnext = tmp;
	(*ptrht)[index] = newItem;
}

/*
** TRP s explicitně zřetězenými synonymy.
** Tato funkce zjišťuje hodnotu datové části položky zadané klíčem.
** Pokud je položka nalezena, vrací funkce ukazatel na položku
** Pokud položka nalezena nebyla, vrací se funkční hodnota NULL
**
** Využijte dříve vytvořenou funkci HTSearch.
*/

tData* htRead ( tHTable* ptrht, tKey key ) {
	if(!ptrht || !key)
		return NULL;
	//vyhladanie polozky s danym klucom
	tHTItem *tmp = htSearch(ptrht,key);
	//ak sme uspesne vyhladali polozku vraciame ukazatel na jej data
	if(tmp){
		return &(tmp->data);
	}
	return NULL;
}

/*
** TRP s explicitně zřetězenými synonymy.
** Tato procedura vyjme položku s klíčem key z tabulky
** ptrht.  Uvolněnou položku korektně zrušte.  Pokud položka s uvedeným
** klíčem neexistuje, dělejte, jako kdyby se nic nestalo (tj. nedělejte
** nic).
**
** V tomto případě NEVYUŽÍVEJTE dříve vytvořenou funkci HTSearch.
*/

void htDelete ( tHTable* ptrht, tKey key ) {
	if(!ptrht || !key)
		return;
	unsigned index = hashCode(key);
	tHTItem *tmp = NULL;
	tHTItem *prev = NULL;
	tHTItem *next = NULL;
	//vyhladanie polozky ktoru chceme odstranit
	for(tmp = (*ptrht)[index]; tmp;){
		next = tmp->ptrnext;				//uchovanie hodnoty na nasladujuci prvok aktualneho prvku
		if(!(strcmp(key,tmp->key))){
			//ak sa prvok ktory chceme odstranit nachadza na zaciatku zoznamu tak potom ukazatel
			//na zaciatok zoznamu bude ukazovat na nasledovnika aktualneho prvku
			if(tmp == (*ptrht)[index])			
				(*ptrht)[index] = tmp->ptrnext;	
			//ak sa prvok ktory chceme odstranit nenachadza na zaciatku zoznamu tak predchadzajuci prvok
			//aktualneho prvku musi ukazovat na nasledujuci prvok aktualneho prvku
			else
				prev->ptrnext = tmp->ptrnext;
			free(tmp->key);
			free(tmp);
			return;
		}
		prev = tmp;		//predchadzajuci prvok sa v dalsom opakovani cyklu bude rovnak momentalne aktualnemu prvku
		tmp = next;		//v dalsom opakovani cyklu sa aktualny prvok bude rovnat nasledujucemu prvku momentalne aktualneho
	}	
}

/* TRP s explicitně zřetězenými synonymy.
** Tato procedura zruší všechny položky tabulky, korektně uvolní prostor,
** který tyto položky zabíraly, a uvede tabulku do počátečního stavu.
*/

void htClearAll ( tHTable* ptrht ) {
	if(!ptrht)
		return;
	tHTItem *tmp = NULL;
	for(unsigned i = 0; i < HTSIZE; i++){
		//ak polozka na aktualnom indexe neexistuje pokracujeme na dalsiu
		if(!((*ptrht)[i]))
			continue;
		//ak polozka existuje odstranime vsetky jej synonyma
		while((*ptrht)[i]){
			tmp = (*ptrht)[i];
			(*ptrht)[i] = (*ptrht)[i]->ptrnext;
			if(tmp->key)
				free(tmp->key);
			free(tmp);
		}
	}
}
