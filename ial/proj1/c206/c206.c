	
/* c206.c **********************************************************}
{* Téma: Dvousměrně vázaný lineární seznam
**
**                   Návrh a referenční implementace: Bohuslav Křena, říjen 2001
**                            Přepracované do jazyka C: Martin Tuček, říjen 2004
**                                            Úpravy: Kamil Jeřábek, říjen 2017
**
** Implementujte abstraktní datový typ dvousměrně vázaný lineární seznam.
** Užitečným obsahem prvku seznamu je hodnota typu int.
** Seznam bude jako datová abstrakce reprezentován proměnnou
** typu tDLList (DL znamená Double-Linked a slouží pro odlišení
** jmen konstant, typů a funkcí od jmen u jednosměrně vázaného lineárního
** seznamu). Definici konstant a typů naleznete v hlavičkovém souboru c206.h.
**
** Vaším úkolem je implementovat následující operace, které spolu
** s výše uvedenou datovou částí abstrakce tvoří abstraktní datový typ
** obousměrně vázaný lineární seznam:
**
**      DLInitList ...... inicializace seznamu před prvním použitím,
**      DLDisposeList ... zrušení všech prvků seznamu,
**      DLInsertFirst ... vložení prvku na začátek seznamu,
**      DLInsertLast .... vložení prvku na konec seznamu, 
**      DLFirst ......... nastavení aktivity na první prvek,
**      DLLast .......... nastavení aktivity na poslední prvek, 
**      DLCopyFirst ..... vrací hodnotu prvního prvku,
**      DLCopyLast ...... vrací hodnotu posledního prvku, 
**      DLDeleteFirst ... zruší první prvek seznamu,
**      DLDeleteLast .... zruší poslední prvek seznamu, 
**      DLPostDelete .... ruší prvek za aktivním prvkem,
**      DLPreDelete ..... ruší prvek před aktivním prvkem, 
**      DLPostInsert .... vloží nový prvek za aktivní prvek seznamu,
**      DLPreInsert ..... vloží nový prvek před aktivní prvek seznamu,
**      DLCopy .......... vrací hodnotu aktivního prvku,
**      DLActualize ..... přepíše obsah aktivního prvku novou hodnotou,
**      DLSucc .......... posune aktivitu na další prvek seznamu,
**      DLPred .......... posune aktivitu na předchozí prvek seznamu, 
**      DLActive ........ zjišťuje aktivitu seznamu.
**
** Při implementaci jednotlivých funkcí nevolejte žádnou z funkcí
** implementovaných v rámci tohoto příkladu, není-li u funkce
** explicitně uvedeno něco jiného.
**
** Nemusíte ošetřovat situaci, kdy místo legálního ukazatele na seznam 
** předá někdo jako parametr hodnotu NULL.
**
** Svou implementaci vhodně komentujte!
**
** Terminologická poznámka: Jazyk C nepoužívá pojem procedura.
** Proto zde používáme pojem funkce i pro operace, které by byly
** v algoritmickém jazyce Pascalovského typu implemenovány jako
** procedury (v jazyce C procedurám odpovídají funkce vracející typ void).
**/

#include "c206.h"

int solved;
int errflg;

void DLError() {
/*
** Vytiskne upozornění na to, že došlo k chybě.
** Tato funkce bude volána z některých dále implementovaných operací.
**/	
    printf ("*ERROR* The program has performed an illegal operation.\n");
    errflg = TRUE;             /* globální proměnná -- příznak ošetření chyby */
    return;
}

void DLInitList (tDLList *L) {
/*
** Provede inicializaci seznamu L před jeho prvním použitím (tzn. žádná
** z následujících funkcí nebude volána nad neinicializovaným seznamem).
** Tato inicializace se nikdy nebude provádět nad již inicializovaným
** seznamem, a proto tuto možnost neošetřujte. Vždy předpokládejte,
** že neinicializované proměnné mají nedefinovanou hodnotu.
**/
	L->First = NULL;
	L->Act = NULL;
	L->Last = NULL;	
}

void DLDisposeList (tDLList *L) {
/*
** Zruší všechny prvky seznamu L a uvede seznam do stavu, v jakém
** se nacházel po inicializaci. Rušené prvky seznamu budou korektně
** uvolněny voláním operace free. 
**/
	tDLElemPtr tmp = NULL;		//pomocny ukazatel na prvok zoznamu
	/* cyklus sa bude opakovat kym sa neodstrania vsetky prvky
	 */
	while(L->First != NULL){
		tmp = L->First;							//uchovame minulu hodnotu prveho prvku zoznamu
		L->First = L->First->rptr;  //hodnotu prveho prvku nastavime na nasledujuci prvok
		free(tmp);									//uvolnime prvok ktory bol prvy
	}	
	L->Last = NULL;		//uvedieme do stavu po inicializacii
	L->Act = NULL;	  //uvedieme do stavu po inicializacii
}

void DLInsertFirst (tDLList *L, int val) {
/*
** Vloží nový prvek na začátek seznamu L.
** V případě, že není dostatek paměti pro nový prvek při operaci malloc,
** volá funkci DLError().
**/
	tDLElemPtr newItem = (tDLElemPtr)malloc(sizeof(struct tDLElem)); //alokacia noveho prvku
	
	/* ak sa alokacia nepodari chlasime chybu a ukoncime funckiu
	 */
	if(newItem == NULL){	
		DLError();
		return;
	}
	newItem->data = val;			//nastavime data noveho prvku
	newItem->lptr = NULL;			//kedze novy prvok sa ma stat prvym prvkom zoznamu nema predchadzajuci prvok
	newItem->rptr = L->First; //jeho nasledovnikom sa stava doposial prvy prvok zoznamu
	
	/* ak je zoznam prazdny novy prvok sa stava prvym a zaroven poslednym prvokom
	 */
	if(L->First == NULL){		
		L->First = newItem;			
		L->Last = newItem;
	}
	/* ak zoznam nie je prazdny musime zaistit aby sa novy prvok stal prvym prvkom zoznamu
	 */	
	else{
		L->First->lptr = newItem;				//doposial prvy prvok zoznamu musi mat za predchodcu novy prvok
		L->First = newItem;							//novy prvok sa stava prvym prvkom
	}
}

void DLInsertLast(tDLList *L, int val) {
/*
** Vloží nový prvek na konec seznamu L (symetrická operace k DLInsertFirst).
** V případě, že není dostatek paměti pro nový prvek při operaci malloc,
** volá funkci DLError().
**/ 	
	tDLElemPtr newItem = (tDLElemPtr)malloc(sizeof(struct tDLElem));	//alokacia noveho prvku
	
	/* ak sa alokacia nepodari chlasime chybu a ukoncime funkciu
	 */
	if(newItem == NULL){
		DLError();
		return;
	}
	newItem->data = val;		 //nastavime data noveho prvku
	newItem->lptr = L->Last; //kedze novy prvok sa ma stat poslednym prvkom zoznamu jeho predchodca sa stava doposial posledny prvok
	newItem->rptr = NULL;  	 //posledny prvok nema nasledovnika
	
	/* ak je zoznam prazdny novy prvok sa stava prvym a zaroven posledny prvkom
	 */
	if(L->First == NULL){
		L->First = newItem;
		L->Last = newItem;
	}
	/* ak zoznam nie je prazdny musime zaistit aby sa novy prvok stal poslednym prvok zoznamu
	 */
	else{
		L->Last->rptr = newItem;  	//doposial prvy prvok zoznamu musi mat za nasledovnika novy prvok
		L->Last = newItem;					//novy prvok sa stava poslednym prvkom
	}
}

void DLFirst (tDLList *L) {
/*
** Nastaví aktivitu na první prvek seznamu L.
** Funkci implementujte jako jediný příkaz (nepočítáme-li return),
** aniž byste testovali, zda je seznam L prázdný.
**/
	L->Act = L->First;		//prvy prvok zoznamu sa stava aktivnym
}

void DLLast (tDLList *L) {
/*
** Nastaví aktivitu na poslední prvek seznamu L.
** Funkci implementujte jako jediný příkaz (nepočítáme-li return),
** aniž byste testovali, zda je seznam L prázdný.
**/
	L->Act = L->Last;		 //posledny prvok zoznamu sa stava aktivnym	
}

void DLCopyFirst (tDLList *L, int *val) {
/*
** Prostřednictvím parametru val vrátí hodnotu prvního prvku seznamu L.
** Pokud je seznam L prázdný, volá funkci DLError().
**/
	/* ak je zoznam prazdny chlasime chybu a 
	 * ukoncime funkciu
	 */
	if(L->First == NULL){
		DLError();
		return;
	}
	/* do premmenej val ulozime hodnotu prveho prvku zoznamu
	 */
	*val = L->First->data;
}

void DLCopyLast (tDLList *L, int *val) {
/*
** Prostřednictvím parametru val vrátí hodnotu posledního prvku seznamu L.
** Pokud je seznam L prázdný, volá funkci DLError().
**/
	/* ak je zoznam prazdny chlasime chybu a 
	 * ukoncime funkciu
	 */
	if(L->First == NULL){
		DLError();
		return;
	}
	/* do premmenej val ulzime hodnotu posledneho prvku zoznamu
 	 */
	*val = L->Last->data;
}

void DLDeleteFirst (tDLList *L) {
/*
** Zruší první prvek seznamu L. Pokud byl první prvek aktivní, aktivita 
** se ztrácí. Pokud byl seznam L prázdný, nic se neděje.
**/
	/* ak zoznam nie je prazdny
	 */
	if(L->First != NULL){		
	
		/* ak bol prvy prvok aktivny aktivita sa straca
	   */	
		if(L->First == L->Act){
			L->Act = NULL;
		}
		/* ak zoznam obsahuje iba jeden prvok(je zaroven prvy aj posledny)
		 * uvolnime tento prvok, zoznam uvedieme do stavu po inicializacii
	   * a funkciu ukoncime
		 */
		if(L->First == L->Last){	
  		free(L->First);
			L->First = NULL;
			L->Last = NULL;
    	return;
  	}
		tDLElemPtr tmp = L->First;		//do pomocneho ukazatela na prvok zoznamu ulozime doposial prvy prvok zoznamu
		L->First = L->First->rptr;		//prvym prvokm zoznamu sa stava nasladujuci prvok
		L->First->lptr = NULL;				//odstranime vazbu na doposial prvy prvok zoznamu
		free(tmp);										//uvolnime doposial prvy prvok zoznamu
	}		
}	

void DLDeleteLast (tDLList *L) {
/*
** Zruší poslední prvek seznamu L. Pokud byl poslední prvek aktivní,
** aktivita seznamu se ztrácí. Pokud byl seznam L prázdný, nic se neděje.
**/
	/* ak zoznam nie je prazdny
	 */ 
	if(L->First != NULL){
		
		/* ak bol posledny prvok aktivny aktivita sa straca
 		 */
		if(L->Last == L->Act){
			L->Act = NULL;
		}
		
		/* ak zoznam obsahuje iba jeden prvok (prvy a posledny prvok sa rovnaju)
		 * uvolnime tento prvok, zoznma uvedieme do stavu po inicializacii
		 * a ukoncime funkciu
		 */
		if(L->First == L->Last){
			free(L->Last);
			L->First = NULL;
			L->Last = NULL;
			return;
		}
		tDLElemPtr tmp = L->Last;			//do pomocneho ukazatela na prvok zoznamu ulozime doposial posledny prvok zoznamu
		L->Last = L->Last->lptr;			//poslednym prvok zoznamu sa stava predchadzajuci prvok
		L->Last->rptr = NULL;					//odstranime vazbu na doposial posledny prvok zoznamu
		free(tmp);										//uvolnime doposial posledny prvok zoznamu
	}
}

void DLPostDelete (tDLList *L) {
/*
** Zruší prvek seznamu L za aktivním prvkem.
** Pokud je seznam L neaktivní nebo pokud je aktivní prvek
** posledním prvkem seznamu, nic se neděje.
**/
	/* ak zoznam nieje aktivny alebo je aktivny jeho posledny prvok ukoncime funkciu
	 */
	if(L->Act == NULL || L->Act == L->Last)	
		return;
	tDLElemPtr tmp = L->Act->rptr;	//do pomocneho ukazatela ulozime nasledovnika aktivneho prvku (prvok ktory chceme odstranit)
	
	/* ak prvok ktory chceme odstranit je poslednym prvokm zoznamu
	 */
	if(tmp == L->Last){
		free(tmp);
		L->Act->rptr = NULL;	//aktualny prvok straca nasledovnika
		L->Last = L->Act;			//aktualny prvok sa stava poslednym prvokm zoznamu
		return;
	}
	L->Act->rptr = tmp->rptr;	//nasledovnikom aktualneho prvku sa stava nasledovnik prvku ktory chceme odstranit
	tmp->rptr->lptr = L->Act;	//predchodcom nasledovnika prvku ktory chceme odstranit sa stava aktualny prvok
	free(tmp);								//uvolnime prvok
}

void DLPreDelete (tDLList *L) {
/*
** Zruší prvek před aktivním prvkem seznamu L .
** Pokud je seznam L neaktivní nebo pokud je aktivní prvek
** prvním prvkem seznamu, nic se neděje.
**/
	/* ak zoznam nije aktivny alebo je aktivny jeho posledny prvok ukoncime funkciu
	 */
	if(L->Act == NULL || L->Act == L->First)
		return;
	tDLElemPtr tmp = L->Act->lptr;	//do pomocneho ukazatela ulozime predchodcu aktivneho prvku (prvok ktory chceme odstranit)
	
	/* ak prvok ktory chceme odstranit je prvym prvkom zoznamu
	 */
	if(tmp == L->First){
		free(tmp);						
		L->Act->lptr = NULL;	//aktualny prvok straca predchodcu
		L->First = L->Act;		//aktualny prvok sa stava prvym prvkom zoznamu
		return;
	}
	L->Act->lptr = tmp->lptr;	//predchodca aktualneho prvku sa stava predchodca prvku ktory chceme odstranit
	tmp->lptr->rptr = L->Act;	//nasledovnikom predchodcu prvku ktory chceme odstranit sa stava aktualny prvok
	free(tmp);								//uvolnime prvok
}

void DLPostInsert (tDLList *L, int val) {
/*
** Vloží prvek za aktivní prvek seznamu L.
** Pokud nebyl seznam L aktivní, nic se neděje.
** V případě, že není dostatek paměti pro nový prvek při operaci malloc,
** volá funkci DLError().
**/
	/* ak zoznam nieje aktivny ukoncime funkciu
	 */
	if(L->Act == NULL)
		return;
	tDLElemPtr newItem = (tDLElemPtr)malloc(sizeof(struct tDLElem)); //alokacia noveho prvku

	/* ak sa nepodari alokacia chlasime chybu a ukoncime funkciu
	 */
	if(newItem == NULL){
		DLError();
		return;
	}
	newItem->data = val; //ulozime hodnotu 'val' do noveho prvku
	/* ak je aktivnym prvkom posledny prvok zoznamu
	 */
	if(L->Act == L->Last){
		L->Act->rptr = newItem;		//nasledovnimkom aktualneho prvku sa stava novy prvok
		newItem->lptr = L->Act;		//predchodcom noveho prvku sa stava aktualny
		newItem->rptr = NULL;			//novy prvok nema nasledovnika
		L->Last = newItem;				//novy prvok sa stava poslednym prvok zoznamu
	}
	else{
		newItem->rptr = L->Act->rptr;	//naslednikom noveho prvku sa stava prvok ktory bol doposial nasledovnikom aktualneho prvku
		L->Act->rptr->lptr = newItem;	//predchocom doposial nasledovnika aktivneho prvku sa stava novy prvok
		L->Act->rptr = newItem;				//nasledovnikom aktualneho prvku sa stava novy prvok
		newItem->lptr = L->Act;				//predchodcom noveho prvku sa stava aktivny prvok
	}
}

void DLPreInsert (tDLList *L, int val) {
/*
** Vloží prvek před aktivní prvek seznamu L.
** Pokud nebyl seznam L aktivní, nic se neděje.
** V případě, že není dostatek paměti pro nový prvek při operaci malloc,
** volá funkci DLError().
**/
	/* ak zoznam nieje aktivny ukoncime funkciu
	 */
	if(L->Act == NULL)
		return;
	tDLElemPtr newItem = (tDLElemPtr)malloc(sizeof(struct tDLElem)); //alokacia noveho prvku

	/* ak sa nepodari alokacia chlasime chybu a ukoncime funkciu
	 */
	if(newItem == NULL){
		DLError();
		return;
	}
	newItem->data = val; //ulozime hodnotu 'val' do noveho prvku
	/* ak je aktivnym prvkom prvy prvok zoznamu 
	 */
	if(L->Act == L->First){
		L->Act->lptr = newItem;		//predchodcom aktivneho prvku sa stava novy prvok
		newItem->rptr = L->Act;		//nasledovnikom noveho prvku sa stava aktivny prvok
		newItem->lptr = NULL;			//novy prvok nema predchodcu
		L->First = newItem;				//novy prvok sa stava prvym prvkom zoznamu
	}
	else{
		newItem->lptr = L->Act->lptr;	//predchodcom noveho prvku sa stava prvok ktory bol doposial prechodca aktulaneho prvku
		L->Act->lptr->rptr = newItem;	//nasledovnikom doposial predchocu aktivneho prvku sa stava novy prvok
		L->Act->lptr = newItem;				//predchodcom aktualneho prvku sa stava novy prvok
		newItem->rptr = L->Act;				//nasledovnikom noveho prvku sa stava aktivny prvok
	}
	
}

void DLCopy (tDLList *L, int *val) {
/*
** Prostřednictvím parametru val vrátí hodnotu aktivního prvku seznamu L.
** Pokud seznam L není aktivní, volá funkci DLError ().
**/
	/* ak zoznam nieje aktivny chlasime chybu 
	 * a ukoncime funkciu
	 */
	if(L->Act == NULL){
		DLError();
		return;
	}	
	/* do premennej val ulozime hodnotu aktivneho prvku
	 */
	*val = L->Act->data;
}

void DLActualize (tDLList *L, int val) {
/*
** Přepíše obsah aktivního prvku seznamu L.
** Pokud seznam L není aktivní, nedělá nic.
**/
	/* ak je zoznam aktivny prepisem hodnotu 
	 * aktivenho prvko hodnotou val inak sa 
	 * nvykona nic
	 */
	if(L->Act != NULL){
		L->Act->data = val;
	}	
}

void DLSucc (tDLList *L) {
/*
** Posune aktivitu na následující prvek seznamu L.
** Není-li seznam aktivní, nedělá nic.
** Všimněte si, že při aktivitě na posledním prvku se seznam stane neaktivním.
**/
	/* ak je zoznam aktivny posunieme 
	 * aktivitu na nasledujuci prvok
	 */
	if(L->Act != NULL){
		L->Act = L->Act->rptr;
	}	
}


void DLPred (tDLList *L) {
/*
** Posune aktivitu na předchozí prvek seznamu L.
** Není-li seznam aktivní, nedělá nic.
** Všimněte si, že při aktivitě na prvním prvku se seznam stane neaktivním.
**/
	/* ak je zoznam aktivny posunieme
	 * aktivitu na nasledujuci prvok
	 */
	if(L->Act != NULL){
		L->Act = L->Act->lptr;
	}	
}

int DLActive (tDLList *L) {
/*
** Je-li seznam L aktivní, vrací nenulovou hodnotu, jinak vrací 0.
** Funkci je vhodné implementovat jedním příkazem return.
**/
	/* ak je zoznam aktivny vrati 1 inak 0
	 */
	return (L->Act != NULL);
}

/* Konec c206.c*/
