
/* c401.c: **********************************************************}
{* Téma: Rekurzivní implementace operací nad BVS
**                                         Vytvořil: Petr Přikryl, listopad 1994
**                                         Úpravy: Andrea Němcová, prosinec 1995
**                                                      Petr Přikryl, duben 1996
**                                                   Petr Přikryl, listopad 1997
**                                  Převod do jazyka C: Martin Tuček, říjen 2005
**                                         Úpravy: Bohuslav Křena, listopad 2009
**                                         Úpravy: Karel Masařík, říjen 2013
**                                         Úpravy: Radek Hranický, říjen 2014
**                                         Úpravy: Radek Hranický, listopad 2015
**                                         Úpravy: Radek Hranický, říjen 2016
**
** Implementujte rekurzivním způsobem operace nad binárním vyhledávacím
** stromem (BVS; v angličtině BST - Binary Search Tree).
**
** Klíčem uzlu stromu je jeden znak (obecně jím může být cokoliv, podle
** čeho se vyhledává). Užitečným (vyhledávaným) obsahem je zde integer.
** Uzly s menším klíčem leží vlevo, uzly s větším klíčem leží ve stromu
** vpravo. Využijte dynamického přidělování paměti.
** Rekurzivním způsobem implementujte následující funkce:
**
**   BSTInit ...... inicializace vyhledávacího stromu
**   BSTSearch .... vyhledávání hodnoty uzlu zadaného klíčem
**   BSTInsert .... vkládání nové hodnoty
**   BSTDelete .... zrušení uzlu se zadaným klíčem
**   BSTDispose ... zrušení celého stromu
**
** ADT BVS je reprezentován kořenovým ukazatelem stromu (typ tBSTNodePtr).
** Uzel stromu (struktura typu tBSTNode) obsahuje klíč (typu char), podle
** kterého se ve stromu vyhledává, vlastní obsah uzlu (pro jednoduchost
** typu int) a ukazatel na levý a pravý podstrom (LPtr a RPtr). Přesnou definici typů 
** naleznete v souboru c401.h.
**
** Pozor! Je třeba správně rozlišovat, kdy použít dereferenční operátor *
** (typicky při modifikaci) a kdy budeme pracovat pouze se samotným ukazatelem 
** (např. při vyhledávání). V tomto příkladu vám napoví prototypy funkcí.
** Pokud pracujeme s ukazatelem na ukazatel, použijeme dereferenci.
**/

#include "c401.h"
int solved;

void BSTInit (tBSTNodePtr *RootPtr) {
/*   -------
** Funkce provede počáteční inicializaci stromu před jeho prvním použitím.
**
** Ověřit, zda byl již strom předaný přes RootPtr inicializován, nelze,
** protože před první inicializací má ukazatel nedefinovanou (tedy libovolnou)
** hodnotu. Programátor využívající ADT BVS tedy musí zajistit, aby inicializace
** byla volána pouze jednou, a to před vlastní prací s BVS. Provedení
** inicializace nad neprázdným stromem by totiž mohlo vést ke ztrátě přístupu
** k dynamicky alokované paměti (tzv. "memory leak").
**	
** Všimněte si, že se v hlavičce objevuje typ ukazatel na ukazatel.	
** Proto je třeba při přiřazení přes RootPtr použít dereferenční operátor *.
** Ten bude použit i ve funkcích BSTDelete, BSTInsert a BSTDispose.
**/
	*RootPtr = NULL;	
}	

int BSTSearch (tBSTNodePtr RootPtr, char K, int *Content)	{
/*  ---------
** Funkce vyhledá uzel v BVS s klíčem K.
**
** Pokud je takový nalezen, vrací funkce hodnotu TRUE a v proměnné Content se
** vrací obsah příslušného uzlu.´Pokud příslušný uzel není nalezen, vrací funkce
** hodnotu FALSE a obsah proměnné Content není definován (nic do ní proto
** nepřiřazujte).
**
** Při vyhledávání v binárním stromu bychom typicky použili cyklus ukončený
** testem dosažení listu nebo nalezení uzlu s klíčem K. V tomto případě ale
** problém řešte rekurzivním volání této funkce, přičemž nedeklarujte žádnou
** pomocnou funkci.
**/
	//ak ukazatel na koren je NULL vratime false
	if(!RootPtr){
		return FALSE;
	}
	//nasli sme hladanu polozku content nastavime na hodnotu hladanej polozky a funkcia vrati hodnotu true
	else if(K == RootPtr->Key){
		*Content = RootPtr->BSTNodeCont;
		return TRUE;
	}
	//ak je hodnota kluca hladaneho uzla mensia ako hodnota kluca v aktualnom uzle hladame v lavom podstrome
	else if(K < RootPtr->Key){
		return BSTSearch(RootPtr->LPtr,K,Content);
	}
	//ak je hodnota kluca hladaneho uzla vascia ako hodnota kluca v aktualnom uzle hladame v pravom podstrome
	else if(K > RootPtr->Key){
		return BSTSearch(RootPtr->RPtr,K,Content);
	}
	//ak uzol nenajdeme vraciame false
	return FALSE;	
} 


void BSTInsert (tBSTNodePtr* RootPtr, char K, int Content)	{	
/*   ---------
** Vloží do stromu RootPtr hodnotu Content s klíčem K.
**
** Pokud již uzel se zadaným klíčem ve stromu existuje, bude obsah uzlu
** s klíčem K nahrazen novou hodnotou. Pokud bude do stromu vložen nový
** uzel, bude vložen vždy jako list stromu.
**
** Funkci implementujte rekurzivně. Nedeklarujte žádnou pomocnou funkci.
**
** Rekurzivní implementace je méně efektivní, protože se při každém
** rekurzivním zanoření ukládá na zásobník obsah uzlu (zde integer).
** Nerekurzivní varianta by v tomto případě byla efektivnější jak z hlediska
** rychlosti, tak z hlediska paměťových nároků. Zde jde ale o školní
** příklad, na kterém si chceme ukázat eleganci rekurzivního zápisu.
**/
	tBSTNodePtr tmp = NULL;
	//ak sme nasli misto kde ma byt uzol vlozeny a uzol neexistuje vlozime vytovorime novy
	if(!(*RootPtr)){
		tmp = (tBSTNodePtr)malloc(sizeof(struct tBSTNode));
		if(tmp == NULL)
			return;
		tmp->LPtr = tmp->RPtr = NULL;
		tmp->BSTNodeCont = Content;
		tmp->Key = K;
		*RootPtr = tmp;
	}
	//ak je hodnota kluca uzla ktory chceme vlozit mensia ako hodnota kluca aktualneho uzla chceme vlozit 
	//uzol niekde do laveho podstromu
	else if(K < (*RootPtr)->Key){
		BSTInsert(&((*RootPtr)->LPtr), K, Content);
	}
	//ak je hodnota kluca uzla ktory chceme vlozit vacsia ako hodnota kluca aktualneho uzla chceme vlozit
	//uzol niekde do praveho podstromu
	else if(K > (*RootPtr)->Key){
		BSTInsert(&((*RootPtr)->RPtr), K, Content);
	}
	//ak uzol existuje aktualizujeme hodnotu
	else{
		(*RootPtr)->BSTNodeCont = Content;
	}	
}

void ReplaceByRightmost (tBSTNodePtr PtrReplaced, tBSTNodePtr *RootPtr) {
/*   ------------------
** Pomocná funkce pro vyhledání, přesun a uvolnění nejpravějšího uzlu.
**
** Ukazatel PtrReplaced ukazuje na uzel, do kterého bude přesunuta hodnota
** nejpravějšího uzlu v podstromu, který je určen ukazatelem RootPtr.
** Předpokládá se, že hodnota ukazatele RootPtr nebude NULL (zajistěte to
** testováním před volání této funkce). Tuto funkci implementujte rekurzivně. 
**
** Tato pomocná funkce bude použita dále. Než ji začnete implementovat,
** přečtěte si komentář k funkci BSTDelete(). 
**/

	//kym existuje pravy podstrom volame rekurzivne funkciu az sa dostaneme do najpravejsieho uzla
	if((*RootPtr)->RPtr){
		ReplaceByRightmost(PtrReplaced,&((*RootPtr)->RPtr));
	}
	//presunieme hodnotu z RootPtr do PtrReplaced
	else{
		tBSTNodePtr tmp = *RootPtr;
		PtrReplaced->Key = (*RootPtr)->Key;
		PtrReplaced->BSTNodeCont = (*RootPtr)->BSTNodeCont;
		*RootPtr = (*RootPtr)->LPtr;
		free(tmp);
	}
}

void BSTDelete (tBSTNodePtr *RootPtr, char K) 		{
/*   ---------
** Zruší uzel stromu, který obsahuje klíč K.
**
** Pokud uzel se zadaným klíčem neexistuje, nedělá funkce nic. 
** Pokud má rušený uzel jen jeden podstrom, pak jej zdědí otec rušeného uzlu.
** Pokud má rušený uzel oba podstromy, pak je rušený uzel nahrazen nejpravějším
** uzlem levého podstromu. Pozor! Nejpravější uzel nemusí být listem.
**
** Tuto funkci implementujte rekurzivně s využitím dříve deklarované
** pomocné funkce ReplaceByRightmost.
**/

	//ak je koren NULL funkcia nerobi nic	
	if(!(*RootPtr))
		return;
	//ak je hodnota kluca prvku ktory chceme odstranit mensia ak hodnota klucu
	//aktualneho uzlu chceme mazat uzol v jeho lavom podstrome
	else if(K < (*RootPtr)->Key){
		BSTDelete(&((*RootPtr)->LPtr),K);
	}
	//ak je hodnota kluca prvku ktory chceme odstranit vacsia ak hodnota klucu
	//aktualneho uzlu chceme mazat uzol v jeho pravom podstrome
	else if(K > (*RootPtr)->Key){
		BSTDelete(&((*RootPtr)->RPtr),K);
	}
	//ak sme nasli uzol ktory chceme odstranit
	else{
		tBSTNodePtr tmp = *RootPtr;
		//ak nema potomkov uvolnime ho
		if(!tmp->LPtr && !tmp->RPtr){
			free(tmp);
			*RootPtr = NULL;
		}
		//ak ma iba pravy podstrom tak ho zdedi jeho otec
		else if(!tmp->LPtr){
			*RootPtr = tmp->RPtr;
			free(tmp);
		}
		//ak ma iba lavy podstrom tak ho zdedi jeho otec
		else if(!tmp->RPtr){
			*RootPtr = tmp->LPtr;
			free(tmp);
		}
		//ak ma oboch potomkov tak tak je ruseny uzol nahradeni napravejsim uzlom laveho podstromu
		else{
			ReplaceByRightmost (*RootPtr, (&(*RootPtr)->LPtr));
		}
	}	
} 

void BSTDispose (tBSTNodePtr *RootPtr) {	
/*   ----------
** Zruší celý binární vyhledávací strom a korektně uvolní paměť.
**
** Po zrušení se bude BVS nacházet ve stejném stavu, jako se nacházel po
** inicializaci. Tuto funkci implementujte rekurzivně bez deklarování pomocné
** funkce.
**/
	//ak existuje koren
	if((*RootPtr)){
		BSTDispose(&((*RootPtr)->LPtr));	//zrusime lavu vetvu
		BSTDispose(&((*RootPtr)->RPtr));	//zrusime pravu vetvu
		free((*RootPtr));		//uvolnime koren
		*RootPtr = NULL;	//koren nastavime do stavu po inicializacii
	}
}

/* konec c401.c */

