Pokud je požadováno přemístění nákladu z jednoho místa do druhého, vozík si materiál vyzvedne do 1 minuty. 
    - "náklad" -> materiál : vo zvyšku textu referencia na materiál takže som to zmenil
    - nešpecifikovaná priorita materiálu pri vytvorení požiadavky a teda predpokladám, že
      pri vytvorení požidavky sa stále jedná o neprioritný materiál
    - neurčitosť : "z jednoho místa do druhého" -> nahradené zdrojovou a cieľovou zastávkou
    - dangling else : ale to je vyriešené v ďalšej vete

Pokud se to nestihne, materiálu se nastavuje prioritní vlastnost. 
    - neurčitosť : "se to nestihne" implicitne ale vieme predpokladať a tak som to spojil
      do jednej vety a vytvoril implikáciu

*Ak je vytvorená požiadavka na presun materiálu (požidavka stále uvažuje neprioritný materiál) zo zdrojovej zastávky do cieľovej zastávky, vozík si materiál naloží do 1 minuty, inak sa materiál stáva prioritným.*


Každý prioritní materiál musí být vyzvednutý vozíkem do 1 minuty od nastavení prioritního požadavku. 
    - dangling else: veta obsahuje nutnú podmienku ("musí"), ale čo v prípade, že podmienka
      nenastane. Spojením s nasledujúcou vetou som vytvoril implikáciu a doplnil vyvoalnie
      výnimky v prípade, že sa nakládka nestihne.
Pokud vozík nakládá prioritní materiál, přepíná se do režimu pouze-vykládka. 
    - nieje špecifikovaný defaultný režim
    - nie je explicitne špecifikované fungovanie režimu iba-vykládka
    
*Ak sa prioritný materiál naloží do 1 minuty od nastavenia prioritnej vlastnosti, vozík sa prepne z režimu "normálny" do režimu "iba-vykládka", inak sa vyvolá výnimka X1. V režime iba-vykládka vozík iba vykladá ľubovoľný (prioritný aj neprioritný) materiál no nesmie prijať nový.*

V tomto režimu zůstává, dokud nevyloží všechen takový materiál. 
    - nejasnosti: "V tomto režimu", "všechen takový materiál"
    - opačný smer implikácie. Lepšie od príčiny k dôsledku.
    - chýba akcia, čo sa stane ak sa všetok "taký" náklad vyloži

*Ak sa vyloží všetok prioritný materiál vozík sa prepne do “normálneho” režimu, inak ostáva v režime "iba-vykládka".*

Normálně vozík během své jízdy může nabírat a vykládat další materiály v jiných zastávkách. 
    - nejasnosť: čo znamená "Normálně". Pomenoval som to ako "normálny režim."
    - implicitný predpoklad normálneho režime -> lepšie priamo špecifikovať.
    
*Ak je vozík v predvolenom "normálnom" režime je povolené pri zastávke naložiť nový materiál a vyložiť prevážaný materiál.*

Na jednom místě může vozík akceptovat nebo vyložit jeden i více materiálů.
    - neurčitosť: slovo "může" -> malo by byť vždy jasné a jednoznačné čo sa stane
    - neurčitosť: slovo "nebo" -> exclusive alebo nie ?

*Na jednej zastávke vozík najprv vyloží všetky materiály ktorých cieľová zastávka sa rovná aktuálnej zastávke a následne začne nakladať čakajúci materiál v aktuálnej zastávke.*  

Pořadí vyzvednutí materiálů nesouvisí s pořadím vytváření požadavků. 
    - tým pádom nemá vplyv na testovanie a poradie nakladanie môže byť ľubovoľné

*Poradie vyzdvyhnutia materiálov nesúvisí s poradím vytvárania požiadaviek.*

Vozík neakceptuje materiál, pokud jsou všechny jeho sloty obsazené nebo by jeho převzetím byla překročena maximální nosnost.
    - typ chyby: obemedzenie nakladania súvisí aj s režimom iba-vykládka
    - chýba pozitivný prípad
    - taktiež pre lepšie pochopenie som otočil implikáciu a doplnili dangling else

*Ak má vozík dostatok voľných slotov a zároveň naloženie materiálu neprekročí maximálnu nosnosť a zaroven nieje v rezime iba-vykladka, vozík naloží materiál, inak výnimka X1.*