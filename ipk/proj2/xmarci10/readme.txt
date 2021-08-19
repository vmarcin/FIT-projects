DHCP STARVATION APLIKACIA:

	- ulohou aplikacie je realizacia DHCP Starvation utoku, ktory vycerpa
	 	adresny pool legitimneho DHCP Server
	- aplikacia to robi pomocou posielania velkeho mnozstva DHCP Discover
		(nerealizuje sa cela DHCP komunikacia)

	SPUSTENIE:

	$./ipk-dhcpstarve -i interface
		
		-interface = meno rozhrania podla OS, na ktore utocnik posiela spravy

	LICENCIA:

	- Free Software
		

