#author	Vladimir Marcin 1-BIT
#email	xmarci10@stud.fit.vutbr.cz
#date	3.3.2017

filename = proj2

all:
	latex $(filename).tex
	latex $(filename).tex
	dvips -t a4 $(filename).dvi
	ps2pdf $(filename).ps

view:
	evince $(filename).pdf

clean:
		rm -rf 	$(filename).log $(filename).dvi $(filename).aux $(filename).pdf $(filename).ps
	
