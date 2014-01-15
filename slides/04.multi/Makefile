NOTESDIR := ../../notes/03.multi

multi.pdf: e5-2670.pdf multi.tex \
  	   $(NOTESDIR)/sandy.png \
  	   $(NOTESDIR)/multiprocess1.pdf \
	   $(NOTESDIR)/multiprocess2.pdf \
	   $(NOTESDIR)/crossbar.pdf \
	   $(NOTESDIR)/ccNUMA.pdf \
	   $(NOTESDIR)/MIMD.pdf \
	   $(NOTESDIR)/2Dmesh.pdf \
	   $(NOTESDIR)/2Dtoroid.pdf
	pdflatex -output-directory ../tmp multi.tex
	mv ../tmp/multi.pdf ..
