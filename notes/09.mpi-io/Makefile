mpiio_notes.pdf: mpiio_notes.tex
	  	 # run twice to resolve references
	  	 pdflatex -output-directory ../tmp mpiio_notes.tex
		 cp referanser.bib ../tmp
		 cd ../tmp && bibtex mpiio_notes
	  	 pdflatex -output-directory ../tmp mpiio_notes.tex
	  	 pdflatex -output-directory ../tmp mpiio_notes.tex
	  	 mv ../tmp/mpiio_notes.pdf ..
