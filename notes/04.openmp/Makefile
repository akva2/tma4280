openmp_notes.pdf: CriticalSection.pdf \
		  mpi-openmp-mxv.c \
		  mpi.pdf \
		  openmp-for.c \
		  openmp-for-dynamic.c \
		  openmp-for-dynamic-chunk.c \
		  openmp-for.f \
		  openmp-for-guided-chunk.c \
		  openmp-integrate.c \
		  openmp_notes.tex \
		  openmp.pdf \
		  openmp-sections.c \
		  referanser.bib \
		  serial.c \
		  serial-for.c \
		  serial-for.f \
		  serial-integrate.c \
		  serial-mxv-ref.c \
		  serial-sections.c

		  # run twice to resolve references
		  pdflatex -output-directory ../tmp openmp_notes.tex
		  cp referanser.bib ../tmp
		  cd ../tmp && bibtex openmp_notes
		  pdflatex -output-directory ../tmp openmp_notes.tex
		  pdflatex -output-directory ../tmp openmp_notes.tex
		  mv ../tmp/openmp_notes.pdf ..
