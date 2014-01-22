openmp.pdf: compile-gnu compile-vilje mpi-mxv.c openmp-mxv.c openmp-mxv-ref.c \
	    openmp.tex serial-mxv-2.c serial-mxv.c serial-mxv-ref.c

	pdflatex -output-directory ../tmp openmp.tex
	pdflatex -output-directory ../tmp openmp.tex
	mv ../tmp/openmp.pdf ..
