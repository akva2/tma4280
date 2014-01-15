SRCDIR := ../../examples/crash

crash.pdf: crash.tex $(SRCDIR)/CMakeLists.txt \
  	   $(SRCDIR)/hello2.c $(SRCDIR)/hello3.c $(SRCDIR)/hello4.c \
	   $(SRCDIR)/hello.c $(SRCDIR)/hello.f $(SRCDIR)/Makefile \
	   $(SRCDIR)/timing.c $(SRCDIR)/utils2.c $(SRCDIR)/utils2.h \
	   $(SRCDIR)/utils.c $(SRCDIR)/utils.h test/CMakeLists-openmpopt.txt \
	   test/CMakeLists-openmp.txt \
	   test/CMakeLists.txt test/forcecompilers.sh test/oot.sh test/test.c
	pdflatex -output-directory ../tmp crash.tex
	mv ../tmp/crash.pdf ..
