tmpdir := tmp

FILES := intro.pdf single.pdf git.pdf multi_notes.pdf openmp_notes.pdf basic.pdf
all: $(FILES)

$(tmpdir):
	mkdir tmp

intro.pdf: | $(tmpdir)
	make -C 00.intro

single.pdf: | $(tmpdir)
	make -C 01.single

git.pdf: | $(tmpdir)
	make -C 02.git

multi_notes.pdf: | $(tmpdir)
	make -C 03.multi

openmp_notes.pdf: | $(tmpdir)
	make -C 04.openmp

basic.pdf: | $(tmpdir)
	make -C 05.basic

clean:
	rm -rf tmp $(FILES)
