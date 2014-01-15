tmpdir := tmp

all: intro.pdf single.pdf git.pdf multi_notes.pdf

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

clean:
	rm -rf tmp intro.pdf single.pdf git.pdf multi_notes.pdf
