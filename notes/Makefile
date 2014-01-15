tmpdir := tmp

all: intro.pdf single.pdf git.pdf

$(tmpdir):
	mkdir tmp

intro.pdf: | $(tmpdir)
	make -C 00.intro

single.pdf: | $(tmpdir)
	make -C 01.single

git.pdf: | $(tmpdir)
	make -C 02.git

clean:
	rm -rf tmp intro.pdf single.pdf git.pdf
