tmpdir := tmp

FILES := p1.pdf s1.pdf p2.pdf s2.pdf p3.pdf s3.pdf p4.pdf p5.pdf

all: $(FILES)

$(tmpdir):
	mkdir tmp

p1.pdf: | $(tmpdir)
	make -C ex1

p2.pdf: | $(tmpdir)
	make -C ex2

p3.pdf: | $(tmpdir)
	make -C ex3

p4.pdf: | $(tmpdir)
	make -C ex4

p5.pdf: | $(tmpdir)
	make -C ex5

s1.pdf: | $(tmpdir)
	make -C ex1

s2.pdf: | $(tmpdir)
	make -C ex2

s3.pdf: | $(tmpdir)
	make -C ex3

clean:
	rm -rf tmp $(FILES)
