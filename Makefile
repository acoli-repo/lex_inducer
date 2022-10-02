all: data
	@# test run (-d)
	python3 induce.py en de de-en.dic.gz -s 3 -t 2 -w 0 -c data/bib.de.xml -d

data:
	if [ ! -e data ]; then mkdir data; fi
	wget -nc https://github.com/acoli-repo/acoli-corpora/raw/master/biblical/data/germ/deu_german/German.xml -O data/bib.de.xml
	wget -nc https://github.com/acoli-repo/acoli-corpora/raw/master/biblical/data/indoeuropean-other/baltic/lit_lithuanian/Lithuanian.xml -O data/bib.lt.xml
	wget -nc https://github.com/acoli-repo/acoli-corpora/raw/master/biblical/data/indoeuropean-other/romance_italic/lat_latin/latin_vulgata_clementina_utf8.xml -O data/bib.la.xml
	wget -nc https://github.com/acoli-repo/acoli-corpora/raw/master/biblical/data/germ/en_modern-english/web.xml -O data/bib.en.xml
	