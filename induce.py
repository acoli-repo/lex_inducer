import sys,os,re,gzip,io,requests

import torchtext
import math
import torch
from torch import nn
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import tempfile

def get(vocab,word):
	""" return (vector, index) """
	if not word in vocab.stoi:
		word=word.strip()
	if not word in vocab.stoi:
		word=word.lower()
	idx=-1 # not found
	if word in vocab.stoi:
		idx=vocab.stoi[word]
	return (vocab.get_vecs_by_tokens(word),idx)

def tokenize(text: str, lcase=True):
	text=" ".join(text.split("_"))
	if(lcase):
		text=text.lower()
	text=re.sub(r"([.,;:\"'()<>|!?\[\]{}]+)",r" \1 ",text)
	text=text.strip()
	return text.split()
	

def induce(src_vocab, dict2src_tgt_weight, outfile, lcase=True, corpus_files=[], context=50):
	""" src_vocab: trained vocab with source language embeddings
		dict2src_tgt_weight: key is dict URI or file name, val is a tuple of source column, target column and weight column, weight column may be None (= equally weighted)
		note that we restrict source words to those found in the vocab, target side remains unconstrained
		note that we add weights, this is useful if the dictionary is actually a TSV file with running translations
		corpus_files are TXT or XML files that are used to bootstrap embeddings for unknown words
		if there are corpus_files, context specifies the aggregation window
		outfile is the external copy of the embedding file, if this is found, it is just read
	"""
	try:
		return torchtext.vocab.Vectors(outfile)
	except Exception:
		# e.g., if not os.path.exists(outfile):
		dict2tgt2src2weight={}
		for d,(src_col,tgt_col,weight_col) in dict2src_tgt_weight.items():
			# print(d,(src_col,tgt_col,weight_col))
			dict2tgt2src2weight[d]={}
			if os.path.exists(d):
				myopen=open
				if d.endswith("gz"):
					myopen=gzip.open
				input=myopen(d,"rt",errors="ignore")
			elif d.endswith("gz"): # URL in gz
				input=gzip.GzipFile(fileobj=io.BytesIO(requests.get(d, stream=True).content))
			else: # url, no gzip
				input=StringIO(requests.get(d, stream=True).content)
			for line in input:
				fields=line.strip().decode("utf-8").split("\t")
				# print(fields)
				try:
					src="_".join(fields[src_col].split())
					tgt="_".join(fields[tgt_col].split())
					if not src in src_vocab.stoi:
						src=src.lower()
					print(src,tgt,len(dict2tgt2src2weight[d]))
					if lcase:
						tgt=tgt.lower()
					if src in src_vocab.stoi:
						w=1.0
						if weight_col!=None and len(fields)>weight_col and weight_col>=0:
							w=float(fields[weight_col])
						if not d in dict2tgt2src2weight: dict2tgt2src2weight[d]={}
						if not tgt in dict2tgt2src2weight[d]: dict2tgt2src2weight[d][tgt]={}
						if not src in dict2tgt2src2weight[d][tgt]: 
							dict2tgt2src2weight[d][tgt][src]=w
						else:
							dict2tgt2src2weight[d][tgt][src]+=w
				except Exception:
					pass
			input.close()
			
			# normalize dict entry weights to sum up to 1
			for tgt in dict2tgt2src2weight[d]:
				sum_w=sum(dict2tgt2src2weight[d][tgt].values())
				for src,w in dict2tgt2src2weight[d][tgt].items():
					dict2tgt2src2weight[d][tgt][src]=w/sum_w
		# print(dict2tgt2src2weight)

		tgt2src2weight={}
		# average over all dictionaries
		if len(dict2tgt2src2weight)==1:
			tgt2src2weight=list(dict2tgt2src2weight.values())[0]
		else:
			for d in dict2tgt2src2weight:
				for tgt in dict2tgt2src2weight[d]:
					if not tgt in tgt2src2weight: tgt2src2weight[tgt]={}
					for src,w in dict2tgt2src2weight[d][tgt].items():
						w=w/len(dict2tgt2src2weight)
						if not src in tgt2src2weight[tgt]:
							tgt2src2weight[tgt][src]=w
						else:
							tgt2src2weight[tgt][src]+=w
		# print(tgt2src2weight)

		tgt2emb={}
		for tgt in tgt2src2weight:
			es=[ get(src_vocab,src)[0]*w for src,w in tgt2src2weight[tgt].items() ]
			tgt2emb[tgt]=sum(es) # avg
		
		# print(tgt2emb["übrig"])
		# print(tgt2src2weight["übrig"])
		# print(get(src_vocab,"left"))
		# sys.exit()

		# bootstrap embeddings for OOV words
		if len(corpus_files)>0:
			word2contexts=[]
			for file in corpus_files:
				with open(file, "rt", errors="ignore") as input:
					buffer=[]
					for line in input:
						line=re.sub(r"<[^>]*>","",line)
						line=line.split("<")[0]
						line=line.split(">")[1]
						line=line.strip()
						for tok in tokenize(line,lcase=lcase):
							if re.match(r".*[^.,;:\"'()<>|!?\[\]{}\s0-9_].*",tok): # exclude punctuation and numbers
								buffer.append(tok)
								if len(buffer)>context:
									buffer=buffer[1:]
								if len(buffer)==context:
									word=buffer[int(context/2)]
									if not word in tgt2emb:
										if word in src_vocab.stoi or word.lower() in src_vocab.stoi:
											tgt2emb[word]=get(src_vocab,word)
										else:
											if not word in word2contexts:
												word2contexts[word]=[]
											word2contexts[word].append(buffer)

			tgt2emb_addenda={}
			for word in word2contexts:
				if len(word2contexts[word])>2: # no hapaxes
					es=[]
					for context in word2contexts[word]:
						e = [ tgt2emb[t] for t in c if t in tgt2emb[t] ]
						e = sum(e) / len(e)
						es.append(e)
					tgt2emb_addenda[word]=sum(es)/len(e)
			
			for tgt,emb in tgt2emb_addenda.items():
				if not tgt in tgt2emb:
					tgt2emb[tgt]=emb

		with open(outfile,"wt") as tmpfile:		
			for tgt,e in tgt2emb.items():
				print(tgt,e)
				tmpfile.write(tgt+" "+" ".join([str(val) for val in e.tolist()])+"\n")
				tmpfile.flush()

	return torchtext.vocab.Vectors(outfile)


args=argparse.ArgumentParser(description="""
		lexical inducer, using OPUS and FastText embeddings
		""")
args.add_argument("src", type=str, help="source language, use a BCP47 code")
args.add_argument("tgt", type=str, help="target language")
args.add_argument("dict_file", type=str, help="basename of the dict file (without path), e.g., {src}-{tgt}.dic.gz for OPUS")
args.add_argument("dict_path", type=str, nargs="?", help="directory of dictionary file or URI, defaults to https://object.pouta.csc.fi/OPUS-bible-uedin/v1/dic/, use . for local directory", default="https://object.pouta.csc.fi/OPUS-bible-uedin/v1/dic/")
args.add_argument("-s", "--src_col", type=int, nargs="?", help="source column in dict, defaults to third col (2, OPUS format)", default=2)
args.add_argument("-t","--tgt_col", type=int, nargs="?", help="target column in dict, defaults to fourth col (3, OPUS format)",default=3)
args.add_argument("-w","--weight_col", type=int, nargs="?", help="weight column in dict, defaults to None (i.e., equally weighted); note: OPUS format first col (0)",default=None)
args.add_argument("-o","--outfile", type=str, nargs="?", help="outfile for vocab, defaults to {tgt}_from_{dict_file}.tsv", default=None)
args.add_argument("-d","--debug", action="store_true", help="run on toy data for debugging purposes")
args=args.parse_args()
if args.outfile==None:
	outfile=args.tgt+"_from_"+args.dict_file+".tsv"

if args.debug:
	src_vocab=torchtext.vocab.FastText(language=args.src,max_vectors=1000)
else:
	src_vocab=torchtext.vocab.FastText(language=args.src)
tgt=induce(src_vocab, {os.path.join(args.dict_path,args.dict_file) : (args.src_col, args.tgt_col, args.weight_col)}, outfile)

