# Lexical Inducer: A lexical (symbolic) induction algorithm for multilingual word embeddings

Build and test with

  $> make

## Aspects

### Direct Induction

from one language to another: cf.

  $> python3 induce.py -h

- `induce.py`
  - propagate monolingual word embeddings to other languages
  - using dictionaries or translation tables (weighted average)
  - uses FastText embeddings as a basis (hard-wired)
  - given source language S with embeddings E_S and target language T without embeddings
    - for every t in T: e_s(t) = \sum_{s\in S} P(t|s) e(s)
  - optional: given the initial embeddings E_S, a corpus (`-c FILE1[..n]`) and an integer n (currently, 50)
    - for every s in S for which s not in E_S:
      e_s(s) = avg_c avg_{x in c} e_s(x)
      with c contexts, x words in context and x in E_S

### TODO

- truly multilingual embeddings can be achieved by:
  - for a set of languages L_1,...,L_n, embedding spaces E_L1, ..., E_Ln
  - for any word l in any of these languages, concatenate: e(l) = e_L1(l) x e_L2(l) x ... x e_Ln(l)
  - perform dimensionality reduction, keep the weight matrix in order to be able to project future embeddings
- notes
  - in order to be balanced, multilingual embeddings that are concatenated must all have the same dimensionality
  - avg may be a poor choice for embedding aggregation (drift towards center), maybe better harmonic mean
  - as we want to stay interpretable and deterministic, we can actually do *statistical* instead of *neural* dimensionality reduction, e.g., using [TruncatedSVD](https://stackoverflow.com/questions/35103085/how-can-i-use-lsa-to-reduce-dimensionality)

## extensions

bootstrap contextualized (sense) embeddings
  - in translation studies, it is sometimes assumed that one translation corresponds to one sense
  - over parallel text, we can concat *observed* embeddings instead of *induced* embeddings (and likewise, aggregate over induced embeddings from multiple source languages), so that we arrive at *contextualized* embeddings (each representing an observed translation set ~ sense)
  - using this parallel text as a basis, we can then train Bert (etc.) embeddings and learn their mapping to contextualized sense embeddings => *interpretable* contextualized embeddings
