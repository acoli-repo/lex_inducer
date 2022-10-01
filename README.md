# Lexical Inducer: A lexical (symbolic) induction algorithm for multilingual word embeddings

Idea:
- propagate monolingual word embeddings to other languages
- using dictionaries or translation tables (weighted average)
- given source language S with embeddings E_S and target language T without embeddings
  - for every t in T: e_s(t) = \sum_{s\in S} P(t|s) e(s)
- optional: given the initial embeddings E_S, a corpus and an integer n
  - for every s in S for which s not in E_S:
    e_s(s) = avg_c avg_{x in c} e_s(x)
    with c contexts, x words in context and x in E_S
- truly multilingual embeddings can be achieved by:
  - for a set of languages L_1,...,L_n, embedding spaces E_L1, ..., E_Ln
  - for any word l in any of these languages, concatenate: e(l) = e_L1(l) x e_L2(l) x ... x e_Ln(l)
  - perform dimensionality reduction
- notes
  - in order to be balanced, multilingual embeddings that are concatenated must all have the same dimensionality
  - avg may be a poor choice for embedding aggregation (drift towards center), maybe better harmonic mean
