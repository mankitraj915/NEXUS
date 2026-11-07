# Third-party notices

## NLTK

- **Used for**: the English Snowball stemmer behind the BM25 analyzer.
- **Location**: `backend/app/domains/search/_vendor/nltk_snowball/`
- **Source**: `nltk/stem/snowball.py`, `nltk/stem/api.py` and `nltk/stem/util.py` from the NLTK 3.10.3 wheel (<https://www.nltk.org/>)
- **Copyright**: Copyright (C) 2001-2026 NLTK Project
- **License**: Apache License 2.0, in `LICENSE.txt` in that folder
- **Modifications**: `snowball.py` keeps only `_LanguageSpecificStemmer`, `_StandardStemmer` and `EnglishStemmer`, and its imports are changed. `api.py` and `util.py` are unchanged apart from a one-line origin comment. The header of `snowball.py` lists the changes.
