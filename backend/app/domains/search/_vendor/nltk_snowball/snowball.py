# Vendored from NLTK 3.10.3 (Apache License 2.0; see LICENSE.txt in this folder).
# Source: nltk/stem/snowball.py in the nltk-3.10.3 wheel pinned by uv.lock.
#
# Changes from the original:
#   1. Imports only: `nltk.stem.api` and `nltk.stem.util` now point at the copies in this
#      folder, and `nltk.corpus.stopwords` and `nltk.stem.porter` are no longer imported.
#   2. Only _LanguageSpecificStemmer, _StandardStemmer and EnglishStemmer are kept. Every
#      other class and function in the module is omitted.
# The kept classes are otherwise byte-for-byte unchanged, and are not reformatted.
#
# Consequence of change 1: ignore_stopwords=True raises NameError. NEXUS never passes it.
# Do not edit by hand: gate G1 (scripts.benchmark parity-analyzer) proves this copy matches
# NLTK over SciFact, and tests/search/test_vendored_stemmer.py pins the sensitive stems.

#
# Natural Language Toolkit: Snowball Stemmer
#
# Copyright (C) 2001-2026 NLTK Project
# Author: Peter Michael Stahl <pemistahl@gmail.com>
#         Peter Ljunglof <peter.ljunglof@heatherleaf.se> (revisions)
#         Lakhdar Benzahia <lakhdar.benzahia@gmail.com>  (co-writer)
#         Assem Chelli <assem.ch@gmail.com>  (reviewer arabicstemmer)
#         Abdelkrim Aries <ab_aries@esi.dz> (reviewer arabicstemmer)
# Algorithms: Dr Martin Porter <martin@tartarus.org>
#             Assem Chelli <assem.ch@gmail.com>  arabic stemming algorithm
#             Benzahia Lakhdar <lakhdar.benzahia@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

import re

from app.domains.search._vendor.nltk_snowball.api import StemmerI
from app.domains.search._vendor.nltk_snowball.util import suffix_replace


class _LanguageSpecificStemmer(StemmerI):
    """
    This helper subclass offers the possibility
    to invoke a specific stemmer directly.
    This is useful if you already know the language to be stemmed at runtime.

    Create an instance of the Snowball stemmer.

    :param ignore_stopwords: If set to True, stopwords are
                             not stemmed and returned unchanged.
                             Set to False by default.
    :type ignore_stopwords: bool
    """

    def __init__(self, ignore_stopwords=False):
        # The language is the name of the class, minus the final "Stemmer".
        language = type(self).__name__.lower()
        if language.endswith("stemmer"):
            language = language[:-7]

        self.stopwords = set()
        if ignore_stopwords:
            try:
                for word in stopwords.words(language):
                    self.stopwords.add(word)
            except OSError as e:
                raise ValueError(
                    "{!r} has no list of stopwords. Please set"
                    " 'ignore_stopwords' to 'False'.".format(self)
                ) from e

    def __repr__(self):
        """
        Print out the string representation of the respective class.

        """
        return f"<{type(self).__name__}>"


class _StandardStemmer(_LanguageSpecificStemmer):
    """
    This subclass encapsulates two methods for defining the standard versions
    of the string regions R1, R2, and RV.

    """

    def _r1r2_standard(self, word, vowels):
        """
        Return the standard interpretations of the string regions R1 and R2.

        R1 is the region after the first non-vowel following a vowel,
        or is the null region at the end of the word if there is no
        such non-vowel.

        R2 is the region after the first non-vowel following a vowel
        in R1, or is the null region at the end of the word if there
        is no such non-vowel.

        :param word: The word whose regions R1 and R2 are determined.
        :type word: str or unicode
        :param vowels: The vowels of the respective language that are
                       used to determine the regions R1 and R2.
        :type vowels: unicode
        :return: (r1,r2), the regions R1 and R2 for the respective word.
        :rtype: tuple
        :note: This helper method is invoked by the respective stem method of
               the subclasses DutchStemmer, FinnishStemmer,
               FrenchStemmer, GermanStemmer, ItalianStemmer,
               PortugueseStemmer, RomanianStemmer, and SpanishStemmer.
               It is not to be invoked directly!
        :note: A detailed description of how to define R1 and R2
               can be found at http://snowball.tartarus.org/texts/r1r2.html

        """
        r1 = ""
        r2 = ""
        for i in range(1, len(word)):
            if word[i] not in vowels and word[i - 1] in vowels:
                r1 = word[i + 1 :]
                break

        for i in range(1, len(r1)):
            if r1[i] not in vowels and r1[i - 1] in vowels:
                r2 = r1[i + 1 :]
                break

        return (r1, r2)

    def _rv_standard(self, word, vowels):
        """
        Return the standard interpretation of the string region RV.

        If the second letter is a consonant, RV is the region after the
        next following vowel. If the first two letters are vowels, RV is
        the region after the next following consonant. Otherwise, RV is
        the region after the third letter.

        :param word: The word whose region RV is determined.
        :type word: str or unicode
        :param vowels: The vowels of the respective language that are
                       used to determine the region RV.
        :type vowels: unicode
        :return: the region RV for the respective word.
        :rtype: unicode
        :note: This helper method is invoked by the respective stem method of
               the subclasses ItalianStemmer, PortugueseStemmer,
               RomanianStemmer, and SpanishStemmer. It is not to be
               invoked directly!

        """
        rv = ""
        if len(word) >= 2:
            if word[1] not in vowels:
                for i in range(2, len(word)):
                    if word[i] in vowels:
                        rv = word[i + 1 :]
                        break

            elif word[0] in vowels and word[1] in vowels:
                for i in range(2, len(word)):
                    if word[i] not in vowels:
                        rv = word[i + 1 :]
                        break
            else:
                rv = word[3:]

        return rv


class EnglishStemmer(_StandardStemmer):
    """
    The English Snowball stemmer.

    :cvar __vowels: The English vowels.
    :type __vowels: unicode
    :cvar __double_consonants: The English double consonants.
    :type __double_consonants: tuple
    :cvar __li_ending: Letters that may directly appear before a word final 'li'.
    :type __li_ending: unicode
    :cvar __step0_suffixes: Suffixes to be deleted in step 0 of the algorithm.
    :type __step0_suffixes: tuple
    :cvar __step1a_suffixes: Suffixes to be deleted in step 1a of the algorithm.
    :type __step1a_suffixes: tuple
    :cvar __step1b_suffixes: Suffixes to be deleted in step 1b of the algorithm.
    :type __step1b_suffixes: tuple
    :cvar __step2_suffixes: Suffixes to be deleted in step 2 of the algorithm.
    :type __step2_suffixes: tuple
    :cvar __step3_suffixes: Suffixes to be deleted in step 3 of the algorithm.
    :type __step3_suffixes: tuple
    :cvar __step4_suffixes: Suffixes to be deleted in step 4 of the algorithm.
    :type __step4_suffixes: tuple
    :cvar __step5_suffixes: Suffixes to be deleted in step 5 of the algorithm.
    :type __step5_suffixes: tuple
    :cvar __special_words: A dictionary containing words
                           which have to be stemmed specially.
    :type __special_words: dict
    :note: A detailed description of the English
           stemming algorithm can be found under
           http://snowball.tartarus.org/algorithms/english/stemmer.html
    """

    __vowels = "aeiouy"
    __double_consonants = ("bb", "dd", "ff", "gg", "mm", "nn", "pp", "rr", "tt")
    __li_ending = "cdeghkmnrt"
    __step0_suffixes = ("'s'", "'s", "'")
    __step1a_suffixes = ("sses", "ied", "ies", "us", "ss", "s")
    __step1b_suffixes = ("eedly", "ingly", "edly", "eed", "ing", "ed")
    __step2_suffixes = (
        "ization",
        "ational",
        "fulness",
        "ousness",
        "iveness",
        "tional",
        "biliti",
        "lessli",
        "entli",
        "ation",
        "alism",
        "aliti",
        "ousli",
        "iviti",
        "fulli",
        "enci",
        "anci",
        "abli",
        "izer",
        "ator",
        "alli",
        "bli",
        "ogi",
        "li",
    )
    __step3_suffixes = (
        "ational",
        "tional",
        "alize",
        "icate",
        "iciti",
        "ative",
        "ical",
        "ness",
        "ful",
    )
    __step4_suffixes = (
        "ement",
        "ance",
        "ence",
        "able",
        "ible",
        "ment",
        "ant",
        "ent",
        "ism",
        "ate",
        "iti",
        "ous",
        "ive",
        "ize",
        "ion",
        "al",
        "er",
        "ic",
    )
    __step5_suffixes = ("e", "l")
    __special_words = {
        "skis": "ski",
        "skies": "sky",
        "dying": "die",
        "lying": "lie",
        "tying": "tie",
        "idly": "idl",
        "gently": "gentl",
        "ugly": "ugli",
        "early": "earli",
        "only": "onli",
        "singly": "singl",
        "sky": "sky",
        "news": "news",
        "howe": "howe",
        "atlas": "atlas",
        "cosmos": "cosmos",
        "bias": "bias",
        "andes": "andes",
        "inning": "inning",
        "innings": "inning",
        "outing": "outing",
        "outings": "outing",
        "canning": "canning",
        "cannings": "canning",
        "herring": "herring",
        "herrings": "herring",
        "earring": "earring",
        "earrings": "earring",
        "proceed": "proceed",
        "proceeds": "proceed",
        "proceeded": "proceed",
        "proceeding": "proceed",
        "exceed": "exceed",
        "exceeds": "exceed",
        "exceeded": "exceed",
        "exceeding": "exceed",
        "succeed": "succeed",
        "succeeds": "succeed",
        "succeeded": "succeed",
        "succeeding": "succeed",
    }

    def stem(self, word):
        """
        Stem an English word and return the stemmed form.

        :param word: The word that is stemmed.
        :type word: str or unicode
        :return: The stemmed form.
        :rtype: unicode

        """
        word = word.lower()

        if word in self.stopwords or len(word) <= 2:
            return word

        elif word in self.__special_words:
            return self.__special_words[word]

        # Map the different apostrophe characters to a single consistent one
        word = (
            word.replace("\u2019", "\x27")
            .replace("\u2018", "\x27")
            .replace("\u201b", "\x27")
        )

        if word.startswith("\x27"):
            word = word[1:]

        if word.startswith("y"):
            word = "".join(("Y", word[1:]))

        for i in range(1, len(word)):
            if word[i - 1] in self.__vowels and word[i] == "y":
                word = "".join((word[:i], "Y", word[i + 1 :]))

        step1a_vowel_found = False
        step1b_vowel_found = False

        r1 = ""
        r2 = ""

        if word.startswith(("gener", "commun", "arsen")):
            if word.startswith(("gener", "arsen")):
                r1 = word[5:]
            else:
                r1 = word[6:]

            for i in range(1, len(r1)):
                if r1[i] not in self.__vowels and r1[i - 1] in self.__vowels:
                    r2 = r1[i + 1 :]
                    break
        else:
            r1, r2 = self._r1r2_standard(word, self.__vowels)

        # STEP 0
        for suffix in self.__step0_suffixes:
            if word.endswith(suffix):
                word = word[: -len(suffix)]
                r1 = r1[: -len(suffix)]
                r2 = r2[: -len(suffix)]
                break

        # STEP 1a
        for suffix in self.__step1a_suffixes:
            if word.endswith(suffix):
                if suffix == "sses":
                    word = word[:-2]
                    r1 = r1[:-2]
                    r2 = r2[:-2]

                elif suffix in ("ied", "ies"):
                    if len(word[: -len(suffix)]) > 1:
                        word = word[:-2]
                        r1 = r1[:-2]
                        r2 = r2[:-2]
                    else:
                        word = word[:-1]
                        r1 = r1[:-1]
                        r2 = r2[:-1]

                elif suffix == "s":
                    for letter in word[:-2]:
                        if letter in self.__vowels:
                            step1a_vowel_found = True
                            break

                    if step1a_vowel_found:
                        word = word[:-1]
                        r1 = r1[:-1]
                        r2 = r2[:-1]
                break

        # STEP 1b
        for suffix in self.__step1b_suffixes:
            if word.endswith(suffix):
                if suffix in ("eed", "eedly"):
                    if r1.endswith(suffix):
                        word = suffix_replace(word, suffix, "ee")

                        if len(r1) >= len(suffix):
                            r1 = suffix_replace(r1, suffix, "ee")
                        else:
                            r1 = ""

                        if len(r2) >= len(suffix):
                            r2 = suffix_replace(r2, suffix, "ee")
                        else:
                            r2 = ""
                else:
                    for letter in word[: -len(suffix)]:
                        if letter in self.__vowels:
                            step1b_vowel_found = True
                            break

                    if step1b_vowel_found:
                        word = word[: -len(suffix)]
                        r1 = r1[: -len(suffix)]
                        r2 = r2[: -len(suffix)]

                        if word.endswith(("at", "bl", "iz")):
                            word = "".join((word, "e"))
                            r1 = "".join((r1, "e"))

                            if len(word) > 5 or len(r1) >= 3:
                                r2 = "".join((r2, "e"))

                        elif word.endswith(self.__double_consonants):
                            word = word[:-1]
                            r1 = r1[:-1]
                            r2 = r2[:-1]

                        elif (
                            r1 == ""
                            and len(word) >= 3
                            and word[-1] not in self.__vowels
                            and word[-1] not in "wxY"
                            and word[-2] in self.__vowels
                            and word[-3] not in self.__vowels
                        ) or (
                            r1 == ""
                            and len(word) == 2
                            and word[0] in self.__vowels
                            and word[1] not in self.__vowels
                        ):
                            word = "".join((word, "e"))

                            if len(r1) > 0:
                                r1 = "".join((r1, "e"))

                            if len(r2) > 0:
                                r2 = "".join((r2, "e"))
                break

        # STEP 1c
        if len(word) > 2 and word[-1] in "yY" and word[-2] not in self.__vowels:
            word = "".join((word[:-1], "i"))
            if len(r1) >= 1:
                r1 = "".join((r1[:-1], "i"))
            else:
                r1 = ""

            if len(r2) >= 1:
                r2 = "".join((r2[:-1], "i"))
            else:
                r2 = ""

        # STEP 2
        for suffix in self.__step2_suffixes:
            if word.endswith(suffix):
                if r1.endswith(suffix):
                    if suffix == "tional":
                        word = word[:-2]
                        r1 = r1[:-2]
                        r2 = r2[:-2]

                    elif suffix in ("enci", "anci", "abli"):
                        word = "".join((word[:-1], "e"))

                        if len(r1) >= 1:
                            r1 = "".join((r1[:-1], "e"))
                        else:
                            r1 = ""

                        if len(r2) >= 1:
                            r2 = "".join((r2[:-1], "e"))
                        else:
                            r2 = ""

                    elif suffix == "entli":
                        word = word[:-2]
                        r1 = r1[:-2]
                        r2 = r2[:-2]

                    elif suffix in ("izer", "ization"):
                        word = suffix_replace(word, suffix, "ize")

                        if len(r1) >= len(suffix):
                            r1 = suffix_replace(r1, suffix, "ize")
                        else:
                            r1 = ""

                        if len(r2) >= len(suffix):
                            r2 = suffix_replace(r2, suffix, "ize")
                        else:
                            r2 = ""

                    elif suffix in ("ational", "ation", "ator"):
                        word = suffix_replace(word, suffix, "ate")

                        if len(r1) >= len(suffix):
                            r1 = suffix_replace(r1, suffix, "ate")
                        else:
                            r1 = ""

                        if len(r2) >= len(suffix):
                            r2 = suffix_replace(r2, suffix, "ate")
                        else:
                            r2 = "e"

                    elif suffix in ("alism", "aliti", "alli"):
                        word = suffix_replace(word, suffix, "al")

                        if len(r1) >= len(suffix):
                            r1 = suffix_replace(r1, suffix, "al")
                        else:
                            r1 = ""

                        if len(r2) >= len(suffix):
                            r2 = suffix_replace(r2, suffix, "al")
                        else:
                            r2 = ""

                    elif suffix == "fulness":
                        word = word[:-4]
                        r1 = r1[:-4]
                        r2 = r2[:-4]

                    elif suffix in ("ousli", "ousness"):
                        word = suffix_replace(word, suffix, "ous")

                        if len(r1) >= len(suffix):
                            r1 = suffix_replace(r1, suffix, "ous")
                        else:
                            r1 = ""

                        if len(r2) >= len(suffix):
                            r2 = suffix_replace(r2, suffix, "ous")
                        else:
                            r2 = ""

                    elif suffix in ("iveness", "iviti"):
                        word = suffix_replace(word, suffix, "ive")

                        if len(r1) >= len(suffix):
                            r1 = suffix_replace(r1, suffix, "ive")
                        else:
                            r1 = ""

                        if len(r2) >= len(suffix):
                            r2 = suffix_replace(r2, suffix, "ive")
                        else:
                            r2 = "e"

                    elif suffix in ("biliti", "bli"):
                        word = suffix_replace(word, suffix, "ble")

                        if len(r1) >= len(suffix):
                            r1 = suffix_replace(r1, suffix, "ble")
                        else:
                            r1 = ""

                        if len(r2) >= len(suffix):
                            r2 = suffix_replace(r2, suffix, "ble")
                        else:
                            r2 = ""

                    elif suffix == "ogi" and word[-4] == "l":
                        word = word[:-1]
                        r1 = r1[:-1]
                        r2 = r2[:-1]

                    elif suffix in ("fulli", "lessli"):
                        word = word[:-2]
                        r1 = r1[:-2]
                        r2 = r2[:-2]

                    elif suffix == "li" and word[-3] in self.__li_ending:
                        word = word[:-2]
                        r1 = r1[:-2]
                        r2 = r2[:-2]
                break

        # STEP 3
        for suffix in self.__step3_suffixes:
            if word.endswith(suffix):
                if r1.endswith(suffix):
                    if suffix == "tional":
                        word = word[:-2]
                        r1 = r1[:-2]
                        r2 = r2[:-2]

                    elif suffix == "ational":
                        word = suffix_replace(word, suffix, "ate")

                        if len(r1) >= len(suffix):
                            r1 = suffix_replace(r1, suffix, "ate")
                        else:
                            r1 = ""

                        if len(r2) >= len(suffix):
                            r2 = suffix_replace(r2, suffix, "ate")
                        else:
                            r2 = ""

                    elif suffix == "alize":
                        word = word[:-3]
                        r1 = r1[:-3]
                        r2 = r2[:-3]

                    elif suffix in ("icate", "iciti", "ical"):
                        word = suffix_replace(word, suffix, "ic")

                        if len(r1) >= len(suffix):
                            r1 = suffix_replace(r1, suffix, "ic")
                        else:
                            r1 = ""

                        if len(r2) >= len(suffix):
                            r2 = suffix_replace(r2, suffix, "ic")
                        else:
                            r2 = ""

                    elif suffix in ("ful", "ness"):
                        word = word[: -len(suffix)]
                        r1 = r1[: -len(suffix)]
                        r2 = r2[: -len(suffix)]

                    elif suffix == "ative" and r2.endswith(suffix):
                        word = word[:-5]
                        r1 = r1[:-5]
                        r2 = r2[:-5]
                break

        # STEP 4
        for suffix in self.__step4_suffixes:
            if word.endswith(suffix):
                if r2.endswith(suffix):
                    if suffix == "ion":
                        if word[-4] in "st":
                            word = word[:-3]
                            r1 = r1[:-3]
                            r2 = r2[:-3]
                    else:
                        word = word[: -len(suffix)]
                        r1 = r1[: -len(suffix)]
                        r2 = r2[: -len(suffix)]
                break

        # STEP 5
        if r2.endswith("l") and word[-2] == "l":
            word = word[:-1]
        elif r2.endswith("e"):
            word = word[:-1]
        elif r1.endswith("e"):
            if len(word) >= 4 and (
                word[-2] in self.__vowels
                or word[-2] in "wxY"
                or word[-3] not in self.__vowels
                or word[-4] in self.__vowels
            ):
                word = word[:-1]

        word = word.replace("Y", "y")

        return word
