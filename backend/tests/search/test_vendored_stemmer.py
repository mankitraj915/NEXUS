"""Pins the stems NLTK 3.10.3 produced for the tokens where other Snowball ports differ.

These 83 tokens are every SciFact token on which snowballstemmer 3.1.1 disagreed with NLTK
(docs/parity-analyzer-rejected-candidates.md). The expected values are NLTK's, written out
so this test needs no NLTK install and cannot follow a later change to it.
"""

import pytest

from app.domains.search._vendor.nltk_snowball.snowball import EnglishStemmer
from app.domains.search.analyzer import analyze_text

NLTK_STEMS = {
    "interval": "interv",
    "organization": "organ",
    "organisms": "organ",
    "international": "intern",
    "intervals": "interv",
    "internal": "intern",
    "lateral": "later",
    "emergency": "emerg",
    "added": "ad",
    "university": "univers",
    "interfering": "interf",
    "emergence": "emerg",
    "organism": "organ",
    "universal": "univers",
    "internalization": "intern",
    "organized": "organ",
    "oncologists": "oncologist",
    "adding": "ad",
    "organizing": "organ",
    "organizer": "organ",
    "internalized": "intern",
    "organizations": "organ",
    "pathologists": "pathologist",
    "universally": "univers",
    "biologists": "biologist",
    "organize": "organ",
    "pulmonologist": "pulmonologist",
    "emergent": "emerg",
    "organic": "organ",
    "evening": "even",
    "interfered": "interf",
    "psychologists": "psychologist",
    "gynecologists": "gynecologist",
    "ionization": "ionize",
    "organism's": "organ",
    "radiologists": "radiologist",
    "emergencies": "emerg",
    "hematologists": "hematologist",
    "laterally": "later",
    "neonatologists": "neonatologist",
    "organization's": "organ",
    "universities": "univers",
    "urologists": "urologist",
    "biologist": "biologist",
    "cardiologists": "cardiologist",
    "lateralized": "later",
    "lateralizing": "later",
    "neonatologist": "neonatologist",
    "nephrologists": "nephrologist",
    "neurologist": "neurologist",
    "neurologists": "neurologist",
    "oncologist": "oncologist",
    "organizers": "organ",
    "organizes": "organ",
    "pathologist's": "pathologist",
    "pulmonologists": "pulmonologist",
    "realization": "realize",
    "transnationalism": "transnate",
    "analgesiologists": "analgesiologist",
    "anthropologists": "anthropologist",
    "embryologists": "embryologist",
    "epidemiologists": "epidemiologist",
    "ethnopharmacologists": "ethnopharmacologist",
    "gastroenterologist": "gastroenterologist",
    "gastroenterologists": "gastroenterologist",
    "gynaecologist": "gynaecologist",
    "hematologist": "hematologist",
    "immunologists": "immunologist",
    "internalize": "intern",
    "internalizing": "intern",
    "internationally": "intern",
    "laterality": "later",
    "microbiologists": "microbiologist",
    "mutationally": "mutate",
    "nephrologist": "nephrologist",
    "neuropathologists": "neuropathologist",
    "neurophysiologists": "neurophysiologist",
    "neuroradiologist": "neuroradiologist",
    "ophthalmologists": "ophthalmologist",
    "pathologist": "pathologist",
    "psychologist": "psychologist",
    "radiologist": "radiologist",
    "translationally": "translate",
}


def test_all_recorded_tokens_are_pinned():
    assert len(NLTK_STEMS) == 83


@pytest.mark.parametrize(("token", "stem"), NLTK_STEMS.items())
def test_vendored_stemmer_keeps_the_nltk_stem(token, stem):
    assert EnglishStemmer().stem(token) == stem


@pytest.mark.parametrize(("token", "stem"), NLTK_STEMS.items())
def test_analyzer_keeps_the_nltk_stem(token, stem):
    assert analyze_text(token, remove_stopwords=False) == [stem]
