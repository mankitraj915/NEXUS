# Analyzer parity (gate G1): SCIFACT

Compares `nltk` SnowballStemmer("english") with `snowballstemmer` over the
analyzer's tokens, stopwords kept (a superset of the stopword-filtered stream).

- **Result**: FAIL
- **Texts**: 6292 (5183 corpus title+abstract, 1109 queries)
- **Tokens**: 1143237
- **mismatched_texts**: 1050
- **Measured**: 2026-11-07 08:27 UTC at commit `7c85aac`
- **Hardware**: Intel(R) Core(TM) i5-10200H CPU @ 2.40GHz, 8 logical CPUs, Windows-11-10.0.26300-SP0
- **Versions**: Python 3.14.2, nltk 3.10.3, snowballstemmer 3.1.1

## Reproduce

`uv sync --group ml`, then `uv run python -m scripts.benchmark parity-analyzer --dataset scifact`.

## Differing stems (at most 50)

| Token | NLTK | snowballstemmer |
|---|---|---|
| internal | intern | internal |
| organization | organ | organiz |
| adding | ad | add |
| emergence | emerg | emergenc |
| interfering | interf | interfer |
| organism | organ | organism |
| oncologists | oncologist | oncolog |
| organic | organ | organic |
| organisms | organ | organism |
| universal | univers | universal |
| gynecologists | gynecologist | gynecolog |
| cardiologists | cardiologist | cardiolog |
| interval | interv | interval |
| international | intern | internat |
| realization | realize | realiz |
| added | ad | add |
| emergent | emerg | emergent |
| ionization | ionize | ioniz |
| university | univers | universiti |
| universities | univers | universiti |
| internalized | intern | internal |
| organize | organ | organiz |
| universally | univers | universal |
| organized | organ | organiz |
| emergency | emerg | emergenc |
| organizer | organ | organiz |
| lateral | later | lateral |
| biologists | biologist | biolog |
| intervals | interv | interval |
| evening | even | evening |
| internalization | intern | internal |
| oncologist | oncologist | oncolog |
| laterally | later | lateral |
| gastroenterologist | gastroenterologist | gastroenterolog |
| organizing | organ | organiz |
| epidemiologists | epidemiologist | epidemiolog |
| organizers | organ | organiz |
| hematologist | hematologist | hematolog |
| organism's | organ | organism |
| radiologists | radiologist | radiolog |
| hematologists | hematologist | hematolog |
| neurologists | neurologist | neurolog |
| interfered | interf | interfer |
| pathologists | pathologist | patholog |
| psychologists | psychologist | psycholog |
| biologist | biologist | biolog |
| urologists | urologist | urolog |
| emergencies | emerg | emergenc |
| organizations | organ | organiz |
| analgesiologists | analgesiologist | analgesiolog |

The table above is capped at 50. All differing tokens are listed below.

## Scope of the failure (diagnostic, measured at the same commit)

| Stream | Texts | Mismatched texts | Distinct tokens | Differing distinct | Token occurrences | Differing occurrences |
|---|---:|---:|---:|---:|---:|---:|
| Corpus, stopwords removed | 5183 | 1041 | 47757 | 83 | 753513 | 1630 |
| Queries, stopwords removed | 1109 | 9 | 3328 | 5 | 9903 | 9 |
| All, stopwords kept (gate stream) | 6292 | 1050 | 48194 | 83 | 1143237 | 1639 |

## All 83 differing tokens, by occurrences

| Token | Occurrences | NLTK 3.10.3 | snowballstemmer 3.1.1 |
|---|---:|---|---|
| interval | 366 | interv | interval |
| organization | 179 | organ | organiz |
| organisms | 128 | organ | organism |
| international | 93 | intern | internat |
| intervals | 70 | interv | interval |
| internal | 66 | intern | internal |
| lateral | 60 | later | lateral |
| emergency | 55 | emerg | emergenc |
| added | 54 | ad | add |
| university | 54 | univers | universiti |
| interfering | 52 | interf | interfer |
| emergence | 41 | emerg | emergenc |
| organism | 41 | organ | organism |
| universal | 40 | univers | universal |
| internalization | 25 | intern | internal |
| organized | 25 | organ | organiz |
| oncologists | 24 | oncologist | oncolog |
| adding | 21 | ad | add |
| organizing | 17 | organ | organiz |
| organizer | 15 | organ | organiz |
| internalized | 14 | intern | internal |
| organizations | 14 | organ | organiz |
| pathologists | 13 | pathologist | patholog |
| universally | 13 | univers | universal |
| biologists | 12 | biologist | biolog |
| organize | 12 | organ | organiz |
| pulmonologist | 11 | pulmonologist | pulmonolog |
| emergent | 8 | emerg | emergent |
| organic | 7 | organ | organic |
| evening | 6 | even | evening |
| interfered | 6 | interf | interfer |
| psychologists | 5 | psychologist | psycholog |
| gynecologists | 4 | gynecologist | gynecolog |
| ionization | 4 | ionize | ioniz |
| organism's | 4 | organ | organism |
| radiologists | 4 | radiologist | radiolog |
| emergencies | 3 | emerg | emergenc |
| hematologists | 3 | hematologist | hematolog |
| laterally | 3 | later | lateral |
| neonatologists | 3 | neonatologist | neonatolog |
| organization's | 3 | organ | organiz |
| universities | 3 | univers | universiti |
| urologists | 3 | urologist | urolog |
| biologist | 2 | biologist | biolog |
| cardiologists | 2 | cardiologist | cardiolog |
| lateralized | 2 | later | lateral |
| lateralizing | 2 | later | lateral |
| neonatologist | 2 | neonatologist | neonatolog |
| nephrologists | 2 | nephrologist | nephrolog |
| neurologist | 2 | neurologist | neurolog |
| neurologists | 2 | neurologist | neurolog |
| oncologist | 2 | oncologist | oncolog |
| organizers | 2 | organ | organiz |
| organizes | 2 | organ | organiz |
| pathologist's | 2 | pathologist | patholog |
| pulmonologists | 2 | pulmonologist | pulmonolog |
| realization | 2 | realize | realiz |
| transnationalism | 2 | transnate | transnat |
| analgesiologists | 1 | analgesiologist | analgesiolog |
| anthropologists | 1 | anthropologist | anthropolog |
| embryologists | 1 | embryologist | embryolog |
| epidemiologists | 1 | epidemiologist | epidemiolog |
| ethnopharmacologists | 1 | ethnopharmacologist | ethnopharmacolog |
| gastroenterologist | 1 | gastroenterologist | gastroenterolog |
| gastroenterologists | 1 | gastroenterologist | gastroenterolog |
| gynaecologist | 1 | gynaecologist | gynaecolog |
| hematologist | 1 | hematologist | hematolog |
| immunologists | 1 | immunologist | immunolog |
| internalize | 1 | intern | internal |
| internalizing | 1 | intern | internal |
| internationally | 1 | intern | internat |
| laterality | 1 | later | lateral |
| microbiologists | 1 | microbiologist | microbiolog |
| mutationally | 1 | mutate | mutat |
| nephrologist | 1 | nephrologist | nephrolog |
| neuropathologists | 1 | neuropathologist | neuropatholog |
| neurophysiologists | 1 | neurophysiologist | neurophysiolog |
| neuroradiologist | 1 | neuroradiologist | neuroradiolog |
| ophthalmologists | 1 | ophthalmologist | ophthalmolog |
| pathologist | 1 | pathologist | patholog |
| psychologist | 1 | psychologist | psycholog |
| radiologist | 1 | radiologist | radiolog |
| translationally | 1 | translate | translat |

## Diagnostics for the decision (not gate results; no code was changed)

- **snowballstemmer 2.2.0**, same command, ephemeral overlay: also FAIL, with 9 mismatched
  texts and 5 differing tokens: realization (realize / realiz), ionization (ionize / ioniz),
  mutationally (mutate / mutat), translationally (translate / translat), transnationalism
  (transnate / transnat).
- **Import footprint**, Windows, Python 3.14.2, three runs each, not the Linux container:
  `import app.main` is 187 MB RSS and about 3 s with NLTK (which loads scipy and sklearn),
  and 89 to 90 MB and 1.1 to 1.4 s with the NLTK import stubbed out.
- **Retrieval impact of any stemmer change**: unmeasured.
