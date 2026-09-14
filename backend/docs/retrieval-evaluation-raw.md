# NEXUS Retrieval Evaluation - Raw Results

| Metric | BM25 | Dense |
|---|---|---|
| NDCG@10 | 0.6736 | 0.6642 |
| Recall@100 | 0.9064 | 0.9277 |
| Latency | 67.23ms | 125.50ms |

**Jaccard Index (Top 10):** 0.2292

## 2. Failure Analysis
**BM25 Failed / Dense Succeeded:** 18 queries
- `0-dimensional biomaterials show inductive properties.` (Relevant: ['31715818'])
- `FoxO3a activation in neuronal death is mediated by reactive oxygen species (ROS).` (Relevant: ['28937856'])
- `Hematopoietic Stem Cell purification reaches purity rate of up to 50%.` (Relevant: ['13980338'])
- `High dietary calcium intakes are unnecessary for prevention of secondary hyperparathyroidism in subjects with 25(OH)D levels above 75 nmol/liter.` (Relevant: ['16256507'])
- `Hypertension is frequently observed in type 1 diabetes patients.` (Relevant: ['39368721'])
- `ITAM phosphorylation prevents the transfer of the T cell receptor (TCR) signal from the echo-domain to the cytoplasmic tail of the T cell receptor (TCR).` (Relevant: ['33499189'])
- `Mercaptopurine is anabolized into the inactive methylmercaptopurine by thiopurine methyltrasnferase (TPMT).` (Relevant: ['6421792'])
- `Mice without IFN-γ or its receptor are resistant to EAM induced with α-MyHC/CFA.` (Relevant: ['40632104'])
- `Modifying the epigenome in the brain affects the normal human aging process by affecting certain genes related to neurogenesis.` (Relevant: ['22543403'])
- `PPAR-RXRs are inhibited by PPAR ligands.` (Relevant: ['3203590'])

**Dense Failed / BM25 Succeeded:** 21 queries
- `Arterioles have a larger lumen diameter than venules.` (Relevant: ['8290953'])
- `Cellular aging closely links to an older appearance.` (Relevant: ['14079881'])
- `Crossover hot spots are not found within gene promoters in Saccharomyces cerevisiae.` (Relevant: ['10874408'])
- `DMRT1 is a sex-determining gene that is epigenetically regulated by the MHM region.` (Relevant: ['4388470'])
- `De novo assembly of sequence data has more specific contigs than unassembled sequence data.` (Relevant: ['6173523'])
- `Deamination of cytidine to uridine on the minus strand of viral DNA results in catastrophic G-to-A mutations in the viral genome.` (Relevant: ['4347374'])
- `Homozygous deletion of murine Sbds gene from osterix-expressing mesenchymal stem and progenitor cells (MPCs) prevents oxidative stress.` (Relevant: ['3863543'])
- `In transgenic mice harboring green florescent protein under the control of the Sox2 promoter, less than ten percent of the cells with green florescent colocalize with cell proliferation markers.` (Relevant: ['16999023'])
- `Increased vessel density along with a reduction in fibrosis decreases the efficacy of chemotherapy treatments.` (Relevant: ['20888849', '2565138'])
- `Infection of human T-cell lymphotropic virus type 1 is most frequent in individuals of African origin.` (Relevant: ['24512064'])
