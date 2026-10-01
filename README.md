# MaizeGuard: lesion-grounded vision-language evidence with few-label conformal calibration

Code for the manuscript *Lesion-Grounded Vision-Language Evidence With Few-Label Conformal Calibration for Open-World Maize Disease Diagnosis and Progression Forecasting Under Field Shift*.

Everything sits in this one folder. The scripts create the working sub-directories they need at run time: `data/`, `models/`, `feats/`, `runs/` and `results/`. Every number in the paper comes from these scripts; none were edited by hand.

## 1. Data

| Source | How to obtain | Place in |
|---|---|---|
| PlantVillage maize (color + segmented) | `git clone --filter=blob:none --no-checkout https://github.com/spMohanty/PlantVillage-Dataset`, then sparse-checkout `raw/color/Corn_(maize)*` and `raw/segmented/Corn_(maize)*` | `data/PlantVillage-Dataset` |
| PlantDoc maize | `git clone https://github.com/pratikkayal/PlantDoc-Dataset` (folders `*/Corn*`) | `data/PlantDoc-Dataset` |
| CD&S | OSF project s6ru5: `Dataset_Original`, `Dataset_Severity`, `Dataset_Annotated` (images resized to a 640-px longest side) | `data/cds/cds/{orig,sev,ann}` |
| Weather | `champion_climate.txt` and `hyderabad_climate.txt` from the `aquacrop` PyPI wheel (`aquacrop/data/`) | `weather/` |
| CLIP ViT-B/32 | `onnx/vision_model_fp16.onnx`, `onnx/text_model_fp16.onnx` and `tokenizer.json` from `Xenova/clip-vit-base-patch32`, renamed to `mg_vision_fp16.onnx`, `mg_text_fp16.onnx` and `mg_tokenizer.json` | `models/` |

`index_pv_pd.csv` and `index_cds.csv` record the exact images used. Perceptual-hash group IDs are included so that splits never separate near-duplicate leaves.

## 2. Pipeline

```
python prep_index.py && python prep_cds.py       # indices, near-duplicate groups
python pseudo_lesion.py                            # colour-derived token targets (PlantVillage)
python splits.py                                   # fixed test split + per-seed partitions
python extract.py list_pvpd.txt feats/pvpd_clean.npz
python extract_jobs.py nontest_aug test_blur_1 ... # degraded copies (see corrupt.py)
python extract_jobs.py list:list_cds_orig_sev.txt:feats/cds_clean.npz
for s in 0 1 2 3 4; do python run_main.py $s mg,zs,zsdesc,lp,clipad,tipf,abmil; done
for s in 0 1 2; do python run_main.py $s mg_noles,mg_nodesc,mg_noqual,mg_noaug,mg_nocons,mg_notok,mg_adapter; done
python predict_ext.py feats/cds_clean.npz cds all  # and every feats/test_*.npz, feats/cdst_*.npz
python analyze.py 0,1,2,3,4 zs,zsdesc,lp,clipad,tipf,abmil,mg
python analyze.py 0,1,2 mg_noles,mg_nodesc,mg_noqual,mg_noaug,mg_nocons,mg_notok,mg_adapter
python localize.py; python sev_baselines.py; python naming.py; python eval_cdst.py; python stats.py; python efficiency.py
for s in 0 1 2; do python run_temporal.py $s; done
python make_numbers.py
python fig_arch.py fig_arch.png; python plots.py robust ood conformal; python plots2.py severity temporal maps
```

## 3. File map

| Group | Files |
|---|---|
| Encoder & features | `clipfeat.py` (ONNX CLIP, value-path dense tokens), `extract.py`, `extract_jobs.py`, `corrupt.py` |
| Data | `prep_index.py`, `prep_cds.py`, `pseudo_lesion.py`, `splits.py`, `featstore.py`, `prompts.py` |
| Models | `mgmodel.py` (MaizeGuard and baselines), `run_main.py` (training), `predict_ext.py` |
| Decision | `conformal2.py` (dual label + lesion-evidence test), `analysis_core.py`, `metrics.py` |
| Evaluation | `analyze.py`, `localize.py`, `sev_baselines.py`, `naming.py`, `eval_cdst.py`, `stats.py`, `efficiency.py` |
| Progression | `progression.py` (weather-driven benchmark), `temporal.py` (filter, GRU), `run_temporal.py` |
| Design selection | `dev_*.py`: the decisions taken on the CD&S *training* folder only (no adapter, PlantDoc-validation selection, lesion-embedding evidence, filter training length) |
| Figures | `fig_arch.py` (with automatic overlap and arrow-routing checks), `plots.py`, `plots2.py` |
| Results | `results_*.json` copies of the result files used in the manuscript |

## 4. Environment

The code was tested with Python 3.11 on a 2-core CPU. Package versions are in `requirements.txt`. A full run takes roughly 8–10 CPU hours, and most of that time is feature extraction.

## 5. Notes

- The progression benchmark is a mechanistic simulation. The weather and the observed images are real, but the lesion trajectories are simulated.
- Model selection never uses CD&S test images.
