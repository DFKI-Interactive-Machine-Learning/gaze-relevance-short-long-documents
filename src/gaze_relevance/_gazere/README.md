# Vendored gazeRE code

Copied from the official gazeRE dataset repository
<https://github.com/DFKI-Interactive-Machine-Learning/gazeRE-dataset> (commit `9af27c6`, GPL-3.0),
so that the 17 eye-tracking features are computed with exactly the code that produced the dataset's features.

Changes, all marked with `PATCH` in the code:
1. package-relative imports (the original uses top-level `data_loading` / `features` packages);
2. `np.float` replaced by `float` (the alias was removed in NumPy 1.24);
3. `gazeRE_DataLoader` keeps its state per instance (the original used class attributes, so a second loader
   accumulated the first loader's data);
4. the feature extractor no longer calls `logging.basicConfig`.

With these patches, the extracted features are identical (max. absolute difference < 1e-12) to the feature tables used
in the paper for all 288 g-REL and 1,504 GoogleNQ samples.
