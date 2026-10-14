# Model Workshop

**[Open the live learning guide](https://rohitjz.github.io/machine-learning-model-lab/)**

Learn machine-learning mechanisms by changing their behavior, rather than
memorizing a universal ranking.

## Workshops

- Fit linear regression and see how outliers affect least squares.
- Separate logistic probabilities from classification thresholds.
- Grow a regression tree and compare training with validation error.
- Average independent bootstrap trees.
- Build a squared-error boosting model one correction at a time.
- Inspect categorical target leakage and prefix-only statistics.
- Rotate a PCA projection and explore the effect of feature scaling.
- Select model candidates from a task and evaluate them fairly.

The guide includes first-principles explanations, knowledge checks, recall
prompts, and official mechanism sources. All data is synthetic. Metrics are
computed in the browser, not taken from performance benchmarks.

## Source and scope

The topic was inspired by [baniascodes' Instagram reel](https://www.instagram.com/reels/DecWDZtMVdQ/).
Its publicly readable caption names linear regression, XGBoost, CatBoost,
and PCA. The video did not play during preparation, so the guide does not
reconstruct or endorse its exact rankings. Other workshops are supporting
concepts added for learning.

No Instagram video, audio, images, or transcript is redistributed.
The explanations, diagrams, datasets, and browser code are original.

The boosting lab demonstrates generic residual boosting for squared error.
The categorical lab demonstrates one simplified ordered target statistic.
Neither runs the XGBoost or CatBoost library. The forest lab isolates
one-feature bootstrap averaging and does not demonstrate feature subsampling.

Sources, checked on 14 October 2026:

- [scikit-learn: linear models](https://scikit-learn.org/stable/modules/linear_model.html)
- [scikit-learn: decision trees](https://scikit-learn.org/stable/modules/tree.html)
- [scikit-learn: random forests](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [XGBoost: introduction to boosted trees](https://xgboost.readthedocs.io/en/stable/tutorials/model.html)
- [CatBoost: categorical transformations](https://catboost.ai/docs/en/concepts/algorithm-main-stages_cat-to-numberic)
- [CatBoost: training stages](https://catboost.ai/docs/en/concepts/algorithm-main-stages)
- [scikit-learn: PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [scikit-learn: cross-validation and leakage](https://scikit-learn.org/stable/modules/cross_validation.html)

## Design

A white instrument-board layout puts the plot before the explanation.
Blue circles identify observations, a crimson line identifies the fitted
model, amber dashed segments identify residuals, and outlined diamonds
identify validation points. A night view is available. Shapes and labels
carry meaning alongside color.

Direction and execution were informed by the `creative-direction`,
`frontend-design`, and `ui-ux-pro-max` skills. The style database did not
return a verified subject-specific aesthetic match; the final composition
is an original direction. Color and typography tokens come from the
`web-artifacts-builder` skill, with a white canvas and semantic chart colors
rather than the previous guide's warm page treatment.

## Run locally

Open [index.html](./index.html) in a browser. It is a self-contained file
without external assets, model calls, or package dependencies. Progress and
the selected lesson are encoded in the URL. No browser-profile storage is
required.

For local HTTP testing:

```bash
python3 -m http.server 8765 --bind 127.0.0.1
```

Then open `http://127.0.0.1:8765/`.

## Test

The numerical mechanisms have dependency-free Node.js regression tests:

```bash
node --test tests/model-workshop.test.cjs
```

They verify least-squares fitting, held-out overfitting, deterministic
bootstrap aggregation, loss reduction in squared-error boosting,
exclusion of each row's own label from ordered encoding, and PCA's
variance and reconstruction identities.

## Publishing

GitHub Pages serves the repository root from the `main` branch over HTTPS.
Update [index.html](./index.html) to change the live workshop.
