<h1 align="center">CorrFlow: Self-Correcting Flow Matching</h1>

<p align="center">
  <a href="https://fogradio.github.io/">Hangwei Zhang</a><sup>1</sup> &nbsp;
  <a href="https://www.linkedin.com/in/nathanrouille/">Nathan Rouillé</a><sup>1,2</sup> &nbsp;
  <a href="https://physicalvision.github.io/people/~chuanxia">Chuanxia Zheng</a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup> PVG, Nanyang Technological University<br>
  <sup>2</sup> Télécom Paris, Institut Polytechnique de Paris
</p>

<p align="center">
  <a href="https://fogradio.github.io/corrflow-project/"><b>Project Page</b></a> ·
  <a href="https://arxiv.org/abs/2610.24750"><b>Paper</b></a> ·
  <b>Model</b>
</p>

**CorrFlow learns to correct its own predictions for image and video generation.** A lightweight refiner revises the base model's velocity before each sampling update.

<p align="center">
  <a href="https://fogradio.github.io/corrflow-project/#method">
    <img src="assets/teaser-poster.jpg" alt="CorrFlow intuition: a learned residual corrects the base model's imperfect clean prediction." width="900">
  </a>
  <br>
  <em>Predict, reflect, refine. Click to watch the method video.</em>
</p>

## Method

The refiner predicts a velocity residual from the base model's clean prediction, hidden states, conditioning, and noise level. Each correction reuses the same base-model forward pass, adding one refiner evaluation per step. The clean reference shown above is used only during training.

<p align="center">
  <img src="assets/method.svg" alt="CorrFlow architecture: a refiner reuses base-model predictions and hidden states to correct the sampling velocity." width="1000">
</p>

## Image Training and Inference

The image implementation is available in [`image/`](image/README.md), including
joint DiT/refiner training, sampling, and evaluation.
See the [setup and usage guide](image/README.md) for dependencies, data,
and commands.

## Acknowledgements

Our image implementation builds on [RAEv2](https://github.com/nanovisionx/RAEv2).
We thank the RAEv2 authors for sharing their code and research. Its upstream license is
preserved in [`image/LICENSE`](image/LICENSE); the root MIT license does not
replace the terms for upstream-derived code.

## Citation

```bibtex
@misc{zhang2026corrflow,
      title={CorrFlow: Self-Correcting Flow Matching},
      author={Hangwei Zhang and Nathan Rouillé and Chuanxia Zheng},
      year={2026},
      eprint={2610.24750},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.24750},
}
```

## Release Plan

- [x] Release image generation training and inference code.
- [ ] Release image generation checkpoints.
- [ ] Release video generation training and inference code.
- [ ] Release video generation checkpoints.
