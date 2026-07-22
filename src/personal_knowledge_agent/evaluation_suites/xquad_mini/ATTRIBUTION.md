# XQuAD-mini-v1 attribution

XQuAD-mini-v1 is a derivative fixture made from the official XQuAD (Cross-lingual Question Answering Dataset) repository maintained by Google DeepMind.

## Upstream source and integrity

- Repository: https://github.com/google-deepmind/xquad
- Pinned revision: `7d30520c717524000f0d9d2f9c10a069acd9d285`
- English source: https://raw.githubusercontent.com/google-deepmind/xquad/7d30520c717524000f0d9d2f9c10a069acd9d285/xquad.en.json
- English SHA-256: `e4c57d1c9143aaa1c5d265ba5987a65f4e69528d2a98f29d6e75019b10344f29`
- Chinese source: https://raw.githubusercontent.com/google-deepmind/xquad/7d30520c717524000f0d9d2f9c10a069acd9d285/xquad.zh.json
- Chinese SHA-256: `691d0b3359bc6b8faa8de931dfdfe21d50a65861ae348e32a0d1a0190b0c8835`
- Packaged `cases.json` plus 24 documents aggregate SHA-256: `c255777a49cee10455ec20fa58046df9b325ea41f0dbafff146e10b630ec9ccb`

The revision and both SHA-256 values are fixed so the derivative can be reproduced from the exact source bytes.

## Derivative fixture notice

This repository does not redistribute the full XQuAD corpus. It packages the first eligible question from each of the first 12 distinct aligned English/Chinese context pairs encountered in English source order, yielding 24 cases and 24 language-specific context documents. The transformation selects a subset, deduplicates answer aliases in original order, adds a logical-ID Markdown title to each unchanged context, and reshapes question metadata into `cases.json`. No answer annotation is included in a Markdown document.

## Citation

Please cite the original XQuAD work:

Mikel Artetxe, Sebastian Ruder, and Dani Yogatama. “On the Cross-lingual Transferability of Monolingual Representations.” arXiv:1910.11856, 2019. https://arxiv.org/abs/1910.11856

```bibtex
@article{Artetxe:etal:2019,
  author = {Mikel Artetxe and Sebastian Ruder and Dani Yogatama},
  title = {On the Cross-lingual Transferability of Monolingual Representations},
  journal = {CoRR},
  volume = {abs/1910.11856},
  year = {2019},
  archivePrefix = {arXiv},
  eprint = {1910.11856}
}
```

## License

XQuAD is distributed under the Creative Commons Attribution-ShareAlike 4.0 International license (CC BY-SA 4.0): https://creativecommons.org/licenses/by-sa/4.0/legalcode

This derivative fixture is redistributed under the same CC BY-SA 4.0 license. The complete license text accompanies it in `LICENSE.txt`. Attribution belongs to the original XQuAD authors named above; the subset selection and packaging changes are described in this notice.
