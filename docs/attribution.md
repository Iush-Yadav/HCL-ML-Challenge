# Dependencies and development tools

The implementation uses NumPy for arrays and FFTs, SciPy for resampling and softmax, and scikit-learn for text hashing and logistic regression. FFmpeg is used only to prepare media. Dataset annotations and media come from [MELD](https://github.com/declare-lab/MELD); they are not included here.

AI assistance was used to generate the initial implementation, tests, response templates, and documentation, and to reorganize the repository. The checks actually run are recorded in [validation.md](validation.md). MELD training and held-out evaluation have not been run. No manual authorship, independent review, or unperformed experiment is claimed.

Before submission, the project owner should review the implementation, run the remaining experiments, and be able to explain and modify the features, fusion rule, uncertainty thresholds, and response policy.
