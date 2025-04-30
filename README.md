# Visualize robust finetuning

You can run this streamlit by simply cloning the repository, installing numpy, pickle, matplotlib, plotly and streamlit. Then simply run it from the terminal with: streamlit run app.py

You have the option to select different plot types for different Datasets and learning rates. The plot options are:
1. PCA: A 2D PCA representation of the interpolation and the optimization path, giving intuition about how the interpolation between zeroshot and the fine-tuned model differs from the training steps
2. Training: The test accuracy and losses of the checkpoints at every epoch of the training. Between epoch interpolations between the parameters of the checkpoints are also shown.
3. Compare: Shows a comparison between the checkpoints of the fine-tuning and the interpolations between zeroshot and the final fine-tuned model
4. Plane: Shows evaluations on the 2D plane between the zeroshot model, the fine-tuned model (10 epochs) and a model from the training procedure (5 epochs). This can give insights into whether the straight interpolation between zeroshot and fine-tuned is optimal or if the direction of the optimization process can give a better performance.