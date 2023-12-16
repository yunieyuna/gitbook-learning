# Fine Tuning an LLM

Ref: [https://dr-bruce-cottman.medium.com/part-1-eight-major-methods-for-finetuning-an-llm-6f746c7259ee](https://dr-bruce-cottman.medium.com/part-1-eight-major-methods-for-finetuning-an-llm-6f746c7259ee)

Ref: [https://qiita.com/mshinoda88/items/fc562ec6a84f45e89e70](https://qiita.com/mshinoda88/items/fc562ec6a84f45e89e70)



## 1. Universal Language Model Finetuning (ULMFiT)

Paper: [https://arxiv.org/abs/1801.06146](https://arxiv.org/abs/1801.06146)

ULMFiT establishes a baseline of techniques from which all of the other four fine-tuning methods can be described and compared.

**Step 1**: Language Model training: We train a Language Model on a large corpus of unlabeled text. A standard first step.

**Step 2**: Target Model Layer (Parameter) Fine-tuning: We take a dataset specific to our target task, which is labeled and smaller than the original training dataset, and train the model further on this data. The goal is to adapt the model to the specific language patterns and concepts of the task at hand using supervised learning.

**Step 3**: Target Task Model Fine-tuning with Classifier Layers: In the final step, we add one or more task-specific classifier layers on top of the fine-tuned language model. These additional layers are randomly initialized and trained using the labeled data from the target task. The rest of the language model remains frozen during this step. The classifier layers allow the model to make task-specific predictions based on the information learned from both the general language model and the fine-tuning stages.

The "ULMFiT" paper found three techniques for improving NLPs, that resulted in "Selective Parameter Subset fine-tuning":

1. Gradual Unfreezing while training the classifier: you don't immediately train all layers. Instead, you gradually "unfreeze" layers starting from the final ones, and moving towards the initial layers. This method, called "gradual unfreezing", helps prevent catastrophic forgetting which can occur when a model abruptly changes to accomodate new learning. The earlier layers which have learned more generic features, are updated slowly. The later layers, which are learning more task-specific features, are updated more aggressively.
2. [Slanted Triangular Learning Rates (STLR)](https://paperswithcode.com/method/slanted-triangular-learning-rates): The next unique approach of ULMFiT is the application of slanted triangular learning rates during classifier fine-tuning. This means that, in the beginning, learning rates slowly increase and then gradually decrease at a much higher rate. This strategy is effective because it first allows the model to explore a wider part of the parameter space (when the learning rate is higher), and then to converge to an optimal solution (when the learning rate decreases).
3. Discriminative Fine-tuning: Lastely, each layer of the model is fine-tuned with different learning rates, a practive termed discriminative fine-tuning. This means that each layer in the neural network can have its own learning rate. The reasoning behind this is that different layers of the model capture different types of information, so they should be fine-tuned at different speeds.

## 2. [Gradient-based parameter importance ranking](https://www.sciencedirect.com/science/article/pii/S2352484721004315) (PIR) or Random Forest-based importance ranking

The heart of PIR is its parameter selection process. It analyzes the task at hand and assigns importance ranking to all parameters in the model. It then selects tha ones that have the highest impact on the task's performance.

ULMFiT freezes layers, leaving the parameters in those layers unchanged. PIR freezes non-selected parameters, no matter what layer the parameters are in. By freezing parameters, not layers, the PIR leaves LLM's knowledge in areas that aren't directly relevant to the task.

Two prominent ways to implement PIR are:

* Gradient-based importance ranking: Gradient-based importance ranking works by calculating the gradient of the loss function with respect to each parameter in the LLM. The gradient of a parameter is a measure of how much the parameter affects the loss function. The parameters with the highest gradients are considered to be the most important.
* [Random forest importance ranking](https://academic.oup.com/bib/article/12/4/369/241163?view=extract\&login=false): Random forest importance ranking works by training a random forest model on the parameters of the LLM. The parameters that are most important for predicting the output of the random forest model are considered to be the most important parameters in the LLM.

In practice, PIR has been abandoned due to the number of parameters in an LLM.

## 3. LoRA: Low Ranking Adaptation ... of Parameters for LLMs, A Current Method

Before discussing LoRA, I go through a brief refresher on methods for reducing a matrix to a low-rank approximation.

[SVD](https://langvillea.people.cofc.edu/DISSECTION-LAB/Emmie'sLSI-SVDModule/p5module.html) (singular value decomposition) decomposes a matrix into three smaller matrices: a diagonal matrix of singular values, a matrix of left singular vectors, and a matrix of right singular vectors. The central matrix (V) singular values represent the magnitudes of the principal components, the left singular vectors represent the directions of the principal components, and the right singualr vectors represent the weights of the principal components.

PCA (principal component analysis) is a special case of SVD where the singular values are all unique. PCA is used to reduce the dimensionality of a dataset by projecting the data onto the first few principal components.

### LoRA Fine-tuning

* LoRA (低ランク パラメーター化更新行列) は、ニューラル ネットワーク、特に LLM のアテンション層に適用されます。
* トランスフォーマー層などの高密度層を使用する他の深層学習モデルにも適用できます。
* ただし、これらの層は疎ではなく密 (パラメーターの独立した層) であるため、行列は下位のランクの近似はよくない。

{% embed url="https://arxiv.org/abs/2106.09685" %}

### The problems of LoRA:

* LoRA can be less effective (it does not work) than traditional fine-tuning methods for tasks that require the model to learn specific patterns. For example, in language-like tasks, such as classification, summarization, and question-answer pairs, from fine-tuning GPT-3, LoRA scores great. For arithmetic or playing poker LoRA will not work.
* LoRA can be more difficult to implement than traditional fine-tuning methods.

### Implementations of LoRA include:

{% embed url="https://github.com/microsoft/LoRA?source=post_page-----6f746c7259ee--------------------------------" %}

{% embed url="https://github.com/cloneofsimo/lora?source=post_page-----6f746c7259ee--------------------------------" %}







