# Chapter 8: Prompt Engineering

## 8 Customizing LLMs and their output

### Join our book community on Discord

[https://packt.link/EarlyAccessCommunity](https://packt.link/EarlyAccessCommunity)

![Qr code Description automatically generated](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file54.png)

This chapter is about techniques and best practices to improve the reliability and performance of large language models (LLMs) in certain scenarios such as on complex reasoning and problem-solving tasks. Generally, this process of adapting a model for a certain task or making sure that our model output corresponds to what we expect is called conditioning. In this chapter, we’ll discuss fine-tuning and prompting as methods for conditioning.Fine-tuning involves training the pre-trained base model on specific tasks or datasets relevant to the desired application. This process allows the model to adapt and become more accurate and contextually aligned for the intended use case. Similarly, by providing additional input or context at inference time, large language models (LLMs) can generate text tailored to a particular tasks or style.Prompt design is highly significant for unlocking LLM reasoning capabilities, the potential for future advancements in models and prompting techniques, and these principles and techniques form a valuable toolkit for researchers and practitioners working with large language models. Understanding how LLMs generate text token-by-token helps create better reasoning prompts.Prompting is still an empirical art - trying variations to see what works is often needed. But some prompt engineering insights can transfer across models and tasks. We’ll discuss the tools in LangChain to enable advanced prompt engineering strategies like few-shot learning, dynamic example selection, and chained reasoning.Throughout the chapter, we’ll work on fine-tuning and prompting with LLMs, which you can find in the `notebooks` directory in the Github repository for the book at [https://github.com/benman1/generative\_ai\_with\_langchain](https://github.com/benman1/generative\_ai\_with\_langchain)The main sections are:

* Conditioning and Alignment
* Fine-Tuning
* Prompt Engineering

Let’s start by discussing conditioning and alignment, why it’s important, and how we can achieve it.

### Conditioning and alignment

Alignment, in the context of generative AI models, refers to ensuring that the outputs of these models are consistent with human values, intentions, or desired outcomes. It involves guiding the model's behavior to be in line with what is considered ethical, appropriate, or relevant within a given context. The concept of alignment is crucial to avoid generating outputs that may be biased, harmful, or deviate from the intended purpose. Addressing alignment involves careful attention to the biases present in training data, iterative feedback loops involving human reviewers, refining objective functions during training/fine-tuning stages, leveraging user feedback, and continuous monitoring during deployment to ensure ongoing alignment.There are several reasons one may want to condition a large language model. The first is to control the content and style of the outputs. For example, conditioning on certain keywords or attributes like formality level can produce more relevant and high-quality text. Conditioning also encompasses safety measures to prevent the generation of malicious or harmful content. For example, avoiding generating misleading information, inappropriate suggestions, or potentially dangerous instructions, or – more generally aligning the model with certain values.The potential benefits of conditioning large language models are numerous. By providing more specific and relevant input, we can achieve outputs that are tailored to our needs. For instance, in a customer support chatbot, conditioning the model with user queries allows it to generate responses that address their concerns accurately. Conditioning also helps control biased or inappropriate outputs by constraining the model's creativity within specific boundaries.Furthermore, by conditioning large language models, we can make them more controllable and adaptable. We can fine-tune and shape their behavior according to our requirements and create AI systems that are reliable in specific domains such as legal advice or technical writing.However, there are potential downsides to consider as well. Conditioning models too heavily might result in overfitting, where they become excessively reliant on specific inputs and struggle with generating creative or diverse outputs in different contexts. Moreover, conditioning should be utilized responsibly since large language models have the tendency to amplify biases present in the training data. Care must be taken not to exacerbate issues related to bias or controversial topics when conditioning these models.

> **Benefits of alignment** include:

* Enhanced User Experience: Aligned models generate outputs that are relevant to user queries or prompts.
* Trust-building: Ensuring ethical behavior helps build trust among users/customers.
* Brand Reputation: By aligning with business goals regarding branding consistency and desired tone/style guidelines.
*   Mitigating Harmful Effects: Alignment with safety, security, and privacy considerations helps prevent the generation of harmful or malicious content.

    > **Potential downsides** include:
* Challenging Balance: Striking a balance between extreme alignment (overly conservative) and creative freedom (overly permissive) can be difficult.
* Limitations of Automated Metrics: Quantitative evaluation metrics might not capture alignment nuances fully.
* Subjectivity: Alignment judgments are often subjective, requiring careful consideration and consensus building on desired values and guidelines.

Pre-training a large model on diverse data to learn patterns and language understanding results in a base model that has a broad understanding of various topics but lacks specificity or alignment to any particular context. While base models such as GPT-4 are capable of generating impressive text on a wide range of topics, conditioning them can enhance their capabilities in terms of task relevance, specificity, and coherence, and make their outputs more relevant and on-topic. Without conditioning, these models tend to generate text that may not always align perfectly with the desired context. By conditioning them, we can guide the language models to produce outputs that are more closely related to the given input or instructions. The major advantage of conditioning is that it allows guiding the model without extensive retraining. It also enables interactive control and switching between modes. Conditioning can happen at different stages of the model development cycle—from fine-tuning to output generation in various contexts. There are several options for achieving alignment of large language models. One approach is to condition during fine-tuning, by training the model on a dataset reflective of the desired outputs. This allows the model to specialize, but requires access to relevant training data. Another option is to dynamically condition the model at inference time by providing conditional input along with the main prompt. This is more flexible but introduces some complexity during deployment.In the next section, I will summarize key methods for alignment such as fine-tuning and prompt engineering, discuss the rationale, and examine their relative pros and cons.

#### Methods for alignment

With the advent of large pre-trained language models like GPT-3, there has been growing interest in techniques to adapt these models for downstream tasks. This process is known as fine-tuning. Fine-tuning allows pre-trained models to be customized for specific applications while leveraging the vast linguistic knowledge acquired during pre-training. The idea of adapting pre-trained neural networks originated in computer vision research in the early 2010s. In NLP, Howard and Ruder (2018) demonstrated the effectiveness of fine-tuning pre-trained contextual representations like ELMo and ULMFit on downstream tasks. The seminal BERT model (Devlin and others., 2019) established fine-tuning of pre-trained transformers as the de facto approach in NLP.The need for fine-tuning arises because pre-trained LMs are designed to model general linguistic knowledge, not specific downstream tasks. Their capabilities manifest only when adapted to particular applications. Fine-tuning allows pre-trained weights to be updated for target datasets and objectives. This enables knowledge transfer from the general model while customizing it for specialized tasks. Several approaches have been proposed for alignment, with trade-offs in efficacy and efficiency and it’s worth to delve a bit more into the details of each of these alignment methods. **Full Fine-Tuning** involves updating all the parameters of the pre-trained language model during fine-tuning. The model is trained end-to-end on the downstream tasks, allowing the weights to be updated globally to maximize performance on the target objectives. FFT consistently achieves strong results across tasks but requires extensive computational resources and large datasets to avoid overfitting or forgetting.In **Adapter Tuning** additional trainable adapter layers are inserted, usually bottleneck layers, into the pre-trained model while keeping the original weights frozen. Only the newly added adapter layers are trained on the downstream tasks. This makes tuning parameter-efficient as only a small fraction of weights are updated. However, as the pre-trained weights remain fixed, adapter tuning has a risk of underfitting to the tasks. The insertion points and capacity of the adapters impact overall efficacy.**Prefix Tuning**: This prepends trainable vectors to each layer of the LM, which are optimized during fine-tuning while the base weights remain frozen. The prefixes allow injection of inductive biases into the model. Prefix tuning has a smaller memory footprint compared to adapters but has not been found to be as effective. The length and initialization of prefixes impact efficacy.In **Prompt Tuning**, the input text is appended with trainable prompt tokens which provide a soft prompt to induce the desired behavior from the LM. For example, a task description can be provided as a prompt to steer the model. Only the added prompt tokens are updated during training while the pre-trained weights are frozen. Performance is heavily influenced by prompt engineering. Automated prompting methods are being explored.**Low-Rank Adaptation (LoRA)** adds pairs of low-rank trainable weight matrices to the frozen LM weights. For example, to each weight W, low-rank matrices B and A are added such that the forward pass uses W + BA. Only B and A are trained, keeping the base W frozen. LoRA achieves reasonable efficacy with greater parameter efficiency than full tuning. The choice of rank r impacts tradeoffs. LoRA enables tuning giant LMs on limited hardware.Another way to ensure proper alignment of outputs is through **human oversight** methods like human-in-the-loop systems. These systems involve human reviewers who provide feedback and make corrections if necessary. Human involvement helps align generated outputs with desired values or guidelines set by humans.Here is a table summarizing the different techniques for steering generative AI outputs:

| **Stage**                   | **Technique**                     | **Examples**                              |
| --------------------------- | --------------------------------- | ----------------------------------------- |
| Training                    | Pre-training                      | Training on diverse data                  |
|                             | Objective Function                | Careful design of training objective      |
|                             | Architecture and Training Process | Optimizing model structure and training   |
| Fine-Tuning                 | Specialization                    | Training on specific datasets/tasks       |
| Inference-Time Conditioning | Dynamic Inputs                    | Prefixes, control codes, context examples |
| Human Oversight             | Human-in-the-Loop                 | Human review and feedback                 |

Figure 8.1: Steering generative AI outputs.

Combining these techniques provides developers with more control over the behavior and outputs of generative AI systems. The ultimate goal is to ensure that human values are incorporated at all stages, from training to deployment, in order to create responsible and aligned AI systems.Furthermore, careful design choices in the pre-training objective function also impact what behaviors and patterns the language model learns initially. By incorporating ethical considerations into these objective functions, developers can influence the initial learning process of large language models.We can distinguish a few more approaches in fine-tuning such as online and offline. InstructGPT was considered a game-changer because it demonstrated the potential to significantly improve language models, such as GPT-3, by incorporating reinforcement learning from human feedback (RLHF). Let’s talk about the reasons why InstructGPT had such a transformative impact.

**Reinforcement learning with human feedback**

In their March 2022 paper, Ouyang and others from OpenAI demonstrated using reinforcement learning from human feedback (RLHF) with proximal policy optimization (PPO) to align large language models like GPT-3 with human preferences. Reinforcement learning from human feedback (RLHF) is an online approach that fine-tunes LMs using human preferences. It has three main steps:

1. Supervised pre-training: The LM is first trained via standard supervised learning on human demonstrations.
2. Reward model training: A reward model is trained on human ratings of LM outputs to estimate reward.
3. RL fine-tuning: The LM is fine-tuned via reinforcement learning to maximize expected reward from the reward model using an algorithm like PPO.

The main change, RLHF, allows incorporating nuanced human judgments into language model training through a learned reward model. As a result, human feedback can steer and improve language model capabilities beyond standard supervised fine-tuning. This new model can be used to follow instructions that are given in natural language, and it can answer questions in a way that’s more accurate and relevant than GPT-3. InstructGPT outperformed GPT-3 on user preference, truthfulness, and harm reduction, despite having 100x fewer parameters.Starting in March 2022, OpenAI started releasing the GPT-3.5 series models, upgraded versions of GPT-3, which include fine-tuning with RLHF.There are three advantages of fine-tuning that were immediately obvious to users of these models:

1. Steerability: the capability of models to follow instructions (instruction-tuning)
2. Reliable output-formatting: this became important, for example, for API calls/function calling)
3. Custom tone: this makes it possible to adapt the output style as appropriate to task and audience.

InstructGPT opened up new avenues for improving language models by incorporating reinforcement learning from human feedback methods beyond traditional fine-tuning approaches. RL training can be unstable and computationally expensive, notwithstanding, its success inspired further research into refining RLHF techniques, reducing data requirements for alignment, and developing more powerful and accessible models for a wide range of applications.

**Offline Approaches**

Offline methods circumvent the complexity of online RL by directly utilizing human feedback. We can distinguish between ranking-based on language-based approaches:

* Ranking-based: Human rankings of LM outputs are used to define optimization objectives for fine-tuning, avoiding RL entirely. This includes methods like Preference Ranking Optimization (PRO; Song et al., 2023) and Direct Preference Optimization (DPO; Rafailov et al., 2023).
* Language-based: Human feedback is provided in natural language and utilized via standard supervised learning. For example, Chain of Hindsight (CoH; Liu et al., 2023) converts all types of feedback into sentences and uses them to fine-tune the model, taking advantage of the language comprehension capabilities of language models.

Direct Preference Optimization (DPO) is a simple and effective method for training language models to adhere to human preferences without needing to explicitly learn a reward model or use reinforcement learning. While it optimizes the same objective as existing RLHF methods, it is much simpler to implement, more stable, and achieves strong empirical performance.Researchers from Meta, in the paper “_LIMA: Less Is More for Alignment_”, simplified alignment by minimizing a supervised loss on only 1,000 carefully curated prompts in fine-training the LLaMa model. Based on the favorable human preferences when comparing their outputs to DaVinci003 (GPT-3.5), they conclude that fine-training has only minimal importance. This they refer to as the superficial alignment hypothesis.Offline approaches offer more stable and efficient tuning. However, they are limited by static human feedback. Recent methods try to combine offline and online learning.While both DPO and RLHF with PPO aim to align LLMs with human preferences, they differ in terms of complexity, data requirements, and implementation details. DPO offers simplicity but achieves strong performance by directly optimizing probability ratios. On the other hand, RLHF with PPO in InstructGPT introduces more complexity but allows for nuanced alignment through reward modeling and reinforcement learning optimization.

**Low-Rank Adaptation**

LLMs have achieved impressive results in Natural Language Processing and are now being used in other domains such as Computer Vision and Audio. However, as these models become larger, it becomes difficult to train them on consumer hardware and deploying them for each specific task becomes expensive. There are a few methods that reduce computational, memory, and storage costs, while improving performance in low-data and out-of-domain scenarios.Low-Rank Adaptation (LoRA) freezes the pre-trained model weights and introduces trainable rank decomposition matrices into each layer of the Transformer architecture to reduce the number of trainable parameters. LoRA achieves comparable or better model quality compared to fine-tuning on various language models (RoBERTa, DeBERTa, GPT-2, and GPT-3) while having fewer trainable parameters and higher training throughput.The QLORA method is an extension of LoRA, which enables efficient fine-tuning of large models by backpropagating gradients through a frozen 4-bit quantized model into learnable low-rank adapters. This allows fine-tuning a 65B parameter model on a single GPU. QLORA models achieve 99% of ChatGPT performance on Vicuna using innovations like new data types and optimizers. In particular, QLORA reduces the memory requirements for fine-tuning a 65B parameter model from >780GB to <48GB without affecting runtime or predictive performance.

> **Quantization** refers to techniques for reducing the numerical precision of weights and activations in neural networks like large language models (LLMs). The main purpose of quantization is to reduce the memory footprint and computational requirements of large models.
>
> > Some key points about quantization of LLMs:

* It involves representing weights and activations using fewer bits than standard single-precision floating point (FP32). For example, weights could be quantized to 8-bit integers.
* This allows shrinking model size by up to 4x and improving throughput on specialized hardware.
* Quantization typically has a minor impact on model accuracy, especially with re-training.
* Common quantization methods include scalar, vector, and product quantization which quantize weights separately or in groups.
* Activations can also be quantized by estimating their distribution and binning appropriately.
* Quantization-aware training adjusts weights during training to minimize quantization loss.
* LLMs like BERT and GPT-3 have been shown to work well with 4-8 bit quantization via fine-tuning.

Parameter-Efficient Fine-tuning (PEFT) methods enable the use of small checkpoints for each task, making the models more portable. These small trained weights can be added on top of the LLM, allowing the same model to be used for multiple tasks without replacing the entire model.In the next section, we’ll discuss methods for conditioning large language models (LLMs) at inference time.

**Inference-Time conditioning**

One commonly used approach is **conditioning at inference time** (output generation phase) where specific inputs or conditions are provided dynamically to guide the output generation process. LLM fine-tuning may not always be feasible or beneficial in certain scenarios:

1. Limited Fine-Tuning Services: Some models are only accessible through APIs that lack or have restricted fine-tuning capabilities.
2. Insufficient Data: In cases where there is a lack of data for fine-tuning, either for the specific downstream task or relevant application domain.
3. Dynamic Data: Applications with frequently changing data, such as news-related platforms, may struggle to fine-tune models frequently, leading to potential drawbacks.
4. Context-Sensitive Applications: Dynamic and context-specific applications like personalized chatbots cannot perform fine-tuning based on individual user data.

For conditioning at inference time, most commonly, we provide a textual prompt or instruction at the beginning of the text generation process. This prompt can be a few sentences or even a single word, acting as an explicit indication of the desired output. Some common techniques for dynamic inference-time conditioning include:

* Prompt tuning: Providing natural language guidance for intended behavior. Sensitive to prompt design.
* Prefix tuning: Prepending trainable vectors to LLM layers.
* Constraining tokens: Forcing inclusion/exclusion of certain words
* Metadata: Providing high-level info like genre, target audience, etc.

Prompts can facilitate generating text that adheres to specific themes, styles, or even mimics a particular author's writing style. These techniques involve providing contextual information during inference time such as for in-context learning or retrieval augmentation.An example of prompt tuning is prefixing prompts, where instructions like "Write a child-friendly story about..." are prepended to the prompt. For example, in chatbot applications, conditioning the model with user messages helps it generate responses that are personalized and pertinent to the ongoing conversation. Further examples include prepending relevant documents to prompts to assist LLMs with writing tasks (e.g., news reports, Wikipedia pages, company documents), or retrieving and prepending user-specific data (financial records, health data, emails) before prompting an LLM to ensure personalized answers. By conditioning LLM outputs on contextual information at runtime, these methods can guide models without relying on traditional fine-tuning processes.Often demonstrations are part of the instructions for reasoning tasks, where few-shot examples are provided to induce desired behavior. Powerful LLMs, such as GPT-3, can solve tasks without further training through prompting techniques. In this approach, the problem to be solved is presented to the model as a text prompt, possibly with some text examples of similar problems and their solutions. The model must provide a completion of the prompt via inference. **Zero-shot prompting** involves no solved examples, while few-shot prompting includes a small number of examples of similar (problem, solution) pairs. It has shown that prompting provides easy control over large frozen models like GPT-3 and allows steering model behavior without extensive fine-tuning. Prompting enables conditioning models on new knowledge with low overhead, but careful prompt engineering is needed for best results. This is what we’ll discuss as part of this chapter.In prefix tuning, continuous task-specific vectors are trained and supplied to models at inference time. continuous task-specific vectors. Similar ideas have been proposed for adapter-approaches such as parameter efficient transfer learning (PELT) or Ladder Side-Tuning (LST).Conditioning at inference time can also happen during sampling such as grammar-based sampling, where the output can be constrained to be compatible with certain well-defined patterns, such as a programming language syntax.

**Conclusions**

Full fine-tuning consistently achieves strong results but often requires extensive resources, and trade-offs exist between efficacy and efficiency. Methods like adapters, prompts, and LoRA reduce this burden via sparsity or freezing, but can be less effective. The optimal approach depends on constraints and objectives. Future work on improved techniques tailor-made for large LMs could push the boundaries of both efficacy and efficiency. Recent work blends offline and online learning for improved stability. Integrating world knowledge and controllable generation remain open challenges.Prompt-based techniques allow flexible conditioning of LLMs to induce desired behaviors without intensive training. Careful prompt design, optimization, and evaluation is key to effectively controlling LLMs. Prompt-based techniques allow conditioning LLMs on specific behaviors or knowledge in a flexible, low-resource manner.

#### Evaluations

Alignment is evaluated on alignment benchmarks like HUMAN and generalization tests like FLAN. There are a few core benchmarks with high differentiability to accurately assess model strengths and weaknesses such as these:

* English knowledge: MMLU
* Chinese knowledge: C-Eval
* Reasoning: GSM8k / BBH (Algorithmic)
* Coding: HumanEval / MBPP

After balancing these directions, additional benchmarks like MATH (high-difficulty reasoning) and Dialog could be pursued.A particularly interesting evaluation is in math or reasoning, where generalization abilities would be expected to be very strong. The MATH benchmark demonstrates high-level difficulty, and GPT-4 achieves varying scores based on prompting methods. Results range from naive prompting via few-shot evaluations to PPO + process-based reward modeling. If fine-tuning involves dialog data only, it might negatively affect existing capabilities such as MMLU or BBH. Prompt engineering is essential, as biases and query difficulty impact evaluations.There are quantitative metrics like perplexity (measuring how well a model predicts data) or BLEU score (capturing similarity between generated text and reference text). These metrics provide rough estimates but may not fully capture semantic meaning or alignment with higher-level goalsOther metrics include user preferences ratings through human evaluation, pairwise preference, utilizing pre-trained reward model for online small/medium models or automated LLM-based assessments (for example GPT-4). Human evaluations can sometimes be problematic since humans can be swayed by subjective criteria such as an authoritative tone in the response rather than the actual accuracy. Conducting evaluations where users assess the quality, relevance, appropriateness of generated text against specific criteria set beforehand provides more nuanced insights into alignment. Fine-tuning is not intended to solely improve user preferences on a given set of prompts. Its primary purpose is to address AI safety concerns by reducing the occurrence of undesirable outputs such as illegal, harmful, abusive, false, or deceptive content. This focus on mitigating risky behavior is crucial in ensuring the safety and reliability of AI systems. Evaluating and comparing models based purely on user preferences without considering the potential harm they may cause can be misleading and prioritize suboptimal models over safer alternatives. In summary, evaluating LLM alignment requires careful benchmark selection, consideration of differentiability, and a mix of automatic evaluation methods and human judgments. Attention to prompt engineering and specific evaluation aspects is necessary to ensure accurate assessment of model performance.In the next section, we’ll fine-tune a small open-source LLM (OpenLLaMa) for a question answering with PEFT and quantization, and we’ll deploy it on HuggingFace.

### Fine-Tuning

As we’ve discussed in the first section of this chapter, the goal of model fine-tuning for LLMs is to optimize a model to generate outputs that are more specific to a task and context than the original foundation model. Amongst the multitude of tasks and scenarios, where we might want to apply this approach are these:

* Software Development
* Document classification
* Question-Answering
* Information Retrieval
* Customer Support

In this section, we’ll fine-tune a model for question answering. This recipe is not specific to LangChain, but we’ll point out a few customizations, where LangChain could come in. For performance reasons, we'll run this on Google Colab instead of the usual local environment.

> **Google Colab** is a computation environment that provides different means for hardware acceleration of computation tasks such as Tensor Processing Units (TPUs) and Graphical Processing Units (GPUs). These are available both in free and professional tiers. For the purpose of the task in this section, the free tier is completely sufficient. You can sign into a Colab environment at this url: [https://colab.research.google.com/](https://colab.research.google.com/)

Please make sure you set your google colab machine settings in the top menu to TPU or GPU in order to make sure you have sufficient resources to run this and that the training doesn't take too long. We’ll install all required libraries in the Google Colab environment – I am adding the versions of these libraries that I’ve used in order to make our fine-tuning repeatable:

* peft: Parameter-Efficient Fine-Tuning (PEFT; version 0.5.0)
* trl: Proximal Policy Optimization (0.6.0)
* bitsandbytes: k-bit optimizers and matrix multiplication routines, needed for quantization (0.41.1)
* accelerate: train and use PyTorch models with multi-GPU, TPU, mixed-precision (0.22.0)
* transformers: HuggingFace transformers library with backends in JAX, PyTorch and TensorFlow (4.32.0)
* datasets: community-driven open-source library of datasets (2.14.4)
* sentencepiece: Python wrapper for fast tokenization (0.1.99)
* wandb: for monitoring the training progress on Weights and Biases (0.15.8)
* langchain for loading the model back as a langchain llm after training (0.0.273)

We can install these libraries from the Colab notebook as follows:

```
!pip install -U accelerate bitsandbytes datasets transformers peft trl sentencepiece wandb langchain
```

In order to download and train models from HuggingFace, we need to authenticate with the platform. Please note that if you want to push your model to HuggingFace later, you need to generate a new API token with write permissions on HuggingFace: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file55.png" alt="Figure 8.3: Creating a new API token on HuggingFace write permissions." height="600" width="754"><figcaption><p>Figure 8.3: Creating a new API token on HuggingFace write permissions.</p></figcaption></figure>

We can authenticate from the notebook like this:

```
import notebook_login
notebook_login()
```

When prompted, paste your HuggingFace access token.

> A note of caution before, we start: when executing the code, you need to log into different services so make sure you pay attention when running the notebook!

Weights and Biases (W\&B) is an MLOps platform that can help developers to monitor and document ML training workflows from end to end. As mentioned earlier, we will use W\&B to get an idea of how well the training is working, in particular if the model is improving over time.For W\&B, we need to name the project; alternatively, we can use wandb's `init()` method:

```
import os
os.environ["WANDB_PROJECT"] = "finetuning"
```

In order to authenticate with W\&B, you need to create a free account with them for this at [https://www.wandb.ai](https://www.wandb.ai/) You can find your API key on the Authorize page: [https://wandb.ai/authorize](https://wandb.ai/authorize)Again, we need to paste in our API token. If the previous training run is still active – this could be from a previous execution of the notebook if you are running a second time –, let's make sure we start a new one! This will ensure that we get new reports and dashboard on W\&B:

```
if wandb.run is not None:
    wandb.finish()
```

Next, we’ll need to choose a dataset against which we want to optimize. We can use lots of different datasets here that are appropriate for coding, storytelling, tool use, SQL generation, grade-school math questions (GSM8k), or many other tasks. HuggingFace provides a wealth of datasets, which can be viewed at this url: [https://huggingface.co/datasets](https://huggingface.co/datasets)These cover a lot of different, even the most niche tasks. We can also customize our own dataset. For example, we can use langchain to set up training data. There are quite a few methods available for filtering that could help reduce redundancy in the dataset. It would have been appealing to show data collection as a practical recipe in this chapter. However, because of the complexity I am leaving it out of scope for the book.It might be harder to filter for quality from web data, but there are a lot of possibilities. For code models, we could apply code validation techniques to score segments as a quality filter. If the code comes from Github, we can filter by stars or by stars by repo owner. For texts in natural language, quality filtering is not trivial. Search engine placement could serve as a popularity filter since it's often based on user engagement with the content. Further, knowledge distillation techniques could be tweaked as a filter by fact density and accuracy.In this recipe, we are fine-tuning for question answering performance with the Squad V2 dataset. You can see a detailed dataset description on HuggingFace: [https://huggingface.co/spaces/evaluate-metric/squad\_v2](https://huggingface.co/spaces/evaluate-metric/squad\_v2)

```
from datasets import load_dataset
dataset_name = "squad_v2" 
dataset = load_dataset(dataset_name, split="train")
eval_dataset = load_dataset(dataset_name, split="validation") 
```

We are taking both training and validation splits. The Squad V2 dataset bas a part that’s supposed to be used in training and another one in validation as we can see in the output of `load_dataset(dataset_name)`:

```
DatasetDict({
    train: Dataset({
        features: ['id', 'title', 'context', 'question', 'answers'],
        num_rows: 130319
    })
    validation: Dataset({
        features: ['id', 'title', 'context', 'question', 'answers'],
        num_rows: 11873
    })
}) 
```

We’ll use the validation splits for early stopping. Early stopping will allow us to stop training when the validation error begins to degrade.The Squad V2 dataset is composed of various features, which we can see here:

```
{'id': Value(dtype='string', id=None), 
 'title': Value(dtype='string', id=None), 
 'context': Value(dtype='string', id=None), 
 'question': Value(dtype='string', id=None), 
 'answers': Sequence(feature={'text': Value(dtype='string', id=None), 
 'answer_start': Value(dtype='int32', id=None)}, length=-1, id=None)}
```

The basic idea in training is prompting the model with a question and comparing the answer to the dataset.We want a small model that we can run locally at a decent token rate. LLaMa-2 models require signing a license agreement with your email address and to get confirmed (which, to be fair, can be very fast) as it comes with restrictions to commercial use. LLaMa derivates such as OpenLLaMa have been performing quite well as can be evidenced on the HF leaderboard: [https://huggingface.co/spaces/HuggingFaceH4/open\_llm\_leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open\_llm\_leaderboard)OpenLLaMa version 1 cannot be used for coding tasks, because of the tokenizer. Therefore, let's use v2! We’ll use a 3 billion parameter model, which we’ll be able to use even on older hardware:

```
model_id = "openlm-research/open_llama_3b_v2" 
new_model_name = f"openllama-3b-peft-{dataset_name}"
```

We can use even smaller models such as `EleutherAI/gpt-neo-125m` which can also give a very good compromise between resource use and performance.Let’s load the model:

```
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
device_map="auto"
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
base_model.config.use_cache = False
```

The Bits and Bytes configuration makes it possible to quantize our model in 8, 4, 3 or even 2 bits with a much-accelerated inference and lower memory footprint without a incurring a big cost in terms of performance. We are going to store model checkpoints on Google Drive; you need to confirm your login to your google account:

```
from google.colab import drive
drive.mount('/content/gdrive')
```

We’ll need to authenticate with Google for this to work. We can set our output directory for model checkpoints and logs to our Google Drive:

```
output_dir = "/content/gdrive/My Drive/results"
```

If you don't want to use google drive, just set this to a directory on your computer.For training, we need to set up a tokenizer:

```
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```

Now we’ll define our training configuration. We’ll set up LORA and other training arguments:

```
from transformers import TrainingArguments, EarlyStoppingCallback
from peft import LoraConfig
# More info: https://github.com/huggingface/transformers/pull/24906
base_model.config.pretraining_tp = 1
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)
training_args = TrainingArguments(
    output_dir=output_dir, 
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=2000,
    num_train_epochs=100,
    evaluation_strategy="steps",
    eval_steps=5,
    save_total_limit=5 
    push_to_hub=False, 
    load_best_model_at_end=True,
    report_to="wandb"
)
```

A few comments to explain some of these parameters are in order. The `push_to_hub` argument means that we can push the model checkpoints to the HuggingSpace Hub regularly during training. For this to work you need to set up the HuggingSpace authentication (with write permissions as mentioned). If we opt for this, as `output_dir` we can use `new_model_name`. This will be the repository name under which the model will be available here on HuggingFace: [https://huggingface.co/models](https://huggingface.co/models)Alternatively, as I’ve done here, we can save your model locally or to the cloud, for example google drive to a directory.I’ve set `max_steps` and `num_train_epochs` very high, because I’ve noticed that training can still improve after many steps. We are using early stepping together with a high number of maximum training steps to get the model to converge to higher performance. For early stopping, we need to set the `evaluation_strategy` as `"steps"` and `load_best_model_at_end=True`.`eval_steps` is the number of update steps between two evaluations. `save_total_limit=5` means that only last 5 models are saved. Finally, `report_to="wandb"` means that we’ll send training stats, some model metadata, and hardware information to W\&B, where we can look at graphs and dashboards for each run. The training can then use our configuration:

```
from trl import SFTTrainer
trainer = SFTTrainer(
    model=base_model,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    dataset_text_field="question",  # this depends on the dataset!
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=200)]
)
trainer.train()
```

The training can take quite a while, even running on TPU device. The evaluating and early stopping slows the training down by a lot. If you disable the early stopping, you can make this much faster.We should see some statistics as the training progresses, but it’s nicer to show the graph of performance as we can see it on W\&B:

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file56.png" alt="Figure 8.4: Fine-tuning training loss over time (steps)." height="1387" width="2642"><figcaption><p>Figure 8.4: Fine-tuning training loss over time (steps).</p></figcaption></figure>

After training is done, we can save the final checkpoint on disk for re-loading:

```
trainer.model.save_pretrained(
    os.path.join(output_dir, "final_checkpoint"),
)
```

We can now share our final model with friends in order to brag about the performance we've achieved by manually pushing to HuggingFace:

```
trainer.model.push_to_hub(
    repo_id=new_model_name
)
```

We can now load the model back using the combination of our HuggingFace username and the repository name (new model name).Let’s quickly show how to use this model in LangChain. usually, the peft model is stored as an adapter, not as a full model, therefore the loading is a bit different:

```
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
model_id = 'openlm-research/open_llama_3b_v2'
config = PeftConfig.from_pretrained("benji1a/openllama-3b-peft-squad_v2")
model = AutoModelForCausalLM.from_pretrained(model_id)
model = PeftModel.from_pretrained(model, "benji1a/openllama-3b-peft-squad_v2")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=256
)
llm = HuggingFacePipeline(pipeline=pipe)
```

We’ve done everything so far on Google Colab, but we can equally execute this locally, just note that you need to have the huggingface peft library installed!So far, we’ve shown how to fine-tune and deploy an open-source LLM. Some commercial models can be fine-tuned on custom data as well. For example, both OpenAI’s GPT-3.5 and Google’s PaLM model offer this capability. This has been integrated with a few Python libraries. With the Scikit-LLM library, this is only a few lines of code in either case:Fine-tuning a PaLM model for text classification can be done like this:

```
from skllm.models.palm import PaLMClassifier
clf = PaLMClassifier(n_update_steps=100)
clf.fit(X_train, y_train) # y_train is a list of labels
labels = clf.predict(X_test)
```

Similarly, you can fine-tune the GPT-3.5 model for text classification like this:

```
from skllm.models.gpt import GPTClassifier
clf = GPTClassifier(
        base_model = "gpt-3.5-turbo-0613",
        n_epochs = None, # int or None. When None, will be determined automatically by OpenAI
        default_label = "Random", # optional
)
clf.fit(X_train, y_train) # y_train is a list of labels
labels = clf.predict(X_test)
```

Interestingly, in the fine-tuning available on OpenAI, all inputs are passed through a moderation system to make sure that the inputs are compatible with safety standards.This concludes fine-tuning. On the extreme end, LLMs can be deployed and queried without any task-specific tuning. By prompting, we can accomplish few-shot learning or even zero-shot learning as we’ll discuss in the next section.

### Prompt Engineering

Prompts are important for steering the behavior of large language models (LLMs) because they allow aligning the model outputs to human intentions without expensive retraining. Carefully engineered prompts can make LLMs suitable for a wide variety of tasks beyond what they were originally trained for. Prompts act as instructions that demonstrate to the LLM what is the desired input-output mapping. The picture below shows a few examples for prompting different language models (source: “Pre-train, Prompt, and Predict - A Systematic Survey of Prompting Methods in Natural Language Processing” by Liu and colleagues, 2021):

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file57.png" alt="Figure 8.5: Prompt examples, particularly knowledge probing in cloze form, and summarization." height="862" width="1178"><figcaption><p>Figure 8.5: Prompt examples, particularly knowledge probing in cloze form, and summarization.</p></figcaption></figure>

Prompt engineering, also known as in-context learning, refers to techniques for steering large language model (LLM) behavior through carefully designed prompts, without changing the model weights. The goal is to align the model outputs with human intentions for a given task. By designing good prompt templates, models can achieve strong results, sometimes comparable to fine-tuning. But how do good prompts look like?

#### Structure of Prompts

Prompts consist of three main components:

* Instructions that describe the task requirements, goals and format of inputs/outputs
* Examples that demonstrate the desired input-output pairs
* The input that the model must act on to generate the output

Instructions explain the task to the model unambiguously. Examples provide diverse demonstrations of how different inputs should map to outputs. The input is what the model must generalize to.Basic prompting methods include zero-shot prompting with just the input text, and few-shot prompting with a few demonstration examples showing desired input-output pairs. Researchers have identified biases like majority label bias and recency bias that contribute to variability in few-shot performance. Careful prompt design through example selection, ordering, and formatting can help mitigate these issues.More advanced prompting techniques include instruction prompting, where the task requirements are described explicitly rather than just demonstrated. Self-consistency sampling generates multiple outputs and selects the one that aligns best with the examples. Chain-of-thought (CoT) prompting generates explicit reasoning steps leading to the final output. This is especially beneficial for complex reasoning tasks. CoT prompts can be manually written or generated automatically via methods like augment-prune-select.This table gives a brief overview of a few methods of prompting compared to fine-tuning:

| **Technique**                 | **Method**                                                                                                | **Key Idea**                                                                      | **Results**                                              |
| ----------------------------- | --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------- |
| Fine-tuning                   | Fine-tune on explanation dataset generated via prompting                                                  | Improves model's reasoning abilities                                              | 73% accuracy on commonsense QA dataset                   |
| Zero-shot prompting           | Simply feeding the task text to the model and asking for results.                                         | Text: "i'll bet the video game is a lot more fun than the film."\<br>- Sentiment: |                                                          |
| Chain-of-Thought (CoT)        | Prefix responses with "Let's think step by step"                                                          | Gives model space to reason before answering                                      | Quadrupled accuracy on math dataset                      |
| Few-shot prompting            | Provide few demos consisting of input and desired output to help the model understand                     | Shows desired reasoning format                                                    | Tripled accuracy on grade school math                    |
| Least-to-most prompting       | Prompt model for simpler subtasks to solve incrementally. "To solve {question}, we need to first solve: " | Decomposes problems into smaller pieces                                           | Boosted accuracy from 16% to 99.7% on some tasks         |
| Selection-inference prompting | Alternate selection and inference prompts                                                                 | Guides model through reasoning steps                                              | Lifts performance on long reasoning tasks                |
| Self-consistency              | Pick most frequent answer from multiple samples                                                           | Increases redundancy                                                              | Gained 1-24 percentage points across benchmarks          |
| Verifiers                     | Train separate model to evaluate responses                                                                | Filters out incorrect responses                                                   | Lifted grade school math accuracy \~20 percentage points |

Figure 8.6: Pronpting techniques for LLMs.

Some prompting techniques incorporate external information retrieval to provide missing context to the LLM before generating the output. For open-domain QA, relevant paragraphs can be retrieved via search engines and incorporated into the prompt. For closed-book QA, few-shot examples with evidence-question-answer format work better than question-answer format.There are different techniques to improve the reliability of large language models (LLMs) in complex reasoning tasks:

1. Prompting and Explanation: prompting the model to explain its reasoning step-by-step before answering using prompts like "Let's think step by step" (as in CoT) significantly improves accuracy in reasoning tasks.
2. Providing few-shot examples of reasoning chains helps demonstrate the desired format and guides LLMs in generating coherent explanations.
3. Alternate Selection and Inference Prompts: Utilizing a combination of specialized selection prompts (narrow down the answer space) and inference prompts (generate the final response) leads to better results compared to generic reasoning prompts alone.
4. Problem Decomposition: Breaking down complex problems into smaller subtasks or components using a least-to-most prompting approach helps improve reliability, as it allows for a more structured and manageable problem-solving process.
5. Sampling Multiple Responses: Sampling multiple responses from LLMs during generation and picking the most common answer increases consistency, reducing reliance on a single output. In particular, training separate verifier models that evaluate candidate responses generated by LLMs helps filter out incorrect or unreliable answers, improving overall reliability.

Finally, fine-tuning LLMs on explanation datasets generated through prompting enhances their performance and reliability in reasoning tasks

> **Few-shot learning** presents the LLM with just a few input-output examples relevant to the task, without explicit instructions. This allows the model to infer the intentions and goals purely from demonstrations. Carefully selected, ordered and formatted examples can greatly improve the model's inference abilities. However, few shot learning can be prone to biases and variability across trials. Adding explicit instructions can make the intentions more transparent to the model and improve robustness. Overall, prompts combine the strengths of instructions and examples to maximize steering of the LLM for the task at hand.

Instead of hand-engineering prompts, methods like automatic prompt tuning learn optimal prompts by directly optimizing prefix tokens on the embedding space. The goal is to increase the likelihood of desired outputs given inputs. Overall, prompt engineering is an active area of research for aligning large pre-trained LLMs with human intentions for a wide variety of tasks. Careful prompt design can steer models without expensive retraining.In this section, we’ll go through a few (but not all) of the techniques mentioned beforehand. Let’s discuss the tools that LangChain provides tools to create prompt templates in Python!

#### Templating

Prompts are the instructions and examples we provide to language models to steer their behavior. Prompt templating refers to creating reusable templates for prompts that can be configured with different parameters.LangChain provides tools to create prompt templates in Python. Templates allow prompts to be dynamically generated with variable input. We can create a basic prompt template like this:

```
from langchain import PromptTemplate
prompt = PromptTemplate("Tell me a {adjective} joke about {topic}")
```

This template has two input variables - {adjective} and {topic}. We can format these with values:

```
prompt.format(adjective="funny", topic="chickens")
# Output: "Tell me a funny joke about chickens"
```

The template format defaults to Python f-strings, but Jinja2 is also supported.Prompt templates can be composed into pipelines, where the output of one template is passed as input to the next. This allows modular reuse.

**Chat Prompt Templates**

For conversational agents, we need chat prompt templates:

```
from langchain.prompts import ChatPromptTemplate 
template = ChatPromptTemplate.from_messages([
  ("human", "Hello, how are you?"),
  ("ai", "I am doing great, thanks!"),
  ("human", "{user_input}"),
])
template.format_messages(user_input="What is your name?")
```

This formats a list of chat messages instead of a string. This can be useful for taking the history of a conversation into account. We’ve looked at different memory methods in Chapter 5. These are similarly relevant in this context to make sure model outputs are relevant and on point.Prompt templating enables reusable, configurable prompts. LangChain provides a Python API for conveniently creating templates and formatting them dynamically. Templates can be composed into pipelines for modularity. Advanced prompt engineering can further optimize prompting.

#### Advanced Prompt Engineering

LangChain provides tools to enable advanced prompt engineering strategies like few-shot learning, dynamic example selection, and chained reasoning.

**Few-Shot Learning**

The `FewShotPromptTemplate` allows showing the model just a few demonstration examples of the task to prime it, without explicit instructions. For instance:

```
from langchain.prompts import FewShotPromptTemplate, PromptTemplate 
example_prompt = PromptTemplate("{input} -> {output}")
examples = [
  {"input": "2+2", "output": "4"},
  {"input": "3+3", "output": "6"}
]
prompt = FewShotPromptTemplate(
  examples=examples,
  example_prompt=example_prompt
)
```

The model must infer what to do from the examples alone.

**Dynamic Example Selection**

To choose examples tailored to each input, `FewShotPromptTemplate` can accept an `ExampleSelector` rather than hardcoded examples:

```
from langchain.prompts import SemanticSimilarityExampleSelector
selector = SemanticSimilarityExampleSelector(...) 
prompt = FewShotPromptTemplate(
   example_selector=selector,
   example_prompt=example_prompt
)
```

`ExampleSelector` implementations like `SemanticSimilarityExampleSelector` automatically find the most relevant examples for each input.

**Chained Reasoning**

When asking an LLM to reason through a problem, it is often more effective to have it explain its reasoning before stating the final answer. For example:

```
from langchain.prompts import PromptTemplate
reasoning_prompt = "Explain your reasoning step-by-step. Finally, state the answer: {question}"
prompt = PromptTemplate(
  reasoning_prompt=reasoning_prompt,
  input_variables=["questions"]
)
```

This encourages the LLM to logically think through the problem first, rather than just guessing the answer and trying to justify it after. This is called **Zero-Shot Chain of Thought**. Asking an LLM to explain its thought process aligns well with its core capabilities.**Few-Shot Chain of Thought** prompting is a few-shot prompt, where the reasoning is explained as part of the example solutions, with the idea to encourage an LLM to explain its reasoning before making a decision. It has been shown that this kind of prompting can lead to more accurate results, however, this performance boost was found to be proportional to the size of the model, and the improvements seemed to be negligible or even negative in smaller models.In **Tree of Thoughts (ToT)** prompting, we are generating multiple problem-solving steps or approaches for a given prompt and then using the AI model to critique these steps. The critique will be based on the model’s judgment of the solution’s suitability to the problem. Let's walk through a more detailed example of implementing ToT using LangChain.First, we'll define our 4 chain components with `PromptTemplates`. We need a solution template, an evaluation template, a reasoning template, and a ranking template. Let’s first generate solutions:

```
solutions_template = """
Generate {num_solutions} distinct solutions for {problem}. Consider factors like {factors}.
Solutions:
"""
solutions_prompt = PromptTemplate(
   template=solutions_template,
   input_variables=["problem", "factors", "num_solutions"]
)
```

Let’s ask the LLM to evaluate these solutions:

```
evaluation_template = """
Evaluate each solution in {solutions} by analyzing pros, cons, feasibility, and probability of success.
Evaluations:
"""
evaluation_prompt = PromptTemplate(
  template=evaluation_template,
  input_variables=["solutions"]  
)
```

Now we’ll reason a bit more about them:

```
reasoning_template = """
For the most promising solutions in {evaluations}, explain scenarios, implementation strategies, partnerships needed, and handling potential obstacles. 
Enhanced Reasoning: 
"""
reasoning_prompt = PromptTemplate(
  template=reasoning_template,
  input_variables=["evaluations"]
)
```

Finally, we can rank these solutions given our reasoning so far:

```
ranking_template = """
Based on the evaluations and reasoning, rank the solutions in {enhanced_reasoning} from most to least promising.
Ranked Solutions:
"""
ranking_prompt = PromptTemplate(
  template=ranking_template, 
  input_variables=["enhanced_reasoning"]
)
```

Next, we create chains from these templates before we’ll put it all together:

```
chain1 = LLMChain(
   llm=SomeLLM(),
   prompt=solutions_prompt,
   output_key="solutions"  
)
chain2 = LLMChain(
   llm=SomeLLM(),
   prompt=evaluation_prompt,
   output_key="evaluations"
)
```

Finally. we connect these chains into a `SequentialChain`:

```
tot_chain = SequentialChain(
   chains=[chain1, chain2, chain3, chain4],
   input_variables=["problem", "factors", "num_solutions"], 
   output_variables=["ranked_solutions"]
)
tot_chain.run(
   problem="Prompt engineering",
   factors="Requirements for high task performance, low token use, and few calls to the LLM",
   num_solutions=3
)
```

This allows us to leverage the LLM at each stage of the reasoning process. The ToT approach helps avoid dead-ends by fostering exploration.These techniques collectively enhance the accuracy, consistency, and reliability of large language models' reasoning capabilities on complex tasks by providing clearer instructions, fine-tuning with targeted data, employing problem breakdown strategies, incorporating diverse sampling approaches, integrating verification mechanisms, and adopting probabilistic modeling frameworks.Prompt design is highly significant for unlocking LLM reasoning capabilities, the potential for future advancements in models and prompting techniques, and these principles and techniques form a valuable toolkit for researchers and practitioners working with large language models.Let’s summarize!

### Summary

In Chapter 1, we discussed the basic principles of generative models, particularly LLMs, and their training. We focused mostly on the pre-training step, which is – generally speaking – adjusting the models to the correlations within words and wider segments of texts. Alignment it the assessment of model outputs against expectations and conditioning is the process of making sure the output is according to expectations. Conditioning allows steering generative AI to improve safety and quality, but it is not a complete solution. In this chapter, the focus is on conditioning, in particular through fine-tuning and prompting. In fine-tuning the language model is trained on many examples of tasks formulated as natural language instructions, along with appropriate responses. Often this is done through reinforcement learning with human feedback (RLHF), which involves training on a dataset of human-generated (prompt, response) pairs, followed by reinforcement learning from human feedback, however, other techniques have been developed that have been shown to produce competitive results with lower resource footprints. In the first recipe of this chapter, we’ve implemented a fine-tuning of a small open-source model for question answering.There are many techniques to improve the reliability of LLMs in complex reasoning tasks including step-by-step prompting, alternate selection, and inference prompts, problem decomposition, sampling multiple responses, and employing separate verifier models. These methods have shown to enhance accuracy and consistency in reasoning tasks. We’ve discussed and compared several techniques. LangChain provides building blocks to unlock advanced prompting strategies like few-shot learning, dynamic example selection, and chained reasoning decomposition as we’ve shown in the examples.Careful prompt engineering is key to aligning language models with complex objectives. Reliability in reasoning can be improved by breaking down problems and adding redundancy. The principles and techniques that we’ve discussed in this chapter provide a toolkit for experts working with LLMs. We can expect future advancements in both model training and prompting techniques. As these methods and LLMs continue to develop, they will likely become even more effective and useful for a broader range of applications.Let’s see if you remember some more key points from this chapter!

### Questions

Please have a look to see if you can come up with the answers to these questions from memory. I’d recommend you go back to the corresponding sections of this chapter, if you are unsure about any of them:

1. What’s alignment in the context of LLMs?
2. What are different methods of conditioning and how can we distinguish them?
3. How’s moderation related to conditioning?
4. What is instruction tuning and what’s its importance?
5. What is quantization?
6. What are a few methods for fine-tuning?
7. What is few-shot learning?
8. What is Chain of Thought prompting?
9. Explain Tree of Thought prompting!
