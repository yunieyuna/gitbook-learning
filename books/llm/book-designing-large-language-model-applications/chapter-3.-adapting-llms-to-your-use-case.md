# Chapter 3. Adapting LLMs To Your Use Case

## Chapter 3. Adapting LLMs To Your Use Case

## A NOTE FOR EARLY RELEASE READERS

With Early Release ebooks, you get books in their earliest form—the author’s raw and unedited content as they write—so you can take advantage of these technologies long before the official release of these titles.

This will be the 5th chapter of the final book. Please note that the GitHub repo will be made active later on.

If you have comments about how we might improve the content and/or examples in this book, or if you notice missing material within this chapter, please reach out to the author at [mcronin@oreilly.com](mailto:mcronin@oreilly.com).

In this chapter, we will continue with our journey through the LLM landscape, exploring the various LLMs available for commercial use and provide pointers on how to choose the right LLM for your task. We will also examine how to load LLMs of various sizes and run inference on them. We will then decipher various decoding strategies for text generation. We will also investigate how to interpret the outputs and intermediate results from language models, surveying various interpretability tools including LIT-NLP.

## Navigating the LLM Landscape

Seemingly there is a new LLM being released every few days, many of them claiming to be state-of-the-art. Most of these LLMs are not too different from each other, so you need not necessarily spend too much time tracking new LLM releases. The accompanying Github repo to this book attempts to keep a track of the major releases [here](https://github.com/piesauce/llm-playbooks), but I don’t promise it will be complete.

Nevertheless, it is a good idea to have a broad understanding of the different types of LLM providers out there, the kinds of LLMs being made available, and the copyright and licensing implications. Therefore, let’s now explore the LLM landscape from this lens and understand the choices at our disposal.

### Who are the LLM providers?

LLM providers can be broadly categorized into the following types:

* **Companies providing proprietary LLMs**: These include companies like Open AI ([GPT](https://openai.com/product)), Google ([PaLM](https://developers.generativeai.google/products/palm)), Anthropic ([Claude](https://www.anthropic.com/product)), [Cohere](https://cohere.com/models/command), [AI21](https://www.ai21.com/studio) etc. who train proprietary LLMs and make them available as an API endpoint (LLM-as-a-service). Many of these companies have also partnered with cloud providers who facilitate access to these models as a fully managed service. The relevant offerings from the major cloud providers are [AWS Bedrock](https://aws.amazon.com/bedrock/) and [Sagemaker JumpStart](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-jumpstart.html) by Amazon, [Vertex AI](https://cloud.google.com/model-garden) by Google, and [Azure Open AI](https://azure.microsoft.com/en-us/products/ai-services/openai-service) by Microsoft.
* **Companies providing open-source LLMs**: These include companies who make the LLM weights public and monetize through providing deployment services ([Together AI](https://together.ai/)), companies whose primary business would benefit from more LLM adoption ([Cerebras](https://www.cerebras.net/)), and research labs who have been releasing LLMs since the early days of Transformers (Microsoft, Google, Meta, Salesforce, etc.). Note that companies like Google have released both propreitary and open-source LLMs.
* **Self-organizing open-source collectives and community research organizations**: This includes the pioneering community research organization [Eleuther AI](https://www.eleuther.ai/), and [Big Science](https://bigscience.huggingface.co/). These organizations rely on donations for compute infrastructure.
* **Academia and government**: Due to the high capital costs, not many LLMs have come out of academia so far. Examples of LLMs from government/academia include the Abu Dhabi government funded [Technology Innovation Institute](https://www.tii.ae/), which released the [Falcon](https://falconllm.tii.ae/falcon-models.html) model, and Tsinghua University, which released the [GLM](http://keg.cs.tsinghua.edu.cn/glm-130b/posts/glm-130b/) model.

Note that open-source models exist on a continuum in terms of the permissibility of their licenses. Models can be as restrictive as being allowed to be used for academic purposes only, all the way up to being fully permitted for commercial use without attribution.

[Table 3-1](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch03.html#llm-provider-categories) shows the various players in the LLM space, the category of entity they belong to, and the various pre-trained models they have published.

| Name                            | Category                         | Pre-trained Models Released                                                                              |
| ------------------------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Google                          | Company                          | BERT, MobileBERT, T5, Flan-T5, ByT5, Canine, UL2, Flan-UL2, Pegasus PaLM, PaLMV2, ELECTRA, Tapas, Switch |
| Microsoft                       | Company                          | DeBERTa, DialoGPT, BioGPT, MPNet                                                                         |
| Open AI                         | Company                          | GPT-2, GPT-3, GPT-3.5, GPT-4                                                                             |
| Amazon                          | Company                          | Titan                                                                                                    |
| Anthropic                       | Company                          | Claude, Claude-2                                                                                         |
| Cohere                          | Company                          | Cohere Command, Cohere Base                                                                              |
| Meta                            | Company                          | RoBERTa, Llama, Llama2, BART, OPT, Galactica                                                             |
| Salesforce                      | Company                          | CTRL, Xgen, EinsteinGPT                                                                                  |
| MosaicML                        | Company (Acquired by Databricks) | MPT                                                                                                      |
| Cerebras                        | Company                          | Cerebras-GPT, BTLM                                                                                       |
| Databricks                      | Company                          | Dolly-V1, Dolly-V2                                                                                       |
| Stability AI                    | Company                          | StableLM                                                                                                 |
| Together AI                     | Company                          | RedPajama                                                                                                |
| Ontocord AI                     | Non-profit                       | MDEL                                                                                                     |
| Eleuther AI                     | Non-profit                       | Pythia, GPT-Neo, GPT Neo-X, GPT-J                                                                        |
| Big Science                     | Non-profit                       | BLOOM                                                                                                    |
| Tsinghua University             | Academic                         | GLM                                                                                                      |
| Technology Innovation Institute | Academic                         | Falcon                                                                                                   |
| UC Berkeley                     | Academic                         | OpenLlaMA                                                                                                |
| Adept AI                        | Company                          | Persimmon                                                                                                |
| Mistral AI                      | Company                          | Mistral                                                                                                  |
| AI21 Labs                       | Company                          | Jurassic                                                                                                 |
| X.AI                            | Company                          | Grok                                                                                                     |

### Model flavors

Each model is usually released with multiple variants. It is customary to release different-sized variants of the same model. As an example, Llama2 comes in 7B, 13B, and 70B sizes, where these numbers refer to the number of parameters in the model.

These days, LLM providers augment their pre-trained models in various ways to make them more amenable to user tasks. The augmentation process typically involves fine-tuning the model in some way, often incorporating human supervision. Some of these fine-tuning exercises can cost millions of dollars in terms of human annotations. We will refer to pre-trained models that have not undergone any augmentation as base models.

Here are some of the popular augmentation types:

#### Instruct-models

Instruct-models, or Instruction-tuned models, are specialized in following instructions written in natural language. While base models possess powerful capabilities, they are akin to a rebellious teenager; effectively interacting with them is possible only after tediously engineering the right prompts through trial-and-error, which tend to be brittle. This is because the base models are trained on either denoising objectives or next-word prediction objectives, which is different from the tasks users typically want to solve. By instruction-tuning the base model, the resulting model is able to more effectively respond to human instructions and be helpful.

A typical instruction-tuning dataset consists of a diverse set of tasks expressed in natural language, along with input-output pairs. In Chapter 6, we will explore various techniques to construct instruction-tuning datasets, and demonstrate how to perform instruction-tuning on a model.

Here is an example from a popular instruction-tuning dataset called [FLAN](https://github.com/google-research/FLAN/tree/main/flan/v2).

Input:

“What is the sentiment of the following review? The pizza was ok but the service was terrible. I stopped in for a quick lunch and got the slice special but it ended up taking an hour after waiting several minutes for someone at the front counter and then again for the slices. The place was empty other than myself, yet I couldn’t get any help/service. OPTIONS: - negative - positive”

Target:

“Negative”

In this example, the input consists of an instruction ‘What is the sentiment of the following review’ expressed in a way that humans would naturally express, along with the input and output. The input is the actual review and the output is the solution to the task, either generated by a model or annotated by a human.

[Figure 3-1](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch03.html#instruction-tuning1) demonstrates the instruction-tuning process

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/figure521.png" alt="Instruction tuning process" height="140" width="600"><figcaption></figcaption></figure>

**Figure 3-1. Instruction-tuning process**

This form of fine-tuning is also called Supervised Fine-tuning (SFT). In addition to improving the ability of a model to respond effectively to user tasks, SFT-based approaches can also be used to make it less harmful, by training on safety datasets that help align model outputs with the values and preferences of the model creators.

More advanced techniques to achieve this alignment include reinforcement learning-based methods like RLHF(Reinforcement Learning from Human Feedback) and RLAIF (Reinforcement Learning from AI Feedback).

In RLHF training, human annotators select or rank candidate outputs based on certain criteria, like helpfulness and harmlessness. These annotations are used to iteratively train a reward model which ultimately leads to the LLM being more controllable, for example, by refusing to answer inappropriate requests from users.

[Figure 3-2](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch03.html#rlhf-1) shows the RLHF training process.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/figure522.png" alt="RLHF" height="175" width="600"><figcaption></figcaption></figure>

**Figure 3-2. Reinforcement Learning from Human Feedback**

We will cover RLHF in detail in Chapter 6, including algorithms like PPO (Proximal Policy Optimization) and Rejection Sampling, as well as pointers on how to facilitate the human feedback process.

Instead of human feedback, one can also leverage LLMs to choose between outputs based on their adherence to a set of principles (don’t be racist, don’t be rude etc). This technique was introduced by Anthropic and is called RLAIF. In this technique, humans only provide a desired set of principles and values (referred to as Constitutional AI), and the LLM is tasked with determining whether its outputs adhere to these principles.

Examples of instruction-tuned models include Open AI’s GPT-3.5-turbo-instruct, Cohere’s Command model, MPT-Instruct, RedPajama-Instruct etc.

#### Chat-models

Chat-models are a type of instruction-tuned models that are optimized for multi-turn dialog. Examples include ChatGPT, Llama2-Chat, MPT-Chat, OpenAssistant etc. In Chapter 6 we will discuss how to generate and structure dialog datasets including the ChatML format used by many models.

#### Long-context models

As discussed in Chapter 2, Transformer-based LLMs have a limited context length. To recap, context length typically refers to the sum of the number of input and output tokens processed by the model per invocation. Typical context lengths of modern LLMs range from 2,000 to 8,000 tokens, with some models like Anthopic’s Claude 2 supporting as much as 100,000 tokens. Some models are released with a long-context variant - for example GPT 3.5 comes with a default 4k context size but also has a 16k context size variant. [MPT](https://huggingface.co/mosaicml/mpt-7b-storywriter) also has a long-context variant that has been trained on 65k context length but can potentially be used for even longer contexts during inference.

**TIP**

There is no free lunch with long-context models. It has been [shown](https://arxiv.org/abs/2307.03172) that recall is negatively impacted with longer context. LLMs tend to forget things in the middle of the context window. This is because of the characteristics of the documents that LLMs are trained on, wherein the most relevant context of a document necessary to predict the next token is more often found near the beginning or end of the context. In my experiments, I have noticed that 3k context size is the tipping point beyond which performance starts to degrade. You also can’t just stuff your entire context with instructions - LLMs can only handle a limited set of instructions in a prompt beyond which performance drops.

#### Domain-adapted or task-adapted models

LLM providers also might perform fine-tuning on specific tasks like summarization or financial sentiment analysis. They may also produce distilled versions of the model, where a smaller model is fine-tuned on outputs from the larger model for a particular task. Examples include [FinBERT](https://huggingface.co/ProsusAI/finbert), which is fine-tuned on financial sentiment analysis datasets, and [UniversalNER](https://universal-ner.github.io/), which is distilled using named-entity-recognition data.

[Table 3-2](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch03.html#llm-taxonomy) shows the various LLMs available, the licenses under which they are published, their pricing, the sizes they are available in, and the flavors in which they are available. Note that the LLM may be instruction-tuned or chat-tuned by a different entity than the one that pre-trained the LLM.

| Name           | Availability                   | Pricing                                                                                         | Sizes                                   | Variants                                         |
| -------------- | ------------------------------ | ----------------------------------------------------------------------------------------------- | --------------------------------------- | ------------------------------------------------ |
| GPT-4          | Propreitary                    | $0.03 per 1000 input tokens,$0.06 per 1000 output tokens. Price of 32K context model is double. | Unknown                                 | GPT-4 32K context, GPT-4 8K context              |
| GPT-3.5 Turbo  | Propreitary                    | $0.0015 per 1000 input tokens,$0.002 per 1000 output tokens.                                    | Unknown                                 | GPT-3.5 4k context, GPT-3.5 16K context          |
| Claude Instant | Propreitary                    | $1.63 per million input tokens, $5.51 per million output tokens                                 | Unknown                                 | -                                                |
| Claude2        | Propreitary                    | $11.02 per million input tokens, $32.68 per million output tokens                               | Unknown                                 | -                                                |
| MPT            | Open-source, commercial use    | Free                                                                                            | 1B, 7B, 30B                             | MPT 65K storywriter                              |
| CerebrasGPT    | Open-source, commercial use    | Free                                                                                            | 111M, 256M, 590M, 1.3B, 2.7B, 6.7B, 13B | CerebrasGPT                                      |
| Stability LM   | Open-source, commercial use    | Free                                                                                            | Common Crawl                            | 7                                                |
| Red Pajama     | Open-source, commercial use    | Free                                                                                            | 3B, 7B                                  | RedPajama-INCITE-Instruct, RedPajama-INCITE-Chat |
| GPT Neo-X      | Open-source, commercial use    | Free                                                                                            | 20B                                     | -                                                |
| BLOOM          | Open-source, restricted use    | Free                                                                                            | 176B                                    | BLOOMZ                                           |
| LLama          | Open-source, no commercial use | Free                                                                                            | 7B, 13B, 33B, 65B                       | -                                                |
| LLama2         | Open-source, commercial use    | Free                                                                                            | 7B, 13B, 70B                            | Llama2-Chat                                      |

## INSTRUCTION TUNING CAN HAVE SIDE EFFECTS

Is it beneficial to always prefer using an instruction-tuned variant over the base model for your tasks? In most cases, yes. However, keep in mind that any tuning on top of a base model inevitably causes some regressions, thus losing access to some of the capabilities possessed by the base model.

An example of this was demonstrated by [Chung et al.](https://arxiv.org/abs/2210.11416) They noticed that instruction-tuning using the FLAN dataset worsened chain-of-thought capabilities (explained later in this chapter), which are crucial for reasoning tasks. However, they also observed that adding chain-of-thought data to their instruction-tuning datasets increased the reasoning capabilities of the model compared to the base variant.

The side effects of instruction-tuning are not well explored, so it is a good idea to experiment with the base model and see if you are losing out on any capabilities.

Similarly, RLHF-tuned models are calibrated to respond to user queries in accordance with the principles, values, and ethics of the LLM provider. These may not be the same values that you or your organization hold.

In all these cases you can perform your own instruction-tuning and RLHF training on the base model, details of which we will explore in Chapter 6. In that chapter, we will also analyze when it is worthwhile to perform your own instruction-tuning/RLHF.

## How to choose an LLM for your task

Given the plethora of options available, how do you ensure you choose the right LLM for your task? Depending on your situation, there are a multitude of criteria to consider, including

* Cost - This includes not only inference costs, but also engineering costs associated with maintenance, monitoring, optimization etc (collectively termed as LLMOps).
* Time Per Output Token([TPOT](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)) - This is a metric used to measure the speed of text generation as experienced by the end user.
* Time To First Token - This is a metric that measures the time it takes for the user to view the first output token after they issue a query. This includes the time taken by the LLM to process the input and generate the first token. The importance of this metric depends on whether you intend to interact with the LLM in real-time (rather than processing requests in bulk batches).
* Task performance - How stringent the performance requirements are. For instance, is it worth spending extra resources to move up accuracy from 93.5 to 93.9?
* Type of tasks - The nature of the tasks the LLM will be used for, like summarization, question answering, classification etc.
* Capabilities required - Examples of capabilities include arithmetic reasoning, logical reasoning, planning, task decomposition etc. A lot of these capabilities, to the extent that they actually exist or approximate, are _emergent properties_ of an LLM as discussed in Chapter 1, and are not exhibited by smaller sized models.
* Licensing - You can use only those models that allow your mode of usage. Even models that explicitly allow commercial use can have restrictions on certain types of use cases. For example, the BigScience Open RAIL-M license restricts the usage of the LLM in use cases pertaining to law enforcement, immigration or asylum processes etc.
* MLOps bandwidth - Whether you have adequate engineering/MLOps bandwidth in your team.
* In-house ML talent - The degree of in-house ML talent also determines how much customization you are able to afford.
* Other non-functional criteria - This includes safety, security, privacy etc. Cloud providers and startups are already implementing solutions that can address these issues.

[Figure 3-3](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch03.html#choose-llm) shows a flow chart that illustrates how these critieria interact with each other and how you can make a decision regarding the kind of LLM you might want to choose for your task.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/figure107.png" alt="Flowchart for choosing an LLM" height="119" width="600"><figcaption></figcaption></figure>

**Figure 3-3. Flowchart for choosing an LLM**

### Open-source vs Proprietary LLMs

The open-source vs propreitary debate has been going on in the field of software for several decades now, and we are seeing it become relevant in the field of LLMs as well. The biggest advantage of open-source models are the transparency and flexibility they provide, and not necessarily the cost. Self-hosting open-source LLMs can incur a lot of engineering overhead and compute/memory costs, and using managed services might not always be able to match propreitary models in terms of latency, throughput, and inference cost. Moreover, many open-source LLMs are not easily accessible through managed services and other third-party deployment options. This situation is bound to change dramatically as the field matures, but in the meanwhile, run through your calculations for your specific situation to determine the costs incurred for using each (type of) model.

The flexibility provided by open-source models helps with debuggability, interpretability, and the ability to augment the LLM with any kind of training/fine-tuning you choose, instead of the restricted avenues made available by the LLM provider. This allows you to more substantially align the LLM towards your preferences and values instead of the ones decided by the LLM provider.

Not all open-source models are fully transparent. Several for-profit companies that release open-source LLMs do not make the training datasets public. For instance, Meta hasn’t disclosed all the details of the training datasets used to train the Llama2 model. Knowing which datasets are used to train the model can help you assess whether there is test set contamination, and understand what kind of knowledge you can expect the LLM to possess.

As of this book’s writing, propreitary LLMs like GPT-4 represent the state-of-the-art and haven’t been matched by any open-source counterpart yet.

**TIP**

Always check if the model provider has a active developer community on Github/Discord/Slack, and that the development team is actively engaged in those channels, responding to user comments and questions. I recommend preferring models with active developer communities, provided they satisfy your primary criteria.

## IS GPT-4 GETTING WORSE OVER TIME?

Is GPT-4 getting worse over time? This question feels like the _Which color is this dress?_ question.[1](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch03.html#id158) There are a lot of developers who swear by their experience of noticing quality degradation of GPT-4. However, opponents of this theory suggest that this is just a perception, as the novelty of GPT-4 washes over.

So what is the truth? Firstly, note that as discussed in Chapter 1, capabilities and behavior are two separate concepts. The behavior of the LLM is influenced by the prompting strategy used. Unfortunately, when models get updated, the previously optimized prompts may not be optimal anymore. This phenomenon is called prompt drift. Therefore, while the capabilities of the LLM might have remained the same or even improved, using prompts that are not optimized for the new model causes a degradation in behavior.

Secondly, any kind of training/fine-tuning over an existing model comes with side effects. It is impossible to update a model in a way such that the updated version is strictly better than the original version for every possible input.

The hope is that LLM players update models transparently, and allow users access to the older version of the models, at least for a grace period.

### LLM Evaluation

In order to evaluate LLMs on their task performance, there exist a lot of benchmark datasets that test a wide variety of skills. Not all skills are relevant to your use case, so you can choose to focus on specific benchmarks that test the skills you need the LLM to perform well on.

The leaderboard on these benchmark tests changes very often, especially if only open-source models are being evaluated, but that does not mean you need to change the LLMs you use every time there is a new leader on the leaderboard. Usually, the differences between the top models are quite marginal. The choice of LLM probably isn’t the most important criteria determining the success of your task, and you are better off spending that bandwidth working on understanding and cleaning your data which is still the most important component of the project.

Let’s now look at a few popular ways in which the field is evaluating LLMs.

#### Eleuther AI LM Evaluation Harness

Through the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), Eleuther AI supports benchmarking on over 400 different benchmark tasks, evaluating skills as varied as open-domain question answering, arithmetic and logical reasoning, linguistic tasks, machine translation, toxic language detection etc. You can use this tool to evaluate any model on the [HuggingFace Hub](https://huggingface.co/docs/hub/index), a platform containing thousands of pre-trained and fine-tuned models, on the benchmarks of your choice.

Here is an example from one of the benchmark tasks called _bigbench\_formal\_fallacies\_syllogisms\_negation_.

```
 {
            "input": "\"Some football fans admire various clubs, others love

            only a single team. But who is a fan of whom precisely? The

            following argument pertains to this question: First premise: Mario

            is a friend of FK \u017dalgiris Vilnius. Second premise: Being a

            follower of F.C. Copenhagen is necessary for being a friend of FK

            \u017dalgiris Vilnius. It follows that Mario is a follower of F.C.

            Copenhagen.\"\n Is the argument, given the explicitly stated

            premises, deductively valid or invalid?",

            "target_scores": {

                "valid": 1,

                "invalid": 0
            }
```

In this task, the model is asked to spot logical fallacies by deducing whether the presented argument is valid given the premises.

Let’s evaluate a few models on this task. Follow the instructions [here](https://github.com/EleutherAI/lm-evaluation-harness) to install the harness. Now, you can run the code

```
python main.py \
    --model hf-causal \
    --model_args pretrained=tiiuae/falcon-7b \
    --tasks bigbench_formal_fallacies_syllogisms_negation \
    --device cuda:0
```

Try this for a few other 7B models, including Llama2, MPT, Cerebras, Red Pajama, with both the base versions and the instruction-tuned versions. What do you find?

There is limited support for evaluation of closed-source models using this harness. For example, here is how you would evaluate Open AI models.

```
export OPENAI_API_SECRET_KEY=<Key>
python main.py \
    --model gpt3 \
    --model_args engine=gpt-3.5-turbo \
    --tasks bigbench_formal_fallacies_syllogisms_negation
```

How does GPT-3.5 compare to open-source models on this task?

#### HuggingFace Open LLM Leaderboard

The [Open LLM leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open\_llm\_leaderboard) uses Eleuther AI’s LM evaluation harness to evaluate the performance of models on 4 benchmark tasks. The 4 tasks are:

1. MMLU (Massive Multitask Language Understanding) - This test evaluates the LLM on knowledge-intensive tasks, drawing from fields like US history, biology, mathematics and more than 50 other subjects in a multiple choice framework.
2. ARC (AI2 Reasoning Challenge) - This test evaluates the LLM on multiple-choice grade school science questions, that need complex reasoning as well as world knowledge to answer them.
3. Hellaswag - This test evaluates commonsense reasoning by providing the LLM with a situation and asking it to predict what might happen next out of the given choices, based on commonsense.
4. TruthfulQA - This test evaluates the LLM’s ability to provide answers that don’t contain falsehoods.

[Figure 3-4](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch03.html#llm-leaderboard) shows a snapshot of the LLM leaderboard as of the day of the book’s writing. We can see that

* Larger models perform better.
* Instruction-tuned or fine-tuned models perform better.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/figure504.png" alt="Snapshot of the Open LLM Leaderboard" height="132" width="600"><figcaption></figcaption></figure>

**Figure 3-4. Snapshot of the Open LLM Leaderboard**

The validity of these benchmarks are in question as complete test set decontamination is not guaranteed. Model providers are also optimizing to solve these benchmarks, thus reducing the value of these benchmarks to serve as reliable estimators of general-purpose performance.

#### HELM (Holistic Evaluation of Language Models)

[HELM](https://crfm.stanford.edu/helm/latest/?groups=1) is an evaluation framework by Stanford that aims to calculate a wide variety of metrics over a range of benchmark tasks. 59 metrics are calculated overall, testing accuracy, calibration, robustness, fairness, bias, toxicity, efficiency, summarization performance, copyright infringement, disinformation, and more. The tasks tested include question answering, summarization, text classification, information retrieval, sentiment analysis, and toxicity detection.

[Figure 3-5](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch03.html#helm-leaderboard) shows a snapshot of the HELM leaderboard as of the day of the book’s writing. We can see that for a given task, the leaders differ across different evaluation criteria (efficiency, bias, accuracy etc.)

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/figure505.png" alt="Snapshot of the HELM Leaderboard" height="105" width="600"><figcaption></figcaption></figure>

**Figure 3-5. Snapshot of the HELM Leaderboard**

## BENCHMARK EVALUATION IS UNRELIABLE

There are multiple ways in which you can evaluate the same task. For example, consider the MMLU task. Questions in the MMLU task have four choices as answers - A, B, C, D. How do we evaluate performance on a multiple-choice question answering task?

1. You can pick the token that has the highest output probability out of the four options (A, B, C, D)
2. You can pick the token that has the highest output probability from the entire vocabulary and use that to match it with the correct answer to the question (not the label).
3. You can produce a normalized sum of the probabilities of the token sequence generated by the model, where the expected token sequence is the label followed by the answer text, and use that to match it with the correct answer (label followed by answer text)

Each of these types of calculations can produce a vastly different result, and can lead to different leaders in the leaderboard. [HuggingFace](https://huggingface.co/blog/evaluating-mmlu-leaderboard) published a blog post about this after people noticed discrepancies in their numbers versus third-party evaluations.

#### ELo Rating

Now that we have seen the limitations of quantitative evaluation, let us explore how we can most effectively incorporate human evaluations. One promising framework is the [Elo Rating](https://en.wikipedia.org/wiki/Elo\_rating\_system) system, used in chess to rank players.

[LMSYS ORG](https://lmsys.org/) (Large Model Systems Organization) has implemented an evaluation platform based on the ELo rating system called the [Chatbot Arena](https://chat.lmsys.org/?arena). Chatbot Arena solicits crowdsourced evaluations by inviting people to choose between two randomized and anonymized LLMs by chatting with them side-by-side. The leaderboard is found [here](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard), with models like GPT-4 and Claude holding a clear advantage over the rest.

[Figure 3-6](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch03.html#chatbotarena-leaderboard) shows a snapshot of the Chatbot Arena leaderboard as of the day of the book’s writing. We can see that propreitary models by Open AI and Anthropic dominate the rankings, followed by chat-tuned models like Vicuna and Guanaco.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/figure506.png" alt="Snapshot of the Chatbot Arena Leaderboard" height="344" width="600"><figcaption></figcaption></figure>

**Figure 3-6. Snapshot of the Chatbot Arena Leaderboard**

## ELO RATINGS CAN BE BIASED TOO

ELo ratings are not a panacea to the problem of generating holistic evaluations. Human biases can meaningfully impact the overall ratings even if the LLMs are being evaluated in an anonymous manner.

According to [Wu et al.](https://arxiv.org/abs/2307.03025), these biases include

* Humans tend to prefer longer texts.
* Humans tend to overlook subtle factuality and consistency issues if the style is authoritative or convincing.
* Humans can be indecisive, and tend to grant ties instead of choosing a winner.
* The order in which the LLM answers are presented can influence human ratings. This can be rectified by providing the answers to the user in a randomized fashion.

Wu et al. propose a multi-ELo rating system that asks humans to evaluate the LLM across three different dimensions: helpfulness, accuracy, and language.

## INTERPRETING BENCHMARK RESULTS

How do you interpret evaluation results presented in research papers? Try to methodologically ask as many questions as possible, and see if the answers are covered in the paper or other material. As an example, let us take the Llama2-chat evaluation graphs presented in the [Llama2](https://arxiv.org/abs/2307.09288) paper. In particular, study Figure 1 and 3, which demonstrate how Llama2-chat compares with respect to helpfulness and safety against other chat models. Some of the questions that come to mind are:

1. How does the evaluation dataset look like? Do we have access to it?
2. What is the difficulty level of the test set? Maybe the model is competitive with respect to chatGPT for easier examples but how does it do with more difficult examples?
3. What proportion of examples in the test set can be considered difficult?
4. What are the kinds of scenarios covered in the test set? What degree of overlap do these scenarios have with the chat-tuning sets?
5. What definition do they use for safety?
6. Can there be a bias in the evaluation due to the fact that the models are evaluated on the basis of a particular definition of safety, which Llama2 was also trained to adhere with, while other models may have different definitions of safety?

Rigorously interrogating the results this way helps you develop a deeper understanding of what is being evaluated, and whether it is in alignment with the capabilities you need from the language model for your own tasks.

**WARNING**

Do not trust evaluations performed by GPT-4 or any other LLM. We have no idea what criteria it uses for evaluation nor do we know what biases it possesses.

Robust evaluation of LLMs is further complicated by the sensitivity of the prompts and the probabilistic nature of generative models. For example, I often see papers claiming that _GPT-4 does not have reasoning capabilities_, while not using any prompting techniques for evaluation. In many of these cases, it turns out that GPT-4 can in fact perform the task if prompted with chain-of-thought prompting. While evaluation prompts need not be heavily engineered, using rudimentary techniques like chain-of-thought should be standard practice and not using it means that the model capabilities are being underestimated.

## Accessing and loading LLMs

You can either access LLMs through APIs or by loading them yourself. Let us explore each of these modes in detail.

### Accessing Open AI LLMs

Let’s take Open AI GPT-3.5/GPT-4 as an example. Most other propreitary models expose similar parameters through their API.

GPT-3.5 and GPT-4 can be accessed through the Chat completion API. Here is an example:

```
import os
import openai
openai.api_key = <INSERT YOUR KEY HERE>

output = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are an expert storywriter."},
    {"role": "user", "content": "Write me a short children's story
    about a dog and an elephant stopping
    being friends with each other."}
  ]
)

print(output.choices[0].message)
```

Roles can be either system, user, or assistant, with assistant referring to the model responses. If you are are having a chat session with the LLM you will need to add the entirety of the conversation history in the _messages_ array in the form of a sequence of _user_ and _assistant_ messages.

**NOTE**

What is the difference between the system and user roles? Which instructions should go into the system prompt and which ones into the user prompt? System prompts are used for dictating the high-level overarching behavior of an LLM, like _You are a financial expert well versed in writing formal reports_. If you are allowing your users to directly interact with the LLM, then the system prompt can be used to provide your own instruction to the LLM along with the user request. In my experiments I have noticed that it doesn’t matter much if you place your instructions in the system prompt vs user prompt. What does matter is the length and size of your instruction. Instructions at the end of the prompt are more likely to be adhered to.

Here are some of the parameters made available by Open AI:

**n** - This refers to the number of generations the model has to make for each input. As an example, if we used n=5 in the given example, it would generate five different children’s stories.

**TIP**

For most tasks, I would advice generating multiple generations; i.e. n>1 and then using a postprocessing function (which could involve an LLM call) to choose the best one. This is because of the probabilistic nature of LLMs, where an answer might be wrong/bad just based on an unlucky token sampling. You might have to balance this process against your budget limitations.

**stop** and **max\_tokens** - These are used to limit the length of the generated output. _stop_ allows you to specify end tokens which if generated, would stop the generation process. An example stop sequence is the newline token. If you ask the model to adhere to a particular output format, like a numbered list of sentences, then in order to stop generating after a particular number of sentences have been output, you can just provide the number as a stop parameter.

**presence\_penalty** and **frequency\_penalty** - These are used to limit the repetitiveness of the generated output. By penalizing the logits of the language model outputs for tokens that have already appeared in the output so far, we can ensure that the model isn’t repeating the same topic repeatedly. These parameters can be used while performing more creative tasks.

**logit bias** - We have seen we can reduce the probability or prevent certain tokens from being generated. Can we do the opposite and make it more probable that some tokens will be generated? The _logit bias_ parameter can be used to do that. In fact, it is also able to reduce the probability of a token being generated, if you provide negative values to the parameter.

**top\_p** and **temperature** - Both these parameters relate to decoding strategies. Generative models produce a distribution of token probabilities, and will use these probabilities to generate the next token. There are many strategies to choose the next token to generate given the probabilities for each token. We will discuss them in detail later in the chapter.

### Loading LLMs and running inference on them

If you have access to GPUs, you can load LLMs in memory and run inference on them. Choosing a GPU depends on cost, the size of the model, whether you are training the model or just running inference, and support for optimizations. Tim Dettmers has developed a great [flowchart](https://i0.wp.com/timdettmers.com/wp-content/uploads/2023/01/gpu\_recommendations.png?ssl=1) that you can use to figure out which GPU best serves your needs.

Let’s figure out the amount of GPU RAM needed to load an LLM of a given size. LLMs can be loaded in various _precisions_:

1. Float32 - 32-bit floating point representation, each parameter occupies 4 bytes of storage
2. Float16 - 16-bit floating point representation. Only 5 bits are reserved for the exponent as opposed to 8 bits in Float32. This means that using Float16 comes with overflow/underflow problems for very large and small numbers.
3. bfloat16 (BF16) - 16-bit floating point representation. Just like Float32, 8 bits are reserved for the exponent, thus alleviating the underflow/overflow problems observed in Float16
4. Int8 - 8-bit integer representation. Running inference in 8-bit mode is around 20 percent slower than running in Float16
5. FP8, FP4 - 8-bit and 4-bit floating point representation.

We will explore these formats in detail in Chapter 9. Generally, running inference on a model with 7B parameters will need around 7GB of GPU RAM if running in 8-bit mode, and around 14GB if running in BF16. If you intend to fine-tune the whole model, you will need a lot more memory. We will discuss the memory requirements for fine-tuning models in Chapter 6.

### HuggingFace Accelerate

You can run inference on models even if they don’t fit in the GPU RAM. _accelerate_ library by HuggingFace facilitates this by loading parts of the model into CPU RAM if the GPU RAM is filled up, and then loading parts of the model into disk if the CPU RAM is also filled up. This [video](https://www.youtube.com/watch?v=MWCSGj9jEAo) shows how accelerate operates under the hood. This whole process is abstracted from the user, so all you need to load a large model is to run the following code:

```
!pip install transformers accelerate
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20B")
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neox-20B")


input_ids = tokenizer("Language models are", return_tensors="pt")
gen_tokens = model.generate(**input_ids, max_new_tokens =1)
```

## Decoding strategies

Now that we have learned how to load and run inference on a model, let’s understand how to effectively generate text in the autoregressive setting. Several _decoding_ strategies have been devised in the past few years. Let’s go through them in detail.

### Greedy decoding

The simplest form of decoding is to just generate the token that has the highest probability. The drawback of this approach is that it causes repetitiveness in the output. Here is an example

```
input = tokenizer('The keyboard suddenly came to life. It ventured up the',

return_tensors='pt').to(torch_device)
output = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

You will notice that the output starts getting repetitive. Therefore, greedy decoding is not suitable unless if you are generating really short sequences, like a token just providing a classification output.

[Figure 3-7](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch03.html#greedy-decoding) shows an example of greedy decoding using the FLAN-T5 model. Note that we missed out on some great sequences because one of the desired tokens has slightly lower probability, ensuring it never gets picked.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/figure523.png" alt="Greedy decoding" height="168" width="600"><figcaption></figcaption></figure>

**Figure 3-7. Greedy decoding**

### Beam Search

One of the most popular alternatives to greedy decoding is beam search. In beam search, the model uses a beam (sequences of tokens) to determine the cumulative probability of the sequences of tokens and picks the beam with the highest probability. In HuggingFace, the _num\_beams_ parameter of the model.generate() function determines the size of the beam. Here is how the decoding code would look like if we used beam search:

```
output = model.generate(**inputs, max_new_tokens=50, num_beams = 3)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

[Figure 3-8](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch03.html#beam-search) shows an example of beam search using the FLAN-T5 model. Note that the repetitiveness problem hasn’t really been solved using beam search. The text also sounds very constricted and un-humanlike, due to the complete absence of lower probability words.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/figure515.png" alt="Beam Search" height="106" width="600"><figcaption></figcaption></figure>

**Figure 3-8. Beam Search**

To resolve these issues, we will need to start introducing some randomness and begin sampling from the probability distribution to ensure not just the top 2-3 tokens get generated all the time.

### Top-K sampling

In top-k sampling, the model samples from a distribution of just the K tokens of the output distribution that have the highest probability. The probability mass is redistributed over the K tokens and the model samples from this distribution to generate the next token. HuggingFace provides the _top\_k_ parameter in its generate function.

```
output = model.generate(**inputs, max_new_tokens=50, do_sample=True, top_k=40)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

[Figure 3-9](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch03.html#topk-sampling) shows an example of top-k sampling using the FLAN-T5 model. Note that this is a vast improvement from greedy or beam search. However, top-p leads to problematic generations when used in cases where the probability is dominated by a few tokens, meaning that tokens with very low probability end up being included in the top-K.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/figure524.png" alt="Top-K Sampling" height="138" width="600"><figcaption></figcaption></figure>

**Figure 3-9. Top-K Sampling**

### Top-P sampling

Top-p sampling solves the problem with top-k sampling by making the number of candidate tokens dynamic. Top-p involves choosing the smallest number of tokens whose cumulative distribution exceeds a given probability p. As seen earlier in the chapter, top-p sampling is available for GPT3.5 and GPT-4 models. Here is how you can implement this in HuggingFace

```
output = model.generate(**inputs, max_new_tokens=50, top_p=0.9)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

[Figure 3-10](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch03.html#topp-sampling) shows an example of top-p sampling using the FLAN-T5 model. Top-p sampling, also called nucleus sampling, is the most popular sampling strategy used today.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/figure525.png" alt="Top-P Sampling" height="131" width="600"><figcaption></figcaption></figure>

**Figure 3-10. Top-P Sampling**

## Model debugging and interpretability

Now that we are comfortable with loading LLMs and generating text using them, we would like to be able to understand model behavior and explore the examples for which the model fails. Google’s open-source tool called [LIT-NLP](https://pair-code.github.io/lit/) is a handy tool that supports visualizations of model behavior as well as various debugging workflows.

[Figure 3-11](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch03.html#lit-NLP) shows an example of LIT-NLP in action, providing interpretability for a T-5 model running a summarization task.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/figure513.png" alt="lit-NLP" height="327" width="600"><figcaption></figcaption></figure>

**Figure 3-11. LIT-NLP**

Here are some features available in LIT-NLP that help you debug your models:

* Visualization of the attention mechanism.
* Salience maps, which show parts of the input that is most paid attention to by the model.
* Visualization of embeddings.
* Counterfactual analysis that shows how your model behavior changes after a change to the input like adding or removing a token.

## Summary

In this chapter, we journeyed through the LLM landscape and took note of the various options we have at our disposal. We learned how to determine the criteria most relevant to our tasks and choose the right LLM accordingly. We explored the various LLM benchmarks and showed how to interpret their results. We learned how to load LLMs and run inference on them, along with efficient decoding strategies. Finally, we showcased interpretability tools like LIT-NLP that can help us understand what is going on behind the scenes in the Transformer architecture.

In the next chapter, we will go through advanced fine-tuning methods like PEFT (Parameter Efficient Fine Tuning). We will showcase various types of PEFT techniques including prefix tuning, adapters, LoRA (Low Rank Adaptation), QLoRA (Quantized Low Rank Adaptation). We will discuss instruction-tuning and show the different techniques to create your own instruction tuning datasets. We will also enter the world of reinforcement learning and learn how to conduct RLHF training to train our own chat-models.

[1](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch03.html#id158-marker) If you haven’t had endless arguments with your friends and family about the color of the dress, now would be the time to do so. For more context, see [_https://en.wikipedia.org/wiki/The\_dress_](https://en.wikipedia.org/wiki/The\_dress)
