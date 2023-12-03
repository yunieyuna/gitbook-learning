# Chapter 9: LLM applications in Production

## 9 Generative AI in Production

### Join our book community on Discord

[https://packt.link/EarlyAccessCommunity](https://packt.link/EarlyAccessCommunity)

![Qr code Description automatically generated](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file58.png)

In this book so far, we’ve talked about models, agents, and LLM apps as well as different use cases, but there are many issues that become important when performance and regulatory requirements needs to be ensured, models and applications need to be deployed at scale, and finally monitoring has to be in place. In this chapter, we’ll discuss evaluation and observability, summarizing a broad range of topics that encompass the governance and lifecycle management of operationalized AI and decision models, including generative AI models. While offline evaluation provides a preliminary understanding of a model's abilities in a controlled setting, observability in production offers continuing insights into its performance in live environments. Both are crucial at different stages of a model's life cycle and complement each other to ensure optimal operation and results from large language models. We’ll discuss a few tools for either case and we’ll give examples.We’ll also discuss deploying of models and applications built around LLMs giving an overview over available tools and examples for deployment with Fast API and Ray Serve.Throughout the chapter, we’ll work on … with LLMs, which you can find in the GitHub repository for the book at [https://github.com/benman1/generative\_ai\_with\_langchain](https://github.com/benman1/generative\_ai\_with\_langchain)The main sections of this chapter are:

* Introduction
* How to evaluate your LLM app?
* How to deploy your LLM app?
* How to observe your LLM app?

Let’s start by introducing MLOps for LLMs and other generative models, what it means and includes.

### Introduction

As we’ve discussed in this book, LLMs have gained significant attention in recent years due to their ability to generate human-like text. From creative writing to conversational chatbots, these generative AI models have diverse applications across industries. However, taking these complex neural network systems from research to real-world deployment comes with significant challenges. This chapter explores the practical considerations and best practices for productionizing generative AI responsibly. We discuss the computational requirements for inference and serving, techniques for optimization, and critical issues around data quality, bias, and transparency. Architectural and infrastructure decisions can make or break a generative AI solution when scaled to thousands of users. At the same time, maintaining rigorous testing, auditing, and ethical safeguards is essential for trustworthy deployment.Deploying applications consisting of models and agents with their tools in production comes with several key challenges that need to be addressed to ensure their effective and safe use:

* **Data Quality and Bias**: Training data can introduce biases that get reflected in model outputs. Careful data curation and monitoring model outputs is crucial.
* **Ethical/Compliance Considerations**: LLMs can generate harmful, biased or misleading content. Review processes and safety guidelines must be established to prevent misuse. Adhering to regulations like HIPAA in specialized industries such as healthcare.
* **Resource Requirements**: LLMs require massive compute resources for training and serving. Efficient infrastructure is critical for cost-effective deployment at scale.
* **Drift or Performance Degradation**: Models need continuous monitoring to detect issues like data drift or performance degradation over time.
* **Lack of Interpretability**: LLMs are often black boxes, making their behaviors and decisions opaque. Interpretability tools are important for transparency.

Taking a trained LLM from research into real-world production involves navigating many complex challenges around aspects like scalability, monitoring, and unintended behaviors. Responsibly deploying capable yet unreliable models involves diligent planning around scalability, interpretability, testing, and monitoring. Techniques like fine-tuning, safety interventions, and defensive design enable developing applications that are helpful, harmless, and honest. With care and preparation, generative AI holds immense potential to benefit industries from medicine to education.Several key patterns can help address the challenges highlighted above:

* **Evaluations**: Solid benchmark datasets and metrics are essential to measure model capabilities, regressions, and alignment with goals. Metrics should be carefully selected based on the task.
* **Retrieval Augmentation**: Retrieving external knowledge provides useful context to reduce hallucinations and add recent information beyond pre-training data.
* **Fine-tuning**: Further tuning LLMs on task-specific data improves performance on target use cases. Techniques like adapter modules reduce overhead.
* **Caching**: Storing model outputs can significantly reduce latency and costs for repeated queries. But cache validity needs careful consideration.
* **Guardrails**: Validating model outputs syntactically and semantically ensures reliability. Guidance techniques directly shape output structure.
* **Defensive UX**: Design anticipating inaccuracies, such as disclaimers on limitations, attributions, and collecting rich user feedback.
* **Monitoring**: Continuously tracking metrics, model behaviors, and user satisfaction provides insight into model issues and business impact.

In Chapter 5, we’ve already covered safety-aligned techniques like Constitutional AI for mitigating risks like generating harmful outputs. Further, LLMs have the potential to generate harmful or misleading content. It is essential to establish ethical guidelines and review processes to prevent the dissemination of misinformation, hate speech, or any other harmful outputs. Human reviewers can play a crucial role in evaluating and filtering the generated content to ensure compliance with ethical standards.Not only for legal, ethical, and reputational reasons, but also in order to maintain performance, we need to continuously evaluate model performance and outputs in order to detect issues like data drift or loss of capabilities. We’ll be discussing techniques to interpret model behaviors and decisions. Improving transparency in high-stakes domains.LLMs or generative AI models require significant computational resources for deployment due to their size and complexity. This includes high-performance hardware, such as GPUs or TPUs, to handle the massive amount of computations involved. Scaling large language models or generative AI models can be challenging due to their resource-intensive nature. As the size of the model increases, the computational requirements for training and inference also increase exponentially. Distributed techniques, such as data parallelism or model parallelism, are often used to distribute the workload across multiple machines or GPUs. This allows for faster training and inference times. Scaling also involves managing the storage and retrieval of large amounts of data associated with these models. Efficient data storage and retrieval systems are required to handle the massive model sizes.Deployment also involves considerations for optimizing inference speed and latency. Techniques like model compression, quantization, or hardware-specific optimizations may be employed to ensure efficient deployment. We’ve discussed some of this in _Chapter 8_. LLMs or generative AI models are often considered black boxes, meaning it can be difficult to understand how they arrive at their decisions or generate their outputs. Interpretability techniques aim to provide insights into the inner workings of these models. This can involve methods like attention visualization, feature importance analysis, or generating explanations for model outputs. Interpretability is crucial in domains where transparency and accountability are important, such as healthcare, finance, or legal systems.As we discussed in _Chapter 8_, Large language models can be fine-tuned on specific tasks or domains to improve their performance on specific use cases. Transfer learning allows models to leverage pre-trained knowledge and adapt it to new tasks. Transfer learning and fine-tuning on domain-specific data unlocks new use cases while requiring additional diligence.With insightful planning and preparation, generative AI promises to transform industries from creative writing to customer service. But thoughtfully navigating the complexities of these systems remains critical as they continue permeating diverse domains. This chapter aims to provide a practical guide for teams of the pieces that we’ve left out so far aiming to build impactful and responsible generative AI applications. We mention strategies for data curation, model development, infrastructure, monitoring, and transparency. Before we continue our discussion, a few words on terminology is in place.

#### Terminology

**MLOps** is a paradigm that focuses on deploying and maintaining machine learning models in production reliably and efficiently. It combines the practices of DevOps with machine learning to transition algorithms from experimental systems to production systems. MLOps aims to increase automation, improve the quality of production models, and address business and regulatory requirements. **LLMOps** is a specialized sub-category of MLOps. It refers to the operational capabilities and infrastructure necessary for fine-tuning and operationalizing large language models as part of a product. While it may not be drastically different from the concept of MLOps, the distinction lies in the specific requirements connected to handling, refining, and deploying massive language models like GPT-3, which houses 175 billion parameters.The term **LMOps** is more inclusive than LLMOps as it encompasses various types of language models, including both large language models and smaller generative models. This term acknowledges the expanding landscape of language models and their relevance in operational contexts.**FOMO** **(Foundational Model Orchestration)** specifically addresses the challenges faced when working with foundational models. It highlights the need for managing multi-step processes, integrating with external resources, and coordinating workflows involving these models.The term **ModelOps** focuses on the governance and lifecycle management of AI and decision models as they are deployed. Even more broadly, **AgentOps** involves the operational management of LLMs and other AI agents, ensuring their appropriate behavior, managing their environment and resource access, and facilitating interactions between agents while addressing concerns related to unintended outcomes and incompatible objectives.While FOMO emphasizes the unique challenges of working specifically with foundational models, LMOps provides a more inclusive and comprehensive coverage of a wider range of language models beyond just the foundational ones. LMOps acknowledges the versatility and increasing importance of language models in various operational use cases, while still falling under the broader umbrella of MLOps. Finally, AgentOps explicitly highlights the interactive nature of agents consisting of generative models operating with certain heuristics and includes tools. The emergence of all very specialized terms underscores the rapid evolution of the field; however, their long-term prevalence is unclear. MLOps is an established term widely used in the industry, with significant recognition and adoption. Therefore, we’ll stick to MLOps for the remainder of this chapter.Before productionizing any agent or model, we should first evaluate its output, so we should start with this. We will focus on the evaluation methods provided by LangChain.

### How to evaluate your LLM apps?

Evaluating LLMs either as standalone entities or in conjunction with an agent chain is crucial to ensure they function correctly and produce reliable results, and is an integral part of the machine learning lifecycle. The evaluation process determines the performance of the models in terms of effectiveness, reliability, and efficiency. The goal of evaluating large language models is to understand their strengths and weaknesses, enhancing accuracy and efficiency while reducing errors, thereby maximizing their usefulness in solving real-world problems. This evaluation process typically occurs offline during the development phase. Offline evaluations provide initial insights into model performance under controlled test conditions and include aspects like hyper-parameter tuning, benchmarking against peer models or established standards. They offer a necessary first step towards refining a model before deployment.Evaluations provide insights into how well an LLM can generate outputs that are relevant, accurate, and helpful. In LangChain, there are various ways to evaluate outputs of LLMs, including comparing chain outputs, pairwise string comparisons, string distances, and embedding distances. The evaluation results can be used to determine the preferred model based on the comparison of outputs. Confidence intervals and p-values can also be calculated to assess the reliability of the evaluation results. LangChain provides several tools for evaluating the outputs of large language models. A common approach is to compare the outputs of different models or prompts using the `PairwiseStringEvaluator`. This prompts an evaluator model to choose between two model outputs for the same input and aggregates the results to determine an overall preferred model.Other evaluators allow assessing model outputs based on specific criteria like correctness, relevance, and conciseness. The `CriteriaEvalChain` can score outputs on custom or predefined principles without needing reference labels. Configuring the evaluation model is also possible by specifying a different chat model like ChatGPT as the evaluator.Let’s compare outputs of different prompts or LLMs with the `PairwiseStringEvaluator`, which prompts an LLM to select the preferred output given a specific input.

#### Comparing two outputs

This evaluation requires an evaluator, a dataset of inputs, and two or more LLMs, chains, or agents to compare. The evaluation aggregates the results to determine the preferred model.The evaluation process involves several steps:

1. Create the Evaluator: Load the evaluator using the `load_evaluator()` function, specifying the type of evaluator (in this case, `pairwise_string`).
2. Select Dataset: Load a dataset of inputs using the `load_dataset()` function.
3. Define Models to Compare: Initialize the LLMs, Chains, or Agents to compare using the necessary configurations. This involves initializing the language model and any additional tools or agents required.
4. Generate Responses: Generate outputs for each of the models before evaluating them. This is typically done in batches to improve efficiency.
5. Evaluate Pairs: Evaluate the results by comparing the outputs of different models for each input. This is often done using a random selection order to reduce positional bias.

Here’s an example from the documentation for pairwise string comparisons:

```
from langchain.evaluation import load_evaluator
evaluator = load_evaluator("labeled_pairwise_string")
evaluator.evaluate_string_pairs(
    prediction="there are three dogs",
    prediction_b="4",
    input="how many dogs are in the park?",
    reference="four",
)
```

The output from the evaluator should look as follows:

```
    {'reasoning': 'Both responses are relevant to the question asked, as they both provide a numerical answer to the question about the number of dogs in the park. However, Response A is incorrect according to the reference answer, which states that there are four dogs. Response B, on the other hand, is correct as it matches the reference answer. Neither response demonstrates depth of thought, as they both simply provide a numerical answer without any additional information or context. \n\nBased on these criteria, Response B is the better response.\n',
     'value': 'B',
     'score': 0}
```

The evaluation result includes a score between 0 and 1, indicating the effectiveness of the agent, sometimes along with reasoning that outlines the evaluation process and justifies the score.In this illustration of against the reference, both results are factually incorrect based on the input. We could remove the reference and let an LLM judge the outputs instead, however, this is potentially dangerous since the specified can also be incorrect.

#### Comparing against criteria

LangChain provides several predefined evaluators for different evaluation criteria. These evaluators can be used to assess outputs based on specific rubrics or criteria sets. Some common criteria include conciseness, relevance, correctness, coherence, helpfulness, and controversiality.The `CriteriaEvalChain` allows you to evaluate model outputs against custom or predefined criteria. It provides a way to verify if an LLM or Chain's output complies with a defined set of criteria. You can use this evaluator to assess correctness, relevance, conciseness, and other aspects of the generated outputs.The `CriteriaEvalChain` can be configured to work with or without reference labels. Without reference labels, the evaluator relies on the LLM's predicted answer and scores it based on the specified criteria. With reference labels, the evaluator compares the predicted answer to the reference label and determines its compliance with the criteria.The evaluation LLM used in LangChain, by default, is GPT-4. However, you can configure the evaluation LLM by specifying other chat models, such as ChatAnthropic or ChatOpenAI, with the desired settings (for example, temperature). The evaluators can be loaded with a custom LLM by passing the LLM object as a parameter to the `load_evaluator()` function.LangChain supports both custom criteria and predefined principles for evaluation. Custom criteria can be defined using a dictionary of `criterion_name: criterion_description pairs`. These criteria can be used to assess outputs based on specific requirements or rubrics.Here’s is a simple example:

```
custom_criteria = {
    "simplicity": "Is the language straightforward and unpretentious?",
    "clarity": "Are the sentences clear and easy to understand?",
    "precision": "Is the writing precise, with no unnecessary words or details?",
    "truthfulness": "Does the writing feel honest and sincere?",
    "subtext": "Does the writing suggest deeper meanings or themes?",
}
evaluator = load_evaluator("pairwise_string", criteria=custom_criteria)
evaluator.evaluate_string_pairs(
    prediction="Every cheerful household shares a similar rhythm of joy; but sorrow, in each household, plays a unique, haunting melody.",
    prediction_b="Where one finds a symphony of joy, every domicile of happiness resounds in harmonious,"
    " identical notes; yet, every abode of despair conducts a dissonant orchestra, each"
    " playing an elegy of grief that is peculiar and profound to its own existence.",
    input="Write some prose about families.",
)
```

We can get a very nuanced comparison of the two outputs as this result shows:

```
{'reasoning': 'Response A is simple, clear, and precise. It uses straightforward language to convey a deep and sincere message about families. The metaphor of music is used effectively to suggest deeper meanings about the shared joys and unique sorrows of families.\n\nResponse B, on the other hand, is less simple and clear. The language is more complex and pretentious, with phrases like "domicile of happiness" and "abode of despair" instead of the simpler "household" used in Response A. The message is similar to that of Response A, but it is less effectively conveyed due to the unnecessary complexity of the language.\n\nTherefore, based on the criteria of simplicity, clarity, precision, truthfulness, and subtext, Response A is the better response.\n\n[[A]]', 'value': 'A', 'score': 1}
```

Alternatively, you can use the predefined principles available in LangChain, such as those from Constitutional AI. These principles are designed to evaluate the ethical, harmful, and sensitive aspects of the outputs. The use of principles in evaluation allows for a more focused assessment of the generated text.

#### String and semantic comparisons

LangChain supports string comparison and distance metrics for evaluating LLM outputs. String distance metrics like Levenshtein and Jaro provide a quantitative measure of similarity between predicted and reference strings. Embedding distances using models like SentenceTransformers calculate semantic similarity between generated and expected texts.Embedding distance evaluators can use embedding models, such as those based on GPT-4 or Hugging Face embeddings, to compute vector distances between predicted and reference strings. This measures the semantic similarity between the two strings and can provide insights into the quality of the generated text. Here’s a quick example from the documentation:

```
from langchain.evaluation import load_evaluator
evaluator = load_evaluator("embedding_distance")
evaluator.evaluate_strings(prediction="I shall go", reference="I shan't go")
```

The evaluator returns the score 0.0966466944859925. You can change the embeddings used with the `embeddings` parameter in the `load_evaluator()` call. This often gives better results than older string distance metrics, but these are also available and allows for simple unit testing and assessment of accuracy. String comparison evaluators compare predicted strings against reference strings or inputs.String distance evaluators use distance metrics, such as Levenshtein or Jaro distance, to measure the similarity or dissimilarity between predicted and reference strings. This provides a quantitative measure of how similar the predicted string is to the reference string.Finally, there’s an agent trajectory evaluator, where the `evaluate_agent_trajectory()` method is used to evaluate the input, prediction, and agent trajectory.We can also use LangSmith to compare our performance against a dataset. We’ll talk about this companion project for LangChain – LangSmith – more in the section on observability.

#### Benchmark dataset

With LangSmith, we can evaluate the model performance against a dataset. Let’s step through an example.First of all, please make sure you create an account on LangSmith here: [https://smith.langchain.com/](https://smith.langchain.com/) You can obtain an API key and set it as `LANGCHAIN_API_KEY` in your environment. We can also set environment variables for project id and tracing:

```
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "My Project"
```

This configures LangChain to log traces. If we don’t tell LangChain the project id, it will log against the `default` project. After this setup, when we run our LangChain agent or chain, we’ll be able to see the traces on LangSmith. Let’s log a run!

```
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI()
llm.predict("Hello, world!")
```

We’ll see this on LangSmith like this:LangSmith allows us to list all runs so far on the LangSmith project page: [https://smith.langchain.com/projects](https://smith.langchain.com/projects)

```
from langsmith import Client
client = Client()
runs = client.list_runs()
print(runs)
```

We can list runs from a specific project or with by `run_type`, for example "chain". Each run comes with inputs and outputs as we can see here:

```
print(f"inputs: {runs[0].inputs}")
print(f"outputs: {runs[0]. outputs}")
```

We can create a dataset from existing agent runs with the `create_example_from_run()` function – or from anything else. Here’s how to create a dataset with a set of questions:

```
questions = [
    "A ship's parts are replaced over time until no original parts remain. Is it still the same ship? Why or why not?",  # The Ship of Theseus Paradox
    "If someone lived their whole life chained in a cave seeing only shadows, how would they react if freed and shown the real world?",  # Plato's Allegory of the Cave
    "Is something good because it is natural, or bad because it is unnatural? Why can this be a faulty argument?",  # Appeal to Nature Fallacy
    "If a coin is flipped 8 times and lands on heads each time, what are the odds it will be tails next flip? Explain your reasoning.",  # Gambler's Fallacy
    "Present two choices as the only options when others exist. Is the statement \"You're either with us or against us\" an example of false dilemma? Why?",  # False Dilemma
    "Do people tend to develop a preference for things simply because they are familiar with them? Does this impact reasoning?",  # Mere Exposure Effect
    "Is it surprising that the universe is suitable for intelligent life since if it weren't, no one would be around to observe it?",  # Anthropic Principle
    "If Theseus' ship is restored by replacing each plank, is it still the same ship? What is identity based on?",  # Theseus' Paradox
    "Does doing one thing really mean that a chain of increasingly negative events will follow? Why is this a problematic argument?",  # Slippery Slope Fallacy
    "Is a claim true because it hasn't been proven false? Why could this impede reasoning?",  # Appeal to Ignorance
]
shared_dataset_name = "Reasoning and Bias"
ds = client.create_dataset(
    dataset_name=shared_dataset_name, description="A few reasoning and cognitive bias questions",
)
for q in questions:
    client.create_example(inputs={"input": q}, dataset_id=ds.id)
```

We can then run an LLM agent or chain on the dataset like this:

```
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
llm = ChatOpenAI(model="gpt-4", temperature=0.0)
def construct_chain():
    return LLMChain.from_string(
        llm,
        template="Help out as best you can.\nQuestion: {input}\nResponse: ",
    )
```

We use a constructor function to initialize for each input. In order to evaluate the model performance against this dataset, we need to define an evaluator as we’ve seen in the previous section.

```
from langchain.evaluation import EvaluatorType
from langchain.smith import RunEvalConfig
evaluation_config = RunEvalConfig(
    evaluators=[
        # Arbitrary criterion as a key: value pair in the criteria dict:
        RunEvalConfig.Criteria({"helpfulness": "Is the response helpful?"}),
        RunEvalConfig.Criteria({"insightful": "Is the response carefully thought out?"})
    ]
)
```

We’ll pass a dataset and evaluators to `run_on_dataset()` to generate metrics and feedback:

```
from langchain.smith import run_on_dataset
results = run_on_dataset(
  client=client,
  dataset=dataset,
  llm_factory=lambda: my_agent, 
  evaluation=evaluation_config
)
```

Similarly, we could pass a dataset and evaluators to arun\_on\_dataset`()` to generate metrics and feedback asynchronously.We can view the evaluator feedback in the LangSmith UI to identify areas for improvement:

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file59.png" alt="Figure 9.1: Evaluators in LangSmith." height="1204" width="2336"><figcaption><p>Figure 9.1: Evaluators in LangSmith.</p></figcaption></figure>

We can click on any of these evaluations to see some detail, for example, for the careful thinking evaluator, we get this prompt that includes the original answer by the LLM:

```
You are assessing a submitted answer on a given task or input based on a set of criteria. Here is the data:
[BEGIN DATA]
***
[Input]: Is something good because it is natural, or bad because it is unnatural? Why can this be a faulty argument?
***
[Submission]: The argument that something is good because it is natural, or bad because it is unnatural, is often referred to as the "appeal to nature" fallacy. This argument is faulty because it assumes that what is natural is automatically good or beneficial, and what is unnatural is automatically bad or harmful. However, this is not always the case. For example, many natural substances can be harmful or deadly, such as certain plants or animals. Conversely, many unnatural things, such as modern medicine or technology, can greatly benefit our lives. Therefore, whether something is natural or unnatural is not a reliable indicator of its value or harm.
***
[Criteria]: insightful: Is the response carefully thought out?
***
[END DATA]
Does the submission meet the Criteria? First, write out in a step by step manner your reasoning about each criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then print only the single character "Y" or "N" (without quotes or punctuation) on its own line corresponding to the correct answer of whether the submission meets all criteria. At the end, repeat just the letter again by itself on a new line.
```

We get this evaluation:

```
The criterion is whether the response is insightful and carefully thought out. 
The submission provides a clear and concise explanation of the "appeal to nature" fallacy, demonstrating an understanding of the concept. It also provides examples to illustrate why this argument can be faulty, showing that the respondent has thought about the question in depth. The response is not just a simple yes or no, but a detailed explanation that shows careful consideration of the question. 
Therefore, the submission does meet the criterion of being insightful and carefully thought out.
Y
Y
```

A way to improve performance for a few types of problems is to do few-shot prompting. LangSmith can help us with this as well. You can find more examples for this in the LangSmith documentation.This concludes evaluation. Now that we’ve evaluated our agents, let’s say we are happy with the performance and we deploy it!

### How to deploy your LLM apps?

Given the increasing use of LLMs in various sectors, it's imperative to understand how to effectively deploy models and apps into production. Deployment Services and Frameworks can help to scale the technical hurdles. There are lots of different ways to productionize LLM-apps or applications with generative AI. Deployment for production requires research into and knowledge of the generative AI ecosystem, which encompasses different aspects including:

* Models and LLM-as-a-Service: LLMs and other models either run directly or offered as an API on vendor-provided infrastructure.
* Reasoning Heuristics: Retrieval Augmented Generation (RAG), Tree-of-Thought, and others.
* Vector Databases: Aid in retrieving contextually relevant information for prompts.
* Prompt Engineering Tools: These facilitate in-context learning without requiring expensive fine-tuning or sensitive data.
* Pre-training and fine-tuning: For models specialized for specific tasks or domains.
* Prompt Logging, Testing, and Analytics: An emerging sector inspired by the desire to understand and improve the performance of Large Language Models.
* Custom LLM Stack: A set of tools for shaping and deploying solutions built on open-source models.

We’ve discussed models in _Chapter 1_ and _Chapter 3_, reasoning heuristics in chapters 4-7, vector databases in Chapter 5, and prompts and fine-tuning in _Chapter 8_. In this chapter, we’ll focus on logging, monitoring, and custom tools for deployment.LLMs are typically utilized using external LLM providers or self-hosted models. With external providers, computational burdens are shouldered by companies like OpenAI or Anthropic, while LangChain facilitates business logic implementation. However, self-hosting open-source LLMs can significantly decrease costs, latency, and privacy concerns.Some tools with infrastructure offer the full package. For example, you can deploy LangChain agents with Chainlit creating ChatGPT-like UIs with Chainlit. Some of the key features include intermediary steps visualisation, element management & display (images, text, carousel, and others) as well as cloud deployment. BentoML is a framework that enables the containerization of machine learning applications to use them as microservices running and scaling independently with automatic generation of OpenAPI and gRPC endpoints.You can also deploy LangChain to different cloud service endpoints, for example, an Azure Machine Learning Online Endpoint. With Steamship, LangChain developers can rapidly deploy their apps, which includes: production-ready endpoints, horizontal scaling across dependencies, persistent storage of app state, multi-tenancy support, and more.Here is a table summarizing services and frameworks for deploying large language model applications:

| **Name**                  | **Description**                                                  | **Type**      |
| ------------------------- | ---------------------------------------------------------------- | ------------- |
| Streamlit                 | Open-source Python framework for building and deploying web apps | Framework     |
| Gradio                    | Lets you wrap models in an interface and host on Hugging Face    | Framework     |
| Chainlit                  | Build and deploy conversational ChatGPT-like apps                | Framework     |
| Apache Beam               | Tool for defining and orchestrating data processing workflows    | Framework     |
| Vercel                    | Platform for deploying and scaling web apps                      | Cloud Service |
| FastAPI                   | Python web framework for building APIs                           | Framework     |
| Fly.io                    | App hosting platform with autoscaling and global CDN             | Cloud Service |
| DigitalOcean App Platform | Platform to build, deploy and scale apps                         | Cloud Service |
| Google Cloud              | Services like Cloud Run to host and scale containerized apps     | Cloud Service |
| Steamship                 | ML infrastructure platform for deploying and scaling models      | Cloud Service |
| Langchain-serve           | Tool to serve LangChain agents as web APIs                       | Framework     |
| BentoML                   | Framework for model serving, packaging and deployment            | Framework     |
| OpenLLM                   | Provides open APIs to commercial LLMs                            | Cloud Service |
| Databutton                | No-code platform to build and deploy model workflows             | Framework     |
| Azure ML                  | Managed ML ops service on Azure for models                       | Cloud Service |

Figure 9.2: Services and frameworks for deploying large language model applications.

All of these are well-documented with different use cases, often directly referencing LLMs. We’ve already shown examples with Streamlit and Gradio, and we’ve discussed how to deploy them to HuggingFace Hub as an example.There are a few main requirements for running LLM applications:

* Scalable infrastructure to handle computationally intensive models and potential spikes in traffic
* Low latency for real-time serving of model outputs
* Persistent storage for managing long conversations and app state
* APIs for integration into end-user applications
* Monitoring and logging to track metrics and model behavior

Maintaining cost efficiency can be challenging with large volumes of user interactions and high costs associated with LLM services. Strategies to manage efficiency include self-hosting models, auto-scaling resource allocations based on traffic, using spot instances, independent scaling, and batching requests to better utilize GPU resources.The choice of the tools and the infrastructure determines trade-offs between these requirements. Flexibility and ease is very important, because we want to be able to iterate rapidly, which is vital due to the dynamic nature of ML and LLM landscapes. It's crucial to avoid getting tied to one solution. A flexible, scalable serving layer that accommodates various models is key. Model composition and cloud providers' selection forms part of this flexibility equation.For most flexibility, Infrastructure as Code (IaC) tools like Terraform, CloudFormation, or Kubernetes YAML files can recreate your infrastructure reliably and quickly. Moreover, continuous integration and continuous delivery (CI/CD) pipelines can automate testing and deployment processes to reduce errors and facilitate quicker feedback and iteration.Designing a robust LLM application service can be a complex task requiring an understanding the trade-offs and critical considerations when evaluating serving frameworks. Leveraging one of these solutions for deployment allows developers to focus on developing impactful AI applications rather than infrastructure. As mentioned LangChain plays nicely with several open-source projects and frameworks like Ray Serve, BentoML, OpenLLM, Modal, and Jina. In the next section, we’ll deploy a chat service webserver based on FastAPI.

#### Fast API webserver

FastAPI is a very popular choice for deployment of webservers. Designed to be fast, easy to use, and efficient, it is a modern, high-performance web framework for building APIs with Python. Lanarky is a small, open-source library for deploying LLM applications that provides convenient wrappers around Flask API as well as Gradio for deployment of LLM applications. This means you can get a REST API endpoint as well as the in-browser visualization at once and you only need a few lines of code.

> A **REST API** (Representational State Transfer Application Programming Interface) is a set of rules and protocols that allows different software applications to communicate with each other over the internet. It follows the principles of REST, which is an architectural style for designing networked applications. A REST API uses HTTP methods (such as GET, POST, PUT, DELETE) to perform operations on resources, and it typically sends and receives data in a standardized format, such as JSON or XML.

In the library documentation, there are several examples including a Retrieval QA with Sources Chain, a Conversational Retrieval app, and a Zero Shot Agent. Following another example, we’ll implement a chatbot webserver with Lanarky. We’ll set up a web server using Lanarky that integrates with Gradio, creates a `ConversationChain` instance with an LLM model and settings, and defines routes for handling HTTP requests.First, we’ll import the necessary dependencies, including FastAPI for creating the web server, `mount_gradio_app` for integrating with Gradio, `ConversationChain` and `ChatOpenAI` from Langchain for handling LLM conversations, and other required modules:

```
from fastapi import FastAPI
from lanarky.testing import mount_gradio_app
from langchain import ConversationChain
from langchain.chat_models import ChatOpenAI
from lanarky import LangchainRouter
from starlette.requests import Request
from starlette.templating import Jinja2Templates
```

Please note that you need to set your environment variables as explained in chapter 3. A `create_chain()` function is defined to create an instance of `ConversationChain`, specifying the LLM model and its settings:

```
def create_chain():
    return ConversationChain(
        llm=ChatOpenAI(
            temperature=0,
            streaming=True,
        ),
        verbose=True,
    )
```

We set the chain as a `ConversationChain`.

```
chain = create_chain()
```

The app variable is assigned to `mount_gradio_app`, which creates a `FastAPI` instance titled _ConversationChainDemo_ and integrates it with Gradio:

```
app = mount_gradio_app(FastAPI(title="ConversationChainDemo"))
```

The templates variable gets set to a `Jinja2Templates` class, specifying the directory where templates are located for rendering. This specifies how the webpage will be shown allowing all kind of customization:

```
templates = Jinja2Templates(directory="webserver/templates")
```

An endpoint for handling HTTP GET requests at the root path (`/`) is defined using the FastAPI decorator `@app.get`. The function associated with this endpoint returns a template response for rendering the index.html template:

```
@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
```

The router object is created as a `LangchainRouter` class. This object is responsible for defining and managing the routes associated with the `ConversationChain` instance. We can add additional routes to the router for handling JSON-based chat that even work with WebSocket requests:

```
langchain_router = LangchainRouter(
    langchain_url="/chat", langchain_object=chain, streaming_mode=1
)
langchain_router.add_langchain_api_route(
    "/chat_json", langchain_object=chain, streaming_mode=2
)
langchain_router.add_langchain_api_websocket_route("/ws", langchain_object=chain)
app.include_router(langchain_router)
```

Now our application knows how to handle requests made to the specified routes defined within the router, directing them to the appropriate functions or handlers for processing.We will use Uvicorn to run our application. Uvicorn excels in supporting high-performance, asynchronous frameworks like FastAPI and Starlette. It is known for its ability to handle a large number of concurrent connections and perform well under heavy loads due to its asynchronous nature.We can run the webserver from the terminal like this:

```
uvicorn webserver.chat:app –reload
```

This command starts a webserver, which you can view in your browser, at this local address: [http://127.0.0.1:8000](http://127.0.0.1:8000/)The reload switch (`--reload`) is particularly handy, because it means the server will be automatically restarted once you’ve made any changes.Here’s a snapshot of the chatbot application we’ve just deployed:

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file60.png" alt="Figure 9.3: Chatbot in Flask/Lanarky" height="1206" width="1214"><figcaption><p>Figure 9.3: Chatbot in Flask/Lanarky</p></figcaption></figure>

I think this looks quite nice for little work we’ve put in. It also comes with a few nice features such as REST API, a web UI, and a websocket interface. While Uvicorn itself does not provide built-in load balancing functionality, it can work together with other tools or technologies such as Nginx or HAProxy to achieve load balancing in a deployment setup, which distribute incoming client requests across multiple worker processes or instances. The use of Uvicorn with load balancers enables horizontal scaling to handle large traffic volumes, improves response times for clients, enhances fault tolerance.In the next section, we’ll see how to build robust and cost-effective generative AI applications with Ray. We'll built a simple search engine using LangChain for text processing and Ray for scaling indexing and serving.

#### Ray

Ray provides a flexible framework to meet infrastructure challenges of complex neural networks in production by scaling out generative AI workloads across clusters. Ray helps with common deployment needs like low-latency serving, distributed training, and large-scale batch inference. Ray also makes it easy to spin up on-demand fine-tuning or scale existing workloads from one machine to many. Some capability includes:

* Schedule distributed training jobs across GPU clusters using Ray Train
* Deploy pre-trained models at scale for low-latency serving with Ray Serve
* Run large batch inference in parallel across CPUs and GPUs with Ray Data
* Orchestrate end-to-end generative AI workflows combining training, deployment, and batch processing

We'll use LangChain and Ray to build a simple search engine for the Ray documentation following an example implemented by Waleed Kadous for the anyscale Blog and on the langchain-ray repository on Github. You can see this as an extension of the recipe in _Channel 5_. You can see the full code for this recipe under semantic search here: [https://github.com/benman1/generative\_ai\_with\_langchain](https://github.com/benman1/generative\_ai\_with\_langchain) You’ll also see how to run this as a FastAPI server.First, we'll ingest and index the Ray docs so we can quickly find relevant passages for a search query:

```
# Load the Ray docs using the LangChain loader
loader = RecursiveUrlLoader("docs.ray.io/en/master/") 
docs = loader.load()
# Split docs into sentences using LangChain splitter
chunks = text_splitter.create_documents(
    [doc.page_content for doc in docs],
    metadatas=[doc.metadata for doc in docs])
# Embed sentences into vectors using transformers
embeddings = LocalHuggingFaceEmbeddings('multi-qa-mpnet-base-dot-v1')  
# Index vectors using FAISS via LangChain
db = FAISS.from_documents(chunks, embeddings) 
```

This builds our search index by ingesting the docs, splitting into sentences, embedding the sentences, and indexing the vectors. Alternatively, we can accelerate the indexing by parallelizing the embedding step:

```
# Define shard processing task
@ray.remote(num_gpus=1)  
def process_shard(shard):
  embeddings = LocalHuggingFaceEmbeddings('multi-qa-mpnet-base-dot-v1')
  return FAISS.from_documents(shard, embeddings)
# Split chunks into 8 shards
shards = np.array_split(chunks, 8)  
# Process shards in parallel
futures = [process_shard.remote(shard) for shard in shards]
results = ray.get(futures)
# Merge index shards
db = results[0]
for result in results[1:]:
  db.merge_from(result)
```

By running embedding on each shard in parallel, we can significantly reduce indexing time. We save the database index to disk:

```
db.save_local(FAISS_INDEX_PATH)
```

`FAISS_INDEX_PATH` is an arbitrary file name. I’ve set it to `faiss_index.db`.Next, we’ll see how we can serve search queries with Ray Serve.

```
# Load index and embedding
db = FAISS.load_local(FAISS_INDEX_PATH)
embedding = LocalHuggingFaceEmbeddings('multi-qa-mpnet-base-dot-v1')
@serve.deployment
class SearchDeployment:
  def __init__(self):
    self.db = db
    self.embedding = embedding
  
  def __call__(self, request):   
    query_embed = self.embedding(request.query_params["query"])
    results = self.db.max_marginal_relevance_search(query_embed) 
    return format_results(results) 
deployment = SearchDeployment.bind()
# Start service
serve.run(deployment)
```

This lets us serve search queries as a web endpoint! Running this gives me this output:

```
Started a local Ray instance. 
View the dashboard at 127.0.0.1:8265
```

We can now query it from Python:

```
import requests
query = "What are the different components of Ray"
         " and how can they help with large language models (LLMs)?”
response = requests.post("http://localhost:8000/", params={"query": query})
print(response.text)
```

For me, the server fetches the Ray use cases page at: [http://https://docs.ray.io/en/latest/ray-overview/use-cases.html](http://https/docs.ray.io/en/latest/ray-overview/use-cases.html)What I really liked was the monitoring with the Ray Dashboard, which looks like this:

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file61.png" alt="Figure 9.4: Ray Dashboard." height="1252" width="2404"><figcaption><p>Figure 9.4: Ray Dashboard.</p></figcaption></figure>

This dashboard very powerful as it can give you a whole bunch of metrics and other information. Collecting metrics is really easy, since all you have to do is setting and updating variables of type `Counter`, `Gauge`, `Histogram` or other types within the deployment object or actor. For time series charts you should have either Prometheus or Grafana server installed. As you can see in the full implementation on Github, we can also spin this up as a FastAPI server. This concludes our simple semantic search engine with LangChain and Ray. As models and LLM apps grow more sophisticated and highly interwoven into the fabric of business applications, observability and monitoring during production become necessary to ensure their accuracy, efficiency, and reliability ongoing. The next section focuses on the significance of monitoring LLMs and highlights key metrics to track for a comprehensive monitoring strategy.

### How to observe LLM apps?

The dynamic nature of real-world operations means that the conditions assessed during offline evaluations hardly cover all potential scenarios that LLMs may encounter in production systems. Thus comes the need for observability in production – a more on-going, real-time observation to capture anomalies that offline tests could not anticipate.Observability allows monitoring behaviors and outcomes as the model interacts with actual input data and users in production. It includes logging, tracking, tracing and alerting mechanisms to ensure healthy system functioning, performance optimization and catching issues like model drift early.As discussed, LLMs have become increasingly important components of many applications in sectors like health, e-commerce, and education.

> Tracking, tracing, and monitoring are three important concepts in the field of software operation and management. While all related to understanding and improving a system's performance, they each have distinct roles. While tracking and tracing are about keeping detailed historical records for analysis and debugging, monitoring is aimed at real-time observation and immediate awareness of issues to ensure optimal system functionality at all times. All three of these concepts fall within the category of observability.
>
> > **Monitoring** is the ongoing process of overseeing the performance of a system or application. This might involve continuously collecting and analyzing metrics related to system health such as memory usage, CPU utilization, network latency, and the overall application/service performance (like response time). Effective monitoring includes setting up alert systems for anomalies or unexpected behaviors – sending notifications when certain thresholds are exceeded. While tracking and tracing are about keeping detailed historical records for analysis and debugging, monitoring is aimed at real-time observation & immediate awareness of issues to ensure optimal system functionality at all times.

The chief aim for monitoring and observability is to provide insights into model performance and behavior through real-time data. This helps in:

* **Preventing model drift**: Models can degrade over time due to changes in the characteristics of input data or user behavior. Regular monitoring can identify such situations early and apply corrective measures.
* **Performance optimization**: By tracking metrics like inference times, resource usage, and throughput, you can make adjustments to improve the efficiency and effectiveness of LLMs in production.
* **A/B Testing**: It helps compare how slight differences in models may result in different outcomes which aids in decision-making towards model improvements.
* **Debugging Issues**: Monitoring helps identify unforeseen problems that can occur during runtime, enabling rapid resolution.

It’s important to consider the monitoring strategy that consists of a few considerations:

* **Metrics to monitor**: Define key metrics of interest such as prediction accuracy, latency, throughput etc. based on desired model performance.
* **Monitoring Frequency**: Frequency should be determined based on how critical the model is to operations - a highly critical model may require near real-time monitoring.
* **Logging**: Logs should provide comprehensive details regarding every relevant action performed by the LLM so analysts can track back any anomalies.
* **Alerting Mechanism**: The system should raise alerts if it detects anomalous behavior or drastic performance drops.

Monitoring LLMs serves multiple purposes, including assessing model performance, detecting abnormalities or issues, optimizing resource utilization, and ensuring consistent and high-quality outputs. By continuously evaluating the behavior and performance of LLMs via validation, shadow launches, and interpretation along with dependable offline evaluation, organizations can identify and mitigate potential risks, maintain user trust, and provide an optimal experience.Here’s a list of relevant metrics:

* **Inference Latency**: Measure the time it takes for the LLM to process a request and generate a response. Lower latency ensures a faster and more responsive user experience.
* **Query per Second (QPS)**: Calculate the number of queries or requests that the LLM can handle within a given time frame. Monitoring QPS helps assess scalability and capacity planning.
* **Token Per Second (TPS)**: Track the rate at which the LLM generates tokens. TPS metrics are useful for estimating computational resource requirements and understanding model efficiency.
* **Token Usage**: The number of tokens correlates with the resource usage such as hardware utilization, latency, and costs.
* **Error Rate**: Monitor the occurrence of errors or failures in LLM responses, ensuring error rates are kept within acceptable limits to maintain the quality of outputs.
* **Resource Utilization**: Measure the consumption of computational resources, such as CPU, memory, and GPU, to optimize resource allocation and avoid bottlenecks.
* **Model Drift**: Detect changes in LLM behavior over time by comparing its outputs to a baseline or ground truth, ensuring the model remains accurate and aligned with expected outcomes.
* **Out-of-Distribution Inputs**: Identify inputs or queries falling outside the intended distribution of the LLM's training data, which can cause unexpected or unreliable responses.
* **User Feedback Metrics**: Monitor user feedback channels to gather insights on user satisfaction, identify areas for improvement, and validate the effectiveness of the LLM.

Data scientists and machine learning engineers should check for staleness, incorrect learning, and bias using model interpretation tools like LIME and SHAP. The most predictive features changing suddenly could indicate a data leak. Offline metrics like AUC do not always correlate with online impacts on conversion rate, so it is important to find dependable offline metrics that translate to online gains relevant to the business ideally direct metrics like clicks and purchases that the system impacts directly.Effective monitoring enables the successful deployment and utilization of LLMs, boosting confidence in their capabilities and fostering user trust. It should be cautioned, however, that you should study the privacy and data protection policy when relying on cloud service platforms.In the next section, we’ll look at monitoring the trajectory of an agent.

#### Tracking and Tracing

> **Tracking** generally refers to the process of recording and managing information about a particular operation or series of operations within an application or system. For example, in machine learning applications or projects, tracking can involve keeping a record of parameters, hyperparameters, metrics, outcomes across different experiments or runs. It provides a way to document the progress and changes over time.
>
> > **Tracing** is a more specialized form of tracking. It involves recording the execution flow through software/systems. Particularly in distributed systems where a single transaction might span multiple services, tracing helps in maintaining an audit or breadcrumb trail, a detailed information about that request path through the system. This granular view enables developers to understand the interaction between various microservices and troubleshoot issues like latency or failures by identifying exactly where they occurred in the transaction path.

Tracking the trajectory of agents can be challenging due to their broad range of actions and generative capabilities. LangChain comes with functionality for trajectory tracking and evaluation. Seeing the traces of an agent is actually really easy! You just have to set the return\_`intermediate_steps` parameter to `True` when initializing an agent or an LLM. Let’s have a quick look at this. I’ll skip the imports and setting up the environment. You can find the full listing on github under monitoring at this address: [https://github.com/benman1/generative\_ai\_with\_langchain/](https://github.com/benman1/generative\_ai\_with\_langchain/)We’ll define a tool. It’s very convenient to use the `@tool` decorator, which will use the function docstring as description of the tool. The first tool sends a ping to a website address and returns information about packages transmitted and latency or – in the case of an error – the error message:

```
@tool
def ping(url: HttpUrl, return_error: bool) -> str:
    """Ping the fully specified url. Must include https:// in the url."""
    hostname = urlparse(str(url)).netloc
    completed_process = subprocess.run(
        ["ping", "-c", "1", hostname], capture_output=True, text=True
    )
    output = completed_process.stdout
    if return_error and completed_process.returncode != 0:
        return completed_process.stderr
    return output]
```

Now we set up an agent that uses this tool with an LLM to make the calls given a prompt:

```
llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0)
agent = initialize_agent(
    llm=llm,
    tools=[ping],
    agent=AgentType.OPENAI_MULTI_FUNCTIONS,
    return_intermediate_steps=True,  # IMPORTANT!
)
result = agent("What's the latency like for https://langchain.com?")
```

The agent reports this:

```
The latency for https://langchain.com is 13.773 ms
```

In `results[`"`intermediate_steps`"`]` we can see all lot of information about the agent’s actions:

```
[(_FunctionsAgentAction(tool='ping', tool_input={'url': 'https://langchain.com', 'return_error': False}, log="\nInvoking: `ping` with `{'url': 'https://langchain.com', 'return_error': False}`\n\n\n", message_log=[AIMessage(content='', additional_kwargs={'function_call': {'name': 'tool_selection', 'arguments': '{\n  "actions": [\n    {\n      "action_name": "ping",\n      "action": {\n        "url": "https://langchain.com",\n        "return_error": false\n      }\n    }\n  ]\n}'}}, example=False)]), 'PING langchain.com (35.71.142.77): 56 data bytes\n64 bytes from 35.71.142.77: icmp_seq=0 ttl=249 time=13.773 ms\n\n--- langchain.com ping statistics ---\n1 packets transmitted, 1 packets received, 0.0% packet loss\nround-trip min/avg/max/stddev = 13.773/13.773/13.773/0.000 ms\n')]
```

By providing visibility into the system and aiding in problem identification and optimization efforts, this kind of tracking and evaluation can be very helpful. The LangChain documentation demonstrates how to use a trajectory evaluator to examine the full sequence of actions and responses they generate, and grade an OpenAI functions agent. Let’s have a look beyond LangChain and see what’s out there for observability!

#### Observability tools

There are quite a few tools available as integrations in LangChain or through callbacks:

* **Argilla**: Argilla is an open-source data curation platform that can integrate user feedback (human-in-the-loop workflows) with prompts and responses to curate datasets for fine-tuning.
* **Portkey**: Portkey adds essential MLOps capabilities like monitoring detailed metrics, tracing chains, caching, and reliability through automatic retries to LangChain.
* **Comet.ml**: Comet offers robust MLOps capabilities for tracking experiments, comparing models and optimizing AI projects.
* **LLMonitor**: Tracks lots of metrics including cost and usage analytics (user tracking), tracing, and evaluation tools (open-source).
* **DeepEval**: Logs default metrics like relevance, bias, and toxicity. Can also help in testing and in monitoring model drift or degradation.
* **Aim**: An open-source visualization and debugging platform for ML models. It logs inputs, outputs, and the serialized state of components, enabling visual inspection of individual LangChain executions and comparing multiple executions side-by-side.
* **Argilla**: An open-source platform for tracking training data, validation accuracy, parameters, and more across machine learning experiments.
* **Splunk**: Splunk's Machine Learning Toolkit can provide observability into your machine learning models in production.
* **ClearML**: An open-source tool for automating training pipelines, seamlessly moving from research to production.
* **IBM Watson OpenScale**: A platform providing insights into AI health with fast problem identification and resolution to help mitigate risks.
* **DataRobot MLOps**: Monitors and manages models to detect issues before they impact performance.
* **Datadog APM Integration**: This integration allows you to capture LangChain requests, parameters, prompt-completions, and visualize LangChain operations. You can also capture metrics such as request latency, errors, and token/cost usage.
* **Weights and Biases (W\&B**) **Tracing**: We’ve already shown an example of using (W\&B) for monitoring of fine-training convergence, but it can also fulfill the role of tracking other metrics and of logging and comparing prompts.
* **Langfuse**: With this open-source tool, we can conveniently monitor detailed information along the traces regarding latency, cost, scores of our LangChain agents and tools.

Most of these integrations are very easy to integrate into LLM pipelines. For example, For W\&B, you can enable tracing by setting the `LANGCHAIN_WANDB_TRACING` environment variable to `True`. Alternatively, you can use a context manager with `wandb_tracing_enabled()` to trace a specific block of code. With Langfuse, we can hand over a `langfuse.callback.CallbackHandler()` as an argument to the `chain.run()` call.Some of these tools are open-source, and what’s great about these platforms is that it allows full customization and on-premise deployment for use cases, where privacy is important. For example, Langfuse is open-source and provides an option of self-hosting. Choose the option that best suits your needs and follow the instructions provided in the LangChain documentation to enable tracing for your agents.Having been released only recently, I am sure there’s much more to come for the platform, but it’s already great to see traces of how agents execute, detecting loops and latency issues. It enables sharing traces and stats with collaborators to discuss improvements.

#### LangSmith

LangSmith is a framework for debugging, testing, evaluating, and monitoring LLM applications developed and maintained by LangChain AI, the organization behind LangChain. LangSmith serves as an effective tool for MLOps, specifically for LLMs, by providing features that cover multiple aspects of the MLOps process. It can help developers take their LLM applications from prototype to production by providing features for debugging, monitoring, and optimizing. LangSmith aims to reduce the barrier to entry for those without a software background by providing a simple and intuitive user interface. LangSmith is a platform for debugging, testing, and monitoring large language models (LLMs) built with LangChain. It allows you to:

* Log traces of runs from your LangChain agents, chains, and other components
* Create datasets to benchmark model performance
* Configure AI-assisted evaluators to grade your models
* View metrics, visualizations, and feedback to iterate and improve your LLMs

LangSmith fulfils the requirements for MLOps for agents by providing features and capabilities that enable developers to debug, test, evaluate, monitor, and optimize language model applications. Its integration within the LangChain framework enhances the overall development experience and facilitates the full potential of language model applications. By using both two platforms, developers can take their LLM applications from prototype to production stage and optimize latency, hardware efficiency, and cost.We can get a large set of graphs for a bunch of important statistics as we can see here:

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file62.png" alt="Figure 9.5: Evaluator metrics in LangSmith." height="1326" width="2198"><figcaption><p>Figure 9.5: Evaluator metrics in LangSmith.</p></figcaption></figure>

The monitoring dashboard includes the following graphs that can be broken down into different time intervals:

| Statistics                                                                                                    | Category  |
| ------------------------------------------------------------------------------------------------------------- | --------- |
| Trace Count, LLM Call Count, Trace Success Rates, LLM Call Success Rates                                      | Volume    |
| Trace Latency (s), LLM Latency (s), LLM Calls per Trace, Tokens / sec                                         | Latency   |
| Total Tokens, Tokens per Trace, Tokens per LLM Call                                                           | Tokens    |
| % Traces w/ Streaming, % LLM Calls w/ Streaming, Trace Time-to-First-Token (ms), LLM Time-to-First-Token (ms) | Streaming |

Figure 9.6: Statistisc in LangSmith.

Here’s a tracing example in LangSmith for the benchmark dataset run that we’ve seen in the section on evaluation:

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file63.png" alt="Figure 9.7: Tracing in LangSmith." height="1326" width="2326"><figcaption><p>Figure 9.7: Tracing in LangSmith.</p></figcaption></figure>

The platform itself is not open source, however, LangChain AI, the company behind LangSmith and LangChain, provide some support for self-hosting for organizations with privacy concerns. There are however a few alternatives to LangSmith such as Langfuse, Weights and Biases, Datadog APM, Portkey, and PromptWatch, with some overlap in features. We’ll focus about LangSmith here, because it has a large set of features for evaluation and monitoring, and because it integrates with LangChain.In the next section, we’ll demonstrate the utilization of PromptWatch.io for prompt tracking of LLMs in production environments.

#### PromptWatch

PromptWatch records information about the prompt and the generated output during this interaction.Let’s get the inputs out of the way.

```
from langchain import LLMChain, OpenAI, PromptTemplate
from promptwatch import PromptWatch
from config import set_environment
set_environment()
```

As mentioned in Chapter 3, I’ve set all API keys in the environment in the set\_environment() function.

```
prompt_template = PromptTemplate.from_template("Finish this sentence {input}")
my_chain = LLMChain(llm=OpenAI(), prompt=prompt_template)
```

Using the `PromptTemplate` class, the prompt template is set to with one variable, `input`, indicating where the user input should be placed within the prompt.Inside the `PromptWatch` block, the `LLMChain` is invoked with an input prompt as an example of the model generating a response based on the provided prompt.

```
with PromptWatch() as pw:
    my_chain("The quick brown fox jumped over")
```

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file64.png" alt="Figure 9.8: Prompt tracking at PromptWatch.io." height="734" width="2094"><figcaption><p>Figure 9.8: Prompt tracking at PromptWatch.io.</p></figcaption></figure>

This seems quite useful. By leveraging PromptWatch.io, developers and data scientists can effectively monitor and analyze LLMs’ prompts, outputs, and costs in real-world scenarios.PromptWatch.io offers comprehensive chain execution tracking and monitoring capabilities for LLMs. With PromptWatch.io, you can track all aspects of LLM chains, actions, retrieved documents, inputs, outputs, execution time, tool details, and more for complete visibility into your system. The platform allows for in-depth analysis and troubleshooting by providing a user-friendly, visual interface that enables users to identify the root causes of issues and optimize prompt templates. PromptWatch.io can also help with unit testing and for versioning prompt templates.Let’s summarize this chapter!

### Summary

Successfully deploying LLMs and other generative AI models in a production setting is a complex but manageable task that requires careful consideration of numerous factors. It requires addressing challenges related to data quality, bias, ethics, regulatory compliance, interpretability, resource requirements, and ongoing monitoring and maintenance, among others.The evaluation of LLMs is an important step in assessing their performance and quality. LangChain supports comparative evaluation between models, checking outputs against criteria, simple string matching, and semantic similarity metrics. These provide different insights into model quality, accuracy, and appropriate generation. Systematic evaluation is key to ensuring large language models produce useful, relevant, and sensible outputs. Monitoring LLMs is a vital aspect of deploying and maintaining these complex systems. With the increasing adoption of LLMs in various applications, ensuring their performance, effectiveness, and reliability is of utmost importance. We’ve discussed the significance of monitoring LLMs, highlighted key metrics to track for a comprehensive monitoring strategy, and have given examples of how to track metrics in practice.LangSmith provides powerful capabilities to track, benchmark, and optimize large language models built with LangChain. Its automated evaluators, metrics, and visualizations help accelerate LLM development and validation.Let’s see if you remember the key points from this chapter!

### Questions

Please have a look to see if you can come up with the answers to these questions from. If you are unsure about any of them, you might want to refer to the corresponding section in the chapter:

1. In your opinion what is the best term for describing the operationalization of language models, LLM apps, or apps that rely on generative models in general?
2. How can we evaluate LLMs apps?
3. Which tools can help for evaluating LLM apps?
4. What are considerations for production deployment of agents?
5. Name a few tools for deployment?
6. What are important metrics for monitoring LLMs in production?
7. How can we monitor these models?
8. What’s LangSmith?
