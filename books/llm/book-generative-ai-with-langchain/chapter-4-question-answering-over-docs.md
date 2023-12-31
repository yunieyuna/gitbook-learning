# Chapter 4: Question Answering over Docs

## 4 Querying with Tools

### Join our book community on Discord

[https://packt.link/EarlyAccessCommunity](https://packt.link/EarlyAccessCommunity)

![Qr code Description automatically generated](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file26.png)

In today's fast-paced business and research landscape, keeping up with the ever-increasing volume of information can be a daunting task. For engineers and researchers in fields like computer science and artificial intelligence staying updated with the latest developments is crucial. However, reading and comprehending numerous papers can be time-consuming and labor-intensive. This is where automation comes into play. In this chapter, we’ll describe an approach to automate the summarization of research papers and answering questions, making it easier for researchers to digest and stay informed. By leveraging language models and a series of questions, the summarization we’ll develop can summarize the core assertions, implications, and mechanics of a paper in a concise and simplified format. This can not only save time and effort while researching a topic, but it also ensures we can effectively navigate the accelerated pace of scientific progress. We’ll also have a play around with functions in OpenAI models, and their application to information extraction. We’ll see how they work (or not yet) for an application as parsers of curriculum vitae (CVs). This function syntax is specific to OpenAI’s API and has many applications, however, LangChain provides a platform that allows the creation of tools for any large language models (LLMs), which enhance their capabilities. These tools enable LLMs to interact with APIs, access real-time information, and perform various tasks such as retrieval search, database queries, writing emails, or even making phone calls. We’ll implement a question-answering app with Retrieval Augmented Generation (RAG). This is a technique to update large language models (LLMs) like GPT by injecting relevant data into the context. Finally, we’ll discuss different strategies of decision making in agents. We’ll implement two strategies, plan-and-execute (or plan-and-solve) and one-shot-agent, and we’ll integrate them into a visual interface as a visual app in the browser (using Streamlit) for question answering.The main sections are:

* What are hallucinations?
* How to summarize long documents?
* Extracting information from documents
* Answer questions with tools
* Reasoning strategies

We'll begin the chapter by discussing the problem of reliability of LLMs.

### What are hallucinations?

The rapid development of generative language models, such as GPT-3, Llama, and Claude 2, has brought attention to their limitations and potential risks. One major concern are hallucinations, where the models generate output that is nonsensical, incoherent, or unfaithful to the provided input. Hallucination poses performance and safety risks in real-world applications, such as medical or machine translation. There’s another aspect to hallucinations, which is the case, where LLMs generate text that includes sensitive personal information, such as email addresses, phone numbers, and physical addresses. This poses significant privacy concerns as it suggests that language models can memorize and recover such private data from their training corpus, despite not being present in the source input.

> **Hallucination** in the context of LLMs refers to the phenomenon where generated text is unfaithful to the intended content or nonsensical. This term draws a parallel to psychological hallucinations, which involve perceiving something that does not exist. In NLG, hallucinated text may appear fluent and natural grounded in the provided context, but lacks specificity or verifiability. **Faithfulness**, where the generated content stays consistent and truthful to the source, is considered an antonym of hallucination.
>
> > **Intrinsic hallucinations** occur when the generated output contradicts the source content, while **extrinsic hallucinations** involve generating information that cannot be verified or supported by the source material. Extrinsic hallucinations can sometimes include factually correct external information, but their unverifiability raises concerns from a factual safety perspective.

Efforts to address hallucination are ongoing, but there is a need for a comprehensive understanding across different tasks to develop effective mitigation methods.Hallucinations in LLMs can be caused by various factors:

1. Imperfect representation learning by the encoder.
2. Erroneous decoding, including attending to the wrong part of the source input and the choice of decoding strategy.
3. Exposure bias, which is the discrepancy between training and inference time.
4. Parametric knowledge bias, where pre-trained models prioritize their own knowledge over the input, leading to the generation of excessive information.

> **Hallucination mitigation methods** can be categorized into two groups: Data-Related Methods and Modeling and Inference Methods (after “Survey of Hallucination in Natural Language Generation”, Ziwei Ji and others, 2022):
>
> > **Data-Related Methods:**
>
> > Building a Faithful Dataset: Constructing datasets with clean and faithful targets from scratch or rewriting real sentences while ensuring semantic consistency.
>
> > Cleaning Data Automatically: Identifying and filtering irrelevant or contradictory information in the existing corpus to reduce semantic noise.
>
> > Information Augmentation: Augmenting inputs with external information, such as retrieved knowledge or synthesized data, to improve semantic understanding and address source-target divergence.
>
> > **Modeling and Inference Methods:**
>
> > Architecture: Modifying encoder architecture to enhance semantic interpretation, attention mechanisms to prioritize source information, and decoder structures to reduce hallucination and enforce implicit or explicit constraints.
>
> > Training: Incorporating planning, reinforcement learning (RL), multi-task learning, and controllable generation techniques to mitigate hallucination by improving alignment, optimizing reward functions, and balancing faithfulness and diversity.
>
> > Post-Processing: Correcting hallucinations in the output through generate-then-refine strategies or refining the results specifically for faithfulness using post-processing correction methods.

A result of hallucinations, where automatic fact checking can be applied, is the danger of spreading incorrect information or misuse for political purposes. **Misinformation**, including **disinformation**, deceptive news, and rumors, poses a significant threat to society, especially with the ease of content creation and dissemination through social media. The threats to society include distrust in science, public health narratives, social polarization, and democratic processes. Journalism and archival studies have extensively studied the issue, and fact-checking initiatives have grown in response. Organizations dedicated to fact-checking provide training and resources to independent fact-checkers and journalists, allowing for the scaling of expert fact-checking efforts. Addressing misinformation is crucial to preserving the integrity of information and combating its detrimental impact on society.In the literature, this kind of problem is called textual entailment, where models predict the directional truth relation between a text pair (i.e. “sentence t entails h” if, typically, a human reading t would infer that h is most likely true). In this chapter, we’ll focus on **automatic fact-checking** through information augmentation and post-processing. Facts can be retrieved either from LLMs or using external tools. In the former case, pre-trained language models can take the place of the knowledge base and retrieval modules, leveraging their vast knowledge to answer open-domain questions and using prompts to retrieve specific facts. We can see the general idea in this diagram (source: https://github.com/Cartus/Automated-Fact-Checking-Resources by Zhijiang Guo):

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file27.png" alt="Figure 4.1: Automatic fact-checking pipeline in three stages." height="481" width="1652"><figcaption><p>Figure 4.1: Automatic fact-checking pipeline in three stages.</p></figcaption></figure>

We can distinguish three stages:

1. Claim detection - identify claims that require verification
2. Retrieval - retrieve evidence to find sources supporting or refuting the claim
3. Claim verification - assess the veracity of the claim based on the evidence

Starting with 24-layer BERT-Large in 2018, language models have been pre-trained on large knowledge bases such as Wikipedia, and therefore would be able to answer knowledge questions from Wikipedia or - since their training set increasingly includes other sources - the internet, textbooks, arxiv, and Github. Querying for facts works with simple prompts such as masking prompts. For example, in order to answer the question “Where is Microsoft’s headquarter?”, the question would be rewritten as “Microsoft’s headquarter is in \[MASK]” and fed into a language model for the answer. In this approach, the activations in the final Interestingly, if an LLM that has not received the source text (unconditional LLM) yields a smaller loss in generating targets than an LLM that has received the source text (conditioned LLM), this indicates that the generated token is hallucinatory (Fillippova, 2020). The ratio of hallucinated tokens to the total number of target tokens can serve as a measure of the degree of hallucination in the generated output. In LangChain, we have a chain available for fact checking with prompt-chaining, where a model actively questions the assumptions that went into a statement. In this self-checking chain, `LLMCheckerChain`, the model is prompted sequentially times, first to make the assumptions explicit - this looks like this:

```
Here’s a statement: {statement}\nMake a bullet point list of the assumptions you made when producing the above statement.\n
```

Please note that this is a string template, where the elements in curly brackets will be replaced by variables. Next, these assumptions are fed back to the model in order to check them one by one with a prompt like this:

```
Here is a bullet point list of assertions:
    {assertions}
    For each assertion, determine whether it is true or false. If it is false, explain why.\n\n
```

Finally, the model is tasked to make a final judgment:

```
In light of the above facts, how would you answer the question '{question}'
```

The `LLMCheckerChain` does this all by itself as this example shows:

```
from langchain.chains import LLMCheckerChain
from langchain.llms import OpenAI
llm = OpenAI(temperature=0.7)
text = "What type of mammal lays the biggest eggs?"
checker_chain = LLMCheckerChain.from_llm(llm, verbose=True)
checker_chain.run(text)
```

The model can return different results to this question, some of which are wrong, and some of which it would correctly identify as false. When I was trying this out, I got results such as the blue whale, the North American beaver, or the extinct Giant Moa. I think this is the right answer:

```
Monotremes, a type of mammal found in Australia and parts of New Guinea, lay the largest eggs in the mammalian world. The eggs of the American echidna (spiny anteater) can grow as large as 10 cm in length, and dunnarts (mouse-sized marsupials found in Australia) can have eggs that exceed 5 cm in length.
• Monotremes can be found in Australia and New Guinea
• The largest eggs in the mammalian world are laid by monotremes
• The American echidna lays eggs that can grow to 10 cm in length
• Dunnarts lay eggs that can exceed 5 cm in length
• Monotremes can be found in Australia and New Guinea – True
• The largest eggs in the mammalian world are laid by monotremes – True
• The American echidna lays eggs that can grow to 10 cm in length – False, the American echidna lays eggs that are usually between 1 to 4 cm in length. 
• Dunnarts lay eggs that can exceed 5 cm in length – False, dunnarts lay eggs that are typically between 2 to 3 cm in length.
The largest eggs in the mammalian world are laid by monotremes, which can be found in Australia and New Guinea. Monotreme eggs can grow to 10 cm in length.
> Finished chain.
```

So, while this doesn’t guarantee correct answers, it can put a stop to some incorrect results.As for augmented retrieval (or RAG), we’ve seen the approach in this chapter, in the section about question answering. Fact-checking approaches involve decomposing claims into smaller checkable queries, which can be formulated as question-answering tasks. Tools designed for searching domain datasets can assist fact-checkers in finding evidence effectively. Off-the-shelf search engines like Google and Bing can also retrieve both topically and evidentially relevant content to capture the veracity of a statement accurately. In the next section, we’ll apply this approach to return results based on web searches and other tools. In the next section, we’ll implement a chain to summarize documents. We can ask any question to be answered from these documents.

### How to summarize long documents?

In this section, we’ll discuss automating the process of summarizing long texts and research papers. In today's fast-paced business and research landscape, keeping up with the ever-increasing volume of information can be a daunting task. For engineers and researchers in fields like computer science and artificial intelligence staying updated with the latest developments is crucial. However, reading and comprehending numerous papers can be time-consuming and labor-intensive. This is where automation comes into play. As engineers, we are driven by their desire to build and innovate, avoid repetitive tasks by automating them through the creation of pipelines and processes. This approach, often mistaken for laziness, allows engineers to focus on more complex challenges and utilize their skills more efficiently.Here, we’ll build an automation tool that can quickly summarize the content of long texts in a more digestible format. This tool is intended to help researchers keep up with the volume of papers being published daily, particularly in fast-moving fields such as artificial intelligence. By automating the summarization process, researchers can save time and effort, while also ensuring that they stay informed about the latest developments in their field. The tool will be based on LangChain and utilize large language models (LLMs) to summarize the core assertions, implications, and mechanics of a paper in a more concise and simplified manner. It can also answer specific questions about the paper, making it a valuable resource for literature reviews and accelerating scientific research. The author plans to further develop the tool to allow automatic processing of multiple documents and customization for specific research domains. Overall, the approach aims to benefit researchers by providing a more efficient and accessible way to stay updated with the latest research.LangChain supports a map reduce approach for processing documents using LLMs, which allows for efficient processing and analysis of documents. When reading in large texts, and splitting them into documents (chunks) that are suitable for the token context length of the LLM, a chain can be applied to each document individually and then combine the outputs into a single document. The core assertion is that the map reduce process involves two steps:

* Map step - the LLM chain is applied to each document individually, treating the output as a new document, and
* Reduce step - all the new documents are passed to a separate combine documents chain to obtain a single output.

This is illustrated in the figure here:

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file28.jpg" alt="Figure 4.2: Map reduce chain in LangChain (source: LangChain documentation)." height="847" width="2639"><figcaption><p>Figure 4.2: Map reduce chain in LangChain (source: LangChain documentation).</p></figcaption></figure>

The implications of this approach are that it allows for parallel processing of documents and enables the use of LLMs for reasoning, generating, or analyzing the individual documents as well as combining their outputs. The mechanics of the process involve compressing or collapsing the mapped documents to ensure they fit within the combine documents chain, which may also involve utilizing LLMs. The compression step can be performed recursively if needed. Here’s a simple example of loading a PDF document and summarizing it:

```
from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI
from langchain.document_loaders import PyPDFLoader
pdf_loader = PyPDFLoader(pdf_file_path)
docs = pdf_loader.load_and_split()
llm = OpenAI()
chain = load_summarize_chain(llm, chain_type="map_reduce")
chain.run(docs)
```

The variable `pdf_file_path` is a string with the path of a PDF file.The default prompt for both the map and reduce steps is this:

```
Write a concise summary of the following:
{text}
CONCISE SUMMARY:
```

We can specify any prompt for each step. In the text summarization application developed for this chapter on Github, we can see how to pass other prompts. On LangChainHub, we can see the qa-with sources prompt, which takes a reduce/combine prompt like this:

```
Given the following extracted parts of a long document and a question, create a final answer with references (\"SOURCES\"). \nIf you don't know the answer, just say that you don't know. Don't try to make up an answer.\nALWAYS return a \"SOURCES\" part in your answer.\n\nQUESTION: {question}\n=========\nContent: {text}
```

In this prompt, we would formulate a concrete question, but equally we could give the LLM a more abstract instruction to extract assumption and implications. The text would be the summaries from the map steps. An instruction like that would help against hallucinations.Other examples of instructions could be translating the document into a different language or rephrasing in a certain style.Once we start doing a lot of calls, especially here in the map step, we’ll see costs increasing. We are doing a lot of calls and using a lot of tokens in total. Time to give this some visibility!

#### Token usage

When using models, especially in long loops such as with map operations, it’s important to track the token usage and understand how much money you are spending. For any serious usage of generative AI, we need to understand the capabilities, pricing options, and use cases for different language models. OpenAI provides different models namely GPT-4, ChatGPT, and InstructGPT that cater to various natural language processing needs. GPT-4 is a powerful language model suitable for solving complex problems with natural language processing. It offers flexible pricing options based on the size and number of tokens used.ChatGPT models, like GPT-3.5-Turbo, specialize in dialogue applications such as chatbots and virtual assistants. They excel in generating responses with accuracy and fluency. The pricing for ChatGPT models is based on the number of tokens used.InstructGPT models are designed for single-turn instruction following and are optimized for quick and accurate response generation. Different models within the InstructGPT family, such as Ada and Davinci, offer varying levels of speed and power. Ada is the fastest model, suitable for applications where speed is crucial, while Davinci is the most powerful model, capable of handling complex instructions. The pricing for InstructGPT models depends on the model's capabilities and ranges from low-cost options like Ada to more expensive options like Davinci.OpenAI's DALL·E, Whisper, and API services for image generation, speech transcription, translation, and access to language models. DALL·E is an AI-powered image generation model that can be seamlessly integrated into apps for generating and editing novel images and art. OpenAI offers three tiers of resolution, allowing users to choose the level of detail they need. Higher resolutions offer more complexity and detail, while lower resolutions provide a more abstract representation. The price per image varies based on the resolution.Whisper is an AI tool that can transcribe speech into text and translate multiple languages into English. It helps capture conversations, facilitates communication, and improves understanding across languages. The cost for using Whisper is based on a per-minute rate.OpenAI's API provides access to powerful language models like GPT-3, enabling developers to create advanced applications. When signing up for the API, users are given an initial token usage limit, representing the number of tokens available to interact with the language model within a specific timeframe. As users' track record and usage increase, OpenAI may increase the token usage limit, granting more access to the model. Users can also request a quota increase if they require more tokens for their applications.We can track the token usage in OpenAI models by hooking into the OpenAI callback:

```
with get_openai_callback() as cb:
    response = llm_chain.predict(text=”Complete this text!”)
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")
```

In this example, the line with `llm_chain` could be any usage of an OpenAI model. We should see an output with the costs and tokens.There are two other ways of getting the token usage. Alternative to the OpenAI Callback, the `generate()` method of the `llm` class returns a response of type `LLMResult` instead of string. This includes token usages and finish reason, for example (from the LangChain docs):

```
input_list = [
    {"product": "socks"},
    {"product": "computer"},
    {"product": "shoes"}
]
llm_chain.generate(input_list)
```

The result looks like this:

```
    LLMResult(generations=[[Generation(text='\n\nSocktastic!', generation_info={'finish_reason': 'stop', 'logprobs': None})], [Generation(text='\n\nTechCore Solutions.', generation_info={'finish_reason': 'stop', 'logprobs': None})], [Generation(text='\n\nFootwear Factory.', generation_info={'finish_reason': 'stop', 'logprobs': None})]], llm_output={'token_usage': {'prompt_tokens': 36, 'total_tokens': 55, 'completion_tokens': 19}, 'model_name': 'text-davinci-003'})
```

Finally, the chat completions response format in the OpenAI API includes a usage object with token information, for example it could look like this (excerpt):

```
  {
  "model": "gpt-3.5-turbo-0613",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 17,
    "prompt_tokens": 57,
    "total_tokens": 74
  }
}
```

Next, we’ll look at how to extract certain pieces of information from documents using OpenAI functions with LangChain.

### Extracting information from documents

In June 2023, OpenAI announced updates to OpenAI's API, including new capabilities for function calling, which opens up an enhanced functionality. Developers can now describe functions to the gpt-4-0613 and gpt-3.5-turbo-0613 models and have the models intelligently generate a JSON object containing arguments to call those functions. This feature aims to enhance the connection between GPT models and external tools and APIs, providing a reliable way to retrieve structured data from the models.Function calling enables developers to create chatbots that can answer questions using external tools or OpenAI plugins. It also allows for converting natural language queries into API calls or database queries, and extracting structured data from text. The mechanics of the update involve using new API parameters, namely `functions`, in the `/v1/chat/completions` endpoint. The functions parameter is defined through a name, description, parameters, and the function to call itself. Developers can describe functions to the model using JSON Schema and specify the desired function to be called. In LangChain, we can use these function calls in OpenAI for information extraction or for calling plugins. For information extraction, we can specific entities and their properties from a text and their properties from a document in an extraction chain with OpenAI chat models. For example, this can help identifying the people mentioned in the text. By using the OpenAI functions parameter and specifying a schema, it ensures that the model outputs the desired entities and properties with their appropriate types.The implications of this approach are that it allows for precise extraction of entities by defining a schema with the desired properties and their types. It also enables specifying which properties are required and which are optional.The default format for the schema is a dictionary, but we can also define properties and their types in Pydantic providing control and flexibility in the extraction process.Here’s an example for a desired schema for information in a Curricum Vitae (CV):

```
from typing import Optional
from pydantic import BaseModel
class Experience(BaseModel):
    start_date: Optional[str]
    end_date: Optional[str]
    description: Optional[str]
class Study(Experience):
    degree: Optional[str]
    university: Optional[str]
    country: Optional[str]
    grade: Optional[str]
class WorkExperience(Experience):
    company: str
    job_title: str
class Resume(BaseModel):
    first_name: str
    last_name: str
    linkedin_url: Optional[str]
    email_address: Optional[str]
    nationality: Optional[str]
    skill: Optional[str]
    study: Optional[Study]
    work_experience: Optional[WorkExperience]
    hobby: Optional[str]
```

We can use this for information extraction from a CV. Here’s an example CV from [https://github.com/xitanggg/open-resume](https://github.com/xitanggg/open-resume)

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file29.png" alt="Figure 4.3: Extract of an example CV." height="504" width="842"><figcaption><p>Figure 4.3: Extract of an example CV.</p></figcaption></figure>

We are going to try to parse the information from this resume. Utilizing the `create_extraction_chain_pydantic()` function in LangChain, we can provide our schema as input, and an output will be an instantiated object that adheres to it. In its most simple terms, we can try this code snippet:

```
from langchain.chains import create_extraction_chain_pydantic
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
pdf_loader = PyPDFLoader(pdf_file_path)
docs = pdf_loader.load_and_split()
# please note that function calling is not enabled for all models!
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613")
chain = create_extraction_chain_pydantic(pydantic_schema=Resume, llm=llm)
return chain.run(docs)
```

We should be getting an output like this:

```
[Resume(first_name='John', last_name='Doe', linkedin_url='linkedin.com/in/john-doe', email_address='hello@openresume.com', nationality=None, skill='React', study=None, work_experience=WorkExperience(start_date='May 2023', end_date='Present', description='Lead a cross-functional team of 5 engineers in developing a search bar, which enables thousands of daily active users to search content across the entire platform. Create stunning home page product demo animations that drives up sign up rate by 20%. Write clean code that is modular and easy to maintain while ensuring 100% test coverage.', company='ABC Company', job_title='Software Engineer'), hobby=None)]
```

It’s far from perfect - only one work experience gets parsed out. But it’s a good start given the little effort we’ve put in so far. Please see the example on Github for the full example. We could add more functionality, for example to guess personality or leadership capability.OpenAI injects these function calls into the system message in a certain syntax, which their models have been optimized for. This implies that functions count against the context limit and are correspondingly billed as input tokens. LangChain natively has functionality to inject function calls as prompts. This means we can use model providers other than OpenAI for function calls within LLM apps. We’ll look at this now, and we’ll build this into an interactive web-app with Streamlit.

### Answer questions with tools

LLMs are trained on general corpus data and may not be as effective for tasks that require domain-specific knowledge. On their own LLMs can’t interact with the environment and access external data sources, however, LangChain provides a platform for creating tools that access real-time information, and perform tasks such as weather forecasting, making reservations, suggesting recipes, and managing tasks. Tools within the framework of agents and chains allow for the development of applications powered by LLMs that are data-aware and agentic, and opens up a wide range of approaches to solving problems with LLMs, expanding their use cases and making them more versatile and powerful.One important aspect of tools is their capability to work within specific domains or process specific inputs. For example, an LLM lacks inherent mathematical capabilities. However, a mathematical tool like a calculator can accept mathematical expressions or equations as an input and calculate the outcome. The LLM combined with such a mathematical tool perform calculations and provide accurate answers.Generally, this combination of retrieval methods and LLMs is called **Retrieval Augmented Generation** (**RAG**), and addresses the limitations of LLMs by retrieving relevant data from external sources and injecting it into the context. This retrieved data serves as additional information to augment the prompts given to the LLMs.By grounding LLMs with use-case specific information through RAG, the quality and accuracy of responses are improved. Through retrieval of relevant data, RAG helps in reducing hallucinations responses from LLMs.For example, an LLM used in a healthcare application could retrieve relevant medical information from external sources such as medical literature or databases during inference. This retrieved data can then be incorporated into the context to enhance the generated responses and ensure they are accurate and aligned with domain-specific knowledge. The benefits of implementing RAG in this scenario are twofold. Firstly, it allows for incorporating up-to-date information into responses despite the model's training data cutoff date. This ensures that users receive accurate and relevant information even for recent events or evolving topics.Secondly, RAG enhances ChatGPT's ability to provide more detailed and contextual answers by leveraging external sources of information. By retrieving specific context from sources like news articles or websites related to a particular topic, the responses will be more accurate.

> RAG (Retrieval Augmented Generation) works by retrieving information from data sources to supplement the prompt given to the language model, providing the model with the needed context to generate accurate responses. RAG involves several steps:

* **Prompt**: The user provides a prompt to the chatbot, describing their expectations for the output.
* **Research**: A contextual search is performed and retrieves relevant information from various data sources. This could involve querying a database, searching indexed documents based on keywords, or invoking APIs to retrieve data from external sources.
* **Update Resource**: The retrieved context is injected into the original prompt, augmenting it with additional factual information related to the user's query. This enhanced prompt improves accuracy as it provides access to factual data.
* **Narrate**: Based on this augmented input, the LLM generates a response that includes factually correct information and sends it back to the chatbot for delivery to the user.

Therefore, by combining external data sources and injecting relevant context into prompts, RAG enhances LLMs' ability to generate responses that are accurate, up-to-date, and aligned with specific domains or topics of interest.An illustration of augmenting LLMs through tools and reasoning is shown here (source: https://github.com/billxbf/ReWOO, implementation for the paper “Decoupling Reasoning from Observations for Efficient Augmented Language Models Resources” by Binfeng Xu and others, May 2023):

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file30.jpg" alt="Figure 4.4: Tool-augmented LM paradigm, leveraging foreseeable reasoning ability of language models to improve system parameter and prompt efficiency" height="715" width="1217"><figcaption><p>Figure 4.4: Tool-augmented LM paradigm, leveraging foreseeable reasoning ability of language models to improve system parameter and prompt efficiency</p></figcaption></figure>

Let’s see this in action!We have quite a few tools available in LangChain, and - if that’s not enough - it’s not hard to roll out our own tools. Let’s set up an agent with a few tools:

```
from langchain.agents import (
    AgentExecutor, AgentType, initialize_agent, load_tools
)
from langchain.chat_models import ChatOpenAI
def load_agent() -> AgentExecutor:
    llm = ChatOpenAI(temperature=0, streaming=True)
    # DuckDuckGoSearchRun, wolfram alpha, arxiv search, wikipedia
    # TODO: try wolfram-alpha!
    tools = load_tools(
        tool_names=["ddg-search", "wolfram-alpha", "arxiv", "wikipedia"],
        llm=llm
    )
    return initialize_agent(
        tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
```

It’s an important detail to know that an `AgentExecutor` is a chain, and therefore - if we wanted we could integrate it into a larger chain if we wanted. We could have initialized this chain using a different syntax like this:

```
return MRKLChain.from_chains(llm, chains, verbose=True)
```

In this syntax, we would pass the tools as chain configurations. MRKL stands for Modular Reasoning, Knowledge and Language. The Zero-Shot Agent is the most general-purpose action agent in the MRKL framework.Please notice the parameter `streaming` in the `ChatOpenAI` constructor, which is set to `True`. This makes for a better use experience, since it means that the text response will be updated as it comes in, rather than once all the text has been completed. Currently only the OpenAI, ChatOpenAI, and ChatAnthropic implementations support streaming. All the tools mentioned have their specific purpose that’s part of the description, which is passed to the language model. These tools here are plugged into the agent:

* DuckDuckGo - a search engine that focuses on privacy; an added advantage is that it doesn’t require developer signup
* Wolfram Alpha - an integration that combines natural language understanding with math capabilities, for questions like “What is 2x+5 = -3x + 7?”
* Arxiv - search in academic pre-print publications; this is useful for research-oriented questions
* Wikipedia - for any question about entities of significant notoriety

Please note that in order to use Wolfram Alpha, you have to set up an account and set the `WOLFRAM_ALPHA_APPID` environment variable with the developer token you create at [https://developer.wolframalpha.com/](https://developer.wolframalpha.com/)There are lot of other search tools integrated in LangChain apart from DuckDuckGo that let you utilize Google or Bing search engines or work with meta search engines. There’s an Open-Meteo - integration for weather information, however, this information is also available through search.Let’s make our agent available as a streamlit app.

> **Streamlit** is an open-source app framework for Machine Learning and Data Science teams. It allows users to create beautiful web apps in minutes using Python.

Let’s write the code for this using the `load_agent()` function we’ve just defined:

```
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
chain = load_agent()
st_callback = StreamlitCallbackHandler(st.container())
if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = chain.run(prompt, callbacks=[st_callback])
        st.write(response)
```

Please notice that we are using the callback handler in the call to the chain, which means that we’ll see responses as they come back from the model. We can start the app locally from the terminal like this:

```
PYTHONPATH=. streamlit run question_answering/app.py
```

> Deployment of Streamlit applications can be local or on a server. Alternatively, you can deploy this on the Streamlit Community Cloud or on Hugging Face Spaces.

* For **Streamlit Community Cloud** do this:
* 1\. Create a Github repository
* 2\. Go to Streamlit Community Cloud, click on "New app" and select the new repo
* 3\. Click "Deploy!"
* As for **Hugging Face Spaces** it works like this:
* 1\. Create a Github repo
* 2\. Create a Hugging Face account at https://huggingface.co/
* 3\. Go to “Spaces” and click “Create new Space”. In the form, fill in a name, type of space as “Streamlit”, and choose the new repo.

Here’s a screenshot from the app looks:

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file31.png" alt="Figure 4.5: Question answering app in Streamlit." height="470" width="764"><figcaption><p>Figure 4.5: Question answering app in Streamlit.</p></figcaption></figure>

The search works quite well, although depending on the tools used it might still come up with the wrong results. For the question about the mammal with the largest egg, using DuckDuckGo it comes back with a result that discusses eggs in birds and mammals, and sometimes concludes that the ostrich is the mammal with the largest egg, although Platypus also comes back sometimes. Here’s the log output (shortened) for the correct reasoning:

```
> Entering new AgentExecutor chain...
I'm not sure, but I think I can find the answer by searching online.
Action: duckduckgo_search
Action Input: "mammal that lays the biggest eggs"
Observation: Posnov / Getty Images. The western long-beaked echidna ...
Final Answer: The platypus is the mammal that lays the biggest eggs.
> Finished chain.
```

You can see that with a powerful framework for automation and problem solving at your behest, you can compress work that can take hundreds of hours into minutes. You can play around with different research questions to see how the tools are used. The actual implementation in the repository for the book allows you to try out different tools, and has an option for self-verification.Retrieval Augmented Generation (RAG) with LLMs can significantly improve the accuracy and quality of responses by injecting relevant data from outside sources into the context. By grounding LLMs with use-case-specific knowledge, we can reduce hallucinations, and make them more useful in real-world scenarios. RAG is more cost-effective and efficient than retraining the models. You can see a very advanced example of augmented information retrieval with LangChain in the BlockAGI project, which is inspired by BabyAGI and AutoGPT, at [https://github.com/blockpipe/BlockAGI](https://github.com/blockpipe/BlockAGI)In the following sections, we’ll compare the main types of agents by their decision making strategies.

### Reasoning strategies

The current generation of generative models like LLMs excel at finding patterns in real-world data, such as visual and audio information, and unstructured texts, however, they struggle with symbol manipulation operations required for tasks that involve structured knowledge representation and reasoning. Reasoning problems pose challenges for LLMs, and there are different reasoning strategies that can complement the pattern completion abilities inherent in neural networks that are generative models. By focusing on enabling symbol-manipulation operations on extracted information, these hybrid systems can enhance the capabilities of language models.**Modular Reasoning, Knowledge, and Language** (**MRKL**) is a framework that combines language models and tools to perform reasoning tasks. In LangChain this consists of three parts:

1. tools,
2. an `LLMChain`, and
3. the agent itself.

The tools are the available resources that the agent can use, such as search engines or databases. The LLMChain is responsible for generating text prompts and parsing the output to determine the next action. The agent class uses the output of the LLMChain to decide which action to take.We’ve discussed tool use strategies in _Chapter 2_, _Introduction to LangChain_. We can see the reasoning with observation pattern in this diagram:

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file32.png" alt="Figure 4.6: Reasoning with observation (source: https://arxiv.org/abs/2305.18323; Binfeng Xu and others, May 2023)." height="588" width="633"><figcaption><p>Figure 4.6: Reasoning with observation (source: https://arxiv.org/abs/2305.18323; Binfeng Xu and others, May 2023).</p></figcaption></figure>

**Observation-dependent reasoning** involves making judgments, predictions, or choices based on the current state of knowledge or the evidence fetched through observation. In each iteration, the agent is providing a context and examples to a language model (LLM). A user's task is first combined with the context and examples and given to the LLM to initiate reasoning. The LLM generates a thought and an action, and then waits for an observation from tools. The observation is added to the prompt to initiate the next call to the LLM. In LangChain, this is an **action agent** (also: **Zero-Shot Agent**, `ZERO_SHOT_REACT_DESCRIPTION`), which is the default setting when you create an agent.As mentioned, plans can also be made ahead of any actions. This strategy, in LangChain called the **plan-and-execute agent**, is illustrated in the diagram here:

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file33.png" alt="Figure 4.7: Decoupling Reasoning from Observations (source: https://arxiv.org/abs/2305.18323; Binfeng Xu and others, May 2023)." height="631" width="690"><figcaption><p>Figure 4.7: Decoupling Reasoning from Observations (source: https://arxiv.org/abs/2305.18323; Binfeng Xu and others, May 2023).</p></figcaption></figure>

The planner (an LLM), which can be fine-tuned for planning and tool usage, produces a list of plans (P) and calls a worker (in LangChain: the agent) to gather evidence (E) through using tools. P and E are combined with the task, and then fed into the Solver (an LLM) for the final answer. We can write a pseudo algorithm like this:

* Plan out all the steps (Planner)
* For step in steps:
  * Determine the proper tools to accomplish the step

The Planner and the Solver can be distinct language models. This opens up the possibility of using smaller, specialized models for Planner and Solver, and using fewer tokens for each of the calls.We can implement plan-and-solve in our research app, let’s do it!First, let’s add a `strategy` variable to the `load_agent()` function. It can take two values, either “plan-and-solve” or “one-shot-react”. For “one-shot-react”, the logic stays the same. For “plan-and-solve”, we’ll define a planner and an executor, which we’ll use to create a `PlanAndExecute` agent executor:

```
from typing import Literal
from langchain.experimental import load_chat_planner, load_agent_executor, PlanAndExecute
ReasoningStrategies = Literal["one-shot-react", "plan-and-solve"]
def load_agent(
        tool_names: list[str],
        strategy: ReasoningStrategies = "one-shot-react"
) -> Chain:
    llm = ChatOpenAI(temperature=0, streaming=True)
    tools = load_tools(
        tool_names=tool_names,
        llm=llm
    )
    if strategy == "plan-and-solve":
        planner = load_chat_planner(llm)
        executor = load_agent_executor(llm, tools, verbose=True)
        return PlanAndExecute(planner=planner, executor=executor, verbose=True)
    return initialize_agent(
        tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
```

For the sake of brevity, I’ve omitted imports that we had already earlier.Let’s define a new variable that’s set through a radio button in Streamlit. We’ll pass this variable over to the `load_agent()` function:

```
strategy = st.radio(
    "Reasoning strategy",
    ("plan-and-solve", "one-shot-react", ))
```

You might have noticed that the `load_agent()` take a list of strings, `tool_names`. This can be chosen in the user interface (UI) as well:

```
tool_names = st.multiselect(
    'Which tools do you want to use?',
    [
        "google-search", "ddg-search", "wolfram-alpha", "arxiv",
        "wikipedia", "python_repl", "pal-math", "llm-math"
    ],
    ["ddg-search", "wolfram-alpha", "wikipedia"])
```

Finally, still in the app, the agent is loaded like this:

```
agent_chain = load_agent(tool_names=tool_names, strategy=strategy)
```

We can see the UI here:

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file34.png" alt="Figure 4.8: Implementing plan-and-execute in our research app." height="486" width="741"><figcaption><p>Figure 4.8: Implementing plan-and-execute in our research app.</p></figcaption></figure>

Please have a look at the app and see the different steps for the question “What is a plan and solve agent in the context of large language models?”. Just briefly, the first step, the plan looks as follows:

1. Define large language models: Large language models are AI models that are trained on vast amounts of text data and can generate human-like text based on the input they receive.
2. Understand the concept of a plan in the context of large language models: In the context of large language models, a plan refers to a structured outline or set of steps that the model generates to solve a problem or answer a question.
3. Understand the concept of a solve agent in the context of large language models: A solve agent is a component of a large language model that is responsible for generating plans to solve problems or answer questions.
4. Recognize the importance of plans and solve agents in large language models: Plans and solve agents help organize the model's thinking process and provide a structured approach to problem-solving or question-answering tasks.
5. Given the above steps taken, please respond to the user's original question: In the context of large language models, a plan is a structured outline or set of steps generated by a solve agent to solve a problem or answer a question. A solve agent is a component of a large language model that is responsible for generating these plans.

Accordingly, the first step is to perform a look up of LLMs:

```
Action:
{
"action": "Wikipedia",
"action_input": "large language models"
}
```

We didn’t discuss another aspect of this, which is the prompting strategy used in these steps. For example, different prompting strategies offer ways to address challenges in complex reasoning problems for LLMs. One approach is **few-shot chain-of-thought** (**CoT**) prompting, where LLMs are guided through step-by-step reasoning demonstrations. For example, in arithmetic reasoning, an LLM can be shown demonstration examples of solving equations to aid its understanding of the process.Another strategy is **zero-shot-CoT** prompting, which eliminates the need for manual demonstrations. Instead, a generic prompt like "Let's think step by step" is appended to the problem statement provided to the LLM. This allows the model to generate reasoning steps without prior explicit examples. In arithmetic reasoning, the problem statement could be augmented with this prompt and fed into an LLM.**Plan-and-Solve (PS) prompting**, involves dividing a complex task into smaller subtasks and executing them step by step according to a plan. For instance, in math reasoning problems like solving equations or word problems involving multiple steps, PS prompting enables an LLM to devise a plan for approaching each sub step, such as extracting relevant variables and calculating intermediate results.To further enhance the quality of reasoning steps and instructions, **PS+** prompting is introduced. It includes more detailed instructions, such as emphasizing the extraction of relevant variables and considering calculation and commonsense. PS+ prompting ensures that the LLMs have a better understanding of the problem and can generate accurate reasoning steps. For example, in arithmetic reasoning, PS+ prompting can guide the LLM to identify key variables, perform calculations correctly, and apply commonsense knowledge during the reasoning process.This concludes our discussion of reasoning strategies. All strategies have their problems with can manifest as calculation errors, missing-step errors, and semantic misunderstandings. However, they help improve the quality of generated reasoning steps, increase accuracy in problem-solving tasks, and enhance LLMs' ability to handle various types of reasoning problems.

### Summary

In this chapter, we’ve talked about the problem of hallucinations, automatic fact-checking, and how to make LLMs more reliable. Of particular emphasis were tools and prompting strategies. We’ve first looked at and implemented prompting strategies to break down and summarize documents. This can be very helpful for digesting large research articles or analyses. Once we get into making a lot of chained calls to LLMs, this can mean we incur a lot of costs. Therefore, I’ve dedicated a section to token usage.Tools provide creative solutions to problems and open up new possibilities for LLMs in various domains. For example, a tool could be developed to enable an LLM to perform advanced retrieval search, query a database for specific information, automate email writing, or even handle phone calls.The OpenAI API implements functions, which we can use, among other things, for information extraction in documents. We’ve implemented a very simple version of a CV parser as an example of this functionality.Tools and function calling is not unique to OpenAI, however. With Streamlit, we can implement different agents that call tools. We’ve implemented an app that can help answer research questions by relying on external tools such as search engines or Wikipedia.We’ve then looked at different strategies employed by the agents to make decisions. The main distinction is the point of decision making. We’ve implemented a plan-and-solve and a one-shot agent into a Streamlit app.I hope this goes to show that in a few lines we can implement apps that can be very impressive in a few cases. It’s important to be clear, however, that the apps developed in this chapter have limitations. They can help you significantly increase your efficiency, however, you - as the human - have to apply judgment and improve the writing to make sure it’s coherent and makes sense.Let’s see if you remember some of the key takeaways from this chapter!

### Questions

Please have a look to see if you can come up with the answers to these questions from memory. I’d recommend you go back to the corresponding sections of this chapter, if you are unsure about any of them:

1. What is a hallucination?
2. How does automated fact-checking work?
3. What can we do in LangChain to make sure the output is valid?
4. What is map-reduce in LangChain?
5. How can we count the tokens we are using (and why should we)?
6. What does RAG stand for and what are the advantages of using it?
7. What tools are available in LangChain?
8. Please define plan-and-solve agents
9. Please define one-shot agents
10. How can we implement text input fields and radio buttons in Streamlit?
