# Chapter 2: LangChain: Core Fundamentals

## 2 Introduction to LangChain

### Join our book community on Discord

[https://packt.link/EarlyAccessCommunity](https://packt.link/EarlyAccessCommunity)

![Qr code Description automatically generated](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file8.png)

In this chapter, we discuss limitations of LLMs, and how combining LLMs with tools can overcome these challenges thereby building innovative language-based applications. There are a few powerful frameworks that empowers developers by providing robust tools for prompt engineering, chaining, data retrieval, and more. Whether you're a developer, data scientist or simply curious about technological advancements in natural language processing (NLP) or generative AI, you should learn about the most powerful and popular of these frameworks, LangChain.LangChain addresses pain points associated with working with LLMs and provides an intuitive framework to create customized NLP solutions. In LangChain, components like LLMs, internet searches, and database lookups can be chained together, which refers to executing different tasks one after another in a sequence based on requirements by the data or the tasks. By leveraging its features, developers can build dynamic and data-aware applications that harness the recent technological breakthroughs that we discussed in chapter 1. We'll include a few use cases to illustrate how the framework can help businesses and organizations in different domains.LangChain's support for agents and memory makes it possible to build a variety of applications that are more powerful and flexible than those that can be built by simply calling out to a language model via an API. We will talk about important concepts related to the framework such as agents, chains, action plan generation and memory. All these concepts are important to understand the how LangChain works.The main sections are:

* What are the limitations of LLMs?
* What is an LLM app?
* What is LangChain?
* How does LangChain work?

We'll start off the chapter by going over the limitations of LLMs.

### What are the limitations of LLMs?

**Large language models** (**LLMs**) have gained significant attention and popularity due to their ability to generate human-like text and understand natural language, which makes them useful in scenarios that revolve around content generation, text classification, and summarization. While **LLMs** offer impressive capabilities, they suffer from limitations that can hinder their effectiveness in certain scenarios. Understanding these limitations is crucial when developing applications. Some pain points associated with large language models include:

1. **Outdated Knowledge**: **LLMs** are unable to provide real-time or recent data as they rely solely on the training data provided to them.
2. **Inability to act**: **LLMs** cannot perform actions or interact with external systems, limiting their functionality. For example, they cannot initiate web searches, query databases in real-time, or use a calculator for multiplying numbers.
3. **Lack of context and additional information**: **LLMs** may struggle to understand and incorporate context from previous prompts or conversations. They may not remember previously mentioned details or fail to provide additional relevant information beyond the given prompt.
4. **Complexity and Learning Curve**: Developing applications using large language models often requires a deep understanding of AI concepts, complex algorithms, and APIs. This can pose a challenge for developers who may not have expertise in these areas.
5. **Hallucinations**: **LLMs** have a lot of general knowledge about the world implicit in their weights. However, they may have an insufficient understanding about certain subjects, and generate responses that are not factually correct or coherent. For example, they might produce information that does not exist or provide inaccurate details.
6. **Bias and Discrimination**: Depending on the data they were trained on, LLMs can exhibit biases, which can be of religious, ideological, political, and other nature.

LLMs don't have information on current events because they don't have a connection to the outside world and they wouldn't know about anything that they weren't trained on, such as anything after the cutoff date, which is when the training data were generated. More than that, they struggle with contextual understanding beyond training data limitations. For example, since the models cannot perform actions or interact with external systems directly, they wouldn't know the weather, don't have access to your documents.This cutoff day issue is illustrated here in the OpenAI ChatGPT chat interface asking about **LangChain**:

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file9.png" alt="Figure 1.1: ChatGPT - lack of up-to-date information." height="227" width="715"><figcaption><p>Figure 1.1: ChatGPT - lack of up-to-date information.</p></figcaption></figure>

In this case, the model was able to correctly catch the problem and give the correct feedback. However, if we ask the same question in the **GPT-3** playground, we'll get this response:

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file10.png" alt="Figure 1.2: OpenAI playground with GPT 3.5 - Hallucination." height="384" width="684"><figcaption><p>Figure 1.2: OpenAI playground with GPT 3.5 - Hallucination.</p></figcaption></figure>

In this case, we can see that the model makes up the term and invents a decentralized platform by the name. This is a hallucination. It's important to watch out for these problems.This problem can be remedied by accessing external data, such as weather APIs, user preferences, or relevant information from the web, and this is essential for creating personalized and accurate language-driven applications.**LLMs** are proficient at generating text but lack true understanding and reasoning capability. However, they might struggle with logical reasoning. As an example, even advanced **LLMs** perform poorly at high-school level math, and can't perform simple math operations that they haven't seen before.Again, we can illustrate this with a simple demonstration:

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file11.png" alt="Figure 1.3: ChatGPT math solving." height="308" width="464"><figcaption><p>Figure 1.3: ChatGPT math solving.</p></figcaption></figure>

So, the model comes up with the correct response for the first question, but fails with the second. Just in case if you were wondering what the true result is - if we use a calculator we get this:

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file12.png" alt="Figure 1.4: Multiplication with a Calculator (BC)." height="132" width="492"><figcaption><p>Figure 1.4: Multiplication with a Calculator (BC).</p></figcaption></figure>

The LLM hasn't stored the result of the calculation of hasn't encountered it often enough in the training data for it to be reliably remembered as in encoded in its weights. Therefore, it fails to correctly come up with the solution. A transformer-based LLM is not the suitable tool for the job in this case.The output of **LLMs** might need to be monitored and corrected for accuracy and for bias and inappropriate language before deployment of an app in domains such as customer service, education, and marketing. It's not hard to come up with examples for bias in Chatbots - just recall the Tay Chatbot, which turned a public relations disaster for Microsoft because of racial slurs and other xenophobic comments.For all of these concerns, **LLMs** need to be integrated with external data sources, memory, and capability in order to interact dynamically with their environment and respond appropriately based on the provided data. However, connecting large language models with different data sources and computations can be tricky and specific customized tools need to be developed and carefully tested. As a result, building data-responsive applications with Generative AI can be complex and can require extensive coding and data handling.Finally, working with **LLM** models directly can be challenging and time-consuming. This starts with the prompt engineering, but extends much further. The inherent challenge lies in navigating these sophisticated models, providing prompts that work, and parsing their output.

### What's an LLM app?

To address the aforementioned challenges and limitations, **LLMs** can be combined with calls to other programs or services. The main idea is that the ability of LLMs can be augmented through the use of tools by connecting them together. Combining **LLMs** with other tools into applications using specialized tooling, **LLM**-powered applications have the potential to transform our digital world. Often this is done via a chain of one or multiple prompted calls to **LLMs**, but can also make use of other external services (such as APIs or data sources) in order to achieve particular tasks.

> An **LLM app** is an application that uses large language models (LLMs) like ChatGPT to assist with various tasks. It operates by sending prompts to the language models to generate responses, and it can also integrate with other external services, like APIs or data sources, to accomplish specific goals.

In order to illustrate how an **LLM** app can look like, here's a very simple **LLM** app that includes a prompt and an **LLM** (source: [https://github.com/srush/MiniChain](https://github.com/srush/MiniChain)):

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file13.png" alt="Figure 1.5: A simple LLM app that combines a prompt with an LLM." height="236" width="607"><figcaption><p>Figure 1.5: A simple LLM app that combines a prompt with an LLM.</p></figcaption></figure>

**LLM** apps have significant potential for humans as they enhance our capabilities, streamline processes, and provide valuable assistance in various domains. Here are some key reasons why **LLM** apps are important:

* **Efficiency and Productivity**: **LLM** apps automate tasks, enabling faster and more accurate completion of repetitive or complex operations. They can handle data processing, analysis, pattern recognition, and decision-making with speed and accuracy that surpasses human capacity. This improves efficiency and productivity in areas such as data analysis, customer service, content generation, and more.
* **Task Simplification**: **LLM** apps simplify complex tasks by breaking them down into manageable steps or providing intuitive interfaces for users to interact with. These tools can automate complex workflows, making them accessible to a wider range of users without specialized expertise.
* **Enhanced Decision-Making**: **LLM** apps offer advanced analytics capabilities that enable data-driven decision-making. They can analyze large volumes of information quickly, identify trends or patterns that may not be apparent to humans alone, and provide valuable insights for strategic planning or problem-solving.
* **Personalization**: AI-powered recommendation systems personalize user experiences based on individual preferences and behavior patterns. These apps consider user data to provide tailored suggestions, recommendations, and personalized content across various domains like e-commerce, entertainment, and online platforms.

A particular area of growth is the usage of company data, especially customer data, with **LLMs**. However, we have to be careful and consider implications for privacy and data protection. We should never feed **personally identifiable** (**PII**) data into public API endpoints. For these use cases, deploying models on in-house infrastructure or private clouds is essential, and where fine-tuning and even training specialized models provide important improvements. This is what we'll talk about in chapter 9, _LLM Apps in Production_.Let's compare a few frameworks that can help to build **LLM** apps.

#### Framework Comparison

**LLM** application frameworks have been developed to provide specialized tooling that can harness the power of **LLMs** effectively to solve complex problems. A few libraries have emerged meeting the requirements of effectively combining generative AI models with other tools to build **LLM** applications.There are several open-source frameworks for **building** dynamic **LLM** applications. They all offer value in developing cutting-edge LLM applications. This graph shows their popularity over time (data source: github star history; [https://star-history.com/](https://star-history.com/)):

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file14.png" alt="Figure 1.6: Comparison of popularity between different framework in Python. We can see the number of stars on github over time for each project." height="1041" width="1380"><figcaption><p>Figure 1.6: Comparison of popularity between different framework in Python. We can see the number of stars on github over time for each project.</p></figcaption></figure>

We can see in the chart that Haystack is the oldest of the compared frameworks having been started early 2020 (as per github commits). It is also the least popular in terms of stars on github. Langchain, **LlamaIndex** (previously called GPTIndex), and **SuperAGI** were started late 2022 or early 2023, and they have all short to popularity in a very short time with **LangChain** growing most impressively. In this book, we'll see why its popularity is exploding right now.**LlamaIndex** focuses on advanced retrieval rather than on the broader aspects of **LLM** apps. Similarly, Haystack focuses on creating large-scale search systems with components designed specifically for scalable information retrieval using retrievers, readers, and other data handlers combined with semantic indexing via pre-trained models.**LangChain** excels at chaining **LLMs** together using agents for delegating actions to the models. Its use cases emphasize prompt optimization and context-aware information retrieval/generation, however with its Pythonic highly modular interface and its huge collection of tools, it is the number one tool to implement complex business logic.**SuperAGI** has similar features to **LangChain**. It even comes with a Marketplace, a repository for tools and agents. However, it's not as extensive and well-supported as **LangChain**.I haven't included **AutoGPT** (and similar tools like **AutoLlama**), a recursive application that breaks down tasks, because its reasoning capability, based on human and LLM feedback, is very limited compared to **LangChain**. As a consequence, it's often caught in logic loops and often repeats steps. I've also omitted a few libraries that concentrate on prompt engineering, for example Promptify. There are other LLM app frameworks in languages such as Rust, Javascript, Ruby, and Java. For example, Dust, written in Rust, focuses on the design of LLM Apps and their deployment.Let's look a bit more at **LangChain**.

### What is LangChain?

LangChain is a framework for developing applications powered by language models and enables users to build applications using **LLMs** more effectively. It provides a standard interface for connecting language models to other sources of data, as well as for building agents that can interact with their environment. LangChain is designed to be modular and extensible, making it easy to build complex applications that can be adapted to a variety of domains. LangChain is open source, and is written in Python, although companion projects exist implemented in JavaScript or - more precisely - Typescript (LangChain.js), and the fledgling Langchain.rb project for Ruby, which comes with a Ruby interpretor for code execution. In this book, we focus on the Python flavor of the framework.

> **LangChain** is an open-source framework that allows AI developers to combine LLMs like ChatGPT with other sources of computation and information.

Started in October 2022 by Harrison Chase as an open-source project on github, it is licensed under the MIT license, a common license, which allows commercial use, modification, distribution, and private use, however, restricts liability and warranty. LangChain is still quite new, however, it already features 100s of integrations and tools. There are active discussions on a discord chat server, there's blog, and regular meetups are taking place in both San Francisco and London. There's even a Chatbot, ChatLangChain, that can answer questions about the LangChain documentation built with with LangChain and FastAPI, which is available online through the documentation website!The project has attracted millions in venture capital funding from the likes of Sequoia Capital and Benchmark, who provided funding to Apple, Cisco, Google, WeWork, Dropbox, and many other successful companies. LangChain comes with many extensions and a larger ecosystem that is developing around it. As mentioned, it has an immense number of integrations already, with many new ones every week. This screenshot showcases a few of the integrations (source: `integrations.langchain.com/trending`):

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file15.png" alt="Figure 1.7: LangChain integrations." height="507" width="1060"><figcaption><p>Figure 1.7: LangChain integrations.</p></figcaption></figure>

For example, LangChainHub is a repository of artifacts that are useful for working with **LangChain** such as prompts, chains and agents, which combine together to form complex LLM applications. Taking inspiration from HuggingFace Hub, which is a collection of models, it aims repository is to be a central resource for sharing and discovering high quality **LangChain** primitives and applications. Currently, the hub solely contains a collection of prompts, but - hopefully - as the community is adding to this collection, you might be able to find chains and agents soon.Further, the **LlamaHub** library extends both LangChain and LlamaIndex with more data loaders and readers for example for **Google Docs**, **SQL Databases**, **PowerPoints**, **Notion**, **Slack**, and **Obsidian**. **LangFlow** is a UI, which allows chaining **LangChain** components in an executable flowchart by dragging sidebar components onto the canvas and connecting them together to create your pipeline. This is a quick way to experiment and prototype pipelines.This is illustrated in the following screenshot of a basic chat pipeline with a prompt template and a conversation buffer as a memory:

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file16.png" alt="Figure 1.8: LangFlow UI with a basic chat." height="566" width="822"><figcaption><p>Figure 1.8: LangFlow UI with a basic chat.</p></figcaption></figure>

In the sidebar of the browser interface (not shown here), you can see all the different **LangChain** components like a zero-shot prompt, data loaders, and language model wrappers. These flows can be either exported and loaded up in **LangChain** directly, or they can be called through API calls to the local server.**LangChain** and **LangFlow** can be deployed locally, for example using the Chainlit library, or on different platforms including Google Cloud. The langchain-serve library helps to deploy both **LangChain** and **LangFlow** on **Jina AI cloud** as LLM Apps as-a-service with a single command.**LangChain** provides an intuitive framework that makes it easier for developers, data scientists, and even those new to NLP technology to create applications using large language models. It's important to note that **LangChain** is neither a model nor a provider but essentially a framework that facilitates seamless interaction with diverse models. With **LangChain**, you don't need to be an expert in AI or complex algorithms — it simplifies the process and reduces the learning curve.

> Please note that although the main focus of LangChain is LLMs, and this is largely what we'll talk about in this book, there are also integrations for image generation.

By being data-aware and agentic, **LangChain** allows for easy integration with various data sources, including **Google Drive**, **Notion**, **Wikipedia**, **Apify Actors**, and more. This data-awareness enables applications to generate personalized and contextually relevant responses based on user preferences or real-time information from external sources.Let's explore why **LangChain** is important and then what it is used for.

#### Why is LangChain relevant?

**LangChain** fills a lot of the needs that we outlined before starting with the limitations of **LLMs** and the emergence of **LLM** apps. Simply put, it simplifies and streamlines the development process of applications using **LLMs**.It provides a way to build applications that are more powerful and flexible than those that can be built by simply calling out to a language model via an API. Particularly, **LangChain's** support for agents and memory allows developers to build applications that can interact with their environment in a more sophisticated way, and that can store and reuse information over time.**LangChain** can be used to improve the performance and reliability of applications in a variety of domains. In the healthcare domain, it can be used to build chatbots that can answer patient questions and provide medical advice. In this context, we have to be very careful with regulatory and ethical constraints around reliability of the information and confidentiality. In the finance domain, the framework can be used to build tools that can analyze financial data and make predictions. Here, we have to look at considerations around interpretability of these models. In the education domain, **LangChain** can be used to build tools that can help students learn new concepts. This is possibly one of the most exciting domains, where complete syllabi can be broken down by LLMs and delivered in customized interactive sessions, personalized to the individual learner.**LangChain's** versatility allows it to be used in several dynamic ways like building virtual personal assistants capable of recalling previous interactions; extracting analyzing structured datasets; creating Q\&A apps providing interaction with APIs offering real-time updates; performing code understanding extracting interacting source codes from GitHub enriching developer experiences robustly enhanced codified performances.There are several benefits to using **LangChain**, including:

* **Increased flexibility**: It provides a wide range of tools and features for building powerful applications. Further, it's modular design makes it easy to build complex applications that can be adapted to a variety of domains.
* **Improved performance**: The support for action plan generation can help to improve the performance of applications.
* **Enhanced reliability**: LangChain's support for memory can help to improve the reliability of applications by storing and reusing information over time, and - by access to external information - it can reduce hallucinations.
* **Open source**: An open business-friendly license coupled with a large community of developers and users means that you can customize it to your needs and rely on broad support.

In conclusion: there are many reasons to use **LangChain**. However, I should caution that since **LangChain** is still quite new, there might be some bugs or issues that have not yet been resolved. The documentation is already relatively comprehensive and big, however, in construction in a few places.

#### What can I build with LangChain?

**LangChain** empowers various NLP use cases such as virtual assistants, content generation models for summaries or translations, question answering systems, and more. It has been used to solve a variety of real-world problems. For example, **LangChain** has been used to build chatbots, question answering systems, and data analysis tools. It has also been used in a number of different domains, including healthcare, finance, and education.You can build a wide variety of applications with **LangChain**, including:

* **Chatbots**: It can be used to build chatbots that can interact with users in a natural way.
* **Question answering**: **LangChain** can be used to build question answering systems that can answer questions about a variety of topics.
* **Data analysis**: You can use it for automated data analysis and visualization to extract insights.
* **Code generation**: You can set up software pair programming assistants that can help to solve business problems.
* And much more!

### How does LangChain work?

With **LangChain**, you can build dynamic applications that harness the power of recent breakthroughs in **natural language processing** (**NLP**). By connecting components from multiple modules (chaining), you can create unique applications tailored around a large language model. From sentiment analysis to chatbots, the possibilities are vast.The principal value proposition of the LangChain framework consists of the following parts:

* **Components**:
  * **Model I/O**: This component provides LLM wrappers as a standardized interface for connecting to a language model.
  * **Prompt Templates**: This allows you to manage and optimize prompts.
  * **Memory**: Indexes are used to store and reuse information between calls of a chain/agent.
* **Agents**: Agents allow LLMs to interact with their environment. They decide the actions to take and take the action.
* **Chains**: These assemble components together in order to solve tasks. They can be comprised of sequences of calls to language models and other utilities.

Here's a visual representation of these parts:

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file17.png" alt="Figure 1.9: LangChain components." height="514" width="771"><figcaption><p>Figure 1.9: LangChain components.</p></figcaption></figure>

There's a lot to unwrap about these parts. Let's go into a bit of detail!Although **LangChain** doesn't supply models itself, it supports integration through **LLM** wrappers with various different language model providers enabling the app to interact with chat models as well text embedding model providers. Supported providers include **OpenAI**, HuggingFace, Azure, and Anthropic. Providing a standardized interface, means being able effortlessly swap out models in order to save money and energy or get better performance. A core building block of **LangChain** is the prompt class, which allows users to interact with **LLMs** by providing concise instructions or examples. Prompt engineering helps optimize prompts for optimal model performance. Templates give flexibility in terms of the input and the available collection of prompts are battle-tested in a range of applications. Vector stores come in when working with large documents, where the document needs to be chunked up in order to be passed to the **LLM**. These parts of the document would be stored as embeddings, which means that they are vector representation of the information. All these tools enhance the **LLMs'** knowledge and improve their performance in applications like question answering and summarization.There are numerous integrations for vector storage. These include Alibaba Cloud OpenSearch, AnalyticDB for PostgreSQL, Meta AI's Annoy library for **Approximate Nearest Neighbor** (**ANN**) **Search**, **Cassandra**, **Chroma**, **ElasticSearch**, **Facebook** **AI Similarity Search** (**Faiss**), **MongoDB** **Atlas** **Vector** **Search**, **PGVector** as a vector similarity search for **Postgres**, **Pinecone**, **Scikit-Learn** (`SKLearnVectorStore` for k-nearest neighbor search), and many more.Some other modules are these:

* **Data connectors and loaders**: These components provide interfaces for connecting to external data sources.
* **Callbacks**: Callbacks are used to log and stream intermediate steps of any chain.

Data connectors include modules for storing data and utilities for interacting with external systems like web searches or databases, and most importantly data retrieval. Examples are Microsoft Doc (docx), HyperText Markup Language (HTML), and other common formats such as PDF, text files, JSON, and CSV. Other tools will send emails to prospective customers, tweet funny puns to your followers, or send slack messages to your coworkers.Let's see a bit more in detail, what agents can be good for and how they make their decisions.

#### What is an agent?

Agents are used in **LangChain** to control the flow of execution of an application to interact with users, the environment, and other agents. Agents can be used to make decisions about which actions to take, to interact with external data sources, and to store and reuse information over time. Agents can transfer money, book flights, or talk to your customers.

> An **agent** is a software entity that can perform actions and tasks in the world and interact with its environment. In **LangChain**, agents take tools and chains and combine them for a task taking decisions on which to use.

Agents can establish a connection to the outside world. For example, a search engine or vector database can be utilized to find up-to-date and relevant information. This information can then be provided to models. This is called **retrieval augmentation**. By integrating external information sources, **LLMs** can draw from current information and extended knowledge. This is an example of how agents can overcome the weaknesses inherent in **LLMs** and enhance them by combining tools with the models.In the section about the limitations of **LLMs** we've seen that for calculations a simple calculator outperforms a model consisting of billions of parameters. In this case, an agent can decide to pass the calculation to a calculator or to a Python interpreter. We can see a simple app connecting an OpenAI language model output to a Python function here:

![chapter2/langflow\_python\_function.png](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file18.png) Figure 1.10: A simple LLM app with a Python function visualized in LangFlow.

We will see this in practice in _Chapter 3_, _Getting Started with LangChain_. Agents in **LangChain** can be used to perform a variety of tasks, such as, for example:

* Searching for information
* Calling APIs
* Accessing databases
* Code execution

Each agent can decide on which tool to use and when. Since this is crucial for understanding how **LangChain** works, let's see this in a bit of detail.

**Action execution**

Each agent is equipped with these subcomponents:

* Tools, which are functional components,
* Toolkits (these are collections of tools), and
* Agent Executors.

The **agent executor** is the execution mechanism that allows choosing between tools. The agent executor can be seen as the intermediary between the agent and the execution environment. It receives the requests or commands from the agent and translates them into actions that can be performed by the underlying system or software. It manages the execution of these actions and provides feedback or results back to the agent. We have different types of execution or decision patterns as we'll see.The **ReAct pattern** (published as "**ReACT**: Synergizing Reasoning and Acting in Language Models" by researchers at Princeton and Google DeepMind, May 2023), short for Reason and Act, where the agent actively assigns a task to an appropriate tool, customizes input for it, and parses its output in order to resolve the task. In the paper, a document store was utilized, where answers would be searched - this is implemented as the **ReAct document store pattern**.In **LangChain**, by default, agents follow the **Zero-shot ReAct pattern** (`ZERO_SHOT_REACT_DESCRIPTION`), where the decision is based only on the tool's description. This mechanism can be extended with memory in order to take into account the full conversation history. With **ReAct**, instead of asking an **LLM** to autocomplete on your text, you can prompt it to respond in a thought/act/observation loop. The prompt for the **LLM** is to respond step by step and associating actions with these steps. The result from these steps, for example search results, is then passed back into the **LLM** for its next deliberation as it iterates towards its goal. For the ZeroShot pattern, the prompt is really important, which is created from joining prefix, a string describing the tools and what they are good for, the format instructions, and the suffix:

```
PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""
FORMAT_INSTRUCTIONS = """Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""
SUFFIX = """Begin!
Question: {input}
Thought:{agent_scratchpad}"""
```

To see this in practice, for example, we can ask for the difference between **LangChain** agent executor and **LangChain** execution plan. Here's the log in **LangChain** - first the question goes to the language model:

```
I'm not familiar with these terms, so I should search for information about them.
Action: Search
Action Input: "difference between langchain agent executor and langchain execution plan"
Observation: The Concept of Agents in LangChain Action Agents decide an action to take and execute that action one step at a time. They are more conventional and suitable for small tasks. On the other hand, Plan-and-Execute Agents first decide a plan of actions to take, and then execute those actions one at a time.
Thought:Based on the observation, a langchain agent executor is an agent that decides and executes actions one step at a time, while a langchain execution plan is an agent that first decides a plan of actions and then executes them one at a time.
Final Answer: A langchain agent executor executes actions one step at a time, while a langchain execution plan first decides a plan of actions and then executes them.
```

There are a few more implemented mechanisms. Researchers at the University of Washington, **MosaicAI**, Meta AI Research, and Allen Institute (in the paper "Measuring and Narrowing the Compositionality Gap in Language Models" by in October 2022) found that **LLMs** might often not come up with the correct and complete answer for questions that require compositional reasoning, where multiple pieces of information have to be put together. The **self-ask with search** pattern decomposes a question into constituents and calls a search engine method in order to retrieve the necessary information in order to answer questions. An example for this powerful mechanism is discussed on LangChain's github by user nkov. The question is how lived longer, Muhammad Ali or Alan Turing, and the conversation develops thus:

```
Question: Who lived longer, Muhammad Ali or Alan Turing?
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
```

```
So the final answer is: Muhammad Ali
```

In each step, the **LLM** decides if follow-up searches are needed and this information is fed back to the **LLM**.Recently, OpenAI models (gpt-3.5-turbo-0613, gpt-4-0613) have been fine-tuned to detect when **function calls** should be executed and which input should be fed into the functions. For this to work, functions can also be described in API calls to these language models. This is also implemented in **LangChain**.There are a few strategies that are not (yet) implemented as execution mechanism in **LangChain**:

* **Recursively Criticizes and Improves** its output (**RCI**) methods ("Language Models can Solve Computer Tasks"; Kim and others, June 2023) use **LLM** as a planner to construct an agent, where the former uses an **LLM** to generate thoughts before executing the action, whereas the latter prompts an LLM to think up lessons learned for improving subsequent episodes.
* The **Tree of Thought** (**ToT**) algorithm (published as "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" in May 2023 by researchers at Princeton and Google DeepMind) advances model reasoning by traversing a search tree. Basic strategies can be depth-first or breadth-first tree traversal, however many others can and have been tested such as Best First, Monte Carlo, and A\*. These strategies have been found to significantly improve the success rate at problem solving.

These decisions can be planned out ahead or can be taken at each step. This process of creating a sequence of actions that an agent can take to achieve a goal is called the **action plan generation**. There are two different types of agents by action plan generation, which can be chosen based on the required dynamism of the task:

* **Action agents** decide at each iteration on the next action based on the outputs of all previous actions.
* **Plan-and-execute agents** decide on the full plan of actions at the start. They then execute all these actions without updating the sequence. This implementation in **LangChain** was inspired by **BabyAGI**.

Generally, action agents are more flexible, while plan-and-execute agents are better at maintaining long-term objectives. If we want to be as flexible as possible we can specify a Zero-shot **ReAct** mechanism for our agent to make decisions at every turn.Let's have a look at chains now!

#### What's a chain?

The core idea in **LangChain** is the compositionality of **LLMs** and other components to work together. For examples, users and developers can put together multiple **LLM** calls and other components in a sequence to create complex applications like chatbot-like social interactions, data extraction, and data analysis.

> In most generic terms, a **chain** is as a sequence of calls to components, which can include other chains.

For example, prompt chaining is a technique that can be used to improve the performance of LangChain applications. Prompt chaining involves chaining together multiple prompts to autocomplete a more complex response.Simply put, both chains and agents are wrappers around components. Both can also extend the functionality of LLMs by enabling them to interact with external systems and gather up-to-date information. This modularization of the applications into building blocks like chains and agents can make it easier to debug and maintain them.The most innocuous example for a chain is probably the `PromptTemplate`, which passes a formatted response to a language model. More interesting examples for chains include `LLMMath` for math-related queries and `SQLDatabaseChain` for querying databases. These are called **utility chains**, because they combine language models with specific tools. A few chains can make autonomous decision. Similar to agents, router chains can make decisions on which tool from a selection to use based on their descriptions. A `RouterChain` can dynamically select which retrieval system such as prompts or indexes to use.**LangChain** implements chains to make sure the content of the output is not toxic or otherwise violates OpenAI's moderation rules (`OpenAIModerationChain`) or that it conforms to ethical, legal, or custom principles (`ConstitutionalChain`).The LLMCheckerChain can prevent hallucinations and reduce inaccurate responses by verifying assumptions underlying provided statements and questions. In a paper by researchers at Carnegie Mellon, Allen Institute, University of Washington, NVIDIA, UC San Diego, and Google Research in May 2023 ("SELF-REFINE: Iterative Refinement with Self-Feedback) this strategy has been found to improve task performance by about 20% absolute on average across a benchmark including dialogue responses, math reasoning, and code reasoning. Let's have a look at the memory strategies!

#### What is memory?

**LLMs** and tools are stateless in the sense that they don't retain any information about previous responses and conversations. Memory is a key concept in LangChain and can be used to improve the performance of LangChain applications by storing the results of previous calls to the language model, the user, the state of the environment that the agent is operating in, and the agent's goals. This can help to reduce the number of times that the language model needs to be called and can help to ensure that the agent can continue to operate even if the environment changes.

> **Memory** is a data structure that is used to store and reuse information over time.

Memory helps provide context to the application and can make the LLM outputs more coherent and contextually relevant. For example, we can store all the conversation (`ConversationBufferMemory`) or use a buffer to retain the last messages in a conversation using the `ConversationBufferWindowMemory`. The recorded messages are included in the model's history parameter during each call. We should note however, that this will increase the token usage (and therefore API fees) and the latency of the responses. It could also affect the token limit of the model. There is also a conversation summary memory strategy, where an LLM is used to summarize the conversation history - this might incur extra costs for the additional API calls.There are a few exciting nuances about these memory options. For example, an interesting feature is that the conversation with the LLM can be encoded as a Knowledge Graph (`ConversationKGMemory`), which can be integrated back into prompts or used to predict responses without having to go to the LLM.

> A **knowledge graph** is a representation of data that uses a graph-structured data model to integrate data typically in the shape of triplets, a subject, a predicate, and an object, for example subject=Sam, predicate=loves, object=apples. This graph stores information about entities like people, places, or events), and the connections between them.

In summary, memory in **LangChain** can be used to store a variety of information, including:

* The results of previous calls to the language model
* The state of the environment that the agent is operating in
* The goals that the agent is trying to achieve.

Now, we'll have a look at the different tools at our disposal.

#### What kind of tools are there?

Tools are components in **LangChain** that can be combined with models to extend their capability. **LangChain** offers tools like document loaders, indexes, and vector stores, which facilitate the retrieval and storage of data for augmenting data retrieval in **LLMs**. There are many tools available, and here are just a few examples of you can do with tools:

* **Machine Translator**: A language model can use a machine translator to better comprehend and process text in multiple languages. This tool enables non-translation-dedicated language models to understand and answer questions in different languages.
* **Calculator**: Language models can utilize a simple calculator tool to solve math word problems. The calculator supports basic arithmetic operations, allowing the model to accurately solve mathematical queries in datasets specifically designed for math problem-solving.
* **Map**: By connecting with Bing Map API or similar services, language models can retrieve location information, assist with route planning, provide driving distance calculations, and offer details about nearby points of interest.
* **Weather**: Weather APIs provide language models with real-time weather information for cities worldwide. Models can answer queries about current weather conditions or forecast the weather for specific locations within varying time frames.
* **Stock**: Connecting with stock market APIs like Alpha Vantage allows language models to query specific stock market information such as opening and closing prices, highest and lowest prices, and more.
* **Slides**: Language models equipped with slide-making tools can create slides using high-level semantics provided by APIs such as python-pptx library or image retrieval from the internet based on given topics. These tools facilitate tasks related to slide creation required in various professional fields.
* **Table Processing**: APIs built with pandas DataFrame enable language models to perform data analysis and visualization tasks on tables. By connecting to these tools, models can provide users with a more streamlined and natural experience for handling tabular data.
* **Knowledge Graphs**: Language models can query knowledge graphs using APIs that mimic human querying processes, such as finding candidate entities or relations, sending SPARQL queries, and retrieving results. These tools assist in answering questions based on factual knowledge stored in knowledge graphs.
* **Search Engine**: By utilizing search engine APIs like Bing Search, language models can interact with search engines to extract information and provide answers to real-time queries. These tools enhance the model's ability to gather information from the web and deliver accurate responses.
* **Wikipedia**: Language models equipped with Wikipedia search tools can search for specific entities on Wikipedia pages, look up keywords within a page, or disambiguate entities with similar names. These tools facilitate question-answering tasks using content retrieved from Wikipedia.
* **Online Shopping**: Connecting language models with online shopping tools allows them to perform actions like searching for items, loading detailed information about products, selecting item features, going through shopping pages, and making purchase decisions based on specific user instructions.

Additional tools include AI Painting, which allows language models to generate images using AI image generation models; 3D Model Construction, enabling language models to create three-dimensional (3D) models using a sophisticated 3D rendering engine; Chemical Properties, assisting in resolving scientific inquiries about chemical properties using APIs like PubChem; Database tools facilitating natural language access to database data for executing SQL queries and retrieving results.These various tools provide language models with additional functionalities and capabilities to perform tasks beyond text processing. By connecting with these tools via APIs, language models can enhance their abilities in areas such as translation, math problem-solving, location-based queries, weather forecasting, stock market analysis, slides creation, table processing and analysis, image generation, text-to-speech conversion and many more specialized tasks. All these tools can give us advanced AI functionality, and there's virtually no limit to tools. We can easily build custom tools to extend the capability of LLMs as we'll see in the next chapter 3. The use of different tools expands the scope of applications for language models and enables them to handle various real-world tasks more efficiently and effectively.Let's summarize!

### Summary

In today's world, understanding and processing language accurately is crucial in the development of smart applications and for creating personalized and effective user experiences. Therefore, **large language models** (**LLMs**) are ideally suited to lend that capability to applications. However, as we've discussed in this chapter, standalone **LLMs** have their limitations. If we supplement **LLMs** with tools, we can overcome some of these limitations and greatly augment their performance creating **LLM** applications. This is where LangChain comes in, which is a framework aimed at AI developers to set up applications of agents - these are composed of computing entities such as LLMs and other tools that can perform certain tasks autonomously. We've discussed its important concepts, first of all concepts such as agents and chains.In conclusion, LangChain is a valuable open-source framework for simplifying the development of applications using **large language models** (**LLMs**) from providers and platforms such as OpenAI and Hugging Face among many others. This framework offers immense value in unlocking the power of generative AI. In the following chapters, we'll build on these core principals of **LangChain** by building **LLM** applications. By leveraging **LangChain's** capabilities, developers can unlock the full potential of **LLMs**. In the _Chapter 3_, _Getting Started with LangChain_, we'll implement our first apps with **Langchain**!Let's see if you remember some of the key takeaways from this chapter!

### Questions

Please have a look to see if you can come up with the answers to these questions. I'd recommend you go back to the corresponding sections of this chapter, if you are unsure about any of them:

1. What are limitations of LLMs?
2. What are LLM-applications?
3. What is LangChain and why should you use it?
4. What are LangChain's key features?
5. What is an agent in LangChain?
6. What is action plan generation?
7. What is a chain?
8. Why do you need memory for LangChain applications?
9. What kind of tools are available?
10. How does LangChain work?
