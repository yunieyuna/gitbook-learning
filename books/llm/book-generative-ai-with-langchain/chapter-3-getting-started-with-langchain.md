# Chapter 3: Getting started with LangChain

## 3 Getting Started with LangChain

### Join our book community on Discord

[https://packt.link/EarlyAccessCommunity](https://packt.link/EarlyAccessCommunity)

![Qr code Description automatically generated](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file19.png)

In this chapter, we'll first set up **LangChain** and the libraries needed for this book giving instructions for common dependency management tools such as **Docker**, **Conda**, **Pip**, and **Poetry**. Then we'll go through model integrations that we can use such as **OpenAI's** **Chatgpt**, models on Huggingface and Jina AI, and others. We'll introduce, set up, and work with a few providers in turn. We'll get an API key tokens and then do a short practical example. This will give us a bit more context at using **LangChain**, and introduce tips and tricks for using it effectively. As the final part, we'll develop a **LangChain** application, a practical example that illustrate a way that **LangChain** can be applied in a real-world business use case in customer service.The main sections are:

* How to Set Up **LangChain**?
* Model Integrations
* Customer Service Helper

We'll start off the chapter by setting up **LangChain** on your computer.

### How to Set Up LangChain?

In this book, we are talking about LangChain. We can install LangChain by simply typing `pip install langchain` from a terminal however, in this book, we'll also be using a variety of other tools and integrations in a few different use cases. In order to make sure, all the examples and code snippets work as intended and they don't just work on my machine, but for anyone installing this, I am providing different ways to set up an environment.There are various approaches to setting up a Python environment. Here, we describe four popular methods for installing related dependencies: Docker, Conda, Pip, and Poetry. In case you encounter issues during the installation process, consult the respective documentation or raise an issue on the Github repository of this book. The different installations have been tested at the time of the release of this book, however, things can change, and we will update the Github readme online to include workarounds for possible problems that could arise.Please find a `Dockerfile` for docker, a `requirements.txt` for pip a `pyproject.toml` for poetry and a `langchain_ai.yml` file for **Conda** in the book's repository at [https://github.com/benman1/generative\_ai\_with\_langchain](https://github.com/benman1/generative\_ai\_with\_langchain)Let's set up our environment starting with Python.

#### Python installation

Before setting up a Python environment and installing related dependencies, you should usually have Python itself installed. I assume, most people who have bought this book will have Python installed, however, just in cases, let's go through it. You may download the latest version from python.org for your operating system or use your platform's package manager. Let's see this with Homebrew for MacOS and apt-get for Ubuntu.On MacOS, with Homebrew, we can do:

```
brew install python
```

For Ubuntu, with apt-get we can do:

```
sudo apt-get updatesudo apt-get install python3.10
```

> **Tip**: If you are new to programming or Python, it is advised to follow some beginner-level tutorials before proceeding with LangChain and the applications in this book.

An important tool for interactively trying out data processing and models is the Jupyter notebook and the lab. Let's have a look at this now.

#### Jupyter Notebook and JupyterLab

Jupyter Notebook and JupyterLab are open-source web-based interactive environments for creating, sharing, and collaborating on computational documents. They enable users to write code, display visualizations, and include explanatory text in a single document called a notebook. The primary difference between the two lies in their interface and functionality.

> **Jupyter Notebook** aims to support various programming languages like Julia, Python, and R - in fact, the project name is a reference to these three languages. Jupyter Notebook offers a simple user interface that allows users to create, edit, and run notebooks with a linear layout. It also supports extensions for additional features and customization.
>
> > **JupyterLab**, on the other hand, is an enhanced version of Jupyter Notebook. Introduced in 2018, JupyterLab offers a more powerful and flexible environment for working with notebooks and other file types. It provides a modular, extensible, and customizable interface where users can arrange multiple windows (for example, notebooks, text editors, terminals) side-by-side, facilitating more efficient workflows.

You can start up a notebook server on your computer from the terminal like this:

```
jupyter notebook
```

You should see your browser opening a new tab with the Jupyter notebook like this:

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file20.png" alt="Figure 3.1: Jupyter Notebook with a LangChain Agent." height="446" width="1161"><figcaption><p>Figure 3.1: Jupyter Notebook with a LangChain Agent.</p></figcaption></figure>

Alternatively, we can also use JupyterLab, the next-generation notebook server that brings significant improvements in usability. You can start up a JupyterLab notebook server from the terminal like this:

```
jupyter lab
```

We should see something like this:

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file21.png" alt="Figure 3.2: Jupyter Lab with a LangChain Agent." height="387" width="1014"><figcaption><p>Figure 3.2: Jupyter Lab with a LangChain Agent.</p></figcaption></figure>

Either one of these two, the `Jupyter notebook` or `JupyterLab`, will give you an **integrated development environment** (**IDE**) to work on some of the code that we'll be introducing in this book. After installing Python and the notebook or lab, let's quickly explore the differences between dependency management tools (**Docker**, **Conda**, **Pip**, and **Poetry**) and use them to fully set up our environment for our projects with LangChain!

#### Environment management

Before we explore various methods to set up a Python environment for working with generative models in **LangChain**, it's essential to understand the differences between primary dependency management tools: **Docker**, **Conda**, **Pip**, and **Poetry**. All four are tools widely used in the realm of software development and deployment.

> **Docker** is an open-source platform that provides OS-level virtualization through containerization. It automates the deployment of applications inside lightweight, portable containers, which run consistently on any system with Docker installed.
>
> > **Conda** is a cross-platform package manager and excels at installing and managing packages from multiple channels, not limited to Python. Geared predominantly toward data science and machine learning projects, it can robustly handle intricate dependency trees, catering to complex projects with numerous dependencies.
>
> > **Pip** is the most commonly used package manager for Python, allowing users to install and manage third-party libraries easily. However, Pip has limitations when handling complex dependencies, increasing the risk of dependency conflicts arising during package installation.
>
> > **Poetry** is a newer package manager that combines the best features of both Pip and Conda. Boasting a modern and intuitive interface, robust dependency resolution system, and support for virtual environments creation, Poetry offers additional functionalities such as dependency isolation, lock files, and version control.

Poetry and Conda both streamline virtual environment management, whereas working with Pip typically involves utilizing a separate tool like virtualenv. Conda is the installation method recommended here. We'll provide a requirements file for pip as well and instructions for poetry, however, some tweaking might be required in a few cases.We'll go through installation with these different tools in turn. For all instructions, please make sure you have the book's repository downloaded (using the Github user interface) or cloned on your computer, and you've changed into the project's root directory.Here's how you can find the download option on Github:

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file22.png" alt="Figure 3.3: Download options in the Github User Interface (UI)." height="379" width="411"><figcaption><p>Figure 3.3: Download options in the Github User Interface (UI).</p></figcaption></figure>

If you are new to git, you can press **Download ZIP**, and then unzip the archive using your favorite tool.Alternatively, to clone the repository using git and change to the project directory, you can type the following commands:

```
git clone https://github.com/benman1/generative_ai_with_langchain.git
cd generative_ai_with_langchain
```

Now that we have the repository on our machine, let's start with Docker!

**Docker**

Docker is a platform that enables developers to automate deployment, packaging, and management of applications. Docker uses containerization technology, which helps standardize and isolate environments. The advantage of using a container is that it protects your local environment from any - potentially unsafe - code that you run within the container. The downside is that the image might require time to build and might require around 10 Gigabytes in storage capacity.Similar to the other tools for environment management, Docker is useful because you can create a reproducible environment for your project. You can use Docker to create an environment with all the libraries and tools you need for your project, and share that environment with others.To start with Docker, follow these steps:

1. Install Docker on your machine. You can go to the Docker website.in your web browser and follow the installation instructions here: [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)
2. In the terminal, run the following command to build the Docker image (please note: you need to be in the project root for this to work).

```
docker build -t langchain_ai
```

This will pull, the continuumio/miniconda3 image from Docker Hub, and build the image.

1. Start the Docker container interactively using the image created:

```
docker run -it langchain_ai
```

This should start our notebook within the container. We should be able to navigate to the `Jupyter Notebook` from your browser. We can find it at this address: `http://localhost:8080/`Let's look at conda next.

**Conda**

`Conda` allows users to manage multiple environments for different projects. It works with Python, R, and other languages, and helps with the installation of system libraries as well by maintaining lists of libraries associated with Python libraries.The best way to get started with conda is to install anaconda or miniconda by following the instructions from this link: [https://docs.continuum.io/anaconda/install/](https://docs.continuum.io/anaconda/install/)While the `conda` environment takes up less disk space than Docker, starting from anaconda, the full environment should still take up about 2.5 Gigabytes. The miniconda setup might save you a bit of disk space.There's also a graphical interface to `conda`, Anaconda Navigator, which can be installed on macOS and Windows, and which can install any dependencies as well as the `conda` tool from the terminal.Let's continue with the `conda` tool and install the dependencies of this book.To create a new environment, execute the following command:

```
conda env create --file langchain_ai.yml
```

`Conda` lets us create environments with lots of different libraries, but also different versions of Python. We are using Python 3.10 throughout this book. Activate the environment by running:

```
conda activate langchain_ai
```

This is all, we are done. We can see this should be painless and straightforward. You can now spin up a `jupyter notebook` or `jupyter lab` within the environment, for example:

```
jupyter notebook
```

Let's have a look at pip, an alternative to `conda`.

**Pip**

`Pip` is the default package manager for Python. It allows you to easily install and manage third-party libraries. We can install individual libraries, but also maintain a full list of Python libraries.If it's not already included in your Python distribution, install pip following instructions on [https://pip.pypa.io/](https://pip.pypa.io/)To install a library with pip, use the following command. For example, to install the NumPy library, you would use the following command:

```
pip install numpy
```

You can also use `pip` to install a specific version of a library. For example, to install version 1.0 of the NumPy library, you would use the following command:

```
pip install numpy==1.0
```

In order to set up a full environment, we can start with a list of requirements - by convention, this list is in a file called `requirements.txt`. I've included this file in the project's root directory, which lists all essential libraries.You can install all the libraries using this command:

```
pip install -r requirements.txt
```

Please note, however, as mentioned, that Pip doesn't take care of the environments. Virtualenv is a tool that can help to maintain environments, for example different versions of libraries. Let's see this quickly:

```
# create a new environment myenv:
virtualenv myenv
# activate the myenv environment:
source myenv/bin/activate
# install dependencies or run python, for example:
python
# leave the environment again:
deactivate
Please note that in Windows, the activation command is slightly different – you'd run a shell script:
# activate the myenv environment:
myenv\Scripts\activate.bat
```

Let's do Poetry next.

**Poetry**

Poetry is a dependency management tool for Python that streamlines library installation and version control. Installation and usage is straightforward as we'll see. Here's a quick run-through of poetry:

1. Install poetry following instructions on [https://python-poetry.org/](https://python-poetry.org/)
2. Run `poetry install` in the terminal (from the project root as mentioned before)

The command will automatically create a new environment (if you haven't created one already) and install all dependencies. This concludes the setup for Poetry. We'll get to model providers now.

### Model Integrations

Before properly starting with generative AI, we need to set up access to models such as **large language models** (**LLMs**) or text to image models so we can integrate them into our applications. As discussed in _Chapter 1_, _What are Generative Models_, there are various **LLMs** by tech giants, like **GPT-4** by **OpenAI**, **BERT** and **PaLM-2** by **Google**, **LLaMA** by **Meta AI**, and many more.With the help of **LangChain**, we can interact with all of these, for example through **Application Programming Interface** (**APIs**), or we can call open-source models that we have downloaded on our computer. Several of these integrations support text generation and embeddings. We'll focus on text generation in this chapter, and discuss embeddings, vector databases, and neural search in _Chapter 5_, _Building a Chatbot like ChatGPT_.There are many providers for model hosting. For **LLMs**, currently, **OpenAI**, **Hugging Face**, **Cohere**, **Anthropic**, **Azure**, **Google Cloud Platform Vertex AI** (**PaLM-2**), and **Jina AI** are among the many providers supported in **LangChain**, however this list is growing all the time. You can all the supported integrations for **LLMs** at [https://integrations.langchain.com/llms](https://integrations.langchain.com/llms)As for image models, the big developers include **OpenAI** (**DALL-E**), **Midjourney**, Inc. (Midjourney), and Stability AI (**Stable Diffusion**). **LangChain** currently doesn't have out-of-the-box handling of models that are not for text, however, its docs describe how to work with Replicate, which also provides an interface to Stable Diffusion models.For each of these providers, to make calls against their Application Programming Interface (API), you'll first need to create an account and obtain an API key. This is free for all of them. With some of them you don't even have to give them your credit card details.In order to set an API key in an environment, in Python we can do:

```
import os
os.environ["OPENAI_API_KEY"] = "<your token>"
```

Here `OPENAI_API_KEY` is the environment key appropriate for OpenAI. Setting the keys in your environment has the advantage that we don't include them in our code.You can also expose these variables from your terminal like this:

```
export OPENAI_API_KEY=<your token>
```

Let's go through a few prominent model providers in turn. We'll give an example usage for each of them.Let's' start with a Fake LLM that's used for testing so we can show the basic idea!

#### Fake LLM

The Fake LM is for testing. The LangChain documentation has an example for the tool use with LLMs. You can execute this example in either Python directly or in a notebook.

```
from langchain.llms.fake import FakeListLLM
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
tools = load_tools(["python_repl"])
responses = ["Action: Python_REPL\nAction Input: print(2 + 2)", "Final Answer: 4"]
llm = FakeListLLM(responses=responses)
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
agent.run("whats 2 + 2")
```

We connect a tool, a Python **Read-Eval-Print Loop** (**REPL**) that will be called depending on the output of the **LLM**. The Fake List **LLM** will give two responses, `responses`, that won't change based on the input. We set up an agent that makes decisions based on the ReAct strategy that we explained in chapter 2, Introduction to LangChain (`ZERO_SHOT_REACT_DESCRIPTION`). We run the agent with a text, the question "what's 2 + 2".We can observe how the Fake LLM output, leads to a call to the Python Interpreter, which returns 4. Please note that the action has to match the `name` attribute of the tool, the `PythonREPLTool`, which is starts like this:

```
class PythonREPLTool(BaseTool):
    """A tool for running python code in a REPL."""
    name = "Python_REPL"
    description = (
        "A Python shell. Use this to execute python commands. "
        "Input should be a valid python command. "
        "If you want to see the output of a value, you should print it out "
        "with `print(...)`."
    )
```

The names and descriptions of the tools are passed to the **LLM**, which then decides based on the provided information.The output of the Python interpreter is passed to the Fake **LLM**, which ignores the observation and returns 4. Obviously, if we change the second response to "`Final Answer: 5`", the output of the agent wouldn't correspond to the question.In the next sections, we'll make this more meaningful by using an actual **LLM** rather than a fake one. One of the first providers that anyone will think of at the moment is OpenAI.

#### OpenAI

As explained in _Chapter 1_, _What are Generative Models?_, OpenAI is an American AI research laboratory that is the current market leader in generative AI models, especially LLMs. They offer a spectrum of models with different levels of power suitable for different tasks. We'll see in this chapter how to interact with OpenAI models with **LangChain's** and the OpenAI Python client library. OpenAI also offers an Embedding class for text embedding models.We will mostly use OpenAI for our applications. There are several models to choose from - each model has its own pros, token usage counts, and use cases. The main LLM models are GPT-3.5 and GPT-4 with different token length. You can see the pricing of different models at [https://openai.com/pricing](https://openai.com/pricing)We need to obtain an OpenAI API key first. In order to create an API key, follow these steps:

1. You need to create a login at [https://platform.openai.com/](https://platform.openai.com/)
2. Set up your billing information.
3. You can see the **API keys** under _Personal -> View API Keys_.
4. Click **Create new secret key** and give it a **Name**.

Here's how this should look like on the OpenAI platform:

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file23.png" alt="Figure 3.4: OpenAI API platform - Create new secret key." height="353" width="784"><figcaption><p>Figure 3.4: OpenAI API platform - Create new secret key.</p></figcaption></figure>

After pressing "**Create secret key**", you should see the message "API key generated." You need to copy the key to your clipboard and keep it. We can set the key as an environment variable (`OPENAI_API_KEY`) or pass it as a parameter every time you construct a class for OpenAI calls.We can use the `OpenAI` language model class to set up an **LLM** to interact with. Let's create an agent that calculates using this model - I am omitting the imports from the previous example:

```
from langchain.llms import OpenAI
llm = OpenAI(temperature=0., model="text-davinci-003")
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
agent.run("whats 4 + 4")
```

We should be seeing this output:

```
> Entering new  chain...
 I need to add two numbers
Action: Python_REPL
Action Input: print(4 + 4)
Observation: 8
Thought: I now know the final answer
Final Answer: 4 + 4 = 8
> Finished chain.
'4 + 4 = 8'
```

This looks quite promising, I think. Let's move on to the next provider and more examples!

#### Hugging Face

Hugging Face is a very prominent player in the NLP space, and has considerable traction in open-source and hosting solutions. The company is an American company that develops tools for building machine learning applications. Its employees develop and maintain the Transformers Python library, which is used for natural language processing tasks, includes implementations of state-of-the-art and popular models like BERT and GPT-2, and is compatible with **PyTorch**, **TensorFlow**, and **JAX**.Hugging Face also provides the Hugging Face Hub, a platform for hosting Git-based code repositories, machine learning models, datasets, and web applications, which provides over 120k models, 20k datasets, and 50k demo apps (Spaces) for machine learning. It is an online platform where people can collaborate and build ML together.These tools allow users to load and use models, embeddings, and datasets from Hugging Face. The `HuggingFaceHub` integration, for example, provides access to different models for tasks like text generation and text classification. The `HuggingFaceEmbeddings` integration allows users to work with sentence-transformers models.They offer various other libraries within their ecosystem, including Datasets for dataset processing, _Evaluate_ for model evaluation, _Simulate_ for simulation, and _Gradio_ for machine learning demos.In addition to their products, Hugging Face has been involved in initiatives such as the BigScience Research Workshop, where they released an open large language model called BLOOM with 176 billion parameters. They have received significant funding, including a $40 million Series B round and a recent Series C funding round led by Coatue and Sequoia at a $2 billion valuation. Hugging Face has also formed partnerships with companies like Graphcore and Amazon Web Services to optimize their offerings and make them available to a broader customer base.In order to use Hugging Face as a provider for your models, you can create an account and API keys at [https://huggingface.co/settings/profile](https://huggingface.co/settings/profile)You can make the token available in your environment as `HUGGINGFACEHUB_API_TOKEN`.Let's see an example, where we use an open-source model developed by Google, the Flan-T5-XXL model:

```
from langchain.llms import HuggingFaceHub
llm = HuggingFaceHub(
    model_kwargs={"temperature": 0.5, "max_length": 64},
    repo_id="google/flan-t5-xxl"
)
prompt = "In which country is Tokyo?"
completion = llm(prompt)
print(completion)
```

We get the response "`japan`."The **LLM** takes a text input, a question in this case, and returns a completion. The model has a lot of knowledge and can come up with answers to knowledge questions. We can also get simple recommendations:

#### Azure

Azure, the cloud computing platform run by Microsoft, integrates with OpenAI to provide powerful language models like GPT-3, Codex, and Embeddings. It offers access, management, and development of applications and services through their global data centers for use cases such as writing assistance, summarization, code generation, and semantic search. It provides capabilities such as **software as a service** (**SaaS**), **platform as a service** (**PaaS**), and **infrastructure as a service** (**IaaS**).Authenticating either through Github or Microsoft credentials, we can create an account on Azure under [https://azure.microsoft.com/](https://azure.microsoft.com/)You can then create new API keys under _Cognitive Services -> Azure OpenAI_. There are a few more steps involved, and personally, I found this process annoying and frustrating, and I gave up. After set up, the models should be accessible through the `AzureOpenAI()` llm class in **LangChain**.

#### Google Cloud

There are many models and functions available through **Google Cloud Platform** (**GCP**) and Vertex its machine learning platform. Google Cloud provide access to **LLMs** like **LaMDA**, **T5**, and **PaLM**. Google has also updated the Google Cloud **Natural Language** (**NL**) API with a new LLM-based model for Content Classification. This updated version offers an expansive pre-trained classification taxonomy to help with ad targeting, and content-based filtering. The **NL** API's improved v2 classification model is enhanced with over 1,000 labels and support for 11 languages with improved accuracy. For models with GCP, you need to have gcloud **command line interface** (**CLI**) installed. You can find the instructions here: [https://cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install)You can then authenticate and print a key token with this command from the terminal:

```
gcloud auth application-default login
```

You also need to enable Vertex for your project. If you haven't enabled it, you should get a helpful error message pointing you to the right website, where you have to click on "Enable".Let's run a model!

```
from langchain.llms import VertexAI
from langchain import PromptTemplate, LLMChain
template = """Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm = VertexAI()
llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"
llm_chain.run(question)
```

We should see this response:

```
[1m> Entering new chain...[0m
Prompt after formatting:
[[Question: What NFL team won the Super Bowl in the year Justin Beiber was born?
Answer: Let's think step by step.[0m
[1m> Finished chain.[0m
Justin Beiber was born on March 1, 1994. The Super Bowl in 1994 was won by the San Francisco 49ers.
```

I've set verbose to True in order to see the reasoning process of the model. It's quite impressive it comes up with the right response even given a misspelling of the name. The step by step prompt instruction is key to the correct answer.There are various models available through Vertex such as these:

| **Model**      | **Description**                                                            | **Properties**                                                                                   |
| -------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| text-bison     | Fine-tuned to follow natural language instructions                         | <p>Max input token: 8,192</p><p>Max output tokens: 1,024</p><p>Training data: Up to Feb 2023</p> |
| chat-bison     | Fine-tuned for multi-turn conversation                                     | Max input token: 4,096 Max output tokens: 1,024 Training data: Up to Feb 2023 Max turns : 2,500  |
| code-bison     | Fine-tuned to generate code based on a natural language description        | Max input token: 4,096 Max output tokens: 2,048                                                  |
| codechat-bison | Fine-tuned for chatbot conversations that help with code-related questions | Max input token: 4,096 Max output tokens: 2,048                                                  |
| code-gecko     | Fine-tuned to suggest code completion                                      | Max input tokens: 2,048 Max output tokens: 64                                                    |

Table 3.1: Models available in Vertex Generative AI. You can check out the documentation at [https://cloud.google.com/vertex-ai/docs/generative-ai](https://cloud.google.com/vertex-ai/docs/generative-ai)We can also generate code. Let's see if **Code-Bison** model can solve FizzBuzz, a common interview question for entry and mid-level software developer positions:

```
question = """
Given an integer n, return a string array answer (1-indexed) where:
answer[i] == "FizzBuzz" if i is divisible by 3 and 5.
answer[i] == "Fizz" if i is divisible by 3.
answer[i] == "Buzz" if i is divisible by 5.
answer[i] == i (as a string) if none of the above conditions are true.
"""
llm = VertexAI(model_name="code-bison")
llm_chain = LLMChain(prompt=prompt, llm=llm)
print(llm_chain.run(question))
```

We are getting this response:

````
```python
answer = []
for i in range(1, n + 1):
    if i % 3 == 0 and i % 5 == 0:
        answer.append("FizzBuzz")
    elif i % 3 == 0:
        answer.append("Fizz")
    elif i % 5 == 0:
        answer.append("Buzz")
    else:
        answer.append(str(i))
return answer
```
````

Would you hire code-bison into your team?

#### Anthropic

Anthropic is an AI startup and public-benefit corporation based in the United States. It was founded in 2021 by former members of OpenAI, including siblings Daniela Amodei and Dario Amodei. The company specializes in developing general AI systems and language models with a focus on responsible AI usage. As of July 2023, Anthropic has raised $1.5 billion in funding. They have also worked on projects such as Claude, an AI chatbot similar to OpenAI's ChatGPT, and have conducted research on the interpretability of machine learning systems, specifically the transformer architecture.Unfortunately, Claude is not available to the general public (yet). You need to apply for access to use Claude and set the `ANTHROPIC_API_KEY` environment variable.

#### Jina AI

Jina AI, founded in February 2020 by Han Xiao and Xuanbin He, is a German AI company based in Berlin that specializes in providing cloud-native neural search solutions with models for text, image, audio, and video. Their open-source neural search ecosystem enables businesses and developers to easily build scalable and highly available neural search solutions, allowing for efficient information retrieval. Recently, **Jina AI** launched _Finetuner_, a tool that enables fine-tuning of any deep neural network to specific use cases and requirements.The company has raised a total of $37.5 million in funding through three rounds, with their most recent funding coming from a Series A round in November 2021. Notable investors in **Jina AI** include **GGV Capital** and **Canaan Partners**.You can set up a login under [https://cloud.jina.ai/](https://cloud.jina.ai/)On the platform, we can set up APIs for different use cases such as image caption, text embedding, image embedding, visual question answering, visual reasoning, image upscale, or Chinese text embedding.Here, we are setting up a visual question answering API with the recommended model:

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file24.png" alt="Figure 3.5: Visual Question Answering API in Jina AI." height="440" width="734"><figcaption><p>Figure 3.5: Visual Question Answering API in Jina AI.</p></figcaption></figure>

We get examples for client calls in Python and cURL, and a demo, where we can ask a question. This is cool, unfortunately, these APIs are not available yet through **LangChain**. We can implement such calls ourselves by subclassing the `LLM` class in **LangChain** as a custom **LLM** interface.Let's set up another chatbot, this time powered by Jina AI. We can generate the API token, which we can set as `JINA_AUTH_TOKEN`, at [https://chat.jina.ai/api](https://chat.jina.ai/api)Let's translate from English to French here:

```
from langchain.chat_models import JinaChat
from langchain.schema import HumanMessage
chat = JinaChat(temperature=0.)
messages = [
    HumanMessage(
        content="Translate this sentence from English to French: I love generative AI!"
    )
]
chat(messages)
```

```
We should be seeing 
```

```
AIMessage(content="J'adore l'IA générative !", additional_kwargs={}, example=False).
```

We can set different temperatures, where a low temperature makes the responses more predictable. In this case it makes very little difference. We are starting the conversation with a system message clarifying the purpose of the chatbot.Let's ask for some food recommendations:

```
chat = JinaChat(temperature=0.)
chat(
    [
        SystemMessage(
            content="You help a user find a nutritious and tasty food to eat in one word."
        ),
        HumanMessage(
            content="I like pasta with cheese, but I need to eat more vegetables, what should I eat?"
        )
    ]
)
```

I am seeing this response in Jupyter:

```
AIMessage(content='A tasty and nutritious option could be a vegetable pasta dish. Depending on your taste, you can choose a sauce that complements the vegetables. Try adding broccoli, spinach, bell peppers, and zucchini to your pasta with some grated parmesan cheese on top. This way, you get to enjoy your pasta with cheese while incorporating some veggies into your meal.', additional_kwargs={}, example=False)
```

It ignored the one-word instruction, but I actually liked reading the ideas. I think I could try this for my son. With other chatbots, I am getting the suggestion of Ratatouille.It's important to understand the difference in LangChain between LLMs and Chat Models. LLMs are text completion models that take a string prompt as input and output a string completion. Chat models are similar to LLMs but are specifically designed for conversations. They take a list of chat messages as input, labeled with the speaker, and return a chat message as output. Both LLMs and Chat Models implement the Base Language Model interface, which includes methods such as `predict()` and `predict_messages()`. This shared interface allows for interchangeability between different types of models in applications as well as between Chat and LLM models.

#### Replicate

Established in 2019, Replicate Inc. is a San Francisco-based startup that presents a streamlined process to AI developers, where they can implement and publish AI models with minimal code input through the utilization of cloud technology. The platform works with private as well as public models and enables model inference and fine-tuning. The firm, deriving its most recent funding from a Series A funding round of which the invested total was $12.5 million, was spearheaded by Andreessen Horowitz, and involved the participation of Y Combinator, Sequoia, and various independent investors. Ben Firshman, who drove open-source product efforts at Docker, and Andreas Jansson, a former machine learning engineer at Spotify, co-founded Replicate Inc. with the mutual aspiration to eliminate the technical barriers that were hindering the mass acceptance of AI. Consequently, they created Cog, an open-source tool that packs machine learning models into a standard production-ready container that can run on any current operating system and automatically generates an API. These containers can also be deployed on clusters of GPUs through the replicate platform. As a result, developers can concentrate on other essential tasks, thereby enhancing their productivity.You can authenticate with your Github credentials on [https://replicate.com/](https://replicate.com/)If you then click on your user icon on the top left, you'll find the API tokens - just copy the API key and make it available in your environment as `REPLICATE_API_TOKEN`. In order to run bigger jobs, you need to set up your credit card (under billing).You can find a lot of models available at [https://replicate.com/explore](https://replicate.com/explore)Here's a simple example for creating an image:

```
from langchain.llms import Replicate
text2image = Replicate(
    model="stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
    input={"image_dimensions": "512x512"},
)
image_url = text2image("a book cover for a book about creating generative ai applications in Python")
```

I got this image:

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file25.png" alt="Figure 3.7: A Book Cover for a book about generative AI with Python - Stable Diffusion." height="512" width="512"><figcaption><p>Figure 3.7: A Book Cover for a book about generative AI with Python - Stable Diffusion.</p></figcaption></figure>

I think it's a nice image - is that an AI chip that creates art?Let's see quickly how to run a model locally in Huggingface transformers or Llama.cpp!

#### Local Models

We can also run local models from LangChain. Let's preface this with a note of caution: an LLM is big, which means that it'll take up a lot of space. If you have an old computer, you can try hosted services such as google colabs. These will let you run on machines with a lot of memory and different hardware including Tensor Processing Units (TPUs) or GPUs.Since both these use cases can take very long to run or crash the Jupyter notebook, I haven't included this code in the notebook or the dependencies in the setup instructions. I think it's still worth discussing it here. The advantages of running models locally are complete control over the model and not sharing any data over the internet.Let's see this first with the transformers library by Hugging Face.

**Hugging Face transformers**

I'll quickly show the general recipe of setting up and running a pipeline:

```
from transformers import pipeline
import torch
generate_text = pipeline(
    model="aisquared/dlite-v1-355m",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    framework="pt"
)
generate_text("In this chapter, we'll discuss first steps with generative AI in Python.")
```

This model is quite small (355 million parameters), but relative performant, and instruction tuned for conversations.Please note that we don't need an API token for local models!This will download everything that's needed for the model such as the tokenizer and model weights. We can then run a text completion to give us some content for this chapter.In order to plug in this pipeline into a LangChain agent or chain, we can use it the same way that we've seen before:

```
from langchain import PromptTemplate, LLMChain
template = """Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=generate_text)
question = "What is electroencephalography?"
print(llm_chain.run(question))
```

In this example, we also see the use of a `PromptTemplate` that gives specific instructions for the task.Let's do Llama.cpp next.

**Llama.cpp**

Llama.cpp is a C++ program that executes models based on architectures based on Llama, one of the first large open-source models released by Meta AI, which spawned the development of many other models in turn. Please note that you need to have an md5 checksum tool installed. This is included by default in several Linux distributions such as Ubuntu. On MacOs, you can install it with brew like this:

```
brew install md5sha1sum
```

We need to download the llama.cpp repository from Github. You can do this online choosing one of the download options on Github, or you can use a git command from the terminal like this:

```
git clone https://github.com/ggerganov/llama.cpp.git
```

Then we need to install the python requirements, which we can do with the pip package installer - let's also switch to the llama.cpp project root directory for convenience:

```
cd llama.cpp
pip install -r requirements.txt
```

You might want to create a Python environment before you install requirements - but this is up to you. Now we need to compile llama.cpp:

```
make -C . -j4 # runs make in subdir with 4 processes
```

We can parallelize the build with 4 processes. In order to get the Llama model weights, you need to sign up with the T\&Cs and wait for a registration email from Meta. There are tools such as the llama model downloader in the pyllama project, but please be advised that they might not conform with the license stipulations by Meta. You can download models from Hugging Face - these models should be compatible with llama.cpp, such as Vicuna or Alpaca. Let's assume you have downloaded the model weights for the 7B Llama model into the models/7B directory.You can download models in much bigger sizes such as 13B, 30B, 65B, however, a note of caution is in order here: these models are fairly big both in terms of memory and disk space.We have to convert the model to llama.cpp format, which is called **ggml**, using the convert script. Then we can optionally quantize the models to save memory when doing inference. Quantization refers to reducing the number of bits that are used to store weights.

```
python3 convert.py models/7B/
./quantize ./models/ggml-model-f16.bin ./models/7B/ggml-model-q4_0.bin q4_0
```

This last file is much smaller than the previous files and will take up much less space in memory as well, which means that you can run it on smaller machines.Once we have chosen a model that we want to run, we can integrate it into an agent or a chain for example as follows:

```
llm = LlamaCpp(
    model_path="./ggml-model-q4_0.bin",
    verbose=True
```

)

This concludes the introduction to model providers. Let's build an application!

### Customer Service Helper

In this section, we'll build a text classification app in LangChain for customer service agents. Given a document such as an email, we want to classify it into different categories related to intent, extract the sentiment, and provide a summary.Customer service agents are responsible for answering customer inquiries, resolving issues and addressing complaints. Their work is crucial for maintaining customer satisfaction and loyalty, which directly affects a company's reputation and financial success. Generative AI can assist customer service agents in several ways:

* Sentiment classification: this helps identify customer emotions and allows agents to personalize their response.
* Summarization: this enables agents to understand the key points of lengthy customer messages and save time.
* Intent classification: similar to summarization, this helps predict the customer's purpose and allows for faster problem-solving.
* Answer suggestions: this provides agents with suggested responses to common inquiries, ensuring that accurate and consistent messaging is provided.

These approaches combined can help customer service agents respond more accurately and in a timely manner, ultimately improving customer satisfaction.Here, we will concentrate on the first three points. We'll document lookups, which we can use for answer suggestions in _Chapter 5_, _Building a Chatbot like ChatGPT_.**LangChain** is a very flexible library with many integrations that can enable us to tackle a wide range of text problems. We have a choice between many different integrations to perform these tasks. We could ask any LLM to give us an open-domain (any category) classification or choose between multiple categories. In particular, because of their large training size, LLMs are very powerful models, especially when given few-shot prompts, for sentiment analysis that don't need any additional training. This was analyzed by Zengzhi Wang and others in their April 2023 study "Is ChatGPT a Good Sentiment Analyzer? A Preliminary Study". A prompt, for an LLM for sentiment analysis could be something like this:

```
Given this text, what is the sentiment conveyed? Is it positive, neutral, or negative?
Text: {sentence}
Sentiment:
```

LLMs can be also very effective at summarization, much better than any previous models. The downside can be that these model calls are slower than more traditional ML models and more expensive.If we want to try out more traditional or smaller models. Cohere and other providers have text classification and sentiment analysis as part of their capabilities. For example, NLP Cloud's model list includes spacy and many others: [https://docs.nlpcloud.com/#models-list](https://docs.nlpcloud.com/#models-list)Many Hugging Face models are supported for these tasks including:

* document-question-answering
* summarization
* text-classification
* text-question-answering
* translation

We can execute these models either locally, by running a `pipeline` in transformer, remotely on the Hugging Face Hub server (`HuggingFaceHub`), or as tool through the `load_huggingface_tool()` loader.Hugging Face contains thousands of models, many fine-tuned for particular domains. For example, `ProsusAI/finbert` is a BERT model that was trained on a dataset called Financial PhraseBank, and can analyze sentiment of financial text. We could also use any local model. For text classification, the models tend to be much smaller, so this would be less of a drag on resources. Finally, text classification could also be a case for embeddings, which we'll discuss in _Chapter 5_, _Building a Chatbot like ChatGPT_.I've decided to try make do as much as I can with smaller models that I can find on Hugging Face for this exercise.We can list the 5 most downloaded models on Hugging Face Hub for text classification through the huggingface API:

```
def list_most_popular(task: str):
    for rank, model in enumerate(
        list_models(filter=task, sort="downloads", direction=-1)
):
        if rank == 5:
            break
        print(f"{model.id}, {model.downloads}\n")
list_most_popular("text-classification")
```

Let's see the list:

| **Model**                                        | **Downloads** |
| ------------------------------------------------ | ------------- |
| nlptown/bert-base-multilingual-uncased-sentiment | 5805259       |
| SamLowe/roberta-base-go\_emotions                | 5292490       |
| cardiffnlp/twitter-roberta-base-irony            | 4427067       |
| salesken/query\_wellformedness\_score            | 4380505       |
| marieke93/MiniLM-evidence-types                  | 4370524       |

Tablee 3.2: The most popular text classification models on Hugging Face Hub.We can see that these models are about small ranges of categories such as sentiment, emotions, irony, or well-formedness. Let's use the sentiment model.

```
I've asked GPT-3.5 to put together a long rambling customer email complaining about a coffee machine. You can find the email on GitHub. Let's see what our sentiment model has to say:
from transformers import pipeline
sentiment_model = pipeline(
    task="sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)
print(sentiment_model(customer_email))
```

I am getting this result:

```
[{'label': '2 stars', 'score': 0.28999224305152893}]
```

Not a happy camper. Let's move on!Let's see the 5 most popular models for summarization as well:

| **Model**                        | **Downloads** |
| -------------------------------- | ------------- |
| t5-base                          | 2710309       |
| t5-small                         | 1566141       |
| facebook/bart-large-cnn          | 1150085       |
| sshleifer/distilbart-cnn-12-6    | 709344        |
| philschmid/bart-large-cnn-samsum | 477902        |

Table 3.3: The most popular summarization models on Hugging Face Hub.

All these models have a relatively small footprint compared to large models. Let's execute the summarization model remotely on a server:

```
from langchain import HuggingFaceHub
summarizer = HuggingFaceHub(
    repo_id="facebook/bart-large-cnn",
    model_kwargs={"temperature":0, "max_length":180}
)
def summarize(llm, text) -> str:
    return llm(f"Summarize this: {text}!")
summarize(summarizer, customer_email)
```

Please note that you need to have your `HUGGINGFACEHUB_API_TOKEN` set for this to work.I am seeing this summary:A customer's coffee machine arrived ominously broken, evoking a profound sense of disbelief and despair. "This heartbreaking display of negligence shattered my dreams of indulging in daily coffee perfection, leaving me emotionally distraught and inconsolable," the customer writes. "I hope this email finds you amidst an aura of understanding, despite the tangled mess of emotions swirling within me as I write to you," he adds.This summary is just passable, but not very convincing. There's still a lot of rambling in the summary. We could try other models or just go for an LLM with a prompt asking to summarize. Let's move on.It could be quite useful to know what kind of issue the customer is writing about. Let's ask **VertexAI**:

```
from langchain.llms import VertexAI
from langchain import PromptTemplate, LLMChain
template = """Given this text, decide what is the issue the customer is concerned about. Valid categories are these:
* product issues
* delivery problems
* missing or late orders
* wrong product
* cancellation request
* refund or exchange
* bad support experience
* no clear reason to be upset
Text: {email}
Category:
"""
prompt = PromptTemplate(template=template, input_variables=["email"])
llm = VertexAI()
llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
print(llm_chain.run(customer_email))
```

We get "product issues" back, which is correct for the long email example that I am using here.I hope it was exciting to see how quickly we can throw a few models and tools together in LangChain to get something that looks actually useful. We could easily expose this in an interface for customer service agents to see.Let's summarize.

### Summary

In this chapter, we've walked through four distinct ways of installing LangChain and other libraries needed in this book as an environment. Then, we've introduced several providers of models for text and images. For each of them, we explained where to get the API token, and demonstrated how to call a model. Finally, we've developed an LLM app for text classification in a use case for customer service. By chaining together various functionalities in LangChain, we can help reduce response times in customer service and make sure answers are accurate and to the point.In the chapters 3 and 4, we'll dive more into use cases such as question answering through augmented retrieval using tools such as web searches and chatbots relying on document search through indexing. Let's see if you remember some of the key takeaways from this chapter!

### Questions

Please have a look to see if you can come up with the answers to these questions. I'd recommend you go back to the corresponding sections of this chapter, if you are unsure about any of them:

1. How do you install LangChain?
2. List at least 4 cloud providers of LLMs apart from OpenAI!
3. What are Jina AI and Hugging Face?
4. How do you generate images with LangChain?
5. How do you run a model locally on your own machine rather than through a service?
6. How do you perform text classification in LangChain?
7. How can we help customer service agents in their work through generative AI?
