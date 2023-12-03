# Chapter 5. Advancing LLM Capabilities with the LangChain Framework and Plug-ins

## Chapter 5. Advancing LLM Capabilities with the LangChain Framework and Plug-ins

This chapter explores the worlds of the LangChain framework and GPT-4 plug-ins. We’ll look at how LangChain enables interaction with different language models and the importance of plug-ins in expanding the capabilities of GPT-4. This advanced knowledge will be fundamental in developing sophisticated, cutting-edge applications that rely on LLMs.

## The LangChain Framework

LangChain is a new framework dedicated to developing LLM-powered apps. You will find that the code integrating LangChain is much more elegant than the example provided in [Chapter 3](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch03.html#building\_apps\_with\_gpt\_4\_and\_chatgpt). The framework also provides many additional possibilities.

Installing LangChain is fast and easy with `pip install langchain`.

**WARNING**

At the time of this writing, LangChain is still in beta version 0.0.2_XX_, and new versions are released almost daily. Functionalities may be subject to change, so we recommend using caution when working with this framework.

LangChain’s key functionalities are divided into modules, as depicted in [Figure 5-1](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch05.html#fig\_1\_langchain\_modules).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098152475/files/assets/dagc_0501.png" alt="" height="554" width="579"><figcaption></figcaption></figure>

**Figure 5-1. LangChain modules**

Following are brief descriptions of these modules:

Models

The Models module is a standard interface provided by LangChain through which you can interact with various LLMs. The framework supports different model-type integrations from various providers, including OpenAI, Hugging Face, Cohere, GPT4All, and more.

Prompts

Prompts are becoming the new standard for programming LLMs. The Prompts module includes many tools for prompt management.

Indexes

This module allows you to combine LLMs with your data.

Chains

With this module, LangChain provides the Chain interface that allows you to create a sequence of calls that combine multiple models or prompts.

Agents

The Agents module introduces the Agent interface. An agent is a component that can process user input, make decisions, and choose the appropriate tools to accomplish a task. It works iteratively, taking action until it reaches a solution.

Memory

The Memory module allows you to persist state between chain or agent calls. By default, chains and agents are stateless, meaning they process each incoming request independently, as do the LLMs.

LangChain is a generic interface for different LLMs; you can review all the integrations [on its documentation page](https://oreil.ly/n5yNV). OpenAI and many other LLM providers are in this list of integrations. Most of these integrations need their API key to make a connection. For the OpenAI models, you can do this setup as we saw in [Chapter 2](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch02.html#a\_deep\_dive\_into\_the\_gpt\_4\_and\_chatgpt\_apis), with the key set in an `OPENAI_API_KEY` environment variable.

### Dynamic Prompts

The easiest way to show you how LangChain works is to present you with a simple script. In this example, OpenAI and LangChain are used to do a simple text completion:

```
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
template = """Question: {question}
Let's think step by step.
Answer: """
prompt = PromptTemplate(template=template, input_variables=["question"])
llm = ChatOpenAI(model_name="gpt-4")
llm_chain = LLMChain(prompt=prompt, llm=llm)
question = """ What is the population of the capital of the country where the
Olympic Games were held in 2016? """
llm_chain.run(question)
```

The output is as follows:

```
Step 1: Identify the country where the Olympic Games were held in 2016.
Answer: The 2016 Olympic Games were held in Brazil.
Step 2: Identify the capital of Brazil.
Answer: The capital of Brazil is Brasília.
Step 3: Find the population of Brasília.
Answer: As of 2021, the estimated population of Brasília is around 3.1 million.
So, the population of the capital of the country where the Olympic Games were 
held in 2016 is around 3.1 million. Note that this is an estimate and may
vary slightly.'
```

The `PromptTemplate` is responsible for constructing the input for the model. As such, it is a reproducible way to generate a prompt. It contains an input text string called a _template_, in which values can be specified via `input_variables`. In our example, the prompt we define automatically adds the “Let’s think step by step” part to the question.

The LLM used in this example is GPT-4; currently, the default model is `gpt-3.5-turbo`. The model is placed in the variable `llm` via the `ChatOpenAI()` function. This function assumes an OpenAI API key is set in the environment variable `OPENAI_API_KEY`, like it was in the examples in the previous chapters.

The prompt and the model are combined by the function `LLMChain()`, which forms a chain with the two elements. Finally, we need to call the `run()` function to request completion with the input question. When the `run()` function is executed, the `LLMChain` formats the prompt template using the input key values provided (and also memory key values, if available), passes the formatted string to the LLM, and finally returns the LLM output. We can see that the model automatically answers the question by applying the “Let’s think step by step” rule.

As you can see, dynamic prompts is a simple yet very valuable feature for complex applications and better prompt management.

### Agents and Tools

Agents and tools are the key functionalities of the LangChain framework: they can make your application extremely powerful. They allow you to solve complex problems by making it possible for LLMs to perform actions and integrate with various capabilities.

A _tool_ is a particular abstraction around a function that makes it easier for a language model to interact with it. An agent can use a tool to interact with the world. Specifically, the interface of a tool has a single text input and a single text output. There are many predefined tools in LangChain. These include Google search, Wikipedia search, Python REPL, a calculator, a world weather forecast API, and others. To get a complete list of tools, check out the [Tools page](https://oreil.ly/iMtOU) in the documentation provided by LangChain. You can also [build a custom tool](https://oreil.ly/\_dyBW) and load it into the agent you are using: this makes agents extremely versatile and powerful.

As we learned in [Chapter 4](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch04.html#advanced\_gpt\_4\_and\_chatgpt\_techniques), with “Let’s think step by step” in the prompt, you can increase, in a sense, the reasoning capacity of your model. Adding this sentence to the prompt asks the model to take more time to answer the question.

In this section, we introduce an agent for applications that require a series of intermediate steps. The agent schedules these steps and has access to various tools, deciding which to use to answer the user’s query efficiently. In a way, as with “Let’s think step by step,” the agent will have more time to plan its actions, allowing it to accomplish more complex tasks.

The high-level pseudocode of an agent looks like this:

1. The agent receives some input from the user.
2. The agent decides which tool, if any, to use and what text to enter into that tool.
3. That tool is then invoked with that input text, and an output text is received from the tool.
4. The tool’s output is fed into the context of the agent.
5. Steps 2 through 4 are repeated until the agent decides that it no longer needs to use a tool, at which point it responds directly to the user.

You might notice that this seems close to what we did in [Chapter 3](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch03.html#building\_apps\_with\_gpt\_4\_and\_chatgpt), with the example of the personal assistant who could answer questions and perform actions. LangChain agents allow you to develop this kind of behavior… but much more powerfully.

To better illustrate how an agent uses tools in LangChain, [Figure 5-2](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch05.html#fig\_2\_interaction\_between\_an\_agent\_and\_tools\_in\_langchai) provides a visual walkthrough of the interaction.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098152475/files/assets/dagc_0502.png" alt="" height="233" width="600"><figcaption></figcaption></figure>

**Figure 5-2. Interaction between an agent and tools in LangChain**

For this section, we want to be able to answer the following question: What is the square root of the population of the capital of the country where the Olympic Games were held in 2016? This question has no real interest, but it is a good demonstration of how LangChain agents and tools can add reasoning capabilities to LLMs.

If we ask the question as-is to GPT-3.5 Turbo, we get the following:

```
The capital of the country where the Olympic Games were held in 2016 is Rio de
Janeiro, Brazil. The population of Rio de Janeiro is approximately 6.32 million
people as of 2021. Taking the square root of this population, we get 
approximately 2,513.29. Therefore, the square root of the population of 
the capital of the country where the Olympic Games were held in 2016 is
approximately 2,513.29.
```

This answer is wrong on two levels: Brazil’s capital is Brasilia, not Rio de Janeiro, and the square root of 6.32 million is 2,513.96. We might be able to get better results by adding “Think step by step” or by using other prompt engineering techniques, but it would still be difficult to trust the result because of the model’s difficulties with reasoning and mathematical operations. Using LangChain gives us better guarantees of accuracy.

The following code gives a simple example of how an agent can use two tools in LangChain: Wikipedia and a calculator. After the tools are created via the function `load_tools()`, the agent is created with the function `initialize_agent()`. An LLM is needed for the agent’s reasoning; here, GPT-3.5 Turbo is used. The parameter `zero-shot-react-description` defines how the agent chooses the tool at each step. By setting the `verbose` value to `true`, we can view the agent’s reasoning and understand how it arrives at the final decision:

```
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
tools = load_tools(["wikipedia", "llm-math"], llm=llm)
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
question = """What is the square root of the population of the capital of the
Country where the Olympic Games were held in 2016?"""
agent.run(question)
```

**NOTE**

To run the Wikipedia tool, it is necessary to have installed the corresponding Python package `wikipedia`. This can be done with `pip install wikipedia`.

As you can see, the agent decides to query Wikipedia for information about the 2016 Summer Olympics:

```
> Entering new chain...
I need to find the country where the Olympic Games were held in 2016 and then find
the population of its capital city. Then I can take the square root of that population.
Action: Wikipedia
Action Input: "2016 Summer Olympics"
Observation: Page: 2016 Summer Olympics
[...]
```

The next lines of the output contain an extract from Wikipedia about the Olympics. Next, the agent uses the Wikipedia tool two additional times:

```
Thought:I need to search for the capital city of Brazil.
Action: Wikipedia
Action Input: "Capital of Brazil"
Observation: Page: Capitals of Brazil
Summary: The current capital of Brazil, since its construction in 1960, is
Brasilia. [...]
Thought: I have found the capital city of Brazil, which is Brasilia. Now I need 
to find the population of Brasilia.
Action: Wikipedia
Action Input: "Population of Brasilia"
Observation: Page: Brasilia
[...]
```

As a next step, the agent uses the calculator tool:

```
Thought: I have found the population of Brasilia, but I need to calculate the
square root of that population.
Action: Calculator
Action Input: Square root of the population of Brasilia (population: found in 
previous observation)
Observation: Answer: 1587.051038876822
```

And finally:

```
Thought:I now know the final answer
Final Answer: The square root of the population of the capital of the country
where the Olympic Games were held in 2016 is approximately 1587.
> Finished chain.
```

As you can see, the agent demonstrated complex reasoning capabilities: it completed four different steps before coming up with the final answer. The LangChain framework allows developers to implement these kinds of reasoning capabilities in just a few lines of code.

**TIP**

Although several LLMs can be used for the agent and GPT-4 is the most expensive among them, we have empirically obtained better results with GPT-4 for complex problems; we have observed that the results could quickly become inconsistent when smaller models are used for the agent’s reasoning. You may also receive errors because the model cannot answer in the expected format.

### Memory

In some applications, it is crucial to remember previous interactions, both in the short and long terms. With LangChain, you can easily add states to chains and agents to manage memory. Building a chatbot is the most common example of this capability. You can do this very quickly with `ConversationChain`—essentially turning a language model into a chat tool with just a few lines of code.

The following code uses the `text-ada-001` model to make a chatbot. It is a small model capable of performing only elementary tasks. However, it is the fastest model in the GPT-3 series and has the lowest cost. This model has never been fine-tuned to behave like a chatbot, but we can see that with only two lines of code with LangChain, we can use this simple completion model to chat:

```
from langchain import OpenAI, ConversationChain
chatbot_llm = OpenAI(model_name='text-ada-001')
chatbot = ConversationChain(llm=chatbot_llm , verbose=True)
chatbot.predict(input='Hello')
```

In the last line of the preceding code, we executed `predict(input='Hello')`. This results in the chatbot being asked to respond to our `'Hello'` message. And as you can see, the model answers:

```
> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is
talkative and provides lots of specific details from its context. If the AI
does not know the answer to a question, it truthfully says it does not know.
Current conversation:
Human: Hello
AI:
> Finished chain.
' Hello! How can I help you?'
```

Thanks to `verbose=True` in `ConversationChain`, we can look at the whole prompt used by LangChain. When we executed `predict(input='Hello')`, the LLM `text-ada-001` received not simply the `'Hello'` message but a complete prompt, which is between the tags `> Entering new ConversationChain chain…` and `> Finished chain`.

If we continue the conversation, you can see that the function keeps a conversation history in the prompt. If we then ask “Can I ask you a question? Are you an AI?” the history of the conversation will also be in the prompt:

```
> Entering new ConversationChain chain...
Prompt after formatting:
The following [...] does not know.
Current conversation:
Human: Hello
AI:  Hello! How can I help you?
Human: Can I ask you a question? Are you an AI?
AI:
> Finished chain.
'\n\nYes, I am an AI.'
```

The `ConversationChain` object uses prompt engineering techniques and memory techniques to transform any LLM that does text completion into a chat tool.

**WARNING**

Even if this LangChain feature allows all the language models to have chat capabilities, this solution is not as powerful as models like `gpt-3.5-turbo` and `gpt-4`, which have been fine-tuned specifically for chat. Furthermore, OpenAI has announced the deprecation of `text-ada-001`.

### Embeddings

Combining language models with your own text data is a powerful way to personalize the knowledge of the models you use in your apps. The principle is the same as that discussed in [Chapter 3](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch03.html#building\_apps\_with\_gpt\_4\_and\_chatgpt): the first step is _information retrieval_, which refers to taking a user’s query and returning the most relevant documents. The documents are then sent to the model’s input context to ask it to answer the query. This section shows how easy it is to do this with LangChain and embeddings.

An essential module in LangChain is `document_loaders`. With this module, you can quickly load your text data from different sources into your application. For example, your application can load CSV files, emails, PowerPoint documents, Evernote notes, Facebook chats, HTML pages, PDF documents, and many other formats. A complete list of loaders is available [in the official documentation](https://oreil.ly/t7nZx). Each of them is super easy to set. This example reuses the PDF of the [_Explorer’s Guide for The Legend of Zelda: Breath of the Wild_](https://oreil.ly/ZGu3z).

If the PDF is in the current working directory, the following code loads its contents and divides it by page:

```
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("ExplorersGuide.pdf")
pages = loader.load_and_split()
```

**NOTE**

To use the PDF loader, it is necessary to have the Python `pypdf` package installed. This can be done with `pip install pypdf`.

To do information retrieval, it is necessary to embed each loaded page. As we discussed in [Chapter 2](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch02.html#a\_deep\_dive\_into\_the\_gpt\_4\_and\_chatgpt\_apis), _embeddings_ are a technique used in information retrieval to convert non-numerical concepts, such as words, tokens, and sentences, into numerical vectors. The embeddings allow models to process relationships between these concepts efficiently. With OpenAI’s embeddings endpoint, developers can obtain numerical vector representations of input text, and LangChain has a wrapper to call these embeddings:

```
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
```

**NOTE**

To use `OpenAIEmbeddings`, install the `tiktoken` Python package with `pip install tiktoken`.

Indexes save pages’ embeddings and make searches easy. LangChain is centered on vector databases. It is possible to choose among many vector databases; a complete list is available [in the official documentation](https://oreil.ly/nJLCI). The following code snippet uses the [FAISS vector database](https://oreil.ly/7TMdI), a library for similarity search developed primarily at Meta’s [Fundamental AI Research group](https://ai.facebook.com/):

```
from langchain.vectorstores import FAISS
db = FAISS.from_documents(pages, embeddings)
```

**NOTE**

To use FAISS, it is necessary to install the `faiss-cpu` Python package with `pip install faiss-cpu`.

To better illustrate how the PDF document’s content is converted into pages of embeddings and stored in the FAISS vector database, [Figure 5-3](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch05.html#fig\_3\_creating\_and\_saving\_embeddings\_from\_a\_pdf\_document) visually summarizes the process.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098152475/files/assets/dagc_0503.png" alt="" height="153" width="600"><figcaption></figcaption></figure>

**Figure 5-3. Creating and saving embeddings from a PDF document**

And now it’s easy to search for similarities:

```
q = "What is Link's traditional outfit color?"
db.similarity_search(q)[0]
```

From the preceding code, we get the following:

```
Document(page_content='While Link’s traditional green 
              tunic is certainly an iconic look, his 
              wardrobe has expanded [...] Dress for Success', 
          metadata={'source': 'ExplorersGuide.pdf', 'page': 35})   
```

The answer to the question is that Link’s traditional outfit color is green, and we can see that the answer is in the selected content. The output says that the answer is on page 35 of _ExplorersGuide.pdf_. Remember that Python starts to count from zero; therefore, if you return to the original PDF file of the _Explorer’s Guide for The Legend of Zelda: Breath of the Wild_, the solution is on page 36 (not page 35).

[Figure 5-4](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch05.html#fig\_4\_the\_information\_retrieval\_looks\_for\_pages\_most\_sim) shows how the information retrieval process uses the embedding of the query and the vector database to identify the pages most similar to the query.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098152475/files/assets/dagc_0504.png" alt="" height="165" width="600"><figcaption></figcaption></figure>

**Figure 5-4. The information retrieval looks for pages most similar to the query**

You might want to integrate your embedding into your chatbot to use the information it has retrieved when it answers your questions. Again, with LangChain, this is straightforward to do in a few lines of code. We use `RetrievalQA`, which takes as inputs an LLM and a vector database. We then ask a question to the obtained object in the usual way:

```
from langchain.chains import RetrievalQA
from langchain import OpenAI
llm = OpenAI()
chain = RetrievalQA.from_llm(llm=llm, retriever=db.as_retriever())
q = "What is Link's traditional outfit color?"
chain(q, return_only_outputs=True)
```

We get the following answer:

```
{'result': " Link's traditional outfit color is green."}
```

[Figure 5-5](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch05.html#fig\_5\_to\_answer\_the\_user\_s\_question\_the\_retrieved\_infor) shows how `RetrievalQA` uses information retrieval to answer the user’s question. As we can see in this figure, “Make context” groups together the pages found by the information retrieval system and the user’s initial query. This enriched context is then sent to the language model, which can use the additional information added in the context to correctly answer the user’s question.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098152475/files/assets/dagc_0505.png" alt="" height="149" width="600"><figcaption></figcaption></figure>

**Figure 5-5. To answer the user’s question, the retrieved information is added to the context of the LLM**

You may wonder why it is necessary to do the information retrieval before sending the information from the document as input to the context of the language model. Indeed, current language models cannot consider large files with hundreds of pages. Therefore, we prefilter the input data if it is too large. This is the task of the information retrieval process. In the near future, as the size of input contexts increases, there will likely be situations for which the use of information retrieval techniques will not be technically necessary.

## GPT-4 Plug-ins

While language models, including GPT-4, have proven helpful in various tasks, they have inherent limitations. For example, these models can only learn from the data on which they were trained, which is often outdated or inappropriate for specific applications. In addition, their capabilities are limited to text generation. We have also seen that LLMs do not work for some tasks, such as complex calculations.

This section focuses on a groundbreaking feature of GPT-4: plug-ins (note that the GPT-3.5 model doesn’t have access to plug-in functionality). In the evolution of AI, plug-ins have emerged as a new transformative tool that redefines interaction with LLMs. The goal of plug-ins is to provide the LLM with broader capabilities, allowing the model to access real-time information, perform complex mathematical computations, and utilize third-party services.

We saw in [Chapter 1](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch01.html#gpt\_4\_and\_chatgpt\_essentials) that the model was not capable of performing complex calculations such as 3,695 × 123,548. In [Figure 5-6](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch05.html#fig\_6\_gpt\_4\_s\_use\_of\_the\_calculator\_plug\_in), we activate the Calculator plug-in and we can see that the model automatically calls the calculator when it needs to do a calculation, allowing it to find the right solution.

With an iterative deployment approach, OpenAI incrementally adds plug-ins to GPT-4, which enables OpenAI to consider practical uses for plug-ins as well as any security and customization challenges that they may introduce. While plug-ins have been available to all paying users since May 2023, the ability to create new plug-ins was not yet available for all developers at the time of this writing.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098152475/files/assets/dagc_0506.png" alt="" height="507" width="600"><figcaption></figcaption></figure>

**Figure 5-6. GPT-4’s use of the Calculator plug-in**

OpenAI’s goal is to create an ecosystem where plug-ins can help shape the future dynamics of human–AI interaction. Today it is inconceivable for a serious business not to have its own website, but maybe soon, every company will need to have its own plug-in. Indeed, several early plug-ins have already been brought to life by companies such as Expedia, FiscalNote, Instacart, KAYAK, Klarna, Milo, OpenTable, Shopify, and Zapier.

Beyond their primary function, plug-ins serve to extend the functionality of GPT-4 in several ways. In a sense, some similarities exist between plug-ins and the agents and tools discussed in [“The LangChain Framework”](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch05.html#the\_langchain\_framework). For example, plug-ins can enable an LLM to retrieve real-time information such as sports scores and stock prices, extract data from knowledge bases such as corporate documents, and perform tasks at the demand of users, such as booking a flight or ordering a meal. Both are designed to help AI access up-to-date information and perform calculations. However, the plug-ins in GPT-4 focus more on third-party services than LangChain’s tools.

This section introduces the fundamental concepts for creating a plug-in by exploring the key points of the examples presented on the OpenAI website. We will use the example of a to-do list definition plug-in. Plug-ins are still in a limited beta version as we write this book, so readers are encouraged to visit the [OpenAI reference page](https://platform.openai.com/docs/plugins/introduction) for the latest information. Note also that during the beta phase, users must manually enable their plug-in in ChatGPT’s user interface, and as a developer, you can share your plug-in with no more than 100 users.

### Overview

As a plug-in developer, you must create an API and associate it with two descriptive files: a plug-in manifest and an OpenAPI specification. When the user starts interacting with GPT-4, OpenAI sends a hidden message to GPT if your plug-in is installed. This message briefly introduces your plug-in, including its description, endpoints, and examples.

The model then becomes an intelligent API caller. When a user asks questions about your plug-in, the model can call your plug-in API. The decision to call the plug-in is made based on the API specification and a natural language description of the circumstances in which your API should be used. Once the model has decided to call your plug-in, it incorporates the API results into its context to provide its response to the user. Therefore, the plug-in’s API responses must return raw data instead of natural language responses. This allows GPT to generate its own natural language response based on the returned data.

For example, if a user asks “Where should I stay in New York?”, the model can use a hotel booking plug-in and then combine the plug-in’s API response with its language generation capabilities to provide an answer that is both informative and user friendly.

### The API

Here is a simplified version of the code example of the to-do list definition plug-in provided on[ OpenAI’s GitHub](https://oreil.ly/un13K):

```
import json
import quart
import quart_cors
from quart import request
app = quart_cors.cors(
    quart.Quart(__name__), allow_origin="https://chat.openai.com"
)
# Keep track of todo's. Does not persist if Python session is restarted.
_TODOS = {}
@app.post("/todos/<string:username>")
async def add_todo(username):
    request = await quart.request.get_json(force=True)
    if username not in _TODOS:
        _TODOS[username] = []
    _TODOS[username].append(request["todo"])
    return quart.Response(response="OK", status=200)
@app.get("/todos/<string:username>")
async def get_todos(username):
    return quart.Response(
        response=json.dumps(_TODOS.get(username, [])), status=200
    )
@app.get("/.well-known/ai-plugin.json")
async def plugin_manifest():
    host = request.headers["Host"]
    with open("./.well-known/ai-plugin.json") as f:
        text = f.read()
        return quart.Response(text, mimetype="text/json")
@app.get("/openapi.yaml")
async def openapi_spec():
    host = request.headers["Host"]
    with open("openapi.yaml") as f:
        text = f.read()
        return quart.Response(text, mimetype="text/yaml")
def main():
    app.run(debug=True, host="0.0.0.0", port=5003)
if __name__ == "__main__":
    main()
```

This Python code is an example of a simple plug-in that manages a to-do list. First the variable `app` is initialized with `quart_cors.cors()`. This line of code creates a new Quart application and configures it to allow cross-origin resource sharing (CORS) from [_https://chat.openai.com_](https://chat.openai.com/). Quart is a Python web microframework, and Quart-CORS is an extension that enables control over CORS. This setup allows the plug-in to interact with the ChatGPT application hosted at the specified URL.

Then the code defines several HTTP routes corresponding to different functionalities of the to-do list plug-in: the `add_todo` function, associated with a `POST` request, and the `get_todos` function, associated with a `GET` request.

Next, two additional endpoints are defined: `plugin_manifest` and `openapi_spec`. These endpoints serve the plug-in’s manifest file and the OpenAPI specification, which are crucial for the interaction between GPT-4 and the plug-in. These files contain detailed information about the plug-in and its API, which GPT-4 uses to know how and when to use the plug-in.

### The Plug-in Manifest

Each plug-in requires an _ai-plugin.json_ file on the API’s domain. So, for example, if your company provides service on _thecompany.com_, you must host this file at _https://thecompany.com/.well-known_. OpenAI will look for this file in _/.well-known/ai-plugin.json_ when installing the plug-in. Without this file, the plug-in can’t be installed.

Here is a minimal definition of the required _ai-plugin.json_ file:

```
{
    "schema_version": "v1",
    "name_for_human": "TODO Plugin",
    "name_for_model": "todo",
    "description_for_human": "Plugin for managing a TODO list. \
        You can add, remove and view your TODOs.",
    "description_for_model": "Plugin for managing a TODO list. \
        You can add, remove and view your TODOs.",
    "auth": {
        "type": "none"
    },
    "api": {
        "type": "openapi",
        "url": "http://localhost:3333/openapi.yaml",
        "is_user_authenticated": false
    },
    "logo_url": "http://localhost:3333/logo.png",
    "contact_email": "support@thecompany.com",
    "legal_info_url": "http://www.thecompany.com/legal"
}      
```

The fields are detailed in [Table 5-1](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch05.html#table-5-1).

| Field name              | Type   | Description                                                                                                                                                                                |
| ----------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `name_for_model`        | String | A short name the model uses to know your plug-in. It can only include letters and numbers, and it can have no more than 50 characters.                                                     |
| `name_for_human`        | String | The name people see. It could be your company’s full name, but it must be fewer than 20 characters.                                                                                        |
| `description_for_human` | String | A simple explanation of what your plug-in does. It’s for people to read and should be fewer than 100 characters.                                                                           |
| `description_for_model` | String | A detailed explanation that helps the AI understand your plug-in. Therefore, explaining the plug-in’s purpose to the model is crucial. The description can be up to 8,000 characters long. |
| `logo_url`              | String | The URL of your plug-in’s logo. The logo should ideally be 512 × 512 pixels.                                                                                                               |
| `contact_email`         | String | An email address people can use if they need help.                                                                                                                                         |
| `legal_info_url`        | String | A web address that lets users find more details about your plug-in.                                                                                                                        |

### The OpenAPI Specification

The next step in creating your plug-in is to create the _openapi.yaml_ file with your API specification. This file must follow the OpenAPI standard (see [“Understanding the OpenAPI Specification ”](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch05.html#understandingopenapi)). The GPT model only knows your API through the information detailed in this API specification file and the manifest file.

Here is an example with the first line of an _openapi.yaml_ file for the to-do list definition plug-in:

```
openapi: 3.0.1
info:
  title: TODO Plugin
  description: A plugin that allows the user to create and manage a TODO list
  using ChatGPT. If you do not know the user's username, ask them first before
  making queries to the plugin. Otherwise, use the username "global".
  version: 'v1'
servers:
  - url: http://localhost:5003
paths:
  /todos/{username}:
    get:
      operationId: getTodos
      summary: Get the list of todos
      parameters:
      - in: path
        name: username
        schema:
            type: string
        required: true
        description: The name of the user.
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/getTodosResponse'
[...]
```

Think of the OpenAPI Specification as descriptive documentation that should be enough by itself to understand and use your API. When a search is performed in GPT-4, the description in the info section is used to determine the relevance of the plug-in to the user’s search. The rest of the OpenAPI Specification follows the standard OpenAPI format. Many tools can automatically generate OpenAPI specifications based on your existing API code or the other way around.

## UNDERSTANDING THE OPENAPI SPECIFICATION

The [OpenAPI Specification](https://oreil.ly/1asy5) (previously known as the Swagger Specification) is a standard for describing HTTP APIs. An OpenAPI definition allows consumers to interact with the remote service without requiring additional documentation or access to the source code. An OpenAPI document can serve as a foundation for various valuable use cases, such as generating API documentation, creating servers and clients in multiple programming languages through code generation tools, facilitating testing processes, and much more.

An OpenAPI document, in JSON or YAML format, defines or describes the API and the API’s elements. The basic OpenAPI documentation starts with the version, title, description, and version number.

If you want to delve further into this topic, the [OpenAPI GitHub repository](https://github.com/OAI/OpenAPI-Specification) contains documentation and various examples.

### Descriptions

When a user request could potentially benefit from a plug-in, the model initiates a scan of the endpoint descriptions within the OpenAPI Specification, as well as the `description_for_model` attribute in the manifest file. Your goal is to create the most appropriate response, which often involves testing different requests and descriptions.

The OpenAPI document should provide a wide range of details about the API, such as the available functions and their respective parameters. It should also contain attribute-specific “description” fields that provide valuable, naturally written explanations of what each function does and what type of information a query field expects. These descriptions guide the model in making the most appropriate use of the API.

A key element in this process is the `description_for_model` attribute. This gives you a way to inform the model on how to use the plug-in. Creating concise, clear, and descriptive instruction is highly recommended.

However, following certain best practices when writing these descriptions is essential:

* Do not attempt to influence the mood, personality, or exact responses of GPT.
* Avoid directing GPT to use a specific plug-in unless the user explicitly requests that category of service.
* Do not prescribe specific triggers for GPT to use the plug-in, as it is designed to autonomously determine when the use of a plug-in is appropriate.

To recap, developing a plug-in for GPT-4 involves creating an API, specifying its behavior in an OpenAPI specification, and describing the plug-in and its usage in a manifest file. With this setup, GPT-4 can effectively act as an intelligent API caller, expanding its capabilities beyond text generation.

## Summary

The LangChain framework and GPT-4 plug-ins represent a significant leap forward in maximizing the potential of LLMs.

LangChain, with its robust suite of tools and modules, has become a central framework in the field of LLM. Its versatility in integrating different models, managing prompts, combining data, sequencing chains, processing agents, and employing memory management opens new avenues for developers and AI enthusiasts alike. The examples in [Chapter 3](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch03.html#building\_apps\_with\_gpt\_4\_and\_chatgpt) proved the limits of writing complex instructions from scratch with the ChatGPT and GPT-4 models. Remember, the true potential of LangChain lies in the creative use of these features to solve complex tasks and transform the generic language models into powerful, fine-grained applications.

GPT-4 plug-ins are a bridge between the language model and the contextual information available in real time. This chapter showed that developing plug-ins requires a well-structured API and descriptive files. Therefore, providing detailed and natural descriptions in these files is essential. This will help GPT-4 make the best use of your API.

The exciting world of LangChain and GPT-4 plug-ins is a testament to the rapidly evolving landscape of AI and LLMs. The insights provided in this chapter are just a tiny taste of the transformative potential of these tools.

## Conclusion

This book has equipped you with the necessary foundational and advanced knowledge to harness the power of LLMs and implement them in real-world applications. We covered everything from foundational principles and API integrations to advanced prompt engineering and fine-tuning, leading you toward practical use cases with OpenAI’s GPT-4 and ChatGPT models. We ended the book with a detailed look at how the LangChain framework and plug-ins can enable you to unleash the power of LLMs and build truly innovative applications.

You now have the tools at your disposal to pioneer further into the realm of AI, developing innovative applications that leverage the strength of these advanced language models. But remember, the AI landscape is continuously evolving; so it’s essential to keep on eye on advancements and adapt accordingly. This journey into the world of LLMs is only the beginning, and your exploration should not stop here. We encourage you to use your new knowledge to explore the future of technology with artificial intelligence.
