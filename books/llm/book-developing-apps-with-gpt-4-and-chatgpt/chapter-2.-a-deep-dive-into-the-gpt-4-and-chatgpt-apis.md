# Chapter 2. A Deep Dive into the GPT-4 and ChatGPT APIs

## Chapter 2. A Deep Dive into the GPT-4 and ChatGPT APIs

This chapter examines the GPT-4 and ChatGPT APIs in detail. The goal of this chapter is to give you a solid understanding of the use of these APIs so that you can effectively integrate them into your Python applications. By the end of this chapter, you will be well equipped to use these APIs and exploit their powerful capabilities in your own development projects.

We’ll start with an introduction to the OpenAI Playground. This will allow you to get a better understanding of the models before writing any code. Next, we will look at the OpenAI Python library. This includes the login information and a simple “Hello World” example. We will then cover the process of creating and sending requests to the APIs. We will also look at how to manage API responses. This will ensure that you know how to interpret the data returned by these APIs. In addition, this chapter will cover considerations such as security best practices and cost management.

As we progress, you will gain practical knowledge that will be very useful in your journey as a Python developer working with GPT-4 and ChatGPT. All the Python code included in this chapter is available in [the book’s GitHub repository](https://oreil.ly/DevAppsGPT\_GitHub).

**NOTE**

Before going any further, please check the [OpenAI usage policies](https://openai.com/policies/usage-policies), and if you don’t already have an account, create one on the [OpenAI home page](https://openai.com/). You can also have a look at the other legal documentation on the [Terms and Policies page](https://openai.com/policies). The concepts introduced in [Chapter 1](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch01.html#gpt\_4\_and\_chatgpt\_essentials) are also essential for using the OpenAI API and libraries.

## Essential Concepts

OpenAI offers several models that are designed for various tasks, and each one has its own pricing. On the following pages, you will find a detailed comparison of the available models and tips on how to choose which ones to use. It’s important to note that the purpose for which a model was designed—whether for text completion, chat, or editing—impacts how you would use its API. For instance, the models behind ChatGPT and GPT-4 are chat based and use a chat endpoint.

The concept of prompts was introduced in [Chapter 1](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch01.html#gpt\_4\_and\_chatgpt\_essentials). Prompts are not specific to the OpenAI API but are the entry point for all LLMs. Simply put, prompts are the input text that you send to the model, and they are used to instruct the model on the specific task you want it to perform. For the ChatGPT and GPT-4 models, prompts have a chat format, with the input and output messages stored in a list. We will explore the details of this prompt format in this chapter.

The concept of tokens was also described in [Chapter 1](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch01.html#gpt\_4\_and\_chatgpt\_essentials). Tokens are words or parts of words. A rough estimate is that 100 tokens equal approximately 75 words for an English text. Requests to the OpenAI models are priced based on the number of tokens used: that is, the cost of a call to the API depends on the length of both the input text and the output text. You will find more details on managing and controlling the number of input and output tokens in [“Using ChatGPT and GPT-4”](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch02.html#using\_chatgpt\_and\_gpt\_4) and [“Using Other Text Completion Models”](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch02.html#using\_other\_text\_completion\_models).

These concepts are summarized in [Figure 2-1](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch02.html#fig\_1\_essential\_concepts\_for\_using\_the\_openai\_api).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098152475/files/assets/dagc_0201.png" alt="" height="91" width="600"><figcaption></figcaption></figure>

**Figure 2-1. Essential concepts for using the OpenAI API**

Now that we have discussed the concepts, let’s move on to the details of the models.

## Models Available in the OpenAI API

The OpenAI API gives you access to [several models developed by OpenAI](https://platform.openai.com/docs/models). These models are available as a service over an API (through a direct HTTP call or a provided library), meaning that OpenAI runs the models on distant servers, and developers can simply send queries to them.

Each model comes with a different set of features and pricing. In this section, we will look at the LLMs provided by OpenAI through its API. It is important to note that these models are proprietary, so you cannot directly modify the code to adapt the models to your needs. But as we will see later, you can fine-tune some of them on your specific data via the OpenAI API.

**NOTE**

Some older OpenAI models, including the GPT-2 model, are not proprietary. While you can download the GPT-2 model from [Hugging Face](https://oreil.ly/39Bu5) or [GitHub](https://oreil.ly/CYPN6), you cannot access it through the API.

Since many of the models provided by OpenAI are continually updated, it is difficult to give a complete list of them in this book; an updated list of models that OpenAI provides is available in the [online documentation](https://platform.openai.com/docs/models). Therefore, here we will focus on the most important models:

InstructGPT

This family of models can process many single-turn completion tasks. The `text-ada-001` model is only capable of simple completion tasks but is also the fastest and least expensive model in the GPT-3 series. Both `text-babbage-001` and `text-curie-001` are a little more powerful but also more expensive. The `text-davinci-003` model can perform all completion tasks with excellent quality, but it is also the most expensive in the family of GPT-3 models.

ChatGPT

The model behind ChatGPT is `gpt-3.5-turbo`. As a chat model, it can take a series of messages as input and return an appropriately generated message as output. While the chat format of `gpt-3.5-turbo` is designed to facilitate multiturn conversations, it is also possible to use it for single-turn tasks without dialogue. In single-turn tasks, the performance of `gpt-3.5-turbo` is comparable to that of `text-davinci-003`, and since `gpt-3.5-turbo` is one-tenth the price, with more or less equivalent performance, it is recommended that you use it by default for single-turn tasks. The `gpt-3.5-turbo` model has a context size of 4,000 tokens, which means it can receive 4,000 tokens as input. OpenAI also provides another model, called `gpt-3.5-turbo-16k`, with the same capabilities as the standard `gpt-3.5-turbo` model but with four times the context size.

GPT-4

This is the largest model released by OpenAI. It has also been trained on the most extensive multimodal corpus of text and images. As a result, it has knowledge and expertise in many domains. GPT-4 can follow complex natural language instructions and solve difficult problems accurately. It can be used for both chat and single-turn tasks with high accuracy. OpenAI offers two GPT-4 models: `gpt-4` has a context size of 8,000 tokens, and `gpt-4-32k` has a context size of 32,000 tokens. A context of 32,000 represents approximately 24,000 words, which is a context of approximately 40 pages.

Both GPT-3.5 Turbo and GPT-4 are continually updated. When we refer to the models `gpt-3.5-turbo`, `gpt-3.5-turbo-16k`, `gpt-4`, and `gpt-4-32k`, we are referring to the latest version of these models.

Developers often need more stability and visibility into the LLM version they are using in their applications. It can be difficult for developers to use model languages in which versions can change from one night to the next and can behave differently for the same input prompt. For this purpose, static snapshot versions of these models are also available. At the time of this writing, the most recent snapshot versions were `gpt-3.5-turbo-0613`, `gpt-3.5-turbo-16k-0613`, `gpt-4-0613`, and `gpt-4-32k-0613`.

As discussed in [Chapter 1](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch01.html#gpt\_4\_and\_chatgpt\_essentials), OpenAI recommends using the InstructGPT series rather than the original GPT-3–based models. These models are still available in the API under the names `davinci`, `curie`, `babbage`, and `ada`. Given that these models can provide strange, false, and misleading answers, as seen in [Chapter 1](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch01.html#gpt\_4\_and\_chatgpt\_essentials), caution in their use is advised.&#x20;

**NOTE**

The SFT model (presented in [Chapter 1](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch01.html#gpt\_4\_and\_chatgpt\_essentials)) obtained after the supervised fine-tuning stage, which did not go through the RLHF stage, is also available in the API under the name `davinci-instruct-beta`.

## Trying GPT Models with the OpenAI Playground

An excellent way to test the different language models provided by OpenAI directly, without coding, is to use the OpenAI Playground, a web-based platform that allows you to quickly test the various LLMs provided by OpenAI on specific tasks. The Playground lets you write prompts, select the model, and easily see the output that is generated.

Here’s how to access the Playground:

1. Navigate to the [OpenAI home page](https://openai.com/) and click Developers, then Overview.
2. If you already have an account and are not logged in, click Login at the upper right of the screen. If you don’t have an account with OpenAI, you will need to create one in order to use the Playground and most of the OpenAI features. Click Sign Up at the upper right of the screen. Note that because there is a charge for the Playground and the API, you will need to provide a means of payment.
3. Once you are logged in, you will see the link to join the Playground at the upper left of the web page. Click the link, and you should see something similar to [Figure 2-2](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch02.html#fig\_2\_the\_openai\_playground\_interface\_in\_text\_completion).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098152475/files/assets/dagc_0202.png" alt="" height="601" width="600"><figcaption></figcaption></figure>

**Figure 2-2. The OpenAI Playground interface in Text Completion mode**

**NOTE**

The ChatGPT Plus option is independent of using the API or the Playground. If you have subscribed to the ChatGPT Plus service, you will still be charged for using the API and the Playground.

The main whitespace in the center of the interface is for your input message. After writing your message, click Submit to generate a completion to your message. In the example in [Figure 2-2](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch02.html#fig\_2\_the\_openai\_playground\_interface\_in\_text\_completion), we wrote “As Descartes said, I think therefore”, and after we clicked Submit, the model completed our input with “I am”.

**WARNING**

Every time you click Submit, your OpenAI account is billed for the usage. We give more information on prices later in this chapter, but as an example, this completion cost almost $0.0002.

There are many options around the sides of the interface. Let’s start at the bottom. To the right of the Submit button is an undo button \[labeled (A) in the figure] that deletes the last generated text. In our case, it will delete “I am”. Next is the regenerate button \[labeled (B) in the figure], which regenerates text that was just deleted. This is followed by the history button \[labeled (C)], which contains all your requests from the previous 30 days. Note that once you are in the history menu, it is easy to delete requests if necessary for privacy reasons.

The options panel on the right side of the screen provides various settings related to the interface and the chosen model. We will only explain some of these options here; others will be covered later in the book. The first drop-down list on the right is the Mode list \[labeled (D)]. At the time of this writing, the available modes are Chat (default), Complete, and Edit.

**NOTE**

Complete and Edit modes are marked as legacy at the time of this book’s writing and will probably disappear in January 2024.

As demonstrated previously, the language model strives to complete the user’s input prompt seamlessly in the Playground’s default mode.

[Figure 2-3](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch02.html#fig\_3\_the\_openai\_playground\_interface\_in\_chat\_mode) shows an example of using the Playground in Chat mode. On the left of the screen is the System pane \[labeled (E)]. Here you can describe how the chat system should behave. For instance, in [Figure 2-3](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch02.html#fig\_3\_the\_openai\_playground\_interface\_in\_chat\_mode), we asked it to be a helpful assistant who loves cats. We also asked it to only talk about cats and to give short answers. The dialogue that results from having set these parameters is displayed in the center of the screen.

If you want to continue the dialogue with the system, click “Add message” \[(F)], enter your message, and click Submit \[(G)]. It is also possible to define the model on the right \[(H)]; here we use GPT-4. Note that not all models are available in all modes. For instance, only GPT-4 and GPT-3.5 Turbo are available in Chat mode.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098152475/files/assets/dagc_0203.png" alt="" height="366" width="600"><figcaption></figcaption></figure>

**Figure 2-3. The OpenAI Playground interface in Chat mode**

Another mode available in the Playground is Edit. In this mode, shown in [Figure 2-4](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch02.html#fig\_4\_the\_openai\_playground\_interface\_in\_edit\_mode), you provide some text \[(I)] and instruction \[(J)], and the model will attempt to modify the text accordingly. In this example, a text describing a young man who is going on a trip is given. The model is instructed to change the subject of the text to an old woman, and you can see that the result respects the instructions \[(K)].

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098152475/files/assets/dagc_0204.png" alt="" height="216" width="600"><figcaption></figcaption></figure>

**Figure 2-4. The OpenAI Playground interface in Edit mode**

On the right side of the Playground interface, below the Mode drop-down list, is the Model drop-down list \[(L)]. As you have already seen, this is where you choose the LLM. The models available in the drop-down list depend on the selected mode. Below the Model drop-down list are parameters, such as Temperature \[(M)], that define the model’s behavior. We will not go into the details of these parameters here. Most of them will be explored when we closely examine how these different models work.

At the top of the screen is the “Load a preset” drop-down list \[(N)] and four buttons. In [Figure 2-2](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch02.html#fig\_2\_the\_openai\_playground\_interface\_in\_text\_completion), we used the LLM to complete the sentence “As Descartes said, I think therefore”, but it is possible to make the model perform particular tasks by using appropriate prompts. [Figure 2-5](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch02.html#fig\_5\_drop\_down\_list\_of\_examples) shows a list of common tasks the model can perform associated with an example of a preset.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098152475/files/assets/dagc_0205.png" alt="" height="679" width="600"><figcaption></figcaption></figure>

**Figure 2-5. Drop-down list of examples**

It should be noted that the proposed presets define not only the prompt but also some options on the right side of the screen. For example, if you click Grammatical Standard English, you will see in the main window the prompt displayed in [Figure 2-6](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch02.html#fig\_6\_example\_prompt\_for\_grammatical\_standard\_english).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098152475/files/assets/dagc_0206.png" alt="" height="153" width="600"><figcaption></figcaption></figure>

**Figure 2-6. Example prompt for Grammatical Standard English**

If you click Submit, you will obtain the following response: “She did not go to the market.” You can use the prompts proposed in the drop-down list as a starting point, but you will always have to modify them to fit your problem. OpenAI also provides a [complete list of examples](https://platform.openai.com/examples) for different tasks.

Next to the “Load a preset” drop-down list in [Figure 2-4](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch02.html#fig\_4\_the\_openai\_playground\_interface\_in\_edit\_mode) is the Save button \[(O)]. Imagine that you have defined a valuable prompt with a model and its parameter for your task, and you want to easily reuse it later in the Playground. This Save button will save the current state of the Playground as a preset. You can give your preset a name and a description, and once saved, your preset will appear in the “Load a preset” drop-down list.

The second-to-last button at the top of the interface is called “View code” \[(P)]. It gives the code to run your test in the Playground directly in a script. You can request code in Python, Node.js, or cURL to interact directly with the OpenAI remote server in a Linux terminal. If the Python code of our example “As Descartes said, I think therefore” is asked, we get the following:

```
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
response = openai.Completion.create(
    model="text-davinci-003",
    prompt="As Descartes said, I think therefore",
    temperature=0.7,
    max_tokens=3,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
)
```

Now that you understand how to use the Playground to test OpenAI language models without coding, let’s discuss how to obtain and manage your API keys for OpenAI services.

## Getting Started: The OpenAI Python Library

In this section, we’ll focus on how to use API keys in a small Python script, and we’ll perform our first test with this OpenAI API.

OpenAI provides GPT-4 and ChatGPT as a service. This means users cannot have direct access to the models’ code and cannot run the models on their own servers. However, OpenAI manages the deployment and running of its models, and users can call these models as long as they have an account and a secret key.

Before completing the following steps, make sure you are logged in on the [OpenAI web page](https://platform.openai.com/login?launch).

### OpenAI Access and API Key

OpenAI requires you to have an API key to use its services. This key has two purposes:

* It gives you the right to call the API methods.
* It links your API calls to your account for billing purposes.

You must have this key in order to call the OpenAI services from your application.

To obtain the key, navigate to the [OpenAI platform](https://platform.openai.com/) page. In the upper-right corner, click your account name and then “View API keys,” as shown in [Figure 2-7](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch02.html#fig\_7\_openai\_menu\_to\_select\_view\_api\_keys).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098152475/files/assets/dagc_0207.png" alt="" height="716" width="600"><figcaption></figcaption></figure>

**Figure 2-7. OpenAI menu to select “View API keys”**

When you are on the “API keys” page, click “Create new secret key” and make a copy of your key. This key is a long string of characters starting with _sk-_.

**WARNING**

Keep this key safe and secure because it is directly linked to your account, and a stolen key could result in unwanted costs.

Once you have your key, the best practice is to export it as an environment variable. This will allow your application to access the key without writing it directly in your code. Here is how to do that.

For Linux or Mac:

```
# set environment variable OPENAI_API_KEY for current session
export OPENAI_API_KEY=sk-(...)
# check that environment variable was set
echo $OPENAI_API_KEY
```

For Windows:

```
# set environment variable OPENAI_API_KEY for current session
set OPENAI_API_KEY=sk-(...)
# check that environment variable was set
echo %OPENAI_API_KEY%
```

The preceding code snippets will set an environment variable and make your key available to other processes that are launched from the same shell session. For Linux systems, it is also possible to add this code directly to your _.bashrc_ file. This will allow access to your environment variable in all your shell sessions. Of course, do not include these command lines in the code you push to a public repository.

To permanently add/change an environment variable in Windows 11, press the Windows key + R key simultaneously to open the Run Program Or File window. In this window, type **sysdm.cpl** to go to the System Properties panel. Then click the Advanced tab followed by the Environment Variables button. On the resulting screen, you can add a new environment variable with your OpenAI key.

**TIP**

OpenAI provides a detailed [page on API key safety](https://oreil.ly/2Qobg).

Now that you have your key, it’s time to write your first “Hello World” program with the OpenAI API.

### “Hello World” Example

This section shows the first lines of code with the OpenAI Python library. We will start with a classic “Hello World” example to understand how OpenAI provides its services.

Install the Python library with _pip_:

```
pip install openai
```

Next, access the OpenAI API in Python:

```
import openai
# Call the openai ChatCompletion endpoint
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello World!"}],
)
# Extract the response
print(response["choices"][0]["message"]["content"])
```

You will see the following output:

```
Hello there! How may I assist you today?
```

Congratulations! You just wrote your first program using the OpenAI Python library.

Let’s go through the details of using this library.

**TIP**

The OpenAI Python library also provides a command-line utility. The following code, running in a terminal, is equivalent to executing the previous “Hello World” example:

```
openai api chat_completions.create -m gpt-3.5-turbo \
    -g user "Hello world"
```

It is also possible to interact with the OpenAI API through HTTP requests or the official Node.js library, as well as other [community-maintained libraries](https://platform.openai.com/docs/libraries).

As you may have observed, the code snippet does not explicitly mention the OpenAI API key. This is because the OpenAI library is designed to automatically look for an environment variable named `OPENAI_API_KEY`. Alternatively, you can point the `openai` module at a file containing your key with the following code:

```
# Load your API key from file
openai.api_key_path = <PATH>, 
```

Or you can manually set the API key within your code using the following method:

```
# Load your API key 
openai.api_key = os.getenv("OPENAI_API_KEY")
```

Our recommendation is to follow a widespread convention for environment variables: store your key in a _.env_ file, which is removed from source control in the _.gitignore_ file. In Python, you can then run the `load_dotenv` function to load the environment variables and import the _openai_ library:

```
from dotenv import load_dotenv
load_dotenv()
import openai
```

It is important to have the `openai` import declaration after loading the _.env_ file; otherwise, the settings for OpenAI will not be applied correctly.

Now that we’ve covered the basic concepts of ChatGPT and GPT-4, we can move on to the details of their use.

## Using ChatGPT and GPT-4

This section discusses how to use the model running behind ChatGPT and GPT-4 with the OpenAI Python library.

At the time of this writing, GPT 3.5 Turbo is the least expensive and most versatile model. Therefore, it is also the best choice for most use cases. Here is an example of its use:

```
import openai
# For GPT 3.5 Turbo, the endpoint is ChatCompletion
openai.ChatCompletion.create(
    # For GPT 3.5 Turbo, the model is "gpt-3.5-turbo"
    model="gpt-3.5-turbo",
    # Conversation as a list of messages.
    messages=[
        {"role": "system", "content": "You are a helpful teacher."},
        {
            "role": "user",
            "content": "Are there other measures than time complexity for an \
            algorithm?",
        },
        {
            "role": "assistant",
            "content": "Yes, there are other measures besides time complexity \
            for an algorithm, such as space complexity.",
        },
        {"role": "user", "content": "What is it?"},
    ],
)
```

In the preceding example, we used the minimum number of parameters—that is, the LLM used to do the prediction and the input messages. As you can see, the conversation format in the input messages allows multiple exchanges to be sent to the model. Note that the API does not store previous messages in its context. The question `"What is it?"` refers to the previous answer and only makes sense if the model has knowledge of this answer. The entire conversation must be sent each time to simulate a chat session. We will discuss this further in the next section.

The GPT 3.5 Turbo and GPT-4 models are optimized for chat sessions, but this is not mandatory. Both models can be used for multiturn conversations and single-turn tasks. They also work well for traditional completion tasks if you specify a prompt asking for a completion.

Both ChatGPT and GPT-4 use the same endpoint: `openai.ChatCompletion`. Changing the model ID allows developers to switch between GPT-3.5 Turbo and GPT-4 without any other code changes.

### Input Options for the Chat Completion Endpoint

Let’s look in more detail at how to use the `openai.ChatCompletion` endpoint and its `create` method.

**NOTE**

The `create` method lets users call OpenAI models. Other methods are available but aren’t helpful for interacting with the models. You can access the Python library code on OpenAI’s GitHub [Python library repository](https://oreil.ly/MQ2aQ).

#### Required input parameters

The `openai.ChatCompletion` endpoint and its `create` method have several input parameters, but only two are required, as outlined in [Table 2-1](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch02.html#table-2-1).

| Field name | Type   | Description                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ---------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`    | String | The ID of the model to use. Currently, the available models are `gpt-4`, `gpt-4-0613`, `gpt-4-32k`, `gpt-4-32k-0613`, `gpt-3.5-turbo`, `gpt-3.5-turbo-0613`, `gpt-3.5-turbo-16k`, and `gpt-3.5-turbo-16k-0613`. It is possible to access the list of available models with another endpoint and method provided by OpenAI, `openai.Model.list()`. Note that not all available models are compatible with the `openai.ChatCompletion` endpoint. |
| `messages` | Array  | An array of `message` objects representing a conversation. A `message` object has two attributes: `role` (possible values are `system`, `user`, and `assistant`) and `content` (a string with the conversation message).                                                                                                                                                                                                                       |

A conversation starts with an optional system message, followed by alternating user and assistant messages:

The system message helps set the behavior of the assistant.

The user messages are the equivalent of a user typing a question or sentence in the ChatGPT web interface. They can be generated by the user of the application or set as an instruction.

The assistant messages have two roles: either they store prior responses to continue the conversation or they can be set as an instruction to give examples of desired behavior. Models do not have any memory of past requests, so storing prior messages is necessary to give context to the conversation and provide all relevant information.

#### Length of conversations and tokens

As seen previously, the total length of the conversation will be correlated to the total number of tokens. This will have an impact on the following:

Cost

The pricing is by token.

Timing

The more tokens there are, the more time the response will take—up to a couple of minutes.

The model working or not

The total number of tokens must be less than the model’s maximum limit. You can find examples of token limits in [“Considerations”](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch02.html#considerations).

As you can see, it is necessary to carefully manage the length of the conversation. You can control the number of input tokens by managing the length of your messages and control the number of output tokens via the `max_tokens` parameter, as detailed in the next subsection.

**TIP**

OpenAI provides a library named [_tiktoken_](https://oreil.ly/zxRIi) that allows developers to count how many tokens are in a text string. We highly recommend using this library to estimate costs before making the call to the endpoint.

#### Additional optional parameters

OpenAI provides several other options to fine-tune how you interact with the library. We will not detail all the parameters here, but we recommend having a look at [Table 2-2](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch02.html#table-2-2).

| Field name      | Type                                                  | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| --------------- | ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `functions`     | Array                                                 | An array of available functions. See [“From Text Completions to Functions”](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch02.html#from\_text\_completions\_to\_functions) for more details on how to use `functions`.                                                                                                                                                                                                                                                                                                                                                                     |
| `function_call` | String or object                                      | <p>Controls how the model responds:</p><ul><li><code>none</code> means the model must respond to the user in a standard way.</li><li><code>{"name":"my_function"}</code> means the model must give an answer that uses the specified function.</li><li><code>auto</code> means the model can choose between a standard response to the user or a function defined in the <code>functions</code> array.</li></ul>                                                                                                                                                                                                           |
| `temperature`   | Number (default: 1; accepted values: between 0 and 2) | A temperature of `0` means the call to the model will likely return the same completion for a given input. Even though the responses will be highly consistent, OpenAI does not guarantee a deterministic output. The higher the value is, the more random the completion will be. LLMs generate answers by predicting a series of tokens one at a time. Based on the input context, they assign probabilities to each potential token. When the temperature parameter is set to `0`, the LLM will always choose the token with the highest probability. A higher temperature allows for more varied and creative outputs. |
| `n`             | Integer (default: 1)                                  | With this parameter, it is possible to generate multiple chat completions for a given input message. However, with a temperature of `0` as the input parameter, you will get multiple responses, but they will all be identical or very similar.                                                                                                                                                                                                                                                                                                                                                                           |
| `stream`        | Boolean (default: false)                              | As its name suggests, this parameter will allow the answer to be in a stream format. This means partial messages will be sent gradually, like in the ChatGPT interface. This can make for a better user experience when the completions are long.                                                                                                                                                                                                                                                                                                                                                                          |
| `max_tokens`    | Integer                                               | This parameter signifies the maximum number of tokens to generate in the chat completion. This parameter is optional, but we highly recommend setting it as a good practice to keep your costs under control. Note that this parameter may be ignored or not respected if it is too high: the total length of the input and generated tokens is capped by the model’s token limitations.                                                                                                                                                                                                                                   |

You can find more details and other parameters on the [official documentation page](https://platform.openai.com/docs/api-reference/chat).

### Output Result Format for the Chat Completion Endpoint

Now that you have the information you need to query chat-based models, let’s see how to use the results.

Following is the complete response for the “Hello World” example:

```
{
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "Hello there! How may I assist you today?",
                "role": "assistant",
            },
        }
    ],
    "created": 1681134595,
    "id": "chatcmpl-73mC3tbOlMNHGci3gyy9nAxIP2vsU",
    "model": "gpt-3.5-turbo",
    "object": "chat.completion",
    "usage": {"completion_tokens": 10, "prompt_tokens": 11, "total_tokens": 21},
}
```

The generated output is detailed in [Table 2-3](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch02.html#table-2-3).

| Field name | Type                     | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ---------- | ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `choices`  | Array of “choice” object | <p>An array that contains the actual response of the model. By default, this array will only have one element, which can be changed with the parameter <code>n</code> (see <a href="https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch02.html#additional_optional_parameters">“Additional optional parameters”</a>). This element contains the following:</p><ul><li><code>finish_reason - string</code>: The reason the answer from the model is finished. In our “Hello World” example, we can see the <code>finish_reason</code> is <code>stop</code>, which means we received the complete response from the model. If there is an error during the output generation, it will appear in this field.</li><li><code>index - integer</code>: The index of the <code>choice</code> object from the <code>choices</code> array.</li><li><code>message - object</code>: Contains a <code>role</code> and either a <code>content</code> or a <code>function_call</code>. The <code>role</code> will always be <code>assistant</code>, and the <code>content</code> will include the text generated by the model. Usually we want to get this string: <code>response['choices'][0]​['mes⁠sage']['content']</code>. For details on how to use <code>function_call</code>, see <a href="https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch02.html#from_text_completions_to_functions">“From Text Completions to Functions”</a>.</li></ul> |
| `created`  | Timestamp                | The date in a timestamp format at the time of the generation. In our “Hello World” example, this timestamp translates to Monday, April 10, 2023 1:49:55 p.m.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `id`       | String                   | A technical identifier used internally by OpenAI.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `model`    | String                   | The model used. This is the same as the model set as input.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `object`   | String                   | Should always be `chat.completion` for GPT-4 and GPT-3.5 models, as we are using the chat completion endpoint.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `usage`    | String                   | Gives information on the number of tokens used in this query and therefore gives you pricing information. The `prompt_tokens` represents the number of tokens used in the input, the `completion_tokens` is the number of tokens in the output, and as you might have guessed, `total_tokens` = `prompt_tokens` + `completion_tokens`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |

**TIP**

If you want to have multiple choices and use an `n` parameter higher than 1, you will see that the `prompt_tokens` value will not change, but the `completion_tokens` value will be roughly multiplied by `n`.

### From Text Completions to Functions

OpenAI introduced the possibility for its models to output a JSON object containing arguments to call functions. The model will not be able to call the function itself, but rather will convert a text input into an output format that can be executed programmatically by the caller.

This is particularly useful when the result of the call to the OpenAI API needs to be processed by the rest of your code: instead of creating a complicated prompt to ensure that the model answers in a specific format that can be parsed by your code, you can use a function definition to convert natural language into API calls or database queries, extract structured data from text, and create chatbots that answer questions by calling external tools.

As you saw in [Table 2-2](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch02.html#table-2-2), which details the input options for the chat completion endpoint, function definitions need to be passed as an array of function objects. The function object is detailed in [Table 2-4](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch02.html#table-2-4).

| Field name    | Type              | Description                                                                                                                                |
| ------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `name`        | String (required) | The name of the function.                                                                                                                  |
| `description` | String            | The description of the function.                                                                                                           |
| `parameters`  | Object            | The parameters expected by the function. These parameters are expected to be described in a [JSON Schema](http://json-schema.org/) format. |

As an example, imagine that we have a database that contains information relative to company products. We can define a function that executes a search against this database:

```
# Example function
def find_product(sql_query):
    # Execute query here
    results = [
        {"name": "pen", "color": "blue", "price": 1.99},
        {"name": "pen", "color": "red", "price": 1.78},
    ]
    return results
```

Next, we define the specifications of the functions:

```
# Function definition
functions = [
    {
        "name": "find_product",
        "description": "Get a list of products from a sql query",
        "parameters": {
            "type": "object",
            "properties": {
                "sql_query": {
                    "type": "string",
                    "description": "A SQL query",
                }
            },
            "required": ["sql_query"],
        },
    }
]
```

We can then create a conversation and call the `openai.ChatCompletion` endpoint:

```
# Example question
user_question = "I need the top 2 products where the price is less than 2.00"
messages = [{"role": "user", "content": user_question}]
# Call the openai.ChatCompletion endpoint with the function definition
response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613", messages=messages, functions=functions
)
response_message = response["choices"][0]["message"]
messages.append(response_message)
```

The model has created a query that we can use. If we print the `function_call` object from the response, we get:

```
"function_call": {
        "name": "find_product",
        "arguments": '{\n  "sql_query": "SELECT * FROM products \
    WHERE price < 2.00 ORDER BY price ASC LIMIT 2"\n}',
    }
```

Next, we execute the function and continue the conversation with the result:

```
# Call the function
function_args = json.loads(
    response_message["function_call"]["arguments"]
)
products = find_product(function_args.get("sql_query"))
# Append the function's response to the messages
messages.append(
    {
        "role": "function",
        "name": function_name,
        "content": json.dumps(products),
    }
)
# Format the function's response into natural language
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=messages,
)
```

And finally, we extract the final response and obtain the following:

```
The top 2 products where the price is less than $2.00 are:
1. Pen (Blue) - Price: $1.99
2. Pen (Red) - Price: $1.78
```

This simple example demonstrates how functions can be useful to build a solution that allows end users to interact in natural language with a database. The function definitions allow you to constrain the model to answer exactly as you want it to, and integrate its response into an application.

## Using Other Text Completion Models

As mentioned, OpenAI provides several additional models besides the GPT-3 and GPT-3.5 series. These models use a different endpoint than the ChatGPT and GPT-4 models. At the time of this writing, this endpoint is compatible with `gpt-3.5-turbo-instruct`, `babbage-002`, `davinci-002` and other deprecated models.

**NOTE**

OpenAI has marked this endpoint as legacy.

There is an important difference between text completion and chat completion: as you might guess, both generate text, but chat completion is optimized for conversations. As you can see in the following code snippet, the main difference with the `openai.ChatCompletion` endpoint is the prompt format. Chat-based models must be in conversation format; for completion, it is a single prompt:

```
import openai
# Call the openai Completion endpoint
response = openai.Completion.create(
    model="text-davinci-003", prompt="Hello World!"
)
# Extract the response
print(response["choices"][0]["text"])
```

The preceding code snippet will output a completion similar to the following:

```
"\n\nIt's a pleasure to meet you. I'm new to the world"
```

The next section goes through the details of the text completion endpoint’s input options.

### Input Options for the Text Completion Endpoint

The set of input options for `openai.Completion.create` is very similar to what we saw previously with the chat endpoint. In this section, we will discuss the main input parameters and consider the impact of the length of the prompt.

#### Main input parameters

The required input parameters and a selection of optional parameters that we feel are most useful are described in [Table 2-5](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch02.html#table-2-5).

| Field name   | Type                                         | Description                                                                                                                                                                                                                                                                                                                                                 |
| ------------ | -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`      | String (required)                            | ID of the model to use (the same as with `openai.ChatCompletion`). This is the only required option.                                                                                                                                                                                                                                                        |
| `prompt`     | String or array (default: `<\|endoftext\|>`) | The prompt to generate completions for. This is the main difference from the `openai.ChatCompletion` endpoint. The `openai.Completion.create` endpoint should be encoded as a string, array of strings, array of tokens, or array of token arrays. If no prompt is provided to the model, it will generate text as if from the beginning of a new document. |
| `max_tokens` | Integer                                      | The maximum number of tokens to generate in the chat completion. The default value of this parameter is `16`, which may be too low for some use cases and should be adjusted according to your needs.                                                                                                                                                       |
| `suffix`     | String (default: null)                       | The text that comes after the completion. This parameter allows adding a suffix text. It also allows making insertions.                                                                                                                                                                                                                                     |

#### Length of prompts and tokens

Just as with the chat models, pricing will depend on the input you send and the output you receive. For the input message, you must carefully manage the length of the prompt parameter, as well as the suffix if one is used. For the output you receive, use `max_tokens.` It allows you to avoid unpleasant surprises.

#### Additional optional parameters

Also as with `openai.ChatCompletion`, additional optional parameters may be used to further tweak the behavior of the model. These parameters are the same as those used for `openai.ChatCompletion`, so we will not detail them again. Remember that you can control the output with the `temperature` or `n` parameter, control your costs with `max_tokens`, and use the `stream` option if you wish to have a better user experience with long completions.

### Output Result Format for the Text Completion Endpoint

Now that you have all the information needed to query text-based models, you will find that the results are very similar to the chat endpoint results. Here is an example output for our “Hello World” example with the `davinci` model:

```
{
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "logprobs": null,
            "text": "<br />\n\nHi there! It's great to see you.",
        }
    ],
    "created": 1681883111,
    "id": "cmpl-76uutuZiSxOyzaFboxBnaatGINMLT",
    "model": "text-davinci-003",
    "object": "text_completion",
    "usage": {"completion_tokens": 15, "prompt_tokens": 3, "total_tokens": 18},
}
```

**NOTE**

This output is very similar to what we got with the chat models. The only difference is in the `choice` object: instead of having a message with `content` and `role` attributes, we have a simple `text` attribute containing the completion generated by the model.

## Considerations

You should consider two important things before using the APIs extensively: cost and data privacy.

### Pricing and Token Limitations

OpenAI keeps the pricing of its models listed on its [pricing page](https://openai.com/pricing). Note that OpenAI is not bound to maintain this pricing, and the costs may change over time.

At the time of this writing, the pricing is as shown in [Table 2-6](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch02.html#table-2-6) for the OpenAI models used most often.

| Family          | Model               | Pricing                                                                           | Max tokens |
| --------------- | ------------------- | --------------------------------------------------------------------------------- | ---------- |
| Chat            | `gpt-4`             | <p>Prompt: $0.03 per 1,000 tokens</p><p>Completion: $0.06 per 1,000 tokens</p>    | 8,192      |
| Chat            | `gpt-4-32k`         | <p>Prompt: $0.06 per 1,000 tokens</p><p>Completion: $0.012 per 1,000 tokens</p>   | 32,768     |
| Chat            | `gpt-3.5-turbo`     | <p>Prompt: $0.0015 per 1,000 tokens</p><p>Completion: $0.002 per 1,000 tokens</p> | 4,096      |
| Chat            | `gpt-3.5-turbo-16k` | <p>Prompt: $0.003 per 1,000 tokens</p><p>Completion: $0.004 per 1,000 tokens</p>  | 16,384     |
| Text completion | `text-davinci-003`  | $0.02 per 1,000 tokens                                                            | 4,097      |

There are several things to note from [Table 2-6](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch02.html#table-2-6):

The `davinci` model is more than 10 times the cost of the GPT-3.5 Turbo 4,000-context model. Since `gpt-3.5-turbo` can also be used for single-turn completion tasks and since both models are nearly equal in accuracy for this type of task, it is recommended to use GPT-3.5 Turbo (unless you need special features such as insertion, via the parameter suffix, or if `text-davinci-003` outperforms `gpt-3.5-turbo` for your specific task).

GPT-3.5 Turbo is less expensive than GPT-4. The differences between GPT-4 and GPT-3.5 are irrelevant for many basic tasks. However, in complex inference situations, GPT-4 far outperforms any previous model.

The chat models have a different pricing system than the `davinci` models: they differentiate input (prompt) and output (completion).

GPT-4 allows a context twice as long as GPT-3.5 Turbo, and can even go up to 32,000 tokens, which is equivalent to more than 25,000 words of text. GPT-4 enables use cases such as long-form content creation, advanced conversation, and document search and analysis… for a cost.

### Security and Privacy: Caution!

As we write this, OpenAI claims the data sent as input to the models will not be used for retraining unless you decide to opt in. However, your inputs are retained for 30 days for monitoring and usage compliance-checking purposes. This means OpenAI employees as well as specialized third-party contractors may have access to your API data.

**WARNING**

Never send sensitive data such as personal information or passwords through the OpenAI endpoints. We recommend that you check [OpenAI’s data usage policy](https://openai.com/policies/api-data-usage-policies) for the latest information, as this can be subject to change. If you are an international user, be aware that your personal information and the data you send as input can be transferred from your location to the OpenAI facilities and servers in the United States. This may have some legal impact on your application creation.

More details on how to build LLM-powered applications while taking into account security and privacy issues can be found in [Chapter 3](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch03.html#building\_apps\_with\_gpt\_4\_and\_chatgpt).

## Other OpenAI APIs and Functionalities

Your OpenAI account gives you access to functionalities besides text completion. We selected several of these functionalities to explore in this section, but if you want a deep dive into all the API possibilities, look at [OpenAI’s API reference page](https://platform.openai.com/docs/api-reference).

### Embeddings

Since a model relies on mathematical functions, it needs numerical input to process information. However, many elements, such as words and tokens, aren’t inherently numerical. To overcome this, _embeddings_ convert these concepts into numerical vectors. Embeddings allow computers to process the relationships between these concepts more efficiently by representing them numerically. In some situations, it can be useful to have access to embeddings, and OpenAI provides a model that can transform a text into a vector of numbers. The embeddings endpoint allows developers to obtain a vector representation of an input text. This vector representation can then be used as input to other ML models and NLP algorithms.

At the time of this writing, OpenAI recommends using its latest model, `text-embedding-ada-002`, for nearly all use cases. It is very simple to use:

```
result = openai.Embedding.create(
    model="text-embedding-ada-002", input="your text"
)
```

The embedding is accessed with:

```
result['data']['embedding']
```

The resulting embedding is a vector: an array of floats.

**TIP**

The complete documentation on embeddings is available in [OpenAI’s reference documents](https://platform.openai.com/docs/api-reference/embeddings).

The principle of embeddings is to represent text strings meaningfully in some space that captures their semantic similarity. With this idea, you can have various use cases:

Search

Sort results by relevance to the query string.

Recommendations

Recommend articles that contain text strings related to the query string.

Clustering

Group strings by similarity.

Anomaly detection

Find a text string that is not related to the other strings.

## HOW EMBEDDINGS TRANSLATE LANGUAGE FOR MACHINE LEARNING

In the world of ML, especially when dealing with language models, we encounter an important concept called _embeddings_. Embeddings transform categorical data—such as tokens, typically single words or groups of these tokens that form sentences—into a numerical format, specifically vectors of real numbers. This transformation is essential because ML models rely on numerical data and aren’t ideally equipped to process categorical data directly.

To visualize this, think of embeddings as a sophisticated language interpreter that translates the rich world of words and sentences into the universal language of numbers that ML models understand fluently. A truly remarkable feature of embeddings is their ability to preserve _semantic similarity_, meaning that words or phrases with similar meanings tend to be mapped closer together in numerical space.

This property is fundamental in a process called _information retrieval_, which involves extracting relevant information from a large dataset. Given the way embeddings inherently capture similarities, they are an excellent tool for such operations.

Modern LLMs make extensive use of embeddings. Typically, these models deal with embeddings of about 512 dimensions, providing a high-dimension numerical representation of the language data. The depth of these dimensions allows the models to distinguish a wide range of complex patterns. As a result, they perform remarkably well in various language tasks, ranging from translation and summarization to generating text responses that convincingly resemble human discourse.

Embeddings have the property that if two texts have a similar meaning, their vector representation will be similar. As an example, in [Figure 2-8](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch02.html#fig\_8\_example\_of\_two\_dimensional\_embedding\_of\_three\_sent), three sentences are shown in two-dimensional embeddings. Although the two sentences “The cat chased the mouse around the house.” and “Around the house, the mouse was pursued by the cat.” have different syntaxes, they convey the same general meaning, and therefore they should have similar embedding representations. As the sentence “The astronaut repaired the spaceship in orbit.” is unrelated to the topic of the previous sentences (cats and mice) and discusses an entirely different subject (astronauts and spaceships), it should have a significantly different embedding representation. Note that in this example, for clarity we show the embedding as having two dimensions, but in reality, they are often in a much higher dimension, such as 512.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098152475/files/assets/dagc_0208.png" alt="" height="396" width="600"><figcaption></figcaption></figure>

**Figure 2-8. Example of two-dimensional embedding of three sentences**

We refer to the embeddings API several times in the remaining chapters, as embeddings are an essential part of processing natural language with AI models.

### Moderation Model

As mentioned earlier, when using the OpenAI models you must respect the rules described in the [OpenAI usage policies](https://openai.com/policies/usage-policies). To help you respect these rules, OpenAI provides a model to check whether the content complies with these usage policies. This can be useful if you build an app in which user input will be used as a prompt: you can filter the queries based on the moderation endpoint results. The model provides classification capabilities that allow you to search for content in the following categories:

Hate

Promoting hatred against groups based on race, gender, ethnicity, religion, nationality, sexual orientation, disability, or caste

Hate/threatening

Hateful content that involves violence or severe harm to targeted groups

Self-harm

Content that promotes or depicts acts of self-harm, including suicide, cutting, and eating disorders

Sexual

Content designed to describe a sexual activity or promote sexual services (except for education and wellness)

Sexual with minors

Sexually explicit content involving persons under 18 years of age

Violence

Content that glorifies violence or celebrates the suffering or humiliation of others

Violence/graphic

Violent content depicting death, violence, or serious bodily injury in graphic detail

**NOTE**

Support for languages other than English is limited.

The endpoint for the moderation model is `openai.Moderation.create`, and only two parameters are available: the model and the input text. There are two models of content moderation. The default is `text-moderation-latest`, which is automatically updated over time to ensure that you always use the most accurate model. The other model is `text-moderation-stable`. OpenAI will notify you before updating this model.

**WARNING**

The accuracy of `text-moderation-stable` may be slightly lower than `text-moderation-latest`.

Here is an example of how to use this moderation model:

```
import openai
# Call the openai Moderation endpoint, with the text-moderation-latest model
response = openai.Moderation.create(
    model="text-moderation-latest",
    input="I want to kill my neighbor.",
)
```

Let’s take a look at the output result of the moderation endpoint contained in the `response` object:

```
{
    "id": "modr-7AftIJg7L5jqGIsbc7NutObH4j0Ig",
    "model": "text-moderation-004",
    "results": [
        {
            "categories": {
                "hate": false,
                "hate/threatening": false,
                "self-harm": false,
                "sexual": false,
                "sexual/minors": false,
                "violence": true,
                "violence/graphic": false,
            },
            "category_scores": {
                "hate": 0.0400671623647213,
                "hate/threatening": 3.671687863970874e-06,
                "self-harm": 1.3143378509994363e-06,
                "sexual": 5.508050548996835e-07,
                "sexual/minors": 1.1862029225540027e-07,
                "violence": 0.9461417198181152,
                "violence/graphic": 1.463699845771771e-06,
            },
            "flagged": true,
        }
    ],
}
```

The output result of the moderation endpoint provides the pieces of information shown in [Table 2-7](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch02.html#table-2-7).

| Field name        | Type    | Description                                                                                                                                                                                                                                                                                                                                                                       |
| ----------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`           | String  | This is the model used for the prediction. When calling the method in our earlier example, we specified the use of the model `text-moderation-latest`, and in the output result, the model used is `text-moderation-004`. If we had called the method with `text-moderation-stable`, then `text-moderation-001` would have been used.                                             |
| `flagged`         | Boolean | If the model identifies the content as violating OpenAI’s usage policies, set this to `true`; otherwise, set it to `false`.                                                                                                                                                                                                                                                       |
| `categories`      | Dict    | This includes a dictionary with binary flags for policy violation categories. For each category, the value is `true` if the model identifies a violation and `false` if not. The dictionary can be accessed via `print(type(response['results'][0]​['cate⁠gories']))`.                                                                                                            |
| `category_scores` | Dict    | The model provides a dictionary with category-specific scores that show how confident it is that the input goes against OpenAI’s policy for that category. Scores range from 0 to 1, with higher scores meaning more confidence. These scores should not be seen as probabilities. The dictionary can be accessed via `print(type(response​['re⁠sults'][0]['category_scores']))`. |

**WARNING**

OpenAI will regularly improve the moderation system. As a result, the `category_scores` may vary, and the threshold set to determine the category value from a category score may also change.

### Whisper and DALL-E

OpenAI also provides other AI tools that are not LLMs but can easily be used in combination with GPT models in some use cases. We don’t explain them here because they are not the focus of this book. But don’t worry, using their APIs is very similar to using OpenAI’s LLM APIs.

Whisper is a versatile model for speech recognition. It is trained on a large audio dataset and is also a multitasking model that can perform multilingual speech recognition, speech translation, and language identification. An open source version is available on the [Whisper project’s GitHub page](https://github.com/openai/whisper) of OpenAI.

In January 2021, OpenAI introduced DALL-E, an AI system capable of creating realistic images and artwork from natural language descriptions. DALL-E 2 takes the technology further with higher resolution, greater input text comprehension, and new capabilities. Both versions of DALL-E were created by training a transformer model on images and their text descriptions. You can try DALL-E 2 through the API and via the [Labs interface](https://labs.openai.com/).

## Summary (and Cheat Sheet)

As we have seen, OpenAI provides its models as a service, through an API. In this book, we chose to use the Python library provided by OpenAI, which is a simple wrapper around the API. With this library, we can interact with the GPT-4 and ChatGPT models: the first step to building LLM-powered applications! However, using these models implies several considerations: API key management, pricing, and privacy.

Before starting, we recommend looking at the OpenAI usage policies, and playing with the Playground to get familiar with the different models without the hassle of coding. Remember: GPT-3.5 Turbo, the model behind ChatGPT, is the best choice for most use cases.

Following is a cheat sheet to use when sending input to GPT-3.5 Turbo:

1.  Install the `openai` dependency:

    ```
    pip install openai
    ```
2.  Set your API key as an environment variable:

    ```
    export OPENAI_API_KEY=sk-(...)
    ```
3.  In Python, import `openai`:

    ```
    import openai
    ```
4.  Call the `openai.ChatCompletion` endpoint:

    ```
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Your Input Here"}],
    )
    ```
5.  Get the answer:

    ```
        print(response['choices'][0]['message']['content'])
    ```

**TIP**

Don’t forget to check the [pricing page](https://openai.com/pricing), and use [tiktoken](https://github.com/openai/tiktoken) to estimate the usage costs.

Note that you should never send sensitive data, such as personal information or passwords, through the OpenAI endpoints.

OpenAI also provides several other models and tools. You will find in the next chapters that the embeddings endpoint is very useful for including NLP features in your application.

Now that you know _how_ to use the OpenAI services, it’s time to dive into _why_ you should use them. In the next chapter, you’ll see an overview of various examples and use cases to help you make the most out of the OpenAI ChatGPT and GPT-4 models.
