# Chapter 4. Advanced GPT-4 and ChatGPT Techniques

## Chapter 4. Advanced GPT-4 and ChatGPT Techniques

Now that you are familiar with the basics of LLMs and the OpenAI API, it’s time to take your skills to the next level. This chapter covers powerful strategies that will enable you to harness the true potential of ChatGPT and GPT-4. From prompt engineering, zero-shot learning, and few-shot learning to fine-tuning models for specific tasks, this chapter will give you all the knowledge you need to create any application you can imagine.

## Prompt Engineering

Before we dive into prompt engineering, let’s briefly review the chat model’s `completion` function, as this section will use it extensively. To make the code more compact, we define the function as follows:

```
def chat_completion(prompt, model="gpt-4", temperature=0):
    res = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    print(res["choices"][0]["message"]["content"])
```

This function receives a prompt and displays the completion result in the terminal. The model and the temperature are two optional features set by default, respectively, to GPT-4 and 0.

To demonstrate prompt engineering, we will return to the example text “As Descartes said, I think therefore”. If this input is passed to GPT-4, it is natural for the model to complete the sentence by iteratively adding the most likely tokens:

```
chat_completion("As Descartes said, I think therefore")
```

As a result, we get the following output message:

```
I am. This famous philosophical statement, also known as "Cogito, ergo sum," 
emphasizes the existence of the self through the act of thinking or doubting.
Descartes used this statement as a foundational principle in his philosophy,
arguing that one's own existence is the most certain and indubitable fact
that can be known.
```

_Prompt engineering_ is an emergent discipline focused on developing best practices for building optimal inputs for LLMs in order to produce desirable outputs as programmatically as possible. As an AI engineer, you must know how to interact with AI to obtain exploitable results for your apps, how to ask the right questions, and how to write quality prompts; all topics we will cover in this section.

It should be noted that prompt engineering can affect the cost of using the OpenAI API. The amount of money you will pay to use the API is proportional to the number of tokens you send to and receive from OpenAI. As mentioned in [Chapter 2](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch02.html#a\_deep\_dive\_into\_the\_gpt\_4\_and\_chatgpt\_apis), use of the `max_token` parameter is highly recommended to avoid unpleasant surprises on your bills.

Also note that you should consider the different parameters you can use in the `openai` methods, as you can get significantly different results with the same prompt if you use parameters like `temperature`, `top_p`, and `max_token`.

### Designing Effective Prompts

A lot of tasks can be performed via prompts. They include summarization, text classification, sentiment analysis, and question answering. In all these tasks, it is common to define three elements in the prompt: a role, a context, and a task, as depicted in [Figure 4-1](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch04.html#fig\_1\_an\_effective\_prompt).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098152475/files/assets/dagc_0401.png" alt="" height="175" width="600"><figcaption></figcaption></figure>

**Figure 4-1. An effective prompt**

All three elements are not always necessary, and their order can be changed, but if your prompt is well constructed and the elements are well defined, you should get good results. Note that even when these three elements are used, for complex tasks you may need to use more advanced techniques, such as zero-shot learning, few-shot learning, and fine-tuning. These advanced techniques will be discussed later in this chapter.

#### The context

The first essential element in a prompt is the context. When you write an input text for an LLM, you must detail the context as much as possible. To illustrate this, say you want to use GPT-4 to create an application that suggests the main course for lunch. In the following, we will compare the results we obtain with two different contexts. The first context will have little detail, and the second will have more detail.

With a short input message such as:

```
chat_completion("Give me a suggestion for the main course for today's lunch.")
```

we get the following output message:

```
Grilled chicken with roasted vegetables and quinoa.
```

Now, with more details about the context in the input message:

```
prompt = """
Context: I do 2 hours of sport a day. I am vegetarian, and I don't like green 
vegetables. I am conscientious about eating healthily. 
Task: Give me a suggestion for a main course for today's lunch."""
chat_completion(prompt)
```

we get the following output message:

```
For today's lunch, you can try a Chickpea and Sweet Potato Curry served with
Quinoa. This dish is packed with protein, fiber, and essential nutrients,
while also being vegetarian and not relying on green vegetables. The curry
can be made with chickpeas, sweet potatoes, tomatoes, onions, garlic, and a
blend of spices like turmeric, cumin, and coriander. Cook the quinoa
separately and serve it alongside the curry for a delicious and healthy meal.
```

In the second example, the sentence is more detailed because the model has more context: it suggests a healthy vegetarian dish that is full of protein.

The context guides GPT’s “thinking” to add value to its response. Building a prompt with a good context is an iterative process, often requiring trial-and-error steps. Identifying the essential points the context must contain is sometimes tricky. To help determine possible improvements, you can ask GPT-4 for suggestions. In the following example, we change the prompt: instead of asking the model to answer our question, we instruct it to ask questions about the context in an effort to achieve a better completion.

With the following input message:

```
prompt = """
Context: I do 2 hours of sport a day. I am vegetarian and I don't like green 
vegetables. I am very careful to eat healthily.
Task: Give me a suggestion for a main course for today's lunch?
Do not perform the requested task! Instead, can you ask me questions about the 
context so that when I answer, you can perform the requested task more
efficiently?
"""
chat_completion(prompt)
```

we get the following questions from the GPT-4 model:

```
Sure, I have a few questions to help me suggest a suitable main course for you:
1. What type of sport do you do, and do you have any specific nutritional
  requirements related to it?
2. Are there any other vegetables or ingredients that you particularly enjoy
  or dislike?
3. Do you prefer a hot or cold meal for lunch?
4. Are you open to trying new recipes or do you prefer sticking to
  familiar dishes?
```

The questions asked by the model are often relevant. Based on these questions, you can update your context in your prompt. Therefore, we advise you to use this tip often when building your prompt.

#### The task

The task defines your use of the GPT-4 model and should be well defined and specific. You should provide sufficient information for the task and use appropriate phrases in the prompt to guide the model to the desired outcome.

In the preceding section, the task was for the model to suggest a main course for lunch. We can add details to that task to clarify what we want. Here, we ask the model to add a table with the ingredients and required quantities of each in grams:

```
prompt = """
Context: I do 2 hours of sport a day. I am vegetarian, and I don't like green 
vegetables. I am conscientious about eating healthily.
Task: Give me a suggestion for a main course for today's lunch.
With this suggestion, I also want a table with two columns where each row 
contains an ingredient from the main course.
The first column in the table is the name of the ingredient.
The second column of the table is the number of grams of that ingredient needed 
for one person. Do not give the recipe for preparing the main course.
"""
chat_completion(prompt)
```

The model gives us the following output:

```
Suggestion for today's lunch: Stuffed Bell Peppers with Quinoa and Black Beans
| Ingredient          | Grams per person |
|---------------------|------------------|
| Bell peppers        | 200g             |
| Quinoa              | 60g              |
| Black beans         | 75g              |
| Onion               | 50g              |
| Garlic              | 5g               |
| Olive oil           | 10g              |
| Tomato sauce        | 50g              |
| Cumin               | 1g               |
| Chili powder        | 1g               |
| Salt                | 1g               |
| Pepper              | 1g               |
| Shredded cheese     | 30g              |
```

On the [OpenAI API example page](https://platform.openai.com/examples), there is a list with 48 examples of tasks that GPT models can perform, always with the associated prompt and a demo. While these examples use the GPT-3 models and the completion endpoint, the principle would be the same for the chat endpoint, and the examples nicely illustrate how to give a task to OpenAI models. We won’t go through all of them here, but here are a few of them:

Grammar correction

Corrects sentences to standard English.

Prompt:

```
Correct this to standard English: She no went to the market.
```

Summarize for a second-grader

Translates complex text into more straightforward concepts.

Prompt:

```
Summarize this for a second-grade student: Jupiter is the fifth planet [...]
```

TL;DR summarization

TL;DR stands for “too long; didn’t read.” It has been observed that a text can be summarized by simply adding `Tl;dr` at the end.

Prompt:

```
A neutron star [...] atomic nuclei. Tl;dr
```

Python to natural language

Explain a piece of Python code in a language people can understand.

Prompt:

```
# Python 3 
def hello(x): 
print('hello '+str(x)) 
# Explanation of what the code does
```

Calculate time complexity

Find the time complexity of a function.

Prompt:

```
# Python 3 
def hello(x, n):
     for i in range(n):
        print('hello '+str(x))
# The time complexity of this function is 
```

Python bug fixer

Fixes code containing a bug.

Prompt:

```
### Buggy Python
def hello(x, n):
     for i in rang(n):
        print('hello '+str(x))
### Fixed Python
```

SQL request

Simple SQL query building.

Prompt:

```
Create a SQL request to find all users who live in California and have
over 1000 credits.
```

Analogy maker

Can make an analogy between two words.

Prompt:

```
Create an analogy for this phrase. Questions are arrows in that:
```

Summary notes

Summarize notes from a meeting.

Prompt:

```
Convert my shorthand into a first-hand account of the meeting:
Tom: Profits up 50%
Jane: New servers are online
Kjel: Need more time to fix software
```

#### The role

A final way to influence the model while writing the prompt is to give it a role. Of course, the role and the context can be used independently, but using both increases your control over the model’s output. In the following example, we put the model in the role of a sports nutrition expert:

```
prompt = """
Role: You are a nutritionist designing healthy diets for high-performance 
athletes. You take into account the nutrition needed for a good recovery.
Context: I do 2 hours of sport a day. I am vegetarian, and I don't like green 
vegetables. I am conscientious about eating healthily.
Task: Based on your expertise defined in your role, give me a suggestion for a 
main course for today's lunch. With this suggestion, I also want a table with
two columns where each row in the table contains an ingredient from the main 
course. The first column in the table is the name of the ingredient.
The second column of the table is the number of grams of that ingredient needed
for one person. Do not give the recipe for preparing the main course.
"""
chat_completion(prompt)
```

And we get the following result:

```
Main Course Suggestion: Quinoa and Chickpea Stuffed Bell Peppers
Ingredient Table:
| Ingredient          | Grams per Person |
|---------------------|------------------|
| Quinoa              | 60               |
| Chickpeas           | 100              |
| Bell Peppers        | 200              |
| Olive Oil           | 10               |
| Onion               | 50               |
| Garlic              | 5                |
| Tomato              | 100              |
| Feta Cheese         | 30               |
| Kalamata Olives     | 20               |
| Fresh Basil         | 5                |
| Salt                | 2                |
| Black Pepper        | 1                |
```

As you’ve seen, prompts can be used to condition the probabilistic distribution set of LLMs such as GPT models. They can be seen as a guide that directs the model to produce a particular type of result. While there is no definitive structure for prompt design, a useful framework to consider is the combination of context, role, and task.

It’s important to understand that this is just one approach, and prompts can be created without explicitly defining these elements. Some prompts may benefit from a different structure or require a more creative approach based on the specific needs of your application. Therefore, this context-role-task framework should not limit your thinking, but rather be a tool to help you effectively design your prompts when appropriate.

### Thinking Step by Step

As we know, GPT-4 is not good for computation. It cannot compute 369 × 1,235:

```
prompt = "How much is 369 * 1235?"
chat_completion(prompt)
```

We get the following answer: `454965`

The correct answer is 455,715. Does GPT-4 not solve complex mathematical problems? Remember that the model formulates this answer by predicting each token in the answer sequentially, starting from the left. This means that GPT-4 generates the leftmost digit first, then uses that as part of the context to generate the next digit, and so on, until the complete answer is formed. The challenge here is that each number is predicted independent of the final correct value. GPT-4 considers numbers like tokens; there is no mathematical logic.

**NOTE**

In [Chapter 5](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch05.html#advancing\_llm\_capabilities\_with\_the\_langchain\_fram), we’ll explore how OpenAI has enriched GPT-4 with plug-ins. An example is a calculator plug-in for providing accurate mathematical solutions.

There is a trick to increasing the reasoning capacity of language models. For example, when asked to solve 369 × 1235, we can see that the model tries to answer directly in one shot. Consider that you probably won’t be able to solve this multiplication either without the help of a pencil and a sheet of paper to do the calculations. It is possible to encourage the model to make intermediate reasonings via the prompt. And like you with your pencil and paper, the model can solve more complex problems if you give it time to reason.

Adding “Let’s think step by step” at the end of the prompt has been empirically proven to enable the model to solve more complicated reasoning problems. This technique, called the _zero-shot-CoT strategy_, was introduced in the scientific paper [“Large Language Models Are Zero-Shot Reasoners”](https://oreil.ly/2tHHy) by Kojima et al., published in 2022.

_CoT_ stands for _chain of thought_; it refers here to the technique of using prompts that encourage the model to imitate step-by-step reasoning. The term _zero-shot_ means the model does not rely on task-specific examples to perform this reasoning; it is ready to handle new tasks based on its general training. Unlike techniques such as few-shot learning (which we will examine shortly) that require the model to have seen examples of the task, zero-shot learning tries to generalize without needing task-specific examples.

As we will see, with this sentence added to the prompt, the model begins to reason by breaking the problem into subproblems. As a result, it can take time to reason, allowing it to find solutions to problems it could not solve before in only one shot. Let’s see what happens if we change our input prompt:

```
prompt = "How much is 369 * 1235 ? Let's think step by step."
chat_completion(prompt)
```

We get the following output:

```
Step 1: Multiply 369 by 5 (the last digit of 1235).
369 * 5 = 1845
Step 2: Multiply 369 by 30 (the second last digit of 1235 is 3, so multiply by
   3 and add a 0 at the end).
369 * 30 = 11070
Step 3: Multiply 369 by 200 (the third digit of 1235 is 2, so multiply by 2 and
   add two 0s at the end).
369 * 200 = 73800
Step 4: Multiply 369 by 1000 (the first digit of 1235 is 1, so multiply by 1 and 
   add three 0s at the end).
369 * 1000 = 369000
Step 5: Add the results from steps 1 to 4.
1845 + 11070 + 73800 + 369000 = 455715
So, 369 * 1235 = 455,715.
```

There has been a remarkable shift in the model’s approach. It now breaks down the primary problem into smaller, more manageable steps rather than trying to tackle the problem head-on.

**WARNING**

Despite prompting the model to “think step by step,” it is still crucial that you carefully evaluate its responses, as GPT-4 is not infallible. For a more complex computation such as 3,695 × 123,548, even with this trick the LLM is not able to find the correct solution.

Of course, it’s hard to tell from one example whether this trick generally works or whether we just got lucky. On benchmarks with various math problems, empirical experiments have shown that this trick significantly increased the accuracy of GPT models. Although the trick works well for most math problems, it is not practical for all situations. The authors of “Large Language Models are Zero-Shot Reasoners” found it to be most beneficial for multistep arithmetic problems, problems involving symbolic reasoning, problems involving strategy, and other issues involving reasoning. It was not found to be useful for commonsense problems.

### Implementing Few-Shot Learning

_Few-shot learning_, introduced in [“Language Models Are Few-Shot Learners”](https://oreil.ly/eSoRo) by Brown et al., refers to the ability of the LLM to generalize and produce valuable results with only a few examples in the prompt. With few-shot learning, you give a few examples of the task you want the model to perform, as illustrated in [Figure 4-2](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch04.html#fig\_2\_a\_prompt\_containing\_a\_few\_examples). These examples guide the model to process the desired output format.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098152475/files/assets/dagc_0402.png" alt="" height="158" width="600"><figcaption></figcaption></figure>

**Figure 4-2. A prompt containing a few examples**

In this example, we ask the LLM to convert specific words into emojis. It is difficult to imagine the instructions to put in a prompt to do this task. But with few-shot learning, it’s easy. Give it examples, and the model will automatically try to reproduce them:

```
prompt = """
I go home -->  go 
my dog is sad --> my  is 
I run fast -->  run 
I love my wife -->   my wife
the girl plays with the ball --> the   with the 
The boy writes a letter to a girl --> 
"""
chat_completion(prompt)
```

From the preceding example, we get the following message as output:

```
The   a  to a 
```

The few-shot learning technique gives examples of inputs with the desired outputs. Then, in the last line, we provide the prompt for which we want a completion. This prompt is in the same form as the earlier examples. Naturally, the language model will perform a completion operation considering the pattern of the examples given.

We can see that with only a few examples, the model can reproduce the instructions. By leveraging the extensive knowledge that LLMs have acquired in their training phase, they can quickly adapt and generate accurate answers based on only a few examples.

**NOTE**

Few-shot learning is a powerful aspect of LLMs because it allows them to be highly flexible and adaptable, requiring only a limited amount of additional information to perform various tasks.

When you provide examples in the prompt, it is essential to ensure that the context is clear and relevant. Clear examples improve the model’s ability to match the desired output format and execute the problem-solving process. Conversely, inadequate or ambiguous examples can lead to unexpected or incorrect results. Therefore, writing examples carefully and ensuring that they convey the correct information can significantly impact the model’s ability to perform the task accurately.

Another approach to guiding LLMs is _one-shot learning_. As its name indicates, in this case you provide only one example to help the model execute the task. Although this approach provides less guidance than few-shot learning, it can be effective for more straightforward tasks or when the LLM already has substantial background knowledge about the topic. The advantages of one-shot learning are simplicity, faster prompt generation, and lower computational cost and thus lower API costs. However, for complex tasks or situations that require a deeper understanding of the desired outcome, few-shot learning might be a more suitable approach to ensure accurate results.

**TIP**

Prompt engineering has become a trending topic, and you will find many online resources to delve deeper into the subject. As an example, this [GitHub repository](https://github.com/f/awesome-chatgpt-prompts) contains a list of effective prompts that were contributed by more than 70 different users.

While this section explored various prompt engineering techniques that you can use individually, note that you can combine the techniques to obtain even better results. As a developer, it is your job to find the most effective prompt for your specific problem. Remember that prompt engineering is an iterative process of trial-and-error experimentation.

### Improving Prompt Effectiveness

We have seen several prompt engineering techniques that allow us to influence the behavior of the GPT models to get better results that meet our needs. We’ll end this section with a few more tips and tricks you can use in different situations when writing prompts for GPT models.

#### Instruct the model to ask more questions

Ending prompts by asking the model if it understood the question and instructing the model to ask more questions is an effective technique if you are building a chatbot-based solution. You can add a text like this to the end of your prompts:

```
Did you understand my request clearly? If you do not fully understand my request,
ask me questions about the context so that when I answer, you can
perform the requested task more efficiently.
```

#### Format the output

Sometimes you’ll want to use the LLM output in a longer process: in such cases, the output format matters. For example, if you want a JSON output, the model tends to write in the output before and after the JSON block. If you add in the prompt `the output must be accepted by json.loads` then it tends to work better. This type of trick can be used in many situations.

For example, with this script:

```
prompt = """
Give a JSON output with 5 names of animals. The output must be accepted 
by json.loads.
"""
chat_completion(prompt, model='gpt-4')
```

we get the following JSON block of code:

```
{
  "animals": [
    "lion",
    "tiger",
    "elephant",
    "giraffe",
    "zebra"
  ]
}
```

#### Repeat the instructions

It has been found empirically that repeating instructions gives good results, especially when the prompt is long. The idea is to add to the prompt the same instruction several times, but formulated differently each time.

This can also be done with negative prompts.

#### Use negative prompts

Negative prompts in the context of text generation are a way to guide the model by specifying what you don’t want to see in the output. They act as constraints or guidelines to filter out certain types of responses. This technique is particularly useful when the task is complicated: models tend to follow instructions more precisely when the tasks are repeated several times in different ways.

Continuing with the previous example, we can insist on the output format with negative prompting by adding `Do not add anything before or after the json text.`.

In [Chapter 3](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch03.html#building\_apps\_with\_gpt\_4\_and\_chatgpt), we used negative prompting in the third project:

```
Extract the keywords from the following question: {user_question}. Do not answer
anything else, only the keywords.
```

Without this addition to the prompt, the model tended to not follow the instructions.

#### Add length constraints

A length constraint is often a good idea: if you expect only a single-word answer or 10 sentences, add it to your prompt. This is what we did in [Chapter 3](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch03.html#building\_apps\_with\_gpt\_4\_and\_chatgpt) in the first project: we specified `LENGTH: 100 words` to generate an adequate news article. In the fourth project, our prompt also had a length instruction: `If you can answer the question: ANSWER, if you need more information: MORE, if you can not answer: OTHER. Only answer one word.`. Without that last sentence, the model would tend to formulate sentences rather than follow the instructions.

## Fine-Tuning

OpenAI provides many ready-to-use GPT models. Although these models excel at a broad array of tasks, fine-tuning them for specific tasks or contexts can further enhance their performance.

### Getting Started

Let’s imagine that you want to create an email response generator for your company. As your company works in a specific industry with a particular vocabulary, you want the generated email responses to retain your current writing style. There are two strategies for doing this: either you can use the prompt engineering techniques introduced earlier to force the model to output the text you want, or you can fine-tune an existing model. This section explores the second technique.

For this example, you must collect a large number of emails containing data about your particular business domain, inquiries from customers, and responses to those inquiries. You can then use this data to fine-tune an existing model to learn your company’s specific language patterns and vocabulary. The fine-tuned model is essentially a new model built from one of the original models provided by OpenAI, in which the internal weights of the model are adjusted to fit your specific problem so that the new model increases its accuracy on tasks similar to the examples it saw in the dataset provided for the fine-tuning. By fine-tuning an existing LLM, it is possible to create a highly customized and specialized email response generator tailored explicitly to the language patterns and words used in your particular business.

[Figure 4-3](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch04.html#fig\_3\_the\_fine\_tuning\_process) illustrates the fine-tuning process in which a dataset from a specific domain is used to update the internal weights of an existing GPT model. The objective is for the new fine-tuned model to make better predictions in the particular domain than the original GPT model. It should be emphasized that this is a _new model_. This new model is on the OpenAI servers: as before, you must use the OpenAI APIs to use it, as it cannot be accessed locally.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098152475/files/assets/dagc_0403.png" alt="" height="532" width="600"><figcaption></figcaption></figure>

**Figure 4-3. The fine-tuning process**

**NOTE**

Even after you have fine-tuned an LLM with your own specific data, the new model remains on OpenAI’s servers. You’ll interact with it through OpenAI’s APIs, not locally.

#### Adapting GPT base models for domain-specific needs

`gpt-3.5-turbo` and  `gpt-4` can be fine-tuned. In the following sections, you will see step-by-step how to proceed.&#x20;

#### Fine-tuning versus few-shot learning

Fine-tuning is a process of _retraining_ an existing model on a set of data from a specific task to improve its performance and make its answers more accurate. In fine-tuning, you update the internal parameters of the model. As we saw before, few-shot learning provides the model with a limited number of good examples through its input prompt, which guides the model to produce desired results based on these few examples. With few-shot learning, the internal parameters of the model are not modified.

Both fine-tuning and few-shot learning can serve to enhance GPT models. Fine-tuning produces a highly specialized model that can provide more accurate and contextually relevant results for a given task. This makes it an ideal choice for cases in which a large amount of data is available. This customization ensures that the generated content is more closely aligned with the target domain’s specific language patterns, vocabulary, and tone.

Few-shot learning is a more flexible and data-efficient approach because it does not require retraining the model. This technique is beneficial when limited examples are available or rapid adaptation to different tasks is needed. Few-shot learning allows developers to quickly prototype and experiment with various tasks, making it a versatile and practical option for many use cases. Another essential criterion for choosing between the two methods is that using and training a model that uses fine-tuning is more expensive.

Fine-tuning methods often require vast amounts of data. The lack of available examples often limits the use of this type of technique. To give you an idea of the amount of data needed for fine-tuning, you can assume that for relatively simple tasks or when only minor adjustments are required, you may achieve good fine-tuning results with a few hundred examples of input prompts with their corresponding desired completion. This approach works when the pretrained GPT model already performs reasonably well on the task but needs slight refinements to better align with the target domain. However, for more complex tasks or in situations where your app needs more customization, your model may need to use many thousands of examples for the training. This can, for example, correspond to the use case we proposed earlier, with the automatic response to an email that respects your writing style. You can also do fine-tuning for very specialized tasks for which your model may need hundreds of thousands or even millions of examples. This fine-tuning scale can lead to significant performance improvements and better model adaptation to the specific domain.

**NOTE**

_Transfer learning_ applies knowledge learned from one domain to a different but related environment. Therefore, you may sometimes hear the term _transfer learning_ in relation to fine-tuning.

### Fine-Tuning with the OpenAI API

This section guides you through the process of tuning an LLM using the OpenAI API. We will explain how to prepare your data, upload datasets, and create a fine-tuned model using the API.

#### Preparing your data

To update an LLM model, it is necessary to provide a dataset with examples. The dataset should be in a JSONL file in which each row corresponds to a pair of prompts and completions:

```
{"prompt": "<prompt text>", "completion": "<completion text>"}
{"prompt": "<prompt text>", "completion": "<completion text>"}
{"prompt": "<prompt text>", "completion": "<completion text>"}
…
```

A JSONL file is a text file, with each line representing a single JSON object. You can use it to store large amounts of data efficiently. OpenAI provides a tool that helps you generate this training file. This tool can take various file formats as input (CSV, TSV, XLSX, JSON, or JSONL), requiring only that they contain a prompt and completion column/key, and that they output a training JSONL file ready to be sent for the fine-tuning process. This tool also validates and gives suggestions to improve the quality of your data.

Run this tool in your terminal using the following line of code:

```
$ openai tools fine_tunes.prepare_data -f <LOCAL_FILE>
```

The application will make a series of suggestions to improve the result of the final file; you can accept them or not. You can also specify the option `-q`, which auto-accepts all suggestions.

**NOTE**

This `openai` tool was installed and available in your terminal when you executed `pip install openai`.

If you have enough data, the tool will ask whether dividing the data into training and validation sets is necessary. This is a recommended practice. The algorithm will use the training data to modify the model’s parameters during fine-tuning. The validation set can measure the model’s performance on a dataset that has not been used to update the parameters.

Fine-tuning an LLM benefits from using high-quality examples, ideally reviewed by experts. When fine-tuning with preexisting datasets, ensure that the data is screened for offensive or inaccurate content, or examine random samples if the dataset is too large to review all entries manually.

#### Making your data available

Once your dataset with the training examples is prepared, you need to upload it to the OpenAI servers. The OpenAI API provides different functions to manipulate files. Here are the most important ones:

Uploading a file:

```
openai.File.create(
    file=open("out_openai_completion_prepared.jsonl", "rb"),
    purpose='fine-tune'
)
```

* Two parameters are mandatory: `file` and `purpose`. Set `purpose` to `fine-tune`. This validates the downloaded file format for fine-tuning. The output of this function is a dictionary in which you can retrieve the `file_id` in the `id` field. Currently, the total file size can be up to 1 GB. For more, you need to contact OpenAI.

Deleting a file:

```
openai.File.delete("file-z5mGg(...)")
```

* One parameter is mandatory: `file_id`.

Listing all uploaded files:

```
openai.File.list()
```

* It can be helpful to retrieve the ID of a file, for example, when you start the fine-tuning process.

#### Creating a fine-tuned model

Fine-tuning an uploaded file is a straightforward process. The endpoint `openai.FineTune.create()` creates a job on the OpenAI servers to refine a specified model from a given dataset. The response of this function contains the details of the queued job, including the status of the job, the `fine_tune_id`, and the name of the model at the end of the process.

The main input parameters are described in [Table 4-1](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch04.html#table-4-1).

| Field name        | Type   | Description                                                                                                                                                                                                         |
| ----------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `training_file`   | String | This is the only mandatory parameter containing the `file_id` of the uploaded file. Your dataset must be formatted as a JSONL file. Each training example is a JSON object with the keys `prompt` and `completion`. |
| `model`           | String | At the time of this writing, you can select `gpt-3.5-turbo-1106`, `babbage-002`, `davinci-002` and `gpt-4-0613` (experimental).                                                                                     |
| `validation_file` | String | This contains the `file_id` of the uploaded file with the validation data. If you provide this file, the data will be used to generate validation metrics periodically during fine-tuning.                          |
| `suffix`          | String | This is a string of up to 40 characters that is added to your custom model name.                                                                                                                                    |

#### Listing fine-tuning jobs

It is possible to obtain a list of all the fine-tuning jobs on the OpenAI servers via the following function:

```
openai.FineTune.list()
```

The result is a dictionary that contains information on all the refined models.

#### Canceling a fine-tuning job

It is possible to immediately interrupt a job running on OpenAI servers via the following function:

```
openai.FineTune.cancel()
```

This function has only one mandatory parameter: `fine_tune_id`. The `fine_tune_id` parameter is a string that starts with `ft-`; for example, `ft-Re12otqdRaJ(...)`. It is obtained after the creation of your job with the function `openai.FineTune.​cre⁠ate()`. If you have lost your `fine_tune_id`, you can retrieve it with `openai.FineTune.list()`.

### Fine-Tuning Applications

Fine-tuning offers a powerful way to enhance the performance of models across various applications. This section looks at several use cases in which fine-tuning has been effectively deployed. Take inspiration from these examples! Perhaps you have the same kind of issue in your use cases. Once again, remember that fine-tuning is more expensive than other techniques based on prompt engineering, and therefore, it will not be necessary for most of your situations. But when it is, this technique can significantly improve your results.

#### Legal document analysis

In this use case, an LLM is used to process legal texts and extract valuable information. These documents are often written with specific jargon, which makes it difficult for nonspecialists to understand these types of texts. We already saw in [Chapter 1](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch01.html#gpt\_4\_and\_chatgpt\_essentials) that when tested on the Uniform Bar Exam, GPT-4 had a score in the 90th percentile. In this case, fine-tuning could specialize the model for a specific domain and/or allow it to assist nonspecialists in the legal process. By fine-tuning an LLM on a legal corpus of a particular topic or for a specific type of end user, the model can process the intricacies of legal language better and become more adept at performing tasks related to that particular type of end user.

Fine-tuning an LLM with a large amount of data to analyze legal documents can potentially significantly improve the model’s performance in these tasks by allowing it to better process the nuances of legal language that are often beyond the capabilities of prompt engineering techniques.

#### Automated code review

In this use case, fine-tuning can help the model analyze and suggest improvements for developer-written code. This requires training the model on a dataset comprising code snippets and comments, enabling the model to process language-specific syntax, semantics, and best practices. Note that this use case is similar to what GitHub does with its Copilot tool, which is designed to assist developers in writing code by providing suggestions of code and entire functions in real time. You can use fine-tuning to train the model on a specific task such as code review, and build a project for your own code with specific standards or dedicated to a particular programming framework.

As an example of an input file for this use case, your JSONL file could contain pairs of code and their corresponding review comments. This would help the model learn how to provide accurate and relevant feedback on code quality. One line of the JSONL file could be something like this:

```
{"prompt": "Review the following Python code: 
    'def sum(a, b): return a + b\nresult = sum(5, '5')'",
"completion": "Type error: The 'sum' function is adding an integer and a string. 
    Consider converting the string to an integer using int() before passing it
    to the function."}
```

#### Financial document summarization

In financial document synthesis, a fine-tuned LLM can generate concise and accurate summaries of financial documents, such as earnings reports, financial statements, and analyst reports. By fine-tuning a language model on a dataset related explicitly to financial records, the resulting model can become more accurate in understanding the terminology and context of these documents. For example, the model could take a detailed report about an interest rate increase by a central bank and condense it into a brief yet informative summary.

To adjust an existing model to get better summaries of financial documents, you need to already have a large number of sample summaries. Then you have to provide the model with a JSONL file containing pairs of financial documents and their ideal summaries. For example:

```
{"prompt": "The central bank announced a 0.25% increase in the benchmark 
    interest rate, citing concerns about inflationary pressures. This move is
    expected to affect borrowing costs for consumers and businesses.",
"completion": "Bank increases rate 0.25% for inflation worry. May affect people,
    business borrowing costs."}
```

#### Technical document translation

Using fine-tuning to translate technical documents can significantly improve the performance of a base model compared to what you can obtain with few-shot learning. The main reason is that technical documents often contain specialized vocabulary and complex sentence structures that few-shot learning cannot handle effectively. The base models are GPT-3 models that have not been adjusted, like the InstructGPT-3 models, with reinforcement learning with human feedback techniques. To be used, they need to be fine-tuned. To adapt an existing base model, you have to prepare a JSONL file containing your training data. For the use case of technical document translation, the contents of this file would include translations of technical texts into the target language.

#### News article generation for niche topics

In this use case, a fine-tuned model could generate high-quality, engaging, and contextually relevant news articles for highly specialized topics that the base model probably would not have seen enough of to be accurate. As with all other use cases, you need to create a training dataset to specialize your model to write articles. For that, you need to have at your disposal many articles written on that specific niche topic. This data will be used to create the JSONL file containing prompt-completion pairs. Here is a small example:

```
{"prompt": "Write an introductory article about a new environmentally friendly
    cryptocurrency: 'EcoCoin: The Green Cryptocurrency Taking
    the Market by Storm'",
"completion": "As concerns over the environmental impact of cryptocurrency
    mining (...) mining process and commitment to sustainability."}
```

### Generating and Fine-Tuning Synthetic Data for an Email Marketing Campaign

In this example, we will make a text generation tool for an email marketing agency that utilizes targeted content to create personalized email campaigns for businesses. The emails are designed to engage audiences and promote products or services.

Let’s assume that our agency has a client in the payment processing industry who has asked to help them run a direct email marketing campaign to offer stores a new payment service for ecommerce. The email marketing agency decides to use fine-tuning techniques for this project. Our email marketing agency will need a large amount of data to do this fine-tuning.

In our case, we will need to generate the data synthetically for demonstration purposes, as you will see in the next subsection. Usually, the best results are obtained with data from human experts, but in some cases, synthetic data generation can be a helpful solution.

#### Creating a synthetic dataset

In the following example, we create artificial data from GPT-3.5 Turbo. To do this, we will specify in a prompt that we want promotional sentences to sell the ecommerce service to a specific merchant. The merchant is characterized by a sector of activity, the city where the store is located, and the size of the store. We get promotional sentences by sending the prompts to GPT-3.5 Turbo via the function `chat_completion`, defined earlier.

We start our script by defining three lists that correspond respectively to the type of shop, the cities where the stores are located, and the size of the stores:

```
l_sector = ['Grocery Stores', 'Restaurants', 'Fast Food Restaurants',
              'Pharmacies', 'Service Stations (Fuel)', 'Electronics Stores']
l_city = ['Brussels', 'Paris', 'Berlin']
l_size = ['small', 'medium', 'large'] 
```

Then we define the first prompt in a string. In this prompt, the role, context, and task are well defined, as they were constructed using the prompt engineering techniques described earlier in this chapter. In this string, the three values between the braces are replaced with the corresponding values later in the code. This first prompt is used to generate the synthetic data:

```
f_prompt = """ 
Role: You are an expert content writer with extensive direct marketing 
experience. You have strong writing skills, creativity, adaptability to 
different tones and styles, and a deep understanding of audience needs and
preferences for effective direct campaigns.
Context: You have to write a short message in no more than 2 sentences for a
direct marketing campaign to sell a new e-commerce payment service to stores. 
The target stores have the following three characteristics:
- The sector of activity: {sector}
- The city where the stores are located: {city} 
- The size of the stores: {size}
Task: Write a short message for the direct marketing campaign. Use the skills
defined in your role to write this message! It is important that the message
you create takes into account the product you are selling and the
characteristics of the store you are writing to.
"""
```

The following prompt contains only the values of the three variables, separated by commas. It is not used to create the synthetic data; only for fine-tuning:

```
f_sub_prompt = "{sector}, {city}, {size}"
```

Then comes the main part of the code, which iterates over the three value lists we defined earlier. We can see that the code of the block in the loop is straightforward. We replace the values in the braces of the two prompts with the appropriate values. The variable `prompt` is used with the function `chat_completion` to generate an advertisement saved in `response_txt`. The `sub_prompt` and `response_txt` variables are then added to the _out\_openai\_completion.csv_ file, our training set for fine-tuning:

```
df = pd.DataFrame()
for sector in l_sector:
    for city in l_city:
        for size in l_size:
            for i in range(3):  ## 3 times each
                prompt = f_prompt.format(sector=sector, city=city, size=size)
                sub_prompt = f_sub_prompt.format(
                    sector=sector, city=city, size=size
                )
                response_txt = chat_completion(
                    prompt, model="gpt-3.5-turbo", temperature=1
                )
                new_row = {"prompt": sub_prompt, "completion": response_txt}
                new_row = pd.DataFrame([new_row])
                df = pd.concat([df, new_row], axis=0, ignore_index=True)
df.to_csv("out_openai_completion.csv",  index=False)
```

Note that for each combination of characteristics, we produce three examples. To maximize the model’s creativity, we set the temperature to `1`. At the end of this script, we have a Pandas table stored in the file _out\_openai\_completion.csv_. It contains 162 observations, with two columns containing the prompt and the corresponding completion. Here are the first two lines of this file:

```
"Grocery Stores, Brussels, small",Introducing our new e-commerce payment service - 
the perfect solution for small Brussels-based grocery stores to easily and 
securely process online transactions. "Grocery Stores, Brussels, small",
Looking for a hassle-free payment solution for your small grocery store in
Brussels? Our new e-commerce payment service is here to simplify your
transactions and increase your revenue. Try it now!
```

We can now call the tool to generate the training file from _out\_openai\_completion.csv_ as follows:

```
$ openai tools fine_tunes.prepare_data -f out_openai_completion.csv
```

As you can see in the following lines of code, this tool makes suggestions for improving our prompt-completion pairs. At the end of this text, it even gives instructions on how to continue the fine-tuning process and advice on using the model to make predictions once the fine-tuning process is complete:

```
Analyzing...
- Based on your file extension, your file is formatted as a CSV file
- Your file contains 162 prompt-completion pairs
- Your data does not contain a common separator at the end of your prompts. 
Having a separator string appended to the end of the prompt makes it clearer
to the fine-tuned model where the completion should begin. See
https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset
for more detail and examples. If you intend to do open-ended generation, 
then you should leave the prompts empty
- Your data does not contain a common ending at the end of your completions. 
Having a common ending string appended to the end of the completion makes it
clearer to the fine-tuned model where the completion should end. See
https://oreil.ly/MOff7 for more detail and examples.
- The completion should start with a whitespace character (` `). This tends to
produce better results due to the tokenization we use. See 
https://oreil.ly/MOff7 for more details
Based on the analysis we will perform the following actions:
- [Necessary] Your format `CSV` will be converted to `JSONL`
- [Recommended] Add a suffix separator ` ->` to all prompts [Y/n]: Y
- [Recommended] Add a suffix ending `\n` to all completions [Y/n]: Y
- [Recommended] Add a whitespace character to the beginning of the completion
[Y/n]: Y
Your data will be written to a new JSONL file. Proceed [Y/n]: Y
Wrote modified file to `out_openai_completion_prepared.jsonl`
Feel free to take a look!
Now use that file when fine-tuning:
> openai api fine_tunes.create -t "out_openai_completion_prepared.jsonl"
After you’ve fine-tuned a model, remember that your prompt has to end with the 
indicator string ` ->` for the model to start generating completions, rather
than continuing with the prompt. Make sure to include `stop=["\n"]` so that the
generated texts ends at the expected place.
Once your model starts training, it'll approximately take 4.67 minutes to train
a `curie` model, and less for `ada` and `babbage`. Queue will approximately
take half an hour per job ahead of you.
```

At the end of this process, a new file called _out\_openai\_completion\_prepared.jsonl_ is available and ready to be sent to the OpenAI servers to run the fine-tuning process.

Note that, as explained in the message of the function, the prompt has been modified by adding the string `->` at the end, and a suffix ending with  has been added to all completions.

#### Fine-tuning a model with the synthetic dataset

The following code uploads the file and does the fine-tuning. In this example, we will use `davinci` as the base model, and the name of the resulting model will have `direct_marketing` as a suffix:

```
ft_file = openai.File.create(
    file=open("out_openai_completion_prepared.jsonl", "rb"), purpose="fine-tune"
)
openai.FineTune.create(
    training_file=ft_file["id"], model="davinci", suffix="direct_marketing"
)
```

This will start the update process of the `davinci` model with our data. This fine-tuning process can take some time, but when it is finished, you will have a new model adapted for your task. The time needed for this fine-tuning is mainly a function of the number of examples available in your dataset, the number of tokens in your examples, and the base model you have chosen. To give you an idea of the time needed for fine-tuning, in our example it took less than five minutes. However, we have seen some cases in which fine-tuning took more than 30 minutes:

```
$ openai api fine_tunes.create -t out_openai_completion_prepared.jsonl \ 
                -m davinci --suffix "direct_marketing"
```

```
Upload progress: 100%|| 40.8k/40.8k [00:00<00:00, 65.5Mit/s]
Uploaded file from out_openai_completion_prepared.jsonl: file-z5mGg(...)
Created fine-tune: ft-mMsm(...)
Streaming events until fine-tuning is complete...
(Ctrl-C will interrupt the stream, but not cancel the fine-tune)
[] Created fine-tune: ft-mMsm(...)
[] Fine-tune costs $0.84
[] Fine-tune enqueued. Queue number: 0
[] Fine-tune started
[] Completed epoch 1/4
[] Completed epoch 2/4
[] Completed epoch 3/4
[] Completed epoch 4/4
```

**WARNING**

As the message in the terminal explains, you will break the connection to the OpenAI servers by typing Ctrl+C in the command line, but this will not interrupt the fine-tuning process.

To reconnect to the server and get back the status of a running fine-tuning job, you can use the following command, `fine_tunes.follow`, where `fine_tune_id` is the ID of the fine-tuning job:

<pre><code><strong>$ openai api fine_tunes.follow -i fine_tune_id
</strong></code></pre>

This ID is given when you create the job. In our earlier example, our `fine_tune_id` is `ft-mMsm(...)`. If you lose your `fine_tune_id`, it is possible to display all models via:

```
$ openai api fine_tunes.list
```

To immediately cancel a fine-tune job, use this:

<pre><code><strong>$ openai api fine_tunes.cancel -i fine_tune_id
</strong></code></pre>

And to delete a fine-tune job, use this:

<pre><code><strong>$ openai api fine_tunes.delete -i fine_tune_id
</strong></code></pre>

#### Using the fine-tuned model for text completion

Once your new model is built, it can be accessed in different ways to make new completions. The easiest way to test it is probably via the Playground. To access your models in this tool, you can search for them in the drop-down menu on the righthand side of the Playground interface (see [Figure 4-4](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch04.html#fig\_4\_using\_the\_fine\_tuned\_model\_in\_the\_playground)). All your fine-tuned models are at the bottom of this list. Once you select your model, you can use it to make predictions.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098152475/files/assets/dagc_0404.png" alt="" height="318" width="600"><figcaption></figcaption></figure>

**Figure 4-4. Using the fine-tuned model in the Playground**

We used the fine-tuned LLM in the following example with the input prompt `Hotel, New York, small ->`. Without further instructions, the model automatically generated an advertisement to sell an ecommerce payment service for a small hotel in New York.

We already obtained excellent results with a small dataset comprising only 162 examples. For a fine-tuning task, it is generally recommended to have several hundred instances, ideally several thousand. In addition, our training set was generated synthetically when ideally it should have been written by a human expert in marketing.

To use it with the OpenAI API, we proceed as before with `openai.Completion.​cre⁠ate()`, except that we need to use the name of our new model as an input parameter. Don’t forget to end all your prompts with `->` and to set  as stop words:

```
openai.Completion.create(
  model="davinci:ft-book:direct-marketing-2023-05-01-15-20-35",
  prompt="Hotel, New York, small ->",
  max_tokens=100,
  temperature=0,
  stop="\n"
)
```

We obtain the following answer:

```
<OpenAIObject text_completion id=cmpl-7BTkrdo(...) at 0x7f2(4ca5c220> JSON: {
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "text": " \"Upgrade your hotel's payment system with our new e-commerce \ 
service, designed for small businesses.
    }
  ],
  "created": 1682970309,
  "id": "cmpl-7BTkrdo(...)",
  "model": "davinci:ft-book:direct-marketing-2023-05-01-15-20-35",
  "object": "text_completion",
  "usage": {
    "completion_tokens": 37,
    "prompt_tokens": 8,
    "total_tokens": 45
  }
}
```

As we have shown, fine-tuning can enable Python developers to tailor LLMs to their unique business needs, especially in dynamic domains such as our email marketing example. It’s a powerful approach to customizing the language models you need for your applications. Ultimately, this can easily help you serve your customers better and drive business growth.

### Cost of Fine-Tuning

The use of fine-tuned models is costly. First you have to pay for the training, and once the model is ready, each prediction will cost you a little more than if you had used the base models provided by OpenAI.

Pricing is subject to change, but at the time of this writing, it looks like [Table 4-2](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch04.html#table-4-2).

| Model           | Training                 | Input Usage              | Output usage             |
| --------------- | ------------------------ | ------------------------ | ------------------------ |
| `gpt-3.5-turbo` | $0.0080 per 1,000 tokens | $0.0030 per 1,000 tokens | $0.0060 per 1,000 tokens |
| `davinci-002`   | $0.0060 per 1,000 tokens | $0.0120 per 1,000 tokens | $0.0120 per 1,000 tokens |
| `babbage-002`   | $0.0004 per 1,000 tokens | $0.0016 per 1,000 tokens | $0.0016 per 1,000 tokens |

As a point of comparison, the price of the `gpt-3.5-turbo` model without fine-tuning is $0.002 per 1,000 tokens. As already mentioned, `gpt-3.5-turbo` has the best cost-performance ratio.

To get the latest prices, visit the [OpenAI pricing page](https://openai.com/pricing).

## Summary

This chapter discussed advanced techniques to unlock the full potential of GPT-4 and ChatGPT and provided key actionable takeaways to improve the development of applications using LLMs.

Developers can benefit from understanding prompt engineering, zero-shot learning, few-shot learning, and fine-tuning to create more effective and targeted applications. We explored how to create effective prompts by considering the context, task, and role, which enable more precise interactions with the models. With step-by-step reasoning, developers can encourage the model to reason more effectively and handle complex tasks. In addition, we discussed the flexibility and adaptability that few-shot learning offers, highlighting its data-efficient nature and ability to adapt to different tasks quickly.

[Table 4-3](https://learning.oreilly.com/library/view/developing-apps-with/9781098152475/ch04.html#table-4-3) provides a quick summary of all these techniques, when to use them, and how they compare.

|            | Zero-shot learning                             | Few-shot learning                                                                                     | Prompt engineering tricks                                                                         | Fine-tuning                                                                                                                                                                                                                                                                |
| ---------- | ---------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Definition | Predicting unseen tasks without prior examples | Prompt includes examples of inputs and desired output                                                 | Detailed prompt that can include context, role, and tasks, or tricks such as “think step by step” | Model is further trained on a smaller, specific dataset; prompts used are simple                                                                                                                                                                                           |
| Use case   | Simple tasks                                   | Well-defined but complex tasks, usually with specific output format                                   | Creative, complex tasks                                                                           | Highly complex tasks                                                                                                                                                                                                                                                       |
| Data       | Requires no additional example data            | Requires a few examples                                                                               | Quantity of data depends on the prompt engineering technique                                      | Requires a large training dataset                                                                                                                                                                                                                                          |
| Pricing    | Usage: pricing per token (input + output)      | Usage: pricing per token (input + output); can lead to long prompts                                   | Usage: pricing per token (input + output), can lead to long prompts                               | <p>Training:<br>Usage: pricing per token (input + output) is about 80 times more expensive for fine-tuned <code>davinci</code> compared to GPT-3.5 Turbo. This means that fine-tuning is financially preferable if other techniques lead to a prompt 80 times as long.</p> |
| Conclusion | Use by default                                 | If zero-shot learning does not work because the output needs to be particular, use few-shot learning. | If zero-shot learning does not work because the task is too complex, try prompt engineering.      | If you have a very specific and large dataset and the other solutions do not give good enough results, this should be used as a last resort.                                                                                                                               |

To ensure success in building LLM applications, developers should experiment with other techniques and evaluate the model’s responses for accuracy and relevance. In addition, developers should be aware of LLM’s computational limitations and adjust their prompts accordingly to achieve better results. By integrating these advanced techniques and continually refining their approach, developers can create powerful and innovative applications that unlock the true potential of GPT-4 and ChatGPT.

In the next chapter, you will discover two additional ways to integrate LLM capabilities into your applications: plug-ins and the LangChain framework. These tools enable developers to create innovative applications, access up-to-date information, and simplify the development of applications that integrate LLMs. We will also provide insight into the future of LLMs and their impact on app development.
