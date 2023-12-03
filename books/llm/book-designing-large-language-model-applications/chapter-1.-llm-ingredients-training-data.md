# Chapter 1. LLM Ingredients: Training Data

## Chapter 1. LLM Ingredients: Training Data

## A NOTE FOR EARLY RELEASE READERS

With Early Release ebooks, you get books in their earliest form—the author’s raw and unedited content as they write—so you can take advantage of these technologies long before the official release of these titles.

This will be the 3rd chapter of the final book. Please note that the GitHub repo will be made active later on.

If you have comments about how we might improve the content and/or examples in this book, or if you notice missing material within this chapter, please reach out to the author at [mcronin@oreilly.com](mailto:mcronin@oreilly.com).

In Chapter 1, we defined LLMs, ruminated on their strengths and limitations, explored current and potential use cases, and presented the scaling laws that seemingly govern progress in this field. In Chapter 2, we dug deep into the trenches to understand the most significant advance in machine learning in recent times, the Transformer architecture which makes modern LLMs possible. Armed with this knowledge, let’s set our sights on utilizing these models to build useful applications!

To set the stage for the rest of this book, in this chapter and the next we will discuss the recipe for pre-training LLMs and the ingredients that go into them in detail. We will also take a journey through the LLM landscape and showcase the different pre-trained models available for our use, both open-source and proprietary. We will classify them according to various criteria including training data domain, architecture type, licensing etc.

But wait, this book is about utilizing pre-trained LLMs to design and build user applications. Why do we need to discuss the nuances of pre-training billion parameter models from scratch, something most machine learning practitioners are never going to do in their lives?

Actually, this information is very important because many of the decisions taken during the pre-training process heavily impact downstream performance. As we will notice in subsequent chapters, failure modes are more easily understandable when you have a comprehension of the training process. Just like we appreciate having ingredients listed on packages at our grocery stores, we would like to know the ingredients that go into making a language model before we use it in serious applications.

**NOTE**

There is not much information available in the public realm about some of the proprietary LLMs that are accessible only through an API. This book will provide as much information as has been made public. While the lack of information doesn’t mean that we should avoid using these models, model transparency is something that you might need to take into your calculus while making a final decision regarding what model to use.

## Ingredients of an LLM

Let’s start with the ingredients that go into making an LLM.

Broadly speaking, we have:

1. Pre-training data - **What’s it trained on?** As the old computer science adage ‘Garbage In, Garbage Out’ comes back to bite us, we will explore popular pre-training datasets and dig into the various pre-processing steps taken to ensure _high-quality_ data is fed to the model. We will also showcase some tools that allow us to probe these datasets and understand how pre-training data composition impacts downstream tasks.
2. Vocabulary and tokenizer - **What’s it trained over?** In order to build a model over a language, we have to first determine the vocabulary of the language we are modeling, and rules to break down a stream of text into the right vocabulary units (tokenization). Linguistically, humans process language in terms of meaning-bearing words and sentences. Language models process language in terms of tokens. We will explore the downstream impact when there is a mismatch between the two.
3. Learning objective - **What is it being trained to do?** By pre-training a language model, we aim to imbibe the language model with general skills in syntax, semantics, reasoning and so on, that will hopefully enable it to reliably solve any task you throw at it even if it was not specifically trained on it. We will discuss the various tasks (learning objectives) that pre-trained models are trained on. You might wonder if LLMs are better suited to solving downstream tasks that are similar to the tasks the pre-trained model has been trained to solve. We will test this assumption and discuss the impact various learning objectives have on task performance.
4. Architecture - **What’s its internal structure?** As mentioned in Chapter 2, most modern language models are based on the Transformer architecture. We will discuss the various architectural backbones- specifically encoder-only models, encoder-decoder models, and decoder-only models, and the rationale used by organizations training LLMs for their choice of architecture type.

Let’s look at how these ingredients fit together in ([Figure 1-1](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch01.html#ingredients-of-llm)):

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/figure109.png" alt="LLM Ingredients" height="118" width="600"><figcaption></figcaption></figure>

**Figure 1-1. Figure depicting how all the ingredients come together to make an LLM.**

The language models trained using the process described in this chapter and the next are called _base models_. Lately, model providers have been augmenting the base model by tuning it on much smaller datasets in order to steer them towards being more aligned with human needs and preferences. Some popular tuning modes are:

* Supervised instruction fine-tuning, so that the model is better at following human instructions.
* RLHF (Reinforcement Learning by Human Feedback), so that the model is better aligned with human preferences.
* Domain-adaptive or task-adaptive continued pre-training, so that the model is better attuned to specific domains and tasks.

to name a few. Based on the specific augmentation carried out, the resulting models are called _instruct models_, _chat models_ and so on.

We will cover instruct and chat models in Chapter 6, and domain/task-adaptive pre-training in Chapter 8.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/Fig1-2_highres.png" alt="Derivative Models" height="182" width="600"><figcaption></figcaption></figure>

**Figure 1-2. Figure showing the relationship between base models and their derivatives.**

## LLM PRE-TRAINING CHALLENGES

Pre-training an LLM is a very technically challenging task, and requires a lot of computational resources and exceptional technical skills. For example, GPT-4’s [technical report](https://arxiv.org/abs/2303.08774) credits 343 unique contributors, not including the [annotators](https://time.com/6247678/openai-chatg) in Kenya who contributed to their [RLHF](https://huggingface.co/blog/rlhf) (Reinforcement Learning with Human Feedback) training. Delving into every aspect of pre-training LLMs is an entire book in itself. In this chapter we will not focus on infrastructure or engineering considerations for pre-training LLMs, nor focus on the nuances of distributed and parallel computing. We will instead focus on aspects of the pre-training process that can directly impact your application’s behavior and performance.

However, if you are curious to read more about the challenges involved in pre-training LLMs, here are some useful resources to quench your thirst -

* [Blog post](https://huggingface.co/blog/bloom-megatron-deepspeed) from Big Science that explains the hardware, types of parallelisms employed, and optimizations used in training BLOOM, an open-source 176B parameter multilingual model.
* Training chronicles(log book) from [BLOOM](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md) and [OPT](https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/README.md), which is a 175B parameter LLM released by Meta, documenting the trials and tribulations faced during training, including hardware failures and how to recover from them, training instabilities, loss spikes and the like.
* [Video](https://www.youtube.com/watch?v=p9IxoSkvZ-M) featuring Susan Zhang, the lead author of OPT, who discusses the OPT chronicles in detail.
* [The Deep Learning Tuning book](https://github.com/google-research/tuning\_playbook) by Google, which discusses hyperparameter optimization, multi-host setups, training instabilities and a lot more.

## Pre-training data requirements

Although it has been shown that higher capacity models are relatively more [sample efficient](https://arxiv.org/abs/2001.08361), in general neural networks are very sample inefficient, meaning they need tons of examples to learn a task. It is infeasible to create such a large supervised dataset with human annotations, hence the predominant means to pre-train language models is using _self-supervised_ learning, where the target labels exist within your training inputs.

Using this setup, virtually any type of text is fair game to be included in a pre-training dataset, and theoretically any non-textual signal can be encoded in text and included as part of a pre-training dataset.

From our scaling laws discussion in Chapter 1, we know that most current language models are severely undertrained and can benefit from additional performance gains by just training them longer and on more data. Also, as discussed in Chapter 1, the _consolidation effect_ at play in the field raises expectations on what a single language model is expected to do end-to-end. Today a single model is expected to answer factual questions about the world, employ arithmetic and logical reasoning, write code, and come up with creative ideas.

All this means that the data needs for language model pre-training are enormous. Now, the key question is if textual data available in the world actually contains sufficient and relevant signals needed to learn all the skills we want LLMs to learn.

Note that language models that are trained solely on text only have access to the linguistic form i.e the sequence of characters making up a sentence like ‘Walter White tossed the pizza onto the roof’. In order to understand its meaning, the linguistic form has to be mapped to the communicative intent of the writer/speaker. While a [section](https://aclanthology.org/2020.acl-main.463.pdf) of the research community argues that one cannot learn meaning from form alone, recent language models are increasingly proving otherwise.

In order to have access to the full picture, the linguistic form needs to be grounded to the real world. In the cognitive sciences, grounding is defined as

> The process of establishing what mutual information is required for successful communication between two interlocutors
>
> Chandu et al., _Grounding ‘grounding’ in NLP_

Human text is generally very underspecified, with a lot of communicative intent existing outside the textual context, depending on the reader/listener to use their common sense, world knowledge, ability to detect and understand emotional subtext in order to interpret it.

**NOTE**

It is estimated that only around [12% of information](https://link.springer.com/book/10.1007/978-1-4612-5880-3) we understand from text is explicitly mentioned in text. There are several theories explaining why we communicate thus, including [Zipf’s principle of least effort](https://en.wikipedia.org/wiki/Principle\_of\_least\_effort), which states it is “human nature to want the greatest outcome at the least amount of work”.

The field of NLP has seen [a lot of work](https://aclanthology.org/2021.findings-acl.375.pdf) in grounding language models to the real world. [Multimodal models](https://arxiv.org/abs/2206.06488) that combine different modalities like image, video, speech, text are a promising avenue of research, and are likely to see more widespread usage in the coming years. Imagine a model seeing ‘pizza’ in the training text, but also getting signals on how it looks, how it sounds, and how it tastes!

But do multimodal models really help? Can we achieve the effect of grounding by just feeding the model with massive amounts of diverse text? These are unsolved questions, and there are good arguments in both directions as shown by this [debate](https://www.youtube.com/watch?v=x10964w00zk).

Whether training on massive amounts of text alone can enable language models to learn skills like logical reasoning is another open question. Note that text on the Internet contains a lot of text describing reasoning steps, like theorem proofs, explanations of jokes, step-by-step answers to puzzles and so on. However, there is simply not enough of derivational text going around, which leads us to cover the shortfall by using prompting methods like chain-of-thought (described further in Chapter 5). There is [recent evidence](https://cdn.openai.com/improving-mathematical-reasoning-with-process-supervision/Lets\_Verify\_Step\_by\_Step.pdf) that process supervision, where feedback is provided for each step of the problem-solving process, as opposed to outcome supervision, where feedback is provided only on the final solution, helps improve arithmetic reasoning.

A crucial skill that language models have to learn is dealing with the inherently ambiguous nature of language. Following up on the aforementioned Zipf’s principle of least effort, ambiguity enables speakers to manage the efficiency-clarity tradeoff in communication. Earlier language models struggled heavily with modeling ambiguity. As an example, I long used this sentence as a canonical example in my NLP talks to highlight ambiguity in language.

“WWE’s John Cena surprises Make-A-Wish 7-year-old with cancer.”

While GPT-4 seems to get the correct interpretation of this particular sentence, [recent work](https://arxiv.org/abs/2304.14399) shows that state-of-the-art models like GPT-4 still struggle to deal with ambiguity in general. Whether just scaling up models and data is enough for LLMs to model ambiguity is an open question.

If our only option to resolve all these shortcomings is to scale up dataset sizes, the next question is if we actually have enough data available in the world that is sufficient to enable LLM’s to learn these skills. Are we at risk of running out of training data any time soon? There is a misconception in certain quarters of our field that we are. However, lack of raw data is far away from being a bottleneck in training models. For instance, there are billions of publicly available documents accessible by scraping or via a free API that haven’t yet made it into most pre-training data sets such as parliamentary proceedings, court judgements, and most SEC filings. Moreover, text generated by language models can be used to self-improve them, albeit with the [risk](https://arxiv.org/abs/2305.17493v2) that training on LLM-generated data can be detrimental, as the model deviates from the true distribution of the data.

Of course, one could make a distinction between the volume of available _high-quality_ data vs _low-quality_ data and claim that it is high-quality data that is close to exhaustion , but what exactly makes data high-quality is a very nuanced question.

**NOTE**

LLMs are underfit, and are usually trained with just one epoch or less (each training example is fed to the model only once, unless duplicates of that example exist across the dataset). However, in recent times, there is increasing evidence that you can safely train for multiple epochs (at least \~5) without being in danger of overfitting. The GALACTICA model from Meta was trained on 4 epochs, and noted improved performance. Recent work from [Muennighoff et al.](https://arxiv.org/abs/2305.16264) and [Xue et al.](https://arxiv.org/abs/2305.13230) provide further evidence on this. Therefore, the impending data-apocalypse has been thwarted even further.

## COPYRIGHT ISSUES PERTAINING TO PRE-TRAINING DATASETS

Can LLMs be trained on copyrighted text without the explicit consent of the copyright holder and without attribution? Can LLMs be trained on text that inadvertantly contains sensitive personal information without legal liabilities? These are all fluid legal and moral questions. In the U.S, the ‘fair use’ doctrine has been used to justify training LLMs on copyrighted text. However, this is currently being tested, and as of this book’s writing, a [class action lawsuit](https://www.theverge.com/2022/11/8/23446821/microsoft-openai-github-copilot-class-action-lawsuit-ai-copyright-violation-training-data) has been filed against Github, Microsoft, and OpenAI for using code from Github repositories that were published under restrictive licenses for training Github Copilot, a code completion LLM. The AI community will be watching this case with interest. However, all over the world, laws are [fast loosening](https://petapixel.com/2023/06/05/japan-declares-ai-training-data-fair-game-and-will-not-enforce-copyright/) to permit this type of usage and clear legal hurdles for LLM training and adoption.

As LLM usage expands and they become an integral part of the economy, data used to train them becomes more valuable. Reddit and StackOverflow, both of which have been an important source of data in many influential pre-training datasets, have [recently announced](https://www.wired.com/story/stack-overflow-will-charge-ai-giants-for-training-data/) they will start charging for data access. Expect more such announcements in future.

What are the copyright implications for people and organizations using these language models downstream? We will discuss this in more detail in Chapter 14, where we will provide more background on the various types of software licenses and their degree of permissibility for commercial usage.

## Popular pre-training datasets

A lot of text is not freely available in public. This includes data exposed behind paywalled APIs and login screens, and paywalled books and documents, many of whom may not even be digitized. Larger companies like Google and OpenAI can afford to purchase this data - Elon Musk [revealed](https://twitter.com/elonmusk/status/1599291104687374338?s=20) that Open AI had access to the Twitter database, and Google has access to over [40 million books](https://blog.google/products/search/google-books-library-project/) it has scanned and digitized as part of the Google Books project. Domain specific text is often proprietary and available only to large incumbents (for example Bloomberg trained [BloombergGPT](https://arxiv.org/abs/2303.17564) partly on their proprietary financial data). However, even for models trained by the largest companies, a significant proportion of training data comes from publicly available data sources.

Next, we will cover some of the most popular general purpose pre-training datasets that are being used to train LLMs. While this is not a comprehensive list, most LLMs, including closed-source ones, have at least a large subset of their training data drawn from these sources. We will defer discussion of domain-specific (catered to a particular field like social media, finance, biomedical etc) datasets to Chapter 8.

**TIP**

Most general purpose LLMs are trained to be a jack-of-all-trades - to be able to solve tasks from a variety of domains. If the data domain for your use case happens to be represented in a pre-training dataset, you will see some performance improvement on your downstream task, even though the data in the pre-training dataset is unlabeled. This means that if you intend to use LLMs for a specific well-defined use case in a particular domain, then domain-specific models could likely be more preferable. You can also perform _continued domain-adaptive or task-adaptive pretraining_ to leverage this phenomenon. This will be discussed in detail in Chapter 8.

_Common Crawl/C4:_ The Web is the largest source of openly available textual data. [Common Crawl](https://commoncrawl.org/the-data/) is a non-profit that creates and makes available a snapshot of all web crawl data, updated every month. However, as one could imagine, this is an extremely coarse data set and needs to be significantly cleaned before it is ready to use. Most pre-training datasets have a sizeable portion of their data sources from Common Crawl. Google prepared C4 (Colossal Clean Crawled Corpus), a 750GB dataset after applying a set of pre-processing and filtering steps to a Common Crawl snapshot from 2019 and released the code for it. [Dodge et al](https://arxiv.org/abs/2104.08758). used this script to reproduce C4 and have made it publicly available. C4 has been used for training several well-known LLMs including all models from the T5 family.

_The Pile:_ [The Pile](https://pile.eleuther.ai/) is a 825 GB dataset from Eluether AI, who focused on publishing a dataset that is drawn from more diverse sources. Diversity of data is important since in-domain unlabeled data in pre-training is helpful for downstream performance on that domain, and diverse data sets also enable generalization to previously unseen tasks and domains. To this end, the data from The Pile comes not only from Common Crawl but also PubMed Central, ArXiv, GitHub, the FreeLaw Project, Stack Exchange, the US Patent and Trademark Office, PubMed, Ubuntu IRC, HackerNews, YouTube, PhilPapers, NIH ExPorter, Project Gutenberg, Wikipedia among others. It is one of the most preferred datasets for open-source LLM models today.

_ROOTS:_ The [ROOTS](https://arxiv.org/abs/2303.03915) dataset is a 1.61 TB multilingual dataset released by BigScience, the open source collaboration that trained BLOOM, which at the time of release was the largest multilingual language model in the world. A large proportion of ROOTS data comes from web domains and datasets that were marked by volunteers from across the world as being highly relevant.

_WebText/OpenWebText/OpenWebText2:_ These refer to a subset of web text, and are limited to text from pages representing outbound links on Reddit that have at least 3 _karma_, where karma refers to the absolute difference between upvotes and downvotes. The idea is that the wisdom of the crowds will enable only quality links to surface, that contain information that people actually find interesting. Models that have been trained on this data include GPT-2 and GPT-3.

_Wikipedia_ - A full dump of Wikipedia contains valuable encyclopedic text that provides factual knowledge to the model. Wikipedia’s editorial system ensures that the text follows a highly structured format. However, it is not diverse stylistically, with text written in a formal manner. Hence, it is usually combined with a corpus like the BooksCorpus.

_BooksCorpus/BooksCorpus2_ - Probably the most influential of all pre-training datasets, this dataset was part of the training corpus for well known models like BERT, RoBERTa, GPT-2/3 etc. The BooksCorpus contains over 7,000 free, mostly fiction books written by unpublished authors. It has since been found that several books in the dataset have [restrictive copyright licenses](https://arxiv.org/abs/2105.05241). The original corpus is no longer public. 26% of books in the original dataset belonged to the Romance genre. A replication of the BooksCorpus is present in The Pile as BooksCorpus2.

The following table provides a list of some of the most commonly used datasets, their size, year of release, and the means to access them.

| Name         | Data Source(s)                                                                                       | Size          | Year Released | Public?                        | Models using this dataset                            |
| ------------ | ---------------------------------------------------------------------------------------------------- | ------------- | ------------- | ------------------------------ | ---------------------------------------------------- |
| C4           | Common Crawl                                                                                         | 750GB         | 2019          | Yes (reproduced version)       | T5, Flan-T5, UL2, Llama etc                          |
| The Pile     | Common Crawl, PubMed Central, Wikipedia, ArXiv, Project Gutenburg, Stack Exchange, USPTO, Github etc | 825GB         | 2020          | Yes                            | GPT-Neo/X, GPT-J, Cerebras-GPT, StableLM, Pythia etc |
| RedPajama    | Common Crawl, Github, Wikipedia, arXiv, StackExchange etc                                            | 1.2T tokens   | 2023          | Yes                            | Red Pajama-INCITE, MPT                               |
| BooksCorpus  | Sampled from smashwords.com                                                                          | 74M sentences | 2015          | Original not available anymore | Most models including BERT, GPT etc                  |
| OpenWebText2 | outbound reddit links                                                                                | 65GB          | 2020          | Yes                            | GPT2, GPT3                                           |
| ROOTS        | BigScience Catalogue, Common Crawl, Github                                                           | 1.6T tokens   | 2022          | No (but available on request)  | BLOOM                                                |
| RefinedWeb   | Common Crawl                                                                                         | 5T tokens     | 2023          | Yes (600B subset only)         | Falcon                                               |
| SlimPajama   | Cleaned from RedPajama                                                                               | 627B tokens   | 2023          | Yes                            | N/A                                                  |

As you can see, most models are trained from the same few datasets. In this chapter, we are limiting our coverage to pre-training datasets for base models. We will cover datasets used to augment base models like instruction tuning datasets, RLHF datasets, prompt datasets etc in Chapter 6.

## EXERCISE

Let’s do some sleuthing. Investigate the C4 dataset and explore its characteristics.

* Is your personal data present in C4? Use this [tool](https://c4-search.apps.allenai.org/) to find out.
* Consider a domain of your choice (finance, poetry, biomedical etc), catering to your professional and/or personal interests. What are the popular websites for your domain? To find out what proportion or C4’s data comes from those websites, you can use [this tool](https://www.washingtonpost.com/technology/interactive/2023/ai-chatbot-learning/) from The Washington Post.(scroll down until you find the tool _the websites in C4’s dataset_.)

## Training Data Preprocessing

Once we have collected or procured data, we need to run the data through a preprocessing pipeline in order to create the pre-training dataset. Data preprocessing is the most unglamorous and underappreciated part of the LLM training pipeline, yet perhaps the most important. I would argue that there are a lot of low-hanging gains to be had for LLMs just by focusing more on data pre-processing. As we walk through the data processing pipeline, I hope you come to appreciate the complexity of language text and the difficulty in processing it. Note that since these datasets are enormous, any preprocessing step should also be very efficient (ideally linear time).

[Figure 1-3](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch01.html#data-collection) shows the typical preprocessing steps used to generate a pre-training dataset. The ordering of steps is not fixed, but there are dependencies between some of the steps.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/Fig1-3_highres.png" alt="Data preprocessing pipeline" height="395" width="600"><figcaption></figcaption></figure>

**Figure 1-3. Data collection and pre-processing pipeline**

Let’s go through these steps in detail.

### Data filtering and cleaning

A majority of text extracted from HTML files is gibberish, like menu text from websites, boilerplate text, and random web page artifacts. There is a significant amount of pornography, toxic, and hateful language on the Web too. For example, here is how a text sample from an uncleaned version of the C4 dataset looks like:

_“Skip to Main Content Skip to Footer Skip to Email Signup Skip to Feedback Form MY REWARDS SIGN OUT SIGN IN & EARN REWARDS 0 Keyboard Controls Welcome to the main navigation. This menu has three levels of product categories. Use and keys to navigate between each category in the current level. Use the key to navigate down a level. Use the key to navigate up a level. Hit the key to be taken to the selected category page. Men What’s Hot New Arrivals Brand That Unites Performance Shop Online Exclusives Express Essentials Vacation Getaway Wedding Tuxedos Military Trend 9 Pieces / 33 Looks The Edit x Express NBA Collection Express + NBA Fashion NBA Game Changers Suiting & Blazers Find”_

How useful do you think this text is for language and task learning?

Data from Common Crawl is made available via both raw HTML and WET (web-extracted text) format. While many dataset creators directly use the WET files, the open source organization Eluether AI [noticed](https://arxiv.org/abs/2101.00027) that the quality of the WET files left much to be desired, with HTML boilerplate still prominent as seen above. To create The Pile, Eleuther AI thus used the [justext](https://github.com/miso-belica/jusText) library to more reliably remove boilerplate text from HTML documents.

Let us explore the effect of using justext with an example.In your Google Colab or jupyter-lab notebook, try this out -

```
!pip install justext

import requests
import justext

response = requests.get("https://en.wikipedia.org/wiki/Toronto_Transit_Commission")
text = justext.justext(response.content, justext.get_stoplist("English"))
for content in text:
  if content.is_boilerplate:
    print(content.text)
```

The output displays all the boilerplate that is filtered out from a standard Wikipedia article.

```
Jump to content
Main menu
Main menu
Navigation
Main page
Contents
Current events
Random article
About Wikipedia
Contact us
Donate
Contribute
Help
Learn to edit
Community portal
Recent changes
Upload file
Languages
Language links are at the top of the page across from the title.
Search
Create account
Log in
Personal tools
…
```

justext just so happens to be more aggressive in removing content, but this is generally OK for cleaning pre-trained datasets since there is an abundance of text available. Some alternative libraries used for this task include [dragnet](https://github.com/dragnet-org/dragnet), [html2text](https://github.com/aaronsw/html2text), [inscriptis](https://github.com/weblyzard/inscriptis), [newspaper](https://github.com/codelucas/newspaper/), and [trafilatura](https://github.com/adbar/trafilatura). [According](https://arxiv.org/abs/2101.00027) to the creators of The Pile, dividing the extraction pipeline across multiple libraries can reduce the risk of the resulting dataset being affected by any bias introduced by one of these libraries.

## EXERCISE

Use your favorite news website and open a news article. Use any of the text extraction libraries mentioned, to remove web boilerplate. Is the output desirable on your first try? What kind of additional heuristics might you need?

## PRE-TRAINING ON RAW HTML DOCUMENTS

Do we really need to filter out HTML tags from raw HTML documents before pre-training? What if we pre-trained on raw HTML documents instead? This outlandish yet creative idea was implemented by [Aghajanyan et al.](https://arxiv.org/abs/2107.06955) in their HTLM (Hyper-text Language Model) model. The structured format of HTML enables valuable metadata to be encoded with text. For example, the \<title> tags could represent the summary, and the \<class> tags could provide category information about the text.

Not all of the HTML is useful for pre-training. For example, CSS isn’t very informative for language learning. Therefore, the creators of HTLM convert the raw HTML into a simplified form, by filtering out iframes, headers, footers, forms etc. This process is called _minification._

The results presented in their paper show the model is especially good at summarization, because the access to the category tags helps it focus on the salient aspects of the topic under discussion. However, as of this book’s writing, this pre-training paradigm hasn’t caught on yet.

Once text is extracted, rudimentary filtering steps based on heuristics are applied. While the details differ across datasets, here are some of the steps typically performed:

* Boilerplate Removal: Only lines that end with a punctuation, like the period, exclamation and question mark are retained. This ensures that menu text from websites is removed. Only lines with greater than a particular threshold of words and documents with greater than a particular threshold of sentences are retained. The latter helps in modeling long sequences which is an important capability for language models to have. Documents containing _lorem ipsum…_ and other boilerplate text are filtered out.
* Non-English text removal: Libraries like _langdetect, langid, fasttext, pycld2_ are used to detect the language of the text. For example, C4 retains text that has > 0.99 probability of English as judged by _langdetect._ Note that these libraries can also be used to remove boilerplate and web page artifacts since they give a lower probability of English to those texts.
* SEO text/Spam removal: Documents with a lot of repeated character sequences are removed. Documents with a low proportion of closed class words are removed. Closed class words in English are function words like of, at, the, is etc. If a page is engaged in keyword stuffing and other SEO tricks, then they would have a lower closed class words ratio.
* Pornographic/abusive text removal: Documents containing any words from keyword lists like the [“List of Dirty, Naughty, Obscene or Otherwise Bad Words”](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words) are removed.

Tools like _langdetect_ and _langid_ are helpful for speedy determination of the language in which text is written at scale, but how do they deal with code-switched text (text with multiple languages, where oftentimes it is English interspersed with a local language)?

You can try it out yourself! Here is an example for Taglish (Tagalog + English, which is a common mode of communication in the Philippines). In your notebook, run

```
!pip install langdetect

from langdetect import detect_langs()

detect_langs("""Pag-uwi ko galing sa paaralan, sobrang pagod ako dahil sa dami

ng aking ginawa sa buong araw. Ang traffic din sa kalsada, nakaka-stress

talaga! Pero nang makarating ako sa aking tahanan, nabuhayan ako ng loob dahil

sa masarap na amoy ng ulam na inihanda ni nanay. Excited na akong kumain

kasama ang aking pamilya at i-share ang mga kwento ko tungkol sa aking mga

kaibigan, guro, at mga natutunan ko sa school. After dinner, magre-relax muna

ako habang nanonood ng TV, and then magre-review ng lessons bago matulog. Ito

ang routine ko pag-uwi mula sa school, at masaya ako na dumating sa bahay namay

naghihintay na pamilya na handang makinig at suportahan ako sa aking

pag-aaral.""")
```

Output:

```
[tl:0.9999984631271781]
```

```
detect_langs("""After a long day at school, pagod na pagod talaga ako. The

traffic on the way home didn't help, nakakastress na nga! But upon arriving

home, I felt a sense of relief dahil sa welcoming atmosphere and the delicious
aroma of the ulam na inihanda ni Mommy. Excited na akong mag-share ng

experiences ko today with my family during dinner, kasama ang mga kwento about
my friends, teachers, and interesting lessons sa school. After eating, it's

time for me to chill while watching some TV shows, and then review my lessons

bago ako matulog. This is my daily routine pag-uwi galing school, and I am

grateful na may loving family ako na handang makinig at supportahan ako sa

aking educational journey.""")
```

Output:

```
[en:0.9999954357601804]
```

The second paragraph would get included in the C4 dataset, as per its filtering criteria (probability of English should be greater than .99). Therefore, even datasets that claim to be English-only routinely contain text in other languages, leading to surprising multilingual behavior during inference. Ever wondered why some monolingual models seem to perform well at machine translation? This is a major reason.

The way _langdetect_ is implemented makes it poor at identifying language when short sequences are provided. For example:

```
detect_langs('I love you too.')
```

returns

```
[sk:0.8571379760844766, en:0.14285726700161824]
```

sk refers to Slovak here.

## EXERCISE

C4 is an English language dataset, with text getting less than 0.99 probability of English in _langdetect_ being removed. However, a lot of non-English data persists in this dataset. If you know a second language, then [search for words](https://c4-search.apps.allenai.org/) in that language in C4. In what contexts do these non-English text fragments appear? Could an LLM _learn_ these languages using these leftover fragments?

### Selecting Quality Documents

While LLM’s are trained with the intention of making them a jack-of-all-trades, the Internet is a very vast place and not all data is created equal. There are many websites whose content one would be hard pressed to find relevancy to any potential downstream task, however imaginative you might be. Moreover, as we have seen earlier, the data cleaning process is far from optimal. A common way of filtering out less _useful_ documents from Common Crawl is to build a classifier for quality text. The examples for the positive class are from a dataset known to be useful, like say, Wikipedia, and the examples for the negative class would be random documents from common crawl.

#### Perplexity for quality selection

[Perplexity](http://blog.echen.me/2021/12/23/a-laymans-introduction-to-perplexity-in-nlp/), an intrinsic evaluation measure for language models, has been used in the data-processing stage for document filtering, notably by the creators of [CCNet](https://arxiv.org/abs/1911.00359).

Just like the classifier approach, we select documents from data sources (like Wikipedia) that we deem useful as the positive class. We then train a 5-gram language model using [KenLM](https://github.com/kpu/kenlm) (a library facilitating training of n-gram language models.) over it. Next, we take the dataset we want to filter, and calculate the perplexity of each paragraph in it over the trained language model. The lower the perplexity, the more similar it is to the positive class. We can then discard documents with high perplexity.

Low perplexity may not always be a good thing. Short and repetitive text can have low perplexity. Note that writing style gets factored into perplexity. If the reference language model is trained over Wikipedia, then documents written in an informal style may receive higher perplexity scores. Therefore, it would be beneficial to have a more involved filtering strategy.

To resolve this, the creators of [BERTIN](https://rua.ua.es/dspace/bitstream/10045/122846/1/PLN\_68\_01.pdf) introduced the concept of perplexity sampling. In perplexity sampling, instead of just filtering out low-perplexity text, they utilize perplexity scores in a sampling strategy over their dataset. The sampling strategy is to oversample from the middle part of the perplexity probability distribution.

#### Exploring perplexity with Wikipedia LMs

Download the file [_https://huggingface.co/edugp/kenlm/blob/main/model.py_](https://huggingface.co/edugp/kenlm/blob/main/model.py) After placing the file in your home directory, run this code in a new file

```
from model import KenlmModel
model = KenlmModel.from_pretrained(“wikipedia”, “en”)
model.get_perplexity(“She was a shriveling bumblebee, and he was a bumbling

banshee, but they accepted a position at Gringotts because of their love for

maple syrup”)
```

## EXERCISE

Try out sentences in different styles and topics to see how the perplexity varies! In particular get the perplexities of these types of text:

* Social media text, like Twitter
* SEO spam
* Text with a lot of slang

Additionally, you can train a KenLM model on your own domain dataset. Sample a portion of your dataset and train the model using the instructions provided in their [Github](https://github.com/kpu/kenlm). You can then take the remaining portion of the dataset, break it into chunks, and calculate the perplexity of each chunk. Which chunks have the highest perplexity? Which chunks have the lowest perplexity? After manually inspecting the results, do you think perplexity sampling is a good measure of quality?

**NOTE**

According to an [analysis of C4](https://arxiv.org/abs/2104.08758), the Internet domain that contributed the largest proportion of text in the dataset was patents.google.com. Over 10 percent of the text from this domain is in fact machine translated, with patents from countries like Japan being translated from Japanese to English. So a significant amount of pre-training data is already not generated by humans!

Propelled by LLM’s, the Internet is slated to see widespread prevalence of AI-generated text. Recognizing whether text was written by a human or an LLM is a non-trivial task, and certainly not feasible at scale. How this would affect future LLM performance is an open research question.

Despite all the data cleaning steps, the resulting dataset is still not going to be perfect at this level of scale. For example, Eleuther AI [reported](https://arxiv.org/abs/2101.00027) that the boilerplate sentence “select the forum that you want to visit from the selection below” occurs 180k times in the Pile.

### Deduplication

So far we have discussed data extraction and cleaning, language identification, and quality filtering. Let’s now explore the most contentious step in the pipeline - deduplication.

We know that web-crawled text is ridden with a lot of duplicates. Duplicates form a non-trivial portion of the training dataset, so any decision taken about them will have a noticeable impact on the ensuing model.

How do we define a duplicate? We will make a distinction between three kinds:

* _Exact Matches_: Two sequences with the same text are exact-match duplicates. They are the easiest to handle.
* _Approximate Matches_: In many cases, there are near-duplicates, where sequences of text are identical except for a few characters. Sometimes these sequences are slightly different only due to HTML text extraction artifacts and other filtering processes.
* _Semantic Duplicates_: Duplicates that semantically convey the same content but using different wordings. This is usually treated as out of scope.

Duplicates can also be categorized based on the granularity at which they occur.

* _Document-level Duplicates_: Duplicate documents are removed during the preparation of most pre-training datasets. However, in some datasets like The Pile, certain subsets (like Wikipedia) are deliberately duplicated, so that they are seen more often by the model.
* _Sequence-level Duplicates_: These are sequences in documents that are repeated across multiple documents. In some cases they can be massively duplicated, like Terms of Service text, copyright notices, website prefaces etc.

#### To Deduplicate or to not Deduplicate

The jury is still out on the effectiveness or lack thereof of deduplication.

There is [evidence](https://arxiv.org/abs/2305.16264) that you can train for four epochs without overfitting. This is equivalent to text being duplicated four times. However, there is still a benefit in removing duplicates that are just boilerplate text and occur thousands of times.

On the other hand, here are a few arguments in support of deduplication:

* A small subset of the pre-training dataset is usually kept aside for validation/test. Deduplication can ensure the removal/reduction of overlap between the train and test sets, which is essential for an unbiased evaluation. Without sequence-level deduplication, there is a high likelihood of overlap of common text sequences in the train and test sets.
* [Anthropic’s work](https://arxiv.org/pdf/2205.10487.pdf) shows a surprising double descent phenomenon - this means that data that is duplicated only a few times doesn’t negatively impact model performance too much, data that is duplicated too many times doesn’t negatively impact model performance too much, but in the distribution of duplication frequency, there is a peak in the middle where the damage is maximum.
* Removing duplicate sequences reduces the overall size of the training dataset. However, [Lee et al.](https://arxiv.org/abs/2107.06499) show that this does not affect the perplexity of the model. Thus, the model can be trained for a shorter period yet with the same benefit.
* Deduplication can also reduce the tendency of the model to memorize its training data. Memorization is closely linked to model overfitting, and thwarts the ability of the model to generalize. While there are many ways to quantify memorization, we will focus on _memorization by generation_, where a model is said to have memorized a sequence if it is capable of generating it verbatim. [Lee et al.](https://arxiv.org/abs/2107.06499) have shown that models trained on datasets that have been deduplicated at the sequence level generate ten times less verbatim training data.

**TIP**

One advantage of using models trained on publicly available datasets is that you can search through the dataset to see if the text generated by the model exists verbatim in the dataset. For example, the [ROOTS search tool](https://huggingface.co/spaces/bigscience-data/roots-search) can be used to test generations from the BLOOM model, which was trained on ROOTS.

## SECURITY VULNERABILITIES IN LLMS DUE TO MEMORIZATION

Memorization makes language models vulnerable to security and privacy attacks. Two demonstrated types of attacks are:

* _Membership inference attack_: With just closed-box access to a model, a membership inference attack enables an attacker to determine if a sequence of text has been used to train the model or not.
* _Training data extraction attack_: With just closed-box access to a model, the attacker can prompt the model to generate memorized sensitive information. A naive example involves prompting the model with the text ‘Suhas Pai’s phone number is’ and asking the model to provide the continuation, with the hope that it has memorized Suhas’s number.

[Carlini et al.](https://arxiv.org/abs/2012.07805) show that larger models memorize more easily and thus are most susceptible to these types of attacks. However, it is hard to estimate how much data is memorized by the model, as some memorized data is output by the model only when prompted with a delicately prepared prefix of a longer length. This makes models harder to audit for privacy guarantees.

[Figure 1-4](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch01.html#privacy-attacks-against-llms) demonstrates the flow of a rudimentary training-data extraction attack.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/figure107.png" alt="Privacy attacks" height="119" width="600"><figcaption></figcaption></figure>

**Figure 1-4. Privacy Attacks against LLMs**

**TIP**

Deduplication is computationally intensive, especially when it comes to removing near-duplicates. Some of the efficient algorithms used include MinHash, SimHash, Suffix Array etc.

### Removing PII (Personally Identifiable Information)

While deduplication can reduce the likelihood of the model memorizing training data, it is by no means a panacea to the memorization problem. Even information that appears only once in the training set could potentially be memorized (and leaked). While a lot of content in the training data is innocuous (Terms of Service text) and perhaps even desirable to memorize (factual information, like the capital of Canada), memorization of personally identifiable information (PII) is a major concern.

Let us see what PII entails. The formal definition from [Cornell Law](https://www.law.cornell.edu/cfr/text/2/200.79) is -

> Information that can be used to distinguish or trace an individual’s identity, either alone or when combined with other personal or identifying information that is linked or linkable to a specific individual.

Based on this definition, non-PII can become PII when another piece of information becomes public, which when combined with the non-PII can be used to uniquely identify an individual.

The legal definition of PII varies by jurisdiction. For example, the [GDPR](https://gdpr-info.eu/issues/personal-data) (General Data Protection Regulation) in Europe, says

> Protection should be extended to anything used to directly or indirectly identify a person (or data subject). This may be extended to include characteristics that describe “physical, physiological, genetic, mental, commercial, cultural, or social identity of a person.

Most open-source models are trained on publicly available datasets. These datasets might contain PII, but one might be tempted to say ‘well it is already out in the open, so there is no need for privacy protection’. This argument overlooks the importance of consent and discoverability controls. For instance, I might have shared my PII on my blog which resides in an obscure corner of the Internet and is not easily discoverable through search engines, but if it ends up being added to a pre-training dataset, it suddenly brings this data into the spotlight, without my consent. This concept is called _contextual integrity_ - data should only be shared in the original context in which it was shared.

So ideally, we would like to _detect_ PII in the dataset, and then _remediate_ it in some fashion, so that the PII is no longer present in the training data or at least not memorizable. The presence of _public-figure PII_ adds a layer of complexity to this problem. We would like our model to be able to answer factual questions about public figures like their birth date accurately. The privacy expectations for public figures is lower, showcasing how the values of transparency and openness clash with privacy. Determining who is a public figure and what level of privacy they are entitled to is a complex social and technical challenge.

Data that is considered private includes names, addresses, credit card data, government IDs, medical history and diagnosis data, email IDs and phone numbers, identity and affinity groups the person belongs to (religion, race, union membership), geolocation data and so on.

Attacks can be either targeted or untargeted. In an untargeted attack, the attacker just generates a large body of text using the model, and then runs a membership inference attack to determine text within it that is most likely to be memorized. In a targeted attack, the attacker attempts to recover personal information about a particular individual or a group of individuals. Targeted attacks are more difficult to execute, because while language models are good at memorization, they are bad at _association_ - for instance, identifying that an email ID belongs to a person.

## EXERCISE

Use the instructions in the [ReadMe to run this code](https://github.com/jeffhj/LM\_PersonalInfoLeak) for analyzing privacy attacks on LLMs. It goes without saying, but please do not use this in the real world! Running the code and observing the outputs will give you an understanding of the limitations of this type of attack, and the type of data that is typically memorized by an LM.

Additionally, you can play around with [Google’s Training Data Extraction Challenge](https://github.com/google-research/lm-extraction-benchmark) and make a submission!

**NOTE**

Language models are also susceptible to training data poisoning attacks. Since a large portion of training data is sourced from web-crawled text, bad actors have an opportunity to influence the content of the training set. [Tramer er al.](https://arxiv.org/pdf/2204.00032.pdf) have shown that one can poison less than 0.1 percent of the training set with data whose effect is to make it easier for other data in the training set to leak more easily.

As LLMs increasingly get used as search engines, the demand for LLM SEO will soon crop up. For example, a company could write content on their web sites in a manner that makes it more likely to be chosen in a pre-training dataset creation process that uses perplexity filtering.

Most pre-training datasets have undergone little to no PII remediation. The Privacy working group (of which I was the co-lead) of the Big Science project that trained the BLOOM model developed a pipeline for PII detection and remediation, which we will discuss next.

[Figure 1-5](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch01.html#PII-processing-pipeline) shows a typical PII processing pipeline.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/Fig1-5_highres.png" alt="PII Processing Pipeline" height="552" width="600"><figcaption></figcaption></figure>

**Figure 1-5. PII Processing pipeline**

#### PII Detection

The task of PII detection is similar to the NLP task of Named Entity Recognition, introduced in Chapter 1. However, not all named entities constitute PII. For our task we determined the PII tags to be - PERSON, AGE, NORP (nationality, race, religion, political party affiliation, socio-economic class, union membership), STREET\_ADDRESS, CREDIT\_CARD, GOVT\_ID, EMAIL\_ADDRESS, USER\_ID, PUBLIC\_FIGURE.

We used the PUBLIC\_FIGURE tag to identify information about public figures, since we didn’t want to filter them out. We also assigned fictional characters this tag.

Some of the structured tags in this list like emails and government IDs can be identified using regular expressions. For other tags, we annotated datasets which could then be used to train Transformer-based NER-like models. Interestingly, we observed a very high degree of inter-annotator disagreement (same example being annotated differently by different people) that underscored the cultural nuances of the definition of privacy and what constitutes personal information.

Here is the [regular expression](https://github.com/bigscience-workshop/data\_tooling/blob/master/pii-manager/src/pii\_manager/lang/en/us/social\_security\_number.py) to detect SSN (U.S Social Security Numbers):

```
ssn_pattern = r"(?!000|666|333)0*(?:[0-6][0-9][0-9]|[0-7][0-6][0-9]|

[0-7][0-7][0-2])[-\ ](?!00)[0-9]{2}[-\ ](?!0000)[0-9]{4}"
```

Note that detection is not the same as validation. Not all 9 digit numbers of the form XXX—​XX-XXXX are SSNs! Validation is the process of checking if a sequence of characters maps to a valid identifier. For example, the Canadian equivalent of SSN, the SIN (Social Insurance Number) contains a checksum digit which can be used to validate it.

```
from stdnum.ca import sin
sin_pattern = re.compile(r"\d{3}[-\ ]\d{3}[-\ ]\d{3}", flags=re.X)
for match in sin_pattern.findall(text):
    if sin.is_valid(match):
         print(match)
```

The is\_valid() function uses the [Luhn checksum algorithm](https://en.wikipedia.org/wiki/Luhn\_algorithm) to validate if the sequence of digits maps to a valid SIN. The same algorithm is also used to validate credit cards. Here is the [regex](https://github.com/bigscience-workshop/data\_tooling/blob/master/pii-manager/src/pii\_manager/lang/any/credit\_card.py) for detecting credit card numbers.

```
from stdnum import luhn
cc_base_pattern =  r"\b \d (?:\d[ -]?){14} \d \b"
cc_full_pattern = r"""4[0-9]{12}(?:[0-9]{3})? |
                      (?:5[1-5][0-9]{2}|222[1-9]|22[3-9][0-9]|2[3-6][0-9]{2}|27[01][0-9]|

                      2720)[0-9]{12} |
                      3[47][0-9]{13} |
                      3(?:0[0-5]|[68][0-9])[0-9]{11} |
                      6(?:011|5[0-9]{2})[0-9]{12} |
                      (?:2131|1800|35\d{3})\d{11}"""
```

The regular expression for detecting email address is

```
email_pattern = r"[\w\.=-]+ @ [\w\.-]+ \. [\w]{2,3}"
```

## EXERCISE

These regular expressions were run on the ROOTS dataset. How effective were they in detecting PII? Find out using the [ROOTS search tool](https://huggingface.co/spaces/bigscience-data/roots-search). If you search for ‘gmail.com’, you will find that all entries in the search results have been successfully redacted. Alter the spelling a little and see if it still holds true. Can you improve the regular expession?

**NOTE**

Removing structured PII data while keeping the number of false positives low is hard enough, but detecting and remediating unstructured data is even harder. Due to the complexity of this task and the uncertainty about its impact on the resulting model performance, we decided to not run the Transformer model based PII pipeline over the ROOTS dataset for training the BLOOM model

#### PII Remediation

Once PII has been detected, it can be remediated. [Figure 1-6](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch01.html#PII-remediation-options) depicts one of the remediation schemes.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/figure117.png" alt="PII Remediation Options" height="171" width="600"><figcaption></figcaption></figure>

**Figure 1-6. PII Remediation Options**

Here is a non-exhaustive list of remediation options:

* _Replace by a special token_: For example, a valid phone number can be replaced by the string \<phone number>
* _Replace with a random token of the same entity type_: For example, replace the name ‘Clarietta Richards’ with ‘Natasha Bridges’, or any other name.
* _Replace with a shuffled token_: Entities detected across the dataset can be shuffled.
* _Remove entire document/data source_: If the amount of PII detected in a single document or data source is higher than a specific threshold, it is probably best to remove it. For example, pastebin.com is said to contain a lot of inadvertently placed PII, and is recommended to be not included in training datasets.

Each of these techniques can have a varied effect on downstream performance of the model. How does replacing tokens affect training perplexity? Do downstream tasks like Named Entity Recognition get negatively affected when tuned on the resulting model? How does replacement by special tokens compare to replacement with random tokens? This is a relatively underexplored topic and all these questions are still open.

[Faker](https://github.com/joke2k/faker) is an excellent library for facilitating random token replacement. It supports random token generation for a variety of PII types including names, addresses, credit card numbers, phone numbers etc. One danger in using random tokens is that the replacement process can alter the demographic distribution of the dataset - for example, if the replacement names were all or mostly Anglo-Saxon names. Faker has localization support to enable replacement with fake data from the same geography/culture. Let’s explore the library in more detail.

```
from faker import Faker
fake = Faker(‘en_IN’)   # Indian locale
Faker.seed(0)
for i in range(5):
   print(fake.aadhaar_id)
```

This code generates 12 digit fake Aadhaar ID’s, which are the Indian equivalent of Social Security Numbers. Note that the generated IDs are all invalid, but still follow the same format. Similarly,

```
for i in range(5):
   print(fake.address)
```

generates fake but representative addresses for the selected locale.

**NOTE**

Removing PII from training datasets is only one of several solutions to prevent data leakage from models. One promising technique is [differential privacy](https://www.oreilly.com/library/view/hands-on-differential-privacy/9781492097730/), which introduces randomness in the inputs or outputs to provide theoretical guarantees for privacy preservation. In neural networks, differential privacy is implemented using the [DP-SGD](https://arxiv.org/abs/1607.00133) algorithm, which involves gradient clipping and noise addition at the end of each update. However, differential privacy significantly slows down training, negatively affects model performance, and disproportionately impacts minority groups in the dataset in terms of model utility degradation. Apart from differential privacy, other methods include adversarial training, [model unlearning](https://arxiv.org/pdf/1912.03817.pdf), [retroactive censoring, and ‘memfree’ decoding](https://arxiv.org/abs/2210.17546).

### Test Set Decontamination

Test set decontamination is a crucial data preprocessing step that helps improve LLM evaluations. A pre-training dataset is said to be contaminated if it contains data from the benchmark test sets used to evaluate its performance. Contamination can happen if the test datasets were constructed from web text, or if the dataset was uploaded on the Web after creation. There are two types of contamination:[1](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch01.html#id142)

* _Input and Label contamination_: In this setting, both the questions (inputs) and answers (target labels) exist in the pre-training dataset. Heard about how GPT-4 can solve all kinds of exams? While the creators of GPT-4 did spend a lot of effort on removing data contamination, in practice it is really hard to remove everything.
* _Input contamination_: In this setting, only the inputs are present in the pre-training dataset but not the target labels. We will describe the effects of input contamination and how we can leverage it for positive use in Chapter 8 and 9.

[Open AI](https://arxiv.org/abs/2005.14165) addressed test set contamination in GPT-3 by finding 13-gram overlaps between text in the test/validation set and the train set, and removing 200 characters before and after the matched texts.

## DATASET ORDERING

After all data pre-processing stages have been completed, the training process can commence. The order in which the data is fed to the model does matter. The area of study to determine the most optimal order is called curriculum learning. To our knowledge, most models do not go beyond some simple ordering heuristics.

One technique is to start the training with shorter training sequences and then gradually increase the sequence lengths. This can be done by either truncating initial sequences to fit a certain length, or by simply reordering the dataset so that shorter sequences are ordered first.

[Researchers](https://openreview.net/pdf?id=y7CNId2RnV) have also experimented with introducing more common words to the model first, by replacing rarer words occurring in early training examples with their part-of-speech tag or with hypernyms (for example, the hypernym of magenta is color).

Now that we have discussed all the important data collection and pre-processing steps for preparing a pre-training dataset, let us see how individual datasets differ in terms of the preprocessing steps they have undergone.

**TIP**

Big Science has developed a visualization tool that helps you understand the effect of various preprocessing functions on the pre-training dataset. Use the [Process Pipeline Visualizer](https://huggingface.co/spaces/bigscience-data/process-pipeline-visualizer) to sequentially run through the preprocessing pipeline yourself!

[Table 1-2](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch01.html#table2-pretraining-datasets) provides a list of the popular pre-training datasets, and the kind of preprocessing they went through.

| Name        | Extraction and Cleaning                                                                                                           | Quality Filtering                                                   | Deduplication                                                                       | Language Identification | Models trained with this dataset                     |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ----------------------- | ---------------------------------------------------- |
| C4          | Remove pages containing word in blocklist, remove code, remove short lines and pages                                              | -                                                                   | Deduplication of 3-sentence spans                                                   | langdetect              | T5, Flan-T5, UL2, Llama etc                          |
| The Pile    | justext library for text extraction                                                                                               | fasttext classifier                                                 | Document level, with MinHashLSH                                                     | pycld2                  | GPT-Neo/X, GPT-J, Cerebras-GPT, StableLM, Pythia etc |
| CCNet       | -                                                                                                                                 | Perplexity filtering                                                | Paragraph level deduplication                                                       | fasttext                | F                                                    |
| RedPajama   | Ccnet pipeline                                                                                                                    | Classifier distinguishing between Wikipedia text and random C4 text | Paragraph level deduplication (for Common Crawl)                                    | fasttext                | Red Pajama-INCITE, MPT                               |
| CleanPajama | low-length filter, NFC normalization                                                                                              | -                                                                   | MinHashLSH                                                                          | -                       | -                                                    |
| RefinedWeb  | URL filtering by blocklists, trafilatura library for text extraction, repetitive content removal                                  | -                                                                   | Fuzzy document level deduplication with MinHash, Exact sequence-level deduplication | fasttext                | Falcon                                               |
| ROOTS       | removal of documents with low ratio of closed class words, high ratio of blocklist words, high ratio of character/word repetition | Perplexity filtering                                                | SimHash, Suffix Array                                                               | fasttext                | BLOOM                                                |

## Leveraging Pre-training Dataset Characteristics

How well do LLM’s do on arithmetic and logical reasoning? The prospects of a very large number of use cases depend on the answer being a positive one. We will investigate this question in more detail in Chapter 11.

But for now, I would like you to dwell a moment on this fascinating observation - there is a correlation between a model’s performance on a given input example and the pre-training corpus frequency of the words present in that input.

[Razeghi et al.](https://arxiv.org/abs/2202.07206) observed this with the GPT-J model - when asked arithmetic questions like addition, subtraction, multiplication etc, the model gets it right sometimes, and wrong other times.If you plot a graph of pre-training frequencies of the numbers versus the performance for arithmetic operations using those numbers, there is a clearly visible trend. The more frequent a number appears in the pre-training dataset, the better the model is at arithmetic operations involving that number.

The effect is most drastic for multiplication tasks. As shown in [Figure 1-7](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch01.html#plot-average-accuracy), the model is more correct at multiplication operations involving the number 24 than ones involving the number 23, and the frequency of the numbers in the dataset show a large difference between the term frequency for these numbers.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/figure118.png" alt="Avg.Accuracy vs Term Frequency" height="413" width="600"><figcaption></figcaption></figure>

**Figure 1-7. Plot of average accuracy plotted against term frequency, using the Snoopy tool. Image taken from Razeghi et al.**

The authors investigate this phenomenon using three types of frequencies. Consider the input

```
Q: What is 40 times 51? A:
```

The frequencies calculated are

1. Unigram frequency: For example, the number of times the number ‘40’ occurs in the dataset
2. Input term co-occurrence: Two input terms co-occurring within a window size of 5. For the current example, it is (40, 51)
3. Input and output term co-occurrence: Two input terms and the output term co-occurring within a window size of 5. For the current example, it is (40, 51, 2040)

The unigram frequencies alone cause noticeable performance gaps. This phenomenon can be replicated across other types of tasks and datasets as well. This means that Open AI’s technique of finding 13-gram overlaps between text in the training set and in benchmark sets isn’t enough to eliminate input contamination.

If your task is well-defined, doesn’t have drastic data drifts, then input contamination may not really be such a bad thing. You can then leverage frequency statistics to design inputs to the model that are more likely to give the right answer!

**TIP**

You can explore this phenomenon on your own by using the [Snoopy](https://nlp.ics.uci.edu/snoopy) tool. Snoopy is a tool built by [Radeghi et al.](https://arxiv.org/abs/2202.07206) for analyzing the impact of pre-training term frequencies on model performance. It uses The Pile, the dataset used to train most open-source models including GPT Neo-X, for analysis. You can experiment with a variety of benchmark tasks.

## EXERCISE

Using the Snoopy tool, try out different benchmark datasets from the drop down dataset and explore the effect of term frequency (both unigram and co-occurrence) on model accuracy. For which tasks is this phenomenon least prevalent? Why could it be?

## Bias and Fairness Issues in Pre-training Datasets

A multitude of ethical questions arise during the productization of large language models. The existence of significant bias and fairness issues in these models often lead to a no-ship condition for a large number of use cases. We will give these issues their due coverage in Chapter 12. For now, in this section we will go through some bias and fairness issues specifically related to the collection and filtering of pre-training data.

The scale of data that LLMs are fed with means that they are not just constructing models of language, but also of the world we inhabit. This gives rise to the question - ‘Do we want to model the world the way it is or do we want to model the world the way we would like it to be?’ The Internet is filled with hate, violence, and abusive language and is often used as an outlet for humanity’s worst impulses. The text in it implicitly encodes long existing biases against groups of people. For example, in The Pile, an [analysis](https://arxiv.org/abs/2101.00027) of word co-occurrence statistics shows the word ‘radical’ co-occurs with the word ‘Muslim’ substantially more than it does for other religions.

The phenomenon of _bias amplification_ makes these problems all the more critical. It has been shown that large language models [amplify the biases](https://arxiv.org/abs/2201.11706) that are encoded in their pre-training data - they make biased predictions against groups of people at higher rates than what the training data statistics would suggest.

So, can we ‘fix’ our training data such that we can model a world that encodes our values and principles which downstream applications will inherit? There is substantial debate in the research community around this. Opponents argue it is hard to identify and fix all societal biases encoded in the data since there are so many dimensions of bias that intersect in complex ways. Values are not universal and model providers would like to be value-neutral in order to cater to all sections of society

However, as Anna Rogers describes in her [paper](https://aclanthology.org/2021.acl-long.170.pdf), this question is already moot. Data curation is already happening, whether we like it or not, and the values and interests of model providers are already being encoded into the models. For example, only a small proportion of available data is ‘selected’ to be part of the pre-training set. This selection process is not value-neutral, even if one might explicitly not think in terms of them.

For example, Wikipedia is one of the more popular datasets used in training LLMs. While this might be a no-brainer to include, let’s explore the implications. Wikipedia is edited by volunteers, a very large proportion of them being men. Since the determination of whether a topic is reputable enough to deserve a Wikipedia page rests with the editors who are largely made up of men, we see disparities like obscure male football players from lower level leagues getting their own pages while a disproportionate number of biography articles about women are slated for deletion.

Similarly, the highly influential WebText dataset is sourced from Reddit outbound links. Reddit is a predominantly male site, with [74% of users](https://blog.gitnux.com/reddit-user-statistics/) being men. Naturally, links posted on Reddit are more likely to be catered to male interests.

Bias can also be introduced during the data filtering stages. Earlier, we noted that keyword lists are often used to filter out pornographic material and abusive text. However, using a naive keyword list is a lazy approach that not only has problems with effectiveness (false negatives), but also inadvertently causes [disproportionately](https://arxiv.org/abs/2104.08758) filtering out positive text written by or about minority communities, as well as text written in dialects like African-American English and Hispanic-aligned English. The fact that words in English have multiple senses has resulted in certain documents about breastfeeding being filtered out of the C4 dataset.

Overall, whether a word is hateful, abusive, or toxic depends on the social context, the intentions of the reader, and the intended audience. Keyword based methods simply do not capture this nuance. The question of whether it is more effective to handle these issues at the pre-training stage or further downstream is an open area of research. We will explore techniques that can be employed downstream in Chapter 12.

The authors of the [Pythia](https://arxiv.org/pdf/2304.01373.pdf) model experimented by replacing masculine pronouns with feminine ones for the last 7 percent of training tokens and noticed a ‘de-biasing’ impact on downstream tasks.

We will further explore bias, fairness, and safety issues and how to integrate these values while designing LLM applications in Chapter 11.

## Summary

In this chapter, we outlined the key ingredients of a language model - the pre-training data, the vocabulary and tokenizer, the language objective, and the model architecture. We walked through the steps involved in creating a pre-training dataset in detail, including language identification, text extraction and cleaning, quality filtering, deduplication, PII removal, and test set decontamination. We also provided a list of commonly used pre-training datasets and the steps taken for pre-processing each of them.

Now that you have a good idea about the data side of LLMs, it is time to explore the model side. In the next chapter, we will provide details on the remaining ingredients of the language model - the vocabulary and tokenizer, learning objective, and model architecture.

[1](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch01.html#id142-marker) from A Case Study on the Colossal Clean Crawled Corpus, Dodge et al., EMNLP 2021
