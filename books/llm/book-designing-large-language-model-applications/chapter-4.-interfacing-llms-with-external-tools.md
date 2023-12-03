# Chapter 4. Interfacing LLMs With External Tools

## Chapter 4. Interfacing LLMs with External Tools

## A NOTE FOR EARLY RELEASE READERS

With Early Release ebooks, you get books in their earliest form—the author’s raw and unedited content as they write—so you can take advantage of these technologies long before the official release of these titles.

This will be the 7th chapter of the final book. Please note that the GitHub repo will be made active later on.

If you have comments about how we might improve the content and/or examples in this book, or if you notice missing material within this chapter, please reach out to the author at [mcronin@oreilly.com](mailto:mcronin@oreilly.com).

In order to effectively harness the power of LLMs in your organization, they have to be integrated into the existing data and software ecosystem. Unlike traditional software components and data stores, LLMs can generate autonomous actions to interact with other components of the ecosystem, thus bringing a degree of flexibility never seen before in the world of software. This flexibility unlocks a whole host of use cases that were previously considered impossible.

There is another reason why we need LLMs to interact with software and external data. As we know too well, LLMs are not yet a mature technology. In Chapter 1, we devoted an entire section to discuss limitations of current LLMs. To recap some key points:

* Since it is expensive to retrain LLMs or keep them continuously updated, they have a knowledge cutoff date and thus possess no knowledge of more recent events.
* Most LLMs perform poorly at mathematical operations beyond rudimentary arithmetic.
* They can’t provide factuality guarantees or accurate citations of their outputs.
* Feeding them your own data is a challenge - fine-tuning is non-trivial and in-context learning is limited by the length of the effective context window. (Revisit Chapter 6. for details on why long-context models are not the panacea to the limited-context problem)

As we have been noticing throughout the book, the consolidation effect is leading us to a future (unless we hit a technological wall) where many of the aforementioned limitations might be addressed within the model itself. But we need not necessarily wait for that moment to arrive - many of these limitations can be addressed today by offloading the tasks/subtasks to external tools.

In this chapter, we will describe the various LLM interaction paradigms and provide guidance on how to adopt them in your application. Broadly speaking, there are two types of external entities that LLMs need to interact with - data stores and tools (software/models). We will describe each of them in detail and showcase how they can be used in tandem to build powerful applications. We will show how to make the best use of libraries like LangChain and LllamaIndex, which have vastly simplified LLM integrations. We will also push the limits of what today’s LLMs are capable of, by demonstrating how they can be deployed as an agent that can make autonomous decisions.

## LLM Interaction Paradigms

Suppose you have a task you want the LLM to solve. There are several possible ways in which this can pan out.

1. The LLM uses its own memory and capabilities encoded in its parameters to solve it.
2. You feed the LLM all the context it needs to solve the task within the prompt, and the LLM uses the provided context and its capabilities to solve it.
3. The LLM doesn’t have the requisite information or skills to process this task, so you update the model parameters (fine-tuning etc., as discussed in detail in Chapters 5 and 6) so that it is able to develop the skills and knowledge to solve it.
4. You don’t know apriori what context is needed to solve the task, so you use mechanisms to automatically fetch the relevant context and insert it into the prompt. (The Passive approach)
5. You provide explicit instructions to the LLM on how to interact with external tools and data stores in order to solve your task, which the LLM follows. (The Explicit approach)
6. The LLM breaks down the task into multiple subtasks if needed, and interacts with its environment to gather the information/knowledge needed to solve the task, and delegates subtasks to external models and tools when it doesn’t have the requisite capabilities to solve that subtask. (The Agentic approach)

As you can see, options 4-6 involve the LLM interacting with its environment. Let’s go through the three interaction paradigms (Passive, Explicit, Agentic) in detail.

### The Passive Approach

[Figure 4-1](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch04.html#passive-interaction) shows the typical workflow of an application that involves an LLM passively interacting with a data store.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/book5.7.png" alt="Passive Interaction" height="218" width="600"><figcaption></figcaption></figure>

**Figure 4-1. An LLM passively interacting with a data store.**

A large number of use cases involve leveraging LLMs to make use of your own data. Examples include building a question answering assistant over your company’s internal knowledge base that is spread over a bunch of Notion documents, or an airline chatbot that responds to customer queries about flight status or booking policies.

In order to allow the LLM to access external information, we need two types of components - _retrieval engines_ and a _data stores_. A retrieval engine can be powered by an LLM itself, or it can be as simple as a keyword matching algorithm. The data store(s) can be a repository of data, like a database, knowledge graph, vector database, or even just a collection of text files. Data in the data store is represented and indexed in a manner that makes retrieval more efficient.

When a user issues a query, the retrieval engine uses the query to find the documents or text segments that are most relevant to answering this query. After ensuring that it fits into the context window of the LLM, it is fed to the LLM along with the query.The LLM is expected to answer the query given the relevant context provided in the prompt.

We will discuss various forms of data stores and retrieval mechanisms later in the chapter. We call this a passive interaction approach because the LLM itself is not actively involved in the selection of the context. This paradigm is often used for building QA assistants or chatbots, where external information is required to understand the context of the conversation.

### The Explicit Approach

[Figure 4-2](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch04.html#explicit-approach) demonstrates the Explicit approach to interface LLMs with external tools.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/book5.3.png" alt="Explicit Approach" height="216" width="600"><figcaption></figcaption></figure>

**Figure 4-2. The Explicit interaction approach in action.**

In this approach, we provide the LLM with explicit instructions on how and when to invoke external information and tools. The LLM just follows the instructions mentioned in the query. This approach is recommended when the interaction sequence is fixed, limited and preferably only a single step. An example would be an AI data analyst asistant where you provide queries in natural language and ask the LLM to generate SQL code that can be run over a database.

**NOTE**

Keep in mind that LLMs don’t have session memory, i.e. they are stateless. Every query is a brand new interaction with the LLM. To simulate session memory, you need to feed all the previous interactions with the LLM in the context of your next query. Yes, this is highly inefficient. Yes, this means you will consume a lot of tokens. But this is what we have to work with.

### The Agentic Approach

[Figure 4-3](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch04.html#agentic-approach) shows how we can turn an LLM into an autonomous agent that can solve complex tasks by itself.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/book5.1.png" alt="Agentic Approach" height="224" width="600"><figcaption></figcaption></figure>

**Figure 4-3. A typical LLM Agent workflow**

The agentic approach, or the Holy Grail approach as I would like to call it, turns an LLM into an autonomous agent that can solve tasks on its own. Here is a typical workflow of an agent:

1. The user formulates their requirements in natural language, optionally providing the format in which they want the LLM to provide the answer.
2. The LLM decomposes the user query into manageable subtasks
3. The LLM synchronously or asynchronously solves each subtask of the problem. Where possible, the LLM uses its own memory and knowledge to solve a specific subtask. For subtasks where the LLM cannot answer on its own, it chooses a tool to invoke from a list of tools available to it. Where possible, the LLM uses the outputs from solutions of already executed subtasks as inputs to other subtasks.
4. The LLM synthesises the final answer using the solutions of the subtasks, generating the output in the requested output format.

This paradigm is general enough to capture just about any use case. It is also a risky paradigm - we are giving the LLM too much responsibility and agency. At this juncture, I would not recommend using this paradigm for any critical applications.

**NOTE**

Why am I calling for caution in deploying agents? Oftentimes, humans underestimate the accuracy requirements for applications. For a lot of use cases, getting right 99% of the time is still not good enough, especially when the failures are unpredictable. The 99% problem is also the one plaguing self-driving cars from getting on the road. This doesn’t mean we can’t deploy autonomous LLM agents at all - we just need clever product design that can shield the user from their failures. We will discuss this more in Chapter 14.

To better understand the agentic paradigm, let me share an example query for the agent I am developing at my company, which operates in the financial domain. Consider this question:

> _Who was the CFO of Apple when its stock price was at its lowest point in the last 10 years?_

Here is how the LLM agent can answer this question. Each item in the numbered list corresponds to a step in the _chain_, the sequence of actions it takes. The system prompt contains a list of available tools and external data stores and their descriptions.

1. First, it decomposes the task into multiple subtasks.
2. To calculate the date range, it needs the current date. If this is not included in the system prompt, it generates code for returning the system time, which is then executed by a code interpreter.
3. Using the current date, it finds the other end of the date range by executing a simple arithmetic operation by itself, or by generating code for it. Step 2 and 3 could also be combined into a single program.
4. It finds a database table that contains stock price information in the data store list. It retrieves the schema of the table and inserts it into the prompt and generates a SQL query for finding the date when the stock price was at its minimum in the last 10 years.
5. With the date in hand, it needs to find the CFO of Apple on that date. It can generate code to call a search engine API to see if there is an explicit mention of the CFO on that particular date.
6. If the search engine query fails to provide a result, it finds a financial API in its tools list and retrieves and inserts the API documentation into its context. It then generates code for an API call to retrieve the list of CFOs and their tenure durations.
7. Finally, it uses its arithmetic reasoning skills to find the duration that matches the date of the lowest stock price, and retrieves the corresponding CFO.
8. It generates the output text with the answer. If there is a requested format, it tries to adhere to that.

As you can see, this is a very powerful paradigm but it is also a very complicated chain involving several LLM calls. Note that it takes several Google searches for even a financial domain expert to find the answer to this question. There are several opportunities for the LLM to fail in this chain, and the earlier it fails the harder it is to recover from it.

Task decomposition is a particularly challenging problem and will be explored further in Chapter 13. However, task decomposition is only needed for the Agentic approach. Let’s now explore how to facilitate interaction between LLMs and external data stores.

**NOTE**

Projects like [BabyAGI](https://github.com/yoheinakajima/babyagi), [AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT), [HuggingGPT](https://github.com/microsoft/JARVIS) (also called Microsoft JARVIS), are notable demos of autonomous LLM agents. Unsurprisingly, none of them are stable enough to be used in production as of now. However, as of today the agentic approach can still be production-ready in limited use cases where accuracy and latency requirements are lax.

## EXERCISE

The accompanying Github for the book contains an AutoGPT-style implementation. Use your GPT-4 key to run the agent code and explore the limitations and potential of autonomous LLM agents. Be careful and set up Open AI billing alerts - a single task might consume a lot of tokens! Try asking the agent ‘Which football team had the 5th highest number of goals scored in the Premier League during the year that Jackie Chan turned 60?' and debug the actions taken by it.

## Retrieval

Now that we have seen how the three interaction paradigms work, let’s discuss Retrieval, a common mode of interaction.

External data can be of just about any type - text files, database tables, knowledge graphs, and so on. Data can range from propreitary domain-specific knowledge bases to intermediate results and outputs generated by LLMs.

As shown in [Figure 4-2](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch04.html#explicit-approach), a typical solution is to calculate some kind of similarity measure between the user query and the data segments in the data store to find the segments that most _match_ the user query i.e. provides the most relevant context that can be used to satisfy the user query. This process is called _retrieval_. The retrieval function often returns a ranked list of results in order of relevance rather than a single result. This process is called _text ranking_. This context is then fed into the LLM prompt along with the user query, and the LLM uses the information provided in the context to answer the user query. This two step-process has traditionally been called the _retriever-reader_ framework.

While structured data can live in databases, unstructured data needs to be first processed in order to make it amenable for retrieval. This usually involves parsing text from the document, splitting it into manageable chunks, associating metadata with each segment, storing a representation of it, and indexing it for easy access.

**TIP**

Unstructured text needs to be split into manageable chunks to facilitate effective retrieval and to allow you to insert matching chunks into the LLM context window. Chunks can be as short as individual sentences, but can also be paragraphs, sections, or even documents, with ideally each chunk containing text about a semantically coherent topic. Make the best use of your knowledge of the document structure to inform the splitting process, or even run topic models on your documents apriori to splitting. If there aren’t hard topic boundaries in the source document, then ensure some overlap between contiguous chunks. Pay more attention to this process - I have seen many retrieval projects fail because the textual units were not well-defined.

If it makes sense for your use case to have the units of text as sentences, NLTK’s Punkt tokenizer is a tried and tested tool for tokenizing text into sentences. Note that sentence tokenization is not a trivial task especially if you have domain-specific text. Naive splitting on end marks (periods, question marks and abbreviations) can only get you so far; for example, abbreviations play spoil sport, among others. You can train the Punkt tokenizer unsupervised over a large body of your target text to ensure it learns your domain-specific rules, as well as provide explicit rules and exceptions yourself. The accompanying Github repo to this book contains one such example. Other tools for sentence tokenization include [spaCy](https://spacy.io/usage), [Stanza](https://stanfordnlp.github.io/stanza/tokenize.html), and [ClarityNLP](https://claritynlp.readthedocs.io/en/latest/developer\_guide/algorithms/sentence\_tokenization.html).

### Retrieval Techniques

Which retrieval technique should you use? The answer depends on the following considerations

* The expected nature of user queries (how complex and abstract they can be)
* The expected degree of vocabulary mismatch between user queries and target documents
* Latency and compute limitations
* The metrics to optimize for (precision/recall/NDCG etc)

Depending on the nature of user queries, keyword matching/probabilistic retrieval techniques like [BM25](https://en.wikipedia.org/wiki/Okapi\_BM25) can be a very strong baseline and can potentially even be _good enough_ for your application. To get around the rigidity of having to match the exact keywords, query-expansion and document-expansion techniques can be employed, which we will discuss later in this chapter. In recent times, embedding based methods (bi-encoders) have become extremely popular.

The retrieval process can be broken into a two-stage or multi-stage process, where the initial stages retrieve a list of chunks that are deemed relevant to the query, followed by one or more _reranking_ stages that takes the list of chunks and sort them by relevance. The reranker is generally a more complex model, usually an encoder or encoder-decoder language model like BERT or T5, that we would generally find expensive to run over the entire dataset. (which is why we don’t use it for the initial retrieval stage).

Let’s go into the different retrieval techniques in detail.

**NOTE**

The IR (information retrieval) research field has been studying these problems for a long time. Now that retrieval is more relevant than ever in the field of NLP, I am noticing a lot of efforts to reinvent the wheel rather than reusing insights from the IR field. For insights in retrieval research, check out papers from leading IR research conferences like SIGIR, ECIR, TREC etc.

### Keyword Match and Probabilistic Methods

There are several traditional methods and frameworks that can be used to perform the first-stage (or depending on the use case, the entirety) of the retrieval process. Lucene/ [ElasticSearch](https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-similarity.html) supports Tf-IDf (Term frequency - inverse document frequency), BM-25 (the current default in ElasticSearch 8.9), DFI (Divergence from Independence), DFR (Divergence from Randomness), IB (Information-based), Dirichlet Similarity, and Jelinek Mercer Similarity. Each of these measures has several tunable parameters. For more insight on these techniques and how to select the parameter values, check out this [video](https://www.youtube.com/watch?v=kKocQdYGVJM). The accompanying Github repo to this book also showcases the differences between these methods.

### Embeddings

We introduced the concept of embeddings in Chapter 2. Let’s now see how they can be used for retrieval.

Embeddings are generated for each of the chunks in the data collection. When a new query comes in, an embedding of the query is generated. The query embedding is compared against the chunk embeddings and the ones that have the highest cosine similarity are selected as candidates to be included in the LLM context or to the next stage of the retrieval process. This process is called _semantic search_, since the embeddings capture meaning of the underlying text.

Embeddings can be generated using both open-source libraries and paywalled API’s. [SBERT](https://www.sbert.net/) (sentence-transformers) is a very well known library for generating embeddings, and provides access to embedding models that still performs competitively with respect to the state of the art, even if the model sizes are much smaller.

**NOTE**

There is a distinction between symmetric semantic search and asymmetric semantic search. If the query text is of similar size as the chunk text, then it is symmetric. If the query text is much smaller than the chunk text, as with search engine and question-answering assistant queries, then it is asymmetric. Different models exist for symmetric and asymmetric semantic search. In some models, the query and chunk texts are encoded using separate models.

As a simple illustrative example, consider two chunks of text, each representing a sentence.

```
chunks = ['The President of the U.S is Joe Biden',
'Ramen consumption has increased in the last 5 months']
```

Given the query ‘president of usa’ we can encode the query and the chunks using SBERT.

```
from sentence_transformers import SentenceTransformer, util
sbert_model = SentenceTransformer('msmarco-distilbert-base-tas-b')

chunk_embeddings = sbert_model.encode(chunks, show_progress_bar=True, device='cuda', normalize_embeddings=True, convert_to_tensor=True)

query_embedding = sbert_model.encode(query, device='cuda', normalize_embeddings=True, convert_to_tensor=True)
matches = util.semantic_search(query_embedding, chunk_embeddings, score_function=util.dot_score)
```

The output is:

```
[[{'corpus_id': 0, 'score': 0.8643729090690613},
  {'corpus_id': 1, 'score': 0.6223753690719604}]]
```

**TIP**

If you set normalize\_embeddings to True, it will normalize the embeddings to unit length. This will ensure that you can compute dot product instead of cosine similarity, which is faster. The creators of SBERT provide separate models trained on dot product and cosine similarity and they mention that dot product models tend to prefer longer chunks during retrieval.

The embedding models provided by SBERT are based-on encoder-only models, by mean pooling (averaging) the encoder outputs. The underlying models are BERT, RoBERTa, MPNet etc., and are typically fine-tuned on paraphrasing/question-answering/natural language inference datasets. These models have smaller maximum sequence lengths (typically 512 tokens), and the embedding dimension size is typically 768, so you will only be able to encode a relatively short sequence in a chunk.

**WARNING**

There is no such thing as infinite compression! Embedding sizes are fixed, so the longer your chunk the lesser information can be encoded in its embedding. Managing this tradeoff differs by use case.

Recently, decoder-based embedding models have started gaining prominence, like the [SGPT](https://github.com/Muennighoff/sgpt) family of models. Open AI exposes a single embedding endpoint for both search and similarity. Open AI embeddings have a much larger maximum sequence length (8192 tokens), and a much larger dimension size (1536). Cohere and Aleph Alpha are some other embedding providers. Aleph Alpha provides more flexibility in the way the final embedding is created from the encoder output including:

* Mean pooling, where the average is taken across all token outputs in the sequence
* Weighted mean, where more weight is given to the last few tokens
* Last token , where the embedding is just the encoder output of the last token. (called \[CLS] token if you are using the BERT model)

Which option should you use? It is not always clear and depends on your data. It doesn’t hurt to experiment a bit. But the differences in performance are not expected to be very large.

**TIP**

Whether the last token (or the first token), contains good representations of the entire sequence depends a lot on the pre-training and the fine tuning objective. BERT’s pre-training objective (next sentence prediction) ensures that the \[CLS] token is much richer in representation than say Roberta, which doesn’t use the next sentence prediction objective and thus its \<s> start sequence token isn’t as informative.

## TRAINING EMBEDDING MODELS

Embeddings generated from base LLM models generally don’t perform well. For these models, it has been [shown](https://aclanthology.org/2022.acl-short.45/) that term frequencies from the pre-training set have an impact on the embedding geometry, leading to distorted cosine similarities. This has led to the cosine similarity between high frequency words to underestimate the similarity between them. To make the models generate usable embeddings, they need to be fine-tuned either in a supervised or an unsupervised/self-supervised manner.

The most promising approach for training sentence embeddings has been to use contrastive learning. In contrastive learning, we take three sentences - an anchor sentence, a sentence that it is very similar to, and a sentence that it is dissimilar to. We then train the model such that it keeps the similar sentences closer in the embedding space and pushes the embeddings of the dissimilar sentences farther apart in the embedding space.

While a similar sentence can be generated by just adding noise/dropping words in the original sentence and comparing them together, it is not very clear what the best dissimilar sentences would be. This [page](https://www.sbert.net/examples/unsupervised\_learning/README.html) shows several techniques used for unsupervised learning of sentence embedding models. The accompanying Github repo to this book shows various techniques to train or fine-tune your own embedding models.

**TIP**

Training or fine-tuning your own embedding model using your data is relatively inexpensive but can potentially come with a lot of benefits. For example, I trained an embedding model on financial text which cost less than $2000 in compute costs but ended up performing better than Open AI embeddings for my use case.

For many applications, embedding similarity is just not enough. To see why semantic search based on cosine similarity of embeddings is limited in what it can do, let’s look at an example.

The semantic similarity task is underspecified. To start with, there are several notions of similarity. For example, are two sentences that have opposite meanings but are talking about similar topics semantically similar? What about sentences or passages that have multiple facets of meanings?

Consider the query

```
query = [‘Who resigned from Edison Corporation in 2019?’]
```

Ideally, we would like to match sentences talking about resignations from Edison Corporation in 2019. But can naive cosine similarity capture this, even if the embedding model was fine-tuned on question-answering datasets?

Let’s say we are matching the query against these chunks

```
chunks = ['Hajian is an expert in cooking pineapple salsa.',

 'Hajian resigned his job at Apple.',

"Edison Corporation is the world's largest movie distributor",

'Roman resigned his job at Apple',

'Roman resigned his job at Edison corporation',

 'Hajian left Edison Corporation in 2017',

'Roman joined Edison Corporation in 2019',

'Hajian did not resign from Edison Corporation']
```

We generate embeddings and calculate similarities

```
model = SentenceTransformer('all-mpnet-base-v2')
chunk_embeddings = model.encode(chunks, show_progress_bar=True, device='cuda', normalize_embeddings=True, convert_to_tensor=True)

query_embedding = model.encode(query, device='cuda', normalize_embeddings=True, convert_to_tensor=True)
hits = util.semantic_search(query_embedding, chunk_embeddings, score_function=util.dot_score)
```

The output is

```
[[{'corpus_id': 5, 'score': 0.710587739944458},

  {'corpus_id': 7, 'score': 0.7007321715354919},

  {'corpus_id': 4, 'score': 0.6919746994972229},

  {'corpus_id': 6, 'score': 0.6464899182319641},

  {'corpus_id': 1, 'score': 0.467547744512558},

  {'corpus_id': 2, 'score': 0.4549838900566101},

  {'corpus_id': 3, 'score': 0.4215250313282013},

  {'corpus_id': 0, 'score': 0.015120428055524826}]]
```

Note that sentences like ‘Roman joined Edison Corporation in 2019’ and ‘Hajian did not resign from Edison Corporation’ have a high similarity score. If we use a similarity threshold of 0.6, then sentences 4,5,6,7 are included in the prompt along with the query to the LLM, and the LLM will correctly answer the question.

As you might have wondered, the precision-recall tradeoff needs to be handled with care, especially since we have a limited context window to feed candidate chunks into the LLM. Read more about precison and recall metrics [here](https://machinelearningmastery.com/precision-recall-and-f-measure-for-imbalanced-classification/).

## EXERCISE

Check how the similarities for these sentences fare when using OpenAI and Cohere embeddings. What do their similarity scores look like? Is it better or worse than what we see here?

So far we have seen that embedding models are specialized for solving a specific task - like semantic search or paraphrasing. A recent development ties together embedding models and the concept of instruction-tuning, which we discussed in Chapter 6. Imagine if you could use the same embedding model to generate different embeddings for the same chunk, based on the task it is going to be used for. These embeddings are called _Instructor Embeddings_. [Instructor Embeddings](https://instructor-embedding.github.io/) allow you to optionally specify the domain, text type (whether it is a sentence, paragraph etc), task, along with the text during encoding.

Here is an example:

```
!pip install InstructorEmbedding

from InstructorEmbedding import INSTRUCTOR
model = INSTRUCTOR('hkunlp/instructor-large')

customized_embeddings = model.encode(
[['Represent the question for retrieving supporting documents:',
  'Who is the CEO of Apple'],
 ['Represent the sentence for retrieval:',
  'Tim Cook is the CEO of Apple'],
 ['Represent the sentence for retrieval:',
  'He is a musically gifted CEO'],
)
```

The creators of InstructorEmbedding recommend using the prompt _Represent the question for retreiving supporting documents_ for queries, and _Represent the sentence for retrieval_ for the chunks.

Another way in which the principle of instruction-tuning can be applied to retrieval is with _description-based retrieval_, where the query can be the description of the text that needs to be retrieved, rather than an instantiation (example) of the text that needs to be retrieved. [Ravfogel et al.](https://arxiv.org/pdf/2305.12517.pdf) have published description-based retrieval models that in my experience are very effective. Note that these models have a dual-encoder setup - separate models are used to encode the query and documents. Check out the accompanying Github repo to this book for examples on how to use these models.

## EVALUATING EMBEDDING MODELS

There are a dizzying number of embedding models available. Which one should you use? [MTEB](https://arxiv.org/abs/2210.07316) (Massive Text Embedding Benchmark) is a benchmark that can help you make the decision. MTEB covers a diverse set of tasks and benchmarks both latency and task performance, enabling you to reason about the tradeoff.

Check out the the current [leaderboard](https://huggingface.co/spaces/mteb/leaderboard), which is updated regularly. While there is no clear winner across all tasks, you can see that Instructor Embeddings generally perform very well, and the SBERT models based on MPNet and MiniLM perform strongly on retrieval tasks. Open AI’s ‘text-embedding-ada-002’ is also in the upper echelons in terms of task performance. Your final decision should generally balance pricing, latency and performance tradeoffs.

### Vector Databases

Depending on your application, you may have to deal with millions or billions of vectors, with the need to add new vectors every day and associate metadata tags to them. Vector databases facilitate this. Both open-source and paid options are available. Weviate, Milvus, Pinecone, Chroma, Qdrant, Redis are some of the popular vector databases. More established players like ElasticSearch, Redis, and Postgres have also started providing vector databases support.

[Table 4-1](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch04.html#vector-db) shows the vector databases, the features they provide, whether they provide hosting or not, along with pricing and licensing information.

| Vector Database Name | Access                                                  | Hosting                     | Pricing                                                    | Other Notes                                      |
| -------------------- | ------------------------------------------------------- | --------------------------- | ---------------------------------------------------------- | ------------------------------------------------ |
| Annoy                | Open-Source                                             | In-memory                   | Free, but you need to generate embeddings outside the tool | Allows you to use static files as indexes        |
| Chroma               | Open-Source                                             | In-memory                   | Free                                                       | Bare bones and easiest to get set up with        |
| DeepLake             | Open-Source                                             | In-memory                   | Free                                                       | Supports data versioning, multimodal data        |
| ElasticSearch        | Open-Source, managed service on Elastic Cloud Available | In-memory and Elastic Cloud | Free, paid service starts from $95 a month                 | Comes with a lot of logging, monitoring features |
| Milvus               | Open-Source, managed service on Zilliz Cloud            | Cloud-native, Zillus Cloud  | Free, around 0.2$ an hour                                  |                                                  |
| PGVector             | Open-source, managed service on AWS, Heroku etc         | database extension          | Bundled with Postgres                                      | Integrated with SQL database                     |
| Pinecone             | Closed                                                  | AWS/Google Cloud            | Starting at $70 a month                                    | Provides many enterprise features                |
| Qdrant               | Open-Source                                             | Self-hosted                 | Free                                                       | Supports distributed deployment                  |
| Redis                | Open-source, with enterprise support on Redis Cloud     | Self-hosted, Redis Cloud    | Free + Redis Cloud                                         | Provides many enterprise features                |
| Weviate              | Open-Source                                             | Cloud-native                | Free                                                       | Known for being extremely fast                   |

Let’s now have a look at how vector DB’s work. Probably the simplest one to get started with is Chroma, which is open-source, and can run locally on your machine or can be deployed on AWS.

```
!pip install chromadb

import chromadb
chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="mango_science")
chunks = ['353 varieties of mangoes are now extinct',
'Mangoes are grown in the tropics']
metadata = [{"topic": "extinction", "chapter": "2"}, {"topic": "regions", "chapter": "5"}]
unique_ids = [str(i) for i in range(len(chunks))]

collection.add(
   documents=chunks,
   metadatas=metadata,
   ids=unique_ids
  )
results = collection.query(
   query_texts=["Where are mangoes grown?"],
   n_results=2,
   where={"chapter": { "$ne": "2"}},
   where_document={"$contains":"grown"}
)
```

Most vector databases offer the following:

* Approximate nearest neighbor search, to reduce latency
* Ability to filter using metadata, like the _where_ option in Chroma
* Ability to integrate keyword search, like the _where\_document_ option in Chroma
* Support Boolean search operations, so that multiple search clauses can be combined with AND or OR operations
* Ability to update or delete entries in the database in real time.

**TIP**

If you are working with less than a million vectors, vector databases might not be necessary, especially if you are not going to constantly add new vectors to your collection.

### Rerankers

In a multi-stage retrieval workflow, the later stages comprise the rerankers, which take the top-k most relevant chunks as determined by the earlier stages, and reranks them in order of relevance. The reranker is usually a language model fine-tuned on the specific task. You can use BERT-like models for building a relevance classifier, where given a query and a chunk, the model outputs the probability of the chunk being relevant to answering the query. These models are called _cross-encoders_, as they capture the interaction between query and document in the same model.

The input sequence for BERT is of the format

```
[CLS] query_text [SEP] chunk_text [SEP]
```

These days, more advanced models like [ColBERT](https://arxiv.org/abs/2004.12832) are used for reranking. The accompanying Github repo to the book contains a tutorial on how to effectively use [ColBERTv2](https://arxiv.org/abs/2112.01488) and similar models for reranking.

In ColBERT-style models, both queries and documents are encoded independently into a set of vectors, by taking the BERT output embeddings for each token in the query or document and down-projecting them. At query time, the cosine similarity between each query token embedding and all the token embeddings of the document are calculated. For each query token embedding, the maximum cosine similarity between the document token embeddings are taken and summed. This type of architecture is called _late interaction_, since the query and document are not encoded together but interact together only later in the process. Late interaction saves times as compared to traditional cross-encoders, as document embeddings can be created and stored in advance.

In general, retrieval is a hard task and does not adequately compensate the limitations of LLMs. Companies adopting this paradigm are realizing that retrieval is becoming the limiting factor that imposes a ceiling on the maximum performance they can get from LLMs.

### LlamaIndex

LlamaIndex facilitates the interfacing of external data stores with LLMs. Let’s explore how to use it to our advantage. In this book, we will be using LlamaIndex version 0.7.20

**WARNING**

While using libraries like LlamaIndex and LangChain which use LLMs under the hood, verify what LLMs are being used as default before running data-heavy workloads on it. For example, as of today LlamaIndex uses ‘text-davinci-003’ as the default model, which is 10 times more expensive than ‘gpt-3.5-turbo’ (the chatGPT model).

To our relief, changing the underlying model is pretty easy -

```
from llama_index import LLMPredictor, ServiceContext
from langchain import OpenAI

llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
```

You can then pass the service\_context while invoking LlamaIndex features.

#### Indices

LlamaIndex provides several types of index data structures for organizing your data for efficient retrieval. The most powerful aspect of this feature is the capacity for compositionality - you can build indices on top of other indices! We will soon see why this is useful. Meanwhile, here are some of the index types they support:

_List Index_ - Each chunk of text is called a Node in LlamaIndex parlance. A list index is simply a sequential list of nodes, ex: a long document split into multiple chunks and arranged contiguously. I strongly recommend you to do the splitting yourself by utilizing your knowledge of the document structure and topic boundaries. Here is how you would create a node by yourself, and set the relationships between different nodes.

[Figure 4-4](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch04.html#list-index) shows how each chunk is represented by a _Node_ and connected to each other sequentially.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/book5.4.png" alt="List Index" height="149" width="600"><figcaption></figcaption></figure>

**Figure 4-4. List Index in LlamaIndex**

```
from llama_index.data_structs.node import Node, DocumentRelationship, GPTListIndex
node1 = Node(text="This is the first chunk", doc_id=1)
node2 = Node(text="This is the second chunk", doc_id=2)
node1.relationships[DocumentRelationship.NEXT] = node2.get_doc_id()
node2.relationships[DocumentRelationship.PREVIOUS] = node1.get_doc_id()
nodes = [node1, node2]
index = GPTListIndex(nodes, service_context = service_context)
```

**TIP**

For some applications, you might be starting off with too much data, or there might be data that is not very likely to be relevant to a user. In these cases, you can perform ‘lazy loading’ of embeddings. You can leverage a List Index for this. Creating a List Index doesn’t involve a call to the underlying LLM or Embedding model. Instead, it supports creating embeddings dynamically at query time.

_Vector Store Index_ - This index type uses a vector store at the backend for storing text along with its embeddings and associated metadata. By default, LlamaIndex uses an in-memory vector store. It supports integration with major vector databases like Chroma, Qdrant, Milvus, Weviate, Pinecone etc.

_Keyword Table Index_ - This is similar to an inverted index used in search engines. Keywords are extracted from Nodes, such that each keyword is potentially mapped to multiple nodes. Note that keywords could include phrases as well. LLamaIndex provides three different methods for building keyword indices.

1. GPTKeywordTableIndex - Uses an LLM to extract keywords
2. GPTRAKEKeywordTableIndex - Uses a heuristics algorithm (RAKE) for extracting keywords
3. GPTSimpleKeywordTableIndex - Uses regular expressions to extract all words and then removes stop words.

For each keyword, the associated nodes are sorted in the order of the number of times the keyword appears in the index.

**WARNING**

GPTKeywordTableIndex (LLM driven keyword extraction) is very expensive!

_Tree Index_ - The tree index is built in a bottom-up fashion by building leaf nodes from chunks, and then constructing their parent nodes by generating summaries of the leaf node texts. The default number of children for each node is 10. As an example, if you are working with a long document, the intermediate levels of the tree could contain chapter summaries etc. By default, LlamaIndex uses the underlying LLM for generating summaries, but you can add your own summaries as well. If you are updating or inserting a new node, the summaries might have to be regenerated.

[Figure 4-5](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch04.html#tree-index) shows the structure of a tree index, and what data each node represents.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/book5.5.png" alt="Tree Index" height="284" width="600"><figcaption></figcaption></figure>

**Figure 4-5. Tree Index in LlamaIndex**

#### Compositional Indices

Pay close attention to the ‘information architecture’ of your data. Most data can be organized conceptually in a hierarchical fashion. For example, Wikipedia pages are organized in terms of hierarchical categories. Exploiting this structure while building indices will make retrieval more effective and efficient. Here are some points you need to take into account while designing your compositional indices.

1. The type of expected user queries and their distribution.
2. Where you land on the precision vs recall tradeoff for your use case.
3. The number of LLM tokens/embeddings needed to construct the indices.
4. The average number of node traversals needed to complete retrieval.

Some common compositional strategies include - creating tree indices for individual documents and creating a list index over them, creating a keyword index over tree indices, and creating a list index over a vector store index.

[Figure 4-6](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch04.html#compositional-index) shows a List Index on top of a Tree Index. The Tree Index can represent splitting a document into multiple chunks and indexing them in a tree form. The List index connects multiple such documents easily, thus enabling the LLM to perform cross-document information retrieval.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/book5.2.png" alt="Compositional Index" height="179" width="600"><figcaption></figcaption></figure>

**Figure 4-6. Compositional Index in LlamaIndex**

#### Retrieval & Response

Once we have created our indices, we can start processing user queries. A typical processing pipeline of a user query consists of the following stages:

1. Query post-processing (Optional)
2. Retrieval
3. Retrieval Post-processing (Optional)
4. Response Synthesis

[Figure 4-7](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch04.html#retrieval-response) depicts the retrieval and response pipeline. Let’s go through each step in detail.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/book5.6.png" alt="Retrieval and Response" height="89" width="600"><figcaption></figcaption></figure>

**Figure 4-7. Retrieval-Response pipeline in the Passive Approach**

_Query Post-processing_ - User queries can be edited and augmented, both to increase the likelihood of retrieving relevant data chunks and to increase the likelihood of the LLM to come up with the right answer (prompt hacking).

The query can be rephrased to ensure higher recall. If you are using a non-embedding based method for retrieval like keyword matching, regular expressions, or BM25, you can use traditional _query expansion_ methods. Naive query expansion would involve adding synonyms of keywords in your query and other topic information.

If you are using embedding based methods, you can use techniques that utilize LLMs for query expansion. Two such examples are Query2Doc and HyDE.

Query2Doc involves generating a pseudo-document for the user query in a few-shot setting. For instance, in the 2-shot setting, we can use the prompt

```
‘Write a passage that answers the given query
Query: <query 1>
Passage: <passage 1>

Query: <query 2>
Passage: <passage 2>

Query: <User Query>
Passage:
```

The passage generated by the LLM is then concatenated to the query. We know that LLMs are susceptible to hallucination, and the generated passage might be factually incorrect and laughably so. But that doesn’t matter to us, because as long as we get enough token and semantic overlaps with the correct answer, we can construct an embedding that would be very similar to the embedding of the real chunk that contains the answer.

Hence, with Query2Doc we have rephrased the query to include the query + LLM generated pseudo-passage.

HyDE is a very similar technique, and is implemented natively by LlamaIndex. The original implementation of HyDE uses a zero-shot setting, and replaces the original query with the generated passage, which it calls a ‘Hypothetical Document’. The Hypothetical Document is then run through a Contriever model to generate embeddings.

Using HyDE in LlamaIndex is as simple as

```
from llama_index.indices.query.query_transform import  (HyDEQueryTransform)

query = "what does critical audit matter mean?"
hyde = HyDEQueryTransform(include_original=True)
```

If include\_original is set to true, then the hypothetical document is appended to the original query.

LlamaIndex also supports query decomposition with the _DecomposeQueryTransform_ class. This uses an LLM to decompose a complex query into multiple subqueries. The hope is that the relevant context for each subquery can be more easily retrieved, and the retrieval output from all subqueries can finally be combined together and fed to the LLM.

## SYSTEM PROMPTS

LlamaIndex uses LLMs in many different scenarios - generating keywords for building keyword indices, selecting the child node to traverse in a tree index, query expansion just to name a few. In order to perform these tasks, it uses default prompts that are defined [here](https://github.com/jerryjliu/llama\_index/blob/main/llama\_index/prompts/default\_prompts.py).

For example, here is the prompt they use for tree traversal

“Some choices are given below. It is provided in a numbered list (1 to {num\_chunks}),where each item in the list corresponds to a summary.\n ---------------------\n {context\_list}\n---------------------\n. Using only the choices above and not prior knowledge, return the top choices (no more than {branching\_factor}, ranked by most relevant to least) that are most relevant to the question: _{query\_str}_\n”. Provide choices in the following format: _ANSWER: \<numbers>_ and explain why these summaries were selected in relation to the question.\n”

The _branching factor_ indicates the maximum number of child nodes that will be traversed. _num\_chunks_ is the number of children of the current node, and _context\_list_ is the summary texts of all the children. Note that this prompt asks it to provide an explanation of why these summaries were generated. You can save some tokens by forgoing the explanation if you want.

## EXERCISE

The prompt _Given the context information and not prior knowledge, answer the question:_ is phrased this way to force the model to use only the retrieved information to answer the question and not use its own memory, which is susceptible to hallucination. The most effective prompt to reduce hallucinations differs by model. Try this yourself - do you notice any hallucinations when using this prompt? On what models is this prompt effective? You can experiment with variations of this prompt yourself. But note that you can’t just prompt your way out of hallucinations, and be very vary of any product that claims to do so!

_Retrieval_ - the next step after post-processing the query (which is an optional step) is to retrieve the relevant Nodes from the indices. The retrieval process varies across index types.

* For ListIndex, either all nodes are retrieved, or you can generate embeddings of the nodes on the fly and retrieve the top-k nodes that are most similar to the query embedding.
* Similarly, the VectorIndex returns the top-k nodes as per the embedding similarity.
* For the KeywordTableIndex, keywords are extracted from the query and the matching nodes are returned.
* For the TreeIndex, the tree is traversed top-down starting from the root node to find the leaf nodes that are relevant to the query. Note that an LLM is used at each level to determine which children to choose to traverse, based on the similarity between the query and the summary of each child.

_Retrieval Postprocessing_ LlamaIndex provides you with the flexibility to exclude or rerank retrieved nodes before feeding them into the LLM. While you can build your own custom postprocessor for your application needs, LlamaIndex comes with some default postprocessors you can use

* _Keyword filtering_ - Exclude nodes that contain particular keywords or mandate the existence of keywords in a node
* _Similarity threshold_ - Filter out nodes with an embedding similarity below a threshold. The threshold varies by embedding model and is dependent on your data domain, so you will have to empirically arrive at it. It is highly recommended to use a threshold, failing which you might have to deal with a lot of irrelevant results.
* _Nearby nodes_ - Include previous or next nodes if you are using a list index, so that you can get additional context. This is especially useful if you used simpler document splitting techniques
* _Temporal filtering_ - Order returned nodes by most recent date and chooses the top-k.

_Response Synthesis_ - After node retrieval and filtering, LlamaIndex calls the LLM with the query and the text from the selected nodes. Three modes of LLM interaction are supported.

_default_ - The default mode uses a strategy called _create-and-refine_. If three Nodes are retrieved, then

1. It would first call the LLM with the query + text from first node
2. It would then call the LLM with the query + text from second node + output from first call + instruction asking it to refine the first output
3. It would then call the LLM with the query + text from third node + output from second call + instruction asking it to refine the second output

_compact_ - in this mode, the prompt is stuffed with text from as many nodes as can fit into the context window. If all nodes don’t fit into the context window, the create-and-refine strategy is used. This reduces the number of calls to the LLM and the number of refine steps. Reducing the number of refine steps could potentially reduce the quality of the final output.

_tree\_summarize_ - Given the retrieved nodes, a tree is dynamically composed bottom up, with the parent nodes being summaries of the child nodes. The root node is returned as the answer. This is useful if you just want a summary of the matching nodes and don’t need specific answers to questions.

#### Token Budgeting

GPT-4 and similar models can be really expensive if you have a high volume of usage/high amount of data. LlamaIndex (and LangChain) provides a cost analyzer that allows you to keep track of the number of tokens spent during both indexing and query time.

You can use the MockLLMPredictor, and MockEmbedding classes to predict the number of tokens that will be used, before actually using them. This allows you to intervene and run token optimizations before interacting with the LLM.

Let’s see the MockLLMPredictor in action

```
from llama_index import GPTKeywordIndex, MockLLMPredictor

mock_llm= MockLLMPredictor(max_tokens=256)
service_context = ServiceContext.from_defaults(llm_predictor=mock_llm)
index = GPTKeywordIndex.from_documents(documents, service_context=service_context)
print(mock_llm.last_token_usage)
```

**TIP**

Language models (even the older ones like BERT etc) are relatively insensitive to word order. For many tasks, removing stop words barely affects performance! This can be a way to reduce your token budget.

### Data Loading & Parsing

To build a retrieval system, you will first have to load and parse the data into a suitable format. Frameworks like [LangChain](https://python.langchain.com/docs/get\_started/introduction.html) and [LlamaIndex](https://gpt-index.readthedocs.io/en/latest/) with their _Document Loaders_ and _Data Connectors_ features respectively can make pulling data from sources like Slack, Notion etc easier. However, you probably will write your own data loaders so that you can exploit your specialized knowledge about the format and content of your data.

As an example, let’s see how we can use LangChain’s ArxivLoader to load and extract text from scientific papers.

```
!pip install langchain arxiv pymupdf

from langchain.document_loaders import ArxivLoader

docs = ArxivLoader(query="LLM Agents", load_max_docs=10, load_all_available_meta=True).load()
```

Note that the query field is free text, and can include a list of arXiv IDs or a text query like ‘LLM Agents’.

You can access the metadata of each document using

```
docs[0].metadata
```

When you issue a query, it is the content in the metadata that is searched.

The parsed text is present in

```
docs[0].page_content
```

**TIP**

Text extraction is a crucial but often difficult, unglamorous, and overlooked aspect of the pipeline. For example, even with the presence of dozens of PDF extraction libraries, many of which use deep learning models under the hood, extracting text from PDF along with all the formatting metadata like subtitles and paragraph boundaries is not 100% accurate. Similarly, removing boilerplate text and artifacts from the document format is not trivial in many cases. The structure and format of the text is important metadata for a retrieval engine. I would strongly recommend spending more time to assess the quality of the text you are extracting and the impact of it on downstream task performance.

## EXERCISE

Load and parse text from this [page](https://www.ourcommons.ca/DocumentViewer/en/44-1/house/sitting-218/hansard) containing the text of a debate in the Canadian Parliament. You can use a library like Unstructured for more custom processing, and libraries like justext for removing boilerplate. How effective do you find these libraries?

## External Tools

So far we have seen how we can augment LLM’s with retrieval. Retrieval augmented LLMs alleviate some of the limitations of LLMs we mentioned at the beginning of this chapter, opening the door to using LLMs’ over your own data, while reducing hallucination risks.

Now, let’s discuss how LLMs can interact with the broader software ecosystem. This includes

* code interpreters/compilers, which the LLM can offload computation to for tasks or subtasks that it is not good at.
* APIs’, which can be queried for information
* Other LLMs and machine learning models

We will henceforth refer to these as _tools_. In almost all cases, the LLM generates code in order to communicate with these tools - for example code representing a mathematical operation that it wants offloaded to a Python interpreter, code representing an API call to extract information in response to a query, SQL queries to access databases and so on.

We can build applications by processing inputs using a complex sequence of operations involving LLMs and multiple tools. We will call this sequence of operations a chain.

LangChain provides a useful framework for implementing chains. It is the most popular library that came out of the recent LLM boom. We will dive into the parts of the library that help in tool interaction.

**TIP**

At some point, you might start questioning yourself - _Why am I even using LangChain/LlamaIndex?_. You are not alone! The abstraction provided by these frameworks reduce flexibility and increase code boilerplate. A lot of abstractions are very thin wrappers over established tools like the Requests library. Overall, frameworks like Langchain is great for experimentation, especially if you are not a machine learning expert. However, they are not a necessary component of a production-ready LLM application.

For an introduction to LangChain, check out their [docs](https://python.langchain.com/docs/get\_started/introduction). In this chapter, we will focus only on the chains and tools features.

The first chain we will explore is the LLM-Requests chain. The workflow for this chain is

1. The Requests library is used to fetch some data from the URL
2. The data is fed to an LLM which parses it and returns results.

As an example, consider an application that takes a user query, uses the Requests library to query Google, and then feeds the results to the LLM which parses it and returns the right answer.

```
from langchain.chains import LLMRequestsChain, LLMChain
from langchain.prompts import PromptTemplate

search_template = """Extract the answer to the question '{query}' using the Google Search results.
Provide at least one well formed sentence.
Use the format
Extracted:<answer or "Search Engine results do not contain the answer">
google search results: {requests_result},
Extracted:"""

requests_prompt = PromptTemplate(
    input_variables=["query", "requests_result"],
    template=search_template
)

llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0)
requests_chain = LLMRequestsChain(llm_chain = LLMChain(llm=llm, prompt=requests_prompt))

query = 'How is the weather in Toronto going to be tomorrow? In Celsius'

inputs = {
            "query": query,

            "url": "https://www.google.com/search?q=" + '+' + query
        }

results = requests_chain.run(inputs)
```

The output is

```
'Tomorrow, the highest temperature in Toronto will be 16°C (60.8°F), while the lowest temperature will be 9°C (48.2°F).'
```

You can see how simple it is for the LLM to gain access to real-time information like the weather!

Next, we will see the LLM Math chain, that allows you to offload mathematical operations to a Python interpreter. The workflow of this chain is

1. The user specifies a mathematical expression in natural language eg: ‘what is 30 percent of 124?’
2. The LLM converts the text into python code.
3. The code is run using Python’s numexpr library (not Python REPL because of code injection issues)
4. The result is parsed by the LLM and fed back to the user.

Here is the code

```
from langchain.chains import LLMMathChain
llm_math = LLMMathChain.from_llm(OpenAI())

llm_math.run("What is 34 percent of 123?")
```

The output is 41.82, as expected. If you try to use the chain for non-math use cases, it will fail.

```
llm_math.run("Who is the prime minister of canada 14-3 years ago?")

ValueError: unknown format from LLM: This does not involve a math problem, so it cannot be translated into an
expression for use with the numexpr library.
```

This is because the prompt for the LLMMathChain is ‘Translate a math problem into a expression that can be executed using Python’s numexpr library. Use the output of running this code to answer the question.’

**WARNING**

Be very vary of running code generated by LLMs. Users can induce the model to generate malicious code!

API’s can be called using the API chain.

```
from langchain.chains import APIChain
llm_math = APIChain.from_llm_and_api_docs(llm, docs, verbose=True
```

The _docs_ variable should contain the relevant API documentation. Note that you will have to ensure that they fit within the context length, so you will need to write some logic for including only the relevant documentation in the docs. One solution is to store the API docs in a data store and then perform retrieval over it to fetch the relevant documentation.

Similarly, the SQLDatabaseChain enables you to query a database. Again, you should have sufficient guard rails to ensure that the LLM doesn’t inadvertently generate code that will update the database. The workflow is

1. Decide the tables which contain the information requested in the query
2. Generate a SQL query for retrieving the results
3. Parse the results and return the answer to the user.

**NOTE**

What is the difference between LangChain and LlamaIndex? I hear this question a lot. They have a lot of overlapping features, but in the end I find them quite complementary. LangChain is good at…well, building chains and LLM agents (the Explicit and the Agentic paradigm), while LlamaIndex has really good support for retrieval augmentation.

Tool use is perhaps the most exciting paradigm in terms of LLM application development, and thus deserves a more comprehensive treatment. In Chapter 13, we will explore tool use in detail, including showing how to fine-tune an LLM with a tool-learning dataset, as well as showing how to create a tool-learning dataset for your own tools.

## Summary

In this Chapter, we have seen how LLMs can be integrated into the software ecosystem, helping them be ubiquitous. We explored the different modes of interaction with external tools and data stores, and discussed some of the most useful tools and data stores one can employ as of today. We covered retrieval augmented models in detail, emphasising the role of embeddings and vector databases. We also had a brief look at LLM agents. In Chapter 13, We will learn more about operationalizing LLM agents, including creating tool-following datasets and fine-tuning your LLM with it.

In the next chapter, we will learn more about designing LLM applications over domain-specific data that is vastly different from the pre-training data seen by LLMs. We will cover various domain adaptation techniques and provide pointers on how to choose between them depending on your application needs and your target domain.
