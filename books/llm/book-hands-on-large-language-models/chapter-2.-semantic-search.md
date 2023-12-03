# Chapter 2. Semantic Search

## Chapter 2. Semantic Search

## A NOTE FOR EARLY RELEASE READERS

With Early Release ebooks, you get books in their earliest form—the author’s raw and unedited content as they write—so you can take advantage of these technologies long before the official release of these titles.

This will be the 3rd chapter of the final book. Please note that the GitHub repo will be made active later on.

If you have comments about how we might improve the content and/or examples in this book, or if you notice missing material within this chapter, please reach out to the editor at _mcronin@oreilly.com_.

Search was one of the first Large Language Model (LLM) applications to see broad industry adoption. Months after the release of the seminal [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) paper, Google announced it was using it to power Google Search and that it [represented ](https://blog.google/products/search/search-language-understanding-bert/)“one of the biggest leaps forward in the history of Search”. Not to be outdone, Microsoft Bing also [stated ](https://azure.microsoft.com/en-us/blog/bing-delivers-its-largest-improvement-in-search-experience-using-azure-gpus/)that “Starting from April of this year, we used large transformer models to deliver the largest quality improvements to our Bing customers in the past year”.

This is a clear testament to the power and usefulness of these models. Their addition instantly and massively improves some of the most mature, well-maintained systems that billions of people around the planet rely on. The ability they add is called _semantic search_, which enables searching by meaning, and not simply keyword matching.

In this chapter, we’ll discuss three major ways of using language models to power search systems. We’ll go over code examples where you can use these capabilities to power your own applications. Note that this is not only useful for web search, but that search is a major component of most apps and products. So our focus will not be just on building a web search engine, but rather on your own dataset. This capability powers lots of other exciting LLM applications that build on top of search (e.g., retrieval-augmented generation, or document question answering). Let’s start by looking at these three ways of using LLMs for semantic search.

## Three Major Categories of Language-Model-based Search Systems

There’s a lot of research on how to best use LLMs for search. Three broad categories of these models are:

1- Dense Retrieval

Say that a user types a search query into a search engine. Dense retrieval systems rely on the concept of embeddings, the same concept we’ve encountered in the previous chapters, and turn the search problem into retrieving the nearest neighbors of the search query (after both the query and the documents are converted into embeddings). [Figure 2-1](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch02.html#fig\_1\_dense\_retrieval\_is\_one\_of\_the\_key\_types\_of\_semanti) shows how dense retrieval takes a search query, consults its archive of texts, and outputs a set of relevant results.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/semantic_search_888356_01.png" alt="Dense retrieval is one of the key types of semantic search  relying on the similarity of text embeddings to retrieve relevant results" height="309" width="600"><figcaption></figcaption></figure>

**Figure 2-1. Dense retrieval is one of the key types of semantic search, relying on the similarity of text embeddings to retrieve relevant results**

2- Reranking

These systems are pipelines of multiple steps. A Reranking LLM is one of these steps and is tasked with scoring the relevance of a subset of results against the query, and then the order of results is changed based on these scores. [Figure 2-2](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch02.html#fig\_2\_rerankers\_the\_second\_key\_type\_of\_semantic\_search) shows how rerankers are different from dense retrieval in that they take an additional input: a set of search results from a previous step in the search pipeline.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/semantic_search_888356_02.png" alt="Rerankers  the second key type of semantic search  take a search query and a collection of results  and re order them by relevance  often resulting in vastly improved results." height="209" width="600"><figcaption></figcaption></figure>

**Figure 2-2. Rerankers, the second key type of semantic search, take a search query and a collection of results, and re-order them by relevance, often resulting in vastly improved results.**

3- Generative Search

The growing LLM capability of text generation led to a new batch of search systems that include a generation model that simply generates an answer in response to a query. [Figure 2-3](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch02.html#fig\_3\_generative\_search\_formulates\_an\_answer\_to\_a\_questi) shows a generative search example.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/semantic_search_888356_03.png" alt="Generative search formulates an answer to a question and cites its information sources." height="285" width="600"><figcaption></figcaption></figure>

**Figure 2-3. Generative search formulates an answer to a question and cites its information sources.**

All three concepts are powerful and can be used together in the same pipeline. The rest of the chapter covers these three types of systems in more detail. While these are the major categories, they are not the only LLM applications in the domain of search.

## Dense Retrieval

Recall that embeddings turn text into numeric representations. Those can be thought of as points in space as we can see in [Figure 2-4](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch02.html#fig\_4\_the\_intuition\_of\_embeddings\_each\_text\_is\_a\_point). Points that are close together mean that the text they represent is similar. So in this example, text 1 and text 2 are similar to each other (because they are near each other), and different from text 3 (because it’s farther away).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/semantic_search_888356_04.png" alt="The intuition of embeddings  each text is a point  texts with similar meaning are close to each other." height="374" width="600"><figcaption></figcaption></figure>

**Figure 2-4. The intuition of embeddings: each text is a point, texts with similar meaning are close to each other.**

This is the property that is used to build search systems. In this scenario, when a user enters a search query, we embed the query, thus projecting it into the same space as our text archive. Then we simply find the nearest documents to the query in that space, and those would be the search results.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/semantic_search_888356_05.png" alt="Dense retrieval relies on the property that search queries will be close to their relevant results." height="375" width="600"><figcaption></figcaption></figure>

**Figure 2-5. Dense retrieval relies on the property that search queries will be close to their relevant results.**

Judging by the distances in [Figure 2-5](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch02.html#fig\_5\_dense\_retrieval\_relies\_on\_the\_property\_that\_search), “text 2” is the best result for this query, followed by “text 1”. Two questions could arise here, however:

Should text 3 even be returned as a result? That’s a decision for you, the system designer. It’s sometimes desirable to have a max threshold of similarity score to filter out irrelevant results (in case the corpus has no relevant results for the query).

Are a query and its best result semantically similar? Not always. This is why language models need to be trained on question-answer pairs to become better at retrieval. This process is explained in more detail in chapter 13.

### Dense Retrieval Example

Let’s take a look at a dense retrieval example by using Cohere to search the Wikipedia page for the film _Interstellar_. In this example, we will do the following:

1. Get the text we want to make searchable, apply some light processing to chunk it into sentences.
2. Embed the sentences
3. Build the search index
4. Search and see the results

To start, we’ll need to install the libraries we’ll need for the example:

```
# Install Cohere for embeddings, Annoy for approximate nearest neighbor search
!pip install cohere tqdm Annoy
```

Get your Cohere API key by signing up at https://cohere.ai/. Paste it in the cell below. You will not have to pay anything to run through this example.

Let’s import the datasets we’ll need:

```
import cohere
import numpy as np
import re
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from annoy import AnnoyIndex
 
# Paste your API key here. Remember to not share publicly
api_key = ''
 
# Create and retrieve a Cohere API key from os.cohere.ai
co = cohere.Client(api_key)
 
```

1.  Getting the text Archive

    Let’s use the first section of the Wikipedia article on the film _Interstellar_. https://en.wikipedia.org/wiki/Interstellar\_(film). We’ll get the text, then break it into sentences.

    ```
    text = """
    Interstellar is a 2014 epic science fiction film co-written, directed, and produced by Christopher Nolan. 
    It stars Matthew McConaughey, Anne Hathaway, Jessica Chastain, Bill Irwin, Ellen Burstyn, Matt Damon, and Michael Caine. 
    Set in a dystopian future where humanity is struggling to survive, the film follows a group of astronauts who travel through a wormhole near Saturn in search of a new home for mankind.
     
    Brothers Christopher and Jonathan Nolan wrote the screenplay, which had its origins in a script Jonathan developed in 2007. 
    Caltech theoretical physicist and 2017 Nobel laureate in Physics[4] Kip Thorne was an executive producer, acted as a scientific consultant, and wrote a tie-in book, The Science of Interstellar. 
    Cinematographer Hoyte van Hoytema shot it on 35 mm movie film in the Panavision anamorphic format and IMAX 70 mm. 
    Principal photography began in late 2013 and took place in Alberta, Iceland, and Los Angeles. 
    Interstellar uses extensive practical and miniature effects and the company Double Negative created additional digital effects.
     
    Interstellar premiered on October 26, 2014, in Los Angeles. 
    In the United States, it was first released on film stock, expanding to venues using digital projectors. 
    The film had a worldwide gross over $677 million (and $773 million with subsequent re-releases), making it the tenth-highest grossing film of 2014. 
    It received acclaim for its performances, direction, screenplay, musical score, visual effects, ambition, themes, and emotional weight. 
    It has also received praise from many astronomers for its scientific accuracy and portrayal of theoretical astrophysics. Since its premiere, Interstellar gained a cult following,[5] and now is regarded by many sci-fi experts as one of the best science-fiction films of all time.
    Interstellar was nominated for five awards at the 87th Academy Awards, winning Best Visual Effects, and received numerous other accolades"""
    # Split into a list of sentences
    texts = text.split('.')
     
    # Clean up to remove empty spaces and new lines
    texts = np.array([t.strip(' \n') for t in texts])
    ```
2.  Embed the texts

    Let’s now embed the texts. We’ll send them to the Cohere API, and get back a vector for each text.

    ```
    # Get the embeddings
    response = co.embed(
      texts=texts,
    ).embeddings
     
    embeds = np.array(response)
    print(embeds.shape)
    ```

    Which outputs:

    (15, 4096)

    Indicating that we have 15 vectors, each one is of size 4096.
3.  Build The Search Index

    Before we can search, we need to build a search index. An index stores the embeddings and is optimized to quickly retrieve the nearest neighbors even if we have a very large number of points.

    ```
    # Create the search index, pass the size of embedding
    search_index = AnnoyIndex(embeds.shape[1], 'angular')
     
    # Add all the vectors to the search index
    for index, embed in enumerate(embeds):
        search_index.add_item(index, embed)
     
    search_index.build(10) 
    search_index.save('test.ann')
    ```
4.  Search the index

    We can now search the dataset using any query we want. We simply embed the query, and present its embedding to the index, which will retrieve the most similar texts.

    Let’s define our search function:

    ```
    def search(query):
      
      # 1. Get the query's embedding
      query_embed = co.embed(texts=[query]).embeddings[0]
     
      # 2. Retrieve the nearest neighbors
      similar_item_ids = search_index.get_nns_by_vector(query_embed, n=3,
                                                      include_distances=True)
      # 3. Format the results
      results = pd.DataFrame(data={'texts': texts[similar_item_ids[0]], 
                                  'distance': similar_item_ids[1]})
      
      # 4. Print and return the results
      print(f"Query:'{query}'\nNearest neighbors:")
      return results
    ```

    We are now ready to write a query and search the texts!

    ```
    query = "How much did the film make?"
    search(query)
    ```

    Which produces the output:

    ```
    Query:'How much did the film make?'
    Nearest neighbors:
    ```

    |                            | <pre><code>texts
    </code></pre>                                                                                                                                              | <pre><code>distance
    </code></pre> |
    | -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
    | <pre><code>0
    </code></pre> | <pre><code>The film had a worldwide gross over $677 million (and $773 million with subsequent re-releases), making it the tenth-highest grossing film of 2014
    </code></pre> | <pre><code>0.815905
    </code></pre> |
    | <pre><code>1
    </code></pre> | <pre><code>It stars Matthew McConaughey, Anne Hathaway, Jessica Chastain, Bill Irwin, Ellen Burstyn, Matt Damon, and Michael Caine
    </code></pre>                            | <pre><code>1.066861
    </code></pre> |
    | <pre><code>2
    </code></pre> | <pre><code>In the United States, it was first released on film stock, expanding to venues using digital projectors
    </code></pre>                                            | <pre><code>1.086919
    </code></pre> |

The first result has the least distance, and so is the most similar to the query. Looking at it, it answers the question perfectly. Notice that this wouldn’t have been possible if we were only doing keyword search because the top result did not include the words “much” or “make”.

To further illustrate the capabilities of dense retrieval, here’s a list of queries and the top result for each one:

Query: “Tell me about the \$$$?”

Top result: The film had a worldwide gross over $677 million (and $773 million with subsequent re-releases), making it the tenth-highest grossing film of 2014

Distance: 1.244138

Query: “Which actors are involved?”

Top result: It stars Matthew McConaughey, Anne Hathaway, Jessica Chastain, Bill Irwin, Ellen Burstyn, Matt Damon, and Michael Caine

Distance: 0.917728

Query: “How was the movie released?”

Top result: In the United States, it was first released on film stock, expanding to venues using digital projectors

Distance: 0.871881

#### Caveats of Dense Retrieval

It’s useful to be aware of some of the drawbacks of dense retrieval and how to address them. What happens, for example, if the texts don’t contain the answer? We still get results and their distances. For example:

```
Query:'What is the mass of the moon?'
Nearest neighbors:
```

|                            | <pre><code>texts
</code></pre>                                                                                                                                              | <pre><code>distance
</code></pre> |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| <pre><code>0
</code></pre> | <pre><code>The film had a worldwide gross over $677 million (and $773 million with subsequent re-releases), making it the tenth-highest grossing film of 2014
</code></pre> | <pre><code>1.298275
</code></pre> |
| <pre><code>1
</code></pre> | <pre><code>It has also received praise from many astronomers for its scientific accuracy and portrayal of theoretical astrophysics
</code></pre>                            | <pre><code>1.324389
</code></pre> |
| <pre><code>2
</code></pre> | <pre><code>Cinematographer Hoyte van Hoytema shot it on 35 mm movie film in the Panavision anamorphic format and IMAX 70 mm
</code></pre>                                   | <pre><code>1.328375
</code></pre> |

In cases like this, one possible heuristic is to set a threshold level -- a maximum distance for relevance, for example. A lot of search systems present the user with the best info they can get, and leave it up to the user to decide if it’s relevant or not. Tracking the information of whether the user clicked on a result (and were satisfied by it), can improve future versions of the search system.

Another caveat of dense retrieval is cases where a user wants to find an exact match to text they’re looking for. That’s a case that’s perfect for keyword matching. That’s one reason why hybrid search, which includes both semantic search and keyword search, is used.

Dense retrieval systems also find it challenging to work properly in domains other than the ones that they were trained on. So for example if you train a retrieval model on internet and Wikipedia data, and then deploy it on legal texts (without having enough legal data as part of the training set), the model will not work as well in that legal domain.

The final thing we’d like to point out is that this is a case where each sentence contained a piece of information, and we showed queries that specifically ask those for that information. What about questions whose answers span multiple sentences? This shows one of the important design parameters of dense retrieval systems: what is the best way to chunk long texts? And why do we need to chunk them in the first place?

### Chunking Long Texts

One limitation of Transformer language models is that they are limited in context sizes. Meaning we cannot feed them very long texts that go above a certain number of words or tokens that the model supports. So how do we embed long texts?

There are several possible ways, and two possible approaches shown in [Figure 2-6](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch02.html#fig\_6\_it\_s\_possible\_to\_create\_one\_vector\_representing\_an) include indexing one vector per document, and indexing multiple vectors per document.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/semantic_search_888356_06.png" alt="It s possible to create one vector representing an entire document  but it s better for longer documents to be split into smaller chunks that get their own embeddings." height="315" width="600"><figcaption></figcaption></figure>

**Figure 2-6. It’s possible to create one vector representing an entire document, but it’s better for longer documents to be split into smaller chunks that get their own embeddings.**

#### One vector per document

In this approach, we use a single vector to represent the whole document. The possibilities here include:

* Embedding only a representative part of the document and ignoring the rest of the text. This may mean embedding only the title, or only the beginning of the document. This is useful to get quickly started with building a demo but it leaves a lot of information unindexed and so unsearchable. As an approach, it may work better for documents where the beginning captures the main points of a document (think: Wikipedia article). But it’s really not the best approach for a real system.
* Embedding the document in chunks, embedding those chunks, and then aggregating those chunks into a single vector. The usual method of aggregation here is to average those vectors. A downside of this approach is that it results in a highly compressed vector that loses a lot of the information in the document.

This approach can satisfy some information needs, but not others. A lot of the time, a search is for a specific piece of information contained in an article, which is better captured if the concept had its own vector.

#### Multiple vectors per document

In this approach, we chunk the document into smaller pieces, and embed those chunks. Our search index then becomes that of chunk embeddings, not entire document embeddings.

The chunking approach is better because it has full coverage of the text and because the vectors tend to capture individual concepts inside the text. This leads to a more expressive search index. Figure X-3 shows a number of possible approaches.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/semantic_search_888356_07.png" alt="A number of possible options for chunking a document for embedding." height="273" width="600"><figcaption></figcaption></figure>

**Figure 2-7. A number of possible options for chunking a document for embedding.**

The best way of chunking a long text will depend on the types of texts and queries your system anticipates. Approaches include:

* Each sentence is a chunk. The issue here is this could be too granular and the vectors don’t capture enough of the context.
* Each paragraph is a chunk. This is great if the text is made up of short paragraphs. Otherwise, it may be that every 4-8 sentences are a chunk.
* Some chunks derive a lot of their meaning from the text around them. So we can incorporate some context via:
  * Adding the title of the document to the chunk
  * Adding some of the text before and after them to the chunk. This way, the chunks can overlap so they include some surrounding text. This is what we can see in [Figure 2-8](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch02.html#fig\_8\_chunking\_the\_text\_into\_overlapping\_segments\_is\_one).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/semantic_search_888356_08.png" alt="Chunking the text into overlapping segments is one strategy to retain more of the context around different segments." height="174" width="600"><figcaption></figcaption></figure>

**Figure 2-8. Chunking the text into overlapping segments is one strategy to retain more of the context around different segments.**

Expect more chunking strategies to arise as the field develops -- some of which may even use LLMs to dynamically split a text into meaningful chunks.

### Nearest Neighbor Search vs. Vector Databases

The most straightforward way to find the nearest neighbors is to calculate the distances between the query and the archive. That can easily be done with NumPy and is a reasonable approach if you have thousands or tens of thousands of vectors in your archive.

As you scale beyond to the millions of vectors, an optimized approach for the retrieval is to rely on approximate nearest neighbor search libraries like Annoy or FAISS. These allow you to retrieve results from massive indexes in milliseconds and some of them can scale to GPUs and clusters of machines to serve very large indices.

Another class of vector retrieval systems are vector databases like Weaviate or Pinecone. A vector database allows you to add or delete vectors without having to rebuild the index. They also provide ways to filter your search or customize it in ways beyond merely vector distances.

### Fine-tuning embedding models for dense retrieval

Just like we’ve seen in the text classification chapter, we can improve the performance of an LLM on a task using fine-tuning. Just like in that case, retrieval needs to optimize text embeddings and not simply token embeddings. The process for this finetuning is to get training data composed of queries and relevant results.

Looking at one example from our dataset, the sentence “Interstellar premiered on October 26, 2014, in Los Angeles.”. Two possible queries where this is a relevant result are:

* Relevant Query 1: “Interstellar release date”
* Relevant Query 2: “When did Interstellar premier”

The fine-tuning process aims to make the embeddings of these queries close to the embedding of the resulting sentence. It also needs to see negative examples of queries that are not relevant to the sentence, for example.

* Irrelevant Query: “Interstellar cast”

Having these examples, we now have three pairs - two positive pairs and one negative pair. Let’s assume, as we can see in [Figure 2-9](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch02.html#fig\_9\_before\_fine\_tuning\_the\_embeddings\_of\_both\_relevan), that before fine-tuning, all three queries have the same distance from the result document. That’s not far-fetched because they all talk about Interstellar.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/semantic_search_888356_09.png" alt="Before fine tuning  the embeddings of both relevant and irrelevant queries may be close to a particular document." height="364" width="600"><figcaption></figcaption></figure>

**Figure 2-9. Before fine-tuning, the embeddings of both relevant and irrelevant queries may be close to a particular document.**

The fine-tuning step works to make the relevant queries closer to the document and at the same time making irrelevant queries farther from the document. We can see this effect in [Figure 2-10](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch02.html#fig\_10\_after\_the\_fine\_tuning\_process\_the\_text\_embedding).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/semantic_search_888356_10.png" alt="After the fine tuning process  the text embedding model becomes better at this search task by incorporating how we define relevance on our dataset using the examples we provided of relevant and irrelevant documents." height="374" width="600"><figcaption></figcaption></figure>

**Figure 2-10. After the fine-tuning process, the text embedding model becomes better at this search task by incorporating how we define relevance on our dataset using the examples we provided of relevant and irrelevant documents.**

## Reranking

A lot of companies have already built search systems. For those companies, an easier way to incorporate language models is as a final step inside their search pipeline. This step is tasked with changing the order of the search results based on relevance to the search query. This one step can vastly improve search results and it’s in fact what Microsoft Bing added to achieve the improvements to the search results using BERT-like models.

[Figure 2-11](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch02.html#fig\_11\_llm\_rerankers\_operate\_as\_a\_part\_of\_a\_search\_pipeli) shows the structure of a rerank search system serving as the second stage in a two-stage search system.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/semantic_search_888356_11.png" alt="LLM Rerankers operate as a part of a search pipeline with the goal of re ordering a number of shortlisted search results by relevance" height="235" width="600"><figcaption></figcaption></figure>

**Figure 2-11. LLM Rerankers operate as a part of a search pipeline with the goal of re-ordering a number of shortlisted search results by relevance**

### Reranking Example

A reranker takes in the search query and a number of search results, and returns the optimal ordering of these documents so the most relevant ones to the query are higher in ranking.

```
import cohere as co
API_KEY = ""
co = cohere.Client(API_KEY)
MODEL_NAME = "rerank-english-02" # another option is rerank-multilingual-02
 
query = "film gross"
```

Cohere’s [Rerank ](https://docs.cohere.com/reference/rerank-1)endpoint is a simple way to start using a first reranker. We simply pass it the query and texts, and get the results back. We don’t need to train or tune it.

```
results = co.rerank(query=query, model=MODEL_NAME, documents=texts, top_n=3)
```

We can print these results:

```
results = co.rerank(query=query, model=MODEL_NAME, documents=texts, top_n=3) # Change top_n to change the number of results returned. If top_n is not passed, all results will be returned.
for idx, r in enumerate(results):
  print(f"Document Rank: {idx + 1}, Document Index: {r.index}")
  print(f"Document: {r.document['text']}")
  print(f"Relevance Score: {r.relevance_score:.2f}")
  print("\n")
```

Output:

```
Document Rank: 1, Document Index: 10
Document: The film had a worldwide gross over $677 million (and $773 million with subsequent re-releases), making it the tenth-highest grossing film of 2014
Relevance Score: 0.92
 
 
Document Rank: 2, Document Index: 12
Document: It has also received praise from many astronomers for its scientific accuracy and portrayal of theoretical astrophysics
Relevance Score: 0.11
 
 
Document Rank: 3, Document Index: 2
Document: Set in a dystopian future where humanity is struggling to survive, the film follows a group of astronauts who travel through a wormhole near Saturn in search of a new home for mankind
Relevance Score: 0.03
 
```

This shows the reranker is much more confident about the first result, assigning it a relevance score of 0.92 while the other results are scored much lower in relevance.

More often, however, our index would have thousands or millions of entries, and we need to shortlist, say one hundred or one thousand results and then present those to the reranker. This shortlisting is called the _first stage_ of the search pipeline.

The dense retriever example we looked at in the previous section is one possible first-stage retriever. In practice, the first stage can also be a search system that incorporates both keyword search as well as dense retrieval.

### Open Source Retrieval and Reranking with Sentence Transformers

If you want to locally setup retrieval and reranking on your own machine, then you can use the Sentence Transformers library. Refer to the documentation in https://www.sbert.net/ for setup. Check the [_Retrieve & Re-Rank_ section](https://www.sbert.net/examples/applications/retrieve\_rerank/README.html) for instructions and code examples for how to conduct these steps in the library.

### How Reranking Models Work

One popular way of building LLM search rerankers present the query and each result to an LLM working as a _cross-encoder_. Meaning that a query and possible result are presented to the model at the same time allowing the model to view the full text of both these texts before it assigns a relevance score. This method is described in more detail in a paper titled [_Multi-Stage Document Ranking with BERT_ ](https://arxiv.org/abs/1910.14424)and is sometimes referred to as monoBERT.

This formulation of search as relevance scoring basically boils down to being a classification problem. Given those inputs, the model outputs a score from 0-1 where 0 is irrelevant and 1 is highly relevant. This should be familiar from looking at the Classification chapter.

To learn more about the development of using LLMs for search, [_Pretrained Transformers for Text Ranking: BERT and Beyond_](https://arxiv.org/abs/2010.06467) is a highly recommended look at the developments of these models until about 2021.

## Generative Search

You may have noticed that dense retrieval and reranking both use representation language models, and not generative language models. That’s because they’re better optimized for these tasks than generative models.

At a certain scale, however, generative LLMs started to seem more and more capable of a form of useful information retrieval. People started asking models like ChatGPT questions and sometimes got relevant answers. The media started painting this as a threat to Google which seems to have started an arms race in using language models for search. Microsoft [launched](https://blogs.microsoft.com/blog/2023/02/07/reinventing-search-with-a-new-ai-powered-microsoft-bing-and-edge-your-copilot-for-the-web/) Bing AI, powered by generative models. Google launched [Bard](https://blogs.microsoft.com/blog/2023/02/07/reinventing-search-with-a-new-ai-powered-microsoft-bing-and-edge-your-copilot-for-the-web/), its own answer in this space.

### What is Generative Search?

Generative search systems include a text generation step in the search pipeline. At the moment, however, generative LLMs aren’t reliable information retrievers and are prone to generate coherent, yet often incorrect, text in response to questions they don’t know the answer to.

The first batch of generative search systems is using search models as simply a summarization step at the end of the search pipeline. We can see an example in [Figure 2-12](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch02.html#fig\_12\_generative\_search\_formulates\_answers\_and\_summaries).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/semantic_search_888356_12.png" alt="Generative search formulates answers and summaries at the end of a search pipeline while citing its sources  returned by the previous steps in the search system ." height="338" width="600"><figcaption></figcaption></figure>

**Figure 2-12. Generative search formulates answers and summaries at the end of a search pipeline while citing its sources (returned by the previous steps in the search system).**

Until the time of this writing, however, language models excel at generating coherent text but they are not reliable in retrieving facts. They don’t yet really know what they know or don’t know, and tend to answer lots of questions with coherent text that can be incorrect. This is often referred to as _hallucination_. Because of it, and for the fact that search is a use case that often relies on facts or referencing existing documents, generative search models are trained to cite their sources and include links to them in their answers.

Generative search is still in its infancy and is expected to improve with time. It draws from a machine learning research area called retrieval-augmented generation. Notable systems in the field include [RAG](https://ai.facebook.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-natural-language-processing-models/), [RETRO](https://jalammar.github.io/illustrated-retrieval-transformer/), [Atlas](https://arxiv.org/pdf/2208.03299.pdf), amongst others.

## Other LLM applications in search

In addition to these three categories, there are plenty of other ways to use LLMs to power or improve search systems. Examples include:

* Generating synthetic data to improve embedding models. This includes methods like [GenQ](https://www.pinecone.io/learn/genq/) and [InPars-v2](https://arxiv.org/abs/2301.01820) that look at documents, generate possible queries and questions about those documents, then use that generated data to fine-tune a retrieval system.
* The growing reasoning capabilities of text generation models are leading to search systems that can tackle complex questions and queries by breaking them down into multiple sub-queries that are tackled in sequence, leading up to a final answer of the original question. One method in this category is described in [_Demonstrate-Search-Predict: Composing retrieval and language models for knowledge-intensive NLP_](https://arxiv.org/abs/2212.14024).

### Evaluation metrics

Semantic search is evaluated using metrics from the Information Retrieval (IR) field. Let’s discuss two of these popular metrics: Mean Average Precision (MAP), and Normalized Discounted Cumulative Gain (nDCG).

Evaluating search systems needs three major [components](https://nlp.stanford.edu/IR-book/html/htmledition/irbook.html), a text archive, a set of queries, and relevance judgments indicating which documents are relevant for each query. We see these components in FIgure 3-13.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/semantic_search_888356_13.png" alt="To evaluate search systems  we need a test suite including queries and relevance judgements indicating which documents in our archive are relevant for each query." height="318" width="600"><figcaption></figcaption></figure>

**Figure 2-13. To evaluate search systems, we need a test suite including queries and relevance judgements indicating which documents in our archive are relevant for each query.**

Using this test suite, we can proceed to explore evaluating search systems. Let’s start with a simple example, let’s assume we pass Query 1 to two different search systems. And get two sets of results. Say we limit the number of results to three results only as we can see in [Figure 2-14](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch02.html#fig\_14\_to\_compare\_two\_search\_systems\_we\_pass\_the\_same\_qu).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/semantic_search_888356_14.png" alt="To compare two search systems  we pass the same query from our test suite to both systems and look at their top results" height="244" width="600"><figcaption></figcaption></figure>

**Figure 2-14. To compare two search systems, we pass the same query from our test suite to both systems and look at their top results**

To tell which is a better system, we turn the relevance judgments that we have for the query. [Figure 2-15](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch02.html#fig\_15\_looking\_at\_the\_relevance\_judgements\_from\_our\_test) shows which of the returned results are relevant.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/semantic_search_888356_15.png" alt="Looking at the relevance judgements from our test suite  we can see that System 1 did a better job than System 2." height="270" width="600"><figcaption></figcaption></figure>

**Figure 2-15. Looking at the relevance judgements from our test suite, we can see that System 1 did a better job than System 2.**

This shows us a clear case where system 1 is better than system 2. Intuitively, we may just count how many relevant results each system retrieved. System A got two out of three correctly, and System 2 got only one out of three correctly.

But what about a case like Figure3-16 where both systems only get one relevant result out of three, but they’re in different positions.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/semantic_search_888356_16.png" alt="We need a scoring system that rewards system 1 for assigning a high position to a relevant result    even though both systems retrieved only one relevant result in their top three results." height="252" width="600"><figcaption></figcaption></figure>

**Figure 2-16. We need a scoring system that rewards system 1 for assigning a high position to a relevant result -- even though both systems retrieved only one relevant result in their top three results.**

In this case, we can intuit that System 1 did a better job than system 2 because the result in the first position (the most important position) is correct. But how can we assign a number or score to how much better that result is? Mean Average Precision is a measure that is able to quantify this distinction.

One common way to assign numeric scores in this scenario is Average Precision, which evaluates System 1’s result for the query to be 0.6 and System 2’s to be 0.1. So let’s see how Average Precision is calculated to evaluate one set of results, and then how it’s aggregated to evaluate a system across all the queries in the test suite.

#### Mean Average Precision (MAP)

To score system 1 on this query, we need to calculate multiple scores first. Since we are looking at only three results, we’ll need to look at three scores - one associated with each position.

The first one is easy, looking at only the first result, we calculate the precision score: we divide the number of correct results by the total number of results (correct and incorrect). [Figure 2-17](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch02.html#fig\_17\_\_to\_calculate\_mean\_average\_precision\_we\_start\_by) shows that in this case, we have one correct result out of one (since we’re only looking at the first position now). So precision here is 1/1 = 1.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/semantic_search_888356_17.png" alt="To calculate Mean Average Precision  we start by calculating precision at each position  starting by position  1." height="192" width="144"><figcaption></figcaption></figure>

**Figure 2-17. To calculate Mean Average Precision, we start by calculating precision at each position, starting by position #1.**

We need to continue calculating precision results for the rest of the position. The calculation at the second position looks at both the first and second position. The precision score here is 1 (one out of two results being correct) divided by 2 (two results we’re evaluating) = 0.5.

[Figure 2-18](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch02.html#fig\_18\_caption\_to\_come) continues the calculation for the second and third positions. It then goes one step further -- having calculated the precision for each position, we average them to arrive at an Average Precision score of 0.61.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/semantic_search_888356_18.png" alt="Caption to come" height="192" width="144"><figcaption></figcaption></figure>

**Figure 2-18. Caption to come**

This calculation shows the average precision for a single query and its results. If we calculate the average precision for System 1 on all the queries in our test suite and get their mean, we arrive at the Mean Average Precision score that we can use to compare System 1 to other systems across all the queries in our test suite.

## Summary

In this chapter, we looked at different ways of using language models to improve existing search systems and even be the core of new, more powerful search systems. These include:

* Dense retrieval, which relies on the similarity of text embeddings. These are systems that embed a search query and retrieve the documents with the nearest embeddings to the query’s embedding.
* Rerankers, systems (like monoBERT) that look at a query and candidate results, and scores the relevance of each document to that query. These relevance scores are then used to order the shortlisted results according to their relevance to the query often producing an improved results ranking.
* Generative search, where search systems that have a generative LLM at the end of the pipeline to formulate an answer based on retrieved documents while citing its sources.

We also looked at one of the possible methods of evaluating search systems. Mean Average Precision allows us to score search systems to be able to compare across a test suite of queries and their known relevance to the test queries.
