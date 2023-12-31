# Chapter 5: Building a Chatbot like ChatGPT/Bard

## 5 Building a Chatbot like ChatGPT

### Join our book community on Discord

[https://packt.link/EarlyAccessCommunity](https://packt.link/EarlyAccessCommunity)

![Qr code Description automatically generated](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file35.png)

In this chapter, we’ll discuss chatbots, what they are, what they can do, and how they can be implemented. We’ll start this chapter by discussing the evolution of chatbots and the current state-of-the-art. Understanding and enhancing the capabilities of current chatbots and Large Language Models (LLMs) has practical implications for their safe and effective use in different domains including regulated ones like medicine and law.Proactive communication, important for engaging with customer needs, requires on the technical side, implementations of mechanisms for context and memory. The focus of this chapter is on retrieval mechanisms including vector storage to improve the accuracy of responses and the faithfulness of chatbots to the available information and the current conversation.We’ll go through the fundamentals of modern chatbots such as retrieval-augmented language models (RALMs), the technical background of what we need to implement them in LangChain. We’ll go into details about methods for loading documents and information including vector storage and embeddings. We’ll further discuss more specific methods for memory, which are about maintaining the knowledge and state of the ongoing conversation. Finally, we discuss another important topic from the reputational and legal perspective: moderation. Let’s make sure our responses are not abusive, intolerant, or against the spirit of the organization. LangChain allows you to pass any text through a moderation chain to check if it contains harmful content. Throughout the chapter, we’ll work on a chatbot implementation with an interface in Streamlit that you can find in the `chat_with_retrieval` directory in the Github repository for the book at [https://github.com/benman1/generative\_ai\_with\_langchain](https://github.com/benman1/generative\_ai\_with\_langchain)The main sections are:

* What is a chatbot?
* Retrieval and vectors
* Implementing a chatbot
* Don’t say anything stupid!

We'll begin the chapter by introducing chatbots and the state-of-the-art of the technology.

### What is a chatbot?

A **chatbot** is an Artificial Intelligence program that can chat with users, provide information and support, book things, and perform various other tasks. It is used to reproduce powerful interactions with users and can be utilized in different industries and for different purposes.Chatbots are beneficial because they can automate tasks, provide instant responses, and offer personalized experiences to users. They can be used for customer support, lead generation, sales, information retrieval, and more. Chatbots can save time, improve efficiency, enhance customer experiences, and streamline business processes. Chatbots work by utilizing natural language processing (NLP) and machine learning algorithms. They analyze user input, understand the intent behind it, and generate appropriate responses. They can be designed to work with text-based messaging platforms or voice-based applications.Some use cases for chatbots in customer service include providing 24/7 support, handling frequently asked questions, assisting with product recommendations, processing orders and payments, and resolving simple customer issues. Some more use cases of chatbots include:

* Appointment Scheduling: Chatbots can help users schedule appointments, book reservations, and manage their calendars.
* Information Retrieval: Chatbots can provide users with specific information, such as weather updates, news articles, or stock prices.
* Virtual Assistants: Chatbots can act as personal assistants, helping users with tasks like setting reminders, sending messages, or making phone calls.
* Language Learning: Chatbots can assist in language learning by providing interactive conversations and language practice.
* Mental Health Support: Chatbots can offer emotional support, provide resources, and engage in therapeutic conversations for mental health purposes.
* Education: In educational settings, virtual assistants are being explored as virtual tutors, helping students learn and assess their knowledge, answer questions, and deliver personalized learning experiences.
* HR and Recruitment: Chatbots can assist in the recruitment process by screening candidates, scheduling interviews, and providing information about job openings.
* Entertainment: Chatbots can engage users in interactive games, quizzes, and storytelling experiences.
* Law: Chatbots can be used to provide basic legal information, answer common legal questions, assist with legal research, and help users navigate legal processes. They can also help with document preparation, such as drafting contracts or creating legal forms.
* Medicine: Chatbots can assist with symptom checking, provide basic medical advice, and offer mental health support. They can improve clinical decision-making by providing relevant information and recommendations to healthcare professionals

These are just a few examples, and the use cases of chatbots continue to expand across various industries and domains. Chat technology in any field has the potential to make information more accessible and provide initial support to individuals seeking assistance.

#### What’s the state-of-the-art?

The Turing Test, named after Alan Turing an English computer scientist, cryptanalyst, and mathematician, is a method of inquiry in artificial intelligence (AI) for determining whether or not a computer is capable of thinking like a human being. Despite much debate about the relevance of the Turing Test today and the validity of the competitions that are based around it, the test still stands as a philosophical starting point for discussing and researching AI. As we continue to make advances in AI and better understand and map how the human brain functions, the Turing Test remains foundational for defining intelligence and is a baseline for the debate about what we should expect from technologies for them to be considered thinking machines.Turing proposed that a computer can be said to possess AI if it can mimic human responses under specific conditions. The original Turing Test requires three terminals, each of which is physically separated from the other two. One terminal is operated by a computer, while the other two are operated by humans. During the test, one of the humans works as the questioner, while the second human and the computer function as respondents. The questioner interrogates the respondents within a specific subject area, using a specified format and context. After a preset length of time or number of questions, the questioner is then asked to decide which respondent was human and which was a computer.Since the formation of the test, many AI have been able to pass; one of the first was Joseph Weizenbaum’s ELIZA. In 1966, he published an article about his chatbot ELIZA, “ELIZA - a computer program for the study of natural language communication between man and machine.” ELIZA was one of the first chatbots ever created and simulated the role of a psychotherapist.Created with a sense of humor to show the limitations of technology, the chatbot employed simplistic rules and vague, open-ended questions as a way of giving an impression of empathic understanding in the conversation, and was an ironic twist often seen as a milestone of artificial intelligence. However, ELIZA had limited knowledge and could only engage in conversations within a specific domain of topics. It also couldn't keep long conversations or learn from the discussion.The Turing Test has been criticized over the years, in particular because historically, the nature of the questioning had to be limited in order for a computer to exhibit human-like intelligence. For many years, a computer might only score high if the questioner formulated the queries, so they had “Yes” or “No” answers or pertained to a narrow field of knowledge. When questions were open-ended and required conversational answers, it was less likely that the computer program could successfully fool the questioner.In addition, a program such as ELIZA could pass the Turing Test by manipulating symbols it does not understand fully. Philosopher John Searle argued that this does not determine intelligence comparable to humans. To many researchers, the question of whether or not a computer can pass a Turing Test has become irrelevant. Instead of focusing on how to convince someone they are conversing with a human and not a computer program, the real focus should be on how to make a human-machine interaction more intuitive and efficient. For example, by using a conversational interface.In 1972, another significant chatbot called PARRY was developed, who acted as a patient with schizophrenia. It had a defined personality, its responses were based on a system of assumptions, and emotional responses were triggered by changes in the user’s utterances. In an experiment in 1979, PARRY was tested by five psychiatrists who had to determine whether the patient they were interacting with was a computer program or a real schizophrenic patient. The results varied, with some psychiatrists giving correct diagnoses and others giving incorrect ones.Although several variations of the Turing Test are often more applicable to our current understanding of AI, the original format of the test is still used to this day. For example, the Loebner Prize has been awarded annually since 1990 to the most human-like computer program as voted by a panel of judges. The competition follows the standard rules of the Turing Test. Critics of the award’s relevance often downplay it as more about publicity than truly testing if machines can think.IBM Watson is a cognitive computing system developed by IBM that uses natural language processing, machine learning, and other AI technologies to respond to complex questions in natural language. It works by ingesting and processing vast amounts of structured and unstructured data, including text, images, and videos. IBM Watson became famous in 2011 when it competed on the quiz show Jeopardy! and defeated two former champions. Watson has been applied in various fields, including healthcare, finance, customer service, and research. In healthcare, Watson has been used to assist doctors in diagnosing and treating diseases, analyzing medical records, and conducting research. It has also been applied in the culinary field, with the Chef Watson application helping chefs create unique and innovative recipes.In 2018, Google Duplex successfully made an appointment with a hairdresser over the phone in front of a crowd of 7,000. The receptionist was completely unaware that they weren’t conversing with a real human. This is considered by some to be a modern-day Turing Test pass, despite not relying on the true format of the test as Alan Turing designed it.Developed by OpenAI, ChatGPT is a language model that uses deep learning techniques to generate human-like responses. It was launched on November 30, 2022, and is built upon OpenAI’s proprietary series of foundational GPT models, including GPT-3.5 and GPT-4. ChatGPT allows users to have coherent, natural, and engaging conversations with the AI, refining and steering the conversation towards their desired length, format, style, level of detail, and language used.ChatGPT is considered a game changer because it represents a significant advancement in conversational AI. Because of its ability to generate contextually relevant responses and to understand and respond to a wide range of topics and questions, the chatbot is thought by some to have the best chance of beating the test in its true form of any technology that we have today. But, even with its advanced text-generation abilities, it can be tricked into answering nonsensical questions and therefore would struggle under the conditions of the Turing Test.Overall, ChatGPT’s capabilities and user-friendly interface have made it a significant advancement in the field of conversational AI, offering new possibilities for interactive and personalized interactions with AI systems.

> Here are some examples of chatbots:

* ELIZA: One of the earliest chatbots, ELIZA was developed in the 1960s and used pattern matching to simulate conversation with users.
* Siri: Siri is a popular voice-based chatbot developed by Apple. It is integrated into Apple devices and can perform tasks, answer questions, and provide information.
* Alexa: Alexa is an intelligent personal assistant developed by Amazon. It can respond to voice commands, play music, provide weather updates, and control smart home devices.
* Google Assistant: Google Assistant is a chatbot developed by Google. It can answer questions, provide recommendations, and perform tasks based on user commands.
* Mitsuku: Mitsuku is a chatbot that has won the Loebner Prize Turing Test multiple times. It is known for its ability to engage in natural and human-like conversations.

> These are just a few examples, and there are many more chatbots available in various industries and applications.

One concern of the use of the Turing test and derivatives is that it focuses on imitation and deceit, when it more meaningful tests should emphasize the need for developers to focus on creating useful and interesting capabilities rather than just performing tricks. The use of benchmarks and academic/professional examinations provides more specific evaluations of AI system performance.The current objective of the researchers in the field is to provide a better benchmark for testing the capabilities of artificial intelligence (AI) systems, specifically large language models (LLMs) such as GPT-4. They aim to understand the limits of LLMs and identify areas where they may fail. The advanced AI systems, including GPT-4, excel in tasks related to language processing but struggle with simple visual logic puzzles. LLMs can generate plausible next words based on statistical correlations but may lack reasoning or understanding of abstract concepts. Researchers have different opinions about the capabilities of LLMs, with some attributing their achievements to limited reasoning abilities. The research on testing LLMs and understanding their capabilities has practical implications. It can help in the safe and effective application of LLMs in real-world domains such as medicine and law. By identifying the strengths and weaknesses of LLMs, researchers can determine how to best utilize them.ChatGPT’s training has made it better at handling hallucinations compared to its predecessors, which means that it is less likely to generate nonsensical or irrelevant responses. However, it is important to note that ChatGPT can still confidently present inaccurate information, so users should exercise caution and verify the information provided.Context and memory play significant roles in ensuring the chatbot’s dialogues deliver accurate information and responses accurately reflective of previous interactions, thus enabling more faithful engagements with users. We’ll discuss this in more detail now.

#### Context and Memory

Context and memory are important aspects of chatbot design. They allow chatbots to maintain conversational context, respond to multiple queries, and store and access long-term memory. They are important factors in conditioning chatbot responses for accuracy and faithfulness. The significance of memory and context in chatbots can be compared to the significance of memory and comprehension in human-human conversations. A conversation without recalling past exchanges or comprehending or knowing about the broader context can be disjointed and cause miscommunication, resulting in an unsatisfactory conversational experience.**Contextual understanding** dramatically impacts the accuracy of chatbot responses. It refers to the ability of the chatbot to comprehend both the entire conversation and what some of the relevant background rather than just the last message from the user. A chatbot that is conscious of the context can maintain a holistic perspective of the conversation, making the chat flow more natural and human-like.**Memory retention** directly influences the faithfulness of the chatbot’s performance, which involves consistency in recognizing and remembering facts from previous conversations for future use. This feature enhances the personalized experience for the user.For instance, if a user says, “Show me the cheapest flights,” and then follows with, “How about hotels in that area?” without the context of the previous messages, the chatbot wouldn’t have a clue what area the user is referring to. In a reversed scenario, a context-aware chatbot would understand that the user is talking about accommodation in the same vicinity as the flight destination.A lack of memory results in inconsistencies throughout conversations (lack of faitfulness). For example, if a user has identified themselves by name in one conversation and the bot forgets this information in the next, it creates an unnatural and impersonal interaction.Both memory and context are vital to making chatbot interactions more productive, accurate, personable, and satisfactory. Without these elements, the bots may come across as deficient, rigid, and unrelatable to their human conversational counterparts. Hence, these characteristics are essential for sophisticated and satisfying interactions between computers and humans.A new aspect of chatbots with LLMs is that they can not only respond to intentions, but more intelligently engage in a dialogue with the user. This is called being proactive.

#### Intentional vs Proactive

In the context of language models or chatbots, **proactive** refers to the ability of the system to initiate actions or provide information without being explicitly prompted by the user. It involves anticipating the user’s needs or preferences based on previous interactions or contextual cues. On the other hand, **intentional** means that the chatbot is designed to understand and fulfill the user’s intentions or requests and is programmed to take specific actions or provide relevant responses based on these intentions and the desired outcome.A proactive chatbot is useful because it can connect with customers and improve their experience, creating a better customer journey. This can enhance the user experience by saving time and effort, and it can also improve customer satisfaction by addressing customer inquiries quickly and efficiently potential issues or questions before they arise.Proactive communication is critical for the success of businesses as it improves customer lifetime value (CLV) and reduces operating costs. By actively anticipating customers’ needs and providing information proactively, businesses can gain control over communication and frame conversations in a favorable light. This builds trust, customer loyalty, and enhances the organization’s reputation. Additionally, proactive communication helps improve organizational productivity by addressing customer inquiries before they ask and reducing incoming support calls.On the technical side, this capability can be achieved through context and memory, and reasoning mechanisms. This is the focus of this chapter. In the next section, we’ll discuss the fundamentals of modern chatbots such as retrieval-augmented language models (RALMs) and the technical background of what we need to implement them.

### Retrieval and vectors

In chapter 4, we discussed Retrieval-Augmented Generation (RAG), which aims to enhance the generation process by leveraging external knowledge and ensuring that the generated text is accurate and contextually appropriate. In this chapter, we’ll further discuss how to combine retrieval and generation techniques to improve the quality and relevance of generated text. Particularly, we’ll discuss Retrieval-Augmented Language Models (RALMs), a specific implementation or application of RAG, which refers to language models that are conditioned on relevant documents from a grounding corpus, a collection of written texts, during generation. In the retrieval, semantic filtering and vector storage is utilized to pre-filter relevant information from a large corpus of documents and incorporating that information into the generation process. This retrieval includes vector storage of documents.

> **Retrieval-Augmented Language Models** (**RALMs**) are language models that incorporate a retrieval component to enhance their performance. Traditional language models generate text based on the input prompt, but RALMs go a step further by retrieving relevant information from a large collection of documents and using that information to generate more accurate and contextually relevant responses.

The benefits of RALMs include:

1. Improved performance: By incorporating active retrieval, LMs can access relevant information from external sources, which can enhance their ability to generate accurate and informative responses.
2. Avoiding input length limitations: Retrieval augmented LMs discard previously retrieved documents and only use the retrieved documents from the current step to condition the next generation. This helps prevent reaching the input length limit of LMs, allowing them to handle longer and more complex queries or tasks.

More in detail, the working mechanism of retrieval augmented LMs involves the following steps:

1. Retrieval: RALMs search for relevant documents or passages from a large corpus. The LM retrieves relevant information from external sources based on a vector-based similarity search of the query and the current context.
2. Conditioning: The retrieved information is used to condition the LLM’s next generation. This means that the LM incorporates the retrieved information into its language model to generate more accurate and contextually appropriate responses.
3. Iterative process: The retrieval and conditioning steps are performed iteratively, with each step building upon the previous one. This iterative process allows the LLM to gradually improve its understanding and generation capabilities by incorporating relevant information from external sources.

The retrieved information can be used in different ways. It can serve as additional context for the language model, helping it generate more accurate and contextually appropriate responses. It can also be used to provide factual information or answer specific questions within the generated text. There are two main strategies for retrieval augmented generation:

1. Single-time Retrieval-Augmented Generation: This strategy involves using the user input as the query for retrieval and generating the complete answer at once. The retrieved documents are concatenated with the user input and used as input to the language model for generation.
2. Active Retrieval Augmented Generation: This strategy involves actively deciding when and what to retrieve during the generation process. At each step of generation, a retrieval query is formulated based on both the user input and the previously generated output. The retrieved documents are then used as input to the language model for generation. This strategy allows for the interleaving of retrieval and generation, enabling the model to dynamically retrieve information as needed.

Within the active retrieval augmented generation framework, there are two forward-looking methods called FLARE (Forward-Looking Active Retrieval Augmented Generation):

* FLARE with Retrieval Instructions: This method prompts the language model to generate retrieval queries when necessary while generating the answer. It uses retrieval-encouraging instructions, such as "\[Search(query)]", to express the need for additional information.
* FLARE Direct: This method directly uses the language model’s generation as search queries. It iteratively generates the next sentence to gain insight into the future topic, and if uncertain tokens are present, it retrieves relevant documents to regenerate the next sentence.

Unlike traditional methods where information is retrieved only once and then used for generation, FLARE follows an iterative process. It involves using a prediction of the upcoming sentence as a query to retrieve relevant documents. This allows the system to regenerate the sentence if the confidence in the initial generation is low. RALMs have shown promising results in tasks like question answering, dialogue systems, and information retrieval. They can provide more accurate and informative responses by leveraging external knowledge sources. Additionally, RALMs can be fine-tuned for specific domains or tasks by training them on domain-specific document collections, further enhancing their usefulness in specialized applications.Overall, by incorporating retrieval, RALMs can leverage the vast amount of knowledge present in the document corpus, making them more powerful and useful for various natural language processing tasks. RALMs leverage active retrieval to enhance their performance and overcome limitations in handling complex queries or tasks.LangChain implements a tool chain of different building blocks for building retrieval systems. This includes data loaders, document transformers, embedding models, vector stores, and retrievers. The relationship between them is illustrated in the diagram here (source: LangChain documentation):

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file36.jpg" alt="Figure 5.1: Vector stores and data loaders." height="913" width="2638"><figcaption><p>Figure 5.1: Vector stores and data loaders.</p></figcaption></figure>

In LangChain, we first load documents through data loaders. Then we can transform them, pass these documents to a vector store as embedding. We can then query the vector store or a retriever associated with the vector store. Retrievers in LangChain can wrap the loading and vector storage into a single step. We’ll mostly skip transformations in this chapter, however, you’ll find explanations with examples of data loaders, embeddings, storage mechanisms, and retrievers.Since we are talking about vector storage, we need to discuss vector search, which is a technique used to search and retrieve vectors (or embeddings) based on their similarity to a query vector. It is commonly used in applications such as recommendation systems, image and text search, and anomaly detection. We’ll look into more of the fundamentals behind RALMs, and we’ll start with embeddings now. Once you understand embeddings, you’ll be able to build everything from search engines to chatbots.

#### Embeddings

An embedding is a numerical representation of a content in a way that machines can process and understand. The essence of the process is to convert an object such as an image or a text into a vector that encapsulates its semantic content while discarding irrelevant details as much as possible. An embedding takes a piece of content, such as a word, sentence, or image, and maps it into a multi-dimensional vector space. The distance between two embeddings indicates the semantic similarity between the corresponding concepts (the original content).

> **Embeddings** are representations of data objects generated by machine learning models to represent. They can represent words or sentences as numerical vectors (lists of float numbers). As for the OpenAI language embedding models, the embedding is a vector of 1,536 floating point numbers that represent the text. These numbers are derived from a sophisticated language model that captures semantic content.
>
> > As an example – let’s say we have the words cat and dog – these could be represented numerically in a space together with all other words in the vocabulary. If the space is 3-dimensional, these could be vectors such as \[0.5, 0.2, -0.1] for cat and \[0.8, -0.3, 0.6] for dog. These vectors encode information about the relationships of these concepts with other words. Roughly speaking, we would expect the concepts cat and dog to be closer (more similar) to the concept of animal than to the concept of computer or embedding.

Embeddings can be created using different methods. For texts, one simple method is the **bag-of-words** approach, where each word is represented by a count of how many times it appears in a text. This approach, which in the scikit-learn library is implemented as `CountVectorizer`, was popular until **word2vec** came about. Word2vec, which – roughly speaking – learns embeddings by predicting the words in a sentence based on other surrounding words ignoring the word order in a linear model. The general idea of embeddings is illustrated in this figure (source: “Analogies Explained: Towards Understanding Word Embeddings” by Carl Allen and Timothy Hospedales, 2019; https://arxiv.org/abs/1901.09813):

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file37.png" alt="Figure 5.2: Word2Vec word embeddings in a 3D space. We can perform simple vector arithmetic with these vectors, for example the vector for king minus man plus the vector for woman gives us a vector that comes closer to queen." height="1106" width="1300"><figcaption><p>Figure 5.2: Word2Vec word embeddings in a 3D space. We can perform simple vector arithmetic with these vectors, for example the vector for king minus man plus the vector for woman gives us a vector that comes closer to queen.</p></figcaption></figure>

As for images, embeddings could come from feature extraction stages such as edge detection, texture analysis, and color composition. These features can be extracted over different window sizes to make the representations both scale-invariant and shift-invariant (**scale-space representations**). Nowadays, often, **Convolutional Neural Networks** (**CNNs**) are pre-trained on large datasets (like ImageNet) to learn a good representation of the image's properties. Since convolutional layers apply a series of filters (or kernels) on the input image to produce a feature map, conceptually this is similar to scale-space. When a pre-trained CNN then runs over a new image, it can output an embedding vector. Today, for most domains including texts and images, embeddings usually come from **transformer-based models**, which consider the context and order of the words in a sentence and the paragraph. Based on the model architecture, most importantly the number of parameters, these models can capture very complex relationships. All these models are trained on large datasets in order to establish the concepts and their relationships.These embeddings can be used in various tasks. By representing data objects as numerical vectors, we can perform mathematical operations on them and measure their similarity or use them as input for other machine learning models. By calculating distances between embeddings, we can perform tasks like search and similarity scoring, or classify objects, for example by topic or category. For example, we could be performing a simple sentiment classifier by checking if embeddings of product reviews are closer to the concept of positive or negative.

> **Distances metrics between embeddings**
>
> > There are different distance metrics used in vector similarity calculations such as:

* The **cosine distance** is a similarity measure that calculates the cosine of the angle between two vectors in a vector space. It ranges from -1 to 1, where 1 represents identical vectors, 0 represents orthogonal vectors, and -1 represents vectors that are diametrically opposed.
* **Euclidean distance**: It measures the straight-line distance between two vectors in a vector space. It ranges from 0 to infinity, where 0 represents identical vectors, and larger values represent increasingly dissimilar vectors.
* **Dot product**: It measures the product of the magnitudes of two vectors and the cosine of the angle between them. It ranges from -∞ to ∞, where a positive value represents vectors that point in the same direction, 0 represents orthogonal vectors, and a negative value represents vectors that point in opposite directions.

In LangChain, you can obtain an embedding by using the `embed_query()` method from the `OpenAIEmbeddings` class. Here is an example code snippet:

```
from langchain.embeddings.openai import OpenAIEmbeddings  
embeddings = OpenAIEmbeddings()  
text = "This is a sample query."  
query_result = embeddings.embed_query(text)  
print(query_result)
print(len(query_result))
```

This code passes a single string input to the embed\_query method and retrieves the corresponding text embedding. The result is stored in the `query_result` variable. The length of the embedding (the number of dimensions) can be obtained using the `len()` function. I am assuming you’ve set the API key as environment variable as recommended in chapter 3, _Getting started in LangChain_.You can also obtain embeddings for multiple document inputs using the `embed_documents()` method. Here is an example:

```
from langchain.embeddings.openai import OpenAIEmbeddings  
words = ["cat", "dog", "computer", "animal"]
embeddings = OpenAIEmbeddings()
doc_vectors = embeddings.embed_documents(words)
```

In this case, the `embed_documents()` method is used to retrieve embeddings for multiple text inputs. The result is stored in the `doc_vectors` variable. We could have retrieved embeddings for long documents – instead, we’ve retrieved the vectors only for single words each.We can also do arithmetic between these embeddings, for example calculate distances between them:

```
from scipy.spatial.distance import pdist, squareform
import pandas as pd
X = np.array(doc_vectors)
dists = squareform(pdist(X))
```

This gives us the Euclidean distances between our words as a square matrix. Let’s plot them:

```
import pandas as pd
df = pd.DataFrame(
    data=dists,
    index=words,
    columns=words
)
df.style.background_gradient(cmap='coolwarm')
```

The distance plot should look like this:

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file38.png" alt="Figure 5.3: Euclidean distances between embeddings of the words cat, dog, computer, animal." height="472" width="1184"><figcaption><p>Figure 5.3: Euclidean distances between embeddings of the words cat, dog, computer, animal.</p></figcaption></figure>

We can confirm: a cat and a dog are indeed closer to an animal than to a computer. There could be many questions here, for example, if a dog is more an animal than a cat, or why a dog and a cat are only little more distant from a computer than from an animal. Although, these questions can be important in certain applications, let’s bear in mind that this is a simple example.In these examples, we’ve used OpenAI embeddings – in the examples further on, we’ll use embeddings from models served by Huggingface. There are a few integrations and tools in LangChain that can help with this process, some of which we’ll encounter further on in this chapter. Additionally, LangChain provides a `FakeEmbeddings` class that can be used to test your pipeline without making actual calls to the embedding providers.In the context of this chapter, we’ll use them for retrieval of related information (semantic search). However, we still need to talk about the integrations of these embeddings into apps and broader systems, and this is where vector storage comes in.

#### How can we store embeddings?

As mentioned, in vector search, each data point is represented as a vector in a high-dimensional space. The vectors capture the features or characteristics of the data points. The goal is to find the vectors that are most similar to a given query vector. In vector search, every data object in a dataset is assigned a vector embedding. These embeddings are arrays of numbers that can be used as coordinates in a high-dimensional space. The distance between vectors can be computed using distance metrics like cosine similarity or Euclidean distance. To perform a vector search, the query vector (representing the search query) is compared to every vector in the collection. The distance between the query vector and each vector in the collection is calculated, and objects with smaller distances are considered more similar to the query.To perform vector search efficiently, vector storage mechanisms are used such as vector databases.

> **Vector search** refers to the process of searching for similar vectors among other stored vectors, for example in a vector database, based on their similarity to a given query vector. Vector search is commonly used in various applications such as recommendation systems, image and text search, and similarity-based retrieval. The goal of vector search is to efficiently and accurately retrieve vectors that are most similar to the query vector, typically using similarity measures such as the dot product or cosine similarity.

A vector storage refers to mechanism used to store vector embeddings, and which is also relevant to how they can be retrieved. Vector storage can be a standalone solution that is specifically designed to store and retrieve vector embeddings efficiently. On the other hand, vector databases are purpose-built to manage vector embeddings and provide several advantages over using standalone vector indices like FAISS. Let’s dive into a few of these concepts a bit more. There are three levels to this:

1. Indexing
2. Vector libraries
3. Vector databases

These components work together for the creation, manipulation, storage, and efficient retrieval of vector embeddings. Indexing organizes vectors to optimize retrieval, structuring them so that vectors can be retrieved quickly. There are different algorithms like k-d trees or Annoy for this. Vector libraries provide functions for vector operations like dot product and vector indexing. Finally, vector databases like Milvus or Pinecone are designed to store, manage, and retrieve large sets of vectors. They use indexing mechanisms to facilitate efficient similarity searches on these vectors. Let’s look at these in turn. There’s a fourth level in LangChain, which is retrievers, and which we’ll cover last.

#### Vector indexing

Indexing in the context of vector embeddings is a method of organizing data to optimize its retrieval and/or the storage. It’s similar to the concept in traditional database systems, where indexing allows quicker access to data records. For vector embeddings, indexing aims to structure the vectors – roughly speaking - so that similar vectors are stored next to each other, enabling fast proximity or similarity searches. A typical algorithm applied in this context is K-dimensional trees (k-d trees), but many others like Ball Trees, Annoy, and FAISS are often implemented, especially for high-dimensional vectors which traditional methods can struggle with.

> **K-Nearest Neighbor** (**KNN**) is a simple and intuitive algorithm used for classification and regression tasks. In KNN, the algorithm determines the class or value of a data point by looking at its k nearest neighbors in the training dataset.
>
> > Here’s how KNN works:

* Choose the value of k: Determine the number of nearest neighbors (k) that will be considered when making predictions.
* Calculate distances: Calculate the distance between the data point you want to classify and all other data points in the training dataset. The most commonly used distance metric is Euclidean distance, but other metrics like Manhattan distance can also be used.
* Find the k nearest neighbors: Select the k data points with the shortest distances to the data point you want to classify.
* Determine the majority class: For classification tasks, count the number of data points in each class among the k nearest neighbors. The class with the highest count becomes the predicted class for the data point. For regression tasks, take the average of the values of the k nearest neighbors.
* Make predictions: Once the majority class or average value is determined, assign it as the predicted class or value for the data point.

> It’s important to note that KNN is a lazy learning algorithm, meaning it does not explicitly build a model during the training phase. Instead, it stores the entire training dataset and performs calculations at the time of prediction.

As for alternatives to KNN, there are several other algorithms commonly used for similarity search indexing. Some of them include:

1. Product Quantization (PQ): PQ is a technique that divides the vector space into smaller subspaces and quantizes each subspace separately. This reduces the dimensionality of the vectors and allows for efficient storage and search. PQ is known for its fast search speed but may sacrifice some accuracy.
2. Locality Sensitive Hashing (LSH): This is a hashing-based method that maps similar data points to the same hash buckets. It is efficient for high-dimensional data but may have a higher probability of false positives and false negatives.
3. Hierarchical Navigable Small World (HNSW): HNSW is a graph-based indexing algorithm that constructs a hierarchical graph structure to organize the vectors. It uses a combination of randomization and greedy search to build a navigable network, allowing for efficient nearest neighbor search. HNSW is known for its high search accuracy and scalability.

Examples for PQ are KD-Trees and Ball Trees. In KD-Trees, a binary tree structure is built up that partitions the data points based on their feature values. It is efficient for low-dimensional data but becomes less effective as the dimensionality increases. Ball Tree: A tree structure that partitions the data points into nested hyperspheres. It is suitable for high-dimensional data but can be slower than KD-Tree for low-dimensional data.Apart from HNSW and KNN, there are other graph-based methods like Graph Neural Networks (GNN) and Graph Convolutional Networks (GCN) that leverage graph structures for similarity search.The Annoy (Approximate Nearest Neighbors Oh Yeah) algorithm uses random projection trees to index vectors. It constructs a binary tree structure where each node represents a random hyperplane. Annoy is simple to use and provides fast approximate nearest neighbor search.These indexing algorithms have different trade-offs in terms of search speed, accuracy, and memory usage. The choice of algorithm depends on the specific requirements of the application and the characteristics of the vector data.

#### Vector libraries

**Vector libraries**, like Facebook (Meta) Faiss or Spotify Annoy, provide functionality for working with vector data. In the context of vector search, a vector library is specifically designed to store and perform similarity search on vector embeddings. These libraries use the Approximate Nearest Neighbor (ANN) algorithm to efficiently search through vectors and find the most similar ones. They typically offer different implementations of the ANN algorithm, such as clustering or tree-based methods, and allow users to perform vector similarity search for various applications.Here’s a quick overview over some open-source libraries for vector storage that shows their popularity in terms of Github stars over time (source: star-history.com):

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file39.png" alt="Figure 5.4: Star history for several popular open-source vector libraries." height="1076" width="2114"><figcaption><p>Figure 5.4: Star history for several popular open-source vector libraries.</p></figcaption></figure>

You can see that faiss has been starred a lot by Github users. Annoy comes second. Others have not found the same popularity yet.Let’s quickly go through these:

* FAISS (Facebook AI Similarity Search) is a library developed by Meta (previously Facebook) that provides efficient similarity search and clustering of dense vectors. It offers various indexing algorithms, including PQ, LSH, and HNSW. FAISS is widely used for large-scale vector search tasks and supports both CPU and GPU acceleration.
* Annoy is a C++ library for approximate nearest neighbor search in high-dimensional spaces maintained and developed by Spotify implementing the Annoy algorithm. It is designed to be efficient and scalable, making it suitable for large-scale vector data. It works with a forest of random projection trees.
* hnswlib is a C++ library for approximate nearest neighbor search using the Hierarchical Navigable Small World (HNSW) algorithm. It provides fast and memory-efficient indexing and search capabilities for high-dimensional vector data.
* nmslib (Non-Metric Space Library) is an open-source library that provides efficient similarity search in non-metric spaces. It supports various indexing algorithms like HNSW, SW-graph, and SPTAG.
* SPTAG by Microsoft implements a distributed approximate nearest neighborhood search (ANN). It comes with kd-tree and relative neighborhood graph (SPTAG-KDT) and balanced k-means tree and relative neighborhood graph (SPTAG-BKT).

Both nmslib and hnswlib are maintained by Leo Boytsov, who works as a senior research scientist at Amazon, and Yury Malkov.There are a lot more libraries. You can see an overview at [https://github.com/erikbern/ann-benchmarks](https://github.com/erikbern/ann-benchmarks)

#### Vector databases

A **vector database** is a type of database that is specifically designed to handle vector embeddings making it easier to search and query data objects. It offers additional features such as data management, metadata storage and filtering, and scalability. While a vector storage focuses solely on storing and retrieving vector embeddings, a vector database provides a more comprehensive solution for managing and querying vector data. Vector databases can be particularly useful for applications that involve large amounts of data and require flexible and efficient search capabilities across various types of vectorized data, such as text, images, audio, video, and more.

> **Vector databases** can be used to store and serve machine learning models and their corresponding embeddings. The primary application is **similarity search** (also: **semantic search**), where efficiently search through large volumes of text, images, or videos, identifying objects matching the query based on the vector representation. This is particularly useful in applications such as document search, reverse image search, and recommendation systems.
>
> > Other use cases for vector databases are continually expanding as the technology evolves, however, some common use cases for vector databases include:

* Anomaly Detection: Vector databases can be used to detect anomalies in large datasets by comparing the vector embeddings of data points. This can be valuable in fraud detection, network security, or monitoring systems where identifying unusual patterns or behaviors is crucial.
* Personalization: Vector databases can be used to create personalized recommendation systems by finding similar vectors based on user preferences or behavior.
* Natural Language Processing (NLP): Vector databases are widely used in NLP tasks such as sentiment analysis, text classification, and semantic search. By representing text as vector embeddings, it becomes easier to compare and analyze textual data.

These databases are popular because they are optimized for scalability and representing and retrieving data in high-dimensional vector spaces. Traditional databases are not designed to efficiently handle large-dimensional vectors, such as those used to represent images or text embeddings.The characteristics of vector databases include:

1. Efficient retrieval of similar vectors: Vector databases excel at finding close embeddings or similar points in a high-dimensional space. This makes them ideal for tasks like reverse image search or similarity-based recommendations.
2. Specialized for specific tasks: Vector databases are designed to perform a specific task, such as finding close embeddings. They are not general-purpose databases and are tailored to handle large amounts of vector data efficiently.
3. Support for high-dimensional spaces: Vector databases can handle vectors with thousands of dimensions, allowing for complex representations of data. This is crucial for tasks like natural language processing or image recognition.
4. Enable advanced search capabilities: With vector databases, it becomes possible to build powerful search engines that can search for similar vectors or embeddings. This opens up possibilities for applications like content recommendation systems or semantic search.

Overall, vector databases offer a specialized and efficient solution for handling large-dimensional vector data, enabling tasks like similarity search and advanced search capabilities.The market for open-source software and databases is currently thriving due to several factors. Firstly, artificial intelligence (AI) and data management have become crucial for businesses, leading to a high demand for advanced database solutions. In the database market, there is a history of new types of databases emerging and creating new market categories. These market creators often dominate the industry, attracting significant investments from venture capitalists (VCs). For example, MongoDB, Cockroach, Neo4J, and Influx are all examples of successful companies that introduced innovative database technologies and achieved substantial market share. The popular Postgres has an extension for efficient vector search: pg\_embedding. Using the Hierarchical Navigable Small Worlds (HNSW) it provides a faster and more efficient alternative to the pgvector extension with IVFFlat indexing.VCs are actively seeking the next groundbreaking type of database, and vector databases, such as Chroma and Marqo, have the potential to be the next big thing. This creates a competitive landscape where companies can raise significant amounts of funding to develop and scale their products.Some examples of vector databases are listed in this table:

| **Database provider** | **Description**                                                                                 | **Business model**       | **First released**                                                                                                                | **License** | **Indexing**                             | **Organization**           |
| --------------------- | ----------------------------------------------------------------------------------------------- | ------------------------ | --------------------------------------------------------------------------------------------------------------------------------- | ----------- | ---------------------------------------- | -------------------------- |
| Chroma                | Commercial open-source embedding store                                                          | (Partly Open) SaaS model | 2022                                                                                                                              | Apache-2.0  | HNSW                                     | Chroma Inc                 |
| Qdrant                | Managed/Self-hosted vector search engine and database with extended filtering support           | (Partly Open) SaaS model | 2021                                                                                                                              | Apache 2.0  | HNSW                                     | Qdrant Solutions GmbH      |
| Milvus                | Vector database built for scalable similarity search                                            | (Partly Open) SaaS       | 2019                                                                                                                              | BSD         | IVF, HNSW, PQ, and more                  | Zilliz                     |
| Weaviate              | Cloud-native vector database that stores both objects and vectors                               | Open SaaS                | started in 2018 as a traditional graph database, first released in 2019                                                           | BSD         | custom HNSW algorithm that supports CRUD | SeMI Technologies          |
| Pinecone              | Fast and scalable applications using embeddings from AI models                                  | SaaS                     | first released in 2019                                                                                                            | proprietary | built on top of Faiss                    | Pinecone Systems Inc       |
| Vespa                 | Commercial Open Source vector database which supports vector search, lexical search, and search | Open SaaS                | originally a web search engine (alltheweb), acquired by Yahoo! in 2003 and later developed into and open sourced as Vespa in 2017 | Apache 2.0  | HNSW, BM25                               | Yahoo!                     |
| Marqo                 | Cloud-native commercial Open Source search and analytics engine                                 | Open SaaS                | 2022                                                                                                                              | Apache 2.0  | HNSW                                     | S2Search Australia Pty Ltd |

Figure 5.5: Vector databases.

I took the liberty to highlight for each search engine the following perspectives:

* Value proposition. What is the unique feature that makes the whole vector search engine stand out from the crowd?
* Business model. General type of this engine: vector database, big data platform. Managed / Self-hosted.
* Indexing. What algorithm approach to similarity / vector search was taken by this search engine and what unique capabilities it offers?
* License: is it open or close source?

I’ve omitted other aspects such as Architecture, for example support for sharding or in-memory processing. There are many vector database providers. I’ve omitted many solutions such as FaissDB or Hasty.ai, and focused on a few ones, which are integrated in LangChain.For the open-source databases, the Github star histories give a good idea of their popularity and traction. Here’s the plot over time (source: star-history.com):

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file40.png" alt="Figure 5.6: Star history of open-source vector databases on Github." height="1006" width="2114"><figcaption><p>Figure 5.6: Star history of open-source vector databases on Github.</p></figcaption></figure>

You can see that milvus is very popular, however other libraries such as qdrant, weviate, and chroma have been catching up.In LangChain, a vector storage can be implemented using the `vectorstores` module. This module provides various classes and methods for storing and querying vectors. One example of a vector store implementation in LangChain is the Chroma vector store. Let’s see two examples for this!

**Chroma**

This vector store is optimized for storing and querying vectors using Chroma as a backend. Chroma takes over for encoding and comparing vectors based on their angular similarity.To use Chroma in LangChain, you need to follow these steps:

1. Import the necessary modules:

```
from langchain.vectorstores import Chroma  
from langchain.embeddings import OpenAIEmbeddings
```

1. Create an instance of Chroma and provide the documents (splits) and the embedding method:

```
vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())
```

The documents (or splits, as seen in chapter 4) will be embedded and stored in the Chroma vector database. We’ll discuss document loaders in another section of this chapter. We can use other embedding integrations or we can feed embeddings like this:

```
vector_store = Chroma()
# Add vectors to the vector store:
vector_store.add_vectors(vectors)
```

Here, vectors is a list of numerical vectors (embeddings) that you want to store.We can query the vector store to retrieve similar vectors:

```
similar_vectors = vector_store.query(query_vector, k)
```

Here, `query_vector` is the vector you want to find similar vectors to, and k is the number of similar vectors you want to retrieve.

**Pinecone**

Here are the steps to integrate Pinecone with LangChain:

1. Start by installing the Pinecone Python client library. You can do this by running the following command in the terminal: `pip install pinecone`.
2. Import Pinecone in your python app: `import pinecone`.
3. Connect to Pinecone: To connect to the Pinecone service, you need to provide your API key. You can obtain an API key by signing up on the Pinecone website. Once you have the API key, pass it to the pinecone wrapper or set it as an environment variable:&#x20;

```
pinecone.init()
```

1. Create a search index like this:

```
Docsearch = Pinecone.from_texts([“dog”, “cat”], embeddings)
```

The embeddings could be `OpenAIEmbeddings`, for example.

1. Now we can find the most similar documents for a query by similarity:

```
docs = docsearch.similarity_search(“terrier”, include_metadata=True)
```

These documents, we can then query again or use in a question answering chain as we’ve seen in _Chapter 4_, _Question Answering_.In LangChain, we can load our documents from many sources and in a bunch of formats through the integrated document loaders. You can use the LangChain integration hub to browse and select the appropriate loader for your data source. Once you have selected the loader, you can load the document using the specified loader. Let’s briefly look at document loaders in LangChain!

#### Document loaders

Document loaders are used to load data from a source as **Document** objects, which consist of text and associated metadata. There are different types of integrations available, such as document loaders for loading a simple .txt file (`TextLoader`), loading the text contents of a web page (`WebBaseLoader`), articles from Arxiv (`ArxivLoader`), or loading a transcript of a YouTube video (`YoutubeLoader`). For webpages, the `Diffbot` integration gives a clean extraction of the content. Other integations exist for images such as providing image captions (`ImageCaptionLoader`).Document loaders have a `load()` method that loads data from the configured source and returns it as documents. They may also have a `lazy_load()` method for loading data into memory as and when they are needed.Here is an example of a document loader for loading data from a text file:

```
from langchain.document_loaders import TextLoader
loader = TextLoader(file_path="path/to/file.txt")
documents = loader.load()
```

The `documents` variable will contain the loaded documents, which can be accessed for further processing. Each document consists of the `page_content` (the text content of the document) and `metadata` (associated metadata such as the source URL or title).Similarly, we can load documents from Wikipedia:

```
from langchain.document_loaders import WikipediaLoader
loader = WikipediaLoader("LangChain")
documents = loader.load()
```

It’s important to note that the specific implementation of document loaders may vary depending on the programming language or framework being used.In LangChain, vector retrieval in agents or chains is done via retrievers, which access the vector storage. Let’s now how this works.

#### Retrievers in LangChain

Retrievers in LangChain are a type of component that is used to search and retrieve information from a given index. In the context of LangChain, a principal type of retriever is a `vectorstore` retriever. This type of retriever utilizes a vector store as a backend, such as Chroma, to index and search embeddings. Retrievers play a crucial role in question answering over documents, as they are responsible for retrieving relevant information based on the given query.Here are a few examples of retrievers:

* BM25 Retriever: This retriever uses the BM25 algorithm to rank documents based on their relevance to a given query. It is a popular information retrieval algorithm that takes into account term frequency and document length.
* TF-IDF Retriever: This retriever uses the TF-IDF (Term Frequency-Inverse Document Frequency) algorithm to rank documents based on the importance of terms in the document collection. It assigns higher weights to terms that are rare in the collection but frequent in a specific document.
* Dense Retriever: This retriever uses dense embeddings to retrieve documents. It encodes documents and queries into dense vectors and calculates the similarity between them using cosine similarity or other distance metrics.
* kNN retriever: This utilizes the well-known k-nearest neighbors’ algorithm to retrieve relevant documents based on their similarity to a given query.

These are just a few examples of retrievers available in LangChain. Each retriever has its own strengths and weaknesses, and the choice of retriever depends on the specific use case and requirements.For example, to use the kNN retriever, you need to create a new instance of the retriever and provide it with a list of texts. Here is an example of how to create a kNN retriever using embeddings from OpenAI:

```
from langchain.retrievers import KNNRetriever  
from langchain.embeddings import OpenAIEmbeddings  
words = ["cat", "dog", "computer", "animal"]
retriever = KNNRetriever.from_texts(words, OpenAIEmbeddings())
```

Once the retriever is created, you can use it to retrieve relevant documents by calling the `get_relevant_documents()` method and passing a query string. The retriever will return a list of documents that are most relevant to the query.Here is an example of how to use the kNN retriever:

```
result = retriever.get_relevant_documents("dog")  
print(result)
```

This will output a list of documents that are relevant to the query. Each document contains the page content and metadata:

```
[Document(page_content='dog', metadata={}),
 Document(page_content='animal', metadata={}),
 Document(page_content='cat', metadata={}),
 Document(page_content='computer', metadata={})]
```

There are a few more specialized retrievers in LangChain such as retrievers from Arxiv, Pubmed, or Wikipedia. For example, the purpose of an **Arxiv retriever** is to retrieve scientific articles from the Arxiv.org archive. It is a tool that allows users to search for and download scholarly articles in various fields such as physics, mathematics, computer science, and more. The functionality of an arxiv retriever includes specifying the maximum number of documents to be downloaded, retrieving relevant documents based on a query, and accessing metadata information of the retrieved documents.A **Wikipedia retriever** allows users to retrieve Wikipedia pages or documents from the Wikipedia website. The purpose of a Wikipedia retriever is to provide easy access to the vast amount of information available on Wikipedia and enable users to extract specific information or knowledge from it.A **PubMed retriever** is a component in LangChain that helps to incorporate biomedical literature retrieval into their language model applications. PubMed contains millions of citations for biomedical literature from various sources.In LangChain, the `PubMedRetriever` class is used to interact with the PubMed database and retrieve relevant documents based on a given query. The `get_relevant_documents()` method of the class takes a query as input and returns a list of relevant documents from PubMed.Here’s an example of how to use the PubMed retriever in LangChain:

```
from langchain.retrievers import PubMedRetriever  
retriever = PubMedRetriever()  
documents = retriever.get_relevant_documents("COVID")
for document in documents:
    print(document.metadata["title"])
```

In this example, the `get_relevant_documents()` method is called with the query “COVID”. The method then retrieves relevant documents related to the query from PubMed and returns them as a list. I am getting the following titles as printed output:

```
The COVID-19 pandemic highlights the need for a psychological support in systemic sclerosis patients.
Host genetic polymorphisms involved in long-term symptoms of COVID-19.
Association Between COVID-19 Vaccination and Mortality after Major Operations.
```

A custom retriever can be implemented in LangChain by creating a class that inherits from the `BaseRetriever` abstract class. The class should implement the `get_relevant_documents()` method, which takes a query string as input and returns a list of relevant documents.Here is an example of how a retriever can be implemented:

```
from langchain.retriever import BaseRetriever  
from langchain.schema import Document  
class MyRetriever(BaseRetriever):  
    def get_relevant_documents(self, query: str) -> List[Document]:  
        # Implement your retrieval logic here  
        # Retrieve and process documents based on the query  
        # Return a list of relevant documents  
        relevant_documents = []  
        # Your retrieval logic goes here…  
        return relevant_documents
```

You can customize this method to perform any retrieval operations you need, such as querying a database or searching through indexed documents.Once you have implemented your retriever class, you can create an instance of it and call the `get_relevant_documents()` method to retrieve relevant documents based on a query.Let’s implement a chatbot with a retriever!

### Implementing a chatbot!

We’ll implement a chatbot now. We start from a similar template as in chapter 4, Question Answering. Same as in the previous chapter, we’ll assume you have the environment installed with the necessary libraries and the API keys as per the instructions in chapter 3, _Getting Started with LangChain_. To implement a simple chatbot in LangChain, you can follow this recipe:

1. Load the document
2. Create a vector storage
3. Set up a chatbot with retrieval from the vector storage

We’ll generalize this with several formats and make this available through an interface in a web browser through Streamlit. You’ll be able to drop in your document and start asking questions. In production, for a corporate deployment for customer engagement, you can imagine that these documents are already loaded in and your vector storage can just be static.Let’s start with the document reader. As mentioned, we want to be able to read different formats:

```
from typing import Any
from langchain.document_loaders import (
  PyPDFLoader, TextLoader, 
  UnstructuredWordDocumentLoader, 
  UnstructuredEPubLoader
)
class EpubReader(UnstructuredEPubLoader):
    def __init__(self, file_path: str | list[str], ** kwargs: Any):
        super().__init__(file_path, **kwargs, mode="elements", strategy="fast")
class DocumentLoaderException(Exception):
    pass
class DocumentLoader(object):
    """Loads in a document with a supported extension."""
    supported_extentions = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".epub": EpubReader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader
    }
```

This gives us interfaces to read PDF, text, EPUB, and word documents with different extensions. We’ll now implement the loader logic.

```
import logging
import pathlib
from langchain.schema import Document
def load_document(temp_filepath: str) -> list[Document]:
    """Load a file and return it as a list of documents."""
    ext = pathlib.Path(temp_filepath).suffix
    loader = DocumentLoader.supported_extentions.get(ext)
    if not loader:
        raise DocumentLoaderException(
            f"Invalid extension type {ext}, cannot load this type of file"
        )
    loader = loader(temp_filepath)
    docs = loader.load()
    logging.info(docs)
    return docs
```

This doesn’t handle a lot of errors at the moment, but this can be extended if needed. Now we can make this loader available from the interface and connect it to vector storage.

```
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.schema import Document, BaseRetriever
def configure_retriever(docs: list[Document]) -> BaseRetriever:
    """Retriever to use."""
    # Split each document documents:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    # Create embeddings and store in vectordb:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Single call to the huggingface model with all texts:
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
    # Define retriever:
    return vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})
```

DocArray is a Python package that provides a high-level API for representing and manipulating multimodal data. It provides various features like advanced indexing, comprehensive serialization protocols, a unified Pythonic interface, and more. Further, it offers efficient and intuitive handling of multimodal data for tasks such as natural language processing, computer vision, and audio processing.We can initialize the DocArray in-memory vector storage with different distance metrics such as cosine and Euclidean – cosine is the default.For the retriever, we have two main options:

1. Similarity-search: We can retrieve document according to similarity, or
2. Maximum Marginal Relevance (MMR): We can apply diversity-based re-ranking of documents during retrieval to get results that cover different results from the documents retrieved so far.

In the similarity-search, we can set a similarity score threshold. We’ve opted for MMR, which should give us better generation. We’ve set the parameter `k` as 2, which means we can get 2 documents back from retrieval.Retrieval can be improved by **contextual compression**, a technique where retrieved documents are compressed and irrelevant information is filtered out. Instead of returning the full documents as-is, contextual compression uses the context of the given query to extract and return only the relevant information. This helps to reduce the cost of processing and improve the quality of responses in retrieval systems.The base compressor is responsible for compressing the contents of individual documents based on the context of the given query. It uses a language model, such as GPT-3, to perform the compression. The compressor can filter out irrelevant information and return only the relevant parts of the document.The base retriever is the document storage system that retrieves the documents based on the query. It can be any retrieval system, such as a search engine or a database. When a query is made to the Contextual Compression Retriever, it first passes the query to the base retriever to retrieve relevant documents. Then, it uses the base compressor to compress the contents of these documents based on the context of the query. Finally, the compressed documents, containing only the relevant information, are returned as the response.We have a few options for contextual compression:

1. `LLMChainExtractor` – this passes over the returned documents and extracts from each only the relevant content.
2. `LLMChainFilter` – this is slightly simpler; it only filters only the relevant documents (rather than the content from the documents).
3. `EmbeddingsFilter` – this applies a similarity filter based on document and the query in terms of embeddings.

The first two compressors require an LLM to call, which means it can be slow and costly. Therefore, the `EmbeddingsFilter` can be a more efficient alternative. We can integrate compression here with a simple switch statement at the end (replacing the return statement):

```
if not use_compression:
    return retriever
embeddings_filter = EmbeddingsFilter(
  embeddings=embeddings, similarity_threshold=0.76
)
return ContextualCompressionRetriever(
  base_compressor=embeddings_filter,
  base_retriever=retriever
)
```

For our chosen compressor, the `EmbeddingsFilter`, we need to include two more additional imports:

```
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
```

We can feed the `use_compression` parameter through the `configure_qa_chain()` to the `configure_retriever()` method (not shown here).Now that we have the mechanism to create the retriever. We can set up the chat chain:

```
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
def configure_chain(retriever: BaseRetriever) -> Chain:
    """Configure chain with a retriever."""
    # Setup memory for contextual conversation
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # Setup LLM and QA chain; set temperature low to keep hallucinations in check
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", temperature=0, streaming=True
    )
    # Passing in a max_tokens_limit amount automatically
    # truncates the tokens when prompting your llm!
    return ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory, verbose=True, max_tokens_limit=4000
    )
```

One final thing for the retrieval logic is taking the documents and passing them to the retriever setup:

```
import os
import tempfile
def configure_qa_chain(uploaded_files):
    """Read documents, configure retriever, and the chain."""
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        docs.extend(load_document(temp_filepath))
    retriever = configure_retriever(docs=docs)
    return configure_chain(retriever=retriever)
```

Now that we have the logic of the chatbot, we need to set up the interface. As mentioned, we’ll use streamlit again:

```
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
st.set_page_config(page_title="LangChain: Chat with Documents", page_icon="🦜")
st.title("🦜 LangChain: Chat with Documents")
uploaded_files = st.sidebar.file_uploader(
    label="Upload files",
    type=list(DocumentLoader.supported_extentions.keys()),
    accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload documents to continue.")
    st.stop()
qa_chain = configure_qa_chain(uploaded_files)
assistant = st.chat_message("assistant")
user_query = st.chat_input(placeholder="Ask me anything!")
if user_query:
    stream_handler = StreamlitCallbackHandler(assistant)
    response = qa_chain.run(user_query, callbacks=[stream_handler])
    st.markdown(response)
```

This gives us a chatbot with retrieval usable via a visual interface with drop-in of custom documents as needed that you can ask questions about.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file41.png" alt="Figure 5.7: Chatbot interface with document loaders in different formats." height="574" width="1548"><figcaption><p>Figure 5.7: Chatbot interface with document loaders in different formats.</p></figcaption></figure>

You can see the full implementation on Github. You can play around with the chatbot, and see how it works, and when it doesn’t.It’s important to note that LangChain has limitations on input size and cost. You may need to consider workarounds to handle larger knowledge bases or optimize the cost of API usage. Additionally, fine-tuning models or hosting the LLM in-house can be more complex and less accurate compared to using commercial solutions. We’ll look at these use cases in _Chapters 8_, _Conditioning and Fine-Tuning_, and _9_, _LLM applications in Production_.Let’s have a look at the available memory mechanisms in LangChain.

#### Memory mechanisms

A memory is a component in the LangChain framework that allows chatbots and language models to remember previous interactions and information. It is essential in applications like chatbots because it enables the system to maintain context and continuity in conversations.We need memory in chatbots to:

1. Remember previous interactions: Memory allows chatbots to retain information from previous messages exchanged with the user. This helps in understanding user queries and providing relevant responses.
2. Maintain context: By recalling previous interactions, chatbots can maintain context and continuity in conversations. This allows for more natural and coherent conversations with users.
3. Extract knowledge: Memory enables the system to extract knowledge and insights from a sequence of chat messages. This information can then be used to improve the performance and accuracy of the chatbot.

In summary, memory is crucial in chatbots to create a more personalized and human-like conversational experience by remembering and building upon past interactions.Here’s a practical example in Python that demonstrates how to use the LangChain memory feature.

```
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
# Creating a conversation chain with memory
memory = ConversationBufferMemory()
chain = ConversationChain(memory=memory)
# User inputs a message
user_input = "Hi, how are you?"
# Processing the user input in the conversation chain
response = chain.predict(input=user_input)
# Printing the response
print(response)
# User inputs another message
user_input = "What's the weather like today?"
# Processing the user input in the conversation chain
response = chain.predict(input=user_input)
# Printing the response
print(response)
# Printing the conversation history stored in memory
print(memory.chat_memory.messages)
```

In this example, we create a conversation chain with memory using ConversationBufferMemory, which is a simple wrapper that stores the messages in a variable. The user’s inputs are processed using the `predict()` method of the conversation chain. The conversation chain retains the memory of previous interactions, allowing it to provide context-aware responses.Instead of constructing the memory separately from the chain, we could have simplified:

```
conversation = ConversationChain(
    llm=llm, 
    verbose=True, 
    memory=ConversationBufferMemory()
)
```

We are setting verbose to True in order to see the prompts.After processing the user inputs, we print the response generated by the conversation chain. Additionally, we print the conversation history stored in memory using `memory.chat_memory.messages`. The `save_context()` method is used to store inputs and outputs. You can use the `load_memory_variables()` method to view the stored content. To get the history as a list of messages, a `return_messages` parameter is set to `True`. We’ll see examples of this in this section.`ConversationBufferWindowMemory` is a memory type provided by LangChain that keeps track of the interactions in a conversation over time. Unlike `ConversationBufferMemory`, which retains all previous interactions, `ConversationBufferWindowMemory` only keeps the last K interactions, where K is the window size specified.Here’s a simple example of how to use `ConversationBufferWindowMemory` in LangChain:

```
from langchain.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(k=1)
```

In this example, the window size is set to 1, meaning that only the last interaction will be stored in memory.We can use the `save_context()` method to save the context of each interaction. It takes two arguments: user\_input and model\_output. These represent the user’s input and the corresponding model’s output for a given interaction.

```
memory.save_context({"input": "hi"}, {"output": "whats up"})
memory.save_context({"input": "not much you"}, {"output": "not much"})
```

We can see the message with `memory.load_memory_variables({})`In LangChain, we can integrate a knowledge graph to enhance the capabilities of language models and enable them to leverage structured knowledge during text generation and inference.

> A **knowledge graph** is a structured knowledge representation model that organizes information in the form of entities, attributes, and relationships. It represents knowledge as a graph, where entities are represented as nodes and relationships between entities are represented as edges.
>
> > Prominent examples of knowledge graphs include Wikidata, which captures structured information from Wikipedia, and Google’s Knowledge Graph, which powers search results with rich contextual information.

In a knowledge graph, entities can be any concept, object, or thing in the world, and attributes describe properties or characteristics of these entities. Relationships capture the connections and associations between entities, providing contextual information and enabling semantic reasoning.There’s functionality in LangChain for knowledge graphs for retrieval, however, LangChain also provides memory components to automatically create a knowledge graph based on our conversation messages. Instantiate the `ConversationKGMemory` class and pass your language model (LLM) instance as the llm parameter:

```
from langchain.memory import ConversationKGMemory
from langchain.llms import OpenAI
llm = OpenAI(temperature=0)
memory = ConversationKGMemory(llm=llm)
```

As the conversation progresses, we can save relevant information from the knowledge graph into the memory using the `save_context()` function of the `ConversationKGMemory`. We can also customize the conversational memory in LangChain, which involves modifying the prefixes used for the AI and Human messages, as well as updating the prompt template to reflect these changes.To customize the conversational memory, you can follow these steps:

1. Import the necessary classes and modules from LangChain:

```
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
llm = OpenAI(temperature=0)
```

1. Define a new prompt template that includes the customized prefixes. You can do this by creating a \`PromptTemplate\` object with the desired template string.

```
template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
Current conversation:
{history}
Human: {input}
AI Assistant:"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
conversation = ConversationChain(
    prompt=PROMPT,
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory(ai_prefix="AI Assistant"),
)
```

In this example, the AI prefix is set to AI Assistant instead of the default AI.The `ConversationSummaryMemory` is a type of memory in LangChain that generates a summary of the conversation as it progresses. Instead of storing all messages verbatim, it condenses the information, providing a summarized version of the conversation. This is particularly useful for long conversation chains where including all previous messages might exceed token limits.To use `ConversationSummaryMemory`, first create an instance of it, passing the language model (llm) as an argument. Then, use the `save_context()` method to save the interaction context, which includes the user input and AI output. To retrieve the summarized conversation history, use the `load_memory_variables()` method.Example:

```
from langchain.memory import ConversationSummaryMemory
from langchain.llms import OpenAI
# Initialize the summary memory and the language model
memory = ConversationSummaryMemory(llm=OpenAI(temperature=0))
# Save the context of an interaction
memory.save_context({"input": "hi"}, {"output": "whats up"})
# Load the summarized memory
memory.load_memory_variables({})
```

LangChain also allows combining multiple memory strategies using the CombinedMemory class. This is useful when you want to maintain different aspects of the conversation history. For instance, one memory could be used to store the complete conversation log and

```
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, CombinedMemory, ConversationSummaryMemory
# Initialize language model (with desired temperature parameter)
llm = OpenAI(temperature=0)
# Define Conversation Buffer Memory (for retaining all past messages)
conv_memory = ConversationBufferMemory(memory_key="chat_history_lines", input_key="input")
# Define Conversation Summary Memory (for summarizing conversation)
summary_memory = ConversationSummaryMemory(llm=llm, input_key="input")
# Combine both memory types
memory = CombinedMemory(memories=[conv_memory, summary_memory])
# Define Prompt Template
_DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
Summary of conversation:
{history}
Current conversation:
{chat_history_lines}
Human: {input}
AI:"""
PROMPT = PromptTemplate(input_variables=["history", "input", "chat_history_lines"], template=_DEFAULT_TEMPLATE)
# Initialize the Conversation Chain
conversation = ConversationChain(llm=llm, verbose=True, memory=memory, prompt=PROMPT)
# Start the conversation
conversation.run("Hi!")
```

In this example, we first instantiate the language model and the different types of memories we're using - `ConversationBufferMemory` for retaining the full conversation history and `ConversationSummaryMemory` for creating a summary of the conversation. We then combine these memories using CombinedMemory. We also define a prompt template that accommodates our memory usage and finally, we create and run the `ConversationChain` by providing our language model, memory, and prompt to it.The `ConversationSummaryBufferMemory` is used to keep a buffer of recent interactions in memory, and compiles old interactions into a summary instead of completely flushing them out. The threshold for flushing interactions is determined by token length and not by the number of interactions. To use this, the memory buffer needs to be instantiated with the LLM model and a `max_token_limit`. `ConversationSummaryBufferMemory` offers a method called `predict_new_summary()` which can be used directly to generate a conversation summary.Zep is a memory store and search engine that is designed to store, summarize, embed, index, and enrich chatbot or AI app histories. It provides developers with a simple and low-latency API to access and manipulate the stored data.A practical example of using Zep is to integrate it as the long-term memory for a chatbot or AI app. By using the `ZepMemory` class, developers can initialize a `ZepMemory` instance with the Zep server URL, API key, and a unique session identifier for the user. This allows the chatbot or AI app to store and retrieve chat history or other relevant information.For example, in Python, you can initialize a ZepMemory instance as follows:

```
from langchain.memory import ZepMemory  
# Set this to your Zep server URL  
ZEP_API_URL = "http://localhost:8000"  
ZEP_API_KEY = "<your JWT token>"  # optional  
session_id = str(uuid4())  # This is a unique identifier for the user  
# Set up ZepMemory instance  
memory = ZepMemory(  
    session_id=session_id,  
    url=ZEP_API_URL,  
    api_key=ZEP_API_KEY,  
    memory_key="chat_history",  
)
```

Once the memory is set up, you can use it in your chatbot’s chain or with your AI agent to store and retrieve chat history or other relevant information.Overall, Zep simplifies the process of persisting, searching, and enriching chatbot or AI app histories, allowing developers to focus on developing their AI applications rather than building memory infrastructure.

### Don’t say anything stupid!

The role of moderation in chatbots is to ensure that the bot’s responses and conversations are appropriate, ethical, and respectful. It involves implementing mechanisms to filter out offensive or inappropriate content, as well as discouraging abusive behavior from users.In the context of moderation, a constitution refers to a set of guidelines or rules that govern the behavior and responses of the chatbot. It outlines the standards and principles that the chatbot should adhere to, such as avoiding offensive language, promoting respectful interactions, and maintaining ethical standards. The constitution serves as a framework for ensuring that the chatbot operates within the desired boundaries and provides a positive user experience.Moderation and having a constitution in chatbots are crucial for creating a safe, respectful, and inclusive environment for users, protecting brand reputation, and complying with legal obligations.Moderation and having a constitution are important in chatbots for several reasons:

1. Ensuring ethical behavior: Chatbots have the potential to interact with a wide range of users, including vulnerable individuals. Moderation helps ensure that the bot’s responses are ethical, respectful, and do not promote harmful or offensive content.
2. Protecting users from inappropriate content: Moderation helps prevent the dissemination of inappropriate or offensive language, hate speech, or any content that may be harmful or offensive to users. It creates a safe and inclusive environment for users to interact with the chatbot.
3. Maintaining brand reputation: Chatbots often represent a brand or organization. By implementing moderation, the developer can ensure that the bot’s responses align with the brand’s values and maintain a positive reputation.
4. Preventing abusive behavior: Moderation can discourage users from engaging in abusive or improper behavior. By implementing rules and consequences, such as the "two strikes" rule mentioned in the example, the developer can discourage users from using provocative language or engaging in abusive behavior.
5. Legal compliance: Depending on the jurisdiction, there may be legal requirements for moderating content and ensuring that it complies with laws and regulations. Having a constitution or set of guidelines helps the developer adhere to these legal requirements.

You can add a moderation chain to an LLMChain to ensure that the generated output from the language model is not harmful.If the content passed into the moderation chain is deemed harmful, there are different ways to handle it. You can choose to throw an error in the chain and handle it in your application, or you can return a message to the user explaining that the text was harmful. The specific handling method depends on your application’s requirements.In LangChain, first, you would create an instance of the `OpenAIModerationChain` class, which is a pre-built moderation chain provided by LangChain. This chain is specifically designed to detect and filter out harmful content.

```
from langchain.chains import OpenAIModerationChain  
moderation_chain = OpenAIModerationChain()
```

Next, you would create an instance of the LLMChain class, which represents your language model chain. This is where you define your prompt and interact with the language model.

```
from langchain.chains import LLMChain  
llm_chain = LLMChain(model_name="gpt-3.5-turbo")
```

To append the moderation chain to the language model chain, you can use the `SequentialChain` class. This class allows you to chain multiple chains together in a sequential manner.

```
from langchain.chains import SequentialChain  
chain = SequentialChain([llm_chain, moderation_chain])
```

Now, when you want to generate text using the language model, you would pass your input text through the moderation chain first, and then through the language model chain.

```
input_text = "Can you generate a story for me?"  
output = chain.generate(input_text)
```

The moderation chain will evaluate the input text and filter out any harmful content. If the input text is deemed harmful, the moderation chain can either throw an error or return a message indicating that the text is not allowed. I’ve added an example for moderation to the chatbot app on Github.Further, Guardrails can be used to define the behavior of the language model on specific topics, prevent it from engaging in discussions on unwanted topics, guide the conversation along a predefined path, enforce a particular language style, extract structured data, and more.

> In the context of large language models, **guardrails** (**rails** for short) refer to specific ways of controlling the output of the model. They provide a means to add programmable constraints and guidelines to ensure the output of the language model aligns with desired criteria.

Here are a few ways guardrails can be used:

* Controlling Topics: Guardrails allow you to define the behavior of your language model or chatbot on specific topics. You can prevent it from engaging in discussions on unwanted or sensitive topics like politics.
* Predefined Dialog Paths: Guardrails enable you to define a predefined path for the conversation. This ensures that the language model or chatbot follows a specific flow and provides consistent responses.
* Language Style: Guardrails allow you to specify the language style that the language model or chatbot should use. This ensures that the output is in line with your desired tone, formality, or specific language requirements.
* Structured Data Extraction: Guardrails can be used to extract structured data from the conversation. This can be useful for capturing specific information or performing actions based on user inputs.

Overall, guardrails provide a way to add programmable rules and constraints to large language models and chatbots, making them more trustworthy, safe, and secure in their interactions with users. By appending the moderation chain to your language model chain, you can ensure that the generated text is moderated and safe for use in your application.

### Summary

In chapter 4, we discussed Retrieval-Augmented Generation (RAG), which involves the utilization of external tools or knowledge resources such as document corpora. In that chapter, we focused on the process. In this chapter, the focus is on methods relevant to building chatbots based on RALMs, and, more particularly, the use of external tools to retrieve relevant information that can be incorporated into the content generation. The main sections of the chapter include an introduction to chatbots, retrieval and vector mechanisms, implementing a chatbot, memory mechanisms, and the importance of appropriate responses.The chapter started with an overview over chatbots. We discussed the evolution and current state of chatbots and language processing models (LLMs) highlighting the practical implications and enhancements of the capabilities of the current technology. We then discussed the importance of proactive communication and the technical implementations required for context, memory, and reasoning. We explored retrieval mechanisms, including vector storage, with the goal to improve the accuracy of chatbot responses. We went into details with methods for loading documents and information, including vector storage and embedding. Additionally, we discussed memory mechanisms for maintaining knowledge and the state of ongoing conversations are examined. The chapter concludes with a discussion on moderation, emphasizing the importance of ensuring responses are respectful and aligned with organizational values. We’ve implemented a chatbot in this chapter, which explores a few features discussed in this chapter, and can serve as a starting point to investigate issues like memory and context, moderation of speech, but can also be interesting for issues like hallucinations or others.In chapter 9, we’ll discuss how you can train LLMs on your documents as another way of conditioning models on your data!Let’s see if you remember some of the key takeaways from this chapter!

### Questions

Please have a look to see if you can come up with the answers to these questions from memory. I’d recommend you go back to the corresponding sections of this chapter, if you are unsure about any of them:

1. Please name 5 different chatbots!
2. What are some important aspects in developing a chatbot?
3. What does RALMs stand for?
4. What is an embedding?
5. What is vector search?
6. What is a vector database?
7. Please name 5 different vector databases!
8. What is a retriever in LangChain?
9. What is memory and what are the memory options in LangChain?
10. What is moderation, what’s a constitution, and how do they work?
