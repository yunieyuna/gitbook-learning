# Chapter 1. Categorizing Text

## Chapter 1. Categorizing Text

## A NOTE FOR EARLY RELEASE READERS

With Early Release ebooks, you get books in their earliest form—the author’s raw and unedited content as they write—so you can take advantage of these technologies long before the official release of these titles.

This will be the 2nd chapter of the final book. Please note that the GitHub repo will be made active later on.

If you have comments about how we might improve the content and/or examples in this book, or if you notice missing material within this chapter, please reach out to the editor at _mcronin@oreilly.com_.

One of the most common tasks in natural language processing, and machine learning in general, is classification. The goal of the task is to train a model to assign a label or class to some input text. Categorizing text is used across the world for a wide range of applications, from sentiment analysis and intent detection to extracting entities and detecting language.

The impact of Large Language Models on categorization cannot be understated. The addition of these models has quickly settled as the default for these kinds of tasks.

In this chapter, we will discuss a variety of ways to use Large Language Modeling for categorizing text. Due to the broad field of text categorization, a variety of techniques, as well as use cases, will be discussed. This chapter also serves as a nice introduction to LLMs as most of them can be used for classification.

We will focus on leveraging pre-trained LLMs, models that already have been trained on large amounts of data and that can be used for categorizing text. Fine-tuning these models for categorizing text and domain adaptation will be discussed in more detail in Chapter 10.

Let’s start by looking at the most basic application and technique, fully-supervised text classification.

## Supervised Text Classification

Classification comes in many flavors, such as few-shot and zero-shot classification which we will discuss later in this chapter, but the most frequently used method is a fully supervised classification. This means that during training, every input has a target category from which the model can learn.

For supervised classification using textual data as our input, there is a common procedure that is typically followed. As illustrated in [Figure 1-1](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch01.html#fig\_1\_an\_example\_of\_supervised\_classification\_can\_we\_pr), we first convert our textual input to numerical representations using a feature extraction model. Traditionally, such a model would represent text as a bag of words, simply counting the number of times a word appears in a document. In this book, however, we will be focusing on LLMs as our feature extraction model.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/categorizing_text_559117_01.png" alt="An example of supervised classification. Can we predict whether a movie review is either positive or negative" height="440" width="600"><figcaption></figcaption></figure>

**Figure 1-1. An example of supervised classification. Can we predict whether a movie review is either positive or negative?**

Then, we train a classifier on the numerical representations, such as embeddings (remember from Chapter X?), to classify the textual data. The classifier can be a number of things, such as a neural network or logistic regression. It can even be the classifier used in many Kaggle competitions, namely XGBoost!

In this pipeline, we always need to train the classifier but we can choose to fine-tune either the entire LLM, certain parts of it, or keep it as is. If we choose not to fine-tune it all, we refer to this procedure as **freezing its layers**. This means that the layers cannot be updated during the training process. However, it may be beneficial to **unfreeze** at least some of its layers such that the Large Language Models can be **fine-tuned** for the specific classification task. This process is illustrated in [Figure 1-2](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch01.html#fig\_2\_a\_common\_procedure\_for\_supervised\_text\_classificat).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/categorizing_text_559117_02.png" alt="A common procedure for supervised text classification. We convert our textual input data to numerical representations through feature extraction. Then  a classifier is trained to predict labels." height="81" width="600"><figcaption></figcaption></figure>

**Figure 1-2. A common procedure for supervised text classification. We convert our textual input data to numerical representations through feature extraction. Then, a classifier is trained to predict labels.**

### Model Selection

We can use an LLM to represent the text to be fed into our classifier. The choice of this model, however, may not be as straightforward as you might think. Models differ in the language they can handle, their architecture, size, inference speed, architecture, accuracy for certain tasks, and many more differences exist.

BERT is a great underlying architecture for representing tasks that can be fine-tuned for a number of tasks, including classification. Although there are generative models that we can use, like the well-known Generated Pretrained Transformers (GPT) such as ChatGPT, BERT models often excel at being fine-tuned for specific tasks. In contrast, GPT-like models typically excel at a broad and wide variety of tasks. In a sense, it is specialization versus generalization.

Now that we know to choose a BERT-like model for our supervised classification task, which are we going to use? BERT has a number of variations, including BERT, RoBERTa, DistilBERT, ALBERT, DeBERTa, and each architecture has been pre-trained in numerous forms, from training in certain domains to training for multi-lingual data. You can find an overview of some well-known Large Language Models in [Figure 1-3](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch01.html#fig\_3\_a\_timeline\_of\_common\_large\_language\_model\_releases).

Selecting the right model for the job can be a form of art in itself. Trying thousands of pre-trained models that can be found on HuggingFace’s Hub is not feasible so we need to be efficient with the models that we choose. Having said that, there are a number of models that are a great starting point and give you an idea of the base performance of these kinds of models. Consider them solid baselines:

* [BERT-base-uncased](https://huggingface.co/bert-base-uncased)
* [Roberta-base](https://huggingface.co/roberta-base)
* [Distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)
* [Deberta-base](https://huggingface.co/microsoft/deberta-base)
* [BERT-tiny](https://huggingface.co/prajjwal1/bert-tiny)
* [Albert-base-v2](https://huggingface.co/albert-base-v2)

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/categorizing_text_559117_03.png" alt="A timeline of common Large Language Model releases." height="239" width="600"><figcaption></figcaption></figure>

**Figure 1-3. A timeline of common Large Language Model releases.**

In this section, we will be using “bert-base-cased” for some of our examples. Feel free to replace “bert-base-cased” with any of the models above. Play around with different models to get a feeling for the trade-off in performance/training speed.



### Data

Throughout this chapter, we will be demonstrating many techniques for categorizing text. The dataset that we will be using to train and evaluate the models is the [“rotten\_tomatoes”](https://huggingface.co/datasets/rotten\_tomatoes); pang2005seeing) dataset. It contains roughly 5000 positive and 5000 negative movie reviews from [Rotten Tomatoes](https://www.rottentomatoes.com/).

We load the data and convert it to a `pandas dataframe` for easier control:

```
import pandas as pd
from datasets import load_dataset
tomatoes = load_dataset("rotten_tomatoes")
 
# Pandas for easier control
train_df = pd.DataFrame(tomatoes["train"])
eval_df = pd.DataFrame(tomatoes["test"])
```

**TIP**

Although this book focuses on LLMs, it is highly advised to compare these examples against classic, but strong baselines such as representing text with TF-IDF and training a LogisticRegression classifier on top of that.



### Classification Head

Using the Rotten Tomatoes dataset, we can start with the most straightforward example of a predictive task, namely binary classification. This is often applied in sentiment analysis, detecting whether a certain document is positive or negative. This can be customer reviews with a label indicating whether that review is positive or negative (binary). In our case, we are going to predict whether a movie review is negative (0) or positive (1).

Training a classifier with transformer-based models generally follows a two-step approach:

First, as we show in [Figure 1-4](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch01.html#fig\_4\_first\_we\_start\_by\_using\_a\_generic\_pre\_trained\_llm), we take an existing transformer model and use it to convert our textual data to numerical representations.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/categorizing_text_559117_04.png" alt="First  we start by using a generic pre trained LLM  e.g.  BERT  to convert our textual data into more numerical representations. During training  we will  freeze  the model such that its weights will not be updated. This speeds up training significantly but is generally less accurate." height="144" width="600"><figcaption></figcaption></figure>

**Figure 1-4. First, we start by using a generic pre-trained LLM (e.g., BERT) to convert our textual data into more numerical representations. During training, we will “freeze” the model such that its weights will not be updated. This speeds up training significantly but is generally less accurate.**

Second, as shown in [Figure 1-5](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch01.html#fig\_5\_after\_fine\_tuning\_our\_llm\_we\_train\_a\_classifier\_o), we put a classification head on top of the pre-trained model. This classification head is generally a single linear layer that we can fine-tune.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/categorizing_text_559117_05.png" alt="After fine tuning our LLM  we train a classifier on the numerical representations and labels. Typically  a Feed Forward Neural Network is chosen as the classifier." height="191" width="600"><figcaption></figcaption></figure>

**Figure 1-5. After fine-tuning our LLM, we train a classifier on the numerical representations and labels. Typically, a Feed Forward Neural Network is chosen as the classifier.**

These two steps each describe the same model since the classification head is added directly to the BERT model. As illustrated in [Figure 1-6](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch01.html#fig\_6\_we\_adopt\_the\_bert\_model\_such\_that\_its\_output\_embed), our classifier is nothing more than a pre-trained LLM with a linear layer attached to it. It is feature extraction and classification in one.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/categorizing_text_559117_06.png" alt="We adopt the BERT model such that its output embeddings are fed into a classification head. This head generally consists of a linear layer but might include dropout beforehand." height="429" width="570"><figcaption></figcaption></figure>

**Figure 1-6. We adopt the BERT model such that its output embeddings are fed into a classification head. This head generally consists of a linear layer but might include dropout beforehand.**

**NOTE**

In Chapter 10, we will use the same pipeline shown in Figures 2-4 and 2-5 but will instead fine-tune the Large Language Model. There, we will go more in-depth into how fine-tuning works and why it improves upon the pipeline as shown here. For now, it is essential to know that fine-tuning this model together with the classification head improves the accuracy during the classification task. The reason for this is that it allows the Large Language Model to better represent the text for classification purposes. It is fine-tuned toward the domain-specific texts.

#### Example

To train our model, we are going to be using the [simpletransformers package](https://github.com/ThilinaRajapakse/simpletransformers). It abstracts most of the technical difficulty away so that we can focus on the classification task at hand. We start by initializing our model:

<pre><code>from simpletransformers.classification import ClassificationModel, ClassificationArgs
 
# Train only the classifier layers
model_args = ClassificationArgs()
<strong>model_args.train_custom_parameters_only = True
</strong>model_args.custom_parameter_groups = [
    {
        "params": ["classifier.weight"],
        "lr": 1e-3,
    },
    {
        "params": ["classifier.bias"],
        "lr": 1e-3,
        "weight_decay": 0.0,
    },
]
 
# Initializing pre-trained BERT model
model = ClassificationModel("bert", "bert-base-cased", args=model_args)
</code></pre>

We have chosen the popular “bert-base-cased” but as mentioned before, there are many other models that we could have chosen instead. Feel free to play around with models to see how it influences performance.

Next, we can train the model on our training dataset and predict the labels of our evaluation dataset:

```
import numpy as np
from sklearn.metrics import f1_score
 
# Train the model
model.train_model(train_df)
 
# Predict unseen instances
result, model_outputs, wrong_predictions = model.eval_model(eval_df, f1=f1_score)
y_pred = np.argmax(model_outputs, axis=1)
```

Now that we have trained our model, all that is left is evaluation:

<pre><code><strong>>>> from sklearn.metrics import classification_report
</strong>>>> print(classification_report(eval_df.label, y_pred))
              precision    recall  f1-score   support
 
           0       0.84      0.86      0.85       533
           1       0.86      0.83      0.84       533
 
    accuracy                           0.85      1066
   macro avg       0.85      0.85      0.85      1066
weighted avg       0.85      0.85      0.85      1066
</code></pre>

Using a pre-trained BERT model for classification gives us an F-1 score of 0.85. We can use this score as a baseline throughout the examples in this section.

**TIP**

The `simpletransformers` package has a number of easy-to-use features for different tasks. For example, you could also use it to create a custom Named Entity Recognition model with only a few lines of code.

### Pre-Trained Embeddings

Unlike the example shown before, we can approach supervised classification in a more classical form. Instead of freezing layers before training and using a feed-forward neural network on top of it, we can completely separate feature extraction and classification training.

This two-step approach completely separates feature extraction from classification:

First, as we can see in [Figure 1-7](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch01.html#fig\_7\_first\_we\_use\_an\_llm\_that\_was\_trained\_specifically), we perform our feature extraction with an LLM, SBERT ([https://www.sbert.net/](https://www.sbert.net/)), which is trained specifically to create embeddings.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/categorizing_text_559117_07.png" alt="First  we use an LLM that was trained specifically to generate accurate numerical representations. These tend to be better representative vectors than we receive from a general Transformer based model like BERT." height="147" width="600"><figcaption></figcaption></figure>

**Figure 1-7. First, we use an LLM that was trained specifically to generate accurate numerical representations. These tend to be better representative vectors than we receive from a general Transformer-based model like BERT.**

Second, as shown in [Figure 1-8](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch01.html#fig\_8\_using\_the\_embeddings\_as\_our\_features\_we\_train\_a\_l), we use the embeddings as input for a logistic regression model. We are completely separating the feature extraction model from the classification model.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/categorizing_text_559117_08.png" alt="Using the embeddings as our features  we train a logistic regression model on our training data." height="176" width="600"><figcaption></figcaption></figure>

**Figure 1-8. Using the embeddings as our features, we train a logistic regression model on our training data.**

In contrast to our previous example, these two steps each describe a different model. SBERT for generating features, namely embeddings, and a Logistic Regression as the classifier. As illustrated in Figure 2-9, our classifier is nothing more than a pre-trained LLM with a linear layer attached to it.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/categorizing_text_559117_09.png" alt="The classifier is a separate model that leverages the embeddings from SBERT to learn from." height="336" width="600"><figcaption></figcaption></figure>

**Figure 1-9. The classifier is a separate model that leverages the embeddings from SBERT to learn from.**

#### Example

Using sentence-transformer, we can create our features before training our classification model:

```
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-mpnet-base-v2')
train_embeddings = model.encode(train_df.text)
eval_embeddings = model.encode(eval_df.text)
```

We created the embeddings for our training (train\_df) and evaluation (eval\_df) data. Each instance in the resulting embeddings is represented by 768 values. We consider these values the features on which we can train our model.

Selecting the model can be straightforward. Instead of using a feed-forward neural network, we can go back to the basics and use a Logistic Regression instead:

```
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=42).fit(train_embeddings, train_df.label)
```

In practice, you can use any classifier on top of our generated embeddings, like Decision Trees or Neural Networks.

Next, let’s evaluate our model:

<pre><code><strong>>>> from sklearn.metrics import classification_report
</strong>>>> y_pred = clf.predict(eval_embeddings)
>>> print(classification_report(eval_df.label, y_pred))
 
              precision    recall  f1-score   support
 
           0       0.84      0.86      0.85       151
           1       0.86      0.83      0.84       149
 
    accuracy                           0.85       300
   macro avg       0.85      0.85      0.85       300
weighted avg       0.85      0.85      0.85       300
</code></pre>

Without needing to fine-tune our LLM, we managed to achieve an F1-score of 0.85. This is especially impressive since it is a much smaller model compared to our previous example.



## Zero-shot Classification

We started this chapter with examples where all of our training data has labels. In practice, however, this might not always be the case. Getting labeled data is a resource-intensive task that can require significant human labor. Instead, we can use zero-shot classification models. This method is a nice example of transfer learning where a model trained for one task is used for a task different than what it was originally trained for. An overview of zero-shot classification is given in Figure 2-11. Note that this pipeline also demonstrates the capabilities of performing multi-label classification if the probabilities of multiple labels exceed a given threshold.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/categorizing_text_559117_10.png" alt="Figure 2 11. In zero shot classification  the LLM is not trained on any of the candidate labels. It learned from different labels and generalized that information to the candidate labels." height="178" width="600"><figcaption></figcaption></figure>

**Figure 1-10. Figure 2-11. In zero-shot classification, the LLM is not trained on any of the candidate labels. It learned from different labels and generalized that information to the candidate labels.**

Often, zero-shot classification tasks are used with pre-trained LLMs that use natural language to describe what we want our model to do. It is often referred to as an emergent feature of LLMs as the models increase in size (wei2022emergent). As we will see later in this chapter on classification with generative models, GPT-like models can often do these kinds of tasks quite well.

### Pre-Trained Embeddings

As we have seen in our supervised classification examples, embeddings are a great and often accurate way of representing textual data. When dealing with no labeled documents, we have to be a bit creative in how we are going to be using pre-trained embeddings. A classifier cannot be trained since we have no labeled data to work with.

Fortunately, there is a trick that we can use. We can describe our labels based on what they should represent. For example, a negative label for movie reviews can be described as “This is a negative movie review”. By describing and embedding the labels and documents, we have data that we can work with. This process, as illustrated in [Figure 1-11](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch01.html#fig\_11\_to\_embed\_the\_labels\_we\_first\_need\_to\_give\_them\_a), allows us to generate our own target labels without the need to actually have any labeled data.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/categorizing_text_559117_11.png" alt="To embed the labels  we first need to give them a description. For example  the description of a negative label could be  A negative movie review . This description can then be embedded through sentence transformers. In the end  both labels as well as all the documents are embedded." height="372" width="600"><figcaption></figcaption></figure>

**Figure 1-11. To embed the labels, we first need to give them a description. For example, the description of a negative label could be “A negative movie review”. This description can then be embedded through sentence-transformers. In the end, both labels as well as all the documents are embedded.**

To assign labels to documents, we can apply cosine similarity to the document label pairs. Cosine similarity, which will often be used throughout this book, is a similarity measure that checks how similar two vectors are to each other.

It is the cosine of the angle between vectors which is calculated through the dot product of the embeddings and divided by the product of their lengths. It definitely sounds more complicated than it is and, hopefully, the illustration in [Figure 1-12](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch01.html#fig\_12\_the\_cosine\_similarity\_is\_the\_angle\_between\_two\_vec) should provide additional intuition.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/categorizing_text_559117_12.png" alt="The cosine similarity is the angle between two vectors or embeddings. In this example  we calculate the similarity between a document and the two possible labels  positive and negative." height="433" width="600"><figcaption></figcaption></figure>

**Figure 1-12. The cosine similarity is the angle between two vectors or embeddings. In this example, we calculate the similarity between a document and the two possible labels, positive and negative.**

For each document, its embedding is compared to that of each label. The label with the highest similarity to the document is chosen. [Figure 1-13](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch01.html#fig\_13\_after\_embedding\_the\_label\_descriptions\_and\_the\_doc) gives a nice example of how a document is assigned a label.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/categorizing_text_559117_13.png" alt="After embedding the label descriptions and the documents  we can use cosine similarity for each label document pair. For each document  the label with the highest similarity to the document is chosen." height="306" width="600"><figcaption></figcaption></figure>

**Figure 1-13. After embedding the label descriptions and the documents, we can use cosine similarity for each label document pair. For each document, the label with the highest similarity to the document is chosen.**

#### Example

We start by generating the embeddings for our evaluation dataset. These embeddings are generated with sentence-transformers as they are quite accurate and are computationally quite fast.

```
from sentence_transformers import SentenceTransformer, util
 
# Create embeddings for the input documents
model = SentenceTransformer('all-mpnet-base-v2')
eval_embeddings = model.encode(eval_df.text)
```

Next, embeddings of the labels need to be generated. The labels, however, do not have a textual representation that we can leverage so we will instead have to name the labels ourselves.

Since we are dealing with positive and negative movie reviews, let’s name the labels “A positive review” and “A negative review”. This allows us to embed those labels:

```
# Create embeddings for our labels
label_embeddings = model.encode(["A negative review", "A positive review"])
```

Now that we have embeddings for our reviews and the labels, we can apply cosine similarity between them to see which label fits best with which review. Doing so requires only a few lines of code:

```
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
 
# Find the best matching label for each document
sim_matrix = cosine_similarity(eval_embeddings, label_embeddings)
y_pred = np.argmax(sim_matrix, axis=1)
```

And that is it! We only needed to come up with names for our labels to perform our classification tasks. Let’s see how well this method works:

```
>>> print(classification_report(eval_df.label, y_pred))
 
              precision    recall  f1-score   support
 
           0       0.83      0.77      0.80       151
           1       0.79      0.84      0.81       149
 
    accuracy                           0.81       300
   macro avg       0.81      0.81      0.81       300
weighted avg       0.81      0.81      0.81       300
```

An F-1 score of 0.81 is quite impressive considering we did not use any labeled data at all! This just shows how versatile and useful embeddings are especially if you are a bit creative with how they are used.

Let’s put that creativity to the test. We decided upon “A negative/positive review” as the names of our labels but that can be improved. Instead, we can make them a bit more concrete and specific towards our data by using “A very negative/positive movie review” instead. This way, the embedding will capture that it is a movie review and will focus a bit more on the extremes of the two labels.

We use the code we used before to see whether this actually works:

```
>>> # Create embeddings for our labels
>>> label_embeddings = model.encode(["A very negative movie review", "A very positive movie review"])
>>> 
>>> # Find the best matching label for each document
>>> sim_matrix = cosine_similarity(eval_embeddings, label_embeddings)
>>> y_pred = np.argmax(sim_matrix, axis=1)
>>> 
>>> # Report results
>>> print(classification_report(eval_df.label, y_pred))
 
              precision    recall  f1-score   support
 
           0       0.90      0.74      0.81       151
           1       0.78      0.91      0.84       149
 
    accuracy                           0.83       300
   macro avg       0.84      0.83      0.83       300
weighted avg       0.84      0.83      0.83       300
```

By only changing the phrasing of the labels, we increased our F-1 score quite a bit!

**TIP**

In the example, we applied zero-shot classification by naming the labels and embedding them. When we have a few labeled examples, embedding them and adding them to the pipeline could help increase the performance. For example, we could average the embeddings of the labeled examples together with the label embeddings. We could even do a voting procedure by creating different types of representations (label embeddings, document embeddings, averaged embeddings, etc.) and see which label is most often found. This would make our zero-shot classification example a few-shot approach.



### Natural Language Inference

Zero-shot classification can also be done using natural language inference (NLI), which refers to the task of investigating whether, for a given premise, a hypothesis is true (entailment) or false (contradiction). [Figure 1-14](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch01.html#fig\_14\_an\_example\_of\_natural\_language\_inference\_nli\_th) shows a nice example how they relate to one another.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/categorizing_text_559117_14.png" alt="An example of natural language inference  NLI . The hypothesis is contradicted by the premise and is not relevant to one another." height="263" width="600"><figcaption></figcaption></figure>

**Figure 1-14. An example of natural language inference (NLI). The hypothesis is contradicted by the premise and is not relevant to one another.**

NLI can be used for zero-shot classification by being a bit creative with how the premise/hypothesis pair is used, as demonstrated in [Figure 1-15](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch01.html#fig\_15\_an\_example\_of\_zero\_shot\_classification\_with\_natura). We use the input document, the review that we want to extract sentiment from and use that as our premise (yin2019benchmarking). Then, we create a hypothesis asking whether the premise is about our target label. In our movie reviews example, the hypothesis could be: “This example is a positive movie review”. When the model finds it to be an entailment, we can label the review as positive and negative when it is a contradiction. Using NLI for zero-shot classification is illustrated with an example in [Figure 1-15](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch01.html#fig\_15\_an\_example\_of\_zero\_shot\_classification\_with\_natura).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/categorizing_text_559117_15.png" alt="An example of zero shot classification with natural language inference  NLI . The hypothesis is supported by the premise and the model will return that the review is indeed a positive movie review." height="263" width="600"><figcaption></figcaption></figure>

**Figure 1-15. An example of zero-shot classification with natural language inference (NLI). The hypothesis is supported by the premise and the model will return that the review is indeed a positive movie review.**

#### Example

With transformers, loading and running a pre-trained NLI model is straightforward. Let’s select “`facebook/bart-large-mnli`” as our pre-trained model. The model was trained on more than 400k premise/hypothesis pairs and should serve well for our use case.

**NOTE**

Over the course of the last few years, Hugging Face has strived to become the Github of Machine Learning by hosting pretty much everything related to Machine Learning. As a result, there is a large amount of pre-trained models available on their hub. For zero-shot classification tasks, you can follow this link: [https://huggingface.co/models?pipeline\_tag=zero-shot-classification](https://huggingface.co/models?pipeline\_tag=zero-shot-classification\&sort=downloads).

We load in our transformers pipeline and run it on our evaluation dataset:

```
from transformers import pipeline
 
# Pre-trained MNLI model
pipe = pipeline(model="facebook/bart-large-mnli")
 
# Candidate labels
candidate_labels_dict = {"negative movie review": 0, "positive movie review": 1}
candidate_labels = ["negative movie review", "positive movie review"]
 
# Create predictions
predictions = pipe(eval_df.text.values.tolist(), candidate_labels=candidate_labels)
```

Since this is a zero-shot classification task, no training is necessary for us to get the predictions that we are interested in. The predictions variable contains not only the prediction but also a score indicating the probability of a candidate label (hypothesis) to entail the input document (premise).

<pre><code><strong>>>> from sklearn.metrics import classification_report
</strong><strong>>>> y_pred = [candidate_labels_dict[prediction["labels"][0]] for prediction in predictions]
</strong>>>> print(classification_report(eval_df.label, y_pred))
 
              precision    recall  f1-score   support
 
           0       0.77      0.89      0.83       151
           1       0.87      0.74      0.80       149
 
    accuracy                           0.81       300
   macro avg       0.82      0.81      0.81       300
weighted avg       0.82      0.81      0.81       300
</code></pre>

Without any fine-tuning whatsoever, it received an F1-score of 0.81. We might be able to increase this value depending on how we phrase the candidate labels. For example, see what happens if the candidate labels were simply “negative” and “positive” instead.

**TIP**

Another great pre-trained model for zero-shot classification is sentence-transformers’ cross-encoder, namely '`cross-encoder/nli-deberta-base`‘. Since training a sentence-transformers model focuses on pairs of sentences, it naturally lends itself to zero-shot classification tasks that leverage premise/hypothesis pairs.

## Classification with Generative Models

Classification with generative large language models, such as OpenAI’s GPT models, works a bit differently from what we have done thus far. Instead of fine-tuning a model to our data, we use the model and try to guide it toward the type of answers that we are looking for.

This guiding process is done mainly through the prompts that you give such as a model. Optimizing the prompts such that the model understands what kind of answer you are looking for is called **prompt engineering**. This section will demonstrate how we can leverage generative models to perform a wide variety of classification tasks.

This is especially true for extremely large language models, such as GPT-3. An excellent paper and read on this subject, “Language Models are Few-Shot Learners”, describes that these models are competitive on downstream tasks whilst needing less task-specific data (brown2020language).

### In-Context Learning

What makes generative models so interesting is their ability to follow the prompts they are given. A generative model can even do something entirely new by merely being shown a few examples of this new task. This process is also called in-context learning and refers to the process of having the model learn or do something new without actually fine-tuning it.

For example, if we ask a generative model to write a haiku (a traditional Japanese poetic form), it might not be able to if it has not seen a haiku before. However, if the prompt contains a few examples of what a haiku is, then the model “learns” from that and is able to create haikus.

We purposely put “learning” in quotation marks since the model is not actually learning but following examples. After successfully having generated the haikus, we would still need to continuously provide it with examples as the internal model was not updated. These examples of in-context learning are shown in [Figure 1-16](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch01.html#fig\_16\_zero\_shot\_and\_few\_shot\_classification\_through\_prom) and demonstrate the creativity needed to create successful and performant prompts.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/categorizing_text_559117_16.png" alt="Zero shot and few shot classification through prompt engineering with generative models." height="455" width="600"><figcaption></figcaption></figure>

**Figure 1-16. Zero-shot and few-shot classification through prompt engineering with generative models.**

In-context learning is especially helpful in few-shot classification tasks where we have a small number of examples that the generative model can follow.

Not needing to fine-tune the internal model is a major advantage of in-context learning. These generative models are often quite large in size and are difficult to run on consumer hardware let alone fine-tune them. Optimizing your prompts to guide the generative model is relatively low-effort and often does not need somebody well-versed in generative AI.

#### Example

Before we go into the examples of in-context learning, we first create a function that allows us to perform prediction with OpenAI’s GPT models.

<pre><code><strong>from tenacity import retry, stop_after_attempt, wait_random_exponential
</strong> 
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
<strong>def gpt_prediction(prompt, document, model="gpt-3.5-turbo-0301"):
</strong>  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content":   prompt.replace("[DOCUMENT]", document)}
  ]
  response = openai.ChatCompletion.create(model=model, messages=messages, temperature=0)
<strong>  return response["choices"][0]["message"]["content"]
</strong></code></pre>

This function allows us to pass a specific `prompt` and `document` for which we want to create a prediction. The `tenacity` module that you also see here allows us to deal with rate limit errors, which happen when you call the API too often. OpenAI, and other external APIs, often want to limit the rate at which you call their API so as not to overload their servers.

This `tenacity` module is essentially a “retrying module” that allows us to retry API calls in specific ways. Here, we implemented something called **exponential backoff** to our `gpt_prediction` function. Exponential backoff performs a short sleep when we hit a rate limit error and then retries the unsuccessful request. Every time the request is unsuccessful, the sleep length is increased until the request is successful or we hit a maximum number of retries.

One easy way to avoid rate limit errors is to automatically retry requests with a random exponential backoff. Retrying with exponential backoff means performing a short sleep when a rate limit error is hit, then retrying the unsuccessful request. If the request is still unsuccessful, the sleep length is increased and the process is repeated. This continues until the request is successful or until a maximum number of retries is reached.

Lastly, we need to sign in to OpenAI’s API with an API-key that you can get from your account:

```
import openai
openai.api_key = "sk-..."
```

**WARNING**

When using external APIs, always keep track of your usage. External APIs, such as OpenAI or Cohere, can quickly become costly if you request too often from their APIs.

#### Zero-shot Classification

Zero-shot classification with generative models is essentially what we typically do when interacting with these types of models, simply ask them if they can do something. In our examples, we ask the model whether a specific document is a positive or negative movie review.

To do so, we create a base template for our zero-shot classification prompt and ask the model if it can predict whether a review is positive or negative:

```
# Define a zero-shot prompt as a base
zeroshot_prompt = """Predict whether the following document is a positive or negative movie review:
 
[DOCUMENT]
 
If it is positive say 1 and if it is negative say 0. Do not give any other answers.
"""
```

You might have noticed that we explicitly say to not give any other answers. These generative models tend to have a mind of their own and return large explanations as to why something is or isn’t negative. Since we are evaluating its results, we want either a 0 or a 1 to be returned.

Next, let’s see if it can correctly predict that the review “unpretentious, charming, quickie, original” is positive:

```
# Define a zero-shot prompt as a base
zeroshot_prompt = """Predict whether the following document is a positive or negative movie review:
 
[DOCUMENT]
 
If it is positive say 1 and if it is negative say 0. Do not give any other answers.
"""
 
# Predict the target using GPT
document = "unpretentious , charming , quirky , original"
gpt_prediction(zeroshot_prompt, document)
```

The output indeed shows that the review was labeled by OpenAI’s model as positive! Using this prompt template, we can insert any document at the “\[DOCUMENT]” tag. These models have token limits which means that we might not be able to insert an entire book into the prompt. Fortunately, reviews tend not to be the sizes of books but are often quite short.

Next, we can run this for all reviews in the evaluation dataset and look at its performance. Do note though that this requires 300 requests to OpenAI’s API:

<pre><code><strong>> from sklearn.metrics import classification_report
</strong>> from tqdm import tqdm
>
<strong>> y_pred = [int(gpt_prediction(zeroshot_prompt, doc)) for doc in tqdm(eval_df.text)]
</strong>> print(classification_report(eval_df.label, y_pred))
 
              precision    recall  f1-score   support
 
           0       0.86      0.96      0.91       151
           1       0.95      0.86      0.91       149
 
    accuracy                           0.91       300
   macro avg       0.91      0.91      0.91       300
weighted avg       0.91      0.91      0.91       300
</code></pre>

An F-1 score of 0.91! That is the highest we have seen thus far and is quite impressive considering we did not fine-tune the model at all.

**NOTE**

Although this zero-shot classification with GPT has shown high performance, it should be noted that fine-tuning generally outperforms in-context learning as presented in this section. This is especially true if domain-specific data is involved which the model during pre-training is unlikely to have seen. A model’s adaptability to task-specific nuances might be limited when its parameters are not updated for the task at hand. Preferably, we would want to fine-tune this GPT model on this data to improve its performance even further!

#### Few-shot Classification

In-context learning works especially well when we perform few-shot classification. Compared to zero-shot classification, we simply add a few examples of movie reviews as a way to guide the generative model. By doing so, it has a better understanding of the task that we want to accomplish.

We start by updating our prompt template to include a few hand-picked examples:

```
# Define a few-shot prompt as a base
fewshot_prompt = """Predict whether the following document is a positive or negative moview review:
 
[DOCUMENT]
 
Examples of negative reviews are:
- a film really has to be exceptional to justify a three hour running time , and this isn't .
- the film , like jimmy's routines , could use a few good laughs .
 
Examples of positive reviews are:
- very predictable but still entertaining
- a solid examination of the male midlife crisis .
 
If it is positive say 1 and if it is negative say 0. Do not give any other answers.
"""
```

We picked two examples per class as a quick way to guide the model toward assigning sentiment to movie reviews.

**NOTE**

Since we added a few examples to the prompt, the generative model consumes more tokens and as a result could increase the costs of requesting the API. However, that is relatively little compared to fine-tuning and updating the entire model.

Prediction is the same as before but replacing the zero-shot prompt with the few-shot prompt:

```
# Predict the target using GPT
document = "unpretentious , charming , quirky , original"
gpt_prediction(fewshot_prompt, document)
```

Unsurprisingly, it correctly assigned sentiment to the review. The more difficult or complex the task is, the bigger the effect of providing examples, especially if they are high-quality.

As before, let’s run the improved prompt against the entire evaluation dataset:

<pre><code><strong>>>> predictions = [gpt_prediction(fewshot_prompt, doc) for doc in tqdm(eval_df.text)]
</strong> 
              precision    recall  f1-score   support
 
           0       0.88      0.97      0.92       151
           1       0.96      0.87      0.92       149
 
    accuracy                           0.92       300
   macro avg       0.92      0.92      0.92       300
weighted avg       0.92      0.92      0.92       300
</code></pre>

The F1-score is now 0.92 which is a very slight increase compared to what we had before. This is not unexpected since its score was already quite high and the task at hand was not particularly complex.

**NOTE**

We can extend the examples of in-context learning to multi-label classification by engineering the prompt. For example, we can ask the model to choose one or multiple labels and return them separated by commas.

### Named Entity Recognition

In the previous examples, we have tried to classify entire texts, such as reviews. There are many cases though where we are more interested in specific information inside those texts. We may want to extract certain medications from textual electronic health records or find out which organizations are mentioned in news posts.

These tasks are typically referred to as token classification or Named Entity Recognition (NER) which involves detecting these entities in text. As illustrated in [Figure 1-17](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch01.html#fig\_17\_an\_example\_of\_named\_entity\_recognition\_that\_detect), instead of classifying an entire text, we are now going to classify certain tokens or token sets.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/categorizing_text_559117_17.png" alt="An example of named entity recognition that detects the entities  place  and  time ." height="64" width="600"><figcaption></figcaption></figure>

**Figure 1-17. An example of named entity recognition that detects the entities “place” and “time”.**

When we think about token classification, one major framework comes into mind, namely SpaCy ([https://spacy.io/](https://spacy.io/)). It is an incredible package for performing many industrial-strength NLP applications and has been the go-to framework for NER tasks. So, let’s use it!

#### Example

To use OpenAI’s models with SpaCy, we will first need to save the API key as an environment variable. This makes it easier for SpaCy to access it without the need to save it locally:

```
import os
os.environ['OPENAI_API_KEY'] = "sk-..."
```

Next, we need to configure our SpaCy pipeline. A “task” and a “backend” will need to be defined. The “task” is what we want the SpaCy pipeline to do, which is Named Entity Recognition. The “backend” is the underlying LLM that is used to perform the “task” which is OpenAI’s GPT-3.5-turbo model. In the task, we can create any labels that we would like to extract from our text. Let’s assume that we have information about patients and we would like to extract some personal information but also the disease and symptoms they developed. We create the entities date, age, location, disease, and symptom:

```
import spacy
 
nlp = spacy.blank("en")
 
# Create a Named Entity Recognition Task and define labels
task = {"task": {
            "@llm_tasks": "spacy.NER.v1",
            "labels": "DATE,AGE,LOCATION, DISEASE, SYMPTOM"}}
 
# Choose which backend to use
backend = {"backend": {
            "@llm_backends": "spacy.REST.v1",
            "api": "OpenAI",
            "config": {"model": "gpt-3.5-turbo"}}}
 
# Combine configurations and create SpaCy pipeline
config = task | backend
nlp.add_pipe("llm", config=config)
```

Next, we only need two lines of code to automatically extract the entities that we are interested in:

<pre><code><strong>> doc = nlp("On February 11, 2020, a 73-year-old woman came to the hospital \n and was diagnosed with COVID-19 and has a cough.")
</strong><strong>> print([(ent.text, ent.label_) for ent in doc.ents])
</strong> 
[('February 11', 'DATE'), ('2020', 'DATE'), ('73-year-old', 'AGE'), ('hospital', 'LOCATION'), ('COVID-19', ' DISEASE'), ('cough', ' SYMPTOM')]
</code></pre>

It seems to correctly extract the entities but it is difficult to immediately see if everything worked out correctly. Fortunately, SpaCy has a display function that allows us to visualize the entities found in the document ([Figure 1-18](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch01.html#fig\_18\_the\_output\_of\_spacy\_using\_openai\_s\_gpt\_3\_5\_model)):

```
from spacy import displacy
from IPython.core.display import display, HTML
 
# Display entities
html = displacy.render(doc, style="ent")
display(HTML(html))
```

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/categorizing_text_559117_18.png" alt="The output of SpaCy using OpenAI s GPT 3.5 model. Without any training  it correctly identifies our custom entities." height="65" width="600"><figcaption></figcaption></figure>

**Figure 1-18. The output of SpaCy using OpenAI’s GPT-3.5 model. Without any training, it correctly identifies our custom entities.**

That is much better! Figure 2-X shows that we can clearly see that the model has correctly identified our custom entities. Without any fine-tuning or training of the model, we can easily detect entities that we are interested in.

**TIP**

Training a NER model from scratch with SpaCy is not possible with only a few lines of code but it is also by no means difficult! Their [documentation and tutorials](https://spacy.io/usage/training) are, in our opinions, state-of-the-art and do an excellent job of explaining how to create a custom model.

## Summary

In this chapter, we saw many different techniques for performing a wide variety of classification tasks. From fine-tuning your entire model to no tuning at all! Classifying textual data is not as straightforward as it may seem on the surface and there is an incredible amount of creative techniques for doing so.

In the next chapter, we will continue with classification but focus instead on unsupervised classification. What can we do if we have textual data without any labels? What information can we extract? We will focus on clustering our data as well as naming the clusters with topic modeling techniques.



