# Topic Modeling with BERTopic

0\. load library

```python
import re
import jaconv
import unicodedata
import pandas as pd
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction

import MeCab
from MeCab import Tagger

# For WordCloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import japanize_matplotlib
```

## 1. Data used for this tutorial

Here is the data overview:

<figure><img src="../../.gitbook/assets/image (197).png" alt=""><figcaption></figcaption></figure>

This is already a column named "分類" where human annotated the results. And "review" is the actually comment left by users for specific products they bought on the E-commerce website.

Here is the value\_counts of "分類" looks like:

```
分類
期待値ギャップ（風味）            512
期待値ギャップ（サイズ、量）       39
期待値ギャップ（見た目、形、色）    29
配送遅延                        28
その他                         12
機能不全（その他）               10
期待値ギャップ（食感）            10
数量不足                        7
梱包不備                        3
商品相違                        2
決済不備                        1
```

## 2. Start Using BERTopic

BERTopic official documentation: [https://maartengr.github.io/BERTopic/getting\_started/quickstart/quickstart.html](https://maartengr.github.io/BERTopic/getting\_started/quickstart/quickstart.html)

### 2.1 Simple Test

```python
topic_model = BERTopic()
topics, probs = topic_model.fit_transform(reviews)

print("Number of topics:", len(set(topics)))
# Number of topics: 2
```

Visualize the topic word socres:

```python
topic_model.visualize_barchart(n_words = 10)
```

<figure><img src="../../.gitbook/assets/image (198).png" alt=""><figcaption></figcaption></figure>

### 2.2 Set models for each step

<figure><img src="../../.gitbook/assets/image (200).png" alt=""><figcaption></figcaption></figure>

#### 2.2.1 Step1 - Extract embeddings

```python
# from huggingface
from sentence_transformers import models
from sentence_transformers import SentenceTransformer

MY_MODEL = "cl-tohoku/bert-base-japanese-v3"   # Japanese Embedding Model
# https://huggingface.co/cl-tohoku/bert-base-japanese-v3

transformer = models.Transformer(MY_MODEL)

pooling = models.Pooling(
    transformer.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False)

embedding_model = SentenceTransformer(modules=[transformer, pooling])
```

#### 2.2.2 Step2 - Reduce dimensionality

[https://maartengr.github.io/BERTopic/getting\_started/dim\_reduction/dim\_reduction.html#umap](https://maartengr.github.io/BERTopic/getting\_started/dim\_reduction/dim\_reduction.html#umap)

```python
umap_model = UMAP(random_state=42)
```

#### 2.2.3 Step3 - Cluster reduced embeddings

[https://maartengr.github.io/BERTopic/getting\_started/clustering/clustering.html#hdbscan](https://maartengr.github.io/BERTopic/getting\_started/clustering/clustering.html#hdbscan)

```python
hdbscan_model = HDBSCAN()
```

#### 2.2.4 Step4 - Tokenize topics

```python
def tokenize_jp(text):
    # Read stopwords from file
    with open('stopwords-ja.txt', encoding='utf-8') as f:
        stopwords = set(f.read().split())

    words = MeCab.Tagger("-Owakati").parse(text).split()
    # Remove stopwords from words
    words = [w for w in words if w not in stopwords]    
    return words

vectorizer_model = CountVectorizer(tokenizer=tokenize_jp)
```

#### 2.2.5 Step5 - Create topic representation

[https://maartengr.github.io/BERTopic/api/ctfidf.html](https://maartengr.github.io/BERTopic/api/ctfidf.html)

```python
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
```

#### 2.2.6 Step6 - (Optional) Fine-tune topic representations

We can use a `KeyBERT-Inspired` model to reduce the appearance of stop words. This also often improves the topic representation.

[https://maartengr.github.io/BERTopic/api/representation/keybert.html](https://maartengr.github.io/BERTopic/api/representation/keybert.html)

```python
representation_model = KeyBERTInspired()
```

#### Test Results

```python
topic_model = BERTopic(
    embedding_model=embedding_model,            # Step 1 - Extract embeddings
    umap_model=umap_model,                      # Step 2 - Reduce dimensionality
    hdbscan_model=hdbscan_model,                # Step 3 - Cluster reduced embeddings
    vectorizer_model=vectorizer_model,          # Step 4 - Tokenize topics
    ctfidf_model=ctfidf_model,                  # Step 5 - Extract topic words
    representation_model=representation_model   # Step 6 - Fine-tune topics
)

topics, probs = topic_model.fit_transform(reviews)


print("Number of topics:", len(set(topics)))
# Number of topics: 29
```

```python
topic_model.get_topic_info().sort_values("Count", ascending=False)
```

<figure><img src="../../.gitbook/assets/image (201).png" alt=""><figcaption></figcaption></figure>

In the graph above, Topic "-1" means "others", and we can see this one has highest count (266).

```python
topic_model.visualize_barchart(n_words = 10)
```

<figure><img src="../../.gitbook/assets/image (202).png" alt=""><figcaption></figcaption></figure>

```python
# Word Cloud
create_wordcloud(topic_model, topic=0)
```

<figure><img src="../../.gitbook/assets/image (203).png" alt=""><figcaption></figcaption></figure>

```python
# Word Cloud
create_wordcloud(topic_model, topic=1)
```

<figure><img src="../../.gitbook/assets/image (204).png" alt=""><figcaption></figcaption></figure>

```python
topic_model.visualize_hierarchy()
```

<figure><img src="../../.gitbook/assets/image (206).png" alt=""><figcaption></figcaption></figure>

### 2.3 Parameters and tricks

Ref: [https://maartengr.github.io/BERTopic/getting\_started/tips\_and\_tricks/tips\_and\_tricks.html](https://maartengr.github.io/BERTopic/getting\_started/tips\_and\_tricks/tips\_and\_tricks.html)

#### 2.3.1 Increasing Number of Topics

Set `nr_topics` = 50

```python
topic_model = BERTopic(
    nr_topics=50,
    
    embedding_model=embedding_model,            # Step 1 - Extract embeddings
    umap_model=umap_model,                      # Step 2 - Reduce dimensionality
    hdbscan_model=hdbscan_model,                # Step 3 - Cluster reduced embeddings
    vectorizer_model=vectorizer_model,          # Step 4 - Tokenize topics
    ctfidf_model=ctfidf_model,                  # Step 5 - Extract topic words
    representation_model=representation_model   # Step 6 - Fine-tune topics
)

topics, probs = topic_model.fit_transform(reviews)

print("Number of topics:", len(set(topics)))
# Number of topics: 29
```

Not working, maybe we need to look deep into the dimensional reduction model.

#### 2.3.2 Reducing number of topics

Set `nr_topics` = 10

```
topic_model = BERTopic(
    nr_topics=10,
    
    embedding_model=embedding_model,            # Step 1 - Extract embeddings
    umap_model=umap_model,                      # Step 2 - Reduce dimensionality
    hdbscan_model=hdbscan_model,                # Step 3 - Cluster reduced embeddings
    vectorizer_model=vectorizer_model,          # Step 4 - Tokenize topics
    ctfidf_model=ctfidf_model,                  # Step 5 - Extract topic words
    representation_model=representation_model   # Step 6 - Fine-tune topics
)

topics, probs = topic_model.fit_transform(reviews)

print("Number of topics:", len(set(topics)))
# Number of topics: 10
```

```python
topic_model.get_topic_info().sort_values("Count", ascending=False)
```

<figure><img src="../../.gitbook/assets/image (207).png" alt=""><figcaption></figcaption></figure>

```python
topic_model.visualize_barchart(n_words = 10)
```

<figure><img src="../../.gitbook/assets/image (208).png" alt=""><figcaption></figcaption></figure>

```python
topic_model.visualize_hierarchy()
```

<figure><img src="../../.gitbook/assets/image (209).png" alt=""><figcaption></figcaption></figure>

#### 2.3.3 Diversify topic representation

After having calculated our top n words per topic there might be many words that essentially mean the same thing. we can use `bertopic.representation.MaximalMarginalRelevance` in BERTopic to diversify words in each topic such that we limit the number of duplicate words we find in each topic.

We do this by specifying a value between 0 and 1, with 0 being not at all diverse and 1 being completely diverse.

```python
from bertopic.representation import MaximalMarginalRelevance

representation_model = MaximalMarginalRelevance(diversity=0.8)

topic_model = BERTopic(
    embedding_model=embedding_model,            # Step 1 - Extract embeddings
    umap_model=umap_model,                      # Step 2 - Reduce dimensionality
    hdbscan_model=hdbscan_model,                # Step 3 - Cluster reduced embeddings
    vectorizer_model=vectorizer_model,          # Step 4 - Tokenize topics
    ctfidf_model=ctfidf_model,                  # Step 5 - Extract topic words
    representation_model=representation_model   # Step 6 - Fine-tune topics
)

topics, probs = topic_model.fit_transform(reviews)

print("Number of topics:", len(set(topics)))
# Number of topics: 29
```

```python
topic_model.visualize_barchart(n_words = 5)
```

<figure><img src="../../.gitbook/assets/image (210).png" alt=""><figcaption></figcaption></figure>

```python
topic_model.visualize_hierarchy()
```

<figure><img src="../../.gitbook/assets/image (211).png" alt=""><figcaption></figcaption></figure>

