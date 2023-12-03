# Chapter 2. LLM Ingredients: Tokenization, Learning Objectives & Architectures

## Chapter 2. LLM Ingredients: Tokenization, Learning Objectives & Architectures

## A NOTE FOR EARLY RELEASE READERS

With Early Release ebooks, you get books in their earliest form—the author’s raw and unedited content as they write—so you can take advantage of these technologies long before the official release of these titles.

This will be the 4th chapter of the final book. Please note that the GitHub repo will be made active later on.

If you have comments about how we might improve the content and/or examples in this book, or if you notice missing material within this chapter, please reach out to the author at [mcronin@oreilly.com](mailto:mcronin@oreilly.com).

In Chapter 3, we dug into the datasets that are used to train the language models of today. Hopefully this foray has underscored how influential pre-training data is to the resulting model. In this chapter, we will go through the remaining ingredients: vocabulary and tokenization, learning objectives, and model architecture.

## Vocabulary and Tokenization

What do you do first when you start learning a new language? You start acquiring its vocabulary, expanding it as you gain more proficiency in the language. Let’s define vocabulary here as

> All the words in a language that are understood by a specific person

The average native English speaker is said to have a vocabulary ranging between [20,000-35,000](https://www.economist.com/johnson/2013/05/29/lexical-facts) words. Similarly, every language model has its own vocabulary, with most vocabulary sizes ranging anywhere between 5,000 to 500,000 _tokens_.

As an example, let us explore the vocabulary of the GPT Neo-X 20B model. Open the file [tokenizer.json](https://huggingface.co/EleutherAI/gpt-neox-20b/resolve/main/tokenizer.json) and ctrl+f for ‘vocab’. You can see that the words comprising the language model vocabulary don’t entirely look like English language words that appear in a dictionary. These word-like units are called ‘types’, and the instantiation of a type (when it appears in a sequence of text) is called a token.

**NOTE**

In recent times, and especially in industry, I have hardly heard anyone use the term ‘type’ except in older NLP textbooks. The term token is broadly used to refer to both the vocabulary units and when it appears in a text sequence. We will henceforth use the word ‘token’ to describe both concepts, even though I personally am not the biggest fan of it.

In the vocabulary file, we see that next to each token is a number, which is called the _input id_ or the _token index_. The vocabulary size of GPT Neo-X is just above 50,000.

The first few hundred tokens are all single character tokens, starting from special characters, digits, capital letters, small letters, and accented characters. Longer words appear later on in the vocabulary. There are a lot of tokens that correspond to just a part of a word, called a _subword_, like ‘impl’, ‘inated’, and so on.

Let’s Ctrl + F for ‘office’. We get nine results -

```
"Ġoffice": 3906
"Ġofficer": 5908
"Ġofficers": 6251
"ĠOffice": 7454
"ĠOfficer": 12743
"Ġoffices": 14145
"office": 30496
"Office": 33577
"ĠOfficers": 37209
```

The Ġ character refers to a space before the word. For instance, in the sentence ‘He stopped going to the office’, the space before the letter ‘o’ is considered part of the token. You can see that the tokens are case-sensitive - there is a separate token for ‘office’ and ‘Office’. Most models these days have case-sensitive vocabularies. Back in the day, BERT came with both a cased and an uncased version.

Cased vocabularies are almost always better, especially when you are training on such a huge body of text such that most tokens are seen by the model enough times so as to learn meaningful embeddings for them. For instance, there is a definite semantic difference between ‘web’ and ‘Web’ and it is good to have separate tokens for them.

Let’s search for some numbers. Ctrl+F for ‘93’. There are only three results

```
"93": 4590
"937": 47508
"930": 48180
```

It seems like not all numbers get their own tokens! Where is the token for 934? It is impractical to give every number its own token, especially if you want to limit your vocabulary size to just 50,000. As discussed in Chapter 2, the vocabulary size determines the size of the embedding layer and we do not want to see it become too large. We will discuss the impact of missing tokens later in this section.

Popular names and places get their own token. There is a token representing Boston, Toronto, and Amsterdam but none representing Mesa or Chennai. There is a token representing Ahmed and Donald, but none for Suhas or Maryam.

You might have noticed that tokens like

```
"]);": 9259
```

exist, indicating that GPT Neo-X is also primed to process programming languages.

## EXERCISE

Go through the [tokenizer.json](https://huggingface.co/EleutherAI/gpt-neox-20b/resolve/main/tokenizer.json) file and explore the vocabulary in detail. Specifically,

* What are some unexpected tokens you see?
* What are the top ten longest tokens?
* Are there tokens representing words from other languages?

How are vocabularies determined? Surely, there was no executive committee holding emergency meetings burning midnight oil, with members making impassioned pleas to include the number _937_ in the vocabulary at the expense of _934_?

Let us revisit the definition of a vocabulary

> All the words in a language that are understood by a specific person

Since we want our language model to be an expert at English, we can just include all words in the English dictionary as part of its vocabulary. Problem solved?

Not nearly. What do you do when you communicate with the language model using a word that it has never seen? This happens a lot more often than you think. New words get invented all the time, words have multiple forms - ‘understanding’, ‘understanding’, ‘understandable’ etc, multiple words can be combined into a single word, and so on. Moreover, there are millions of domain-specific words (biomedical, chemistry etc)

## THE DEFINITION OF A WORD

What exactly is a word, anyway? It is surprisingly very hard to answer this. Conceptually, you could say that a word is the smallest unit of text that has a self-contained meaning. This is not exactly true. For example, the word ‘snowball’ has components that have self-contained meanings of their own. Algorithmically, you can say that a word is just a sequence of characters separated by white space. This isn’t always true either. For example, the word ‘Hong Kong’ is generally regarded as a single word, even if it is separated by white space. Meanwhile the word ‘can’t’ could potentially be regarded as two or three words, even if there is no white space separating them.

**NOTE**

The twitter account [‘NYT first said’](https://twitter.com/NYT\_first\_said) tweets out words when they appear in the New York Times for the first time, excluding proper nouns. An average of 5 new words appear in the American paper of record for the first time each day. On the day I wrote this section, the words were ‘unflippant’, ‘dumbeyed’, ‘dewdrenched’, ‘faceflat’, ‘saporous, and ‘dronescape’. Many of these words might never get added to a dictionary.

A token that doesn’t exist in the vocabulary is called an OOV (Out-of-vocabulary) token. In Chapter 2, we saw how each token is assigned an embedding in the Transformer architecture. The architecture is fixed, and the number of embeddings in the embedding layer equals the size of the vocabulary of the model. Traditionally, OOV tokens were represented using a special \<UNK> token. The \<UNK> token is a placeholder for all tokens that don’t exist in the vocabulary. All OOV tokens share the same embedding (and encode the same meaning), which is undesirable. Moreover, the \<UNK> token cannot be used in generative models. You don’t want your model to output something like

```
‘As a language model, I am trained to <UNK> sequences, and output <UNK> text’.
```

To solve the OOV problem, one possible solution could be to represent tokens in terms of characters instead of words. Each character has its own embedding, and as long as all valid characters are included in the vocabulary, there will never be a chance of encountering an OOV token. However, there are many downsides to this. The number of tokens needed to represent the average sentence becomes much larger. For example, the previous sentence contains 13 tokens with a word tokenization scheme but 81 tokens with a character tokenization scheme. As seen in Chapter 2, the sequence length of a Transformer is limited, and the expanded number of tokens makes both training and inference slower, and reduces the amount of context that can be provided to a model in zero-shot or few-shot settings. Therefore, character-based tokens cannot be adapted without a significant change to the Transformer architecture. There have been attempts to do this including CANINE, ByT5, CharFormer, which we will discuss later in this section.

So, the middle ground and the best of both worlds (or the worst of both worlds, the field hasn’t come to a consensus yet) is using subwords. Subwords are the predominant mode of representing vocabulary units in the language model space today. The GPT Neo-X vocabulary we explored earlier uses subword tokens. [Figure 2-1](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch02.html#subword-tokens) shows the Open AI tokenizer playground that demonstrates how words are split into their constituent subwords.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/figure124.png" alt="Subword Tokens" height="361" width="600"><figcaption></figcaption></figure>

**Figure 2-1. Subword Tokens**

### Tokenizer

A tokenizer has two responsibilities -

1. In the tokenizer pre-training stage, the tokenizer is run over a body of text to generate a vocabulary.
2. While processing input during both training and inference, free-form raw text is run through the tokenizer algorithm to break down the text into tokens. [Figure 2-2](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch02.html#tokenizer-workflow) depicts the roles played by a tokenizer

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/Fig2-2_highres.png" alt="Tokenizer Workflows" height="257" width="600"><figcaption></figcaption></figure>

**Figure 2-2. Tokenizer Workflow**

When we feed raw text to the tokenizer, it breaks down the text into tokens that are part of the vocabulary, and maps the tokens to their token indices. The sequence of token indices (input ids) are then fed to the language model where they are mapped to their corresponding embeddings. Let us explore this process in detail.

This time, let’s experiment with the FlanT5 model. You need a Google Colab Pro or equivalent system to be able to run it.

```
!pip install transformers accelerate sentencepiece
from transformers import T5Tokenizer, T5ForConditionalGeneration


tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto")


input_text = "what is 937 + 934?"
encoded_text = tokenizer.encode(input_text)
tokens = tokenizer.convert_ids_to_tokens(encoded_text)
print(tokens)
```

The output is

```
['▁what', '▁is', '▁9', '37', '▁+', '▁9', '34', '?', '</s>']
```

The encoder() function tokenizes the input text and returns the corresponding token indices. The token indices are mapped to the tokens they represent using the convert\_ids\_to\_tokens() function.

As you can see, the Flan-T5 tokenizer doesn’t have dedicated tokens for the numbers 937 or 934. Therefore, it splits the numbers into ‘9’ and ‘37’. The \</s> token is a special token indicating the end of the string. The ‘\_’ means that the token is preceded by a space.

Let’s try another example.

```
input_text = "Insuffienct adoption of corduroy pants is the reason this

economy is in the dumps!!!"
encoded_text = tokenizer.encode(input_text)
tokens = tokenizer.convert_ids_to_tokens(encoded_text)
print(tokens)
```

The output is

```
['▁In', 's', 'uff', 'i', 'en', 'c', 't', '▁adoption', '▁of', '▁cord', 'u',

'roy', '▁pants', '▁is', '▁the', '▁reason', '▁this', '▁economy', '▁is', '▁in',

'▁the', '▁dump', 's', '!!!', '</s>']
```

I had made a deliberate typo with the word ‘Insufficient’. Note that subword tokenization is rather brittle with typos. But at least the OOV problem has been dealt with by breaking down the words into subwords. The vocabulary also doesn’t seem to have an entry for the word ‘corduroy’, thus confirming its poor sense of fashion. Meanwhile, there is a separate token for three contiguous exclamation points, which is different from the token that represents a single exclamation point. Semantically, they do convey slightly different meanings.

**NOTE**

Very large models trained on a massive body of text are more robust to misspellings. A lot of misspellings already occur in the training set. For example, even the rare misspelling ‘_Insuffienct_’ occurs 14 times in the C4 pre-training dataset. The more common misspelling ‘_insufficent_’ occurs over 1100 times. Larger models can also infer the misspelled word from its context. Smaller models like BERT are quite sensitive to misspellings.

If you are using models from Open AI, you can explore their tokenization scheme using the [tiktoken](https://github.com/openai/tiktoken) library. (no relation to the social media website).

Using tiktoken, let’s see the different vocabularies available in the Open AI ecosystem.

```
!pip install tiktoken

import tiktoken
tiktoken.list_encoding_names()
```

The output is

```
['gpt2', 'r50k_base', 'p50k_base', 'p50k_edit', 'cl100k_base']
```

The numbers like 50k/100k are presumed to be the vocabulary size. Open AI hasn’t revealed much information about these. Their documentation does state that _cl100k\_base_ is used by GPT-4 and GPT 3.5 (chatGPT), while _p50k\_base_ is used by the Codex models, and the Instruct versions of GPT-3.

```
encoding = tiktoken.encoding_for_model("gpt-4")
input_ids = encoding.encode("Insuffienct adoption of corduroy pants is the

reason this economy is in the dumps!!!")
tokens = [encoding.decode_single_token_bytes(token) for token in input_ids]
```

The output is

```
[b'Ins', b'uff', b'ien', b'ct', b' adoption', b' of', b' cord', b'uro', b'y',

b' pants', b' is', b' the', b' reason', b' this', b' economy', b' is', b' in',

b' the', b' dumps', b'!!!']
```

As you can see there is not much of a difference between the tokenization used by GPT-4 and GPT Neo-X.

## EXERCISE

Using tiktoken, find the difference between _p50k\_base_, the encoding used for GPT 3.5 (chatGPT), and _cl100k\_base_, the encoding used for GPT-4. What are the 50,000 extra tokens in the GPT-4 vocabulary representing?

**TIP**

While adapting LLM’s to your use case, If you see strange behavior from the model on a subset of your inputs, it is worthwhile to check how they have been tokenized. While you cannot definitively diagnose your problem just by analyzing the tokenization, it is often helpful in analysis. In my experience, a non-negligible amount of LLM failures can be attributed to the way the text was tokenized. This is especially true if your target domain is different from the pre-training domain.

## TOKENIZATION-FREE MODELS

As discussed in Chapter 1, the _consolidation effect_ has resulted in end-to-end architectures. However, one last hold-out is the tokenization step. You have seen in the code earlier that the tokenization is used as a pre-processing step to prepare the input to be fed into the model. The input to the model is the sequence of token indices and not raw text. But what if we make the model truly end-to-end by removing the tokenization step? Is it possible to directly feed raw text to the model and have it output results?

There have been forays into the world of tokenization-free language modeling, with models like CANINE, ByT5, and CharFormer.

* [CANINE](https://arxiv.org/abs/2103.06874) accepts Unicode codepoints as input. But there are 1,114,112 possible code points, rendering the vocabulary and resulting embedding layer size infeasible. To resolve this, CANINE uses hashed embeddings so that the effective vocabulary space is much smaller.
* [ByT5](https://arxiv.org/abs/2105.13626) accepts input in terms of bytes, so there are only 259 embeddings in the embedding matrix (including a few special tokens), thus reducing the embedding layer size drastically.
* [CharFormer](https://arxiv.org/abs/2106.12672) also accepts input in terms of bytes, and passes it to a gradient-based subword tokenizer module, that constructs latent subwords.

### Tokenization Pipeline

[Figure 2-3](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch02.html#huggingface-tokenizers-pipeline) depics the sequence of steps performed by a tokenizer.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/Fig2-3_highres.png" alt="HuggingFace Tokenizers pipeline" height="60" width="600"><figcaption></figcaption></figure>

**Figure 2-3. HuggingFace Tokenizers Pipeline**

If you are using the tokenizers library from HuggingFace, your input text is run through a [multi-stage tokenization pipeline](https://huggingface.co/docs/tokenizers/pipeline). This pipeline is composed of four components -

* Normalization
* Pre-tokenization
* Tokenization
* Post-processing

Note that different models will have different steps executed within these 4 components.

#### Normalization

Different types of normalization applied include

* Converting text to lowercase (if you are using an uncased model)
* Stripping off accents from characters, like from the word Peña
* Unicode normalization

Let’s see what kind of normalization is applied on the uncased version of BERT:

```
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(tokenizer.backend_tokenizer.normalizer.normalize_str('Pédrò pôntificated at üs:-)')
```

The output is

```
pedro pontificated at us:-)
```

As we see, the accents have been removed and the text has been converted to lowercase.

There isn’t much normalization done in tokenizers for more recent models.

#### Pre-tokenization

Before we run the tokenizer on the text, we can optionally perform a pre-tokenization step. As mentioned earlier, most tokenizers today employ subword tokenization. A common step is to first perform word tokenization and then feed the output of it to the subword tokenization algorithm. This step is called pre-tokenization.

Pre-tokenization is a relatively easy step in English compared to many other languages, since you can start off with a very strong baseline by just splitting text on whitespace. There are outlier decisions to be made - how to deal with punctuation, multiple spaces, numbers etc. In HuggingFace the regular expression

```
\w+|[^\w\s]+
```

is used to split on whitespace.

Let’s run the pre-tokenization step of the T5 tokenizer.

```
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("I'm starting to

suspect - I am 55 years old!   Time to vist New York?")
```

The output is

```
[("▁I'm", (0, 3)),
 ('▁starting', (3, 12)),
 ('▁to', (12, 15)),
 ('▁suspect', (15, 23)),
 ('▁-', (23, 25)),
 ('▁I', (25, 27)),
 ('▁am', (27, 30)),
 ('▁55', (30, 33)),
 ('▁years', (33, 39)),
 ('▁old!', (39, 44)),
 ('▁', (44, 45)),
 ('▁', (45, 46)),
 ('▁Time', (46, 51)),
 ('▁to', (51, 54)),
 ('▁vist', (54, 59)),
 ('▁New', (59, 63)),
 ('▁York?', (63, 69))]
```

Along with the pre-tokens (or word tokens), the character offsets are returned.

The T5 pre-tokenizer splits only on whitespace, doesn’t collapse multiple spaces into one, does’t split on punctuation or numbers. The behavior can be vastly different for other tokenizers.

#### Tokenization

After the optional pre-tokenization step, the actual tokenization step is performed. Some of the important algorithms in this space are BPE (Byte Pair Encoding), Byte BPE, WordPiece, and Unigram LM. The tokenizer comprises a set of rules that is learned during a pre-training phase over a pre-training dataset. Now let’s go through these algorithms in detail.

#### BPE (Byte Pair Encoding)

This algorithm is the simplest and most widely used tokenization algorithm.

_Training stage_

We take a training dataset, run it through the normalization and pre-tokenization steps discussed earlier, and record the unique tokens in the resulting output and their frequencies. We then construct an initial vocabulary consisting of the unique characters that make up these tokens. Starting from this initial vocabulary, we continue adding new tokens using _merge_ rules. The merge rule is simple - we merge the most frequent consecutive pairs of tokens. The merges continue until we reach the desired vocabulary size.

Let’s explore this with an example. Imagine our training dataset is composed of six words, each appearing just once.

```
‘bat’, ‘cat’, ‘cap’, ‘sap’, ‘map’, ‘fan’
```

The initial vocabulary is then made up of

```
‘b’, ‘a’, ‘t’, ‘c’, ‘p’, ‘s’, ‘m’, ‘f’, ‘n’
```

The frequencies of contiguous token pairs are

```
‘ba’ - 1, ‘at’ - 2, ‘ca’ - 2, ‘ap’ - 3, ‘sa’ - 1, ‘ma’ - 1, ‘fa’ - 1, ‘an’ - 1
```

The most frequent pair is ‘ap’, so the first merge rule is to merge ‘a’ and ‘p’. The vocabulary now is

```
‘b’, ‘a’, ‘t’, ‘c’, ‘p’, ‘s’, ‘m’, ‘f’, ‘n’, ‘ap’
```

The new frequencies are -

```
‘ba’ - 1, ‘at’ - 2, ‘cap’ - 1, ‘sap’ - 1, ‘map’ - 1, ‘fa’ - 1, ‘an’ - 1
```

Now,the most frequent pair is ‘at’, so the next merge rule is to merge ‘a’ and ‘t’.This process continues until we reach the vocabulary size.

_Inference stage_

After the tokenizer has been trained, it can be used to divide the text into appropriate subword tokens and feed the text into the model. This happens in a similar fashion as the training step. After normalization and pre-tokenization of the input text, the resulting tokens are broken into individual characters and all the merge rules are applied in order. The tokens remaining after all merge rules have been applied are the final tokens which are then fed to the model.

You can open the [vocabulary file](https://huggingface.co/EleutherAI/gpt-neox-20b/resolve/main/tokenizer.json) for GPT Neo-X again and ctrl+f ‘merges’ to see the merge rules. As expected, the initial merge rules join single characters with each other. At the end of the merge list, you can see larger subwords like ‘out’ and ‘comes’ being merged into a single token.

## EXERCISE

Implement the BPE algorithm by yourself, using a domain dataset of your choice. What tokens do you end up with and how does it differ from the vocabulary of the popular language models? This also gives you a clue on how effective general-purpose LM’s will be for your use case.

**NOTE**

Since all unique individual characters in the tokenizer training set will get their own token, it is guaranteed that there will be no OOV tokens as long as all tokens seen during inference in future are made up of characters that were present in the training set. But Unicode consists of over a million code points and around 150,000 valid characters, which would not fit in a vocabulary of size 30000. This means that if your input text contained a character that wasn’t in the training set, that character would be assigned an \<UNK> token. To resolve this, a variant of BPE called Byte-level BPE is used. Byte-level BPE starts with 256 tokens, representing all the characters that can be represented by a byte. This ensures that every Unicode character can be encoded just by the concatenation of the constituent byte tokens. Hence it also ensures that we will never encounter an \<UNK> token. GPT-n models use this tokenizer.

#### WordPiece

WordPiece is similar to BPE, so we will highlight only the differences.

Instead of the frequency approach used by BPE, WordPiece uses the maximum likelihood approach. The frequency of the token pairs in the dataset is normalized by the product of the frequency of the individual tokens. The pairs with the resulting highest score are then merged.

```
score = freq(a,b)/(freq(a) * freq(b))
```

This means that lower frequency terms are joined first.

In WordPiece, merge rules are not used. Instead, for each pre-tokenized token in the input text, the tokenizer finds the longest subword from the vocabulary in the token and splits on it. For example, if the token is ‘understanding’ and the longest subword in the dictionary within this token is ‘understand’, then it will be split into ‘understand’ and ‘ing’.

#### Postprocessing

The final stage of the tokenizer pipeline is the postprocessing stage. This is where model specific special tokens are added. Common tokens include \[CLS] or the classification token used in many language models, and \[SEP], a separator token used to separate parts of the input.

## THE CURIOUS CASE OF SOLIDMAGIGOLDKARP.

There are weird tokens that end up being part of a language model’s vocabulary, due to the way the tokenization algorithms work. One such token is ‘SolidMagiGoldkarp’, representing a now-deleted Reddit user who was one of the site’s most active posters because of his quest to count to infinity. This was a token in the GPT-2 tokenizer. The same tokenizer was used in GPT-3 models but the pre-training dataset of the model had changed, so now a token existed for SolidMagiGoldkarp but there was no signal in the pre-training dataset to learn from. This leads to some anomalous and hilarious behavior in GPT-N models.

## EXERCISE

Token archaeology is a new hobby for many LLM enthusiasts. This involves finding rare tokens in the vocabulary of language models, and unearthing its origin. This is not just fun and games though, as knowing the origin of rare tokens can give you an insight into the characteristics of the pre-training dataset. Using tiktoken, find some rare vocabulary terms in GPT-3.5 or GPT-4’s vocabulary. Can you figure out their origins?

### Special Tokens

Depending on the model, there are a few special tokens that are added to the vocabulary to facilitate processing. These tokens include

* \<PAD> - to indicate padding, in case the size of the input is lesser than the maximum sequence length.
* \<EOS> - to indicate the end of the sequence. Generative models stop generating after outputting this token.
* \<UNK> - to indicate an OOV term

As we have seen, if our data is domain-specific like healthcare, scientific literature etc, tokenization from a general-purpose tokenizer will be unsatisfactory. GALACTICA by Meta introduced several domain specific tokens in their model and special tokenization rules

* \[START\_REF] and \[END\_REF] for wrapping citations.
* \<WORK> token to wrap tokens that make up an internal working memory, used for reasoning and code generation
* Numbers are handled by assigning each digit in the number its own token
* \[START\_SMILES], \[START\_DNA], \[START\_AMINO], \[END\_SMILES], \[END\_DNA], \[END\_AMINO] for protein sequences, DNA sequences, and amino acid sequences respectively.

**NOTE**

Why is the vocabulary size so large? Surely, having a smaller vocabulary size would be more convenient as the size of the embedding matrix would be smaller. However, the smaller the vocabulary, the more number of tokens needed to represent a sequence, which would make the model slower in both training and inference.

## Learning Objectives

Now that we have discussed the pre-training dataset and vocabulary, let us move on to the next ingredient of the language model - the Learning Objective. Language models are pre-trained in a self-supervised manner. The scale of data we need to train them makes it prohibitively expensive to perform supervised learning, where (input, output) examples need to come from humans. Instead, we use a form of training called self-supervision, where the data itself contains the target labels. The goal of self-supervised learning is to learn a task which acts as a proxy for learning the syntax and semantics of a language, as well as skills like reasoning, arithmetic and logical manipulation, and other cognitive tasks, and (hopefully) eventually leading up to general human intelligence. How does this work?

For example, let’s take the canonical language modeling task - predicting the next word that comes in a sequence. Consider the sequence

```
'Tammy jumped over the'
```

and the language model is asked to predict the next token. The total number of possible answers is the size of the vocabulary. There are a lot of valid continuations to this sequence - like (hedge, fence, barbecue, sandcastle etc), but there are many continuations to this sequence that would violate English grammar rules like (is, of, the). During the training process, after seeing billions of sequences, the model will know that it is highly improbable for the word _the_ to be followed by the word _is_ or _of_, regardless of the surrounding context. Thus, you can see how just predicting the next token is such a powerful tool - in order to correctly predict the next token you can eventually learn more and more complex functions that you can encode in your model connections. However, whether this paradigm is all we need to develop general intelligence is an open question.

Self-supervised learning objectives used for pre-training LLMs can be broadly classified (non-exhaustively) into three types:

* FLM (Full Language Modeling)
* MLM (Masked Language Modeling)
* PrefixLM (Prefix Language Modeling)

Let’s explore these in detail.

### Full Language Modeling

[Figure 2-4](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch02.html#full-language-modeling) shows the canonical FLM objective at work

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/figure121.png" alt="Full Language Modeling" height="127" width="600"><figcaption></figcaption></figure>

**Figure 2-4. Full Language Modeling**

This is the canonical language modeling objective of learning to predict the next token in a sequence.This is currently the simplest and most common training objective, used by GPT-4 and a vast number of open-source models. The loss is computed for every token the model sees, i.e every single token in the training set that is being asked to be predicted by the language model provides a learning signal for the model, making it very efficient.

Let us explore an example, using the GPT-Neo model.

Suppose we continue pre-training the GPT-Neo model from its publicly available checkpoint, using the full language modeling objective. Let’s say the current training sequence is

```
'Language models are ubiquitous'
```

You can run this code

```
import torch
from transformers import AutoTokenizer, GPTNeoForCausalLM


tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")


input_ids = tokenizer("Language models are", return_tensors="pt")
gen_tokens = model.generate(**input_ids, max_new_tokens =1,

output_scores=True, return_dict_in_generate=True)
output_scores = gen_tokens["scores"]
scores_tensor = output_scores[0]
sorted_indices = torch.argsort(scores_tensor[0], descending=True)[:20]


for index in sorted_indices:
    token_id = index
    token_name = tokenizer.decode([token_id.item()])
    token_score = scores_tensor[0][index].item()
    print(f"Token: {token_name}, Score: {token_score}")
```

This code tokenizes the input text _Language models are_ and feeds it to the model by invoking the generate() function. The function predicts the continuation, given the sequence ‘Language models are’. It outputs only one token and stops generating because max\_new\_tokens is set to 1. The rest of the code enables it to output the top 20 list of tokens with the highest score, prior to applying the softmax at the last layer.

The top 20 tokens with the highest prediction score are

```
Output: Token:  a, Score: -1.102203369140625
Token:  used, Score: -1.4315788745880127
Token:  the, Score: -1.7675716876983643
Token:  often, Score: -1.8415470123291016
Token:  an, Score: -2.4652323722839355
Token:  widely, Score: -2.657834053039551
Token:  not, Score: -2.6726579666137695
Token:  increasingly, Score: -2.7568516731262207
Token:  ubiquitous, Score: -2.8688106536865234
Token:  important, Score: -2.902832508087158
Token:  one, Score: -2.9083480834960938
Token:  defined, Score: -3.0815649032592773
Token:  being, Score: -3.2117576599121094
Token:  commonly, Score: -3.3110013008117676
Token:  very, Score: -3.317342758178711
Token:  typically, Score: -3.4478530883789062
Token:  complex, Score: -3.521362781524658
Token:  powerful, Score: -3.5338563919067383
Token:  language, Score: -3.550961971282959
Token:  pervasive, Score: -3.563507080078125
```

Every word in the top 20 seems to be a valid continuation of the sequence. The ground truth is the token ‘ubiquitous’, which we can use to calculate the loss and initiate the backpropagation process for learning.

As an another example, consider the text sequence

```
'I had 25 eggs. I gave away 12. I now have 13'
```

Run the same code as previously, except for this change.

```
input_ids = tokenizer("'I had 25 eggs. I gave away 12. I now have", return_tensors="pt")
```

The top 20 output tokens are:

```
Token:  12, Score: -2.3242850303649902
Token:  25, Score: -2.5023117065429688
Token:  only, Score: -2.5456185340881348
Token:  a, Score: -2.5726099014282227
Token:  2, Score: -2.6731367111206055
Token:  15, Score: -2.6967623233795166
Token:  4, Score: -2.8040688037872314
Token:  3, Score: -2.839219570159912
Token:  14, Score: -2.847306728363037
Token:  11, Score: -2.8585362434387207
Token:  1, Score: -2.877161979675293
Token:  10, Score: -2.9321107864379883
Token:  6, Score: -2.982785224914551
Token:  18, Score: -3.0570476055145264
Token:  20, Score: -3.079172134399414
Token:  5, Score: -3.111320972442627
Token:  13, Score: -3.117424726486206
Token:  9, Score: -3.125835657119751
Token:  16, Score: -3.1476120948791504
Token:  7, Score: -3.1622045040130615
```

The correct answer has the 17th highest score. A lot of numbers appear in the top 10, showing that the model is more or less random guessing the answer, which is not surprising for a smaller model like GPT-Neo

The Open AI API provides the ‘logprobs’ parameter that allows you to specify the number of tokens along with their log probabilities that need to be returned. This is available for GPT-3, but not yet for ChatGPT. The tokens returned are in order of their log probabilities.

```
import openai
openai.api_key = <Insert your Open AI key>


openai.Completion.create(
  model="text-davinci-003",
  prompt="I had 25 eggs. I gave away 12. I now have ",
  max_tokens=1,
  temperature=0,
  logprobs = 10
)
```

This code calls the older ‘text-davinci-003’ (GPT-3) model, asking it to generate a maximum of one token.The output is

```
"top_logprobs": [
          {
            "\n": -0.08367541,
            " 13": -2.8566456,
            "____": -4.579212,
            "_____": -4.978668,
            "________": -6.220278
          }
```

GPT-4 is pretty confident that the answer is 13, and rightfully so. The rest of the top probability tokens are all related to output formatting.

**TIP**

During inference, we don’t necessarily need to generate the token with the highest score. There are several _decoding strategies_ that allow you to generate more diverse text. We will discuss these strategies in Chapter 4.

## EXERCISE

Ask the text-davinci-003 model to solve individual crossword clues in the [Washington Post Daily Crossword](https://www.washingtonpost.com/crossword-puzzles/daily/). You may have to iterate with the prompt. A good start would be ‘Solve this crossword and answer in one word. The clue is \<X> and it is a \<Y> letter word. The answer is ‘. Set max\_tokens = 3 to account for formatting tokens. Analyze the logprobs output. Is it dangerously close to getting it right/wrong? How many clues does it answer correctly?

### Prefix Language Modeling

Prefix LM is similar to the FLM setting. The difference is that FLM is fully causal, i.e in a left-to-right writing system like English, tokens do not attend to tokens to the right (future). In the prefix LM setting, a part of the text sequence, called the prefix, is allowed to attend to future tokens in the prefix. The prefix part is thus non-causal. For training prefix LMs, a random prefix length is sampled, and the loss is calculated over only the tokens in the suffix.

### Masked Language Modeling

[Figure 2-5](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch02.html#masked-language-modeling-bert) shows the canonical MLM objective at work

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/figure122.png" alt="Masked Language Modeling in BERT" height="101" width="600"><figcaption></figcaption></figure>

**Figure 2-5. Masked Language Modeling in BERT**

In the MLM setting, rather than predict the next token in a sequence, we ask the model to predict masked tokens within the sequence. In the most basic form of MLM implemented in the BERT model, 15% of tokens are randomly chosen to be masked and are replaced with a special mask token, and the language model is asked to predict the original tokens.

#### T5

The T5 model creators used a modification of the original MLM objective. In this variant, 15% of tokens are randomly chosen to be removed from a sequence. Consecutive dropped-out tokens are replaced by a single unique special token called the _sentinel token_. The model is then asked to predict and generate the dropped tokens, delineated by the sentinel tokens.

As an example, consider this sequence

```
'Tempura has always been a source of conflict in the family due to unexplained reasons'
```

Let’s say we drop the tokens ‘has’, ‘always’, ‘of’, ‘conflict’. The sequence is now

```
'Tempura <S1> been a source <S2> in the family due to unexplained reasons'
```

with S1, S2 being the sentinel tokens. The model is expected to output

```
‘<S1> has always <S2> of conflict <E>’
```

The output sequence is terminated by another sentinel token indicating the end of the sequence.

Generating only the dropped tokens and not the entire sequence is computationally more efficient and saves training time. Note that unlike in Full Language Modeling, the loss is calculated over only a small proportion of tokens (the masked tokens) in the input sequence.

Let’s explore this on HuggingFace

```
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-3b")
model = T5ForConditionalGeneration.from_pretrained("t5-3b")

input_ids = tokenizer("Tempura <extra_id_0>  been a source <extra_id_1> in the
family due to unexplained reasons", return_tensors="pt").input_ids
targets = tokenizer("<extra_id_0> has always <extra_id_1> of conflict

<extra_id_2>", return_tensors="pt").input_ids
loss = model(input_ids=input_ids, labels=labels).loss
```

The targets can be prepared using a simple templating function.

## EXERCISE

Play around with different masking strategies. Specifically,

* Change the masking rate. What happens if you mask 30% or 50% of tokens?
* Change the masking strategy. Can you do better than random masking? What heuristics would allow you to mask tokens that would contribute more towards learning?

More generally, masked language modeling can be interpreted as a _denoising autoencoder_. You corrupt your input by adding noise(masking, dropping tokens), and then you train a model to regenerate the original input. BART takes this to the next level by using 5 different types of span corruptions:

* Random token masking ala BERT. [Figure 2-6](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch02.html#bart-enoiser-objectives1) depicts the corruption and denoising steps.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/figure122.png" alt="BART Denoiser Objectives1" height="101" width="600"><figcaption></figcaption></figure>

**Figure 2-6. Random token masking in BART**

* Random token deletion. The model needs to predict the positions in the text where tokens have been deleted. [Figure 2-7](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch02.html#bart-enoiser-objectives2) depicts the corruption and denoising steps.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/figure130.png" alt="BART Denoiser Objectives2" height="128" width="600"><figcaption></figcaption></figure>

**Figure 2-7. Random token deletion in BART**

* Text spans are sampled from text, with span lengths coming from a Poisson distribution. This means 0 length spans are possible. The spans are deleted from the text and replaced with a single mask token. Therefore the model now has to also predict the number of tokens deleted. [Figure 2-8](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch02.html#bart-enoiser-objectives3) depicts the corruption and denoising steps.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/figure128.png" alt="BART Denoiser Objectives3" height="143" width="600"><figcaption></figcaption></figure>

**Figure 2-8. Span masking in BART**

* Sentences in the input document are shuffled.The model is taught to arrange them in the right order. [Figure 2-9](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch02.html#bart-enoiser-objectives4) depicts the corruption and denoising steps.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/figure132.png" alt="BART Denoiser Objectives4" height="53" width="600"><figcaption></figcaption></figure>

**Figure 2-9. Document shuffling objective in BART**

* The document is rotated so that it starts from an arbitrary token. The model is trained to detect the correct start of the document. [Figure 2-10](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch02.html#bart-enoiser-objectives5) depicts the corruption and denoising steps.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/figure131.png" alt="BART Denoiser Objectives5" height="54" width="600"><figcaption></figcaption></figure>

**Figure 2-10. Document rotation objective in BART**

### Which learning objectives are better?

It has been shown that models trained with FLM are better at generation, and models trained with MLM are better at classification tasks. However, it is inefficient to use different language models for different use cases. The consolidation effect continues to take hold, with the introduction of [UL2](https://arxiv.org/pdf/2205.05131.pdf), a new paradigm that combines the best of different learning objective types in a single model.

#### UL2

UL2 mimics the effect of PLMs, MLMs, and PrefixLMs in a single paradigm called _Mixture of Denoisers_.

The denoisers used are -

* R-Denoiser - This is similar to the T5 span corruption task. Spans between length 2-5 tokens are replaced by a single mask token. [Figure 2-11](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch02.html#ul2-mixture-denoisers1) depicts the workings of the R-denoiser.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/figure133.png" alt="UL2&#x27;s Mixture of Denoisers1" height="165" width="600"><figcaption></figcaption></figure>

**Figure 2-11. UL2’s R-Denoiser**

* S-Denoiser - Similar to prefix LM, the text is divided into a prefix and a suffix. The suffix is masked, while the prefix has access to bidirectional context. [Figure 2-12](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch02.html#ul2-mixture-denoisers2) depicts the workings of the S-denoiser.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/figure134.png" alt="UL2&#x27;s Mixture of Denoisers2" height="179" width="600"><figcaption></figcaption></figure>

**Figure 2-12. UL2’s S-Denoiser**

* X-Denoiser - This stands for extreme denoising, where a large proportion of text is masked (often over 50%). [Figure 2-13](https://learning.oreilly.com/library/view/designing-large-language/9781098150495/ch02.html#ul2-mixture-denoisers3) depicts the workings of the X-denoiser.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150495/files/assets/figure135.png" alt="UL2&#x27;s Mixture of Denoisers3" height="94" width="600"><figcaption></figcaption></figure>

**Figure 2-13. UL2’s X-Denoiser**

## Architecture

After covering the pre-training dataset, tokenization, and the learning objective, the final piece of the puzzle is the model architecture itself.

As mentioned in Chapter 2, most modern language models are based on the Transformer architecture. Recall that the original Transformer architecture is made up of an encoder and a decoder. In practice, there are three major types of architecture backbones used:

* Encoder-only
* Encoder-Decoder
* Decoder-only

### Encoder-only architectures

Encoder-only architectures were all the rage when Transformer-based language models first burst on the scene. Iconic language models from yesteryears (circa 2018) that use encoder-only architectures include BERT, RoBERTa, etc.

There aren’t really many encoder-only LLM’s being trained these days. Some reasons are:

* It is relatively harder to train them.
* The masked language modeling objective typically used to train them provides a learning signal in only a small percentage of tokens (the masking rate), thus needing a lot more data in order to reach the same level of performance as decoder-only models.
* For every downstream task, you need to train a separate task specific head, making usage inefficient.

The creators of UL2 recommend that encoder-only models should be considered obsolete. While I personally wouldn’t go that far, I generally agree with the arguments made above against using encoder-only LLMs. However, if you already have a satisfactory pipeline for your use case built around encoder-only models, I would say if it ain’t broke, why fix it?

If you still want to explore encoder-only models, here are some rules of thumb you can follow.

* RoBERTa performs better than BERT most of the time, since it is trained a lot longer on more data, and adopts best practices learned after the release of BERT.
* DeBERTa is currently regarded as the most well performing encoder-only model, and also the largest one available (1.5B parameters)
* The distilled versions of encoder-only models like DistillBERT etc, are not too far off from the original models in terms of performance, and should be considered if you are operating under resource constraints.

Several embedding models are built from encoder-only models. For example, perhaps one of the most important libraries in the field of NLP, considered the Swiss Army Knife of NLP tools, _sentence-transformers_, still provides encoder-only model based embedding models that are very widely used. ‘_all-mpnet-base-v2_’, based on an encoder-only model called MPNet, and fine- tuned on several task datasets, is still competitive with much larger embedding models.

### Encoder-Decoder Architectures

This is the original architecture of the Transformer, as it was first proposed. The T5 series of models uses this architectural type.

In encoder-decoder models, the input is text and the output is also text. A standardized interface ensures that the same model and training procedure can be used for multiple tasks. The inputs are handled by an encoder, and the outputs by the decoder.

### Decoder-only Architectures

A majority of LLMs trained today use decoder-only models. Decoder-only models came into fashion starting from the original GPT model from Open AI. Decoder-only models excel at zero shot and few shot learning.

Decoder models can be causal and non causal. Non causal models have bidirectionality over the input sequence, while the output is still autoregressive (you cannot look ahead)

**TIP**

While the field is still evolving, there has been some compelling evidence for the following results:

* Decoder-only models are the best choice for zero-shot and few-shot generationization
* Encoder-decoder models are the best choice for multi-task fine tuning.

The best of both worlds is to combine the two - Start with auto-regressive training, and then in an adaptation step, pre-train further with a non-casual setup using a span corruption objective.

## Putting it all Together

The recipe for training each model is slightly different. As we have seen, at every step of the way there are a multitude of high-impact decisions to be made.

I often get this question from NLP practitioners - ‘Hey, I am tackling this \<insert niche usecase> problem, what language model do you think I should use? There are hundreds of pre-trained models available out there and I have no idea how to choose among them.’ Truth be told, there are dozens of factors that can impact your choice, and sometimes it may not even be the most immediate or right question to ask. In subsequent chapters I will demonstrate how you can navigate tradeoffs and make an informed decision regarding your choice of model, and the various ways you can utilize them in your tasks.

**NOTE**

Depending on your task, the exact choice of pre-trained model used may not be as important as other data-related choices you need to make. Even in the era of GPT-4, your domain expertise and data cleaning skills are crucial for building successful applications. That being said, throughout the book, we will showcase scenarios where the choice of model can play a crucial role.

## Summary

In this chapter, we discussed vocabularies and tokens, and delved into the different tokenization algorithms currently used. We also discussed the tasks that a language model is pre-trained on, and how they are a proxy to learning syntax and semantics. We also discussed the various architectural backbones of the Transformer.

Now that we know the recipe and ingredients behind LLMs, we will next learn how to utilize them to solve our own tasks. We will discuss techniques like fine-tuning, in-context learning, and zero-shot learning. We will also show how to evaluate LLMs for our use cases, and how to select the right model that suits our needs.
