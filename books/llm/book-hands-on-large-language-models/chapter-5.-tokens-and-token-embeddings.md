# Chapter 5. Tokens & Token Embeddings

## Chapter 5. Tokens & Token Embeddings

## A NOTE FOR EARLY RELEASE READERS

With Early Release ebooks, you get books in their earliest form—the author’s raw and unedited content as they write—so you can take advantage of these technologies long before the official release of these titles. In particular, some of the formatting may not match the description in the text: this will be resolved when the book is finalized.

This will be the 8th chapter of the final book. Please note that the GitHub repo will be made active later on.

If you have comments about how we might improve the content and/or examples in this book, or if you notice missing material within this chapter, please reach out to the editor at _mcronin@oreilly.com_.

Embeddings are a central concept to using large language models (LLMs), as you’ve seen over and over in part one of the book. They also are central to understanding how LLMs work, how they’re built, and where they’ll go in the future.

The majority of the embeddings we’ve looked at so far are _text embeddings_, vectors that represent an entire sentence, passage, or document. [Figure 5-1](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch05.html#fig\_1\_\_the\_difference\_between\_text\_embeddings\_one\_vect) shows this distinction.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/tokens_token_embeddings_963889_01.png" alt="The difference between text embeddings  one vector for a sentence or paragraph  and token embeddings  one vector per word or token ." height="151" width="600"><figcaption></figcaption></figure>

**Figure 5-1. The difference between text embeddings (one vector for a sentence or paragraph) and token embeddings (one vector per word or token).**

In this chapter, we begin to discuss token embeddings in more detail. Chapter 2 discussed tasks of token classification like Named Entity Recognition. In this chapter, we look more closely at what tokens are and the tokenization methods used to power LLMs. We will then go beyond the world of text and see how these concepts of token embeddings empower LLMs that can understand images and data modes (other than text, for example video, audio...etc). LLMs that can process modes of data in addition to text are called _multi-modal_ models. We will then delve into the famous word2vec embedding method that preceded modern-day LLMs and see how it’s extending the concept of token embeddings to build commercial recommendation systems that power a lot of the apps you use.

## LLM Tokenization

### How tokenizers prepare the inputs to the language model

Viewed from the outside, generative LLMs take an input prompt and generate a response, as we can see in [Figure 5-2](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch05.html#fig\_2\_\_high\_level\_view\_of\_a\_language\_model\_and\_its\_inpu).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/tokens_token_embeddings_963889_02.png" alt="High level view of a language model and its input prompt." height="246" width="600"><figcaption></figcaption></figure>

**Figure 5-2. High-level view of a language model and its input prompt.**

As we’ve seen in Chapter 5, instruction-tuned LLMs produce better responses to prompts formulated as instructions or questions. At the most basic level of the code, let’s assume we have a generate method that hits a language model and generates text:

```
prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened."
# Placeholder definition. The next code blocks show the actual generation
def generate(prompt, number_of_tokens):
  # TODO: pass prompt to language model, and return the text it generates
  pass
output = generate(prompt, 10)
print(output)
```

Generation:

```
Subject: Apology and Condolences 
Dear Sarah, 
I am deeply sorry for the tragic gardening accident that took place in my backyard yesterday. As you may have heard, ...etc
```

Let us look closer into that generation process to examine more of the steps involved in text generation. Let’s start by loading our model and its tokenizer.

<pre><code><strong>from transformers import AutoModelForCausalLM, AutoTokenizer
</strong># openchat is a 13B LLM
model_name = "openchat/openchat"
# If your environment does not have the required resources to run this model
# then try a smaller model like "gpt2" or "openlm-research/open_llama_3b"
# Load a tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Load a language model
model = AutoModelForCausalLM.from_pretrained(model_name)
</code></pre>

We can then proceed to the actual generation. Notice that the generation code always includes a tokenization step prior to the generation step.

```
prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened."
# Tokenize the input prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
# Generate the text
generation_output = model.generate(
  input_ids=input_ids, 
  max_new_tokens=256
)
# Print the output
print(tokenizer.decode(generation_output[0]))
```

Looking at this code, we can see that the model does not in fact receive the text prompt. Instead, the tokenizers processed the input prompt, and returned the information the model needed in the variable input\_ids, which the model used as its input.

Let’s print input\_ids to see what it holds inside:

```
tensor([[ 1, 14350, 385, 4876, 27746, 5281, 304, 19235, 363, 278, 25305, 293, 16423, 292, 286, 728, 481, 29889, 12027, 7420, 920, 372, 9559, 29889]])
```

This reveals the inputs that LLMs respond to. A series of integers as shown in [Figure 5-3](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch05.html#fig\_3\_\_a\_tokenizer\_processes\_the\_input\_prompt\_and\_prepa). Each one is the unique ID for a specific token (character, word or part of word). These IDs reference a table inside the tokenizer containing all the tokens it knows.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/tokens_token_embeddings_963889_03.png" alt="A tokenizer processes the input prompt and prepares the actual input into the language model  a list of token ids." height="334" width="600"><figcaption></figcaption></figure>

**Figure 5-3. A tokenizer processes the input prompt and prepares the actual input into the language model: a list of token ids.**

If we want to inspect those IDs, we can use the tokenizer’s decode method to translate the IDs back into text that we can read:

```
for id in input_ids[0]:
   print(tokenizer.decode(id))
```

Which prints:

```
<s> 
Write 
an 
email 
apolog 
 izing 
to 
Sarah 
for 
the 
trag 
 ic 
garden 
 ing 
m 
 ish 
 ap 
. 
Exp 
 lain 
how 
it 
happened 
.
```

This is how the tokenizer broke down our input prompt. Notice the following:

* The first token is the token with ID #1, which is \<s>, a special token indicating the beginning of the text
* Some tokens are complete words (e.g., _Write_, _an_, _email_)
* Some tokens are parts of words (e.g., _apolog_, _izing_, _trag_, _ic_)
* Punctuation characters are their own token
* Notice how the space character does not have its own token. Instead, partial tokens (like ‘izing’ and ‘ic') have a special hidden character at their beginning that indicate that they’re connected with the token that precedes them in the text.

There are three major factors that dictate how a tokenizer breaks down an input prompt. First, at model design time, the creator of the model chooses a tokenization method. Popular methods include Byte-Pair Encoding (BPE for short, widely used by GPT models), WordPiece (used by BERT), and SentencePiece (used by LLAMA). These methods are similar in that they aim to optimize an efficient set of tokens to represent a text dataset, but they arrive at it in different ways.

Second, after choosing the method, we need to make a number of tokenizer design choices like vocabulary size, and what special tokens to use. More on this in the “Comparing Trained LLM Tokenizers” section.

Thirdly, the tokenizer needs to be trained on a specific dataset to establish the best vocabulary it can use to represent that dataset. Even if we set the same methods and parameters, a tokenizer trained on an English text dataset will be different from another trained on a code dataset or a multilingual text dataset.

In addition to being used to process the input text into a language model, tokenizers are used on the output of the language model to turn the resulting token ID into the output word or token associated with it as [Figure 5-4](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch05.html#fig\_4\_\_tokenizers\_are\_also\_used\_to\_process\_the\_output\_o) shows.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/tokens_token_embeddings_963889_04.png" alt="Tokenizers are also used to process the output of the model by converting the output token ID into the word or token associated with that ID." height="478" width="600"><figcaption></figcaption></figure>

**Figure 5-4. Tokenizers are also used to process the output of the model by converting the output token ID into the word or token associated with that ID.**

### Word vs. Subword vs. Character vs. Byte Tokens

The tokenization scheme we’ve seen above is called subword tokenization. It’s the most commonly used tokenization scheme but not the only one. The four notable ways to tokenize are shown in [Figure 5-5](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch05.html#fig\_5\_\_there\_are\_multiple\_methods\_of\_tokenization\_that). Let’s go over them:

Word tokens

This approach was common with earlier methods like Word2Vec but is being used less and less in NLP. Its usefulness, however, led it to be used outside of NLP for use cases such as recommendation systems, as we’ll see later in the chapter.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/tokens_token_embeddings_963889_05.png" alt="There are multiple methods of tokenization that break down the text to different sizes of components  words  subwords  characters  and bytes ." height="372" width="600"><figcaption></figcaption></figure>

**Figure 5-5. There are multiple methods of tokenization that break down the text to different sizes of components (words, subwords, characters, and bytes).**

One challenge with word tokenization is that the tokenizer becomes unable to deal with new words that enter the dataset after the tokenizer was trained. It also results in a vocabulary that has a lot of tokens with minimal differences between them (e.g., apology, apologize, apologetic, apologist). This latter challenge is resolved by subword tokenization as we’ve seen as it has a token for '_apolog',_ and then suffix tokens (e.g., _'-y_', '_-ize_', '_-etic_', '-_ist_') that are common with many other tokens, resulting in a more expressive vocabulary.

Subword Tokens

This method contains full and partial words. In addition to the vocabulary expressivity mentioned earlier, another benefit of the approach is its ability to represent new words by breaking the new token down into smaller characters, which tend to be a part of the vocabulary.

When compared to character tokens, this method benefits from the ability to fit more text within the limited context length of a Transformer model. So with a model with a context length of 1024, you may be able to fit three times as much text using subword tokenization than using character tokens (sub word tokens often average three characters per token).

Character Tokens

This is another method that is able to deal successfully with new words because it has the raw letters to fall-back on. While that makes the representation easier to tokenize, it makes the modeling more difficult. Where a model with subword tokenization can represent “play” as one token, a model using character-level tokens needs to model the information to spell out “p-l-a-y” in addition to modeling the rest of the sequence.

Byte Tokens

One additional tokenization method breaks down tokens into the individual bytes that are used to represent unicode characters. Papers like [CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation](https://arxiv.org/abs/2103.06874) outline methods like this which are also called “tokenization free encoding”. Other works like [ByT5: Towards a token-free future with pre-trained byte-to-byte models](https://arxiv.org/abs/2105.13626) show that this can be a competitive method.

One distinction to highlight here: some subword tokenizers also include bytes as tokens in their vocabulary to be the final building block to fall back to when they encounter characters they can’t otherwise represent. The GPT2 and RoBERTa tokenizers do this, for example. This doesn’t make them tokenization-free byte-level tokenizers, because they don’t use these bytes to represent everything, only a subset as we’ll see in the next section.

Tokenizers are discussed in more detail in \[Suhas’ book]

### Comparing Trained LLM Tokenizers

We’ve pointed out earlier three major factors that dictate the tokens that appear within a tokenizer: the tokenization method, the parameters and special tokens we use to initialize the tokenizer, and the dataset the tokenizer is trained on. Let’s compare and contrast a number of actual, trained tokenizers to see how these choices change their behavior.

We’ll use a number of tokenizers to encode the following text:

```
text = """
English and CAPITALIZATION
ߎ堩蟠
show_tokens False None elif == >= else: two tabs:"    " Three tabs: "       "
12.0*50=600
"""
```

This will allow us to see how each tokenizer deals with a number of different kinds of tokens:

* Capitalization
* Languages other than English
* Emojis
* Programming code with its keywords and whitespaces often used for indentation (in languages like python for example)
* Numbers and digits

Let’s go from older to newer tokenizers and see how they tokenize this text and what that might say about the language model. We’ll tokenize the text, and then print each token with a gray background color.

#### bert-base-uncased

Tokenization method: WordPiece, introduced in [Japanese and Korean voice search](https://static.googleusercontent.com/media/research.google.com/ja/pubs/archive/37842.pdf)

Vocabulary size: 30522

Special tokens: ‘unk\_token’: '\[UNK]'

’sep\_token’: '\[SEP]'

‘pad\_token’: '\[PAD]'

‘cls\_token’: '\[CLS]'

‘mask\_token’: '\[MASK]'

Tokenized text:

```
[CLS] english and capital ##ization [UNK] [UNK] show _ token ##s false none eli ##f = = > = else : four spaces : " " two tab ##s : " " 12 . 0 * 50 = 600 [SEP]
```

With the uncased (and more popular) version of the BERT tokenizer, we notice the following:

* The newline breaks are gone, which makes the model blind to information encoded in newlines (e.g., a chat log when each turn is in a new line)
* All the text is in lower case
* The word “capitalization” is encoded as two subtokens capital ##ization . The ## characters are used to indicate this token is a partial token connected to the token the precedes it. This is also a method to indicate where the spaces are, it is assumed tokens without ## before them have a space before them.
* The emoji and Chinese characters are gone and replaced with the \[UNK] special token indicating an “unknown token”.

#### bert-base-cased

Tokenization method: WordPiece

Vocabulary size: 28,996

Special tokens: Same as the uncased version

Tokenized text:

```
[CLS] English and CA ##PI ##TA ##L ##I ##Z ##AT ##ION [UNK] [UNK] show _ token ##s F ##als ##e None el ##if = = > = else : Four spaces : " " Two ta ##bs : " " 12 . 0 * 50 = 600 [SEP]
```

The cased version of the BERT tokenizer differs mainly in including upper-case tokens.

* Notice how “CAPITALIZATION” is now represented as eight tokens: CA ##PI ##TA ##L ##I ##Z ##AT ##ION
* Both BERT tokenizers wrap the input within a starting \[CLS] token and a closing \[SEP] token. \[CLS] and \[SEP] are utility tokens used to wrap the input text and they serve their own purposes. \[CLS] stands for Classification as it’s a token used at times for sentence classification. \[SEP] stands for Separator, as it’s used to separate sentences in some applications that require passing two sentences to a model (For example, in the rerankers in chapter 3, we would use a \[SEP] token to separate the text of the query and a candidate result).

#### gpt2

Tokenization method: BPE, introduced in [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)

Vocabulary size: 50,257

Special tokens: <|endoftext|>

Tokenized text:

English and CAP ITAL IZ ATION

� � � � � �

show \_ t ok ens False None el if == >= else :

Four spaces : " " Two tabs : " "

12 . 0 \* 50 = 600

With the GPT-2 tokenizer, we notice the following:

The newline breaks are represented in the tokenizer

Capitalization is preserved, and the word “CAPITALIZATION” is represented in four tokens

The 🎵 蟠characters are now represented into multiple tokens each. While we see these tokens printed as the � character, they actually stand for different tokens. For example, the 🎵 emoji is broken down into the tokens with token ids: 8582, 236, and 113. The tokenizer is successful in reconstructing the original character from these tokens. We can see that by printing tokenizer.decode(\[8582, 236, 113]), which prints out 🎵

The two tabs are represented as two tokens (token number 197 in that vocabulary) and the four spaces are represented as three tokens (number 220) with the final space being a part of the token for the closing quote character.

**NOTE**

What is the significance of white space characters? These are important for models that understand or generate code. A model that uses a single token to represent four consecutive white space characters can be said to be more tuned to a python code dataset. While a model can live with representing it as four different tokens, it does make the modeling more difficult as the model needs to keep track of the indentation level. This is an example of where tokenization choices can help the model improve on a certain task.

#### google/flan-t5-xxl

Tokenization method: SentencePiece, introduced in [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/pdf/1808.06226.pdf)

Vocabulary size: 32,100

Special tokens:

\- ‘unk\_token’: '\<unk>'

\- ‘pad\_token’: '\<pad>'

Tokenized text:

English and CA PI TAL IZ ATION \<unk> \<unk> show \_ to ken s Fal s e None e l if = = > = else : Four spaces : " " Two tab s : " " 12. 0 \* 50 = 600 \</s>

The FLAN-T5 family of models use the sentencepiece method. We notice the following:

* No newline or whitespace tokens, this would make it challenging for the model to work with code.
* The emoji and Chinese characters are both replaced by the \<unk> token. Making the model completely blind to them.

#### GPT-4

Tokenization method: BPE

Vocabulary size: a little over 100,000

Special tokens:

<|endoftext|>

Fill in the middle tokens. These three tokens enable the GPT-4 capability of generating a completion given not only the text before it but also considering the text after it. This method is explained in more detail in the paper [Efficient Training of Language Models to Fill in the Middle](https://arxiv.org/abs/2207.14255). These special tokens are:

<|fim\_prefix|>

<|fim\_middle|>

<|fim\_suffix|>

Tokenized text:

```
English and CAPITAL IZATION 
� � � � � � 
show _tokens False None elif == >= else :
Four spaces : "     " Two tabs : " 	 	 "
12 . 0 * 50 = 600
```

The GPT-4 tokenizer behaves similarly with its ancestor, the GPT-2 tokenizer. Some differences are:

* The GPT-4 tokenizer represents the four spaces as a single token. In fact, it has a specific token to every sequence of white spaces up until a list of 83 white spaces.
* The python keyword elif has its own token in GPT-4. Both this and the previous point stem from the model’s focus on code in addition to natural language.
* The GPT-4 tokenizer uses fewer tokens to represent most words. Example here include ‘CAPITALIZATION’ (two tokens, vs. four) and ‘tokens’ (one token vs. three).

#### bigcode/starcoder

Tokenization method:

Vocabulary size: about 50,000

Special tokens:

'<|endoftext|>'

FIll in the middle tokens:

'\<fim\_prefix>'

'\<fim\_middle>'

'\<fim\_suffix>'

'\<fim\_pad>'

When representing code, managing the context is important. One file might make a function call to a function that is defined in a different file. So the model needs some way of being able to identify code that is in different files in the same code repository, while making a distinction between code in different repos. That’s why starcoder uses special tokens for the name of the repository and the filename:

'\<filename>'

'\<reponame>

'\<gh\_stars>'

The tokenizer also includes a bunch of the special tokens to perform better on code. These include:

'\<issue\_start>'

'\<jupyter\_start>'

'\<jupyter\_text>'

Paper: [StarCoder: may the source be with you!](https://arxiv.org/abs/2305.06161)

Tokenized text:

```
English and CAPITAL IZATION 
� � � � � 
show _ tokens False None elif == >= else : 
Four spaces : "   " Two tabs : " 	 	 " 
1 2 . 0 * 5 0 = 6 0 0
```

This is an encoder that focuses on code generation.

* Similarly to GPT-4, it encodes the list of white spaces as a single token
* A major difference here to everyone we’ve seen so far is that each digit is assigned its own token (so 600 becomes 6 0 0). The hypothesis here is that this would lead to better representation of numbers and mathematics. In GPT-2, for example, the number 870 is represented as a single token. But 871 is represented as two tokens (8 and 71). You can intuitively see how that might be confusing to the model and how it represents numbers.

#### facebook/galactica-1.3b

The galactica model described in [Galactica: A Large Language Model for Science](https://arxiv.org/abs/2211.09085) is focused on scientific knowledge and is trained on many scientific papers, reference materials, and knowledge bases. It pays extra attention to tokenization that makes it more sensitive to the nuances of the dataset it’s representing. For example, it includes special tokens for citations, reasoning, mathematics, Amino Acid sequences, and DNA sequences.

Tokenization method:

Vocabulary size: 50,000

Special tokens:

\<s>

\<pad>

\</s>

\<unk>

References: Citations are wrapped within the two special tokens:

\[START\_REF]

\[END\_REF]

One example of usage from the paper is:\
Recurrent neural networks, long short-term memory \[START\_REF]Long Short-Term Memory, Hochreiter\[END\_REF]

Step-by-Step Reasoning -

\<work> is an interesting token that the model uses for chain-of-thought reasoning.

Tokenized text:

```
English and CAP ITAL IZATION 
� � � � � � � 
show _ tokens False None elif == > = else : 
Four spaces : "     " Two t abs : " 		 " 
1 2 . 0 * 5 0 = 6 0 0
```

The Galactica tokenizer behaves similar to star coder in that it has code in mind. It also encodes white spaces in the same way - assigning a single token to sequences of whitespace of different lengths. It differs in that it also does that for tabs, though. So from all the tokenizers we’ve seen so far, it’s the only one that’s assigned a single token to the string made up of two tabs ('\t\t')

We can now recap our tour by looking at all these examples side by side:

| bert-base-uncased              | <pre><code>[CLS] english and capital ##ization [UNK] [UNK] show _ token ##s false none eli ##f = = > = else : four spaces : " " two tab ##s : " " 12 . 0 * 50 = 600 [SEP]
</code></pre>                         |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| bert-base-cased                | <pre><code>[CLS] English and CA ##PI ##TA ##L ##I ##Z ##AT ##ION [UNK] [UNK] show _ token ##s F ##als ##e None el ##if = = > = else : Four spaces : " " Two ta ##bs : " " 12 . 0 * 50 = 600 [SEP]
</code></pre> |
| gpt2                           | <pre><code> English and CAP ITAL IZ ATION � � � � � � show _ t ok ens False None el if == >= else : Two tabs :" " Four spaces : " " 12 . 0 * 50 = 600 
</code></pre>                                            |
| google/flan-t5-xxl             | <pre><code>English and CA PI TAL IZ ATION &#x3C;unk> &#x3C;unk> show _ to ken s Fal s e None e l if = = > = else : two tab s : " " Four spaces : " " 12. 0 * 50 = 600 &#x3C;/s> 
</code></pre>                  |
| GPT-4                          | <pre><code>English and CAPITAL IZATION � � � � � � show _tokens False None elif == >= else : Four spaces : " " Two tabs : " " 12 . 0 * 50 = 600
</code></pre>                                                   |
| bigcode/starcoder              | <pre><code>English and CAPITAL IZATION � � � � � show _ tokens False None elif == >= else : Four spaces : " " Two tabs : " " 1 2 . 0 * 5 0 = 6 0 0
</code></pre>                                                |
| facebook/galactica-1.3b        | <pre><code>English and CAP ITAL IZATION � � � � � � � show _ tokens False None elif == > = else : Four spaces : " " Two t abs : " " 1 2 . 0 * 5 0 = 6 0 0
</code></pre>                                         |
| meta-llama/Llama-2-70b-chat-hf | <pre><code>&#x3C;s> English and C AP IT AL IZ ATION � � � � � � � show _ to kens False None elif == >= else : F our spaces : " " Two tabs : " " 1 2 . 0 * 5 0 = 6 0 0
</code></pre>                             |

Notice how there’s a new tokenizer added in the bottom. By now, you should be able to understand many of its properties by just glancing at this output. This is the tokenizer for LLaMA2, the most recent of these models.

### Tokenizer Properties

The preceding guided tour of trained tokenizers showed a number of ways in which actual tokenizers differ from each other. But what determines their tokenization behavior? There are three major groups of design choices that determine how the tokenizer will break down text: The tokenization method, the initialization parameters, and the dataset we train the tokenizer (but not the model) on.

#### Tokenization methods

As we’ve seen, there are a number of tokenization methods with Byte-Pair Encoding (BPE), WordPiece, and SentencePiece being some of the more popular ones. Each of these methods outlines an algorithm for how to choose an appropriate set of tokens to represent a dataset. A great overview of all these methods can be found in the Hugging Face [Summary of the tokenizers page](https://huggingface.co/docs/transformers/tokenizer\_summary).

#### Tokenizer Parameters

After choosing a tokenization method, an LLM designer needs to make some decisions about the parameters of the tokenizer. These include:

Vocabulary size

How many tokens to keep in the tokenizer’s vocabulary? (30K, 50K are often used vocabulary size values, but more and more we’re seeing larger sizes like 100K)

Special tokens

What special tokens do we want the model to keep track of. We can add as many of these as we want, especially if we want to build LLM for special use cases. Common choices include:

* Beginning of text token (e.g., \<s>)
* End of text token
* Padding token
* Unknown token
* CLS token
* Masking token

Aside from these, the LLM designer can add tokens that help better model the domain of the problem they’re trying to focus on, as we’ve seen with Galactica’s \<work> and \[START\_REF] tokens.

Capitalization

In languages such as English, how do we want to deal with capitalization? Should we convert everything to lower-case? (Name capitalization often carries useful information, but do we want to waste token vocabulary space on all caps versions of words?). This is why some models are released in both cased and uncased versions (like [Bert-base cased](https://huggingface.co/bert-base-cased) and the more popular [Bert-base uncased](https://huggingface.co/bert-base-uncased)).

#### The Tokenizer Training Dataset

Even if we select the same method and parameters, tokenizer behavior will be different based on the dataset it was trained on (before we even start model training). The tokenization methods mentioned previously work by optimizing the vocabulary to represent a specific dataset. From our guided tour we’ve seen how that has an impact on datasets like code, and multilingual text.

For code, for example, we’ve seen that a text-focused tokenizer may tokenize the indentation spaces like this (We’ll highlight some tokens in yellow and green):

```
def add_numbers(a, b):
...."""Add the two numbers `a` and `b`."""
....return a + b
```

Which may be suboptimal for a code-focused model. Code-focused models instead tend to make different tokenization choices:

```
def add_numbers(a, b):
...."""Add the two numbers `a` and `b`."""
....return a + b
```

These tokenization choices make the model’s job easier and thus its performance has a higher probability of improving.

A more detailed tutorial on training tokenizers can be found in the [Tokenizers section of the Hugging Face course](https://huggingface.co/learn/nlp-course/chapter6/1?fw=pt). and in [Natural Language Processing with Transformers, Revised Edition](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/).

### A Language Model Holds Embeddings for the Vocabulary of its Tokenizer

After a tokenizer is initialized, it is then used in the training process of its associated language model. This is why a pre-trained language model is linked with its tokenizer and can’t use a different tokenizer without training.

The language model holds an embedding vector for each token in the tokenizer’s vocabulary as we can see in [Figure 5-6](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch05.html#fig\_6\_\_a\_language\_model\_holds\_an\_embedding\_vector\_assoc). In the beginning, these vectors are randomly initialized like the rest of the model’s weights, but the training process assigns them the values that enable the useful behavior they’re trained to perform.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/tokens_token_embeddings_963889_06.png" alt="A language model holds an embedding vector associated with each token in its tokenizer." height="198" width="600"><figcaption></figcaption></figure>

**Figure 5-6. A language model holds an embedding vector associated with each token in its tokenizer.**

### Creating Contextualized Word Embeddings with Language Models

Now that we’ve covered token embeddings as the input to a language model, let’s look at how language models can _create_ better token embeddings. This is one of the main ways of using language models for text representation that empowers applications like named-entity recognition or extractive text summarization (which summarizes a long text by highlighting to most important parts of it, instead of generating new text as a summary).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/tokens_token_embeddings_963889_07.png" alt="Language models produce contextualized token embeddings that improve on raw  static token embeddings" height="298" width="600"><figcaption></figcaption></figure>

**Figure 5-7. Language models produce contextualized token embeddings that improve on raw, static token embeddings**

Instead of representing each token or word with a static vector, language models create contextualized word embeddings (shown in [Figure 5-7](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch05.html#fig\_7\_\_language\_models\_produce\_contextualized\_token\_emb)) that represent a word with a different token based on its context. These vectors can then be used by other systems for a variety of tasks. In addition to the text applications we mentioned in the previous paragraph, these contextualized vectors, for example, are what powers AI image generation systems like Dall-E, Midjourney, and Stable Diffusion, for example.

#### Code Example: Contextualized Word Embeddings From a Language Model (Like BERT)

Let’s look at how we can generate contextualized word embeddings, the majority of this code should be familiar to you by now:

```
from transformers import AutoModel, AutoTokenizer
# Load a tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
# Load a language model
model = AutoModel.from_pretrained("microsoft/deberta-v3-xsmall")
# Tokenize the sentence
tokens = tokenizer('Hello world', return_tensors='pt')
# Process the tokens
output = model(**tokens)[0]
```

This code downloads a pre-trained tokenizer and model, then uses them to process the string “Hello world”. The output of the model is then saved in the output variable. Let’s inspect that variable by first printing its dimensions (we expect it to be a multi-dimensional array).

The model we’re using here is called DeBERTA v3, which at the time of writing, is one of the best-performing language models for token embeddings while being small and highly efficient. It is described in the paper [DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing](https://openreview.net/forum?id=sE7-XhLxHA).

```
output.shape
```

This prints out:

```
torch.Size([1, 4, 384])
```

We can ignore the first dimension and read this as four tokens, each one embedded in 384 values.

But what are these four vectors? Did the tokenizer break the two words into four tokens, or is something else happening here? We can use what we’ve learned about tokenizers to inspect them:

```
for token in tokens['input_ids'][0]:
    print(tokenizer.decode(token))
```

Which prints out:

```
[CLS] 
Hello
world 
[SEP]
```

Which shows that this particular tokenizer and model operate by adding the \[CLS] and \[SEP] tokens to the beginning and end of a string.

Our language model has now processed the text input. The result of its output is the following:

```
tensor([[
[-3.3060, -0.0507, -0.1098, ..., -0.1704, -0.1618, 0.6932], 
[ 0.8918, 0.0740, -0.1583, ..., 0.1869, 1.4760, 0.0751], 
[ 0.0871, 0.6364, -0.3050, ..., 0.4729, -0.1829, 1.0157], 
[-3.1624, -0.1436, -0.0941, ..., -0.0290, -0.1265, 0.7954]
]], grad_fn=<NativeLayerNormBackward0>)
```

This is the raw output of a language model. The applications of large language models build on top of outputs like this.

We can recap the input tokenization and resulting outputs of a language model in [Figure 5-8](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch05.html#fig\_8\_\_a\_language\_model\_operates\_on\_raw\_static\_embeddi). Technically, the switch from token IDs into raw embeddings is the first step that happens inside a language model.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/tokens_token_embeddings_963889_08.png" alt="A language model operates on raw  static embeddings as its input and produces contextual text embeddings." height="446" width="600"><figcaption></figcaption></figure>

**Figure 5-8. A language model operates on raw, static embeddings as its input and produces contextual text embeddings.**

A visual like this is essential for the next chapter when we start to look at how Transformer-based LLMs work under the hood.

## Word Embeddings

Token embeddings are useful even outside of large language models. Embeddings generated by pre-LLM methods like Word2Vec, Glove, and Fasttext still have uses in NLP and beyond NLP. In this section, we’ll look at how to use pre-trained Word2Vec embeddings and touch on how the method creates word embeddings. Seeing how Word2Vec is trained will prime you for the chapter on contrastive training. Then in the following section, we’ll see how those embeddings can be used for recommendation systems.

### Using Pre-trained Word Embeddings

Let’s look at how we can download pre-trained word embeddings using the [Gensim](https://radimrehurek.com/gensim/) library

```
import gensim
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# Download embeddings (66MB, glove, trained on wikipedia, vector size: 50)
# Other options include "word2vec-google-news-300"
# More options at https://github.com/RaRe-Technologies/gensim-data
model = api.load("glove-wiki-gigaword-50")
```

Here, we’ve downloaded the embeddings of a large number of words trained on wikipedia. We can then explore the embedding space by seeing the nearest neighbors of a specific word, ‘king’ for example:

```
model.most_similar([model['king']], topn=11)
```

Which outputs:

```
[('king', 1.0000001192092896), 
('prince', 0.8236179351806641), 
('queen', 0.7839043140411377), 
('ii', 0.7746230363845825), 
('emperor', 0.7736247777938843), 
('son', 0.766719400882721), 
('uncle', 0.7627150416374207), 
('kingdom', 0.7542161345481873), 
('throne', 0.7539914846420288), 
('brother', 0.7492411136627197), 
('ruler', 0.7434253692626953)]
```

### The Word2vec Algorithm and Contrastive Training

The word2vec algorithm described in the paper [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) is described in detail in [The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/). The central ideas are condensed here as we build on them when discussing one method for creating embeddings for recommendation engines in the following section.

Just like LLMs, word2vec is trained on examples generated from text. Let’s say for example, we have the text "_Thou shalt not make a machine in the likeness of a human mind_" from the _Dune_ novels by Frank Herbert. The algorithm uses a sliding window to generate training examples. We can for example have a window size two, meaning that we consider two neighbors on each side of a central word.

The embeddings are generated from a classification task. This task is used to train a neural network to predict if words appear in the same context or not. We can think of this as a neural network that takes two words and outputs 1 if they tend to appear in the same context, and 0 if they do not.

In the first position for the sliding window, we can generate four training examples as we can see in [Figure 5-9](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch05.html#fig\_9\_\_a\_sliding\_window\_is\_used\_to\_generate\_training\_ex).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/tokens_token_embeddings_963889_09.png" alt="A sliding window is used to generate training examples for the word2vec algorithm to later predict if two words are neighbors or not." height="360" width="600"><figcaption></figcaption></figure>

**Figure 5-9. A sliding window is used to generate training examples for the word2vec algorithm to later predict if two words are neighbors or not.**

In each of the produced training examples, the word in the center is used as one input, and each of its neighbors is a distinct second input in each training example. We expect the final trained model to be able to classify this neighbor relationship and output 1 if the two input words it receives are indeed neighbors.

These training examples are visualized in [Figure 5-10](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch05.html#fig\_10\_\_each\_generated\_training\_example\_shows\_a\_pair\_of).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/tokens_token_embeddings_963889_10.png" alt="Each generated training example shows a pair of neighboring words." height="283" width="600"><figcaption></figcaption></figure>

**Figure 5-10. Each generated training example shows a pair of neighboring words.**

If, however, we have a dataset of only a target value of 1, then a model can ace it by output 1 all the time. To get around this, we need to enrich our training dataset with examples of words that are not typically neighbors. These are called negative examples and are shown in [Figure 5-11](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch05.html#fig\_11\_\_we\_need\_to\_present\_our\_models\_with\_negative\_exam).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/tokens_token_embeddings_963889_11.png" alt="We need to present our models with negative examples  words that are not usually neighbors. A better model is able to better distinguish between the positive and negative examples." height="386" width="600"><figcaption></figcaption></figure>

**Figure 5-11. We need to present our models with negative examples: words that are not usually neighbors. A better model is able to better distinguish between the positive and negative examples.**

It turns out that we don’t have to be too scientific in how we choose the negative examples. A lot of useful models are result from simple ability to detect positive examples from randomly generated examples (inspired by an important idea called Noise Contrastive Estimation and described in [Noise-contrastive estimation: A new estimation principle for unnormalized statistical models](https://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf)). So in this case, we get random words and add them to the dataset and indicate that they are not neighbors (and thus the model should output 0 when it sees them.

With this, we’ve seen two of the main concepts of word2vec ([Figure 5-12](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch05.html#fig\_12\_\_skipgram\_and\_negative\_sampling\_are\_two\_of\_the\_ma)): Skipgram - the method of selecting neighboring words and negative sampling - adding negative examples by random sampling from the dataset.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/tokens_token_embeddings_963889_12.png" alt="Skipgram and Negative Sampling are two of the main ideas behind the word2vec algorithm and are useful in many other problems that can be formulated as token sequence problems." height="233" width="600"><figcaption></figcaption></figure>

**Figure 5-12. Skipgram and Negative Sampling are two of the main ideas behind the word2vec algorithm and are useful in many other problems that can be formulated as token sequence problems.**

We can generate millions and even billions of training examples like this from running text. Before proceeding to train a neural network on this dataset, we need to make a couple of tokenization decisions, which, just like we’ve seen with LLM tokenizers, include how to deal with capitalization and punctuation and how many tokens we want in our vocabulary.

We then create an embedding vector for each token, and randomly initialize them, as can be seen in [Figure 5-13](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch05.html#fig\_13\_\_a\_vocabulary\_of\_words\_and\_their\_starting\_random). In practice, this is a matrix of dimensions vocab\_size x embedding\_dimensions.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/tokens_token_embeddings_963889_13.png" alt="A vocabulary of words and their starting  random  uninitialized embedding vectors." height="441" width="600"><figcaption></figcaption></figure>

**Figure 5-13. A vocabulary of words and their starting, random, uninitialized embedding vectors.**

A model is then trained on each example to take in two embedding vectors and predict if they’re related or not. We can see what this looks like in [Figure 5-14](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch05.html#fig\_14\_\_a\_neural\_network\_is\_trained\_to\_predict\_if\_two\_wo):

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/tokens_token_embeddings_963889_14.png" alt="A neural network is trained to predict if two words are neighbors. It updates the embeddings in the training process to produce the final  trained embeddings." height="178" width="600"><figcaption></figcaption></figure>

**Figure 5-14. A neural network is trained to predict if two words are neighbors. It updates the embeddings in the training process to produce the final, trained embeddings.**

Based on whether its prediction was correct or not, the typical machine learning training step updates the embeddings so that the next the model is presented with those two vectors, it has a better chance of being more correct. And by the end of the training process, we have better embeddings for all the tokens in our vocabulary.

This idea of a model that takes two vectors and predicts if they have a certain relation is one of the most powerful ideas in machine learning, and time after time has proven to work very well with language models. This is why we’re dedicating chapter XXX to go over this concept and how it optimizes language models for specific tasks (like sentence embeddings and retrieval).

The same idea is also central to bridging modalities like text and images which is key to AI Image generation models. In that formulation, a model is presented with an image and a caption, and it should predict whether that caption describes this image or not.

## Embeddings for Recommendation Systems

The concept of token embeddings is useful in so many other domains. In industry, it’s widely used for recommendation systems, for example.

### Recommending songs by embeddings

In this section we’ll use the Word2vec algorithm to embed songs using human-made music playlists. Imagine if we treated each song as we would a word or token, and we treated each playlist like a sentence. These embeddings can then be used to recommend similar songs which often appear together in playlists.

The [dataset](https://www.cs.cornell.edu/\~shuochen/lme/data\_page.html) we’ll use was collected by Shuo Chen from Cornell University. The dataset contains playlists from hundreds of radio stations around the US. [Figure 5-15](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch05.html#fig\_15\_\_for\_song\_embeddings\_that\_capture\_song\_similarity) demonstrates this dataset.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/tokens_token_embeddings_963889_15.png" alt="For song embeddings that capture song similarity we ll use a dataset made up of a collection of playlists  each containing a list of songs." height="144" width="600"><figcaption></figcaption></figure>

**Figure 5-15. For song embeddings that capture song similarity we’ll use a dataset made up of a collection of playlists, each containing a list of songs.**

Let’s demonstrate the end product before we look at how it’s built. So let’s give it a few songs and see what it recommends in response.

Let’s start by giving it Michael Jackson’s _Billie Jean_, the song with ID #3822.

```
print_recommendations(3822)
title Billie Jean 
artist Michael Jackson
Recommendations:
```

| id    | title                          | artist                  |
| ----- | ------------------------------ | ----------------------- |
| 4181  | Kiss                           | Prince & The Revolution |
| 12749 | Wanna Be Startin’ Somethin’    | Michael Jackson         |
| 1506  | The Way You Make Me Feel       | Michael Jackson         |
| 3396  | Holiday                        | Madonna                 |
| 500   | Don’t Stop ‘Til You Get Enough | Michael Jackson         |

That looks reasonable. Madonna, Prince, and other Michael Jackson songs are the nearest neighbors.

Let’s step away from Pop and into Rap, and see the neighbors of 2Pac’s California Love:

```
print_recommendations(842)
```

| id   | title                                                  | artist                  |
| ---- | ------------------------------------------------------ | ----------------------- |
| 413  | If I Ruled The World (Imagine That) (w\\/ Lauryn Hill) | Nas                     |
| 196  | I’ll Be Missing You                                    | Puff Daddy & The Family |
| 330  | Hate It Or Love It (w\\/ 50 Cent)                      | The Game                |
| 211  | Hypnotize                                              | The Notorious B.I.G.    |
| 5788 | Drop It Like It’s Hot (w\\/ Pharrell)                  | Snoop Dogg              |

Another quite reasonable list!

```
# Get the playlist dataset file
data = request.urlopen('https://storage.googleapis.com/maps-premium/dataset/yes_complete/train.txt')
# Parse the playlist dataset file. Skip the first two lines as
# they only contain metadata
lines = data.read().decode("utf-8").split('\n')[2:]
# Remove playlists with only one song
playlists = [s.rstrip().split() for s in lines if len(s.split()) > 1]
print( 'Playlist #1:\n ', playlists[0], '\n')
print( 'Playlist #2:\n ', playlists[1])
Playlist #1: ['0', '1', '2', '3', '4', '5', ..., '43'] 
Playlist #2: ['78', '79', '80', '3', '62', ..., '210']
Let's train the model:
model = Word2Vec(playlists, vector_size=32, window=20, negative=50, min_count=1, workers=4)
```

That takes a minute or two to train and results in embeddings being calculated for each song that we have. Now we can use those embeddings to find similar songs exactly as we did earlier with words.

```
song_id = 2172
# Ask the model for songs similar to song #2172
model.wv.most_similar(positive=str(song_id))
```

Which outputs:

```
[('2976', 0.9977465271949768), 
('3167', 0.9977430701255798), 
('3094', 0.9975950717926025), 
('2640', 0.9966474175453186), 
('2849', 0.9963167905807495)]
```

And that is the list of the songs whose embeddings are most similar to song 2172. See the jupyter notebook for the code that links song ids to their names and artist names.

In this case, the song is:

```
title Fade To Black 
artist Metallica
```

Resulting in recommendations that are all in the same heavy metal and hard rock genre:

| id    | title            | artist        |
| ----- | ---------------- | ------------- |
| 11473 | Little Guitars   | Van Halen     |
| 3167  | Unchained        | Van Halen     |
| 5586  | The Last In Line | Dio           |
| 5634  | Mr. Brownstone   | Guns N’ Roses |
| 3094  | Breaking The Law | Judas Priest  |

## Summary

In this chapter, we have covered LLM tokens, tokenizers, and useful approaches to use token embeddings beyond language models.

* Tokenizers are the first step in processing the input to a LLM -- turning text into a list of token IDs.
* Some of the common tokenization schemes include breaking text down into words, subword tokens, characters, or bytes
* A tour of real-world pre-trained tokenizers (from BERT to GPT2, GPT4, and other models) showed us areas where some tokenizers are better (e.g., preserving information like capitalization, new lines, or tokens in other languages) and other areas where tokenizers are just different from each other (e.g., how they break down certain words).
* Three of the major tokenizer design decisions are the tokenizer algorithm (e.g., BPE, WordPiece, SentencePiece), tokenization parameters (including vocabulary size, special tokens, capitalization, treatment of capitalization and different languages), and the dataset the tokenizer is trained on.
* Language models are also creators of high-quality contextualized token embeddings that improve on raw static embeddings. Those contextualized token embeddings are what’s used for tasks including NER, extractive text summarization, and span classification.
* Before LLMs, word embedding methods like word2vec, Glove and Fasttext were popular. They still have some use cases within and outside of language processing.
* The Word2Vec algorithm relies on two main ideas: Skipgram and Negative Sampling. It also uses contrastive training similar to the one we’ll see in the contrastive training chapter.
* Token embeddings are useful for creating and improving recommender systems as we’ve seen in the music recommender we’ve built from curated song playlists.
