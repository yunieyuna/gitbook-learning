# Chapter 4. Multimodel Large Language Models

## Chapter 4. Multimodal Large Language Models

## A NOTE FOR EARLY RELEASE READERS

With Early Release ebooks, you get books in their earliest form—the author’s raw and unedited content as they write—so you can take advantage of these technologies long before the official release of these titles. In particular, some of the formatting may not match the description in the text: this will be resolved when the book is finalized.

This will be the 7th chapter of the final book. Please note that the GitHub repo will be made active later on.

If you have comments about how we might improve the content and/or examples in this book, or if you notice missing material within this chapter, please reach out to the editor at _mcronin@oreilly.com_.

When you think about Large Language Models (LLMs), multimodality might not be the first thing that comes to mind. After all, they are _Language_ Models!

We have seen all manner of emerging behaviors rising from LLMs, from generalization capabilities and reasoning to arithmetic and linguistics. As models grow larger and smarter, so do their skill sets.[1](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch04.html#id209)

The ability to receive and reason with multimodal input might further increase and help emerge capabilities that were previously locked. In practice, Language does not solely live in a vacuum. As an example, your body language, facial expressions, intonation, etc. are all methods of communication that enhance the spoken word.

The same thing applies to Large Language Models, if we can enable them to reason about multimodal information, their capabilities might increase.

In this chapter, we will explore a number of different LLMs that have multimodal capabilities and what that means for practical use cases. We will start by exploring how images are converted to numerical representations using an adaption of the original transformer technique. Then, we will show how LLMs can be extended to include vision tasks using this transformer.

## Transformers for Vision

Throughout the chapters of this book, we have seen the success of using transformer-based models for a variety of language modeling tasks, from classification and clustering to search and generative modeling.

So it might not be surprising that researchers have been looking at a way to generalize some of the transformer’s success to the field of computer vision.

The method they came up with is called the Vision Transformer (ViT) which has been shown to do tremendously well on image recognition tasks compared to the previously default Convolutional Neural Networks (CNNs).[2](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch04.html#id210) Like the original transformer, ViT is used to transform unstructured data, an image, into representations that can be used for a variety of tasks, like classification as illustrated in [Figure 4-1](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch04.html#fig\_1\_both\_the\_original\_transformer\_as\_well\_as\_the\_visio).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/multimodal_large_language_models_406548_01.png" alt="Both the original transformer as well as the vision transformer take unstructured data  convert it to numerical representations  and finally use that for tasks like classification." height="379" width="600"><figcaption></figcaption></figure>

**Figure 4-1. Both the original transformer as well as the vision transformer take unstructured data, convert it to numerical representations, and finally use that for tasks like classification.**

ViT relies on an important component of the transformer architecture, namely the encoder. As we saw in Chapter 1, the encoder is responsible for converting textual input into numerical representations before being passed to the decoder. However, before the encoder can perform its duties, the textual input needs to be tokenized first as is illustrated in [Figure 4-2](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch04.html#fig\_2\_text\_is\_passed\_to\_one\_or\_multiple\_encoders\_by\_firs).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/multimodal_large_language_models_406548_02.png" alt="Text is passed to one or multiple encoders by first tokenizing it using a tokenizer." height="183" width="600"><figcaption></figcaption></figure>

**Figure 4-2. Text is passed to one or multiple encoders by first tokenizing it using a tokenizer.**

Since an image does not consist of words this tokenization process cannot be used for visual data. Instead, the authors of ViT came up with a method for tokenizing images into “words” which allowed them to use the original encoder structure.

Imagine that you have an image of a cat. This image is represented by a number of pixels, let’s say 512 by 512 pixels. Each individual pixel does not convey much information but when you combine patches of pixels, you slowly start to see more information.

ViT uses a principle much like that. Instead of splitting text up into tokens, it converts the original image into patches of images. In other words, it cuts the image into a number of pieces horizontally and vertically as illustrated in [Figure 4-3](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch04.html#fig\_3\_the\_tokenization\_process\_for\_image\_input\_it\_con).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/multimodal_large_language_models_406548_03.png" alt="The  tokenization  process for image input. It converts an image into patches of sub images." height="464" width="600"><figcaption></figcaption></figure>

**Figure 4-3. The “tokenization” process for image input. It converts an image into patches of sub-images.**

Just like we are converting text into tokens of text, we are converting an image into patches of images. The flattened input of image patches can be thought of as the tokens in a piece of text.

However, unlike tokens, we cannot just assign each patch with an ID since these patches will rarely be found in other images, unlike the vocabulary of a text.

Instead, the patches are linearly embedded to create numerical representations, namely embeddings. These can then be used as the input of a transformer model. That way, the patches of images are treated the same way as tokens. The full process is illustrated in [Figure 4-4](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch04.html#fig\_4\_the\_main\_algorithm\_behind\_vit\_after\_patching\_the).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/multimodal_large_language_models_406548_04.png" alt="The main algorithm behind ViT. After patching the images and linearly projecting them  the patch embeddings are passed to the encoder and treated as if they were textual tokens." height="477" width="600"><figcaption></figcaption></figure>

**Figure 4-4. The main algorithm behind ViT. After patching the images and linearly projecting them, the patch embeddings are passed to the encoder and treated as if they were textual tokens.**

For illustrative purposes, the images in the examples were patched into 3 by 3 patches but the original implementation used 16 by 16 patches. After all, the paper is called “An image is worth 16x16 words”.

What is so interesting about this approach is that the moment the embeddings are passed to the encoder, they are treated as if they were textual tokens. From that point forward, there is no difference in how a textual or image trains and their outputs

Due to their similarities, the ViT is often used to make all kinds of language models multimodal. One of the most straightforward ways to use them is during the training of embedding models.

## Multimodal Embedding Models

In previous chapters, like Chapters X, X, and X, we used embedding models to capture the semantic content of textual representations, such as books and documents. We saw that we could use these embeddings or numerical representations to find similar documents, apply classification tasks, and even perform topic modeling.

As we have seen many times before, embeddings often are an important driver behind LLM applications. They are an efficient method for capturing large-scale information and searching for the needle in the haystack of information.

That said, we have only looked at monomodal embedding models thus far. Embedding models that only focus on generating embeddings for textual representations. Although embedding models exist for solely embedding imagery, we will look at embedding models that can capture both textual as well as vision representations. We illustrate this in [Figure 4-5](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch04.html#fig\_5\_multimodal\_embedding\_models\_can\_create\_embeddings).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/multimodal_large_language_models_406548_05.png" alt="Multimodal embedding models can create embeddings for multiple modalities in the same vector space." height="293" width="600"><figcaption></figcaption></figure>

**Figure 4-5. Multimodal embedding models can create embeddings for multiple modalities in the same vector space.**

A big advantage is that it allows for comparing multimodal representations since the resulting embeddings lie in the same vector space, as illustrated in [Figure 4-6](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch04.html#fig\_6\_tmultimodal\_embedding\_models\_can\_create\_embeddings). For instance, using such a multimodal embedding model, we can find images based on input text. What images would we find if we search for images similar to “pictures of a puppy”? Vice versa would also be possible. Which documents are best related to this question?

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/multimodal_large_language_models_406548_06.png" alt="tMultimodal embedding models can create embeddings for multiple modalities in the same vector space." height="343" width="600"><figcaption></figcaption></figure>

**Figure 4-6. Multimodal embedding models can create embeddings for multiple modalities in the same vector space.**

There are a number of multimodal embedding models out there but the most well-known and currently most-used model is CLIP (Contrastive Language-Image Pre-Training).

### CLIP: Connecting Text and Images

CLIP is an embedding model that can compute embeddings of both images and texts. The resulting embeddings lie in the same vector space which means that the embeddings of images can be compared with the embeddings of text.[3](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch04.html#id211)

This capability of comparison makes CLIP, and similar models, usable for tasks such as:

Zeroshot classification

We can compare the embedding of an image with that of the description of its possible classes to find which class is most similar

Clustering

Cluster both images and a collection of keywords to find which keywords belong to which sets of images

Search

Across billions of texts or images, we can quickly find what relates to an input text or image

Generation

Use multimodal embeddings to drive the generation of images (e.g., stable diffusion)

#### How can CLIP generate multimodal embeddings?

The procedure of CLIP is actually quite straightforward. Imagine that you have a dataset with millions of images alongside captions as we illustrate in [Figure 4-7](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch04.html#fig\_7\_the\_type\_of\_data\_that\_is\_needed\_to\_train\_a\_multimo).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/multimodal_large_language_models_406548_07.png" alt="The type of data that is needed to train a multimodal embedding model." height="206" width="600"><figcaption></figcaption></figure>

**Figure 4-7. The type of data that is needed to train a multimodal embedding model.**

This dataset can be used to create two representations for each pair, the image and its caption. To do so, CLIP uses a text encoder to embed text and an image encoder to embed images. As is shown in [Figure 4-8](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch04.html#fig\_8\_in\_the\_first\_step\_of\_training\_clip\_both\_images\_an), the result is an embedding for both the image and its corresponding caption.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/multimodal_large_language_models_406548_08.png" alt="In the first step of training CLIP  both images and text are embedded using an image and text encoder respectively." height="293" width="600"><figcaption></figcaption></figure>

**Figure 4-8. In the first step of training CLIP, both images and text are embedded using an image and text encoder respectively.**

The pair of embeddings that are generated are compared through cosine similarity. As we saw in Chapter 2, cosine similarity is the cosine of the angle between vectors which is calculated through the dot product of the embeddings and divided by the product of their lengths.

When we start training, the similarity between the image embedding and text embedding will be low as they are not yet optimized to be within the same vector space. During training, we optimize for the similarity between the embeddings and want to maximize them for similar image/caption pairs and minimize them for dissimilar image/caption pairs.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/multimodal_large_language_models_406548_09.png" alt="In the second step of training CLIP  the similarity between the sentence and image embedding is calculated using cosine similarity." height="293" width="600"><figcaption></figcaption></figure>

**Figure 4-9. In the second step of training CLIP, the similarity between the sentence and image embedding is calculated using cosine similarity.**

After calculating their similarity, the model is updated and the process starts again with new batches of data and updated representations. This method is called contrastive learning and we will go in-depth into its inner workings in Chapter 13 where we will create our own embedding model.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/multimodal_large_language_models_406548_10.png" alt="In the third step of training CLIP  the text and image encoders are updated to match what the intended similarity should be. This updates the embeddings such that they are closer in vector space if the inputs are similar." height="293" width="600"><figcaption></figcaption></figure>

**Figure 4-10. In the third step of training CLIP, the text and image encoders are updated to match what the intended similarity should be. This updates the embeddings such that they are closer in vector space if the inputs are similar.**

Eventually, we expect the embedding of an image of a cat would be similar to the embedding of the sentence “a picture of a cat”. As we will see in Chapter 13, to make sure the representations are as accurate as possible, negative examples of images and captions that are not related should also be included in the training process.

Modeling similarity is not only knowing what makes things similar to one another but also what makes them different and dissimilar.

#### OpenCLIP

For this example, we are going to be using models from the open-source variant of CLIP, namely OpenCLIP ([https://github.com/mlfoundations/open\_clip](https://github.com/mlfoundations/open\_clip)).

Using OpenCLIP, or any CLIP model, boils down to two things, processing the textual and image inputs before passing them to the main model.

Before doing so, let’s take a look at a small example where we will be using one of the images we have seen before. Namely, an AI-generated image (though stable-diffusion) of a puppy playing in the snow as illustrated in [Figure 4-11](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch04.html#fig\_11\_an\_ai\_generated\_image\_of\_a\_puppy\_playing\_in\_the\_sn):

```
from urllib.request import urlopen
from PIL import Image
 
# Load an AI-generated image of a puppy playing in the snow
image = Image.open(urlopen("https://i.imgur.com/iQ5OtWi.png"))
caption = "a puppy playing in the snow"
```

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/multimodal_large_language_models_406548_11.png" alt="An AI generated image of a puppy playing in the snow." height="512" width="512"><figcaption></figcaption></figure>

**Figure 4-11. An AI-generated image of a puppy playing in the snow.**

Since we have a caption for this image, we can use OpenCLIP to generate embeddings for both.

To do so, we load in three models:

A tokenizer for tokenizing the textual input

A preprocessor to preprocess and resize the image

The main model that converts the previous outputs to embeddings

<pre><code><strong>from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
</strong> 
model_id = "openai/clip-vit-base-patch32"
 
# Load a tokenizer to preprocess the text
tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
 
# Load a processor to preprocess the images
processor = CLIPProcessor.from_pretrained(model_id)
 
# Main model for generating text and image embeddings
model = CLIPModel.from_pretrained(model_id)
</code></pre>

After having loaded in the models, preprocessing our input is straightforward. Let’s start with the tokenizer and see what happens if we preprocess our input:

```
>>> # Tokenize our input
>>> inputs = tokenizer(caption, return_tensors="pt"); inputs
 
{'input_ids': tensor([[49406,   320,  6829,  1629,   530,   518,  2583, 49407]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1]])}
```

Our input text has been converted to input ids. To see what those represent, let’s convert them to tokens:

```
>>> tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
 
['<|startoftext|>',
 'a</w>',
 'puppy</w>',
 'playing</w>',
 'in</w>',
 'the</w>',
 'snow</w>',
 '<|endoftext|>']
```

As we often have seen before, the text is split up into tokens. Additionally, we now also see that the start and end of the text is indicated to separate it from a potential image embedding. You might also notice that the `[CLS]` token is missing. In CLIP, the `[CLS]` token is actually used to represent the image embedding.

Now that we have preprocessed our caption, next up is to create the embedding:

```
>>> # Create a text embedding
>>> text_embedding = model.get_text_features(**inputs)
>>> text_embedding.shape
 
torch.Size([1, 512])
```

Before we can create our image embedding, like the text embedding, we will need to preprocess it as the model expects the input image to have certain characteristics, like its size and shape.

To do so, we can use the processor that we created before:

<pre><code>>>> # Preprocess image
<strong>>>> processed_image = processor(text=None, images=image, return_tensors='pt')['pixel_values']
</strong>>>> processed_image.shape
 
torch.Size([1, 3, 224, 224])
</code></pre>

The original image was 512 by 512 pixels. Notice that the preprocessing of this image reduced its size to 224 by 224 pixels as that is its expected size.

Let’s visualize, in [Figure 4-12](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch04.html#fig\_12\_the\_preprocessed\_input\_image\_by\_clip), the preprocessed image to see what it actually is doing:

```
import numpy as np
 
# Prepare image for visualization
img = processed_image.squeeze(0).T
img = np.einsum('ijk->jik', img)
 
# Visualize preprocessed image
plt.imshow(a)
plt.axis('off')
```

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/multimodal_large_language_models_406548_12.png" alt="The preprocessed input image by CLIP." height="389" width="389"><figcaption></figcaption></figure>

**Figure 4-12. The preprocessed input image by CLIP.**

To convert this preprocessed image into embeddings, we can call the `model` as we did before:

```
>>> # Create the image embedding
>>> image_embedding = model.get_image_features(processed_image)
>>> image_embedding.shape
 
torch.Size([1, 512])
```

Notice that the shape of the resulting image embedding is exactly the same as that of the text embedding. This is important as it allows us to compare their embeddings and see whether they actually are similar.

We can use these embeddings to calculate the probability that the caption belongs to the image by calculating their dot product and taking the softmax:

```
>>> # Calculate the probability of the text belonging to the image
>>> text_probs = (100.0 * image_embedding @ text_embedding.T).softmax(dim=-1)
>>> text_probs
 
tensor([[1.]], grad_fn=<SoftmaxBackward0>)
```

It gives us back a score of 1 indicating that the model is certain that the caption belongs to the image.

We can extend this example by calculating the similarity between the embeddings. By normalizing the embeddings first before calculating the dot product, we get a value that lies between 0 and 1:

<pre><code>>>> # Normalize the embeddings
<strong>>>> text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
</strong><strong>>>> image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
</strong>>>> 
>>> # Calculate their similarity
>>> text_embedding = text_embedding.detach().cpu().numpy()
>>> image_embedding = image_embedding.detach().cpu().numpy()
>>> score = np.dot(text_embedding, image_embedding.T)
>>> score
 
array([[0.33149636]], dtype=float32)
</code></pre>

We get a similarity score of 0.33 which is difficult to interpret considering we do not know what the model considers a low versus a high similarity score.

Instead, let’s extend the example with more images and captions as illustrated in [Figure 4-13](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch04.html#fig\_13\_the\_similarity\_matrix\_between\_three\_images\_and\_thr).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/multimodal_large_language_models_406548_13.png" alt="The similarity matrix between three images and three captions." height="497" width="600"><figcaption></figcaption></figure>

**Figure 4-13. The similarity matrix between three images and three captions.**

It seems that a score of 0.33 is indeed high considering the similarities with other images are quite a bit lower.

**TIP**

In sentence-transformers, there are a few CLIP-based models implemented that make it much easier to create embeddings. It only takes a few lines of code:

```
from sentence_transformers import SentenceTransformer, util
 
# Load SBERT-compatible CLIP model
model = SentenceTransformer('clip-ViT-B-32')
 
# Encode the images
image_embeddings = model.encode(images)
 
# Encode the captions
text_embeddings = model.encode(captions)
 
#Compute cosine similarities 
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)
print(sim_matrix)
```

## Making Text Generation Models Multimodal

Traditionally, text generation models have been, as you might expect, models that interpret textual representations. Models like Llama 2 and ChatGPT excel at reasoning about textual information and responding with natural language.

They are, however, limited to the modality they were trained in, namely text. As we have seen before with multimodal embedding models, the addition of vision can enhance the capabilities of a model.

In the case of text generation models, we would like it to reason about certain input images. For example, we could give it an image of a pizza and ask it what ingredients it contains. You could show it a picture of the Eiffel Tower and ask it when it was built or where it is located. This conversational ability is further illustrated in [Figure 4-14](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch04.html#fig\_14\_a\_multimodal\_text\_generation\_model\_that\_can\_reason).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/multimodal_large_language_models_406548_14.png" alt="A multimodal text generation model that can reason about input images." height="385" width="600"><figcaption></figcaption></figure>

**Figure 4-14. A multimodal text generation model that can reason about input images.**

To bridge the gap between these two domains, attempts have been made to introduce a form of multimodality to existing models. One such method is called BLIP-2: _Bootstrapping Language Image Pre-training for unified vision-language understanding and generation 2_. BLIP-2 introduces an easy-to-use and modular technique that allows for introducing vision capabilities to existing language models.

### BLIP-2: Bridging the Modality Gap

Creating a multimodal language model from scratch requires significant computing power and data. We would have to use billions of images, text, and image-text pairs to create such a model. As you can imagine, this is not easily feasible!

Instead of building the architecture from scratch, BLIP-2 bridges the vision-language gap by building a bridge, named the Q-former, that connects a pre-trained image encoder and a pre-trained LLM.

By leveraging pre-trained models, BLIP-2 only needs to train the bridge without needing to train the image encoder and LLM from scratch. It makes great use of the technology and models that are already out there! This bridge is illustrated in [Figure 4-15](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch04.html#fig\_15\_the\_querying\_transformer\_is\_the\_bridge\_between\_vis).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/multimodal_large_language_models_406548_15.png" alt="The Querying Transformer is the bridge between vision  ViT  and text  LLM  which is the only trainable component of the pipeline." height="176" width="600"><figcaption></figcaption></figure>

**Figure 4-15. The Querying Transformer is the bridge between vision (ViT) and text (LLM) which is the only trainable component of the pipeline.**

To connect the two pre-trained models, the Q-Former, also known as the Querying Transformer, mimics their architectures. It has two modules that share their attention layers:

* An image transformer to interact with the frozen vision transformer for feature extraction
* A text transformer that can interact with the LLM

The Q-Former is trained in two stages, one for each modality as illustrated in [Figure 4-16](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch04.html#fig\_16\_in\_step\_1\_representation\_learning\_is\_applied\_to\_l).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/multimodal_large_language_models_406548_16.png" alt="In step 1  representation learning is applied to learn representations for vision and language simultaneously. In step 2  these representations are converted to soft visual prompts to feed the LLM." height="213" width="600"><figcaption></figcaption></figure>

**Figure 4-16. In step 1, representation learning is applied to learn representations for vision and language simultaneously. In step 2, these representations are converted to soft visual prompts to feed the LLM.**

In step 1, a number of image-document pairs are used to train the Q-Former to represent both images and text. These pairs are generally captions of images, as we have seen before with training CLIP.

The images are fed to the frozen vision transformer to extract vision embeddings. These embeddings are used as the input of Q-Former’s vision transformer. The captions are used as the input of Q-Former’s text transformer.

With these inputs, the Q-Former is then trained on three tasks:

1. Image-Text Contrastive Learning
2. Image-Text Matching
3. Image-grounded Text Generation

These three objectives are jointly optimized to improve the visual representations that are extracted from the frozen vision transformer. In a way, we are trying to inject textual information into the embeddings of the frozen vision transformer so that we can use them in the LLM. This first step of BLIP-2 is illustrated in [Figure 4-17](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch04.html#fig\_17\_in\_step\_1\_the\_output\_of\_the\_frozen\_vision\_transfo).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/multimodal_large_language_models_406548_17.png" alt="In step 1  the output of the frozen vision transformer is used together with its caption and trained on three contrastive like tasks to learn visual text representations." height="283" width="600"><figcaption></figcaption></figure>

**Figure 4-17. In step 1, the output of the frozen vision transformer is used together with its caption and trained on three contrastive-like tasks to learn visual-text representations.**

In step 2, the learnable embeddings derived from step 1 now contain visual information in the same dimensional space as its corresponding textual information.

The learnable embeddings are then passed to the LLM as a soft prompt. In a way, these embeddings contain textual representations of the input image.

The learnable embeddings are then passed to the LLM. In a way, these embeddings serve as soft visual prompts that condition the LLM on the visual representations that were extracted by the Q-Former.

There is also a fully connected linear layer in between them to make sure that the learnable embeddings have the same shape as the LLM expects. This second step of converting vision to language is represented in [Figure 4-18](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch04.html#fig\_18\_in\_step\_2\_the\_learned\_embeddings\_from\_the\_q\_forme).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/multimodal_large_language_models_406548_18.png" alt="In step 2  the learned embeddings from the Q Former are passed to the LLM through a projection layer. The projected embeddings serve as a soft visual prompt." height="218" width="600"><figcaption></figcaption></figure>

**Figure 4-18. In step 2, the learned embeddings from the Q-Former are passed to the LLM through a projection layer. The projected embeddings serve as a soft visual prompt.**

When we put these steps together, they make it possible for the Q-Former to learn visual and textual representations in the same dimensional space which can be used as a soft prompt to the LLM. As a result, the LLM will be given information about the image and is similar to the context you would provide an LLM when prompting. The full in-depth process is illustrated in [Figure 4-19](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch04.html#fig\_19\_the\_full\_procedure\_of\_blip\_2).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/multimodal_large_language_models_406548_19.png" alt="The full procedure of BLIP 2." height="376" width="600"><figcaption></figcaption></figure>

**Figure 4-19. The full procedure of BLIP-2.**

### Preprocessing Multimodal Inputs

Now that we know how BLIP-2 is created, there are a number of interesting use cases for which you can use such a model. Not limited to captioning images, answering visual questions, and even performing prompting.

Before we go through some use cases, let’s first load the model and explore how you can use it:

<pre><code><strong>from transformers import AutoProcessor, Blip2ForConditionalGeneration
</strong>import torch
 
# Load processor and main model
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
 
# Send the model to GPU to speed up inference
<strong>device = "cuda" if torch.cuda.is_available() else "cpu"
</strong>model.to(device)
</code></pre>

**NOTE**

Using `model.vision_model` and `model.language_model` we can see which vision transformer and large language model are respectively used in the BLIP-2 model that we loaded.

We loaded two components that make up our full pipeline, a `processor` and a `model`. The `processor` can be compared to the tokenizer of language models. It converts unstructured input, such as images and text, to representations that the model generally expects.

#### Preprocessing Images

Let’s start by exploring what the `processor` does to images. We start by loading the picture of a very wide image for illustration purposes:

```
from urllib.request import urlopen
from PIL import Image
 
# Load a wide image
link = "https://images.unsplash.com/photo-1524602010760-6eb67b3c63a0?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2631&q=80"
image = Image.open(urlopen(link)).convert("RGB")
image
```

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/multimodal_large_language_models_406548_20.png" alt="Caption to come" height="520" width="492"><figcaption></figcaption></figure>

**Figure 4-20. Caption to come**

The image has 520 by 492 pixels which is generally an unusual format. So let’s see what our processor does to it.

```
>>> np.array(image).shape
 
(520, 492, 3)
```

When we check its shape after converting it to Numpy, it shows us an additional dimension that is of size 3. This represents the RGB coding of each pixel, namely its color.

Next, we pass the original image to the processor so that the image can be processed to the shape the model expects:

```
>>> inputs = processor(image, return_tensors="pt").to(device, torch.float16)
>>> inputs["pixel_values"].shape
 
torch.Size([1, 3, 224, 224])
```

The result is a 224 by 224 sized image. Quite a bit smaller than we initially had! This also means that all different shapes of images will be processed into squares. So be careful inputting very wide or tall images as they might get distorted.

#### Preprocessing Text

Let’s continue this exploration of the `processor` with text instead. First, we can access the tokenizer used to tokenize the input text:

<pre><code>>>> processor.tokenizer
 
<strong>GPT2TokenizerFast(name_or_path='Salesforce/blip2-opt-2.7b', vocab_size=50265, model_max_length=1000000000000000019884624838656, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'bos_token': AddedToken("&#x3C;/s>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'eos_token': AddedToken("&#x3C;/s>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'unk_token': AddedToken("&#x3C;/s>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'pad_token': AddedToken("&#x3C;pad>", rstrip=False, lstrip=False, single_word=False, normalized=True)}, clean_up_tokenization_spaces=True)
</strong></code></pre>

The BLIP-2 model that we are using uses a GPT2Tokenizer. Most tokenizers work very similarly but have slight differences in when and how they tokenize the input text.

To explore how this GPT2Tokenizer works, we can try it out with a small sentence. We start by converting the sentence to token ids before converting them back to tokens:

```
# Preprocess the text
text = "Her vocalization was remarkably melodic"
token_ids = processor(image, text=text, return_tensors="pt").to(device, torch.float16)["input_ids"][0]
 
# Convert input ids back to tokens
tokens = processor.tokenizer.convert_ids_to_tokens(token_ids)
```

When we inspect the tokens, you might notice a strange symbol at the beginning of some tokens. Namely, the Ġ symbol. This is actually supposed to be a space. However, an internal function takes characters in certain code points and moves them up by 256 to make them printable. As a result, the space (code point 32) becomes Ġ (code point 288).

We will convert them to underscores for illustrative purposes:

<pre><code><strong>>>> tokens = [token.replace("Ġ", "_") for token in tokens]
</strong>>>> tokens
 
['&#x3C;/s>', 'Her', '_vocal', 'ization', '_was', '_remarkably', '_mel', 'odic']
</code></pre>

The output shows that the underscore indicates the beginning of a word. That way, words that are made up of multiple tokens can be recognized.

### Use Case 1: Image Captioning

The most straightforward usage of a model like BLIP-2 is to create captions of images that you have in your data. You might be a store that wants to create descriptions of its clothing or perhaps you are a photographer that does not have the time to manually label its 1000+ pictures of a wedding.

The process of captioning an image closely follows the processing. An image is converted to pixel values that the model can read. These pixel values are passed to BLIP-2 to be converted into soft visual prompts that the LLM can use to decide on a proper caption.

Let’s take the image of a supercar and process it using the processor to derive pixels in the expected shape:

```
from urllib.request import urlopen
from PIL import Image
 
# Load an AI-generated image of a supercar
image = Image.open(urlopen("https://i.imgur.com/zehSvAe.png")).convert("RGB")
 
# Convert an image into inputs and preprocess it
inputs = processor(image, return_tensors="pt").to(device, torch.float16)
image
```

The next step is converting the image into token IDs using the BLIP-2 model. After doing so, we can convert the IDs into text which is the generated caption:

<pre><code># Generate token ids using the full BLIP-2 model
generated_ids = model.generate(**inputs, max_new_tokens=20)
 
# Convert the token ids to text
<strong>generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
</strong></code></pre>

When we print out the `generated_text`, we can take a look at the caption:

```
>>> print(generated_text)
 
an orange supercar driving on the road at sunset
```

“An orange supercar driving on the road at sunset” seems like a perfect description for this image!

Image captioning is a great way to get to learn this model before stepping into more complex use cases. Try it out with a few images yourself and see where it performs well and where it performs poorly.

Domain specific images, like pictures of specific cartoon characters or imaginary creations may fail as the model was trained on largely public data.

Let’s end this use case with a fun example, namely an image of the Rorschach which is illustrated in [Figure 4-21](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch04.html#fig\_21\_an\_image\_from\_the\_rorschach\_test\_what\_do\_you\_see). This test is an old psychological test which tests the individual’s perception of inkblots.[4](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch04.html#id212) What someone sees in such an inkblot supposedly tells you something about a person’s personality characteristics.

It is quite a subjective test but that just makes it more fun!

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/Rorschach.png" alt="An image from the Rorschach test. What do you see in it" height="396" width="600"><figcaption></figcaption></figure>

**Figure 4-21. An image from the Rorschach test. What do you see in it?**

Let’s take the image illustrated in Figure 7-X and use that as our input:

<pre><code># Load rorschach image
url = "https://upload.wikimedia.org/wikipedia/commons/7/70/Rorschach_blot_01.jpg"
image = Image.open(urlopen(url)).convert("RGB")
 
# Generate caption
inputs = processor(image, return_tensors="pt").to(device, torch.float16)
generated_ids = model.generate(**inputs, max_new_tokens=20)
<strong>generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
</strong></code></pre>

As before, when we print out the `generated_text`, we can take a look at the caption:

```
>>> print(generated_text)
 
"a black and white ink drawing of a bat"
```

“A black and white ink drawing of a bat”. I can definitely see how the model would caption this image using such a description. Since this is a Rorscharch test, what do you think it says about the model?

### Use Case 2: Multimodal Chat-based Prompting

Although captioning is an important task, we can extend its use case even further. In that example, we showed going from one modality, vision (image), to another, text (caption).

Instead of following this linear structure, we can try to present both modalities simultaneously by performing what is called visual question answering. In this particular use case, we give the model an image along with a question about that specific image for it to answer. The model would need to process both the image as well as the question as once.

To demonstrate, let’s start with the picture of a car and ask BLIP-2 to describe the image. To do so, we first need to preprocess the image as we did a few times before:

```
# Load an AI-generated image of a supercar and process it
image = Image.open(urlopen("https://i.imgur.com/zehSvAe.png")).convert("RGB")
inputs = processor(image, return_tensors="pt").to(device, torch.float16)
```

To perform our visual question answering we need to give BLIP-2 more than just the image, namely the prompt. Without it the model would generate a caption as it did before.

We will ask the model to describe the image we just processed:

<pre><code># Visual Question Answering
prompt = "Question: Write down what you see in this picture. Answer:"
 
# Process both the image and the prompt
inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)
 
# Generate text
generated_ids = model.generate(**inputs, max_new_tokens=30)
<strong>generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
</strong></code></pre>

When we print out the `generated_text`, we can explore the answer it has given to the question we asked it:

```
>>> print(generated_text)
 
A sports car driving on the road at sunset
```

It correctly describes the image. However, this is a rather simple example since our question is essentially asking the model to create a caption. Instead, we can ask it follow-up questions in a chat-based manner.

To do so, we can give the model our previous conversation, including its answer to our question. We then ask it a follow-up question.

<pre><code>>>> # Chat-like prompting
>>> prompt = "Question: Write down what you see in this picture. Answer: A sports car driving on the road >>> at sunset. Question: What would it cost me to drive that car? Answer:"
>>> 
>>> # Generate output
>>> inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)
>>> generated_ids = model.generate(**inputs, max_new_tokens=30)
<strong>>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
</strong>>>> print(generated_text)
 
$1,000,000
</code></pre>

$1,000,000 is highly specific! This shows a more chat-like behavior from BLIP-2 which allows for some interesting conversations.

Finally, we can make this process a bit smoother by creating an interactive chat-bot using ipywidgets, an extension for Jupyter Notebooks that allows us to make interactive buttons, input text, etc.

<pre><code>from IPython.display import HTML, display
import ipywidgets as widgets
 
<strong>def text_eventhandler(*args):
</strong>  question = args[0]["new"]
<strong>  if question:
</strong>    args[0]["owner"].value = ""
 
    # Create prompt
<strong>    if not memory:
</strong>      prompt = " Question: " + question + " Answer:"
<strong>    else:
</strong><strong>      template = "Question: {} Answer: {}."
</strong><strong>      prompt = " ".join([template.format(memory[i][0], memory[i][1]) for i in range(len(memory))]) + " Question: " + question + " Answer:"
</strong>    
    # Generate text
    inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=100)
<strong>    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip().split("Question")[0]
</strong> 
    # Update memory
    memory.append((question, generated_text))
    
    # Assign to output
    output.append_display_data(HTML("&#x3C;b>USER:&#x3C;/b> " + question))
    output.append_display_data(HTML("&#x3C;b>BLIP-2:&#x3C;/b> " + generated_text))
    output.append_display_data(HTML("&#x3C;br>"))
 
# Prepare widgets
in_text = widgets.Text()
<strong>in_text.continuous_update = False
</strong>in_text.observe(text_eventhandler, "value")
output = widgets.Output()
memory = []
 
# Display chat box
display(
    widgets.VBox(
        children=[output, in_text],
        layout=widgets.Layout(display="inline-flex", flex_flow="column-reverse"),
    )
)
</code></pre>

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098150952/files/assets/multimodal_large_language_models_406548_23.png" alt="Figure Caption to come" height="306" width="600"><figcaption></figcaption></figure>

**Figure 4-22. Figure Caption to come**

It seems that we can continue the conversation and ask it a bunch of questions. Using this chat-based approach, we essentially created a chatbot that can reason about images!

## Summary

In this chapter, we explored two methods making language models multimodal.

[1](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch04.html#id209-marker) Wei, J., Tay, Y., Bommasani, R., Raffel, C., Zoph, B., Borgeaud, S., Yogatama, D., Bosma, M., Zhou, D., Metzler, D., & others (2022). Emergent abilities of large language models. arXiv preprint arXiv:2206.07682.\


[2](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch04.html#id210-marker) Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., & others (2020). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.\


[3](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch04.html#id211-marker) Radford, A., Kim, J., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., & others (2021). Learning transferable visual models from natural language supervision. In _International conference on machine learning_ (pp. 8748–8763).\


[4](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch04.html#id212-marker) Schafer, R. (1954). Psychoanalytic interpretation in Rorschach testing: theory and application.
