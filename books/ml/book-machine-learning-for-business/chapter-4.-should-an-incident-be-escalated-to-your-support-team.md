# Chapter 4. Should an incident be escalated to your support team?

_This chapter covers_

* An overview of natural language processing
* (NLP)
* How to approach an NLP machine learning scenario
* How to prepare data for an NLP scenario
* SageMaker’s text analytics engine, BlazingText
* How to interpret BlazingText results

Naomi heads up an IT team that handles customer support tickets for a number of companies. A customer sends a tweet to a Twitter account, and Naomi’s team replies with a resolution or a request for further information. A large percentage of the tweets can be handled by sending links to information that helps customers resolve their issues. But about a quarter of the responses are to people who need more help than that. They need to feel they’ve been heard and tend to get very cranky if they don’t. These are the customers that, with the right intervention, become the strongest advocates; with the wrong intervention, they become the loudest detractors. Naomi wants to know who these customers are as early as possible so that her support team can intervene in the right way.

She and her team have spent the past few years automating responses to the most common queries and manually escalating the queries that must be handled by a person. Naomi wants to build a triage system that reviews each request as it comes in to determine whether the response should be automatic or should be handed off to a person.

Fortunately for Naomi, she has a couple of years of historical tweets that her team has reviewed and decided whether they can be handled automatically or should be handled by a person. In this chapter, you’ll take Naomi’s historical data and use it to decide whether new tweets should be handled automatically or escalated to one of Naomi’s team members.

#### 4.1. What are you making decisions about? <a href="#ch04lev1sec1__title" id="ch04lev1sec1__title"></a>

As always, the first thing you want to look at is what you’re making decisions about. In this chapter, the decision Naomi is making is, _should this tweet be escalated to a person?_

The approach that Naomi’s team has taken over the past few years is to escalate tweets in which the customer appears frustrated. Her team did not apply any hard and fast rules when making this decision. They just had a feeling that the customer was frustrated so they escalated the tweet. In this chapter, you are going to build a machine learning model that learns how to identify frustration based on the tweets that Naomi’s team has previously escalated.

#### 4.2. The process flow <a href="#ch04lev1sec2__title" id="ch04lev1sec2__title"></a>

The process flow for this decision is shown in [figure 4.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04fig01). It starts when a customer sends a tweet to the company’s support account. Naomi’s team reviews the tweet to determine whether they need to respond personally or whether it can be handled by their bot. The final step is a tweet response from either the bot or Naomi’s team.

**Figure 4.1. Tweet response workflow for Naomi’s customer support tickets**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch04fig01\_alt.jpg)

Naomi wants to replace the determination at step 1 of [figure 4.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04fig01) with a machine learning application that can make a decision based on how frustrated the incoming tweet seems. This chapter shows you how to prepare this application.

#### 4.3. Preparing the dataset <a href="#ch04lev1sec3__title" id="ch04lev1sec3__title"></a>

In the previous two chapters, you prepared a synthetic dataset from scratch. For this chapter, you are going to take a dataset of tweets sent to tech companies in 2017. The dataset is published by a company called Kaggle, which runs machine learning competitions.

Kaggle, competition, and public datasets

Kaggle is a fascinating company. Founded in 2010, Kaggle gamifies machine learning by pitting teams of data scientists against each other to solve machine learning problems for prize money. In mid-2017, shortly before being acquired by Google, Kaggle announced it had reached a milestone of one million registered competitors.

Even if you have no intention of competing in data science competitions, Kaggle is a good resource to become familiar with because it has public datasets that you can use in your machine learning training and work.

To determine what data is required to solve a particular problem, you need to focus on the objective you are pursuing and, in this case, think about the minimum data Naomi needs to achieve her objective. Once you have that, you can decide whether you can achieve her objective using only that data supplied, or you need to expand the data to allow Naomi to better achieve her objective.

As a reminder, Naomi’s objective is to identify the tweets that should be handled by a person, based on her team’s past history of escalating tweets. So Naomi’s dataset should contain an incoming tweet and a flag indicating whether it was escalated or not.

The dataset we use in this chapter is based on a dataset uploaded to Kaggle by Stuart Axelbrooke from Thought Vector. (The original dataset can be viewed at [https://www.kaggle.com/thoughtvector/customer-support-on-twitter/](https://www.kaggle.com/thoughtvector/customer-support-on-twitter/).) This dataset contains over 3 million tweets sent to customer support departments for several companies ranging from Apple and Amazon to British Airways and Southwest Air.

Like every dataset you’ll find in your company, you can’t just use this data as is. It needs to be formatted in a way that allows your machine learning algorithm do its thing. The original dataset on Kaggle contains both the original tweet and the response. In the scenario in this chapter, only the original tweet is relevant. To prepare the data for this chapter, we removed all the tweets except the original tweet and used the responses to label the original tweet as escalated or not escalated. The resulting dataset contains a tweet with that label and these columns:

* _tweet\_id_—Uniquely identifies the tweet
* _author\_id_—Uniquely identifies the author
* _created\_at_—Shows the time of the tweet
* _in\_reply\_to_—Shows which company is being contacted
* _text_—Contains the text in the tweet
* _escalate_—Indicates whether the tweet was escalated or not

[Table 4.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04table01) shows the first three tweets in the dataset. Each of the tweets is to Sprint Care, the support team for the US phone company, Sprint. You can see that the first tweet (“and how do you propose we do that”) was not escalated by Naomi’s team. But the second tweet (“I have sent several private messages and no one is responding as usual”) was escalated. Naomi’s team let their automated response system handle the first tweet but escalated the second tweet to a member of her team for a personal response.

**Table 4.1. Tweet dataset**

| tweet\_id | author\_id | created\_at           | in\_reply\_to | text                                                                               | escalate |
| --------- | ---------- | --------------------- | ------------- | ---------------------------------------------------------------------------------- | -------- |
| 2         | 115712     | Tue Oct 31 22:11 2017 | sprintcare    | @sprintcare and how do you propose we do that                                      | False    |
| 3         | 115712     | Tue Oct 31 22:08 2017 | sprintcare    | @sprintcare I have sent several private messages and no one is responding as usual | True     |
| 5         | 115712     | Tue Oct 31 21:49 2017 | sprintcare    | @sprintcare I did.                                                                 | False    |

In this chapter, you’ll build a machine learning application to handle the task of whether to escalate the tweet. But this application will be a little different than the machine learning applications you built in previous chapters. In order to decide which tweets should be escalated, the machine learning application needs to know something about language and meaning, which you might think is pretty difficult to do. Fortunately, some very smart people have been working on this problem for a while. They call it _natural language processing_, or NLP.

#### 4.4. NLP (natural language processing) <a href="#ch04lev1sec4__title" id="ch04lev1sec4__title"></a>

The goal of NLP is to be able to use computers to work with language as effectively as computers can work with numbers or variables. This is a hard problem because of the richness of language. (The previous sentence is a good example of the difficulty of this problem.) The term _rich_ means something slightly different when referring to language than it does when referring to a person. And the sentence “Well, that’s rich!” can mean the opposite of how rich is used in other contexts.

Scientists have worked on NLP since the advent of computing, but it has only been recently that they have made significant strides in this area. NLP originally focused on getting computers to understand the structure of each language. In English, a typical sentence has a subject, verb, and an object, such as this sentence: “Sam throws the ball”; whereas in Japanese, a sentence typically follows a subject, object, verb pattern. But the success of this approach was hampered by the mind-boggling number and variety of exceptions and slowed by the necessity to individually describe each different language. The same code you use for English NLP won’t work for Japanese NLP.

The big breakthrough in NLP occurred in 2013 when NIPS published a paper on word vectors.\[[1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04fn01)] With this approach, you don’t look at parts of the language at all! You just apply a mathematical algorithm to a bunch of text and work with the output of the algorithm. This has two advantages:

> 1
>
> See “Distributed Representations of Words and Phrases and their Compositionality” by Tomas Mikolov et al. at [https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf).

* It naturally handles exceptions and inconsistencies in a language.
* It is language-agnostic and can work with Japanese text as easily as it can work with English text.

In SageMaker, working with word vectors is as easy as working with the data you worked with in [chapters 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02) and [3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03). But there are a few decisions you need to make when configuring SageMaker that require you to have some appreciation of what is happening under the hood.

**4.4.1. Creating word vectors**

Just as you used the pandas function get\_dummies in [chapter 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02) to convert categorical data (such as desk, keyboard, and mouse) to a wide dataset, the first step in creating a word vector is to convert all the words in your text into wide datasets. As an example, the word _queen_ is represented by the dataset 0,1,0,0,0, as shown in [figure 4.2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04fig02). The word _queen_ has 1 under it, while every other word in the row has a 0. This can be described as a _single dimensional vector_.

Using a single dimensional vector, you test for equality and nothing else. That is, you can determine whether the vector is equal to the word _queen_, and in [figure 4.2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04fig02), you can see that it is.

**Figure 4.2. Single dimensional vector testing for equality**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch04fig02.jpg)

Mikolov’s breakthrough was the realization that _meaning_ can be captured by a _multidimensional_ vector with the representation of each word distributed across each dimension. [Figure 4.3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04fig03) shows conceptually how dimensions look in a vector. Each dimension can be thought of as a group of related words. In Mikolov’s algorithm, these groups of related words don’t have labels, but to show how meaning can emerge from multidimensional vectors, we have provided four labels on the left side of the figure: Royalty, Masculinity, Femininity, and Elderliness.

**Figure 4.3. Multidimensional vectors capture meaning in languages.**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch04fig03.jpg)

Looking at the first dimension in [figure 4.3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04fig03), Royalty, you can see that the values in the King, Queen, and Princess columns are higher than the values in the Man and Woman columns; whereas, for Masculinity, the values in the King and Man columns are higher than in the others. From this you start to get the picture that a King is Masculine Royalty whereas a Queen is non-masculine Royalty. If you can imagine working your way through hundreds of vectors, you can see how meaning emerges.

Back to Naomi’s problem, as each tweet comes in, the application breaks the tweet down into multidimensional vectors and compares it to the tweets labeled by Naomi’s team. The application identifies which tweets in the training dataset have similar vectors. It then looks at the label of the trained tweets and assigns that label to the incoming tweet. For example, if an incoming tweet has the phrase “no one is responding as usual,” the tweets in the training data with similar vectors would likely have been escalated, and so the incoming tweet would be escalated as well.

The magic of the mathematics behind word vectors is that it groups the words being defined. Each of these groups is a dimension in the vector. For example, in the tweet where the tweeter says “no one is responding as usual,” the words _as usual_ might be grouped into a dimension with other pairs of words such as _of course_, _yeah obviously_, and _a doy_, which indicate frustration.

The King/Queen, Man/Woman example is used regularly in the explanation of word vectors. Adrian Colyer’s excellent blog, “the morning paper,” discusses word vectors in more detail at [https://blog.acolyer.org/?s=the+amazing+power+of+word+vectors](https://blog.acolyer.org/?s=the+amazing+power+of+word+vectors). [Figures 4.2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04fig02) and [4.3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04fig03) are based on figures from the first part of this article. If you are interested in exploring this topic further, the rest of Adrian’s article is a good place to start.

**4.4.2. Deciding how many words to include in each group**

In order to work with vectors in SageMaker, the only decision you need to make is whether SageMaker should use single words, pairs of words, or word triplets when creating the groups. For example, if SageMaker uses the word pair _as usual_, it can get better results than if it uses the single word _as_ and the single word _usual_, because the word pair expresses a different concept than do the individual words.

In our work, we normally use word pairs but have occasionally gotten better results from triplets. In one project where we were extracting and categorizing marketing terms, using triplets resulted in much higher accuracy, probably because marketing fluff is often expressed in word triplets such as _world class results_, _high powered engine_, and _fat burning machine_.

NLP uses the terms unigram, bigram, and trigram for single-, double-, and triple-word groups. [Figures 4.4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04fig04), [4.5](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04fig05), and [4.6](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04fig06) show examples of single-word (unigram), double-word (bigram), and triple-word (trigram) word groups, respectively.

As [figure 4.4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04fig04) shows, _unigrams_ are single words. Unigrams work well when word order is not important. For example, if you were creating word vectors for medical research, unigrams do a good job of identifying similar concepts.

**Figure 4.4. NLP defines single words as unigrams.**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch04fig04\_alt.jpg)

As [figure 4.5](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04fig05) shows, _bigrams_ are pairs of words. Bigrams work well when word order is important, such as in sentiment analysis. The bigram _as usual_ conveys frustration, but the unigrams _as_ and _usual_ do not.

**Figure 4.5. Pairs of words are known as bigrams.**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch04fig05.jpg)

As [figure 4.6](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04fig06) shows, _trigrams_ are groups of three words. In practice, we don’t see much improvement in moving from bigrams to trigrams, but on occasion there can be. One project we worked on to identify marketing terms delivered significantly better results using trigrams, probably because the trigrams better captured the common pattern _hyperbole noun noun_ (as in _greatest coffee maker_) and the pattern _hyperbole adjective noun_ (as in _fastest diesel car_).

**Figure 4.6. Words grouped into threes are trigrams.**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch04fig06.jpg)

In our case study, the machine learning application will use an algorithm called _BlazingText_. This predicts whether a tweet should be escalated.

#### 4.5. What is BlazingText and how does it work? <a href="#ch04lev1sec5__title" id="ch04lev1sec5__title"></a>

BlazingText is a version of an algorithm, called fastText, developed by researchers at Facebook in 2017. And fastText is a version of the algorithm developed by Google’s own Mikolov and others. [Figure 4.7](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04fig07) shows the workflow after BlazingText is put into use. In step 1, a tweet is sent by a person requiring support. In step 2, BlazingText decides whether the tweet should be escalated to a person for a response. In step 3, the tweet is escalated to a person (step 3a) or handled by a bot (step 3b).

**Figure 4.7. The BlazingText workflow to determine whether a tweet should be escalated**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch04fig07\_alt.jpg)

In order for BlazingText to decide whether a tweet should be escalated, it needs to determine whether the person sending the tweet is feeling frustrated or not. To do this, BlazingText doesn’t actually need to know whether the person is feeling frustrated or even understand what the tweet was about. It just needs to determine how similar the tweet is to other tweets that have been labeled as frustrated or not frustrated. With that as background, you are ready to start building the model. If you like, you can read more about BlazingText on Amazon’s site at [https://docs.aws.amazon.com/sagemaker/latest/dg/blazingtext.html](https://docs.aws.amazon.com/sagemaker/latest/dg/blazingtext.html).

Refresher on how SageMaker is structured

Now that you’re getting comfortable with using Jupyter Notebook, it is a good time to review how SageMaker is structured. When you first set up SageMaker, you created a notebook instance, which is a server that AWS configures to run your notebooks. In [appendix C](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_028.html#app03), we instructed you to select a medium-sized server instance because it has enough grunt to do anything we cover in this book. As you work with larger datasets in your own work, you might need to use a larger server.

When you run your notebook for the case studies in this book, AWS creates two additional servers. The first is a temporary server that is used to train the machine learning model. The second server AWS creates is the endpoint server. This server stays up until you remove the endpoint. To delete the endpoint in SageMaker, click the radio button to the left of the endpoint name, then click the Actions menu item, and click Delete in the menu that appears.

#### 4.6. Getting ready to build the model <a href="#ch04lev1sec6__title" id="ch04lev1sec6__title"></a>

Now that you have a deeper understanding of how BlazingText works, you’ll set up another notebook in SageMaker and make some decisions. You are going to do the following (as you did in [chapters 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02) and [3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03)):

1. Upload a dataset to S3
2. Set up a notebook on SageMaker
3. Upload the starting notebook
4. Run it against the data

**Tip**

If you’re jumping into the book at this chapter, you might want to visit the appendixes, which show you how to do the following:

* [Appendix A](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_021.html#app01): sign up for AWS, Amazon’s web service
* [appendix B](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_024.html#app02): set up S3, AWS’s file storage service
* [Appendix C](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_028.html#app03): set up SageMaker

**4.6.1. Uploading a dataset to S3**

To set up your dataset for this chapter, you’ll follow the same steps as you did in [appendix B](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_024.html#app02). You don’t need to set up another bucket though. You can just go to the same bucket you created earlier. In our example, we called the bucket _mlforbusiness_, but your bucket will be called something different. When you go to your S3 account, you will see something like that shown in [figure 4.8](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04fig08).

**Figure 4.8. Viewing the list of buckets**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch04fig08\_alt.jpg)

Click this bucket to see the ch02 and ch03 folders you created in the previous chapters. For this chapter, you’ll create a new folder called _ch04_. You do this by clicking Create Folder and following the prompts to create a new folder.

Once you’ve created the folder, you are returned to the folder list inside your bucket. There you’ll see you now have a folder called ch04.

Now that you have the ch04 folder set up in your bucket, you can upload your data file and start setting up the decision-making model in SageMaker. To do so, click the folder and download the data file at this link:

[https://s3.amazonaws.com/mlforbusiness/ch04/inbound.csv](https://s3.amazonaws.com/mlforbusiness/ch04/inbound.csv).

Then upload the CSV file into the ch04 folder by clicking Upload. Now you’re ready to set up the notebook instance.

**4.6.2. Setting up a notebook on SageMaker**

Like you did for [chapters 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02) and [3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03), you’ll set up a notebook on SageMaker. If you skipped [chapters 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02) and [3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03), follow the instructions in [appendix C](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_028.html#app03) on how to set up SageMaker.

When you go to SageMaker, you’ll see your notebook instances. The notebook instance you created for [chapters 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02) and [3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03) (or that you’ve just created by following the instructions in [appendix C](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_028.html#app03)) will either say Open or Start. If it says Start, click the Start link and wait a couple of minutes for SageMaker to start. Once it displays Open Jupyter, select that link to open your notebook list.

Once it opens, create a new folder for [chapter 4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04) by clicking New and selecting Folder at the bottom of the dropdown list. This creates a new folder called Untitled Folder. To rename the folder, tick the checkbox next to Untitled Folder, and you will see the Rename button appear. Click Rename and change the name to ch04. Click the ch04 folder, and you will see an empty notebook list.

Just as we already prepared the CSV data you uploaded to S3, we’ve already prepared the Jupyter notebook you’ll now use. You can download it to your computer by navigating to this URL:

[https://s3.amazonaws.com/mlforbusiness/ch04/customer\_support.ipynb](https://s3.amazonaws.com/mlforbusiness/ch04/customer\_support.ipynb).

Click Upload to upload the customer\_support.ipynb notebook to the folder. After uploading the file, you’ll see the notebook in your list. Click it to open it. Now, just like in [chapters 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02) and [3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03), you are a few keystrokes away from being able to run your machine learning model.

#### 4.7. Building the model <a href="#ch04lev1sec7__title" id="ch04lev1sec7__title"></a>

As in [chapters 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02) and [3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03), you will go through the code in six parts:

* Load and examine the data.
* Get the data into the right shape.
* Create training and validation datasets (there’s no need for a test dataset in this example).
* Train the machine learning model.
* Host the machine learning model.
* Test the model and use it to make decisions

Refresher on running code in Jupyter notebooks

SageMaker uses Jupyter Notebook as its interface. Jupyter Notebook is an open-source data science application that allows you to mix code with text. As shown in the figure, the code sections of a Jupyter notebook have a grey background, and the text sections have a white background.

**Sample notebook showing text and code cells**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/pg87fig01\_alt.jpg)

To run the code in the notebook, click into a code cell and press ![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ctrl\_ent.jpg). Alternatively, you can select Run All from the Cell menu item at the top of the notebook.

**4.7.1. Part 1: Loading and examining the data**

As in the previous two chapters, the first step is to say where you are storing the data. To do that, you need to change 'mlforbusiness' to the name of the bucket you created when you uploaded the data and rename its subfolder to the name of the subfolder on S3 where you store the data ([listing 4.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04ex1)).

If you named the S3 folder ch04, then you don’t need to change the name of the folder. If you kept the name of the CSV file that you uploaded earlier in the chapter, then you don’t need to change the inbound.csv line of code. If you changed the name of the CSV file, then update inbound.csv to the name you changed it to. And, as always, to run the code in the notebook cell, click the cell and press ![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ctrl\_ent.jpg).

**Listing 4.1. Say where you are storing the data**

```
data_bucket = 'mlforbusiness'    1
subfolder = 'ch04'               2
dataset = 'inbound.csv'          3
```

* _1_ S3 bucket where the data is stored
* _2_ Subfolder of the S3 bucket where the data is stored
* _3_ Dataset that’s used to train and test the model

The Python modules and libraries imported in [listing 4.2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04ex2) are the same as the import code in [chapters 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02) and [3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03), with the exception of lines 6, 7, and 8. These lines import Python’s json module. This module is used to work with data structured in _JSON_ format (a structured mark-up language for describing data). Lines 6 and 7 import Python’s json and csv modules. These two formats define the data.

The next new library you import is NLTK ([https://www.nltk.org/](https://www.nltk.org/)). This is a commonly used library for getting text ready to use in a machine learning model. In this chapter, you will use NLTK to _tokenize_ words. Tokenizing text involves splitting the text and stripping out those things that make it harder for the machine learning model to do what it needs to do.

In this chapter, you use the standard word\_tokenize function that splits text into words in a way that consistently handles abbreviations and other anomalies. BlazingText often works better when you don’t spend a lot of time preprocessing the text, so this is all you’ll do to prepare each tweet (in addition to applying the labeling, of course, which you’ll do in [listing 4.8](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04ex8)). To run the code, click in the notebook cell and press ![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ctrl\_ent.jpg).

**Listing 4.2. Importing the modules**

```
import pandas as pd                       1
import boto3                              2
import sagemaker                          3
import s3fs                               4
from sklearn.model_selection \
  import train_test_split                 5
import json                               6
import csv                                7
import nltk                               8

role = sagemaker.get_execution_role()     9
s3 = s3fs.S3FileSystem(anon=False)        10
```

* _1_ Imports the pandas Python library
* _2_ Imports the boto3 AWS library
* _3_ Imports SageMaker
* _4_ Imports the s3fs module to make working with S3 files easier
* _5_ Imports only the sklearn train\_test\_split module
* _6_ Imports Python’s json module for working with JSON files
* _7_ Imports the csv module to work with CSV files
* _8_ Imports NLTK to tokenize tweets
* _9_ Creates a role in SageMaker
* _10_ Establishes the connection with S3

You’ve worked with CSV files throughout the book. JSON is a type of structured markup language similar to XML but simpler to work with. The following listing shows an example of an invoice described in JSON format.

**Listing 4.3. Sample JSON format**

```
{
  "Invoice": {
    "Header": {
      "Invoice Number": "INV1234833",
      "Invoice Date": "2018-11-01"
    },
    "Lines": [
      {
        "Description": "Punnet of strawberries",
        "Qty": 6,
        "Unit Price": 3
      },
      {
        "Description": "Punnet of blueberries",
        "Qty": 6,
        "Unit Price": 4
      }
    ]
  }
}
```

Next, you’ll load and view the data. The dataset you are loading has a half-million rows but loads in only a few seconds, even on the medium-sized server we are using in our SageMaker instance. To time and display how long the code in a cell takes to run, you can include the line %%time in the cell, as shown in the following listing.

**Listing 4.4. Loading and viewing the data**

```
%%time                                              1
df = pd.read_csv(
    f's3://{data_bucket}/{subfolder}/{dataset}')    2
display(df.head())                                  3
```

* _1_ Displays how long it takes to run the code in the cell
* _2_ Reads the S3 inbound.csv dataset in [listing 4.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04ex1)
* _3_ Displays the top five rows of the DataFrame

[Table 4.2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04table02) shows the output of running display(df.head()). Note that using the .head() function on your DataFrame displays only the top five rows.

**Table 4.2. The top five rows in the tweet dataset**

| row\_id | tweet\_id | author\_id | created\_at                    | in\_reply\_to | text                                             | escalate |
| ------- | --------- | ---------- | ------------------------------ | ------------- | ------------------------------------------------ | -------- |
| 0       | 2         | 115712     | Tue Oct 31 22:11 2017          | sprintcare    | @sprintcare and how do you propose we do that    | False    |
| 1       | 3         | 115712     | Tue Oct 31 22:08 2017          | sprintcare    | @sprintcare I have sent several private messag…  | True     |
| 2       | 5         | 115712     | Tue Oct 31 21:49 2017          | sprintcare    | @sprintcare I did.                               | False    |
| 3       | 16        | 115713     | Tue Oct 31 20:00:43 +0000 2017 | sprintcare    | @sprintcare Since I signed up with you….Sinc…    | False    |
| 4       | 22        | 115716     | Tue Oct 31 22:16:48 +0000 2017 | Ask\_Spectrum | @Ask\_Spectrum Would you like me to email you a… | False    |

You can see from the first five tweets that only one was escalated. At this point, we don’t know if that is expected or unexpected. [Listing 4.5](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04ex5) shows how many rows are in the dataset and how many were escalated or not. To get this information, you run the pandas shape and value\_counts functions.

**Listing 4.5. Showing the number of escalated tweets in the dataset**

```
print(f'Number of rows in dataset: {df.shape[0]}')   1
print(df['escalated'].value_counts())                2
```

* _1_ Displays the number of rows in the dataset
* _2_ Displays the number of rows where a tweet was escalated and the number of rows where the tweet was not escalated

The next listing shows the output from the code in [listing 4.5](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04ex5).

**Listing 4.6. Total number of tweets and the number of escalated tweets**

```
Number of rows in dataset: 520793
False    417800
True     102993
Name: escalate, dtype: int64
```

Out of the dataset of more than 500,000 tweets, just over 100,000 were manually escalated. If Naomi can have a machine learning algorithm read and escalate tweets, then her team will only have to read 20% of the tweets they currently review.

**4.7.2. Part 2: Getting the data into the right shape**

Now that you can see your dataset in the notebook, you can start working with it. First, you create train and validation data for the machine learning model. As in the previous two chapters, you use sklearn’s train\_test\_split function to create the datasets. With BlazingText, you can see the accuracy of the model in the logs as it is validating the model, so there is no need to create a test dataset.

**Listing 4.7. Creating train and validate datasets**

```
train_df, val_df, _, _ = train_test_split(
    df,
    df['escalate'],
    test_size=0.2,
    random_state=0)                                    1
print(f'{train_df.shape[0]} rows in training data')    2
print(f'{val_df.shape[0]} rows in validation data')    3
```

* _1_ Creates the train and validation datasets
* _2_ Displays the number of rows in the training data
* _3_ Displays the number of rows in the validation data

Unlike the XGBoost algorithm that we worked with in previous chapters, BlazingText _cannot_ work directly with CSV data. It needs the data in a different format, which you will do in [listings 4.8](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04ex8) through [4.10](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04ex10).

Formatting data for BlazingText

BlazingText requires a label in the format \_\_label\_\_0 for a tweet that was not escalated and \_\_label\_\_1 for a tweet that was escalated. The label is then followed by the tokenized text of the tweet. _Tokenizing_ is the process of taking text and breaking it into parts that are linguistically meaningful. This is a difficult task to perform but, fortunately for you, the hard work is handled by the NLTK library.

[Listing 4.8](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04ex8) defines two functions. The first function, preprocess, takes a DataFrame containing either the validation or training datasets you created in [listing 4.7](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04ex7), turns it into a list, and then for each row in the list, calls the second function transform\_ instance to convert the row to the format \_\_label\_\_0 or \_\_label\_\_1, followed by the text of the tweet. To run the preprocess function on the validation data, you call the function on the val\_df DataFrame you created in [listing 4.8](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04ex8).

You’ll run this code first on the validation dataset and then on the training dataset. The validation dataset has 100,000 rows, and this cell will take about 30 seconds to run on that data. The training dataset has 400,000 rows and will take about 2 minutes to run. Most of the time is spent converting the dataset to a DataFrame and back again. This is fine for a dataset of a half-million rows. If you are working with a dataset with millions of rows, you’ll need to start working directly with the csv module rather than using pandas. To learn more about the cvs module, visit [https://docs.python.org/3/library/csv.html](https://docs.python.org/3/library/csv.html).

**Listing 4.8. Transforming each row to the format used by BlazingText**

```
def preprocess(df):
    all_rows = df.values.tolist()                       1
    transformed_rows = list(
      map(transform_instance, all_rows))                2
    transformed_df = pd.DataFrame(transformed_rows)         3
    return transformed_df                               4

def transform_instance(row):
    cur_row = []                                        5
    label = '__label__1' if row[5] == True \
        else '__label__0'                               6
    cur_row.append(label)                               7
    cur_row.extend(
        nltk.word_tokenize(row[4].lower()))             8
    return cur_row                                      9

transformed_validation_rows = preprocess(val_df)        10
display(transformed_validation_rows.head())             11
```

* _1_ Turns the DataFrame into a list
* _2_ Applies transform\_instance to every row in the list
* _3_ Turns it back into a DataFrame
* _4_ Returns the DataFrame
* _5_ Creates an empty list that holds the label followed by each of the words in the tweet
* _6_ Creates a label with the value of 1 if escalated or 0 if not
* _7_ Sets the first element of the cur\_row list as the label
* _8_ Sets each of the words as a separate element in the list
* _9_ Returns the row
* _10_ Runs the preprocess function
* _11_ Displays the first five rows of data

The data shown in [table 4.3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04table03) shows the first few rows of data in the format BlazingText requires. You can see that the first two tweets were labeled 1 (escalate), and the third row is labeled 0 (don’t escalate).

**Table 4.3. Validation data for Naomi’s tweets**

| Labeled preprocessed data                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| \_\_label\_\_1 @ 115990 no joke … this is one of the worst customer experiences i have had verizon . maybe time for @ 115714 @ 115911 @ att? [https://t.co/vqmlkvvwxe](https://t.co/vqmlkvvwxe) \_\_label\_\_1 @ amazonhelp neither man seems to know how to deliver a package . that is their entire job ! both should lose their jobs immediately. \_\_label\_\_0 @ xboxsupport yes i see nothing about resolutions or what size videos is exported only quality i have a 34 '' ultrawide monitor 21:9 2560x1080 what i need [https://t.co/apvwd1dlq8](https://t.co/apvwd1dlq8) |

Now that you have the text in the format BlazingText can work with, and that text is sitting in a DataFrame, you can use the pandas to\_csv to store the data on S3 so you can load it into the BlazingText algorithm. The code in the following listing writes out the validation data to S3.

**Listing 4.9. Transforming the data for BlazingText**

```
s3_validation_data = f's3://{data_bucket}/\
{subfolder}/processed/validation.csv'

data = transformed_validation_rows.to_csv(
        header=False,
        index=False,
        quoting=csv.QUOTE_NONE,
        sep='|',
        escapechar='^').encode()
with s3.open(s3_validation_data, 'wb') as f:
    f.write(data)>
```

Next, you’ll preprocess the training data by calling the preprocess function on the train\_df DataFrame you created in [listing 4.7](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04ex7).

**Listing 4.10. Preprocessing and writing training data**

```
%%time
transformed_train_rows = preprocess(train_df)
display(transformed_train_rows.head())

s3_train_data = f's3://{data_bucket}/{subfolder}/processed/train.csv'

data = transformed_train_rows.to_csv(
        header=False,
        index=False,
        quoting=csv.QUOTE_NONE,
        sep='|',
        escapechar='^').encode()
with s3.open(s3_train_data, 'wb') as f:
    f.write(data)>
```

With that, the train and test datasets are saved to S3 in a format ready for use in the model. The next section takes you though the process of getting the data into SageMaker so it’s ready to kick off the training process.

**4.7.3. Part 3: Creating training and validation datasets**

Now that you have your data in a format that BlazingText can work with, you can create the training and validation datasets.

**Listing 4.11. Creating the training, validation, and test datasets**

```
%%time

train_data = sagemaker.session.s3_input(
    s3_train_data,
    distribution='FullyReplicated',
    content_type='text/plain',
    s3_data_type='S3Prefix')                   1
validation_data = sagemaker.session.s3_input(
    s3_validation_data,
    distribution='FullyReplicated',
    content_type='text/plain',
    s3_data_type='S3Prefix')                   2
```

* _1_ Creates the train\_data dataset
* _2_ Creates the validation\_data dataset

With that, the data is in a SageMaker session, and you are ready to start training the model.

**4.7.4. Part 4: Training the model**

Now that you have prepared the data, you can start training the model. This involves three steps:

* Setting up a container
* Setting the hyperparameters for the model
* Fitting the model

The hyperparameters are the interesting part of this code:

* _epochs_—Similar to the num\_round parameter for XGBoost in [chapters 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02) and [3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03), it specifies how many passes BlazingText performs over the training data. We chose the value of 10 after trying lower values and seeing that more epochs were required. Depending on how the results converge or begin overfitting, you might need to shift this value up or down.
* _vector\_dim_—Specifies the dimension of the word vectors that the algorithm learns; default is 100. We set this to 10 because experience has shown that a value as low as 10 is usually still effective and consumes less server time.
* _early\_stopping_—Similar to early stopping in XGBoost. The number of epochs can be set to a high value, and early stopping ensures that the training finishes when it stops, improving against the validation dataset.
* _patience_—Sets how many epochs should pass without improvement before early stopping kicks in.
* _min\_epochs_—Sets a minimum number of epochs that will be performed even if there is no improvement and the patience threshold is reached.
* _word\_ngrams_—N-grams were discussed in [figures 4.4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04fig04), [4.5](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04fig05), and [4.6](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04fig06) earlier in this chapter. Briefly, unigrams are single words, bigrams are pairs of words, and trigrams are groups of three words.

In [listing 4.12](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04ex12), the first line sets up a container to run the model. A _container_ is just the server that runs the model. The next group of lines configures the server. The set\_hyperparameters function sets the hyperparameters for the model. The final line in the next listing kicks off the training of the model.

**Listing 4.12. Training the model**

```
s3_output_location = f's3://{data_bucket}/\
{subfolder}/output'                                           1
sess = sagemaker.Session()                                    2
container = sagemaker.amazon.amazon_estimator.get_image_uri(
    boto3.Session().region_name,
    "blazingtext",
    "latest")                                                 3

estimator = sagemaker.estimator.Estimator(
    container,                                                4
    role,                                                     5
    train_instance_count=1,                                   6
    train_instance_type='ml.c4.4xlarge',                      7
    train_max_run = 600,                                      8
    output_path=s3_output_location,                           9
    sagemaker_session=sess)                                   10

estimator.set_hyperparameters(
    mode="supervised",                                        11
    epochs=10,                                                11
    vector_dim=10,                                            12
    early_stopping=True,                                      13
    patience=4,                                               14
    min_epochs=5,                                             15
    word_ngrams=2)                                            16

estimator.fit(inputs=data_channels, logs=True)
```

* _1_ Sets the location of the trained model
* _2_ Names this training session
* _3_ Sets up the container (server)
* _4_ Configures the server
* _5_ Assigns the role set up in [listing 4.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04ex1)
* _6_ Sets the number of servers used to train the model
* _7_ Sets the size of the server
* _8_ Maximum number of minutes the server will run before being terminated
* _9_ Specifies the location of the completed model
* _10_ Names this training session
* _11_ Specifies the mode for BlazingText (supervised or unsupervised)
* _12_ Sets the number of epochs (or rounds) of training
* _13_ Sets the number of vectors
* _14_ Enables early stopping
* _15_ Early stops after 4 epochs when no improvement is observed
* _16_ Performs a minimum of 5 epochs even if there’s no improvement after epoch 1
* _17_ Uses bigrams

**Note**

BlazingText can run in supervised or unsupervised mode. Because this chapter uses labeled text, we operate in supervised mode.

When you ran this cell in the current chapter (and in [chapters 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02) and [3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03)), you saw a number of rows with red notifications pop up in the notebook. The red notifications that appear when you run this cell look very different from the XGBoost notifications.

Each type of machine learning model provides information that is relevant to understanding how the algorithm is progressing. For the purposes of this book, the most important information comes at the end of the notifications: the training and validation accuracy scores, which display when the training finishes. The model in the following listing shows a training accuracy of over 98.88% and a validation accuracy of 92.28%. Each epoch is described by the validation accuracy.

**Listing 4.13. Training rounds output**

```
...
-------------- End of epoch: 9
Using 16 threads for prediction!
Validation accuracy: 0.922196
Validation accuracy improved! Storing best weights...
##### Alpha: 0.0005  Progress: 98.95%  Million Words/sec: 26.89 #####
-------------- End of epoch: 10
Using 16 threads for prediction!
Validation accuracy: 0.922455
Validation accuracy improved! Storing best weights...
##### Alpha: 0.0000  Progress: 100.00%  Million Words/sec: 25.78 #####
Training finished.
Average throughput in Million words/sec: 26.64
Total training time in seconds: 3.40

#train_accuracy: 0.9888                1
Number of train examples: 416634

#validation_accuracy: 0.9228           2
Number of validation examples: 104159

2018-10-07 06:56:20 Uploading - Uploading generated training model
2018-10-07 06:56:35 Completed - Training job completed
Billable seconds: 49
```

* _1_ Training accuracy
* _2_ Validation accuracy

**4.7.5. Part 5: Hosting the model**

Now that you have a trained model, you can host it on SageMaker so it is ready to make decisions ([listings 4.14](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04ex14) and [4.15](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04ex15)). We’ve covered a lot of ground in this chapter, so we’ll delve into how the hosting works in a subsequent chapter. For now, just know that it is setting up a server that receives data and returns decisions.

**Listing 4.14. Hosting the model**

```
endpoint_name = 'customer-support'                1
try:
    sess.delete_endpoint(
        sagemaker.predictor.RealTimePredictor(
            endpoint=endpoint_name).endpoint)     2
    print(
        'Warning: Existing endpoint deleted to make way for new endpoint.')
except:
    passalexample>
```

* _1_ In order to not create duplicate endpoints, name your endpoint.
* _2_ Deletes existing endpoint of that name

Next, in [listing 4.15](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04ex15), you create and deploy the endpoint. SageMaker is highly scalable and can handle very large datasets. For the datasets we use in this book, you only need a t2.medium machine to host your endpoint.

**Listing 4.15. Creating a new endpoint to host the model**

```
print('Deploying new endpoint...')
text_classifier = estimator.deploy(
    initial_instance_count = 1,
    instance_type = 'ml.t2.medium',
    endpoint_name=endpoint_name)        1
```

* _1_ Creates a new endpoint

**4.7.6. Part 6: Testing the model**

Now that the endpoint is set up and hosted, you can start making decisions. In [listing 4.16](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04ex16), you set a sample tweet, tokenize it, and then make a prediction.

Try changing the text in the first line and clicking ![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ctrl\_ent.jpg) to test different tweets. For example, changing the text _disappointed_ to _happy_ or _ambivalent_ changes the label from 1 to 0. This means that the tweet “Help me I’m very disappointed” will be escalated, but the tweets “Help me I’m very happy” and “Help me I’m very ambivalent” will not be escalated.

**Listing 4.16. Making predictions using the test data**

```
tweet = "Help me I'm very disappointed!"           1

tokenized_tweet = \
    [' '.join(nltk.word_tokenize(tweet))]          2
payload = {"instances" : tokenized_tweet}          3
response = \
    text_classifier.predict(json.dumps(payload))   4
escalate = pd.read_json(response)                  5
escalate                                           6
```

* _1_ Sample tweet
* _2_ Tokenizes the tweets
* _3_ Creates a payload in a format that the text\_classifier you created in [listing 4.15](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04ex15) can interpret
* _4_ Gets the response
* _5_ Converts the response to a pandas DataFrame
* _6_ Displays the decision

#### 4.8. Deleting the endpoint and shutting down your notebook instance <a href="#ch04lev1sec8__title" id="ch04lev1sec8__title"></a>

It is important that you shut down your notebook instance and delete your endpoint. We don’t want you to get charged for SageMaker services that you’re not using.

**4.8.1. Deleting the endpoint**

Appendix D describes how to shut down your notebook instance and delete your endpoint using the SageMaker console, or you can do that with the code in the next listing.

**Listing 4.17. Deleting the notebook**

```
# Remove the endpoint (optional)
# Comment out this cell if you want the endpoint to persist after Run All
sess.delete_endpoint(text_classifier.endpoint)
```

To delete the endpoint, uncomment the code in the listing, then click ![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ctrl\_ent.jpg) to run the code in the cell.

**4.8.2. Shutting down the notebook instance**

To shut down the notebook, go back to your browser tab where you have SageMaker open. Click the Notebook Instances menu item to view all of your notebook instances. Select the radio button next to the notebook instance name as shown in [figure 4.9](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04fig09), then click Stop on the Actions menu. It takes a couple of minutes to shut down.

**Figure 4.9. Shutting down the notebook**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch04fig09\_alt.jpg)

#### 4.9. Checking to make sure the endpoint is deleted <a href="#ch04lev1sec9__title" id="ch04lev1sec9__title"></a>

If you didn’t delete the endpoint using the notebook (or if you just want to make sure it is deleted), you can do this from the SageMaker console. To delete the endpoint, click the radio button to the left of the endpoint name, then click the Actions menu item and click Delete in the menu that appears.

When you have successfully deleted the endpoint, you will no longer incur AWS charges for it. You can confirm that all of your endpoints have been deleted when you see the text “There are currently no resources” displayed at the bottom of the Endpoints page ([figure 4.10](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04fig10)).

**Figure 4.10. Verifying that you have successfully deleted the endpoint**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch04fig10\_alt.jpg)

Naomi is very pleased with your results. She can now run all the tweets received by her team through your machine learning application to determine whether they should be escalated. And it identifies frustration in about the same way her team members used to identify discontented tweets (because the machine learning algorithm was trained using her team’s past decisions about whether to escalate a tweet). It’s pretty amazing. Imagine how hard it would have been to try to establish rules to identify frustrated tweeters.

#### Summary <a href="#ch04lev1sec10__title" id="ch04lev1sec10__title"></a>

* You determine which tweets to escalate using natural language processing (NLP) that captures meaning by a multidimensional word vector.
* In order to work with vectors in SageMaker, the only decision you need to make is whether SageMaker should use single words, pairs of words, or word triplets when creating the groups. To indicate this, NLP uses the terms unigram, bigram, and trigram, respectively.
* BlazingText is an algorithm that allows you to classify labeled text to set up your data for an NLP scenario.
* NLTK is a commonly used library for getting text ready to use in a machine learning model by tokenizing text.
* Tokenizing text involves splitting the text and stripping out those things that make it harder for the machine learning model to do what it needs to do.
