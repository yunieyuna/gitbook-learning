# Chapter 2. Should you send a purchase order to a technical approver?

_This chapter covers_

* Identifying a machine learning opportunity
* Identifying what and how much data is required
* Building a machine learning system
* Using machine learning to make decisions

In this chapter, you’ll work end to end through a machine learning scenario that makes a decision about whether to send an order to a technical approver or not. And you’ll do it without writing any rules! All you’ll feed into the machine learning system is a dataset consisting of 1,000 historical orders and a flag that indicates whether that order was sent to a technical approver or not. The machine learning system figures out the patterns from those 1,000 examples and, when given a new order, will be able to correctly determine whether to send the order to a technical approver.

In the first chapter, you’ll read about Karen, the person who works in the purchasing department. Her job is to receive requisitions from staff to buy a product or service. For each request, Karen decides which approver needs to review and approve the order; then, after getting approval, she sends the request to the supplier. Karen might not think of herself this way, but she’s a _decision maker_. As requests to buy products and services come in, Karen decides who needs to approve each request. For some products, such as computers, Karen needs to send the request to a technical advisor, who determines if the specification is suitable for the person buying the computer. Does Karen need to send this order to a technical approver, or not? This is the decision you’ll work on in this chapter.

By the end of this chapter, you’ll be able to help Karen out. You’ll be able to put together a solution that will look at the requests before they get to Karen and then recommend whether she should send the request to a technical approver. As you work through the examples, you’ll become familiar with how to use machine learning to make a decision.

#### 2.1. The decision <a href="#ch02lev1sec1__title" id="ch02lev1sec1__title"></a>

[Figure 2.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02fig01) shows Karen’s process from a requester placing an order to the supplier receiving the order. Each person icon in the workflow represents a person taking some action. If they have more than one arrow pointing away from them, they need to make a decision.

**Figure 2.1. Approval workflow for purchasing technical equipment in Karen’s company**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch02fig01\_alt.jpg)

In Karen’s process, there are three decisions (numbered 1, 2, and 3):

1. The first decision is the one we are going to look at in this chapter: should Karen send this order to a technical approver?
2. The second decision is made by the technical approver after Karen routes an order to them: should the technical approver accept the order and send it to finance, or should it be rejected and sent back to the requester?
3. The third decision is made by the financial approver: should they approve the order and send it to the supplier, or should it be rejected and sent back to the requester?

Each of these decisions may be well suited for machine learning—and, in fact, they are. Let’s look at the first decision (Karen’s decision) in more detail to understand why it’s suitable for machine learning.

#### 2.2. The data <a href="#ch02lev1sec2__title" id="ch02lev1sec2__title"></a>

In discussions with Karen, you’ve learned that the approach she generally takes is that if a product looks like an IT product, she’ll send it to a technical approver. The exception to her rule is that if it’s something that can be plugged in and used, such as a mouse or a keyboard, she doesn’t send it for technical approval. Nor does she send it for technical approval if the requester is from the IT department.

[Table 2.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02table01) shows the dataset you will work with in this scenario. This dataset contains the past 1,000 orders that Karen has processed.

It’s good practice when preparing a labeled dataset for supervised machine learning to put the target variable in the first column. In this scenario, your target variable is this: _should Karen send the order to a technical approver?_ In your dataset, if Karen sent the order for technical approval, you put a 1 in the tech\_approval\_required column. If she did not send the order for technical approval, you put a 0 in that column. The rest of the columns are _features_. These are things that you think are going to be useful in determining whether an item should be sent to an approver. Just like target variables come in two types, categorical and continuous, features also come in two types: categorical and continuous.

Categorical features in [table 2.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02table01) are those in the requester\_id, role, and product columns. A _categorical feature_ is something that can be divided into a number of distinct groups. These are often text rather than numbers, as you can see in the following columns:

* _requester\_id_—ID of the requester.
* _role_—If the requester is from the IT department, these are labeled tech.
* _product_—The type of product.

Continuous features are those in the last three columns in [table 2.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02table01). _Continuous features_ are always numbers. The continuous features in this dataset are quantity, price, and total.

**Table 2.1. Technical Approval Required dataset contains information from prior orders received by Karen.**

| tech\_approval\_required | requester\_id | role     | product          | quantity | price | total |
| ------------------------ | ------------- | -------- | ---------------- | -------- | ----- | ----- |
| 0                        | E2300         | tech     | Desk             | 1        | 664   | 664   |
| 0                        | E2300         | tech     | Keyboard         | 9        | 649   | 5841  |
| 0                        | E2374         | non-tech | Keyboard         | 1        | 821   | 821   |
| 1                        | E2374         | non-tech | Desktop Computer | 24       | 655   | 15720 |
| 0                        | E2327         | non-tech | Desk             | 1        | 758   | 758   |
| 0                        | E2354         | non-tech | Desk             | 1        | 576   | 576   |
| 1                        | E2348         | non-tech | Desktop Computer | 21       | 1006  | 21126 |
| 0                        | E2304         | tech     | Chair            | 3        | 155   | 465   |
| 0                        | E2363         | non-tech | Keyboard         | 1        | 1028  | 1028  |
| 0                        | E2343         | non-tech | Desk             | 3        | 487   | 1461  |

The fields selected for this dataset are those that will allow you to replicate Karen’s decision-making process. There are many other fields that you could have selected, and there are some very sophisticated tools being released that help in selecting those features. But for the purposes of this scenario, you’ll use your intuition about the problem you’re solving to select your features. As you’ll see, this approach can quickly lead to some excellent results. Now you’re ready to do some machine learning:

* Your end goal is to be able to submit an order to the machine learning model and have it return a result that recommends sending the order to a technical approver or not.
* You have identified the features you’ll use to make the decision (the type of product and whether the requester is from the IT department).
* You have created your labeled historical dataset (the dataset shown in [table 2.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02table01)).

#### 2.3. Putting on your training wheels <a href="#ch02lev1sec3__title" id="ch02lev1sec3__title"></a>

Now that you have your labeled dataset, you can train a machine learning model to make decisions. But what is a model and how do you train it?

You’ll learn more about how machine learning works in the following chapters. For now, all you need to know is that a _machine learning model_ is a mathematical function that is rewarded for guessing right and punished for guessing wrong. In order to get more guesses right, the function associates certain values in each feature with right guesses or wrong guesses. As it works through more and more samples, it gets better at guessing. When it’s run through all the samples, you say that the model is _trained_.

The mathematical function that underlies a machine learning model is called the _machine learning algorithm_. Each machine learning algorithm has a number of parameters you can set to get a better-performing model. In this chapter, you are going to accept all of the default settings for the algorithm you’ll use. In subsequent chapters, we’ll discuss how to fine tune the algorithm to get better results.

One of the most confusing aspects for machine learning beginners is deciding which machine learning algorithm to use. In the supervised machine learning exercises in this book, we focus on just one algorithm: XGBoost. XGBoost is a good choice because

* It is fairly forgiving; it works well across a wide range of problems without significant tuning.
* It doesn’t require a lot of data to provide good results.
* It is easy to explain why it returns a particular prediction in a certain scenario.
* It is a high-performing algorithm and the go-to algorithm for many participants in machine learning competitions with small datasets.

In a later chapter, you’ll learn how XGBoost works under the hood, but for now, let’s discuss how to use it. If you want to read more about it on the AWS site, you can do so here: [https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html).

**Note**

If you haven’t already installed and configured all the tools you’ll need as you work through this chapter and the book, visit appendixes A, B, and C, and follow the instructions you find there. After working your way through the appendixes (to the end of [appendix C](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_028.html#app03)), you’ll have your dataset stored in S3 (AWS’s file storage service) and a Jupyter notebook set up and running in SageMaker.

#### 2.4. Running the Jupyter notebook and making predictions <a href="#ch02lev1sec4__title" id="ch02lev1sec4__title"></a>

Let’s step through the Jupyter notebook and make predictions about whether to send an order to a technical approver or not. In this chapter, we will look at the notebook in six parts:

* Load and examine the data.
* Get the data into the right shape.
* Create training, validation, and test datasets.
* Train the machine learning model.
* Host the machine learning model.
* Test the model and use it to make decisions.

To follow along, you should have completed two things:

* Load the dataset orders\_with\_predicted\_value.csv into S3.
* Upload the Jupyter notebook tech\_approval\_required.ipynb to SageMaker.

Appendixes B and C take you through how to do each of these steps in detail for the dataset you’ll use in this chapter. In summary:

* Download the dataset at [https://s3.amazonaws.com/mlforbusiness/ch02/orders\_with\_predicted\_value.csv](https://s3.amazonaws.com/mlforbusiness/ch02/orders\_with\_predicted\_value.csv).
* Upload the dataset to your S3 bucket that you have set up to hold the datasets for this book.
* Download the Jupyter notebook at [https://s3.amazonaws.com/mlforbusiness/ch02/tech\_approval\_required.ipynb](https://s3.amazonaws.com/mlforbusiness/ch02/tech\_approval\_required.ipynb).
* Upload the Jupyter notebook to your SageMaker notebook instance.

Don’t be frightened by the code in the Jupyter notebook. As you work through this book, you’ll become familiar with each aspect of it. In this chapter, you’ll run the code rather than edit it. In fact, in this chapter, you do not need to modify any of the code with the exception of the first two lines, where you tell the code which S3 bucket to use and which folder in that bucket contains your dataset.

To start, open the SageMaker service from the AWS console in your browser by logging into the AWS console at [http://console.aws.amazon.com](http://console.aws.amazon.com/). Click Notebook Instances in the left-hand menu on SageMaker ([figure 2.2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02fig02)). This takes you to a screen that shows your notebook instances.

**Figure 2.2. Selecting a notebook instance from the Amazon SageMaker menu**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch02fig02\_alt.jpg)

If the notebook you uploaded in [appendix C](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_028.html#app03) is not running, you will see a screen like the one shown in [figure 2.3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02fig03). Click the Start link to start the notebook instance.

**Figure 2.3. Notebook instance with a Stopped status**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch02fig03\_alt.jpg)

Once you have started your notebook instance, or if your notebook was already started, you’ll see a screen like that shown in [figure 2.4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02fig04). Click the Open link.

**Figure 2.4. Opening Jupyter from a notebook instance**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch02fig04\_alt.jpg)

When you click the Open link, a new tab opens in your browser, and you’ll see the ch02 folder you created in [appendix C](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_028.html#app03) ([figure 2.5](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02fig05)).

**Figure 2.5. Selecting the ch02 folder**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch02fig05\_alt.jpg)

Finally, when you click ch02, you’ll see the notebook you uploaded in [appendix C](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_028.html#app03): tech-approval-required.ipynb. Click this notebook to open it in a new browser tab ([figure 2.6](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02fig06)).

**Figure 2.6. Opening the tech-approval-required notebook**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch02fig06\_alt.jpg)

[Figure 2.7](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02fig07) shows a Jupyter notebook. Jupyter notebooks are an amazing coding environment that combines text with code in sections. An example of the text cell is the text for heading 2.4.1, [_Part 1_](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_009.html#part01)_: Load and examine the data_. An example of a code cell are the following lines:

**Figure 2.7. Inside the Jupyter notebook**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch02fig07\_alt.jpg)

```
data_bucket = 'mlforbusiness'
subfolder = 'ch02'
dataset = 'orders_with_predicted_value.csv'
```

To run the code in a Jupyter notebook, press ![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ctrl\_ent.jpg) when your cursor is in a code cell.

**2.4.1. Part 1: Loading and examining the data**

The code in [listings 2.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex1) through [2.4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex4) loads the data so you can look at it. The only two values in this entire notebook that you need to modify are the data\_bucket and the subfolder. You should use the bucket and subfolder names you set up as per the instructions in [appendix B](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_024.html#app02).

**Note**

This walkthrough will just familiarize you with the code so that, when you see it again in subsequent chapters, you’ll know how it fits into the SageMaker workflow.

The following listing shows how to identify the bucket and subfolder where the data is stored.

**Listing 2.1. Setting up the S3 bucket and subfolder**

```
data_bucket = 'mlforbusiness'                   1
subfolder = 'ch02'                              2
dataset = 'orders_with_predicted_value.csv'     3
```

* _1_ S3 bucket where the data is stored
* _2_ Subfolder of S3 bucket where the data is stored
* _3_ Dataset used to train and test the model

As you’ll recall, a Jupyter notebook is where you can write and run code. There are two ways you can run code in a Jupyter notebook. You can run the code in one of the cells, or you can run the code in more than one of the cells.

To run the code in one cell, click the cell to select it, and then press ![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ctrl\_ent.jpg). When you do so, you’ll see an asterisk (\*) appear to left of the cell. This means that the code in the cell is running. When the asterisk is replaced by a number, the code has finished running. The number shows how many cells have been run since you opened the notebook.

If you want, after you have updated the name of the data bucket and the subfolder ([listing 2.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex1)), you can run the notebook. This loads the data, builds and trains the machine learning model, sets up the endpoint, and generates predictions from the test data. SageMaker takes about 10 min to complete these actions for the datasets you’ll use in this book. It may take longer if you load large datasets from your company.

To run the entire notebook, click Cell in the toolbar at the top of the Jupyter notebook, then click Run All ([figure 2.8](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02fig08)).

**Figure 2.8. Running the Jupyter notebook**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch02fig08\_alt.jpg)

**Setting up the notebook**

Next, you’ll set up the Python libraries required by the notebook ([listing 2.2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex2)). To run the notebook, you don’t need to change any of these values:

* _pandas_—A Python library commonly used in data science projects. In this book, we’ll touch only the surface of what pandas can do. You’ll load pandas as pd. As you’ll see later in this chapter, this means that we will preface any use of any module in the pandas library with pd.
* _boto3_ and _sagemaker_—The libraries created by Amazon to help Python users interact with AWS resources: boto3 is used to interact with S3, and sagemaker, unsurprisingly, is used to interact with SageMaker. You will also use a module called _s3fs_, which makes it easier to use boto3 with S3.
* _sklearn_—The final library you’ll import. It is short for _scikit-learn_, which is a comprehensive library of machine learning algorithms that is used widely in both the commercial and scientific communities. Here we only import the train\_test\_split function that we’ll use later.

You’ll also need to create a role on SageMaker that allows the sagemaker library to use the resources it needs to build and serve the machine learning application. You do this by calling the sagemaker function get\_execution\_role.

**Listing 2.2. Importing modules**

```
import pandas as pd                        1
import boto3                               2
import sagemaker                           3
import s3fs                                4
from sklearn.model_selection \
        import train_test_split            5

role = sagemaker.get_execution_role()      6
s3 = s3fs.S3FileSystem(anon=False)         7
```

* _1_ Imports the pandas Python library
* _2_ Imports the boto3 AWS library
* _3_ Imports SageMaker
* _4_ Imports the s3fs module to make working with S3 files easier
* _5_ Imports only the train\_test\_split module from the sklearn library
* _6_ Creates a role in SageMaker
* _7_ Establishes the connection with S3

As a reminder, as you walk through each of the cells in the Jupyter notebook, to run the code in a cell, click the cell and press ![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ctrl\_ent.jpg).

**LOADING AND VIEWING THE DATA**

Now that you’ve identified the bucket and subfolder and set up the notebook, you can take a look at the data. The best way to view the data is to use the pandas library you imported in [listing 2.2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex2).

The code in [listing 2.3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex3) creates a pandas data structure called a _DataFrame_. You can think of a DataFrame as a table like a spreadsheet. The first line assigns the name _df_ to the DataFrame. The data in the DataFrame is the orders data from S3. It is read into the DataFrame by using the pandas function read\_csv. The line df.head() displays the first five rows of the df DataFrame.

**Listing 2.3. Viewing the dataset**

```
df = pd.read_csv(
    f's3://{data_bucket}/{subfolder}/{dataset}')     1
df.head()                                            2
```

* _1_ Reads the S3 orders\_with\_predicted\_value.csv dataset in [listing 2.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex1)
* _2_ Displays the top five rows of the DataFrame loaded in the line above

Running the code displays the top five rows in the dataset. (To run the code, insert your cursor in the cell and press ![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ctrl\_ent.jpg).) The dataset will look similar to the dataset in [table 2.2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02table02).

**Table 2.2. Technical Approval Required dataset displayed in Excel**

| tech\_approval\_required | requester\_id | role     | product          | quantity | price | total |
| ------------------------ | ------------- | -------- | ---------------- | -------- | ----- | ----- |
| 0                        | E2300         | tech     | Desk             | 1        | 664   | 664   |
| 0                        | E2300         | tech     | Keyboard         | 9        | 649   | 5841  |
| 0                        | E2374         | non-tech | Keyboard         | 1        | 821   | 821   |
| 1                        | E2374         | non-tech | Desktop Computer | 24       | 655   | 15720 |
| 0                        | E2327         | non-tech | Desk             | 1        | 758   | 758   |

To recap, the dataset you uploaded to S3 and are now displaying in the df DataFrame lists the last 1,000 orders that Karen processed. She sent some of those orders to a technical approver, and some she did not.

The code in [listing 2.4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex4) displays how many rows are in the dataset and how many of the rows were sent for technical approval. Running this code shows that out of 1,000 rows in the dataset, 807 were not sent to a technical approver, and 193 were sent.

**Listing 2.4. Determining how many rows were sent for technical approval**

```
print(f'Number of rows in dataset: {df.shape[0]}')     1
print(df[df.columns[0]].value_counts())                2
```

* _1_ Displays the number of rows
* _2_ You don’t need to understand this line of code. The output displays the number of rows that went to a technical approver and the number of rows that did not.

In [listing 2.4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex4), the shape property of a DataFrame provides information about the number of rows and the number of columns. Here df.shape\[0] shows the number of rows, and df.shape\[1] shows the number of columns. The value\_counts property of the df DataFrame shows the number of rows in the dataset where the order was sent to a technical approver. It contains a 1 if it was sent for technical approval and a 0 if it was not.

**2.4.2. Part 2: Getting the data into the right shape**

For this part of the notebook, you’ll prepare the data for use in the machine learning model. You’ll learn more about this topic in later chapters but, for now, it is enough to know that there are standard approaches to preparing data, and we are using one we’ll apply to each of our machine learning exercises.

One important point to understand about most machine learning models is that they typically work with numbers rather than text-based data. We’ll discuss why this is so in a subsequent chapter when we go into the details of the XGBoost algorithm. For now, it is enough to know that you need to convert the text-based data to numerical data before you can use it to train your machine learning model. Fortunately, there are easy-to-use tools that will help with this.

First, we’ll use the pandas get\_dummies function to convert all of the text data into numbers. It does this by creating a separate column for every unique text value. For example, the product column contains text values such as Desk, Keyboard, and Mouse. When you use the get\_dummies function, it turns every value into a column and places a 0 or 1 in the row, depending on whether the row contains a value or not.

[Table 2.3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02table03) shows a simple table with three rows. The table shows the price for a desk, a keyboard, and a mouse.

**Table 2.3. Simple three-row dataset with prices for a desk, a keyboard, and a mouse**

| product  | price |
| -------- | ----- |
| Desk     | 664   |
| Keyboard | 69    |
| Mouse    | 89    |

When you use the get\_dummies function, it takes each of the unique values in the non-numeric columns and creates new columns from those. In our example, this looks like the values in [table 2.4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02table04). Notice that the get\_dummies function removes the product column and creates three columns from the three unique values in the dataset. It also places a 1 in the new column that contains the value from that row and zeros in every other column.

**Table 2.4. The three-row dataset after applying the get\_dummies function**

| price | product\_Desk | product\_Keyboard | product\_Mouse |
| ----- | ------------- | ----------------- | -------------- |
| 664   | 1             | 0                 | 0              |
| 69    | 0             | 1                 | 0              |
| 89    | 0             | 0                 | 1              |

[Listing 2.5](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex5) shows the code that creates [table 2.4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02table04). To run the code, insert your cursor in the cell, and press ![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ctrl\_ent.jpg). You can see that this dataset is very wide (111 columns in our example).

**Listing 2.5. Converting text values to columns**

```
encoded_data = pd.get_dummies(df)      1
encoded_data.head()                    2
```

* _1_ Creates a new pandas DataFrame to store the table with columns for each unique text value in the original table
* _2_ The pandas function to display the first five rows of the table

A machine learning algorithm can now work with this data because it is all numbers. But there is a problem. Your dataset is now probably very wide. In our sample dataset in [figures 2.3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02fig03) and [2.4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02fig04), the dataset went from 2 columns wide to 4 columns wide. In a real dataset, it can go to thousands of columns wide. Even our sample dataset in the SageMaker Jupyter notebook goes to 111 columns when you run the code in the cells.

This is not a problem for the machine learning algorithm. It can easily handle datasets with thousands of columns. It’s a problem for you because it becomes more difficult to reason about the data. For this reason, and for the types of machine learning decision-making problems we look at in this book, you can often get results that are just as accurate by reducing the number of columns to only the most relevant ones. This is important for your ability to explain to others what is happening in the algorithm in a way that is convincing. For example, in the dataset you work with in this chapter, the most highly correlated columns are the ones relating to technical product types and the ones relating to whether the requester has a tech role or not. This makes sense, and it can be explained concisely to others.

A _relevant_ column for this machine learning problem is a column that contains values that are correlated to the value you are trying to predict. You say that two values are correlated when a change in one value is accompanied by a change in another value. If these both increase or decrease together, you say that they are _positively correlated_—they both move in the same direction. And when one goes up and the other goes down (or vice versa), you say that they are _negatively correlated_—they move in opposite directions. For our purposes, the machine learning algorithm doesn’t really care whether a column is positively or negatively correlated, just that it is correlated.

Correlation is important because the machine learning algorithm is trying to predict a value based on the values in other columns in the dataset. The values in the dataset that contribute most to the prediction are those that are correlated to the predicted value.

You’ll find the most correlated columns by applying another pandas function called corr. You apply the corr function by appending .corr() to the pandas DataFrame you named encoded\_data in [listing 2.5](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex5). After the function, you need to provide the name of the column you are attempting to predict. In this case, the column you are attempting to predict is the tech\_approval\_required column. [Listing 2.6](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex6) shows the code that does this. Note that the .abs() function at the end of the listing is simply turning all of the correlations positive.

**Listing 2.6. Identifying correlated columns**

```
corrs = encoded_data.corr()[
            'tech_approval_required'
        ].abs()                        1
columns = corrs[corrs > .1].index      2
corrs = corrs.filter(columns)          3
corrs                                  4
```

* _1_ Creates a series (a DataFrame with just one column) called corrs that lists all the columns in the 111-column dataset you created with the code in [listing 2.5](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex5)
* _2_ Identifies the columns that have a correlation greater than 10%
* _3_ Filters corrs to just the columns with a correlation greater than 10%
* _4_ Jupyter notebooks display the output of the last line in a cell. Because the last line is the name of a variable, it displays the variable when you run the code in a cell.

The code in [listing 2.6](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex6) identifies the columns that have a correlation greater than 10%. You don’t need to know exactly how this code works. You are simply finding all of the columns that have a correlation greater than 10% with the tech\_approval\_required column. Why 10%? It removes the irrelevant noise from the dataset which, while this step does not help the machine learning algorithm, improves your ability to talk about the data in a meaningful way. With fewer columns to consider, you can more easily prepare an explanation of what the algorithm is doing.

[Table 2.5](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02table05) shows the columns with a correlation greater than 10%.

**Table 2.5. Correlation with predicted value**

| Column name               | Correlation with predicted value |
| ------------------------- | -------------------------------- |
| tech\_approval\_required  | 1.000000                         |
| role\_non-tech            | 0.122454                         |
| role\_tech                | 0.122454                         |
| product\_Chair            | 0.134168                         |
| product\_Cleaning         | 0.191539                         |
| product\_Desk             | 0.292137                         |
| product\_Desktop Computer | 0.752144                         |
| product\_Keyboard         | 0.242224                         |
| product\_Laptop Computer  | 0.516693                         |
| product\_Mouse            | 0.190708                         |

Now that you have identified the most highly correlated columns, you need to filter the encoded\_data table to contain just those columns. You do so with the code in [listing 2.7](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex7). The first line filters the columns to just the correlated columns, and the second line displays the table when you run the code. (Remember, to run the code, insert your cursor in the cell and press ![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ctrl\_ent.jpg).)

**Listing 2.7. Showing only correlated columns**

```
encoded_data = encoded_data[columns]      1
encoded_data.head()                       2
```

* _1_ Filters columns to only those that are correlated with the tech\_approval\_required column
* _2_ Displays the table when you run the code in the cell

**2.4.3. Part 3: Creating training, validation, and test datasets**

The next step in the machine learning process is to create a dataset that you’ll use to train the algorithm. While you’re at it, you’ll also create the dataset you’ll use to validate the results of the training and the dataset you’ll use to test the results. To do this, you’ll split the dataset into three parts:

* Train
* Validate
* Test

The machine learning algorithm uses the training data to train the model. You should put the largest chunk of data into the training dataset. The validation data is used by the algorithm to determine whether the algorithm is improving. This should be the next-largest chunk of data. Finally, the test data, the smallest chunk, is used by you to determine how well the algorithm performs. Once you have the three datasets, you will convert them to CSV format and then save them to S3.

In [listing 2.8](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex8), you create two datasets: a dataset with 70% of the data for training and a dataset with 30% of the data for validation and testing. Then you split the validation and test data into two separate datasets: a validation dataset and a test dataset. The validation dataset will contain 20% of the total rows, which equals 66.7% of the validation and test dataset. The test dataset will contain 10% of the total rows, which equals 33.3% of the validation and test dataset. To run the code, insert your cursor in the cell and press ![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ctrl\_ent.jpg).

**Listing 2.8. Splitting data into train, validate, and test datasets**

```
train_df, val_and_test_data = train_test_split(
        encoded_data,
        test_size=0.3,
        random_state=0)              1
val_df, test_df = train_test_split(
        val_and_test_data,
        test_size=0.333,
        random_state=0)              2
```

* _1_ Puts 70% of the data into the train dataset
* _2_ Puts 20% of the data in to the validation data and 10% of the data into the test dataset

**Note**

The random\_state argument ensures that repeating the command splits the data in the same way.

In [listing 2.8](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex8), you split the data into three DataFrames. In [listing 2.9](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex9), you’ll convert the datasets to CSV format.

Input and CSV formats

CSV is one of the two formats you can use as input to the XGBoost machine learning model. The code in this book uses CSV format. That’s because if you want to view the data in a spreadsheet, it’s easy to import into spreadsheet applications like Microsoft Excel. The drawback of using CSV format is that it takes up a lot of space if you have a dataset with lots of columns (like our encoded\_data dataset after using the get\_dummies function in [listing 2.5](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex5)).

The other format that XBGoost can use is libsvm. Unlike a CSV file, where even the columns containing zeros need to be filled out, the libsvm format only includes the columns that do not contain zeros. It does this by concatenating the column number and the value together. So the data you looked at in [table 2.2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02table02) would look like this:

1:664 2:1

1:69 3:1

1:89 4:1

The first entry in each row shows the price (664, 69, or 89). The number in front of the price indicates that this is in the first column of the dataset. The second entry in each row contains the column number (2, 3, or 4) and the non-zero value of the entry (which in our case is always 1). So 1:89 4:1 means that the first column in the row contains the number 89, and the fourth column contains the number 1. All the other values are zero.

You can see that using libsvm over CSV has changed the width of our dataset from four columns to two columns. But don’t get too hung up on this. SageMaker and XGBoost work just fine with CSV files with thousands of columns; but if your dataset is very wide (tens of thousands of columns), you might want to use libsvm instead of CSV. Otherwise, use CSV because it’s easier to work with.

[Listing 2.9](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex9) shows how to use the pandas function to\_csv to create a CSV dataset from the pandas DataFrames you created in [listing 2.8](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex8). To run the code, insert your cursor in the cell and press ![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ctrl\_ent.jpg).

**Listing 2.9. Converting the data to CSV**

```
train_data = train_df.to_csv(None, header=False, index=False).encode()
val_data = val_df.to_csv(None, header=False, index=False).encode()
test_data = test_df.to_csv(None, header=True, index=False).encode()
```

The None argument in the to\_csv function indicates that you do not want to save to a file. The header argument indicates whether the column names will be included in the CSV file or not. For the train\_data and val\_data datasets, you don’t include the column headers (header=False) because the machine learning algorithm is expecting each column to contain only numbers. For the test\_data dataset, it is best to include headers because you’ll be running the trained algorithm against the test data, and it is helpful to have column names in the data when you do so. The index=False argument tells the function to not include a column with the row numbers. The encode() function ensures that the text in the CSV file is in the right format.

**Note**

Encoding text in the right format can be one of the most frustrating parts of machine learning. Fortunately, a lot of the complexity of this is handled by the pandas library, so you generally won’t have to worry about that. Just remember to always use the encode() function if you save the file to CSV.

In [listing 2.9](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex9), you created CSV files from the train, val, and test DataFrames. However, the CSV files are not stored anywhere yet other than in the memory of the Jupyter notebook. In [listing 2.10](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex10), you will save the CSV files to S3.

In [listing 2.2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex2) in the fourth line, you imported a Python module called s3fs. This module makes it easy to work with S3. In the last line of the same listing, you assigned the variable s3 to the S3 filesystem. You will now use this variable to work with S3. To do this, you’ll use Python’s with...open syntax to indicate the filename and location, and the write function to write the contents of a variable to that location ([listing 2.10](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex10)).

Remember to use 'wb' when creating the file to indicate that you are writing the contents of the file in binary mode rather than text mode. (You don’t need to know how this works, just that it allows the file to be read back exactly as it was saved.) To run the code, insert your cursor in the cell and press ![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ctrl\_ent.jpg).

**Listing 2.10. Saving the CSV file to S3**

```
with s3.open(f'{data_bucket}/{subfolder}/processed/train.csv', 'wb') as f:
    f.write(train_data)                                                   1

with s3.open(f'{data_bucket}/{subfolder}/processed/val.csv', 'wb') as f:
    f.write(val_data)                                                     2

with s3.open(f'{data_bucket}/{subfolder}/processed/test.csv', 'wb') as f:
    f.write(test_data)                                                    3
```

* _1_ Writes train.csv to S3
* _2_ Writes val.csv to S3
* _3_ Writes test.csv to S3

**2.4.4. Part 4: Training the model**

Now you can start training the model. You don’t need to understand in detail how this works at this point, just what it is doing, so the code in this part will not be annotated to the same extent as the code in the earlier listings.

First, you need to load your CSV data into SageMaker. This is done using a SageMaker function called s3\_input. In [listing 2.11](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex11), the s3\_input files are called train\_input and test\_input. Note that you don’t need to load the test.csv file into SageMaker because it is not used to train or validate the model. Instead, you will use it at the end to test the results. To run the code, insert your cursor in the cell and press ![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ctrl\_ent.jpg).

**Listing 2.11. Preparing the CSV data for SageMaker**

```
train_input = sagemaker.s3_input(
    s3_data=f's3://{data_bucket}/{subfolder}/processed/train.csv',
    content_type='csv')
val_input = sagemaker.s3_input(
    s3_data=f's3://{data_bucket}/{subfolder}/processed/val.csv',
    content_type='csv')
```

The next listing ([listing 2.12](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex12)) is truly magical. It is what allows a person with no systems engineering experience to do machine learning. In this listing, you train the machine learning model. This sounds simple, but the ease with which you can do this using SageMaker is a massive step forward from having to set up your own infrastructure to train a machine learning model. In [listing 2.12](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex12), you

1. Set up a variable called sess to store the SageMaker session.
2. Define in which container AWS will store the model (use the containers given in [listing 2.12](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex12)).
3. Create the model (which is stored in the variable estimator in [listing 2.12](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex12)).
4. Set hyperparameters for the estimator.

You will learn more about each of these steps in subsequent chapters, so you don’t need to understand this deeply at this point in the book. You just need to know that this code will create your model, start a server to run your model, and then train the model on the data. If you click ![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ctrl\_ent.jpg) in this notebook cell, the model runs.

**Listing 2.12. Training the model**

```
sess = sagemaker.Session()

container = sagemaker.amazon.amazon_estimator.get_image_uri(
    boto3.Session().region_name,
    'xgboost',
    'latest')

estimator = sagemaker.estimator.Estimator(
    container,
    role,
    train_instance_count=1,
    train_instance_type='ml.m5.large',                1
    output_path= \
        f's3://{data_bucket}/{subfolder}/output',     2
    sagemaker_session=sess)

estimator.set_hyperparameters(
    max_depth=5,
    subsample=0.7,
    objective='binary:logistic',                      3
     eval_metric = 'auc',                             4
     num_round=100,                                   5
     early_stopping_rounds=10)                        6

estimator.fit({'train': train_input, 'validation': val_input})
```

* _1_ Sets the type of server that SageMaker uses to run your model
* _2_ Stores the output of the model at this location in S3
* _3_ SageMaker has some very sophisticated hyperparameter tuning capability. To use the tuning function, you just need to make sure you set the objective correctly. For the current dataset, where you are trying to predict a 1 or 0, you should set this as binary:logistic. We’ll cover this in more detail in later chapters.
* _4_ Tells SageMaker to tune the hyperparameters to achieve the best area under curve result. Again, we’ll cover this in more detail in later chapters.
* _5_ The maximum number of rounds of training that will occur. This is covered in more detail in [chapter 3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03).
* _6_ The number of rounds that will occur where there is no improvement in the model before training is terminated

It takes about 5 minutes to train the model, so you can sit back and reflect on how happy you are to not be manually configuring a server and installing the software to train your model. The server only runs for about a minute, so you will only be charged for about a minute of compute time. At the time of writing of this book, the m5-large server was priced at under US$0.10 per hour. Once you have stored the model on S3, you can use it again whenever you like without retraining the model. More on this in later chapters.

**2.4.5. Part 5: Hosting the model**

The next section of the code is also magical. In this section, you will launch another server to host the model. This is the server that you will use to make predictions from the trained model.

Again, at this point in the book, you don’t need to understand how the code in the listing in this section works—just that it’s creating a server that you’ll use to make predictions. The code in [listing 2.13](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex13) calls this endpoint order-approval and uses Python’s try-except block to create it.

A try-except block tries to run some code, and if there is an error, it runs the code after the except line. You do this because you only want to set up the endpoint if you haven’t already set one up with that name. The listing tries to set up an endpoint called order-approval. If there is no endpoint named order-approval, then it sets one up. If there is an order-approval endpoint, the try code generates an error, and the except code runs. The except code in this case simply says to use the endpoint named order-approval.

**Listing 2.13. Hosting the model**

```
endpoint_name = 'order-approval'
try:
    sess.delete_endpoint(
        sagemaker.predictor.RealTimePredictor(
            endpoint=endpoint_name).endpoint)
    print('Warning: Existing endpoint deleted\
to make way for your new endpoint.')
except:
    pass

predictor = estimator.deploy(initial_instance_count=1,
               instance_type='ml.t2.medium',              1
               endpoint_name=endpoint_name)

from sagemaker.predictor import csv_serializer, json_serializer
predictor.content_type = 'text/csv'
predictor.serializer = csv_serializer
predictor.deserializer = None
```

* _1_ The type of server you are using, in this case, an ml.t2.medium server.

The code in [listing 2.13](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex13) sets up a server sized as a t2.medium server. This is a smaller server than the m5.large server we used to train the model because making predictions from a model is less computationally intensive than creating the model. Both the try block and the except block create a variable called predictor that you’ll draw on to test and use the model. The final four lines in the code set up the predictor to work with the CSV file input so you can work with it more easily.

Note that when you click ![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ctrl\_ent.jpg) in the notebook cell, the code will take another 5 minutes or so to run the first time. It takes time because it is setting up a server to host the model and to create an endpoint so you can use the model.

**2.4.6. Part 6: Testing the model**

Now that you have the model trained and hosted on a server (the endpoint named predictor), you can start using it to make predictions. The first three lines of the next listing create a function that you’ll apply to each row of the test data.

**Listing 2.14. Getting the predictions**

```
def get_prediction(row):
    prediction = round(float(predictor.predict(row[1:]).decode('utf-8')))
    return prediction

with s3.open(f'{data_bucket}/{subfolder}/processed/test.csv') as f:
    test_data = pd.read_csv(f)

test_data['prediction'] = test_data.apply(get_prediction, axis=1)
test_data.set_index('prediction', inplace=True)
test_data
```

The function get\_prediction in [listing 2.14](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex14) takes every column in the test data (except for the first column, because that is the value you are trying to predict), sends it to the predictor, and returns the prediction. In this case, the prediction is 1 if the order should be sent to an approver, and 0 if it should _not_ be sent to an approver.

The next two lines open the test.csv file and read the contents into a pandas DataFrame called test\_data. You can now work with this DataFrame in the same way you worked with the original dataset in [listing 2.7](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex7). The final three lines apply the function created in the first three lines of the listing.

When you click ![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ctrl\_ent.jpg) in the cell containing the code in [listing 2.14](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02ex14), you see the results of each row in the test file. [Table 2.6](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02table06) shows the top two rows of the test data. Each row represents a single order. For example, if an order for a desk was placed by a person in a technical role, the role\_tech and the product\_desk columns would have a 1. All other columns would be 0.

**Table 2.6. Test results from the predictor**

| prediction | tech\_approval\_required | role\_non-tech | role\_tech | product\_Chair | product\_Cleaning | product\_Desk | product\_Desktop Computer | product\_Keyboard | product\_Laptop Computer | product\_Mouse |
| ---------- | ------------------------ | -------------- | ---------- | -------------- | ----------------- | ------------- | ------------------------- | ----------------- | ------------------------ | -------------- |
| 1          | 1                        | 1              | 0          | 0              | 0                 | 0             | 1                         | 0                 | 0                        | 0              |
| 0          | 0                        | 1              | 0          | 0              | 1                 | 0             | 0                         | 0                 | 0                        | 0              |

The 1 in the prediction column in first row says that the model predicts that this order should be sent to a technical approver. The 1 in the tech\_approval\_required column says that in your test data, this order was labeled as requiring technical approval. This means that the machine learning model predicted this order correctly.

To see why, look at the values in the columns to the right of the tech\_approval\_ required column. You can see that this order was placed by someone who is not in a technical role because there is a 1 in the role\_non-tech column and a 0 in the role\_tech column. And you can see that the product ordered was a desktop computer because the product\_Desktop Computer column has a 1.

The 0 in the prediction column in the second row says that the model predicts this order does _not_ require technical approval. The 0 in the tech\_approval\_required column, because it is the same value as that in the prediction column, says that the model predicted this correctly.

The 1 in the role\_non-tech column says this order was also placed by a non-technical person. However, the 1 in the product\_Cleaning column indicates that the order was for cleaning products; therefore, it does not require technical approval.

As you look through the results, you can see that your machine learning model got almost every test result correct! You have just created a machine learning model that can correctly decide whether to send orders to a technical approver, all without writing any rules. To determine how accurate the results are, you can test how many of the predictions match the test data, as shown in the following listing.

**Listing 2.15. Testing the model**

```
(test_data['prediction'] == \
    test_data['tech_approval_required']).mean()      1
```

* _1_ Displays the percentage of predictions that match the test dataset

#### 2.5. Deleting the endpoint and shutting down your notebook instance <a href="#ch02lev1sec5__title" id="ch02lev1sec5__title"></a>

It is _important_ that you shut down your notebook instance and delete your endpoints when you are not using them. If you leave them running, you will be charged for each second they are up. The charges for the machines we use in this book are not large (if you were to leave a notebook instance or endpoint on for a month, it would cost about US$20). But, there is no point in paying for something you are not using.

**2.5.1. Deleting the endpoint**

To delete the endpoint, click Endpoints on the left-hand menu you see when you are looking at the SageMaker tab ([figure 2.9](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02fig09)).

**Figure 2.9. Selecting the endpoint for deletion**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch02fig09\_alt.jpg)

You will see a list of all of your running endpoints ([figure 2.10](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02fig10)). To ensure you are not charged for endpoints you are not using, you should delete all of the endpoints you are not using (remember that endpoints are easy to create, so even if you will not use the endpoint for only a few hours, you might want to delete it).

**Figure 2.10. Viewing the active endpoint(s)**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch02fig10\_alt.jpg)

To delete the endpoint, click the radio button to the left of order-approval, click the Actions menu item, then click the Delete menu item that appears ([figure 2.11](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02fig11)).

**Figure 2.11. Deleting the endpoint**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch02fig11\_alt.jpg)

You have now deleted the endpoint, so you’ll no longer incur AWS charges for it. You can confirm that all of your endpoints have been deleted when you see the text “There are currently no resources” displayed on the Endpoints page ([figure 2.12](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02fig12)).

**Figure 2.12. Endpoint deleted**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch02fig12\_alt.jpg)

**2.5.2. Shutting down the notebook instance**

The final step is to shut down the notebook instance. Unlike endpoints, you do not delete the notebook. You just shut it down so you can start it again, and it will have all of the code in your Jupyter notebook ready to go. To shut down the notebook instance, click Notebook instances in the left-hand menu on SageMaker ([figure 2.13](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02fig13)).

**Figure 2.13. Selecting the notebook instance to prepare for shutdown**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch02fig13\_alt.jpg)

To shut down the notebook, select the radio button next to the notebook instance name and click Stop on the Actions menu ([figure 2.14](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02fig14)). After your notebook instance has shut down, you can confirm that it is no longer running by checking the Status to ensure it says Stopped.

**Figure 2.14. Shutting down the notebook**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch02fig14\_alt.jpg)

This chapter was all about helping Karen decide whether to send an order to a technical approver. In this chapter, you worked end to end through a machine learning scenario. The scenario you worked through involved how to decide whether to send an order to a technical approver. The skills you learned in this chapter will be used throughout the rest of the book as you work through other examples of using machine learning to make decisions in business automation.

#### Summary <a href="#ch02lev1sec6__title" id="ch02lev1sec6__title"></a>

* You can uncover machine learning opportunities by identifying decision points.
* It’s simple to set up SageMaker and build a machine learning system using AWS SageMaker and Jupyter notebooks.
* You send data to the machine learning endpoints to make predictions.
* You can test the predictions by sending the data to a CSV file for viewing.
* To ensure you are not charged for endpoints you are not using, you should delete all unused endpoints.
