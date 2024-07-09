# Chapter 3. Should you call a customer because they are at risk of churning?

_This chapter covers_

* Identifying customers who are about to churn
* How to handle imbalanced data in your analysis
* How the XGBoost algorithm works
* Additional practice in using S3 and SageMaker

Carlos takes it personally when a customer stops ordering from his company. He’s the Head of Operations for a commercial bakery that sells high-quality bread and other baked goods to restaurants and hotels. Most of his customers have used his bakery for a long time, but he still regularly loses customers to his competitors. To help retain customers, Carlos calls those who have stopped using his bakery. He hears a similar story from each of these customers: they like his bread, but it’s expensive and cuts into their desired profit margins, so they try bread from another, less expensive bakery. After this trial, his customers conclude that the quality of their meals would still be acceptable even if they served a lower quality bread.

_Churn_ is the term used when you lose a customer. It’s a good word for Carlos’s situation because it indicates that a customer probably hasn’t stopped ordering bread; they’re just ordering it from someone else.

Carlos comes to you for help in identifying those customers who are in the process of trying another bakery. Once he’s identified the customer, he can call them to determine if there’s something he can do to keep them. In Carlos’s conversations with his lost customers, he sees a common pattern:

* Customers place orders in a regular pattern, typically daily.
* A customer tries another bakery, thus reducing the number of orders from Carlos’s bakery.
* The customer negotiates an agreement with the other bakery, which may or may not result in a temporary resurgence in orders placed with Carlos’s bakery.
* Customers stop ordering from his bakery altogether.

In this chapter, you are going to help Carlos understand which customers are at risk of churning so he can call them to determine whether there is some way to address their move to another supplier. To help Carlos, you’ll look at the business process in a similar way that you looked at Karen’s process in [chapter 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02).

For Karen’s process, you looked at how orders moved from requester to approver, and the features that Karen used to make a decision about whether to send the order to a technical approver or not. You then built a SageMaker XGBoost application that automated the decision. Similarly, with Carlos’s decision about whether to call a customer because they are at risk of churning, you’ll build a SageMaker XGBoost application that looks at Carlos’s customers each week and makes a decision about whether Carlos should call them.

#### 3.1. What are you making decisions about? <a href="#ch03lev1sec1__title" id="ch03lev1sec1__title"></a>

At first glance, it looks like you are working with ordering data like you did in [chapter 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02), where Karen looks at an order and decides whether to send it to a technical approver or not. In this chapter, Carlos reviews a customer’s orders and decides whether to call that customer. The difference between Karen’s process flow in [chapter 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02) and Carlos’s process flow in this chapter is that, in [chapter 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02), you made decisions about orders: should Karen send this order to an approver. In this chapter, you make decisions about customers: should Carlos call a customer.

This means that instead of just taking the order data and using that as your dataset, you need to first transform the order data into customer data. In later chapters, you’ll learn how to use some automated tools to do this, but in this chapter, you’ll learn about the process conceptually, and you’ll be provided with the transformed dataset. But before we look at the data, let’s look at the process we want to automate.

#### 3.2. The process flow <a href="#ch03lev1sec2__title" id="ch03lev1sec2__title"></a>

[Figure 3.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03fig01) shows the process flow. You start with the Orders database, which contains records of which customers have bought which products and when.

**Figure 3.1. Carlos’s process flow for deciding which customers to call**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch03fig01\_alt.jpg)

Carlos believes that there is a pattern to customers’ orders before they decide to move to a competitor. This means that you need to turn order data into customer data. One of the easiest ways to think about this is to picture the data as a table like it might be displayed in Excel. Your order data has a single row for each of the orders. If there are 1,000 orders, there will be 1,000 rows in your table. If these 1,000 orders were from 100 customers, when you turn your order data into customer data, your table with 1,000 rows will become a table with 100 rows.

This is shown in step 1 in [figure 3.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03fig01): transform the orders dataset into a customer dataset. You’ll see how to do this in the next section. Let’s move on to step 2 now, which is the primary focus of this chapter. In step 2, you answer the question, should Carlos call a customer?

After you’ve prepared the customer database, you’ll use that data to prepare a SageMaker notebook. When the notebook is complete, you’ll send data about a customer to the SageMaker endpoint and return a decision about whether Carlos should call that customer.

#### 3.3. Preparing the dataset <a href="#ch03lev1sec3__title" id="ch03lev1sec3__title"></a>

The base dataset is very simple. It has the customer code, customer name, date of the order, and the value of the order. Carlos has 3,000 customers who, on average, place 3 orders per week. This means that, over the course of the past 3 months, Carlos received 117,000 orders (3,000 customer × 3 orders per week × 13 weeks).

**Note**

Throughout this book, the datasets you’ll use are simplified examples of datasets you might encounter in your work. We have done this to highlight certain machine learning techniques rather than to devote significant parts of each chapter to understanding the data.

To turn 117,000 rows into a 3,000-row table (one row per customer), you need to group the non-numerical data and summarize the numerical data. In the dataset shown in [table 3.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03table01), the non-numerical fields are customer\_code, customer\_name, and date. The only numerical field is amount.

**Table 3.1. Test results from the predictor**

| customer\_code | customer\_name                   | date       | amount |
| -------------- | -------------------------------- | ---------- | ------ |
| 393            | Gibson Group                     | 2018-08-18 | 264.18 |
| 393            | Gibson Group                     | 2018-08-17 | 320.14 |
| 393            | Gibson Group                     | 2018-08-16 | 145.95 |
| 393            | Gibson Group                     | 2018-08-15 | 280.59 |
| 840            | Meadows, Carroll, and Cunningham | 2018-08-18 | 284.12 |
| 840            | Meadows, Carroll, and Cunningham | 2018-08-17 | 232.41 |
| 840            | Meadows, Carroll, and Cunningham | 2018-08-16 | 235.95 |
| 840            | Meadows, Carroll, and Cunningham | 2018-08-15 | 184.59 |

Grouping customer\_code and customer\_name is easy. You want a single row per customer\_code. And you can simply use the customer name associated with each customer code. In [table 3.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03table01), there are two different customer\_codes in the rows 393 and 840, and each has a company associated with it: Gibson Group and Meadows, Carroll, and Cunningham.

Grouping the dates is the interesting part of the dataset preparation in this chapter. In discussions with Carlos, you learned that he believes there is a pattern to the customers that stop using his bakery. The pattern looks like this:

1. A customer believes they can use a lower quality product without impacting their business.
2. They try another bakery’s products.
3. They set up a contract with the other bakery.
4. They stop using Carlos’s bakery.

Carlos’s ordering pattern will be stable over time, then drop while his customers try a competitor’s products, and then return to normal while a contract with his competitor is negotiated. Carlos believes that this pattern should be reflected in the customers’ ordering behavior.

In this chapter, you’ll use XGBoost to see if you can identify which customers will stop using Carlos’s bakery. Although several tools exist to help you prepare the data, in this chapter, you won’t use those tools because the focus of this chapter is on machine learning rather than data preparation. In a subsequent chapter, however, we’ll show you how to use these tools with great effect. In this chapter, you’ll take Carlos’s advice that most of his customers follow a weekly ordering pattern, so you’ll summarize the data by week.

You’ll apply two transformations to the data:

* Normalize the data
* Calculate the change from week to week

The first transformation is to calculate the percentage spend, relative to the average week. This normalizes all of the data so that instead of dollars, you are looking at a weekly change relative to the _average_ sales. The second transformation is to show the change from week to week. You do this because you want the machine learning algorithm to see the patterns in the weekly changes as well as the relative figures for the same time period.

Note that for this chapter, we’ll apply these transformations for you, but later chapters will go more into how to transform data. Because the purpose of this chapter is to learn more about XGBoost and machine learning, we’ll tell you what the data looks like so you won’t have to do the transformations yourself.

**3.3.1. Transformation 1: Normalizing the data**

For our dataset, we’ll do the following:

1. Take the sum of the total spent over the year for each of Carlos’s customers and call that total\_spend.
2. Find the average spent per week by dividing total\_spend by 52.
3. For each week, calculate the total spent per week divided by the average spent per week to get a weekly spend as a percentage of an average spend.
4. Create a column for each week.

[Table 3.2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03table02) shows the results of this transformation.

**Table 3.2. Customer dataset grouped by week after normalizing the data**

| customer\_code | customer\_name                   | total\_sales | week\_minus\_4 | week\_minus\_3 | week\_minus\_2 | last\_week |
| -------------- | -------------------------------- | ------------ | -------------- | -------------- | -------------- | ---------- |
| 393            | Gibson Group                     | 6013.96      | 1.13           | 1.18           | 0.43           | 2.09       |
| 840            | Meadows, Carroll, and Cunningham | 5762.40      | 0.52           | 1.43           | 0.87           | 1.84       |

**3.3.2. Transformation 2: Calculating the change from week to week**

As [table 3.3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03table03) shows, for each week from the column named week\_minus\_3 to last\_week, you subtract the value from the preceding week and call it the delta between the weeks. For example, in week\_minus\_3, the Gibson Group has sales that are 1.18 times their average week. In week\_minus\_4, their sales are 1.13 times their average sales. This means that their weekly sales rose by 0.05 of their normal sales from week\_minus\_4 to week\_minus\_3. This is the delta between week\_minus\_3 and week\_minus\_4 and is recorded as 0.05 in the 4-3\_delta column.

**Table 3.3. Customer dataset grouped by week, showing changes per week**

| customer\_code | customer\_name                   | total\_sales | week\_minus\_4 | week\_minus\_3 | week\_minus\_2 | last\_week | 4-3\_delta | 3-2\_delta | 2-1\_delta |
| -------------- | -------------------------------- | ------------ | -------------- | -------------- | -------------- | ---------- | ---------- | ---------- | ---------- |
| 393            | Gibson Group                     | 6013.96      | 1.13           | 1.18           | 0.43           | 2.09       | 0.05       | -0.75      | 1.66       |
| 840            | Meadows, Carroll, and Cunningham | 5762.40      | 0.52           | 1.43           | 0.87           | 1.84       | 0.91       | -0.56      | 0.97       |

The following week was a disaster in sales for the Gibson Group: sales decreased by 0.75 times their average weekly sales. This is shown by a –0.75 in the 3-2\_delta column. Their sales rebounded in the last week though, as they went to 2.09 times their average weekly sales. This is shown by the 1.66 in the 2-1\_delta column.

Now that you’ve prepared the data, let’s move on to setting up the machine learning application by first looking at how XGBoost works.

#### 3.4. XGBoost primer <a href="#ch03lev1sec4__title" id="ch03lev1sec4__title"></a>

In [chapter 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02), you used XGBoost to help Karen decide which approver to send an order to, but we didn’t go into much detail on how it works. We’ll cover this now.

**3.4.1. How XGBoost works**

XGBoost can be understood at a number of levels. How deep you go in your understanding depends on your needs. A high-level person will be satisfied with a high-level answer. A more detailed person will require a more detailed understanding. Carlos and Karen will both need to understand the model enough to show their managers they know what’s going on. How deep they have to go really depends on their managers.

At the highest level, in the circle example from [chapter 1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_010.html#ch01) (reproduced in [figure 3.2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03fig02)), we separated the dark circles from the light circles using two approaches:

**Figure 3.2. Machine learning function to identify a group of similar items (reprinted from** [**chapter 1**](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_010.html#ch01)**)**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch03fig02\_alt.jpg)

* Rewarding the function for getting a dark circle on the right and punishing it for getting a dark circle on the left.
* Rewarding the function for getting a light circle on the left and punishing it for getting a light circle on the right.

This could be considered an _ensemble_ machine learning model, which is a model that uses multiple approaches when it learns. In a way, XGBoost is also an ensemble machine learning model, which means it uses a number of different approaches to improve the effectiveness of its learning. Let’s go another level deeper into the explanation.

XGBoost stands for Extreme Gradient Boosting. Consider the name in two parts:

* Gradient boosting
* Extreme

_Gradient boosting_ is a technique where different learners are used to improve a function. You can think of this like ice hockey players’ sticks handling the puck down the ice. Instead of trying to push the puck straight ahead, they use small corrections to guide the puck in the right direction. Gradient boosting follows a similar approach.

The _Extreme_ part of the name is in recognition that XGBoost has a number of other characteristics that makes the model particularly accurate. For example, the model automatically handles regularization of the data so you’re not inadvertently thrown off by a dataset with big differences in the values you look at.

And finally, for the next level of depth, the sidebar gives Richard’s more detailed explanation.

Richie’s explanation of XGBoost

XGBoost is an incredibly powerful machine learning model. To begin with, it supports multiple forms of regularization. This is important because gradient boosting algorithms are known to have potential problems with overfitting. An _overfit model_ is one that is very strongly tied to the unique features of the training data and does not generalize well to unseen data. As we add more rounds to XGBoost, we can see this when our validation accuracy starts deteriorating.

Apart from restricting the number of rounds with early stopping, XGBoost also controls overfitting with column and row subsampling and the parameters eta, gamma, lambda, and alpha. This penalizes specific aspects of the model that tend to make it fit the training data too tightly.

Another feature is that XGBoost builds each tree in parallel on all available cores. Although each step of gradient boosting needs to be carried out serially, XGBoost’s use of all available cores for building each tree gives it a big advantage over other algorithms, particularly when solving more complex problems.

XGBoost also supports _out-of-core computation_. When data does not fit into memory, it divides that data into blocks and stores those on disk in a compressed form. It even supports sharding of these blocks across multiple disks. These blocks are then decompressed on the fly by an independent thread while loading into memory.

XGBoost has been extended to support massively parallel processing big data frameworks such as Spark, Flink, and Hadoop. This means it can be used for building extremely large and complex models with potentially billions of rows and millions of features that run at high speed.

XGBoost is _sparsity aware_, meaning that it handles missing values without any requirement for imputation. We have taken this for granted, but many machine learning algorithms require values for all attributes of all samples; in which case, we would have had to impute an appropriate value. This is not always easy to do without skewing the results of the model in some way. Furthermore, XGBoost handles missing values in a very efficient way: its performance is proportional to the number of present values and is independent of the number of missing values.

Finally, XGBoost implements a highly efficient algorithm for optimizing the objective known as _Newton boosting_. Unfortunately, an explanation of this algorithm is beyond the scope of this book.

You can read more about XGBoost on Amazon’s site: [https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html).

**3.4.2. How the machine learning model determines whether the function is- s getting better or getting worse AUC**

XGBoost is good at learning. But what does it mean to learn? It simply means that the model gets punished less and rewarded more. And how does the machine learning model know whether it should be punished or rewarded? The area under the curve (AUC) is a metric that is commonly used in machine learning as the basis for rewarding or punishing the function. The curve is the “When the function gets a greater area under the curve, it is rewarded” guideline. When the function gets a reduced AUC, it is punished.

To get a feel for how AUC works, imagine you are a celebrity in a fancy resort. You are used to being pampered, and you expect that to happen. One of the staff attending to your every whim is the shade umbrella adjuster. Let’s call him _Function_. When Function fails to adjust the umbrella so that you are covered by its shade, you berate him. When he consistently keeps you in the shade, you give him a tip. That is how a machine learning model works: it rewards Function when he increases the AUC and punishes him when he reduces the AUC. Now over to Richie for a more technical explanation.

Richie’s explanation of the area under the curve (AUC)

When we tell XGBoost that our objective is _binary:logistic_, what we are asking it for is actually not a prediction of a positive or negative label. We are instead asking for the probability of a positive label. As a result, we get a continuous value between 0 and 1. It is then up to us to decide what probability will produce a positive prediction.

It might make sense to choose 0.5 (50%) as our cutoff, but at other times, we might want to be really certain of our prediction before predicting a positive. Typically, we would do this when the cost of the decision associated with a positive label is quite high. In other cases, the cost of missing a positive can be more important and justify choosing a cutoff much less than 0.5.

The plot in this sidebar’s figure shows true positives on the y-axis as a fraction between 0 and 1, and false positives on the x-axis as a fraction between 0 and 1:

* The _true positive rate_ is the portion of all positives that are actually identified as positive by our model.
* The _false positive rate_ is the portion of incorrect positive predictions as a percentage of all negative numbers.

This plot is known as an _ROC curve_.a When we use AUC as our evaluation metric, we are telling XGBoost to optimize our model by maximizing the area under the ROC curve to give us the best possible results when averaged across all cutoff probabilities between 0 and 1.

**The area under the curve (AUC) showing true and false positive values**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/pg58fig01\_alt.jpg)

Whichever value you choose for your cutoff produces both TP (true positive) and corresponding FP (false positive) values. If you choose a probability that allows you to capture most or all of the true positives by picking a low cutoff (such as 0.1), you will also accidentally predict more negatives as positives. Whichever value you pick will be a trade-off between these two competing measures of model accuracy.

When the curve is well above the diagonal (as in this figure), you get an AUC value close to 1. A model that simply matches the TP and FP rates for each cutoff will have an AUC of 0.5 and will directly match the diagonal dotted line in the figure.

a ROC stands for Receiver Operator Characteristic. It was first developed by engineers during World War II for detecting enemy objects in battle, and the name has stuck.

#### 3.5. Getting ready to build the model <a href="#ch03lev1sec5__title" id="ch03lev1sec5__title"></a>

Now that you have a deeper understanding of how XGBoost works, you can set up another notebook on SageMaker and make some decisions. As you did in [chapter 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02), you are going to do the following:

* Upload a dataset to S3.
* Set up a notebook on SageMaker.
* Upload the starting notebook.
* Run it against the data.

Along the way, we’ll go into some details we glossed over in [chapter 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02).

**Tip**

If you’re jumping into the book at this chapter, you might want to visit the appendixes, which show you how to do the following:

* [Appendix A](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_021.html#app01): sign up for AWS, Amazon’s web service
* [appendix B](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_024.html#app02): set up S3, AWS’s file storage service
* [Appendix C](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_028.html#app03): set up SageMaker

**3.5.1. Uploading a dataset to S3**

To set up the dataset for this chapter, you’ll follow the same steps as you did in [appendix B](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_024.html#app02). You don’t need to set up another bucket though. You can just go to the same bucket you created earlier. In our example, we called the bucket _mlforbusiness_, but your bucket will be called something different. When you go to your S3 account, you will see something like that shown in [figure 3.3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03fig03).

**Figure 3.3. Viewing the list of buckets in S3**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch03fig03\_alt.jpg)

Click this bucket to see the ch02 folder you created in the previous chapter. For this chapter, you’ll create a new folder called _ch03_. You do this by clicking Create Folder and following the prompts to create a new folder.

Once you’ve created the folder, you are returned to the folder list inside your bucket. There you will see you now have a folder called ch03.

Now that you have the ch03 folder set up in your bucket, you can upload your data file and start setting up the decision-making model in SageMaker. To do so, click the folder and download the data file at this link:

[https://s3.amazonaws.com/mlforbusiness/ch03/churn\_data.csv](https://s3.amazonaws.com/mlforbusiness/ch03/churn\_data.csv).

Then upload the CSV file into your ch03 folder by clicking the Upload button. Now you’re ready to set up the notebook instance.

**3.5.2. Setting up a notebook on SageMaker**

Like you did in [chapter 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02), you’ll set up a notebook on SageMaker. This process is much faster for this chapter because, unlike in [chapter 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02), you now have a notebook instance set up and ready to run. You just need to run it and upload the Jupyter notebook we prepared for this chapter. (If you skipped [chapter 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02), follow the instructions in [appendix C](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_028.html#app03) on how to set up SageMaker.)

When you go to SageMaker, you’ll see your notebook instances. The notebook instance you created for [chapter 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02) (or that you’ve just created by following the instructions in [appendix C](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_028.html#app03)) will either say Open or Start. If it says Start, click the Start link and wait a couple of minutes for SageMaker to start. Once the screen displays Open Jupyter, select that link to open your notebook list.

Once it opens, create a new folder for [chapter 3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03) by clicking New and selecting Folder at the bottom of the dropdown list ([figure 3.4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03fig04)). This creates a new folder called Untitled Folder.

**Figure 3.4. Creating a new folder in SageMaker**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch03fig04.jpg)

When you tick the checkbox next to Untitled Folder, you will see the Rename button appear. Click it, and change the folder name to ch03 ([figure 3.5](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03fig05)).

**Figure 3.5. Renaming a folder in SageMaker**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch03fig05.jpg)

Click the ch03 folder, and you will see an empty notebook list. Just as we already prepared the CSV data you uploaded to S3 (churn\_data.csv), we’ve already prepared the Jupyter Notebook you’ll now use. You can download it to your computer by navigating to this URL:

[https://s3.amazonaws.com/mlforbusiness/ch03/customer\_churn.ipynb](https://s3.amazonaws.com/mlforbusiness/ch03/customer\_churn.ipynb).

Click Upload to upload the customer-churn.ipynb notebook to the ch03 folder ([figure 3.6](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03fig06)).

**Figure 3.6. Uploading a notebook to SageMaker**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch03fig06\_alt.jpg)

After uploading the file, you’ll see the notebook in your list. Click it to open it. Now, just like in [chapter 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02), you are a few keystrokes away from being able to run your machine learning model.

#### 3.6. Building the model <a href="#ch03lev1sec6__title" id="ch03lev1sec6__title"></a>

As in [chapter 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02), you will go through the code in six parts:

* Load and examine the data.
* Get the data into the right shape.
* Create training, validation, and test datasets.
* Train the machine learning model.
* Host the machine learning model.
* Test the model and use it to make decisions.

**3.6.1. Part 1: Loading and examining the data**

First, you need to tell SageMaker where your data is. Update the code in the first cell of the notebook to point to your S3 bucket and folders ([listing 3.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03ex1)). If you called your S3 folder ch03 and did not rename the churn\_data.csv file, then you just need to update the name of the data bucket to the name of the S3 bucket you uploaded the data to. Once you have done that, you can actually run the entire notebook. As you did in [chapter 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02), to run the notebook, click Cell in the toolbar at the top of the Jupyter Notebook, then click Run All.

**Listing 3.1. Setting up the notebook and storing the data**

```
data_bucket = 'mlforbusiness'    1
subfolder = 'ch03'               2
dataset = 'churn_data.csv'       3
```

* _1_ S3 bucket where the data is stored
* _2_ Subfolder of the S3 bucket where the data is stored
* _3_ Dataset that’s used to train and test the model

When you run the notebook, SageMaker loads the data, trains the model, sets up the endpoint, and generates decisions from the test data. SageMaker takes about 10 minutes to complete these actions, so you have time to get yourself a cup of coffee or tea while this is happening.

When you return with your hot beverage, if you scroll to the bottom of your notebook, you should see the decisions that were made on the test data. But before we get into that, let’s work through the notebook.

Back at the top of the notebook, you’ll see the cell that imports the Python libraries and modules you’ll use in this notebook. You’ll hear more about these in a subsequent chapter. For now, let’s move to the next cell. If you didn’t click Run All in the notebook, click the cell and press ![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ctrl\_ent.jpg) to run the code in this cell, as shown in [listing 3.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03ex1).

Moving on to the next cell, you will now import all of the Python libraries and modules that SageMaker uses to prepare the data, train the machine learning model, and set up the endpoint.

As you learned in [chapter 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02), pandas is one of the most commonly used Python libraries in data science. In the code cell shown in [listing 3.2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03ex2), you’ll import pandas as pd. When you see pd in the cell, it means you are using a pandas function. Other items that you import include these:

* _boto3_—Amazon’s Python library that helps you work with AWS services in Python.
* _sagemaker_—Amazon’s Python module to work with SageMaker.
* _s3fs_—A module that makes it easier to use boto3 to manage files on S3.
* _sklearn.metrics_—A new import (it wasn’t used in [chapter 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02)). This module lets you generate summary reports on the output of the machine learning model.

**Listing 3.2. Importing the modules**

```
import pandas as pd                      1
import boto3                             2
import sagemaker                         3
import s3fs                              4
from sklearn.model_selection \
    import train_test_split              5
import sklearn.metrics as metrics        6

role = sagemaker.get_execution_role()    7
s3 = s3fs.S3FileSystem(anon=False)       8
```

* _1_ Imports the pandas Python library
* _2_ Imports the boto3 AWS library
* _3_ Imports SageMaker
* _4_ Imports the s3fs module to make working with S3 files easier
* _5_ Imports only the train\_test\_split module from the sklearn library
* _6_ Imports the metrics module from the sklearn library
* _7_ Creates a role in SageMaker
* _8_ Establishes the connection with S3

In the cell in [listing 3.3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03ex3), we are using the pandas read\_csv function to read our data and the head function to display the top five rows. This is one of the first things you’ll do in each of the chapters so you can see the data and understand its shape. To load and view the data, click the cell with your mouse to select it, and then press ![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ctrl\_ent.jpg) to run the code.

**Listing 3.3. Loading and viewing the data**

```
df = pd.read_csv(
    f's3://{data_bucket}/{subfolder}/{dataset}')      1
df.head()                                             2
```

* _1_ Reads the S3 churn\_data.csv dataset in [listing 3.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03ex1)
* _2_ Displays the top five rows of the DataFrame

You can see that the data has a single customer per row and that it reflects the format of the data in [table 3.3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03table03). The first column in [table 3.4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03table04) indicates whether the customer churned or did not churn. If the customer churned, the first column contains a 1. If they remain a customer, it should show a 0. Note that these data rows are provided by way of example, and the rows of data you see might be different.

**Table 3.4. Dataset for Carlos’s customers displayed in Excel**

| churned | id | customer\_code | co\_name                        | total\_spend | week\_minus\_4 | week\_minus\_3 | week\_minus\_2 | last\_week | 4-3\_delta | 3-2\_delta | 2-1\_delta |
| ------- | -- | -------------- | ------------------------------- | ------------ | -------------- | -------------- | -------------- | ---------- | ---------- | ---------- | ---------- |
| 0       | 1  | 1826           | Hoffman, Martinez, and Chandler | 68567.34     | 0.81           | 0.02           | 0.74           | 1.45       | 0.79       | -0.72      | -0.71      |
| 0       | 2  | 772            | Lee Martin and Escobar          | 74335.27     | 1.87           | 1.02           | 1.29           | 1.19       | 0.85       | -0.27      | 0.10       |

You can see from the first five customers that none of them have churned. This is what you would expect because Carlos doesn’t lose that many customers.

To see how many rows are in the dataset, you run the pandas shape function as shown in [listing 3.4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03ex4). To see how many customers in the dataset have churned, you run the pandas value\_counts function.

**Listing 3.4. Number of churned customers in the dataset**

```
print(f'Number of rows in dataset: {df.shape[0]}')      1
print(df['churned'].value_counts())                     2
```

* _1_ Displays the total number of rows
* _2_ Displays the number of rows where a customer churned and the number of rows where the customer did not

You can see from this data that out of 2,999 rows of data, 166 customers have churned. This represents a churn rate of about 5% per week, which is higher than the rate that Carlos experiences. Carlos’s true churn rate is about 0.5% per week (or about 15 customers per week).

We did something a little sneaky with the data in this instance to bring the churn rate up to this level. The dataset actually contains the churned customers from the past three months and a random selection of non-churned customers over that same period to bring the total number of customers up to 2,999 (the actual number of customers that Carlos has). We did this because we are going to cover how to handle extremely rare events in a subsequent chapter and, for this chapter, we wanted to use a similar toolset to that which we used in [chapter 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02).

There are risks in the approach we took with the data in this chapter. If there are differences in the ordering patterns of churned customers over the past three months, then our results might be invalid. In discussions with Carlos, he believes that the pattern of churning and normal customers remains steady over time, so we felt confident we can use this approach.

The other point to note is that this approach might not be well received if we were writing an academic paper. One of the lessons you’ll learn as you work with your own company’s data is that you rarely get everything you want. You have to constantly assess whether you can make good decisions based on the data you have.

**3.6.2. Part 2: Getting the data into the right shape**

Now that you can see your dataset in the notebook, you can start working with it. XGBoost can only work with numbers, so we need to either remove our categorical data or encode it.

_Encoding data_ means that you set each distinct value in the dataset as a column and then put a 1 in the rows that contain the value of the column and a zero in the other rows in that column. This worked well for your products in Karen’s dataset, but it will not help you out with Carlos’s dataset. That’s because the categorical data (customer-\_name, customer\_code, and id) are unique—these occur only once in the dataset. Turning these into columns would not improve the model either.

Your best approach in this case is also the simplest approach: just remove the categorical data. To remove the data, use the pandas drop function, and display the first five rows of the dataset again by using the head function. You use axis=1 to indicate that you want to remove columns rather than rows in the pandas DataFrame.

**Listing 3.5. Removing the categorical data**

```
encoded_data = df.drop(
    ['id', 'customer_code', 'co_name'],
    axis=1)                              1
encoded_data.head()                      2
```

* _1_ Removes categorical columns by calling the drop function on the df DataFrame
* _2_ Displays the first five rows of the DataFrame

Removing the columns shows the dataset without the categorical information ([table 3.5](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03table05)).

**Table 3.5. The transformed dataset without categorical information**

| churned | total\_spend | week\_minus\_4 | week\_minus\_3 | week\_minus\_2 | last\_week | 4-3\_delta | 3-2\_delta | 2-1\_delta |
| ------- | ------------ | -------------- | -------------- | -------------- | ---------- | ---------- | ---------- | ---------- |
| 0       | 68567.34     | 0.81           | 0.02           | 0.74           | 1.45       | 0.79       | -0.72      | -0.71      |
| 0       | 74335.27     | 1.87           | 1.02           | 1.29           | 1.19       | 0.85       | -0.27      | 0.10       |

**3.6.3. Part 3: Creating training, validation, and test datasets**

Now that you have your data in a format that XGBoost can work with, you can split the data into test, validation, and training datasets as you did in [chapter 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02). One important difference with the approach we are taking going forward is that we are using the stratify parameter during the split.

The stratify parameter is particularly useful for datasets where the target variable you are predicting is relatively rare. The parameter works by shuffling the deck as it’s building the machine learning model and making sure that the train, validate, and test datasets contain similar ratios of target variables. This ensures that the model does not get thrown off course by an unrepresentative selection of customers in any of the datasets.

We glossed over this code in [chapter 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02). We’ll go into more depth here and show you how to use stratify ([listing 3.6](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03ex6)). You create training and testing samples from a dataset, with 70% allocated to the training data and 30% allocated to the testing and validation samples. The stratify argument tells the function to use y to stratify the data so that a _random_ sample is balanced proportionally according to the distinct values in y.

You might notice that the code to split the dataset is slightly different than the code used in [chapter 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02). Because you are using stratify, you have to explicitly declare your target column (churned in this example). The stratify function returns a couple of additional values that you don’t care about. The underscores in the y = test\_and\_val\_data line (those beginning with val\_df) are simply placeholders for variables. Don’t worry if this seems a bit arcane. You don’t need to understand this part of the code in order to train, validate, and test the model.

You then split the testing and validation data, with two thirds allocated to validation and one third to testing ([listing 3.6](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03ex6)). Over the entire dataset, 70% of the data is allocated to training, 20% to validation, and 10% to testing.

**Listing 3.6. Creating the training, validation, and test datasets**

```
y = encoded_data['churned']                            1
train_df, test_and_val_data, _, _ = train_test_split(
    encoded_data,
    y,
    test_size=0.3,
    stratify=y,
    random_state=0)                                    2

y = test_and_val_data['churned']
val_df, test_df, _, _ = train_test_split(
    testing_data,
    y,
    test_size=0.333,
    stratify=y,
    random_state=0)                                    3

print(train_df.shape, val_df.shape, test_df.shape)
print()
print('Train')
print(train_df['churned'].value_counts())              4
print()
print('Validate')
print(val_df['churned'].value_counts())
print()
print('Test')
print(test_df['churned'].value_counts())
```

* _1_ Sets the target variable for use in splitting the data
* _2_ Creates training and testing samples from dataset df. random\_state and ensures that repeating the command splits the data in the same way
* _3_ Splits testing and validation data into a validation dataset and a testing dataset
* _4_ The value\_counts function shows the number of customers who did not churn (denoted by a 0) and the number of churned customers (denoted by a 1) in the train, validate, and test datasets.

Just as you did in [chapter 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02), you convert the three datasets to CSV and save the data to S3. The following listing creates the datasets that you’ll save to the same S3 folder as your original churn\_data.csv file.

**Listing 3.7. Converting the datasets to CSV and saving to S3**

```
train_data = train_df.to_csv(None, header=False, index=False).encode()
val_data = val_df.to_csv(None, header=False, index=False).encode()
test_data = test_df.to_csv(None, header=True, index=False).encode()

with s3.open(f'{data_bucket}/{subfolder}/processed/train.csv', 'wb') as f:
    f.write(train_data)                                                    1

with s3.open(f'{data_bucket}/{subfolder}/processed/val.csv', 'wb') as f:
    f.write(val_data)                                                      2

with s3.open(f'{data_bucket}/{subfolder}/processed/test.csv', 'wb') as f:
    f.write(test_data)                                                     3

train_input = sagemaker.s3_input(
    s3_data=f's3://{data_bucket}/{subfolder}/processed/train.csv',
    content_type='csv')
val_input = sagemaker.s3_input(
    s3_data=f's3://{data_bucket}/{subfolder}/processed/val.csv',
    content_type='csv')
```

* _1_ Writes the train.csv file to S3
* _2_ Writes the val.csv file to S3
* _3_ Writes the test.csv file to S3

[Figure 3.7](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03fig07) shows the datasets you now have in S3.

**Figure 3.7. CSV file listing for the S3 folder**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch03fig07.jpg)

**3.6.4. Part 4: Training the model**

Now you train the model. In [chapter 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02), we didn’t go into much detail about what is happening with the training. Now that you have a better understanding of XGBoost, we’ll explain the process a bit more.

The interesting parts of the following listing ([listing 3.8](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03ex8)) are the estimator hyperparameters. We’ll discuss max\_depth and subsample in a later chapter. For now, the hyperparameters of interest to us are

* _objective_—As in [chapter 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02), you set this hyperparameter to binary:logistic. You use this setting when your target variable is 1 or 0. If your target variable is a multiclass variable or a continuous variable, then you use other settings, as we’ll discuss in later chapters.
* _eval\_metric_—The evaluation metric you are optimizing for. The metric argument auc stands for area under the curve, as discussed by Richie earlier in the chapter.
* _num\_round_—How many times you want to let the machine learning model run through the training data (the number of rounds). With each loop through the data, the function gets better at separating the dark circles from the light circles, for example (to refer back to the explanation of machine learning in [chapter 1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_010.html#ch01)). After a while though, the model gets too good; it begins to find patterns in the test data that are not reflected in the real world. This is called _overfitting_. The larger the number of rounds, the more likely you are to be overfitting. To avoid this, you set early stopping rounds.
* _early\_stopping\_rounds_—The number of rounds where the algorithm fails to improve.
* _scale\_pos\_weight_—The scale positive weight is used with imbalanced datasets to make sure the model puts enough emphasis on correctly predicting rare classes during training. In the current dataset, about 1 in 17 customers will churn. So we set scale\_pos\_weight to 17 to accommodate for this imbalance. This tells XGBoost to focus more on customers who actually churn rather than on happy customers who are still happy.

**Note**

If you have the time and interest, try retraining your model without setting scale\_pos\_weight and see what effect this has on your results.

**Listing 3.8. Training the model**

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
    max_depth=3,
    subsample=0.7,
    objective='binary:logistic',                      3
    eval_metric='auc',                                4
    num_round=100,                                    5
    early_stopping_rounds=10,                         6
    scale_pos_weight=17)                              7

estimator.fit({'train': train_input, 'validation': val_input})
```

* _1_ Sets the type of server that SageMaker uses to run your model
* _2_ Stores the output of the model at this location in S3
* _3_ Binary logistic objective hyperparameter
* _4_ Area under the curve (AUC) evaluation metric hyperparameter
* _5_ Number of rounds hyperparameter
* _6_ Early stopping rounds hyperparameter
* _7_ Scale positive weight hyperparameter

When you ran this cell in the current chapter (and in [chapter 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02)), you saw a number of rows of red notifications pop up in the notebook. We passed over this without comment, but these actually contain some interesting information. In particular, you can see if the model is overfitting by looking at this data.

Richie’s explanation of overfitting

We touched on overfitting earlier in the XGBoost explanation. _Overfitting_ is the process of building a model that maps too closely or exactly to the provided training data and fails to predict unseen data as accurately or reliably. This is also sometimes known as _a model that does not generalize well_. Unseen data includes test data, validation data, and data that can be provided to our endpoint in production.

When you run the training, the model does a couple of things in each round of training. First, it trains, and second, it validates. The red notifications that you see are the result of that validation process. As you read through the notifications, you can see that the validation score improves for the first 48 rounds and then starts getting worse.

What you are seeing is _overfitting_. The algorithm is improving at building a function that separates the dark circles from the light circles in the training set (as in [chapter 1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_010.html#ch01)), but it is getting worse at doing it in the validation dataset. This means the model is starting to pick up patterns in the test data that do not exist in the real world (or at least in our validation dataset).

One of the great features in XGBoost is that it deftly handles overfitting for you. The early\_stopping\_rounds hyperparameter in [listing 3.8](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03ex8) stops the training when there’s no improvement in the past 10 rounds.

The output shown in [listing 3.9](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03ex9) is taken from the output of the Train the Model cell in the notebook. You can see that round 15 had an AUC of 0.976057 and that round 16 had an AUC of 0.975683, and that neither of these is better than the previous best of 0.980493 from round 6. Because we set early\_stopping\_rounds=10, the training stops at round 16, which is 10 rounds past the best result in round 6.

**Listing 3.9. Training rounds output**

```
[15]#011train-auc:0.98571#011validation-auc:0.976057
[16]#011train-auc:0.986562#011validation-auc:0.975683
Stopping. Best iteration:
[6]#011train-auc:0.97752#011validation-auc:0.980493
```

**3.6.5. Part 5: Hosting the model**

Now that you have a trained model, you can host it on SageMaker so it is ready to make decisions ([listing 3.10](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03ex10)). We’ve covered a lot of ground in this chapter, so we’ll delve into how the hosting works in a subsequent chapter. For now, just know that it is setting up a server that receives data and returns decisions.

**Listing 3.10. Hosting the model**

```
endpoint_name = 'customer-churn'

try:
    sess.delete_endpoint(
        sagemaker.predictor.RealTimePredictor(
            endpoint=endpoint_name).endpoint)
    print(
        'Warning: Existing endpoint deleted to make way for new endpoint.')
except:
    pass

predictor = estimator.deploy(initial_instance_count=1,
               instance_type='ml.t2.medium',           1
               endpoint_name=endpoint_name)

from sagemaker.predictor import csv_serializer, json_serializer
predictor.content_type = 'text/csv'
predictor.serializer = csv_serializer
predictor.deserializer = None
```

* _1_ Indicates the server type (in this case, an ml.t2.medium server)

**3.6.6. Part 6: Testing the model**

Now that the endpoint is set up and hosted, you can start making decisions. Start by running your test data through the system to see how the model works on data it hasn’t seen before.

The first three lines in [listing 3.11](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03ex11) create a function that returns 1 if the customer is more likely to churn and 0 if they are less likely to churn. The next two lines open the test CSV file you created in [listing 3.7](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03ex7). And the last two lines apply the get\_prediction function to every row in the test dataset to display the data.

**Listing 3.11. Making predictions using the test data**

```
def get_prediction(row):
    prob = float(predictor.predict(row[1:]).decode('utf-8'))
    return 1 if prob > 0.5 else 0                             1

with s3.open(f'{data_bucket}/{subfolder}/processed/test.csv') as f:
    test_data = pd.read_csv(f)

test_data['decison'] = test_data.apply(get_prediction, axis=1)
test_data.set_index('decision', inplace=True)
test_data[:10]ple>
```

* _1_ Returns a value between 0 and 1

In your results, you want to only show a 1 or 0. If the prediction is greater than 0.5 (if prob > 0.5), get\_prediction sets the prediction as 1. Otherwise, it sets the prediction as 0.

The results look pretty good ([table 3.6](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03table06)). Every row that has a 1 in the churn column also has a 1 in the decision column. There are some rows with a 1 in the decision column and a 0 in the churn column, which means Carlos is going to call them even though they are not at risk of churning. But this is acceptable to Carlos. Far better to call a customer that’s not going to churn than to not call a customer that will churn.

**Table 3.6. Results of the test**

| decision | churned | total\_spend | week\_minus\_4 | week\_minus\_3 | week\_minus\_2 | last\_week | 4-3\_delta | 3-2\_delta | 2-1\_delta |
| -------- | ------- | ------------ | -------------- | -------------- | -------------- | ---------- | ---------- | ---------- | ---------- |
| 0        | 0       | 17175.67     | 1.47           | 0.61           | 1.86           | 1.53       | 0.86       | –1.25      | 0.33       |
| 0        | 0       | 68881.33     | 0.82           | 2.26           | 1.59           | 1.72       | –1.44      | 0.67       | -0.13      |
| …        | …       | …            | …              | …              | …              | …          | …          | …          | …          |
| 1        | 1       | 71528.99     | 2.48           | 1.36           | 0.09           | 1.24       | 1.12       | 1.27       | –1.15      |

To see how the model performs overall, you can look at how many customers churned in the test dataset compared to how many Carlos would have called. To do this, you use the value\_counts function as shown in the next listing.

**Listing 3.12. Checking the predictions made using the test data**

```
print(test_data['churned'].value_counts())        1
print(test_data['prediction'].value_counts())     2
print(
    metrics.accuracy_score(
        test_data['churned'],
        test_data['prediction']))                 3
```

* _1_ Counts the number of customers who churned
* _2_ Counts the number of customers who did not churn
* _3_ Calculates the accuracy of the prediction

The value\_counts function shows that Carlos would have called 33 customers and that, if he did nothing, 17 would have churned. But this isn’t very helpful for two reasons:

* What this tells us is that 94.67% of our predictions are correct, but that’s not as good as it sounds because only about 6% of Carlos’s customers churned. If we were to guess that none of our customers churned, we would be 94% accurate.
* It doesn’t tell us how many of those he called would have churned.

For this, you need to create a confusion matrix:

```
0    283
1     17
Name: churned, dtype: int64
0    267
1     33
Name: prediction, dtype: int64
0.94.67
```

A confusion matrix is one of the most confusingly named terms in machine learning. But, because it is also one of the most helpful tools in understanding the performance of a model, we’ll cover it here.

Although the term is confusing, creating a confusion matrix is easy. You use an sklearn function as shown in the next listing.

**Listing 3.13. Creating the confusion matrix**

```
print(
    metrics.confusion_matrix(       1
        test_data['churned'],
        test_data['prediction']))
```

* _1_ Creates the confusion matrix

A _confusion matrix_ is a table containing an equal number of rows and columns. The number of rows and columns corresponds to the number of possible values (classes) for the target variable. In Carlos’s dataset, the target variable could be a 0 or a 1, so the confusion matrix has two rows and two columns. In a more general sense though, the rows of the matrix represent the actual class, and the columns represent the predicted class. (Note: Wikipedia currently has rows and columns reversed to this explanation; however, our description is the way the sklearn.confusion\_matrix function works.)

In the following output, the first row represents happy customers (0) and the second row represents churns (1). The left column shows predicted happy customers, and the right column displays predicted churns. For Carlos, the right column also shows how many customers he called. You can see that Carlos called 16 customers who did not churn and 17 customers who did churn.

```
[[267   16]
[  0  17]]
```

Importantly, the 0 at the bottom left shows how many customers who churned were predicted not to churn and that he did not call. To Carlos’s great satisfaction, that number is 0.

Richie’s note on interpretable machine learning

Throughout this book, we focus on providing examples of business problems that can be solved by machine learning using one of several algorithms. We also attempt to explain in high-level terms how these algorithms work. Generally, we use fairly simple metrics, such as accuracy, to indicate whether a model is working or not. But what if you were asked to explain why your model worked?

Which of your features really matter most in determining whether the model works, and why? For example, is the model biased in ways that can harm minority groups in your customer base or workforce? Questions like this are becoming increasingly prevalent, particularly due to the widespread use of neural networks, which are particularly opaque.

One advantage of XGBoost over neural networks (that we have not previously touched on) is that XGBoost supports the examination of feature importances to help address the explainability issue. At the time of writing, Amazon does not support this directly in the SageMaker XGBoost API; however, the model is stored on S3 as an archive named model.tar.gz. By accessing this file, we can view feature importances. The following listing provides sample code on how to do this.

**Listing 3.14. Sample code used to access SageMaker’s XGBoost model.tar.gz**

```
model_path = f'{estimator.output_path}/\
{estimator._current_job_name}/output/model.tar.gz'
s3.get(model_path, 'xgb_tar.gz')
with tarfile.open('xgb_tar.gz') as tar:
    with tar.extractfile('xgboost-model') as m:S
        xgb_model = pickle.load(m)

xgb_scores = xgb_model.get_score()
print(xgb_scores)>
```

Note that we do not include this code in the notebook as it is beyond the scope of what we want to cover here. But for those of you who want to dive deeper, you can do so using this code, or for more details, see the XGBoost documentation at

[https://xgboost.readthedocs.io/en/latest/python/python\_api.html](https://xgboost.readthedocs.io/en/latest/python/python\_api.html).

#### 3.7. Deleting the endpoint and shutting down your notebook instance <a href="#ch03lev1sec7__title" id="ch03lev1sec7__title"></a>

It is important that you shut down your notebook instance and delete your endpoint. We don’t want you to get charged for SageMaker services that you’re not using.

**3.7.1. Deleting the endpoint**

Appendix D describes how to shut down your notebook instance and delete your endpoint using the SageMaker console, or you can do that with the code in the next listing.

**Listing 3.15. Deleting the notebook**

```
# Remove the endpoint (optional)
# Comment out this cell if you want the endpoint to persist after Run All
sess.delete_endpoint(text_classifier.endpoint)
```

To delete the endpoint, uncomment the code in the listing, then click ![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ctrl\_ent.jpg) to run the code in the cell.

**3.7.2. Shutting down the notebook instance**

To shut down the notebook, go back to your browser tab where you have SageMaker open. Click the Notebook Instances menu item to view all of your notebook instances. Select the radio button next to the notebook instance name, as shown in [figure 3.8](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03fig08), then click Stop on the Actions menu. It will take a couple of minutes to shut down.

**Figure 3.8. Shutting down the notebook**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch03fig08\_alt.jpg)

#### 3.8. Checking to make sure the endpoint is deleted <a href="#ch03lev1sec8__title" id="ch03lev1sec8__title"></a>

If you didn’t delete the endpoint using the notebook (or if you just want to make sure it’s deleted), you can do this from the SageMaker console. To delete the endpoint, click the radio button to the left of the endpoint name, then click the Actions menu item and click Delete in the menu that appears.

When you have successfully deleted the endpoint, you will no longer incur AWS charges for it. You can confirm that all of your endpoints have been deleted when you see the text “There are currently no resources” displayed at the bottom of the Endpoints page ([figure 3.9](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03fig09)).

**Figure 3.9. Verifying that you have successfully deleted the endpoint**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch03fig09\_alt.jpg)

#### Summary <a href="#ch03lev1sec9__title" id="ch03lev1sec9__title"></a>

* You created a machine learning model to determine which customers to call because they are at risk of taking their business to a competitor.
* XGBoost is a gradient-boosting, machine learning model that uses an ensemble of different approaches to improve the effectiveness of its learning.
* Stratify is one technique to help you handle imbalanced datasets. It shuffles the deck as it builds the machine learning model, making sure that the train, validate, and test datasets contain similar ratios of target variables.
* A confusion matrix is one of the most confusingly named terms in machine learning, but it is also one of the most helpful tools in understanding the performance of a model.
