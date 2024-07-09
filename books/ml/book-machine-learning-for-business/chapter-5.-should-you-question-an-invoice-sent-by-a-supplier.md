# Chapter 5. Should you question an invoice sent by a supplier?

_This chapter covers_

* What’s the real question you’re trying to answer?
* A machine learning scenario without trained data
* The difference between supervised and unsupervised machine learning
* Taking a deep dive into anomaly detection
* Using the Random Cut Forest algorithm

Brett works as a lawyer for a large bank. He is responsible for checking that the law firms hired by the bank bill the bank correctly. How tough can this be, you ask? Pretty tough is the answer. Last year, Brett’s bank used hundreds of different firms across thousands of different legal matters, and each invoice submitted by a firm contains dozens or hundreds of lines. Tracking this using spreadsheets is a nightmare.

In this chapter, you’ll use SageMaker and the Random Cut Forest algorithm to create a model that highlights the invoice lines that Brett should query with a law firm. Brett can then apply this process to every invoice to keep the lawyers working for his bank on their toes, saving the bank hundreds of thousands of dollars per year. Off we go!

#### 5.1. What are you making decisions about? <a href="#ch05lev1sec1__title" id="ch05lev1sec1__title"></a>

As always, the first thing we want to look at is what we’re making decisions about. In this chapter, at first glance, it appears the question Brett must decide is, should this invoice line be looked at more closely to determine if the law firm is billing us correctly? But, if you build a machine learning algorithm to definitively answer that question correctly 100% of the time, you’ll almost certainly fail. Fortunately for you and Brett, that is not the real question you are trying to answer.

To understand the true value that Brett brings to the bank, let’s look at his process. Prior to Brett and his team performing their functions, the bank found that law firm costs were spiraling out of control. The approach that Brett’s team has taken over the past few years is to manually review each invoice and use gut instinct to determine whether they should query costs with the law firm. When Brett reads an invoice, he usually gets a pretty good feel for whether the costs are in line with the type of case the law firm is working on. He can tell pretty accurately whether the firm has billed an unusually high number of hours from partners rather than junior lawyers on the case, or whether a firm seems to be padding the number of hours their paralegals are spending on a matter.

When Brett comes across an apparent anomaly, an invoice that he feels has incorrect charges, he contacts the law firm and requests that they provide further information on their fees. The law firm responds in one of two ways:

* They provide additional information to justify their fees.
* They reduce their fees to an amount that is more in line with a typical matter of this type.

It is important to note that Brett really doesn’t have a lot of clout in this relationship. If his bank instructs a law firm to work on a case, and they say that a particular piece of research took 5 hours of paralegal time, there is little Brett can do to dispute that. Brett can say that it seems like a lot of time. But the law firm can respond with, “Well, that’s how long it took,” and Brett has to accept that.

But this way of looking at Brett’s job is too restrictive. The interesting thing about Brett’s job is that Brett is effective not because he can identify 100% of the invoice lines that should be queried, but because the law firms know that Brett is pretty good at picking up anomalies. So, if the law firms charge more than they would normally charge for a particular type of service, they know they need to be prepared to justify it.

Lawyers really dislike justifying their costs, not because they can’t, but because it takes time that they’d rather spend billing other clients. Consequently, when lawyers prepare their timesheets, if they know that there is a good chance that a line has more time on it than can easily be justified, they will weigh whether they should adjust their time downward or not. This decision, multiplied over the thousands of lines billed to the bank each year, results in hundreds of thousands of dollars in savings for the bank.

The real question you are trying to answer in this scenario is, what invoice lines does Brett need to query to encourage the law firms to bill the bank correctly?

And this question is fundamentally different from the original question about how to accurately determine which line is an anomaly. If you are trying to correctly identify anomalies, your success is determined by your accuracy. However, in this case, if you are simply trying to identify enough anomalies to encourage the law firms to bill the bank correctly, then your success is determined by how efficiently you can hit the threshold of _enough anomalies_.

What percentage of anomalies is enough anomalies?

A great deal of time and effort can be expended answering this question accurately. If a lawyer knew that one out every thousand anomalous lines would be queried, their behavior might not change at all. But if they knew that 9 out of 10 anomalous lines would be queried, then they would probably prepare their timesheets with a little more consideration.

In an academic paper, you want to clearly identify this threshold. In the business world, you need to weigh the benefits of accuracy against the cost of not being able to work on another project because you are spending time identifying a threshold. In Brett’s case, it is probably sufficient to compare the results of the algorithm against how well a member of Brett’s team can perform the task. If it comes out about the same, then you’ve hit the threshold.

#### 5.2. The process flow <a href="#ch05lev1sec2__title" id="ch05lev1sec2__title"></a>

The process flow for this decision is shown in [figure 5.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05fig01). It starts when a lawyer creates an invoice and sends it to Brett (1). On receiving the invoice, Brett or a member of his team reviews the invoice (2) and then does one of two things, depending on whether the fees listed in the invoice seem reasonable:

**Figure 5.1. Current workflow showing Brett’s process for reviewing invoices received from lawyers**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch05fig01.jpg)

* The invoice is passed on to Accounts Payable for payment (3).
* The invoice is sent back to the lawyer with a request for clarification of some of the charges (4).

With thousands of invoices to review annually, this is a full-time job for Brett and his two staff members.

[Figure 5.2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05fig02) shows the new workflow after you implement the machine learning application you’ll build in this chapter. When the lawyer sends the invoice (1), instead of Brett or his team reviewing the invoice, it is passed through a machine learning model that determines whether the invoice contains any anomalies (2). If there are no anomalies, the invoice is passed through to Accounts Payable without further review by Brett’s team (3). If an anomaly is detected, the application sends the invoice back to the lawyer and requests further information on the fees charged (4). The role Brett plays in this process is to review a certain number of these transactions to ensure the system is functioning as designed (5).

**Figure 5.2. New workflow after implementing machine learning app to catch anomalies in invoices**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch05fig02.jpg)

Now that Brett is not required to review invoices he is able to spend more time on other aspects of his role such as maintaining and improving relationships with suppliers.

#### 5.3. Preparing the dataset <a href="#ch05lev1sec3__title" id="ch05lev1sec3__title"></a>

The dataset you are using in this chapter is a synthetic dataset created by Richie. It contains 100,000 rows of invoice line data from law firms retained by Brett’s bank.

Synthetic data vs. real data

Synthetic data is data created by you, the analyst, as opposed to data found in the real world. When you are working with data from your own company, your data will be real data rather than synthetic data.

A good set of real data is more fun to work with than synthetic data because it is typically more nuanced than synthetic data. With real data, there are interesting patterns you can find in the data that you weren’t expecting to see. Synthetic data, on the other hand, is great in that it shows exactly the concept you want to show, but it lacks the element of surprise and the joy of discovery that working with real data provides.

In [chapters 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02) and [3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03), you worked with synthetic data (purchase order data and customer churn data). In [chapter 4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04), you worked with real data (tweets to customer support teams). In [chapter 6](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06), you’ll be back to working with real data (electricity usage data).

Law firm invoices are usually quite detailed and show how many minutes the firm spent performing each task. Law firms typically have a stratified fee structure, where junior lawyers and paralegals (staff who perform work that doesn’t need to be performed by a qualified lawyer) are billed at a lower cost than senior lawyers and law firm partners. The important information on law firm invoices is the type of material worked on (antitrust, for example), the resource that performed the work (paralegal, junior lawyer, partner, and so on), how many minutes were spent on the activity, and how much it cost. The dataset you’ll use in this chapter contains the following columns:

* _Matter Number_—An identifier for each invoice. If two lines have the same matter number, it means that they are on the same invoice.
* _Firm Name_—The name of the law firm.
* _Matter Type_—The type of matter the invoice relates to.
* _Resource_—The type of resource that performs the activity.
* _Activity_—The type of activity performed by the resource.
* _Minutes_—How many minutes it took to perform the activity.
* _Fee_—The hourly rate for the resource.
* _Total_—The total fee.
* _Error_—A column indicating whether the invoice line contains an error. Note that this column exists in this dataset to allow you to determine how successful the model was at picking the lines with errors. In a real-life dataset, you wouldn’t have this field.

[Table 5.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05table01) shows three invoice lines in the dataset.

**Table 5.1. Dataset invoice lines for the lawyers submitting invoices to the bank**

| Matter Number | Firm Name | Matter Type | Resource  | Activity       | Minutes | Fee | Total   | Error |
| ------------- | --------- | ----------- | --------- | -------------- | ------- | --- | ------- | ----- |
| 0             | Cox Group | Antitrust   | Paralegal | Attend Court   | 110     | 50  | 91.67   | False |
| 0             | Cox Group | Antitrust   | Junior    | Attend Court   | 505     | 150 | 1262.50 | True  |
| 0             | Cox Group | Antitrust   | Paralegal | Attend Meeting | 60      | 50  | 50.00   | False |

In this chapter, you’ll build a machine learning application to pick the lines that contain errors. In machine learning lingo, you are identifying anomalies in the data.

#### 5.4. What are anomalies <a href="#ch05lev1sec4__title" id="ch05lev1sec4__title"></a>

Anomalies are the data points that have something unusual about them. Defining _unusual_ is not always easy. For example, the image in [figure 5.3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05fig03) contains an anomaly that is pretty easy to spot. All the characters in the image are capital _S_'s with the exception of the single number 5.

**Figure 5.3. A simple anomaly. It’s easy to spot the anomaly in this dataset.**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch05fig03.jpg)

**Figure 5.4. A complex anomaly. It’s far more difficult to spot the second anomaly in this dataset.**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch05fig04.jpg)

But what about the image shown in [figure 5.4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05fig04)? The anomaly is less easy to spot.

There are actually two anomalies in this dataset. The first anomaly is similar to the anomaly in [figure 5.3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05fig03). The number 5 in the bottom right of the image is the only number. Every other character is a letter. The last anomaly is difficult: the only characters that appear in pairs are vowels. Admittedly, the last anomaly would be almost impossible for a human to identify but, given enough data, a machine learning algorithm could find it.

Just like the images in [figures 5.3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05fig03) and [5.4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05fig04), Brett’s job is to identify anomalies in the invoices sent to his bank by law firms. Some invoices have anomalies that are easy to find. The invoice might contain a high fee for the resource, such as a law firm charging $500 per hour for a paralegal or junior lawyer, or the invoice might contain a high number of hours for a particular activity, such as a meeting being billed for 360 minutes.

But other anomalies are more difficult to find. For example, antitrust matters might typically involve longer court sessions than insolvency matters. If so, a 500-minute court session for an insolvency matter might be an anomaly, but the same court session for an antitrust matter might not.

One of the challenges you might have noticed in identifying anomalies in [figures 5.3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05fig03) and [5.4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05fig04) is that you did not know what type of anomaly you were looking for. This is not dissimilar to identifying anomalies in real-world data. If you had been told that the anomaly had to do with numbers versus letters, you would have easily identified the 5 in [figures 5.3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05fig03) and [Figure 5.4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05fig04). Brett, who is a trained lawyer and has been reviewing legal invoices for years, can pick out anomalies quickly and easily, but he might not consciously know why he feels that a particular line is an anomaly.

In this chapter, you will not define any rules to help the model determine what lines contain anomalies. In fact, you won’t even tell the model which lines contain anomalies. The model will figure it out for itself. This is called _unsupervised_ machine learning.

#### 5.5. Supervised vs. unsupervised machine learning <a href="#ch05lev1sec5__title" id="ch05lev1sec5__title"></a>

In the example you are working through in this chapter, you could have had Brett label the invoice lines he would normally query and use that to train an XGBoost model in a manner similar to the XGBoost models you trained in [chapters 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02) and [3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03). But what if you didn’t have Brett working for you? Could you still use machine learning to tackle this problem? It turns out you can.

The machine learning application in this chapter uses an unsupervised algorithm called Random Cut Forest to determine whether an invoice should be queried. The difference between a supervised algorithm and an unsupervised algorithm is that with an _unsupervised_ algorithm, you don’t provide any labeled data. You just provide the data and the algorithm decides how to interpret it.

In [chapters 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02), [3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03), and [4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04), the machine learning algorithms you used were _supervised_. In this chapter, the algorithm you will use is unsupervised. In [chapter 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02), your dataset had a column called tech\_approval\_required that the model used to learn whether technical approval was required. In [chapter 3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03), your dataset had a column called churned that the model used to learn whether a customer churned or not. In [chapter 4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04), your dataset had a column called escalate to learn whether a particular tweet should be escalated.

In this chapter, you are not going to tell the model which invoices should be queried. Instead, you are going to let the algorithm figure out which invoices contain anomalies, and you will query the invoices that have anomalies over a certain threshold. This is unsupervised machine learning.

#### 5.6. What is Random Cut Forest and how does it work? <a href="#ch05lev1sec6__title" id="ch05lev1sec6__title"></a>

The machine learning algorithm you’ll use in this chapter, Random Cut Forest, is a wonderfully descriptive name because the algorithm takes random data points (Random), cuts them to the same number of points and creates trees (Cut). It then looks at all of the trees together (Forest) to determine whether a particular data point is an anomaly—hence, _Random Cut Forest_.

A tree is an ordered way of storing numerical data. The simplest type of tree is called a _binary tree_. It’s a great way to store data because it’s easy and fast for a computer to work with. To create a tree, you randomly subdivide the data points until you have isolated the point you are testing to determine whether it is an anomaly. Each time you subdivide the data points, it creates a new level of the tree. The fewer times you need to subdivide the data points before you isolate the target data point, the more likely the data point is to be an anomaly for that sample of data.

In the two sections that follow, you’ll look at two examples of trees with a target data point injected. In the first sample, the target data point will appear to be an anomaly. In the second sample, the target data point will not be an anomaly. When you look at the samples together as a forest, you’ll see that the latter point is not likely to be an anomaly.

**5.6.1. Sample 1**

[Figure 5.5](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05fig05) shows six dark dots that represent six data points that have been pulled at random from the dataset. The white dot represents the target data point that you are testing to determine whether it is an anomaly. Visually, you can see that this white dot sits somewhat apart from the other values in this sample of data, so it might be an anomaly. But how do you determine this algorithmically? This is where the tree representation comes in.

**Figure 5.5. Sample 1: The white dot represents an anomaly.**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch05fig05.jpg)

[Figure 5.6](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05fig06) shows the top level of the tree. The top level is a single node that represents all of the data points in the sample (including the target data point you are testing). If the node contains any data points other than the target point you are testing for, the color of the node is shown as dark. (The top-level node is always dark because it represents all of the data points in the sample.)

**Figure 5.6. Sample 1: Level-1 tree represents a node with all of the data points in one group.**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch05fig06.jpg)

[Figure 5.7](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05fig07) shows the data points after the first subdivision. The dividing line is inserted at random through the data points. Each side of the subdivision represents a node in the tree.

**Figure 5.7. Sample 1: Level-2 data points divided between two nodes after the first subdivision.**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch05fig07.jpg)

[Figure 5.8](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05fig08) shows the next level of the tree. The left side of [figure 5.7](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05fig07) becomes Node B on the left of the tree. The right side of [figure 5.7](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05fig07) becomes Node C on the right of the tree. Both nodes in the tree are shown as dark because both sides of the subdivided diagram in [figure 5.7](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05fig07) contain at least one dark dot.

**Figure 5.8. Sample 1: Level-2 tree represents the data points split into two groups, where both nodes are shown as dark.**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch05fig08.jpg)

The next step is to further subdivide the part of the diagram that contains the target data point. This is shown in [figure 5.9](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05fig09). You can see that Node C on the right is untouched, whereas the left side is subdivided into Nodes D and E. Node E contains only the target data point, so no further subdivision is required.

**Figure 5.9. Sample 1: Level-3 data points separate the target data point from the values in the dataset.**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch05fig09.jpg)

[Figure 5.10](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05fig10) shows the final tree. Node E is shown in white because it contains the target data point. The tree has three levels. The smaller the tree, the greater the likelihood that the point is an anomaly. A three-level tree is a pretty small tree, indicating that the target data point might be an anomaly.

**Figure 5.10. Sample 1: Level-3 tree represents one of the level-2 groups split again to isolate the target data point.**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch05fig10.jpg)

Now, let’s take a look at another sample of six data points that are clustered more closely around the target data point.

**5.6.2. Sample 2**

In the second data sample, the randomly selected data points are clustered more closely around the target data point. It is important to note that our target data point is the same data point that was used in sample 1. The only difference is that a different sample of data points was drawn from the dataset. You can see in [figure 5.11](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05fig11) that the data points in the sample (dark dots) are more closely clustered around the target data point than they were in sample 1.

**Figure 5.11. Sample 2: Level-1 data points and tree represent all of the data points in a single group.**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch05fig11.jpg)

**Note**

In [figure 5.11](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05fig11) and the following figures in this section, the tree is displayed below the diagram of the data points.

Just as in sample 1, [figure 5.12](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05fig12) splits the diagram into two sections, which we have labeled B and C. Because both sections contain dark dots, level 2 of the tree diagram is shown as dark.

**Figure 5.12. Sample 2: Level-2 data points and tree represent the level-1 groups split into two groups.**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch05fig12.jpg)

Next, the section containing the target data point is split again. [Figure 5.13](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05fig13) shows that section B has been split into two sections labeled D and E, and a new level has been added to the tree. Both of these sections contain one or more dark dots, so level 3 of the tree diagram is shown as dark.

**Figure 5.13. Sample 2: Level-3 data points and tree represent one of the level-2 groups split into two groups.**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch05fig13.jpg)

The target data point is in section E, so that section is split into two sections labeled F and G as shown in [figure 5.14](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05fig14).

**Figure 5.14. Sample 2: Level-4 data points and tree represent one of the level-3 groups split into two groups.**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch05fig14.jpg)

The target data point is in section F, so that section is split into two sections labeled H and J as shown in [figure 5.15](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05fig15). Section J contains only the target data point, so it is shown as white. No further splitting is required. The resulting diagram has 5 levels, which indicates that the target data point is not likely to be an anomaly.

**Figure 5.15. Sample 2: Level-5 data points and tree represent one of the level-4 groups split into two groups, isolating the target data point.**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch05fig15.jpg)

The final step performed by the Random Cut Forest algorithm is to combine the trees into a forest. If lots of the samples have very small trees, then the target data point is likely to be an anomaly. If only a few of the samples have small trees, then it is likely to _not_ be an anomaly.

You can read more about Random Cut Forest on the AWS site at [https://docs.aws.amazon.com/sagemaker/latest/dg/randomcutforest.html](https://docs.aws.amazon.com/sagemaker/latest/dg/randomcutforest.html).

Richie’s explanation of the forest part of Random Cut Forest

Random Cut Forest partitions the dataset into the number of trees in the forest (specified by the num\_trees hyperparameter). During training, a total of num\_trees × num\_samples\_per\_tree individual data points get sampled from the full dataset without replacement. For a small dataset, this can be equal to the total number of observations, but for large datasets, it need not be.

During inference, however, a brand new data point gets assigned an anomaly score by cycling through all the trees in the forest and determining what anomaly score to give it from each tree. This score then gets averaged to determine if this point should actually be considered an anomaly or not.

#### 5.7. Getting ready to build the model <a href="#ch05lev1sec7__title" id="ch05lev1sec7__title"></a>

Now that you have a deeper understanding of how Random Cut Forest works, you can set up another notebook on SageMaker and make some decisions. As you did in [chapters 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02), [3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03), and [4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04), you are going to do the following:

1. Upload a dataset to S3
2. Set up a notebook on SageMaker
3. Upload the starting notebook
4. Run it against the data

**Tip**

If you’re jumping into the book at this chapter, you might want to visit the appendixes, which show you how to do the following:

* [Appendix A](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_021.html#app01): sign up for AWS, Amazon’s web service
* [appendix B](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_024.html#app02): set up S3, AWS’s file storage service
* [Appendix C](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_028.html#app03): set up SageMaker

**5.7.1. Uploading a dataset to S3**

To set up the dataset for this chapter, you’ll follow the same steps as you did in [appendix B](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_024.html#app02). You don’t need to set up another bucket though. You can go to the same bucket you created earlier. In our example, we called it _mlforbusiness_, but your bucket will be called something different.

When you go to your S3 account, you will see the bucket you created to hold the data files for previous chapters. Click this bucket to see the ch02, ch03, and ch04 folders you created in previous chapters. For this chapter, you’ll create a new folder called _ch05_. You do this by clicking Create Folder and following the prompts to create a new folder.

Once you’ve created the folder, you are returned to the folder list inside your bucket. There you’ll see you now have a folder called ch05. Now that you have the ch05 folder set up in your bucket, you can upload your data file and start setting up the decision-making model in SageMaker. To do so, click the folder and download the data file at this link:

[https://s3.amazonaws.com/mlforbusiness/ch05/activities.csv](https://s3.amazonaws.com/mlforbusiness/ch05/activities.csv).

Then upload the CSV file into your ch05 folder by clicking Upload. Now you’re ready to set up the notebook instance.

**5.7.2. Setting up a notebook on SageMaker**

Like you did in [chapters 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02), [3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03), and [4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04), you’ll set up a notebook on SageMaker. If you skipped the earlier chapters, follow the instructions in [appendix C](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_028.html#app03) on how to set up SageMaker.

When you go to SageMaker, you’ll see your notebook instances. The notebook instance you created for earlier chapters (or that you’ve just created by following the instructions in [appendix C](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_028.html#app03)) will either say Open or Start. If it says Start, click the Start link and wait a couple of minutes for SageMaker to start. Once the screen displays Open Jupyter, click the Open Jupyter link to open up your notebook list.

Once it opens, create a new folder for [chapter 5](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05) by clicking New and selecting Folder at the bottom of the dropdown list. This creates a new folder called Untitled Folder. When you tick the checkbox next to Untitled Folder, you will see the Rename button appear. Click it and change the folder name to ch05. Click the ch05 folder, and you will see an empty notebook list.

Just as we already prepared the CSV data you uploaded to S3 (activities.csv), we’ve already prepared the Jupyter notebook you’ll now use. You can download it to your computer by navigating to this URL:

[https://s3.amazonaws.com/mlforbusiness/ch05/detect\_suspicious\_lines.ipynb](https://s3.amazonaws.com/mlforbusiness/ch05/detect\_suspicious\_lines.ipynb).

Click Upload to upload the detect\_suspicious\_lines.ipynb notebook to the ch05 folder. After uploading the file, you’ll see the notebook in your list. Click it to open it. Now, just like in the previous chapters, you are a few keystrokes away from being able to run your machine learning model.

#### 5.8. Building the model <a href="#ch05lev1sec8__title" id="ch05lev1sec8__title"></a>

As in the previous chapters, you will go through the code in six parts:

1. Load and examine the data.
2. Get the data into the right shape.
3. Create training and validation datasets (there’s no need for a test dataset in this example).
4. Train the machine learning model.
5. Host the machine learning model.
6. Test the model and use it to make decisions

Refresher on running code in Jupyter notebooks

SageMaker uses Jupyter Notebook as its interface. Jupyter Notebook is an open-source data science application that allows you to mix code with text. As shown in the figure, the code sections of a Jupyter notebook have a gray background, and the text sections have a white background.

**Sample Jupyter notebook showing text and code cells**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/pg116fig01\_alt.jpg)

To run the code in the notebook, click a code cell and press ![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ctrl\_ent.jpg). To run the notebook, you can select Run All from the Cell menu item at the top of the notebook. When you run the notebook, SageMaker loads the data, trains the model, sets up the endpoint, and generates decisions from the test data.

**5.8.1. Part 1: Loading and examining the data**

As in the previous three chapters, the first step is to say where you are storing the data. In [listing 5.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05ex1), you need to change 'mlforbusiness' to the name of the bucket you created when you uploaded the data, then change the subfolder to the name of the subfolder on S3 where you want to store the data. If you named the S3 folder ch05, then you don’t need to change the name of the folder. If you kept the name of the CSV file you uploaded earlier in the chapter, then you don’t need to change the activities.csv line of code either. If you renamed the CSV file, then you need to update the filename with the name you changed it to. To run the code in the notebook cell, click the cell and press ![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ctrl\_ent.jpg).

**Listing 5.1. Say where you are storing the data**

```
data_bucket = 'mlforbusiness'    1
subfolder = 'ch05'               2
dataset = 'activities.csv'       3
```

* _1_ S3 bucket where the data is stored
* _2_ Subfolder of the S3 bucket where the data is stored
* _3_ Dataset that’s used to train and test the model

Next you’ll import all of the Python libraries and modules that SageMaker uses to prepare the data, train the machine learning model, and set up the endpoint. The Python modules and libraries imported in [listing 5.2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05ex2) are the same as the imports you used in previous chapters.

**Listing 5.2. Importing the modules**

```
import pandas as pd                       1
import boto3                              2
import s3fs                               3
import sagemaker                          4
from sklearn.model_selection \
    import train_test_split               5
import json                               6
import csv                                7

role = sagemaker.get_execution_role()     8
s3 = s3fs.S3FileSystem(anon=False)        9
```

* _1_ Imports the pandas Python library
* _2_ Imports the boto3 AWS library
* _3_ Imports the s3fs module to make working with S3 files easier
* _4_ Imports SageMaker
* _5_ Imports only the train\_test\_split module from the sklearn library
* _6_ Imports the json module to work with JSON files
* _7_ Imports the csv module to work with comma separated files
* _8_ Creates a role on SageMaker
* _9_ Establishes the connection with S3

The dataset contains invoice lines from all matters handled by your panel of lawyers over the past 3 months. The dataset has about 100,000 lines covering 2,000 invoices (50 lines per invoice). It contains the following columns:

* _Matter Number_—An identifier for each invoice. If two lines have the same number, it means that these are on the same invoice.
* _Firm Name_—The name of the law firm.
* _Matter Type_—The type of activity the invoice relates to.
* _Resource_—The resource that performs the activity.
* _Activity_—The activity performed by the resource.
* _Minutes_—How many minutes it took to perform the activity.
* _Fee_—The hourly rate for the resource.
* _Total_—The total fee.
* _Error_—Indicates whether the invoice line contains an error.

**Note**

The Error column is not used during training because, in our scenario, this information is not known until you contact the law firm and determine whether the line was in error. This field is included here to allow you to determine how well your model is working.

Next, you’ll load and view the data. In [listing 5.3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05ex3), you read the top 20 rows of the CSV data in activities.csv to display those in a pandas DataFrame. In this listing, you use a different way of displaying rows in the pandas DataFrame. Previously, you used the head() function to display the top 5 rows. In this listing, you use explicit numbers to display specific rows.

**Listing 5.3. Loading and viewing the data**

```
df = pd.read_csv(
     f's3://{data_bucket}/{subfolder}/{dataset}')       1
display(df[5:8])                                        2
```

* _1_ Reads the S3 dataset set in [listing 5.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05ex1)
* _2_ Displays 3 rows of the DataFrame (rows 5, 6, and 7)

In this example, the top 5 rows all show no errors. You can tell if a row shows an error by looking at the rightmost column, Error. Rows 5, 6, and 7 are displayed because they show two rows with Error = False and one row with Error = True. [Table 5.2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05table02) shows the output of running display(df\[5:8]).

**Table 5.2. Dataset invoice lines display the three rows returned from running display(df\[5:8]).**

| Row number | Matter Number | Firm Name | Matter Type | Resource  | Activity       | Minutes | Fee | Total   | Error |
| ---------- | ------------- | --------- | ----------- | --------- | -------------- | ------- | --- | ------- | ----- |
| 5          | 0             | Cox Group | Antitrust   | Paralegal | Attend Court   | 110     | 50  | 91.67   | False |
| 6          | 0             | Cox Group | Antitrust   | Junior    | Attend Court   | 505     | 150 | 1262.50 | True  |
| 7          | 0             | Cox Group | Antitrust   | Paralegal | Attend Meeting | 60      | 50  | 50.00   | False |

In [listing 5.4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05ex4), you use the pandas value\_counts function to determine the error rate. You can see that out of 100,000 rows, about 2,000 have errors, which gives a 2% error rate. Note that in a real-life scenario, you won’t know the error rate, so you would have to run a small project to determine your error rate by sampling lines from invoices.

**Listing 5.4. Displaying the error rate**

```
[id="esc
----
df['Error'].value_counts()    1
----formalexample>
```

* _1_ Displays the error rate: False is no error; True is an error.

The following listing shows the output from the code in [listing 5.4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05ex4).

**Listing 5.5. Total number of tweets and the number of escalated tweets**

```
False    103935
True     2030
Name: escalate, dtype: int64
```

The next listing shows the types of matters, resources, and activities.

**Listing 5.6. Describing the data**

```
print(f'Number of rows in dataset: {df.shape[0]}')
print()
print('Matter types:')
print(df['Matter Type'].value_counts())
print()
print('Resources:')
print(df['Resource'].value_counts())
print()
print('Activities:')
print(df['Activity'].value_counts())
```

The results of the code in [listing 5.6](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05ex6) are shown in [listing 5.7](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05ex7). You can see that there are 10 different matter types, ranging from Antitrust to Securities litigation; four different types of resources, ranging from Paralegal to Partner; and four different activity types, such as Phone Call, Attend Meeting, and Attend Court.

**Listing 5.7. Viewing the data description**

```
Number of rows in dataset: 105965

Matter types:
Antitrust                 23922
Insolvency                16499
IPO                       14236
Commercial arbitration    12927
Project finance           11776
M&A                        6460
Structured finance         5498
Asset recovery             4913
Tax planning               4871
Securities litigation      4863
Name: Matter Type, dtype: int64

Resources:
Partner      26587
Junior       26543
Paralegal    26519
Senior       26316
Name: Resource, dtype: int64
*
Activities:
Prepare Opinion    26605
Phone Call         26586
Attend Court       26405
Attend Meeting     26369
Name: Activity, dtype: int64
```

The machine learning model uses these features to determine which invoice lines are potentially erroneous. In the next section, you’ll work with these features to get them into the right shape for use in the machine learning model.

**5.8.2. Part 2: Getting the data into the right shape**

Now that you’ve loaded the data, you need to get the data into the right shape. This involves several steps:

* Changing the categorical data to numerical data
* Splitting the dataset into training data and validation data
* Removing unnecessary columns

The machine learning algorithm you’ll use in this notebook is the Random Cut Forest algorithm. Just like the XGBoost algorithm you used in [chapters 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02) and [3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03), Random Cut Forest can’t handle text values—everything needs to be a number. And, as you did in [chapters 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02) and [3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03), you’ll use the pandas get\_dummies function to convert each of the different text values in the Matter Type, Resource, and Activity columns and place a 0 or a 1 as the value in the column. For example, the rows shown in the three-column [table 5.3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05table03) would be converted to a four column table.

**Table 5.3. Data before applying the get\_dummies function**

| Matter Number | Matter Type | Resource  |
| ------------- | ----------- | --------- |
| 0             | Antitrust   | Paralegal |
| 0             | Antitrust   | Partner   |

The converted table ([table 5.4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05table04)) has four columns because an additional column gets created for each unique value in any of the columns. Given that there are two different values in the Resource column in [table 5.3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05table03), that column is split into two columns: one for each type of resource.

**Table 5.4. Data after applying the get\_dummies function**

| Matter Number | Matter\_Type\_Antitrust | Resource\_Paralegal | Resource\_Partner |
| ------------- | ----------------------- | ------------------- | ----------------- |
| 0             | 1                       | 1                   | 0                 |
| 0             | 1                       | 0                   | 1                 |

In [listing 5.8](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05ex8), you create a pandas DataFrame called encoded\_df by calling the get\_dummies() function on the original pandas df DataFrame. Calling the head() function here returns the first three rows of the DataFrame.

Note that this can create very wide datasets, as every unique value becomes a column. The DataFrame you work with in this chapter increases from a 9-column table to a 24-column table. To determine how wide your table will be, you need to subtract the number of columns you are applying the get\_dummies function to and add the number of unique elements in each column. So, your original 9-column table becomes a 6-column table once you subtract the 3 columns you apply the get\_dummies function to. Then it expands to a 24-column table once you add 10 columns for each unique element in the Matter Type column and four columns each for the unique elements in the Resource and Activity columns.

**Listing 5.8. Creating the train and validate data**

```
encoded_df = pd.get_dummies(
    df,
    columns=['Matter Type','Resource','Activity'])    1
encoded_df.head(3)                                    2
```

* _1_ Converts three columns into a column for each unique value
* _2_ Displays the top three rows of the DataFrame

**5.8.3. Part 3: Creating training and validation datasets**

You now split the dataset into train and validation data. Note that with this notebook, you don’t have any test data. In a real-world situation, the best way to test the data is often to compare your success at identifying errors _before_ using the machine learning model with your success _after_ you use the machine learning algorithm.

A test size of 0.2 instructs the function to place 80% of the data into a train DataFrame and 20% into a validation DataFrame. If you are splitting a dataset into training and validation data, you typically will place 70% of your data into a training dataset, 20% into test, and 10% into validation. For the dataset in this chapter, you are just splitting the data into training and test datasets as, in Brett’s data, there will be no validation data.

**Listing 5.9. Creating training and validation datasets**

```
train_df, val_df, _, _ = train_test_split(
    encoded_df,
    encoded_df['Error'],
    test_size=0.2,
    random_state=0)                                  1
print(
    f'{train_df.shape[0]} rows in training data')    2
```

* _1_ Creates the training and validation datasets
* _2_ Displays the number of rows in the training data

With that, the data is in a SageMaker session, and you are ready to start training the model.

**5.8.4. Part 4: Training the model**

In [listing 5.10](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05ex10), you import the RandomCutForest function, set up the training parameters, and store the result in a variable called rcf. This all looks very similar to how you set up the training jobs in previous chapters, with the exception of the final two parameters in the RandomCutForest function.

The parameter num\_samples\_per\_tree sets how many samples you include in each tree. Graphically, you can think of it as the number of dark dots per tree. If you have lots of samples per tree, your trees will get very large before the function creates a slice that contains only the target point. Large trees take longer to calculate than small trees. AWS recommends you start with 100 samples per tree, as that provides a good middle ground between speed and size.

The parameter num\_trees is the number of trees (groups of dark dots). This parameter should be set to approximate the fraction of errors expected. In your dataset, about 2% (or 1/50) are errors, so you’ll set the number of trees to 50. The final line of code in the following listing runs the training job and creates the model.

**Listing 5.10. Training the model**

```
from sagemaker import RandomCutForest

session = sagemaker.Session()

rcf = RandomCutForest(role=role,
                      train_instance_count=1,
                      train_instance_type='ml.m4.xlarge',
                      data_location=f's3://{data_bucket}/{subfolder}/',
                      output_path=f's3://{data_bucket}/{subfolder}/output',
                      num_samples_per_tree=100,                            1
                      num_trees=50)                                        2

rcf.fit(rcf.record_set(train_df_no_result.values))
```

* _1_ Number of samples per tree
* _2_ Number of trees

**5.8.5. Part 5: Hosting the model**

Now that you have a trained model, you can host it on SageMaker so it is ready to make decisions. If you have run this notebook already, you might already have an endpoint. To handle this, in the next listing, you delete any existing endpoints you have so you don’t end up paying for a bunch of endpoints you aren’t using.

**Listing 5.11. Hosting the model: deleting existing endpoints**

```
endpoint_name = 'suspicious-lines'                 1
try:
    sess.delete_endpoint(
        sagemaker.predictor.RealTimePredictor(
            endpoint=endpoint_name).endpoint)      2
    print(
        'Warning: Existing endpoint deleted to make way for new endpoint.')
except:
    passalexample>
```

* _1_ So you don’t create duplicate endpoints, name your endpoint.
* _2_ Deletes existing endpoint with that name

Next, in [listing 5.12](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05ex12), you create and deploy the endpoint. SageMaker is highly scalable and can handle very large datasets. For the datasets we use in this book, you only need a t2.medium machine to host your endpoint.

**Listing 5.12. Hosting the model: setting machine size**

```
rcf_endpoint = rcf.deploy(
    initial_instance_count=1,        1
    instance_type='ml.t2.medium'     2
)
```

* _1_ Number of machines to host your endpoint
* _2_ Size of the machine

You now need to set up the code that takes the results from the endpoint and puts them in a format you can easily work with.

**Listing 5.13. Hosting the model: converting to a workable format**

```
from sagemaker.predictor import csv_serializer, json_deserializer

rcf_endpoint.content_type = 'text/csv'
rcf_endpoint.serializer = csv_serializer
rcf_endpoint.accept = 'application/json'
rcf_endpoint.deserializer = json_deserializer
```

**5.8.6. Part 6: Testing the model**

You can now compute anomalies on the validation data as shown in [listing 5.14](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05ex14). Here you use the val\_df\_no\_result dataset because it does not contain the Error column (just as the training data did not contain the Error column). You then create a DataFrame called scores\_df to hold the results from the numerical values returned from the rcf\_endpoint.predict function. Then you’ll combine the scores\_df DataFrame with the val\_df DataFrame so you can see the score from the Random Cut Forest algorithm associated with each row in the training data.

**Listing 5.14. Adding scores to validation data**

```
results = rcf_endpoint.predict(
    val_df_no_result.values)                    1
scores_df = pd.DataFrame(results['scores'])     2
val_df = val_df.reset_index(drop=True)          3
results_df = pd.concat(
    [val_df, scores_df], axis=1)                4
results_df['Error'].value_counts()              5
```

* _1_ Gets the results from the val\_df\_no\_result DataFrame
* _2_ Creates a new DataFrame with the results
* _3_ Resets the index of the val\_df DataFrame so it starts at zero
* _4_ Concatenates the columns in the val\_df and the scores\_df DataFrames
* _5_ Shows how many errors there are in the val\_df DataFrame

To combine the data, we used the pandas concat function in [listing 5.14](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05ex14). This function combines two DataFrames, using the index of the DataFrames. If the axis parameter is 0, you will concatenate rows. If it is 1, you will concatenate columns.

Because we have just created the scores\_df DataFrame, the index for the rows starts at 0 and goes up to 21,192 (as there are 21,193 rows in the val\_df and scores\_df DataFrames). We then reset the index of the val\_df DataFrame so that it also starts at 0. That way when we concatenate the DataFrames, the scores line up with the correct rows in the val\_df DataFrame.

You can see from the following listing that there are 20,791 correct lines in the validation dataset (val\_df) and 402 errors (based on the Errors column in the val\_df DataFrame).

**Listing 5.15. Reviewing erroneous lines**

```
False    20791               1
True       402               2
Name: Error, dtype: int64
```

* _1_ Rows that do not contain an error
* _2_ Rows that do contain an error

Brett believes that he and his team catch about half the errors made by law firms and that this is sufficient to generate the behavior the bank wants from their lawyers: to bill accurately because they know that if they don’t, they will be asked to provide additional supporting information for their invoices.

To identify the errors with scores in the top half of the results, you use the pandas median function to identify the median score of the errors and then create a DataFrame called results\_above\_cutoff to hold the results ([listing 5.16](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05ex16)). To confirm that you have the median, you can look at the value counts of the Errors column in the DataFrame to determine that there are 201 rows in the DataFrame (half the total number of errors in the val\_df DataFrame).

The next listing calculates the number of rows where the score is greater than the median score.

**Listing 5.16. Calculating errors greater than 1.5 (the median score)**

```
score_cutoff = results_df[
    results_df['Error'] == True]['score'].median()     1
print(f'Score cutoff: {score_cutoff}')
results_above_cutoff = results_df[
    results_df['score'] > score_cutoff]                2
results_above_cutoff['Error'].value_counts()           3
```

* _1_ Gets the median score in the results\_df DataFrame
* _2_ Creates a new DataFrame called results\_above\_cutoff that contains rows where the score is greater than the median True score
* _3_ Displays the number of rows in the results\_above\_cutoff DataFrame

And the next listing shows the number of true errors above the median score and the number of false positives.

**Listing 5.17. Viewing false positives**

```
Score cutoff: 1.58626156755      1

True     201                     2
False     67                     3
```

* _1_ Queries only the invoices that have a score greater than 1.586
* _2_ Returns 201 invoice lines over the threshold that are errors
* _3_ Returns 67 invoice lines over the threshold that are identified as errors but are not errors

Because you are looking at the value\_counts of the Errors column, you can also see that for the 67 rows that did not contain errors, you will query the law firm. Brett tells you that this is a better hit rate than his team typically gets. With this information, you are able to prepare the two key ratios that allow you to describe how your model is performing. These two key ratios are _recall_ and _precision_:

* Recall is the proportion of correctly identified errors over the total number of invoice lines with errors.
* Precision is the proportion of correctly identified errors over the total number of invoice lines predicted to be errors.

These concepts are easier to understand with examples. The key numbers in this analysis that allow you to calculate recall and precision are the following:

* There are 402 errors in the validation dataset.
* You set a cutoff to identify half the erroneous lines submitted by the law firms (201 lines).
* When you set the cutoff at this point, you misidentify 67 correct invoice lines as being erroneous.

Recall is the number of identified errors divided by the total number of errors. Because we decided to use the median score to determine the cutoff, the recall will always be 50%.

Precision is the number of correctly identified errors divided by the total number of errors predicted. The total number of errors predicted is 268 (201 + 67). The precision is 201 / 268, or 75%.

Now that you have defined the cutoff, you can set a column in the results\_df DataFrame that sets a value of True for rows with scores that exceed the cutoff and False for rows with scores that are less than the cutoff, as shown in the following listing.

**Listing 5.18. Displaying the results in a pandas DataFrame**

```
results_df['Prediction'] = \
    results_df['score'] > score_cutoff     1
results_df.head()                          2
```

* _1_ Sets the values in the Prediction column to True where the score is greater than the cutoff
* _2_ Displays the results

The dataset now shows the results for each invoice line in the validation dataset.

Exercise:

1. What is the score for row 356 of the val\_df dataset?
2. How would you submit this single row to the prediction function to return the score for only that row?

#### 5.9. Deleting the endpoint and shutting down your notebook instance <a href="#ch05lev1sec9__title" id="ch05lev1sec9__title"></a>

It is important that you shut down your notebook instance and delete your endpoint. We don’t want you to get charged for SageMaker services that you’re not using.

**5.9.1. Deleting the endpoint**

Appendix D describes how to shut down your notebook instance and delete your endpoint using the SageMaker console, or you can do that with the code in this listing.

**Listing 5.19. Deleting the notebook**

```
# Remove the endpoint (optional)
# Comment out this cell if you want the endpoint to persist after Run All
sagemaker.Session().delete_endpoint(rcf_endpoint.endpoint)
```

To delete the endpoint, uncomment the code in the listing, then click ![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ctrl\_ent.jpg) to run the code in the cell.

**5.9.2. Shutting down the notebook instance**

To shut down the notebook, go back to your browser tab where you have SageMaker open. Click the Notebook Instances menu item to view all of your notebook instances. Select the radio button next to the notebook instance name as shown in [figure 5.16](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05fig16), then click Stop on the Actions menu. It takes a couple of minutes to shut down.

**Figure 5.16. Shutting down the notebook**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch05fig16\_alt.jpg)

#### 5.10. Checking to make sure the endpoint is deleted <a href="#ch05lev1sec10__title" id="ch05lev1sec10__title"></a>

If you didn’t delete the endpoint using the notebook (or if you just want to make sure it is deleted), you can do this from the SageMaker console. To delete the endpoint, click the radio button to the left of the endpoint name, then click the Actions menu item and click Delete in the menu that appears.

When you have successfully deleted the endpoint, you will no longer incur AWS charges for it. You can confirm that all of your endpoints have been deleted when you see the text “There are currently no resources” displayed at the bottom of the Endpoints page ([figure 5.17](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_015.html#ch05fig17)).

**Figure 5.17. Verifying that you have successfully deleted the endpoint**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch05fig17\_alt.jpg)

Brett’s team can now run each of the invoices they receive from their lawyers and determine within seconds whether they should query the invoice or not. Now Brett’s team can focus on assessing the adequacy of the law firm’s responses to their query rather than on whether an invoice should be queried. This will allow Brett’s team to handle significantly more invoices with the same amount of effort.

#### Summary <a href="#ch05lev1sec11__title" id="ch05lev1sec11__title"></a>

* Identify what your algorithm is trying to achieve. In Brett’s case in this chapter, the algorithm does not need to identify every erroneous line, it only needs to identify enough lines to drive the right behavior from the law firms.
* Synthetic data is data created by you, the analyst, as opposed to real data found in the real world. A good set of real data is more interesting to work with than synthetic data because it is typically more nuanced.
* Unsupervised machine learning can be used to solve problems where you don’t have any trained data.
* The difference between a supervised algorithm and an unsupervised algorithm is that with an unsupervised algorithm, you don’t provide any labeled data. You just provide the data, and the algorithm decides how to interpret it.
* Anomalies are data points that have something unusual about them.
* Random Cut Forest can be used to address the challenges inherent in identifying anomalies.
* Recall and precision are two of the key ratios you use to describe how your model is performing.
