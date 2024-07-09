# Chapter 6. Forecasting your company’s monthly power usage

_This chapter covers_

* Preparing your data for time-series analysis
* Visualizing data in your Jupyter notebook
* Using a neural network to generate forecasts
* Using DeepAR to forecast power consumption

Kiara works for a retail chain that has 48 locations around the country. She is an engineer, and every month her boss asks her how much energy they will consume in the next month. Kiara follows the procedure taught to her by the previous engineer in her role: she looks at how much energy they consumed in the same month last year, weights it by the number of locations they have gained or lost, and provides that number to her boss. Her boss sends this estimate to the facilities management teams to help plan their activities and then to Finance to forecast expenditure. The problem is that Kiara’s estimates are always wrong—sometimes by a lot.

As an engineer, she reckons there must be a better way to approach this problem. In this chapter, you’ll use SageMaker to help Kiara produce better estimates of her company’s upcoming power consumption.

#### 6.1. What are you making decisions about? <a href="#ch06lev1sec1__title" id="ch06lev1sec1__title"></a>

This chapter covers different material than you’ve seen in earlier chapters. In previous chapters, you used supervised and unsupervised machine learning algorithms to make decisions. You learned how each algorithm works and then you applied the algorithm to the data. In this chapter, you’ll use a _neural network_ to predict how much power Kiara’s company will use next month.

Neural networks are much more difficult to intuitively understand than the machine learning algorithms we’ve covered so far. Rather than attempt to give you a deep understanding of neural networks in this chapter, we’ll focus on how to explain the output from a neural network. Instead of a theoretical discussion of neural networks, you’ll come out of this chapter knowing how to use a neural network to forecast time-series events and how to explain the results of the forecast. Rather than learning in detail the _why_ of neural networks, you’ll learn the _how_.

[Figure 6.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06fig01) shows the predicted versus actual power consumption for one of Kiara’s sites for a six-week period from mid-October 2018 to the end of November 2018. The site follows a weekly pattern with a higher usage on the weekdays and dropping very low on Sundays.

**Figure 6.1. Predicted versus actual power consumption for November 2018 for one of Kiara’s sites**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch06fig01\_alt.jpg)

The shaded area shows the range Kiara predicted with 80% accuracy. When Kiara calculates the average error for her prediction, she discovers it is 5.7%, which means that for any predicted amount, it is more likely to be within 5.7% of the predicted figure than not. Using SageMaker, you can do all of this without an in-depth understanding of how neural networks actually function. And, in our view, that’s OK.

To understand how neural networks can be used for time-series forecasting, you first need to understand why time-series forecasting is a thorny issue. Once you understand this, you’ll see what a neural network is and how a neural network can be applied to time-series forecasting. Then you’ll roll up your sleeves, fire up SageMaker, and see it in action on real data.

**Note**

The power consumption data you’ll use in this chapter is provided by BidEnergy ([http://www.bidenergy.com](http://www.bidenergy.com/)), a company that specializes in power-usage forecasting and in minimizing power expenditure. The algorithms used by BidEnergy are more sophisticated than you’ll see in this chapter, but you’ll get a feel for how machine learning in general, and neural networks in particular, can be applied to forecasting problems.

**6.1.1. Introduction to time-series data**

Time-series data consists of a number of observations at particular intervals. For example, if you created a time series of your weight, you could record your weight on the first of every month for a year. Your time series would have 12 observations with a numerical value for each observation. [Table 6.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06table01) shows what this might look like.

**Table 6.1. Time-series data showing my (Doug's) weight in kilograms over the past year**

| Date       | Weight (kg) |
| ---------- | ----------- |
| 2018-01-01 | 75          |
| 2018-02-01 | 73          |
| 2018-03-01 | 72          |
| 2018-04-01 | 71          |
| 2018-05-01 | 72          |
| 2018-06-01 | 71          |
| 2018-07-01 | 70          |
| 2018-08-01 | 73          |
| 2018-09-01 | 70          |
| 2018-10-01 | 69          |
| 2018-11-01 | 72          |
| 2018-12-01 | 74          |

It’s pretty boring to look at a table of data. It’s hard to get a real understanding of the data when it is presented in a table format. Line charts are the best way to view data. [Figure 6.2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06fig02) shows the same data presented as a chart.

**Figure 6.2. A line chart displays the same time-series data showing my weight in kilograms over the past year.**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch06fig02.jpg)

You can see from this time series that the date is on the left and my weight is on the right. If you wanted to record the time series of body weight for your entire family, for example, you would add a column for each of your family members. In [table 6.2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06table02), you can see my weight and the weight of each of my family members over the course of a year.

**Table 6.2. Time-series data showing the weight in kilograms of family members over a year**

| Date       | Me | Spouse | Child 1 | Child 2 |
| ---------- | -- | ------ | ------- | ------- |
| 2018-01-01 | 75 | 52     | 38      | 67      |
| 2018-02-01 | 73 | 52     | 39      | 68      |
| 2018-03-01 | 72 | 53     | 40      | 65      |
| 2018-04-01 | 71 | 53     | 41      | 63      |
| 2018-05-01 | 72 | 54     | 42      | 64      |
| 2018-06-01 | 71 | 54     | 42      | 65      |
| 2018-07-01 | 70 | 55     | 42      | 65      |
| 2018-08-01 | 73 | 55     | 43      | 66      |
| 2018-09-01 | 70 | 56     | 44      | 65      |
| 2018-10-01 | 69 | 57     | 45      | 66      |
| 2018-11-01 | 72 | 57     | 46      | 66      |
| 2018-12-01 | 74 | 57     | 46      | 66      |

And, once you have that, you can visualize the data as four separate charts, as shown in [figure 6.3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06fig03).

**Figure 6.3. Line charts display the same time-series data showing the weight in kilograms of members of a family over the past year.**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch06fig03\_alt.jpg)

You’ll see the chart formatted in this way throughout this chapter and the next. It is a common format used to concisely display time-series data.

**6.1.2. Kiara’s time-series data: Daily power consumption**

Power consumption data is displayed in a manner similar to our weight data. Kiara’s company has 48 different business sites (retail outlets and warehouses), so each site gets its own column when you compile the data. Each observation is a cell in that column. [Table 6.3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06table03) shows a sample of the electricity data used in this chapter.

**Table 6.3. Power usage data sample for Kiara’s company in 30-minute intervals**

| Time                | Site\_1 | Site\_2 | Site\_3 | Site\_4 | Site\_5 | Site\_6 |
| ------------------- | ------- | ------- | ------- | ------- | ------- | ------- |
| 2017-11-01 00:00:00 | 13.30   | 13.3    | 11.68   | 13.02   | 0.0     | 102.9   |
| 2017-11-01 00:30:00 | 11.75   | 11.9    | 12.63   | 13.36   | 0.0     | 122.1   |
| 2017-11-01 01:00:00 | 12.58   | 11.4    | 11.86   | 13.04   | 0.0     | 110.3   |

This data looks similar to the data in [table 6.2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06table02), which shows the weight of each family member each month. The difference is that instead of each column representing a family member, in Kiara’s data, each column represents a site (office or warehouse location) for her company. And instead of each row representing a person’s weight on the first day of the month, each row of Kiara’s data shows how much power each site used on that day.

Now that you see how time-series data can be represented and visualized, you are ready to see how to use a Jupyter notebook to visualize this data.

#### 6.2. Loading the Jupyter notebook for working with time-series data <a href="#ch06lev1sec2__title" id="ch06lev1sec2__title"></a>

To help you understand how to display time-series data in SageMaker, for the first time in this book, you’ll work with a Jupyter notebook that does not contain a SageMaker machine learning model. Fortunately, because the SageMaker environment is simply a standard Jupyter Notebook server with access to SageMaker models, you can use SageMaker to run ordinary Jupyter notebooks as well. You’ll start by downloading and saving the Jupyter notebook at

[https://s3.amazonaws.com/mlforbusiness/ch06/time\_series\_practice.ipynb](https://s3.amazonaws.com/mlforbusiness/ch06/time\_series\_practice.ipynb)

You’ll upload it to the same SageMaker environment you have used for previous chapters. Like you did for previous chapters, you’ll set up a notebook on SageMaker. If you skipped the earlier chapters, follow the instructions in [appendix C](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_028.html#app03) on how to set up SageMaker.

When you go to SageMaker, you’ll see your notebook instances. The notebook instances you created for the previous chapters (or the one that you’ve just created by following the instructions in [appendix C](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_028.html#app03)) will either say Open or Start. If it says Start, click the Start link and wait a couple of minutes for SageMaker to start. Once it displays Open Jupyter, select that link to open your notebook list ([figure 6.4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06fig04)).

**Figure 6.4. Viewing the Notebook instances list**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch06fig04\_alt.jpg)

Create a new folder for [chapter 6](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06) by clicking New and selecting Folder at the bottom of the dropdown list ([figure 6.5](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06fig05)). This creates a new folder called Untitled Folder.

**Figure 6.5. Creating a new folder in SageMaker**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch06fig05.jpg)

To rename the folder, when you tick the checkbox next to Untitled Folder, you will see the Rename button appear. Click it and change the name to ch06. Click the ch06 folder and you will see an empty notebook list. Click Upload to upload the time\_series \_practice.ipynb notebook to the folder.

After uploading the file, you’ll see the notebook in your list. Click it to open it. You are now ready to work with the time\_series\_practice notebook. But before we set up the time-series data for this notebook, let’s look at some of the theory and practices surrounding time-series analysis.

#### 6.3. Preparing the dataset: Charting time-series data <a href="#ch06lev1sec3__title" id="ch06lev1sec3__title"></a>

Jupyter notebooks and pandas are excellent tools for working with time-series data. In the SageMaker neural network notebook you’ll create later in this chapter, you’ll use pandas and a data visualization library called _Matplotlib_ to prepare the data for the neural network and to analyze the results. To help you understand how it works, you’ll get your hands dirty with a time-series notebook that visualizes the weight of four different people over the course of a year.

To use a Jupyter notebook to visualize data, you’ll need to set up the notebook to do so. As shown in [listing 6.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06ex1), you first need to tell Jupyter that you intend to display some charts in this notebook. You do this with the line %matplotlib inline, as shown in line 1.

**Listing 6.1. Displaying charts**

```
%matplotlib inline                  1

import pandas as pd                 2
import matplotlib.pyplot as plt     3
```

* _1_ Loads Matplotlib capability into Jupyter
* _2_ Imports the pandas library for working with data
* _3_ Imports the Matplotlib library for displaying charts

Matplotlib is a Python charting library, but there are lots of Python charting libraries that you could use. We have selected Matplotlib because it is available in the Python standard library and, for simple things, is easy to use.

The reason line 1 starts with a % symbol is that the line is an _instruction_ to Jupyter, rather than a line in your code. It tells the Jupyter notebook that you’ll be displaying charts, so it should load the software to do this into the notebook. This is called a _magic command_.

The magic command: Is it really magic?

Actually, it is. When you see a command in a Jupyter notebook that starts with % or with %%, the command is known as a magic command. Magic commands provide additional features to the Jupyter notebook, such as the ability to display charts or run external scripts. You can read more about magic commands at [https://ipython.readthedocs.io/en/stable/interactive/magics.html](https://ipython.readthedocs.io/en/stable/interactive/magics.html).

As you can see in [listing 6.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06ex1), after loading the Matplotlib functionality into your Jupyter notebook, you then imported the libraries: pandas and Matplotlib. (Remember that in line 1 of the listing, where you referenced %matplotlib inline, you did not import the Matplotlib library; line 3 is where you imported that library.)

After importing the relevant libraries, you then need to get some data. When working with SageMaker in the previous chapters, you loaded that data from S3. For this notebook, because you are just learning about visualization with pandas and Jupyter Notebook, you’ll just create some data and send it to a pandas DataFrame, as shown in the next listing.

**Listing 6.2. Inputting the time-series data**

```
my_weight = [                                     1
    {'month': '2018-01-01', 'Me': 75},
    {'month': '2018-02-01', 'Me': 73},
    {'month': '2018-03-01', 'Me': 72},
    {'month': '2018-04-01', 'Me': 71},
    {'month': '2018-05-01', 'Me': 72},
    {'month': '2018-06-01', 'Me': 71},
    {'month': '2018-07-01', 'Me': 70},
    {'month': '2018-08-01', 'Me': 73},
    {'month': '2018-09-01', 'Me': 70},
    {'month': '2018-10-01', 'Me': 69},
    {'month': '2018-11-01', 'Me': 72},
    {'month': '2018-12-01', 'Me': 74}
]
df = pd.DataFrame(my_weight).set_index('month')   2
df.index = pd.to_datetime(df.index)               3
df.head()                                         4
```

* _1_ Creates the dataset for data in [figure 6.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06fig01)
* _2_ Converts the dataset to a pandas DataFrame
* _3_ Sets the index of the DataFrame to a time series
* _4_ Displays the first five rows

Now, here’s where the real magic lies. To display a chart, all you need to do is type the following line into the Jupyter notebook cell:

```
df.plot()
```

The Matplotlib library recognizes that the data is time-series data from the index type you set in line 3 of [listing 6.2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06ex2), so it just works. Magic! The output of the df.plot() command is shown in [figure 6.6](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06fig06).

**Figure 6.6. Time-series data returned by df.plot showing my weight in kilograms over the past year**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch06fig06.jpg)

To expand the data to include the weight of the entire family, you first need to set up the data. The following listing shows the dataset expanded to include data from every family member.

**Listing 6.3. Inputting time-series data for the whole family**

```
family_weight = [                                    1
    {'month': '2018-01-01', 'Me': 75, 'spouse': 67,
        'ch_1': 52, 'ch_2': 38},
    {'month': '2018-02-01', 'Me': 73, 'spouse': 68,
        'ch_1': 52, 'ch_2': 39},
    {'month': '2018-03-01', 'Me': 72, 'spouse': 65,
        'ch_1': 53, 'ch_2': 40},
    {'month': '2018-04-01', 'Me': 71, 'spouse': 63,
        'ch_1': 53, 'ch_2': 41},
    {'month': '2018-05-01', 'Me': 72, 'spouse': 64,
        'ch_1': 54, 'ch_2': 42},
    {'month': '2018-06-01', 'Me': 71, 'spouse': 65,
        'ch_1': 54, 'ch_2': 42},
    {'month': '2018-07-01', 'Me': 70, 'spouse': 65,
        'ch_1': 55, 'ch_2': 42},
    {'month': '2018-08-01', 'Me': 73, 'spouse': 66,
        'ch_1': 55, 'ch_2': 43},
    {'month': '2018-09-01', 'Me': 70, 'spouse': 65,
        'ch_1': 56, 'ch_2': 44},
    {'month': '2018-10-01', 'Me': 69, 'spouse': 66,
        'ch_1': 57, 'ch_2': 45},
    {'month': '2018-11-01', 'Me': 72, 'spouse': 66,
        'ch_1': 57, 'ch_2': 46},
    {'month': '2018-12-01', 'Me': 74, 'spouse': 66,
        'ch_1': 57, 'ch_2': 46}
]
df2 = pd.DataFrame(
        family_weight).set_index('month')          2
df2.index = pd.to_datetime(df2.index)              3
df2.head()                                         4
```

* _1_ Creates the dataset for the month and each person’s weight
* _2_ Converts the dataset to a pandas DataFrame
* _3_ Sets the index of the DataFrame to a time series
* _4_ Displays the first five rows

Displaying four charts in Matplotlib is a little more complex than displaying one chart. You need to first create an area to display the charts, and then you need to loop across the columns of data to display the data in each column. Because this is the first loop you’ve used in this book, we’ll go into some detail about it.

**6.3.1. Displaying columns of data with a loop**

The loop is called a _for loop_, which means you give it a list (usually) and step through each item in the list. This is the most common type of loop you’ll use in data analysis and machine learning because most of the things you’ll loop through are lists of items or rows of data.

The standard way to loop is shown in the next listing. Line 1 of this listing defines a list of three items: A, B, and C. Line 2 sets up the loop, and line 3 prints each item.

**Listing 6.4. Standard way to loop through a list**

```
my_list = ['A', 'B', 'C']   1
for item in my_list:        2
    print(item)             3
```

* _1_ Creates a list called my\_list
* _2_ Loops through my\_list
* _3_ Prints each item in the list

Running this code prints A, B, and C, as shown next.

**Listing 6.5. Standard output when you run the command to loop through a list**

```
A
B
C
```

When creating charts with Matplotlib, in addition to looping, you need to keep track of how many times you have looped. Python has a nice way of doing this: it’s called _enumerate_.

To enumerate through a list, you provide two variables to store the information from the loop and to wrap the list you are looping through. The enumerate function returns two such variables. The first variable is the count of how many times you’ve looped (starting with zero), and the second variable is the item retrieved from the list. [Listing 6.6](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06ex6) shows [listing 6.4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06ex4) converted to an enumerated for loop.

**Listing 6.6. Standard way to enumerate through a loop**

```
my_list = ['A', 'B', 'C']              1
for i, item in enumerate(my_list):     2
    print(f'{i}. {item}')              3
```

* _1_ Creates a list called my\_list
* _2_ Loops through my\_list and stores the count in the i variable and the list item in the item variable
* _3_ Prints the loop count (starting with zero) and each item in the list

Running this code creates the same output as that shown in [listing 6.5](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06ex5) but also allows you to display how many items you have looped through in the list. To run the code, click the cell and press ![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ctrl\_ent.jpg). The next listing shows the output from using the enumerate function in your loop.

**Listing 6.7. Output when enumerating through the loop**

```
0. A
1. B
2. Cformalexample>
```

With that as background, you are ready to create multiple charts in Matplotlib.

**6.3.2. Creating multiple charts**

In Matplotlib, you use the subplots functionality to create multiple charts. To fill the subplots with data, you loop through each of the columns in the table showing the weight of each family member ([listing 6.3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06ex3)) and display the data from each column.

**Listing 6.8. Charting time-series data for the whole family**

```
start_date = "2018-01-01"                               1
end_date = "2018-12-31"                                 2
fig, axs = plt.subplots(
    2,
    2,
    figsize=(12, 5),
    sharex=True)                                        3
axx = axs.ravel()                                       4
for i, column in enumerate(df2.columns):                5
    df2[df2.columns[i]].loc[start_date:end_date].plot(
        ax=axx[i])                                      6
    axx[i].set_xlabel("month")                          7
    axx[i].set_ylabel(column)                           8
```

* _1_ Sets the start date
* _2_ Sets the end date
* _3_ Creates the matplotlib to hold the four charts
* _4_ Ensures the plots are stored as a series so you can loop through them
* _5_ Loops through each of the columns of data
* _6_ Sets the chart to display a particular column of data
* _7_ Sets the x-axis label
* _8_ Sets the y-axis label

In line 3 of [listing 6.8](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06ex8), you state that you want to display a grid of 2 charts by 2 charts with a width of 12 inches and height of 5 inches. This creates four Matplotlib objects in a 2-by-2 grid that you can fill with data. You use the code in line 4 to turn the 2-by-2 grid into a list that you can loop through. The variable axx stores the list of Matplotlib subplots that you will fill with data.

When you run the code in the cell by clicking into the cell and pressing ![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ctrl\_ent.jpg), you can see the chart, as shown in [figure 6.7](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06fig07).

**Figure 6.7. Time-series data generated with code, showing the weight of family members over the past year.**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch06fig07\_alt.jpg)

So far in this chapter, you’ve looked at time-series data, learned how to loop through it, and how to visualize it. Now you’ll learn why neural networks are a good way to forecast time-series events.

#### 6.4. What is a neural network? <a href="#ch06lev1sec4__title" id="ch06lev1sec4__title"></a>

Neural networks (sometimes referred to as _deep learning_) approach machine learning in a different way than traditional machine learning models such as XGBoost. Although both XGBoost and neural networks are examples of supervised machine learning, each uses different tools to tackle the problem. XGBoost uses an ensemble of approaches to attempt to predict the target result, whereas a neural network uses just one approach.

A neural network attempts to solve the problem by using layers of interconnected neurons. The neurons take inputs in one end and push outputs to the other side. The connections between the neurons have weights assigned to them. If a neuron receives enough weighted input, then it _fires_ and pushes the signal to the next layer of neurons it’s connected to.

**Definition**

A neuron is simply a mathematical function that receives two or more inputs, applies a weighting to the inputs, and passes the result to multiple outputs if the weighting is above a certain threshold.

Imagine you’re a neuron in a neural network designed to filter gossip based on how salacious it is or how true it is. You have ten people (the interconnected neurons) who tell you gossip, and if it is salacious enough, true enough, or a combination of the two, you’ll pass it on to the ten people who you send gossip to. Otherwise, you’ll keep it to yourself.

Also imagine that some of the people who send you gossip are not very trustworthy, whereas others are completely honest. (The trustworthiness of your sources changes when you get feedback on whether the gossip was true or not and how salacious it was perceived to be.) You might not pass on a piece of gossip from several of your least trustworthy sources, but you might pass on gossip that the most trusted people tell you, even if only one person tells you this gossip.

Now, let’s look at a specific time-series neural network algorithm that you’ll use to forecast power consumption.

What is DeepAR?

DeepAR is Amazon’s time-series neural network algorithm that takes as its input related types of time-series data and automatically combines the data into a global model of all time series in a dataset. It then uses this global model to predict future events. In this way, DeepAR is able to incorporate different types of time-series data (such as power consumption, temperature, wind speed, and so on) into a single model that is used in our example to predict power consumption.

In this chapter, you are introduced to DeepAR, and you’ll build a model for Kiara that uses historical data from her 48 sites. In [chapter 7](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07), you will incorporate other features of the DeepAR algorithm (such as weather patterns) to enhance your prediction.\[[a](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06fna)]

> a
>
> You can read more about DeepAR on this AWS site: [https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html).

#### 6.5. Getting ready to build the model <a href="#ch06lev1sec5__title" id="ch06lev1sec5__title"></a>

Now that you have a deeper understanding of neural networks and of how DeepAR works, you can set up another notebook in SageMaker and make some decisions. As you did in the previous chapters, you are going to do the following:

1. Upload a dataset to S3
2. Set up a notebook on SageMaker
3. Upload the starting notebook
4. Run it against the data

**Tip**

If you’re jumping into the book at this chapter, you might want to visit the appendixes, which show you how to do the following:

* [Appendix A](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_021.html#app01): sign up to AWS, Amazon’s web service
* [appendix B](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_024.html#app02): set up S3, AWS’s file storage service
* [Appendix C](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_028.html#app03): set up SageMaker

**6.5.1. Uploading a dataset to S3**

To set up the dataset for this chapter, you’ll follow the same steps as you did in [appendix B](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_024.html#app02). You don’t need to set up another bucket though. You can just go to the same bucket you created earlier. In our example, we called the bucket mlforbusiness, but your bucket will be called something different.

When you go to your S3 account, you will see a list of your buckets. Clicking the bucket you created for this book, you might see the ch02, ch03, ch04, and ch05 folders if you created these in the previous chapters. For this chapter, you’ll create a new folder called _ch06_. You do this by clicking Create Folder and following the prompts to create a new folder.

Once you’ve created the folder, you are returned to the folder list inside your bucket. There you will see you now have a folder called ch06.

Now that you have the ch06 folder set up in your bucket, you can upload your data file and start setting up the decision-making model in SageMaker. To do so, click the folder and download the data file at this link:

[https://s3.amazonaws.com/mlforbusiness/ch06/meter\_data.csv](https://s3.amazonaws.com/mlforbusiness/ch06/meter\_data.csv)

Then upload the CSV file into the ch06 folder by clicking Upload. Now you’re ready to set up the notebook instance.

**6.5.2. Setting up a notebook on SageMaker**

Just as we prepared the CSV data you uploaded to S3 for this scenario, we’ve already prepared the Jupyter notebook you’ll use now. You can download it to your computer by navigating to this URL:

[https://s3.amazonaws.com/mlforbusiness/ch06/energy\_usage.ipynb](https://s3.amazonaws.com/mlforbusiness/ch06/energy\_usage.ipynb)

Once you have downloaded the notebook to your computer, you can upload it to the same SageMaker folder you used to work with the time\_series\_practice.ipynb notebook. (Click Upload to upload the notebook to the folder.)

#### 6.6. Building the model <a href="#ch06lev1sec6__title" id="ch06lev1sec6__title"></a>

As in the previous chapters, you will go through the code in six parts:

1. Load and examine the data.
2. Get the data into the right shape.
3. Create training and validation datasets.
4. Train the machine learning model.
5. Host the machine learning model.
6. Test the model and use it to make decisions.

**6.6.1. Part 1: Loading and examining the data**

As in the previous chapters, the first step is to say where you are storing the data. To do that, you need to change 'mlforbusiness' to the name of the bucket you created when you uploaded the data, and rename its subfolder to the name of the subfolder on S3 where you want to store the data ([listing 6.9](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06ex9)).

If you named the S3 folder ch06, then you don’t need to change the name of the folder. If you kept the name of the CSV file that you uploaded earlier in the chapter, then you don’t need to change the meter\_data.csv line of code. If you changed the name of the CSV file, then update meter\_data.csv to the name you changed it to. To run the code in the notebook cell, click the cell and press ![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ctrl\_ent.jpg).

**Listing 6.9. Say where you are storing the data**

```
data_bucket = 'mlforbusiness'   1
subfolder = 'ch06'              2
dataset = 'meter_data.csv'      3
```

* _1_ S3 bucket where the data is stored
* _2_ Subfolder of S3 bucket where the data is stored
* _3_ Dataset that’s used to train and test the model

Many of the Python modules and libraries imported in [listing 6.10](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06ex10) are the same as the imports you used in previous chapters, but you’ll also use Matplotlib in this chapter.

**Listing 6.10. Importing the modules**

```
%matplotlib inline                      1

import datetime                         2
import json                             3
import random                           4
from random import shuffle              5

import boto3                            6
import ipywidgets as widgets            7
import matplotlib.pyplot as plt         8
import numpy as np                      9
import pandas as pd                     10
from dateutil.parser import parse       11

import s3fs                             12
import sagemaker                        13

role = sagemaker.get_execution_role()   14
s3 = s3fs.S3FileSystem(anon=False)      15
```

* _1_ Uses plotting in the Jupyter notebook
* _2_ Uses date and time functions
* _3_ Imports Python’s json module to work with JSON files
* _4_ Imports the random module to generate random numbers
* _5_ Imports the shuffle function to shuffle random numbers
* _6_ Imports the boto3 AWS library
* _7_ Imports interactive widgets in Jupyter notebooks
* _8_ Imports plotting functionality from Matplotlib
* _9_ Imports the numpy library to work with arrays of numbers
* _10_ Imports the pandas Python library
* _11_ Date parsing convenience functions
* _12_ Imports the s3fs module to make working with S3 files easier
* _13_ Imports SageMaker
* _14_ Creates a role on SageMaker
* _15_ Establishes the connection with S3

Next, you’ll load and view the data. In [listing 6.11](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06ex11), you read in the CSV data and display the top 5 rows in a pandas DataFrame. Each row of the dataset shows 30-minute energy usage data for a 13-month period from November 1, 2017, to mid-December 2018. Each column represents one of the 48 retail sites owned by Kiara’s company.

**Listing 6.11. Loading and viewing the data**

```
s3_data_path = \
    f"s3://{data_bucket}/{subfolder}/data"           1
s3_output_path = \
    f"s3://{data_bucket}/{subfolder}/output"         2
df = pd.read_csv(
    f's3://{data_bucket}/{subfolder}/meter_data.csv',
    index_col=0)                                     3
df.head()                                            4
```

* _1_ Reads the S3 dataset
* _2_ Displays 3 rows of the DataFrame (rows 5, 6, and 7)
* _3_ Reads in the meter data
* _4_ Displays the top 5 rows

[Table 6.4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06table04) shows the output of running display(df\[5:8]). The table only shows the first 6 sites, but the dataset in the Jupyter notebook has all 48 sites.

**Table 6.4. Power usage data in half-hour intervals**

| Index               | Site\_1 | Site\_2 | Site\_3 | Site\_4 | Site\_5 | Site\_6 |
| ------------------- | ------- | ------- | ------- | ------- | ------- | ------- |
| 2017-11-01 00:00:00 | 13.30   | 13.3    | 11.68   | 13.02   | 0.0     | 102.9   |
| 2017-11-01 00:30:00 | 11.75   | 11.9    | 12.63   | 13.36   | 0.0     | 122.1   |
| 2017-11-01 01:00:00 | 12.58   | 11.4    | 11.86   | 13.04   | 0.0     | 110.3   |

You can see that column 5 in [table 6.4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06table04) shows zero consumption for the first hour and a half in November. We don’t know if this is an error in the data or if the stores didn’t consume any power during that period. We’ll discuss the implications of this as you work through the analysis.

Let’s take a look at the size of your dataset. When you run the code in [listing 6.12](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06ex12), you can see that the dataset has 48 columns (one column per site) and 19,632 rows of 30-minute data usage.

**Listing 6.12. Viewing the number of rows and columns**

```
print(f'Number of rows in dataset: {df.shape[0]}')
print(f'Number of columns in dataset: {df.shape[1]}')
```

Now that you’ve loaded the data, you need to get the data into the right shape so you can start working with it.

**6.6.2. Part 2: Getting the data into the right shape**

Getting the data into the right shape involves several steps:

* Converting the data to the right interval
* Determining if missing values are going to create any issues
* Fixing any missing values if you need to
* Saving the data to S3

First, you will convert the data to the right interval. Time-series data from the power meters at each site is recorded in 30-minute intervals. The fine-grained nature of this data is useful for certain work, such as quickly identifying power spikes or drops, but it is not the right interval for our analysis.

For this chapter, because you are not combining this dataset with any other datasets, you could use the data with the 30-minute interval to run your model. However, in the next chapter, you will combine the historical consumption data with daily weather forecast predictions to better predict the consumption over the coming month. The weather data that you’ll use in [chapter 7](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07) reflects weather conditions set at daily intervals. Because you’ll be combining the power consumption data with daily weather data, it is best to work with the power consumption data using the same interval.

**CONVERTING THE DATASET FROM 30-MINUTE INTERVALS TO DAILY INTERVALS**

[Listing 6.13](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06ex13) shows how to convert 30-minute data to daily data. The pandas library contains many helpful features for working with time-series data. Among the most helpful is the resample function, which easily converts time-series data in a particular interval (such as 30 minutes) into another interval (such as daily).

In order to use the resample function, you need to ensure that your dataset uses the date column as the index and that the index is in date-time format. As you might expect from its name, the index is used to reference a row in the dataset. So, for example, a dataset with the index of 1:30 AM 1 November 2017 can be referenced by 1:30 AM 1 November 2017. The pandas library can take rows referenced by such indexes and convert them into other periods such as days, months, quarters, or years.

**Listing 6.13. Converting data to daily figures**

```
df.index = pd.to_datetime(df.index)    1
daily_df = df.resample('D').sum()      2
daily_df.head()                        3
```

* _1_ Sets the index column to a date-time format
* _2_ Resamples the dataset so that the data is in daily intervals rather than 30-minute intervals
* _3_ Displays the top 5 rows of the dataset

[Table 6.5](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06table05) shows the converted data in daily figures.

**Table 6.5. Power usage data in daily intervals**

| Index      | Site\_1 | Site\_2 | Site\_3 | Site\_4 | Site\_5 | Site\_6 |
| ---------- | ------- | ------- | ------- | ------- | ------- | ------- |
| 2017-11-01 | 1184.23 | 1039.1  | 985.95  | 1205.07 | None    | 6684.4  |
| 2017-11-02 | 1210.9  | 1084.7  | 1013.91 | 1252.44 | None    | 6894.3  |
| 2017-11-03 | 1247.6  | 1004.2  | 963.95  | 1222.4  | None    | 6841    |

The following listing shows the number of rows and columns in the dataset as well as the earliest and latest dates.

**Listing 6.14. Viewing the data in daily intervals**

```
print(daily_df.shape)                                1
print(f'Time series starts at {daily_df.index[0]} \
and ends at {daily_df.index[-1]}')                   2
```

* _1_ Prints the number of rows and columns in the dataset
* _2_ Displays the earliest date and latest date in the dataset

In the output, you can now see that your dataset has changed from 19,000 rows of 30-minute data to 409 rows of daily data, and the number of columns remains the same:

```
(409, 48)
Time series starts at 2017-11-01 00:00:00 and ends at 2018-12-14 00:00:00
```

**HANDLING ANY MISSING VALUES IF REQUIRED**

As you work with the pandas library, you’ll come across certain gems that allow you to handle a thorny problem in an elegant manner. The line of code shown in [listing 6.15](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06ex15) is one such instance.

Basically, the data you are using in this chapter, with the exception of the first 30 days, is in good shape. It is missing a few observations, however, representing a handful of missing data points. This doesn’t impact the training of the data (DeepAR handles missing values well) but you can?t make predictions using data with missing values. To use this dataset to make predictions, you need to ensure there are no missing values. This section shows you how to do that.

**Note**

You need to make sure that the data you are using for your predictions is complete and has no missing values.

The pandas fillna function has the ability to forward fill missing data. This means that you can tell fillna to fill any missing value with the preceding value. But Kiara knows that most of their locations follow a weekly cycle. If one of the warehouse sites (that are closed on weekends) is missing data for one Saturday, and you forward fill the day from Friday, your data will not be very accurate. Instead, the one-liner in the next listing replaces a missing value with the value from 7 days prior.

**Listing 6.15. Replacing missing values**

```
daily_df = daily_df.fillna(daily_df.shift(7))
```

With that single line of code, you have replaced missing values with values from 7 days earlier.

**VIEWING THE DATA**

Time-series data is best understood visually. To help you understand the data better, you can create charts showing the power consumption at each of the sites. The code in [listing 6.16](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06ex16) is similar to the code you worked with in the practice notebook earlier in the chapter. The primary difference is that, instead of looping through every site, you set up a list of sites in a variable called indicies and loop through that.

If you remember, in [listing 6.16](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06ex16), you imported matplotlib.pyplot as plt. Now you can use all of the functions in plt. In line 2 of [listing 6.16](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06ex16), you create a Matplotlib figure that contains a 2-by-5 grid. Line 3 of the listing tells Matplotlib that when you give it data to work with, it should turn the data into a single data series rather than an array.

In line 4, the indices are the column numbers of the sites in the dataset. Remember that Python is zero-based, so 0 would be site 1. These 10 sites display in the 2-by-5 grid you set up in line 2. To view other sites, just change the numbers and run the cell again.

Line 5 is a loop that goes through each of the indices you defined in line 4. For each item in an index, the loop adds data to the Matplotlib figure you created in line 2. Your figure contains a grid 2 charts wide by 5 charts long so it has room for 10 charts, which is the same as the number of indices.

Line 6 is where you put all the data into the chart. daily\_df is the dataset that holds your daily power consumption data for each of the sites. The first part of the line selects the data that you’ll display in the chart. Line 7 inserts the data into the plot. Lines 8 and 9 set the labels on the charts.

**Listing 6.16. Creating charts to show each site over a period of months**

```
print('Number of time series:',daily_df.shape[1])     1
fig, axs = plt.subplots(
    5,
    2,
    figsize=(20, 20),
    sharex=True)                                      2
axx = axs.ravel()                                     3
indices = [0,1,2,3,4,5,40,41,42,43]                   4
for i in indices:                                     5
    plot_num = indices.index(i)                       6
    daily_df[daily_df.columns[i]].loc[
        "2017-11-01":"2018-01-31"].plot(
            ax=axx[plot_num])                         7
    axx[plot_num].set_xlabel("date")                  8
    axx[plot_num].set_ylabel("kW consumption")        9
```

* _1_ Displays the number of columns in the daily\_df dataset. This is 48, the total number of sites.
* _2_ Creates the Matplotlib figure to hold the 10 charts (5 rows of 2 charts)
* _3_ Ensures that the data will be stored as a series
* _4_ Identifies the 10 sites you want to chart
* _5_ Loops through each of the sites
* _6_ Gets each element, one by one, from the list indices
* _7_ Sets the data in the plot to the site referenced by the variable indicies
* _8_ Sets the label for the x-axis
* _9_ Sets the label for the y-axis

Now that you can see what your data looks like, you can create your training and test datasets.

**6.6.3. Part 3: Creating training and testing datasets**

DeepAR requires the data to be in JSON format. JSON is a very common data format that can be read by people and by machines.

A hierarchical structure that you commonly use is the folder system on your computer. When you store documents relating to different projects you are working on, you might create a folder for each project and put the documents relating to each project in that folder. That is a hierarchical structure.

In this chapter, you will create a JSON file with a simple structure. Instead of project folders holding project documents (like the previous folder example), each element in your JSON file will hold daily power consumption data for one site. Also, each element will hold two additional elements, as shown in [listing 6.17](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06ex17). The first element is start, which contains the date, and the second is target, which contains each day’s power consumption data for the site. Because your dataset covers 409 days, there are 409 elements in the target element.

**Listing 6.17. Sample JSON file**

```
{
    "start": "2017-11-01 00:00:00",
    "target": [
        1184.23,
        1210.9000000000003,
        1042.9000000000003,
        ...
        1144.2500000000002,
        1225.1299999999999
    ]
}
```

To create the JSON file, you need to take the data through a few transformations:

* Convert the data from a DataFrame to a list of series
* Withhold 30 days of data from the training dataset so you don’t train the model on data you are testing against
* Create the JSON files

The first transformation is converting the data from a DataFrame to a list of data series, with each series containing the power consumption data for a single site. The following listing shows how to do this.

**Listing 6.18. Converting a DataFrame to a list of series**

```
daily_power_consumption_per_site = []                 1
for column in daily_df.columns:                       2
    site_consumption = site_consumption.fillna(0)     3
    daily_power_consumption_per_site.append(
        site_consumption)                             4

print(f'Time series covers \
{len(daily_power_consumption_per_site[0])} days.')    5
print(f'Time series starts at \
{daily_power_consumption_per_site[0].index[0]}')      6
print(f'Time series ends at \
{daily_power_consumption_per_site[0].index[-1]}')     7
```

* _1_ Creates an empty list to hold the columns in the DataFrame
* _2_ Loops through the columns in the DataFrame
* _3_ Replaces any missing values with zeros
* _4_ Appends the column to the list
* _5_ Prints the number of days
* _6_ Prints the start date of the first site
* _7_ Prints the end date of the first site

In line 1 in the listing, you create a list to hold each of your sites. Each element of this list holds one column of the dataset. Line 2 creates a loop to iterate through the columns. Line 3 appends each column of data to the daily\_power\_consumption\_per \_site list you created in line 1. Lines 4 and 5 print the results so you can confirm that the conversion still has the same number of days and covers the same period as the data in the DataFrame.

Next, you set a couple of variables that will help you keep the time periods and intervals consistent throughout the notebook. The first variable is freq, which you set to D. _D_ stands for day, and it means that you are working with daily data. If you were working with hourly data, you’d use H, and monthly data is M.

You also set your _prediction period_. This is the number of days out that you want to predict. For example, in this notebook, the training dataset goes from November 1, 2017, to October 31, 2018, and you are predicting power consumption for November 2018. November has 30 days, so you set the prediction\_length to 30.

Once you have set the variables in lines 1 and 2 of [listing 6.19](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06ex19), you then define the start and end dates in a timestamp format. _Timestamp_ is a data format that stores dates, times, and frequencies as a single object. This allows for easy transformation from one frequency to another (such as daily to monthly) and easy addition and subtraction of dates and times.

In line 3, you set the start\_date of the dataset to November 1, 2017, and the end date of the training dataset to the end of October 2018. The end date of the test dataset is 364 days later, and the end date of the training dataset is 30 days after that. Notice that you can simply add days to the original timestamp, and the dates are automatically calculated.

**Listing 6.19. Setting the length of the prediction period**

```
freq = 'D'                                                 1
prediction_length = 30                                     2

start_date = pd.Timestamp(
    "2017-11-01 00:00:00",
    freq=freq)                                             3
end_training = start_date + 364                            4
end_testing = end_training + prediction_length             5

print(f'End training: {end_training}, End testing: {end_testing}')
```

* _1_ Sets frequency of the time series to day
* _2_ Sets prediction length to 30 days
* _3_ Sets start\_date as November 1, 2017
* _4_ Sets the end of the training dataset to October 31, 2018
* _5_ Sets the end of the test dataset to November 30, 2018

The DeepAR JSON input format represents each time series as a JSON object. In the simplest case (which you will use in this chapter), each time series consists of a start timestamp (start) and a list of values (target). The JSON input format is a JSON file that shows the daily power consumption for each of the 48 sites that Kiara is reporting on. The DeepAR model requires two JSON files: the first is the training data and the second is the test data.

Creating JSON files is a two-step process. First, you create a Python dictionary with a structure identical to the JSON file, and then you convert the Python dictionary to JSON and save the file.

To create the Python dictionary format, you loop through each of the daily\_power \_consumption\_per\_site lists you created in [listing 6.18](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06ex18) and set the start variable and _target_ list. [Listing 6.20](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06ex20) uses a type of Python loop called a _list comprehension_. The code between the open and close curly brackets (line 2 and 5 of [listing 6.20](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06ex20)) marks the start and end of each element in the JSON file shown in [listing 6.17](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06ex17). The code in lines 3 and 4 inserts the start date and a list of days from the training dataset.

Lines 1 and 7 mark the beginning and end of the list comprehension. The loop is described in line 6. The code states that the list ts will be used to hold each site as it loops through the daily\_power\_consumption\_per\_site list. That is why, in line 4, you see the variable ts\[start\_date:end\_training]. The code ts\[start\_date:end \_training] is a list that contains one site and all of the days in the range start\_date to end\_training that you set in [listing 6.19](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06ex19).

**Listing 6.20. Creating a Python dictionary in same structure as a JSON file**

```
training_data = [                                 1
    {                                             2
        "start": str(start_date),                 3
        "target": ts[
            start_date:end_training].tolist()     4
    }                                             5
    for ts in timeseries                          6
]                                                 7

test_data = [
    {
        "start": str(start_date),
        "target": ts[
            start_date:end_testing].tolist()      8
    }
    for ts in timeseries
]
```

* _1_ Creates a list of dictionary objects to hold the training data
* _2_ Sets the start of each dictionary object
* _3_ Sets the start date
* _4_ Creates a list of power consumption training data for one site
* _5_ Sets the end of each dictionary object
* _6_ List comprehension loop
* _7_ Sets the end of the training data
* _8_ Creates a list of power consumption test data for one site

Now that you have created two Python dictionaries called test\_data and training\_data, you need to save these as JSON files on S3 for DeepAR to work with. To do this, create a helper function that converts a Python dictionary to JSON and then apply that function to the test\_data and training\_data dictionaries, as shown in the following listing.

**Listing 6.21. Saving the JSON files to S3**

```
def write_dicts_to_s3(path, data):                    1
    with s3.open(path, 'wb') as f:                    2
        for d in data:                                3
            f.write(json.dumps(d).encode("utf-8"))    4
            f.write("\n".encode('utf-8'))             5

write_dicts_to_s3(
    f'{s3_data_path}/train/train.json',
    training_data)                                    6
write_dicts_to_s3(
    f'{s3_data_path}/test/test.json',
    test_data)                                        7
```

* _1_ Creates a function that writes the dictionary data to S3
* _2_ Opens an S3 file object
* _3_ Loops through the data
* _4_ Writes the dictionary object in JSON format
* _5_ Writes a newline character so that each dictionary object starts on a new line
* _6_ Applies the function to the training data
* _7_ Applies the function to the test data

Your training and test data are now stored on S3 in JSON format. With that, the data is in a SageMaker session and you are ready to start training the model.

**6.6.4. Part 4: Training the model**

Now that you have saved the data on S3 in JSON format, you can start training the model. As shown in the following listing, the first step is to set up some variables that you will hand to the estimator function that will build the model.

**Listing 6.22. Setting up a server to train the model**

```
s3_output_path = \
    f's3://{data_bucket}/{subfolder}/output'                   1
sess = sagemaker.Session()                                     2
image_name = sagemaker.amazon.amazon_estimator.get_image_uri(
    sess.boto_region_name,
    "forecasting-deepar",
    "latest")                                                  3
```

* _1_ Sets the path to save the machine learning model
* _2_ Creates a variable to hold the SageMaker session
* _3_ Tells AWS to use the forecasting-deepar image to create the model

Next, you hand the variables to the estimator ([listing 6.23](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06ex23)). This sets up the type of machine that SageMaker will fire up to create the model. You will use a single instance of a c5.2xlarge machine. SageMaker creates this machine, starts it, builds the model, and shuts it down automatically. The cost of this machine is about US$0.47 per hour. It will take about 3 minutes to create the model, which means it will cost only a few cents.

**Listing 6.23. Setting up an estimator to hold training parameters**

```
estimator = sagemaker.estimator.Estimator(    1
    sagemaker_session=sess,                   2
    image_name=image_name,                    3
    role=role,                                4
    train_instance_count=1,                   5
    train_instance_type='ml.c5.2xlarge',      6
    base_job_name='ch6-energy-usage',         7
    output_path=s3_output_path                8
)
```

* _1_ Creates the estimator variable for the model
* _2_ Applies the SageMaker session
* _3_ Sets the image
* _4_ Gives the image a role that allows it to run
* _5_ Sets up a single instance for the training machine
* _6_ Sets the size of the machine the model will be trained on
* _7_ Nominates a name for the job
* _8_ Saves the model to the output location you set up on S3

Richie’s note on SageMaker instance types

Throughout this book, you will notice that we have chosen to use the instance type ml.m4.xlarge for all training and inference instances. The reason behind this decision was simply that usage of these instance types was included in Amazon’s free tier at the time of writing. (For details on Amazon’s current inclusions in the free tier, see [https://aws.amazon.com/free](https://aws.amazon.com/free).)

For all the examples provided in this book, this instance is more than adequate. But what should you be using in your workplace if your problem is more complex and/or your dataset is much larger than the ones we have presented? There are no hard and fast rules, but here are a few guidelines:

* See the SageMaker examples on the Amazon website for the algorithm you are using. Start with the Amazon example as your default.
* Make sure you calculate how much your chosen instance type is actually costing you for training and inference ([https://aws.amazon.com/sagemaker/pricing](https://aws.amazon.com/sagemaker/pricing)).
* If you have a problem with training or inference cost or time, don’t be afraid to experiment with different instance sizes.
* Be aware that quite often a very large and expensive instance can actually cost less to train a model than a smaller one, as well as run in much less time.
* XGBoost runs in parallel when training on a compute instance but does not benefit at all from a GPU instance, so don’t waste time on a GPU-based instance (p3 or accelerated computing) for training or inference. However, feel free to try an m5.24xlarge or m4.16xlarge in training. It might actually be cheaper!
* Neural-net-based models will benefit from GPU instances in training, but these usually should not be required for inference as these are exceedingly expensive.
* Your notebook instance is most likely to be memory constrained if you use it a lot, so consider an instance with more memory if this becomes a problem for you. Just be aware that you are paying for every hour the instance is running even if you are not using it!

Once you set up the estimator, you then need to set its parameters. SageMaker exposes several parameters for you. The only two that you need to change are the last two parameters shown in lines 7 and 8 of [listing 6.24](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06ex24): context\_length and prediction\_length.

The _context length_ is the minimum period of time that will be used to make a prediction. By setting this to 90, you are saying that you want DeepAR to use 90 days of data as a minimum to make its predictions. In business settings, this is typically a good value as it allows for the capture of quarterly trends. The _prediction length_ is the period of time you are predicting. For this notebook, you are predicting November data, so you use the prediction\_length of 30 days.

**Listing 6.24. Inserting parameters for the estimator**

```
estimator.set_hyperparameters(                   1
    time_freq=freq,                              2
    epochs="400",                                3
    early_stopping_patience="40",                4
    mini_batch_size="64",                        5
    learning_rate="5E-4",                        6
    context_length="90",                         7
    prediction_length=str(prediction_length)     8
)
```

* _1_ Sets the hyperparameters
* _2_ Sets the frequency to daily
* _3_ Sets the epochs to 400 (leave this value as is)
* _4_ Sets the early stopping to 40 (leave this value as is)
* _5_ Sets the batch size to 64 (leave this value as is)
* _6_ Sets the learning rate to 0.0005 (the decimal conversion of the exponential value 5E-4)
* _7_ Sets the context length to 90 days
* _8_ Sets the prediction length to 30 days

Now you train the model. This can take 5 to 10 minutes.

**Listing 6.25. Training the model**

```
%%time
data_channels = {                                 1
    "train": "{}/train/".format(s3_data_path),    2
    "test": "{}/test/".format(s3_data_path)       3
}
estimator.fit(inputs=data_channels, wait=True)    4
```

* _1_ Pulls in the training and test data to create the model
* _2_ Training data
* _3_ Test data
* _4_ Runs the estimator function that creates the model

After this code runs, the model is trained, so now you can host it on SageMaker so it is ready to make decisions.

**6.6.5. Part 5: Hosting the model**

Hosting the model involves several steps. First, in [listing 6.26](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06ex26), you delete any existing endpoints you have so you don’t end up paying for a bunch of endpoints you aren’t using.

**Listing 6.26. Deleting existing endpoints**

```
endpoint_name = 'energy-usage'
try:
    sess.delete_endpoint(
            sagemaker.predictor.RealTimePredictor(
                endpoint=endpoint_name).endpoint,
                delete_endpoint_config=True)
    print(
        'Warning: Existing endpoint deleted to make way for new endpoint.')
    from time import sleep
    sleep(10)
except:
    passalexample>
```

Next is a code cell that you don’t really need to know anything about. This is a helper class prepared by Amazon to allow you to review the results of the DeepAR model as a pandas DataFrame rather than as JSON objects. It is a good bet that in the future they will make this code part of the M library. For now, just run it by clicking into the cell and pressing ![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ctrl\_ent.jpg).

You are now at the stage where you set up your endpoint to make predictions ([listing 6.27](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06ex27)). You will use an m5.large machine as it represents a good balance of power and price. As of March 2019, AWS charges 13.4 US cents per hour for the machine. So if you keep the endpoint up for a day, the total cost will be US$3.22.

**Listing 6.27. Setting up the predictor class**

```
%%time
predictor = estimator.deploy(          1
    initial_instance_count=1,          2
    instance_type='ml.m5.large',       3
    predictor_cls=DeepARPredictor,     4
    endpoint_name=endpoint_name)       5
```

* _1_ Deploys the estimator to a variable named predictor
* _2_ Sets it up on a single machine
* _3_ Uses an m5.large machine
* _4_ Uses the DeepARPredictor class to return the results as a pandas DataFrame
* _5_ Names the endpoint energy-usage

You are ready to start making predictions.

**6.6.6. Part 6: Making predictions and plotting results**

In the remainder of the notebook, you will do three things:

* Run a prediction for a month that shows the 50th percentile (most likely) prediction and also displays the range of prediction between two other percentiles. For example, if you want to show an 80% confidence range, the prediction will also show you the lower and upper range that falls within an 80% confidence level.
* Graph the results so that you can easily describe the results.
* Run a prediction across all the results for the data in November 2018. This data was not used to train the DeepAR model, so it will demonstrate how accurate the model is.

**PREDICTING POWER CONSUMPTION FOR A SINGLE SITE**

To predict power consumption for a single site, you just pass the site details to the predictor function. In the following listing, you are running the predictor against data from site 1.

**Listing 6.28. Setting up the predictor class**

```
predictor.predict(ts=daily_power_consumption_per_site[0]
                    [start_date+30:end_training],
                  quantiles=[0.1, 0.5, 0.9]).head()        1
```

* _1_ Runs the predictor function against the first site

[Table 6.6](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06table06) shows the result of running the prediction against site 1. The first column shows the day, and the second column shows the prediction from the 10th percentile of results. The third column shows the 50th percentile prediction, and the last column shows the 90th percentile prediction.

**Table 6.6. Predicting power usage data for site 1 of Kiara’s companies**

| Day        | 0.1         | 0.5         | 0.9         |
| ---------- | ----------- | ----------- | ----------- |
| 2018-11-01 | 1158.509766 | 1226.118042 | 1292.315430 |
| 2018-11-02 | 1154.938232 | 1225.540405 | 1280.479126 |
| 2018-11-03 | 1119.561646 | 1186.360962 | 1278.330200 |

Now that you have a way to generate predictions, you can chart the predictions.

**CHARTING THE PREDICTED POWER CONSUMPTION FOR A SINGLE SITE**

The code in [listing 6.29](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06ex29) shows a function that allows you to set up charting. It is similar to the code you worked with in the practice notebook but has some additional complexities that allow you to display graphically the range of results.

**Listing 6.29. Setting up a function that allows charting**

```
def plot(                                         1
    predictor,
    target_ts,
    end_training=end_training,
    plot_weeks=12,
    confidence=80
):
    print(f"Calling served model to generate predictions starting from \
{end_training} to {end_training+prediction_length}")

    low_quantile = 0.5 - confidence * 0.005       2
    up_quantile = confidence * 0.005 + 0.5        3

    plot_history = plot_weeks * 7                 4

    fig = plt.figure(figsize=(20, 3))             5
    ax = plt.subplot(1,1,1)                       6

    prediction = predictor.predict(
        ts=target_ts[:end_training],
        quantiles=[
            low_quantile, 0.5, up_quantile])      7

    target_section = target_ts[
        end_training-plot_history:\
        end_training+prediction_length]           8

    target_section.plot(
        color="black",
        label='target')                           9

    ax.fill_between(                              10
        prediction[str(low_quantile)].index,
        prediction[str(low_quantile)].values,
        prediction[str(up_quantile)].values,
        color="b",
        alpha=0.3,
        label=f'{confidence}% confidence interval'
    )
    prediction["0.5"].plot(
        color="b",
        label='P50')                              11

    ax.legend(loc=2)                              12
    ax.set_ylim(
        target_section.min() * 0.5,
        target_section.max() * 1.5)               13
```

* _1_ Sets the argument for the plot function
* _2_ Calculates the lower range
* _3_ Calculates the upper range
* _4_ Calculates the number of days based on the plot\_weeks argument
* _5_ Sets the size of the chart
* _6_ Sets a single chart to display
* _7_ The prediction function
* _8_ Sets the actual values
* _9_ Sets the color of the line for the actual values
* _10_ Sets the color of the range
* _11_ Sets the color of the prediction
* _12_ Creates the legend
* _13_ Sets the scale

Line 1 of the code creates a function called plot that lets you create a chart of the data for each site. The plot function takes three arguments:

* predictor—The predictor that you ran in [listing 6.28](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06ex28), which generates predictions for the site
* plot\_weeks—The number of weeks you want to display in your chart
* confidence—The confidence level for the range that is displayed in the chart

In lines 2 and 3 in [listing 6.29](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06ex29), you calculate the confidence range you want to display from the confidence value you entered as an argument in line 1. Line 4 calculates the number of days based on the plot\_weeks argument. Lines 5 and 6 set the size of the plot and the subplot. (You are only displaying a single plot.) Line 7 runs the prediction function on the site. Lines 8 and 9 set the date range for the chart and the color of the actual line. In line 10, you set the prediction range that will display in the chart, and in line 11 you define the prediction line. Finally, lines 12 and 13 set the legend and the scale of the chart.

**Note**

In this function to set up charting, we use global variables that were set earlier in the notebook. This is not ideal but keeps the function a little simpler for the purposes of this book.

[Listing 6.30](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06ex30) runs the function. The chart shows the actual and prediction data for a single site. This listing uses site 34 as the charted site, shows a period of 8 weeks before the 30-day prediction period, and defines a confidence level of 80%.

**Listing 6.30. Running the function that creates the chart**

```
site_id = 34           1
plot_weeks = 8         2
confidence = 80        3
plot(                  4
        predictor,
        target_ts=daily_power_consumption_per_site[
            site_id][start_date+30:],
        plot_weeks=plot_weeks,
        confidence=confidence
    )ormalexample>
```

* _1_ Sets the site for charting
* _2_ Sets the number of weeks to include in the chart
* _3_ Sets the confidence level at 80%
* _4_ Runs the plot function

[Figure 6.8](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06fig08) shows the chart you have produced. You can use this chart to show the predicted usage patterns for each of Kiara’s sites.

**Figure 6.8. Chart showing predicted versus actual power consumption for November 2018, for one of Kiara’s sites**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch06fig08\_alt.jpg)

**CALCULATING THE ACCURACY OF THE PREDICTION ACROSS ALL SITES**

Now that you can see one of the sites, it’s time to calculate the error across all the sites. To express this, you use a Mean Absolute Percentage Error (MAPE). This function takes the difference between the actual value and the predicted value and divides it by the actual value. For example, if an actual value is 50 and the predicted value is 45, you’d subtract 45 from 50 to get 5, and then divide by 50 to get 0.1. Typically, this is expressed as a percentage, so you multiply that number by 100 to get 10%. So the MAPE of an actual value of 50 and a predicted value of 45 is 10%.

The first step in calculating the MAPE is to run the predictor across all the data for November 2018 and get the actual values (usages) for that month. [Listing 6.31](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06ex31) shows how to do this.

In line 5, you see a function that we haven’t used yet in the book: the zip function. This is a very useful piece of code that allows you to loop through two lists concurrently and do interesting things with the paired items from each list. In this listing, the interesting thing you’ll do is to print the actual value compared to the prediction.

**Listing 6.31. Running the predictor**

```
predictions= []
for i, ts in enumerate(
    daily_power_consumption_per_site):                  1

    print(i, ts[0])
    predictions.append(
        predictor.predict(
            ts=ts[start_date+30:end_training]
            )['0.5'].sum())                             2

usages = [ts[end_training+1:end_training+30].sum() \
    for ts in daily_power_consumption_per_site]         3

for p,u in zip(predictions,usages):                     4
    print(f'Predicted {p} kwh but usage was {u} kwh,')
```

* _1_ Loops through daily site power consumption
* _2_ Runs predictions for the month of November
* _3_ Gets the usages
* _4_ Prints usages and predictions

The next listing shows the code that calculates the MAPE. Once the function is defined, you will use it to calculate the MAPE.

**Listing 6.32. Calculating the Mean Absolute Percentage Error (MAPE)**

```
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
```

The following listing runs the MAPE function across all the usages and predictions for the 30 days in November 2018 and returns the mean MAPE across all the days.

**Listing 6.33. Running the MAPE function**

```
print(f'MAPE: {round(mape(usages, predictions),1)}%')
```

The MAPE across all the days is 5.7%, which is pretty good, given that you have not yet added weather data. You’ll do that in [chapter 7](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07). Also in [chapter 7](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07), you’ll get to work with a longer period of data so the DeepAR algorithm can begin to detect annual trends.

#### 6.7. Deleting the endpoint and shutting down your notebook instance <a href="#ch06lev1sec7__title" id="ch06lev1sec7__title"></a>

It is important that you shut down your notebook instance and delete your endpoint. We don’t want you to get charged for SageMaker services that you’re not using.

**6.7.1. Deleting the endpoint**

Appendix D describes how to shut down your notebook instance and delete your endpoint using the SageMaker console, or you can do that with the following code.

**Listing 6.34. Deleting the notebook**

```
# Remove the endpoint (optional)
# Comment out this cell if you want the endpoint to persist after Run All
sagemaker.Session().delete_endpoint(rcf_endpoint.endpoint)
```

To delete the endpoint, uncomment the code in the listing, then click ![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ctrl\_ent.jpg) to run the code in the cell.

**6.7.2. Shutting down the notebook instance**

To shut down the notebook, go back to your browser tab where you have SageMaker open. Click the Notebook Instances menu item to view all of your notebook instances. Select the radio button next to the notebook instance name as shown in [figure 6.9](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06fig09), then select Stop from the Actions menu. It takes a couple of minutes to shut down.

**Figure 6.9. Shutting down the notebookx**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch06fig09\_alt.jpg)

#### 6.8. Checking to make sure the endpoint is deleted <a href="#ch06lev1sec8__title" id="ch06lev1sec8__title"></a>

If you didn’t delete the endpoint using the notebook (or if you just want to make sure it is deleted), you can do this from the SageMaker console. To delete the endpoint, click the radio button to the left of the endpoint name, then click the Actions menu item and click Delete in the menu that appears.

When you have successfully deleted the endpoint, you will no longer incur AWS charges for it. You can confirm that all of your endpoints have been deleted when you see the text “There are currently no resources” displayed on the Endpoints page ([figure 6.10](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06fig10)).

**Figure 6.10. Endpoint deleted**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch06fig10\_alt.jpg)

Kiara can now predict power consumption for each site with a 5.7% MAPE and, as importantly, she can take her boss through the charts to show what she is predicting to occur in each site.

#### Summary <a href="#ch06lev1sec9__title" id="ch06lev1sec9__title"></a>

* Time-series data consists of a number of observations at particular intervals. You can visualize time-series data as line charts.
* Jupyter notebooks and the pandas library are excellent tools for transforming time-series data and for creating line charts of the data.
* Matplotlib is a Python charting library.
* Instructions to Jupyter begin with a % symbol. When you see a command in a Jupyter notebook that starts with % or with %%, it’s known as a magic command-.
* A for loop is the most common type of loop you’ll use in data analysis and machine learning. The enumerate function lets you keep track of how many times you have looped through a list.
* A neural network (sometimes referred to as _deep learning_) is an example of supervised machine learning.
* You can build a neural network using SageMaker’s DeepAR model.
* DeepAR is Amazon’s time-series neural network algorithm that takes as its input related types of time-series data and automatically combines the data into a global model of all time series in a dataset to predict future events.
* You can use DeepAR to predict power consumption.
