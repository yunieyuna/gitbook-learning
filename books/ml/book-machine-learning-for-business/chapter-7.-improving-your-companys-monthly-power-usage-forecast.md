# Chapter 7. Improving your company’s monthly power usage forecast

_This chapter covers_

* Adding additional data to your analysis
* Using pandas to fill in missing values
* Visualizing your time-series data
* Using a neural network to generate forecasts
* Using DeepAR to forecast power consumption

In [chapter 6](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06), you worked with Kiara to develop an AWS SageMaker DeepAR model to predict power consumption across her company’s 48 sites. You had just a bit more than one year’s data for each of the sites, and you predicted the temperature for November 2018 with an average percentage error of less that 6%. Amazing! Let’s expand on this scenario by adding additional data for our analysis and filling in any missing values. First, let’s take a deeper look at DeepAR.

#### 7.1. DeepAR’s ability to pick up periodic events <a href="#ch07lev1sec1__title" id="ch07lev1sec1__title"></a>

The DeepAR algorithm was able to identify patterns such as weekly trends in our data from [chapter 6](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06). [Figure 7.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07fig01) shows the predicted and actual usage for site 33 in November. This site follows a consistent weekly pattern.

**Figure 7.1. Predicted versus actual consumption from site 33 using the DeepAR model you built in** [**chapter 6**](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06)

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch07fig01\_alt.jpg)

You and Kiara are heroes. The company newsletter included a two-page spread showing you and Kiara with a large printout of your predictions for December. Unfortunately, when January came around, anyone looking at that photo would have noticed that your predictions weren’t that accurate for December. Fortunately for you and Kiara, not many people noticed because most staff take some holiday time over Christmas, and some of the sites had a mandatory shutdown period.

“Wait a minute!” You and Kiara said at the same time as you were discussing why your December predictions were less accurate. “With staff taking time off and mandatory shut-downs, it’s no wonder December was way off.”

When you have rare but still regularly occurring events like a Christmas shutdown in your time-series data, your predictions will still be accurate, provided you have enough historical data for the machine learning model to pick up the trend. You and Kiara would need several years of power consumption data for your model to pick up a Christmas shutdown trend. But you don’t have this option because the smart meters were only installed in November 2017. So what do you do?

Fortunately for you (and Kiara), SageMaker DeepAR is a neural network that is particularly good at incorporating several different time-series datasets into its forecasting. And these can be used to account for events in your time-series forecasting that your time-series data can’t directly infer.

To demonstrate how this works, [figure 7.2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07fig02) shows time-series data covering a typical month. The x-axis shows days per month. The y-axis is the amount of power consumed on each day. The shaded area is the predicted power consumption with an 80% confidence interval. An 80% confidence interval means that 4 out of every 5 days will fall within this range. The black line shows the actual power consumption for that day. In [figure 7.2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07fig02), you can see that the actual power consumption was within the confidence interval for every day of the month.

**Figure 7.2. Actual versus predicted usage during a normal month**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch07fig02\_alt.jpg)

[Figure 7.3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07fig03) shows a month with a shutdown from the 10th to the 12th day of the month. You can see that the actual power consumption dropped on these days, but the predicted power consumption did not anticipate this.

**Figure 7.3. Actual versus predicted usage during a month with a shutdown**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch07fig03\_alt.jpg)

There are three possible reasons why the power consumption data during this shutdown was not correctly predicted. First, the shutdown _is_ a regularly occurring event, but there is not enough historical data for the DeepAR algorithm to pick up the recurring event. Second, the shutdown _is not_ a recurring event (and so can’t be picked up in the historical data) but is an event that can be identified through other datasets. An example of this is a planned shutdown where Kiara’s company is closing a site for a few days in December. Although the historical dataset won’t show the event, the impact of the event on power consumption can be predicted if the model incorporated planned staff schedules as one of its time series. We’ll discuss this more in the next section. Finally, the shutdown is not planned, and there is no dataset that could be incorporated to show the shutdown. An example of this is a work stoppage due to an employee strike. Unless your model can predict labor activism, there is not much your machine learning model can do to predict power consumption during these periods.

#### 7.2. DeepAR’s greatest strength: Incorporating related time series <a href="#ch07lev1sec2__title" id="ch07lev1sec2__title"></a>

To help the DeepAR model predict trends, you need to provide it with additional data that shows trends. As an example, you know that during the shutdown periods, only a handful of staff are rostered. If you could feed this data into the DeepAR algorithm, then it could use this information to predict power consumption during shutdown periods.\[[1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07fn1)]

> 1
>
> You can read more about DeepAR on the AWS site: [https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html).

[Figure 7.4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07fig04) shows the number of staff rostered in a month during a shutdown. You can see that for most days, there are between 10 and 15 staff members at work, but on the 10th, 11th, and 12th, there are only 4 to 6 staff members.

**Figure 7.4. Number of staff rostered during a month with a shutdown**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch07fig04\_alt.jpg)

If you could incorporate this time series into the DeepAR model, you would better predict upcoming power consumption. [Figure 7.5](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07fig05) shows the prediction when you use both historical consumption and rostering data in your DeepAR model.

**Figure 7.5. Power consumption predictions incorporating historical and staff roster data**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch07fig05\_alt.jpg)

In this chapter, you’ll learn how to incorporate additional datasets into your DeepAR model to improve the accuracy of the model in the face of known upcoming events that are either not periodic or are periodic, but you don’t have enough historical data for the model to incorporate into its predictions.

#### 7.3. Incorporating additional datasets into Kiara’s power consumption model <a href="#ch07lev1sec3__title" id="ch07lev1sec3__title"></a>

In [chapter 6](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06), you helped Kiara build a DeepAR model that predicted power consumption across each of the 41 sites owned by her company. The model worked well when predicting power consumption in November, but performed less well when predicting December’s consumption because some of the sites were on reduced operating hours or shut down altogether.

Additionally, you noticed that there were seasonal fluctuations in power usage that you attributed to changes in temperature, and you noticed that different types of sites had different usage patterns. Some types of sites were closed every weekend, whereas others operated consistently regardless of the day of the week. After discussing this with Kiara, you realized that some of the sites were retail sites, whereas others were industrial or transport-related areas.

In this chapter, the notebook you’ll build will incorporate this data. Specifically, you’ll add the following datasets to the power consumption metering data you used in [chapter 6](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06):

* _Site categories_—Indicates retail, industrial, or transport site
* _Site holidays_—Indicates whether a site has a planned shutdown
* _Site maximum temperatures_—Lists the maximum temperature forecast for each site each day

Then you’ll train the model using these three datasets.

Different types of datasets

The three datasets used in this chapter can be classified into two types of data:

* _Categorical_—Information about the site that doesn’t change. The dataset site categories, for example, contains categorical data. (A site is a retail site and will likely always be a retail site.)
* _Dynamic_—Data that changes over time. Holidays and forecasted maximum temperatures are examples of dynamic data.

When predicting power consumption for the month of December, you’ll use a schedule of planned holidays for December and the forecasted temperature for that month.

#### 7.4. Getting ready to build the model <a href="#ch07lev1sec4__title" id="ch07lev1sec4__title"></a>

As in previous chapters, you need to do the following to set up another notebook in SageMaker and fine tune your predictions:

1. From S3, download the notebook we prepared for this chapter.
2. Set up the folder to run the notebook on AWS SageMaker.
3. Upload the notebook to AWS SageMaker.
4. Download the datasets from your S3 bucket.
5. Create a folder in your S3 bucket to store the datasets.
6. Upload the datasets to your AWS S3 bucket.

Given that you’ve followed these steps in each of the previous chapters, we’ll move quickly through them in this chapter.

**7.4.1. Downloading the notebook we prepared**

We prepared the notebook you’ll use in this chapter. You can download it from this location:

[https://s3.amazonaws.com/mlforbusiness/ch07/energy\_consumption\_additional\_datasets.ipynb](https://s3.amazonaws.com/mlforbusiness/ch07/energy\_consumption\_additional\_datasets.ipynb)

Save this file on your computer. In step 3, you’ll upload it to SageMaker.

**7.4.2. Setting up the folder on SageMaker**

Go to AWS SageMaker at [https://console.aws.amazon.com/sagemaker/home](https://console.aws.amazon.com/sagemaker/home), and select Notebook Instances from the left-hand menu. If your instance is stopped, you’ll need to start it. Once it is started, click Open Jupyter.

This opens a new tab and shows you a list of folders in SageMaker. If you have been following along in earlier chapters, you will have a folder for each of the earlier chapters. Create a new folder for this chapter. We’ve called our folder ch07.

**7.4.3. Uploading the notebook to SageMaker**

Click the folder you’ve just created, and click Upload to upload the notebook. Select the notebook you downloaded in step 1, and upload it to SageMaker. [Figure 7.6](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07fig06) shows what your SageMaker folder might look like after uploading the notebook.

**Figure 7.6. Viewing the uploaded energy\_consumption\_additional\_datasets notebook on S3**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch07fig06\_alt.jpg)

**7.4.4. Downloading the datasets from the S3 bucket**

We stored the datasets for this chapter in one of our S3 buckets. You can download each of the datasets by clicking the following links:

* _Meter data_—[https://s3.amazonaws.com/mlforbusiness/ch07/meter\_data\_daily.csv](https://s3.amazonaws.com/mlforbusiness/ch07/meter\_data\_daily.csv)
* _Site categories_—[https://s3.amazonaws.com/mlforbusiness/ch07/site\_categories.csv](https://s3.amazonaws.com/mlforbusiness/ch07/site\_categories.csv)
* _Site holidays_—[https://s3.amazonaws.com/mlforbusiness/ch07/site\_holidays.csv](https://s3.amazonaws.com/mlforbusiness/ch07/site\_holidays.csv)
* _Site maximums_—[https://s3.amazonaws.com/mlforbusiness/ch07/site\_maximums.csv](https://s3.amazonaws.com/mlforbusiness/ch07/site\_maximums.csv)

The power consumption data you’ll use in this chapter is provided by BidEnergy ([http://www.bidenergy.com](http://www.bidenergy.com/)), a company that specializes in power-usage forecasting and in minimizing power expenditure. The algorithms used by BidEnergy are more sophisticated than you’ll see in this chapter, but you’ll still get a feel for how machine learning in general, and neural networks in particular, can be applied to forecasting problems.

**7.4.5. Setting up a folder on S3 to hold your data**

In AWS S3, go to the bucket you created to hold your data in earlier chapters, and create another folder. You can see a list of your buckets at this link:

[https://s3.console.aws.amazon.com/s3/buckets](https://s3.console.aws.amazon.com/s3/buckets)

The bucket we are using to hold our data is called mlforbusiness. Your bucket will be called something else (a name of your choosing). Once you click into your bucket, create a folder to store your data, naming it something like _ch07_.

**7.4.6. Uploading the datasets to your AWS bucket**

After creating the folder on S3, upload the datasets you downloaded in step 4. [Figure 7.7](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07fig07) shows what your S3 folder might look like.

**Figure 7.7. Viewing the uploaded CSV datasets on S3**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch07fig07.jpg)

#### 7.5. Building the model <a href="#ch07lev1sec5__title" id="ch07lev1sec5__title"></a>

With the data uploaded to S3 and the notebook uploaded to SageMaker, you can now start to build the model. As in previous chapters, you’ll go through the following steps:

1. Set up the notebook.
2. Import the datasets.
3. Get the data into the right shape.
4. Create training and test datasets.
5. Configure the model and build the server.
6. Make predictions and plot results.

**7.5.1. Part 1: Setting up the notebook**

[Listing 7.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07ex1) shows your notebook setup. You will need to change the values in line 1 to the name of the S3 bucket you created on S3, then change line 2 to the subfolder of that bucket where you saved the data. Line 3 sets the location of the training and test data created in this notebook, and line 4 sets the location where the model is stored.

**Listing 7.1. Setting up the notebook**

```
data_bucket = 'mlforbusiness'                  1
subfolder = 'ch07'                             2
s3_data_path = \
    f"s3://{data_bucket}/{subfolder}/data"     3
s3_output_path = \
    f"s3://{data_bucket}/{subfolder}/output"   4
```

* _1_ S3 bucket where the data is stored
* _2_ Subfolder of the bucket where the data is stored
* _3_ Path where training and test data will be stored
* _4_ Path where model will be stored

The next listing imports the modules required by the notebook. This is the same as the imports used in [chapter 6](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06), so we won’t review these here.

**Listing 7.2. Importing Python modules and libraries**

```
%matplotlib inline

from dateutil.parser import parse
import json
import random
import datetime
import os

import pandas as pd
import boto3
import s3fs
import sagemaker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

role = sagemaker.get_execution_role()
s3 = s3fs.S3FileSystem(anon=False)
s3_data_path = f"s3://{data_bucket}/{subfolder}/data"
s3_output_path = f"s3://{data_bucket}/{subfolder}/output"
```

With that done, you are now ready to import the datasets.

**7.5.2. Part 2: Importing the datasets**

Unlike other chapters, in this notebook, you’ll upload four datasets for the meter, site categories, holidays, and maximum temperatures. The following listing shows how to import the meter data.

**Listing 7.3. Importing the meter data**

```
daily_df = pd.read_csv(
    f's3://{data_bucket}/{subfolder}/meter_data_daily.csv',
    index_col=0,
    parse_dates=[0])
daily_df.index.name = None
daily_df.head()le>
```

The meter data you use in this chapter has a few more months of observations. In [chapter 6](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06), the data ranged from October 2017 to October 2018. This dataset contains meter data from November 2017 to February 2019.

**Listing 7.4. Displaying information about the meter data**

```
print(daily_df.shape)
print(f'timeseries starts at {daily_df.index[0]} \
and ends at {daily_df.index[-1]}')
```

[Listing 7.5](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07ex5) shows how to import the site categories data. There are three types of sites:

* Retail
* Industrial
* Transport

**Listing 7.5. Displaying information about the site categories**

```
category_df = pd.read_csv
    (f's3://{data_bucket}/{subfolder}/site_categories.csv',
    index_col=0
    ).reset_index(drop=True)
print(category_df.shape)
print(category_df.Category.unique())
category_df.head()
```

oIn [listing 7.6](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07ex6), you import the holidays. Working days and weekends are marked with a 0; holidays are marked with a 1. There is no need to mark all the weekends as holidays because DeepAR can pick up that pattern from the site meter data. Although you don’t have enough site data for DeepAR to identify annual patterns, DeepAR can work out the pattern if it has access to a dataset that shows holidays at each of the sites.

**Listing 7.6. Displaying information about holidays for each site**

```
holiday_df = pd.read_csv(
    f's3://{data_bucket}/{subfolder}/site_holidays.csv',
    index_col=0,
    parse_dates=[0])
print(holiday_df.shape)
print(f'timeseries starts at {holiday_df.index[0]} \
and ends at {holiday_df.index[-1]}')
holiday_df.loc['2018-12-22':'2018-12-27']
```

[Listing 7.7](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07ex7) shows the maximum temperature reached each day for each of the sites. The sites are located in Australia, so energy usage increases as temperatures rise in the summer due to air conditioning; whereas, in more temperate climates, energy usage increases more as temperatures drop below zero degrees centigrade in the winter due to heating.

**Listing 7.7. Displaying information about maximum temperatures for each site**

```
max_df = pd.read_csv(
    f's3://{data_bucket}/{subfolder}/site_maximums.csv',
    index_col=0,
    parse_dates=[0])
print(max_df.shape)
print(f'timeseries starts at {max_df.index[0]} \
and ends at {max_df.index[-1]}')
```

With that, you are finished loading data into your notebook. To recap, for each site for each day from November 1, 2018, to February 28, 2019, you loaded data from CSV files for

* Energy consumption
* Site category (Retail, Industrial, or Transport)
* Holiday information (1 represents a holiday and 0 represents a working day or normal weekend)
* Maximum temperatures reached on the site

You will now get the data into the right shape to train the DeepAR model.

**7.5.3. Part 3: Getting the data into the right shape**

With your data loaded into DataFrames, you can now get each of the datasets ready for training the DeepAR model. The shape of each of the datasets is the same: each site is represented by a column, and each day is represented by a row.

In this section, you’ll ensure that there are no problematic missing values in each of the columns and each of the rows. DeepAR is very good at handling missing values in training data but cannot handle missing values in data it uses for predictions. To ensure that you don’t have annoying errors when running predictions, you fill in missing values in your prediction range. You’ll use November 1, 2018, to January 31, 2019, to train the data, and you’ll use December 1, 2018, to February 28, 2019, to test the model. This means that for your prediction range, there cannot be any missing data from December 1, 2018, to February 28, 2019. The following listing replaces any zero values with None and then checks for missing energy consumption data.

**Listing 7.8. Checking for missing energy consumption data**

```
daily_df = daily_df.replace([0],[None])
daily_df[daily_df.isnull().any(axis=1)].index
```

You can see from the output that there are several days in November 2018 with missing data because that was the month the smart meters were installed, but there are no days with missing data after November 2018. This means you don’t need to do anything further with this dataset because there’s no missing prediction data.

The next listing checks for missing category data. Again, there is no missing category data, so you can move on to holidays and missing maximum temperatures.

**Listing 7.9. Checking for missing category data and holiday data**

```
print(f'{len(category_df[category_df.isnull().any(axis=1)])} \
sites with missing categories.')
print(f'{len(holiday_df[holiday_df.isnull().any(axis=1)])} \
days with missing holidays.')
```

The following listing checks for missing maximum temperature data. There are several days without maximum temperature values. This is a problem, but one that can be easily solved.

**Listing 7.10. Checking for missing maximum temperature data**

```
print(f'{len(max_df[max_df.isnull().any(axis=1)])} \
days with missing maximum temperatures.')
```

The next listing uses the interpolate function to fill in missing data for a time series. In the absence of other information, the best way to infer missing values for a temperature time series like this is straight line interpolation based on time.

**Listing 7.11. Fixing missing maximum temperature data**

```
max_df = max_df.interpolate(method='time')              1
print(f'{len(max_df[max_df.isnull().any(axis=1)])} \
days with missing maximum temperatures. Problem solved!')
```

* _1_ Interpolates missing values

To ensure you are looking at data similar to the data we used in [chapter 6](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06), take a look at the data visually. In [chapter 6](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06), you learned about using Matplotlib to display multiple plots. As a refresher, [listing 7.12](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07ex12) shows the code for displaying multiple plots. Line 1 sets the shape of the plots as 6 rows by 2 columns. Line 2 creates a series that can be looped over. Line 3 sets which 12 sites will be displayed. And lines 4 through 7 set the content of each plot.

**Listing 7.12. Fixing missing maximum temperature data**

```
print('Number of timeseries:',daily_df.shape[1])
fig, axs = plt.subplots(
    6,
    2,
    figsize=(20, 20),
    sharex=True)                                1
axx = axs.ravel()                               2
indices = [0,1,2,3,26,27,33,39,42,43,46,47]     3
for i in indices:
    plot_num = indices.index(i)                 4
    daily_df[daily_df.columns[i]].loc[
        "2017-11-01":"2019-02-28"
        ].plot(ax=axx[plot_num])                5
    axx[plot_num].set_xlabel("date")            6
    axx[plot_num].set_ylabel("kW consumption")  7
```

* _1_ Sets the shape as 6 rows by 2 columns
* _2_ Creates a series from the 6 x 2 plot table
* _3_ Sets which sites will display in the plots
* _4_ Loops through the list of sites and gets each site number
* _5_ Gets the data for the plot
* _6_ Sets the x-axis label for the plot
* _7_ Sets the y-axis label for the plot

[Figure 7.8](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07fig08) shows the output of [listing 7.12](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07ex12). In the notebook, you’ll see an additional eight charts because the shape of the plot is 6 rows and 2 columns of plots.

**Figure 7.8. Site plots showing temperature fluctuations from November 2017 to February 2019**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch07fig08\_alt.jpg)

With that complete, you can start preparing the training and test datasets.

**7.5.4. Part 4: Creating training and test datasets**

In the previous section, you loaded each of the datasets into pandas DataFrames and fixed any missing values. In this section, you’ll turn the DataFrames into lists to create the training and test data.

[Listing 7.13](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07ex13) converts the category data into a list of numbers. Each of the numbers 0 to 2 represents one of these categories: Retail, Industrial, or Transport.

**Listing 7.13. Converting category data to a list of numbers**

```
cats = list(category_df.Category.astype('category').cat.codes)
print(cats)xample>
```

The next listing turns the power consumption data into a list of lists. Each site is a list, and there are 48 of these lists.

**Listing 7.14. Converting power consumption data to a list of lists**

```
usage_per_site = [daily_df[col] for col in daily_df.columns]
print(f'timeseries covers {len(usage_per_site[0])} days.')
print(f'timeseries starts at {usage_per_site[0].index[0]}')
print(f'timeseries ends at {usage_per_site[0].index[-1]}')
usage_per_site[0][:10]                                       1
```

* _1_ Displays the first 10 days of power consumption from site 0

The next listing repeats this for holidays.

**Listing 7.15. Converting holidays to a list of lists**

```
hols_per_site = [holiday_df[col] for col in holiday_df.columns]
print(f'timeseries covers {len(hols_per_site[0])} days.')
print(f'timeseries starts at {hols_per_site[0].index[0]}')
print(f'timeseries ends at {hols_per_site[0].index[-1]}')
hols_per_site[0][:10]
```

And the next listing repeats this for maximum temperatures.

**Listing 7.16. Converting maximum temperatures to a list of lists**

```
max_per_site = [max_df[col] for col in max_df.columns]
print(f'timeseries covers {len(max_per_site[0])} days.')
print(f'timeseries starts at {max_per_site[0].index[0]}')
print(f'timeseries ends at {max_per_site[0].index[-1]}')
max_per_site[0][:10]
```

With the data formatted as lists, you can split it into training and test data and then write the files to S3. [Listing 7.17](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07ex17) sets the start date for both testing and training as November 1, 2017. It then sets the end date for training as the end of January 2019, and the end date for testing as 28 days later (the end of February 2019).

**Listing 7.17. Setting the start and end dates for testing and training data**

```
freq = 'D'
prediction_length = 28

start_date = pd.Timestamp("2017-11-01", freq=freq)
end_training = pd.Timestamp("2019-01-31", freq=freq)
end_testing = end_training + prediction_length

print(f'End training: {end_training}, End testing: {end_testing}')
```

Just as you did in [chapter 6](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06), you now create a simple function, shown in the next listing, that writes each of the datasets to S3. In [listing 7.19](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07ex19), you’ll apply the function to the test data and training data.

**Listing 7.18. Creating a function that writes data to S3**

```
def write_dicts_to_s3(path, data):
    with s3.open(path, 'wb') as f:
        for d in data:
            f.write(json.dumps(d).encode("utf-8"))
            f.write("\n".encode('utf-8'))
```

The next listing creates the training and test datasets. DeepAR requires categorical data to be separated from dynamic features. Notice how this is done in the next listing.

**Listing 7.19. Creating training and test datasets**

```
training_data = [
    {
        "cat": [cat],                       1
        "start": str(start_date),
        "target": ts[start_date:end_training].tolist(),
        "dynamic_feat": [
            hols[
                start_date:end_training
                ].tolist(),                 2
            maxes[
                start_date:end_training
                ].tolist(),                 3
        ] # Note: List of lists
    }
    for cat, ts, hols, maxes in zip(
        cats,
        usage_per_site,
        hols_per_site,
        max_per_site)
]

test_data = [
    {
        "cat": [cat],
        "start": str(start_date),
        "target": ts[start_date:end_testing].tolist(),
        "dynamic_feat": [
            hols[start_date:end_testing].tolist(),
            maxes[start_date:end_testing].tolist(),
        ] # Note: List of lists
    }
    for cat, ts, hols, maxes in zip(
        cats,
        usage_per_site,
        hols_per_site,
        max_per_site)
]

write_dicts_to_s3(f'{s3_data_path}/train/train.json', training_data)
write_dicts_to_s3(f'{s3_data_path}/test/test.json', test_data)
```

* _1_ Categorical data for site categories
* _2_ Dynamic data for holidays
* _3_ Dynamic data for maximum temperatures

In this chapter, you set up the notebook in a slightly different way than you have in previous chapters. This chapter is all about how to use additional datasets such as site category, holidays, and max temperatures to enhance the accuracy of time series predictions.

To allow you to see the impact of these additional datasets on the prediction, we have prepared a commented-out notebook cell that creates and tests the model without using the additional datasets. If you are interested in seeing this result, you can uncomment that part of the notebook and run the entire notebook again. If you do so, you will see that, without using the additional datasets, the MAPE (Mean Average Percentage Error) for February is 20%! Keep following along in this chapter to see what it drops to when the additional datasets are incorporated into the model.

**7.5.5. Part 5: Configuring the model and setting up the server to build the model**

[Listing 7.20](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07ex20) sets the location on S3 where you will store the model and determines how SageMaker will configure the server that will build the model. At this point in the process, you would normally set a random seed to ensure that each run through the DeepAR algorithm generates a consistent result. At the time of this writing, there is an inconsistency in SageMaker’s DeepAR model—the functionality is not available. It doesn’t impact the accuracy of the results, only the consistency of the results.

**Listing 7.20. Setting up the SageMaker session and server to create the model**

```
s3_output_path = f's3://{data_bucket}/{subfolder}/output'
sess = sagemaker.Session()
image_name = sagemaker.amazon.amazon_estimator.get_image_uri(
    sess.boto_region_name,
    "forecasting-deepar",
    "latest")

data_channels = {
    "train": f"{s3_data_path}/train/",
    "test": f"{s3_data_path}/test/"
}
```

[Listing 7.21](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07ex21) is used to calculate the MAPE of the prediction. It is calculated for each day you are predicting by subtracting the predicted consumption each day from the actual consumption and dividing it by the predicted amount (and, if the number is negative, making it positive). You then take the average of all of these amounts.

For example, if on three consecutive days, you predicted consumption of 1,000 kilowatts of power, and the actual consumption was 800, 900, and 1,150 kilowatts, the MAPE would be the average of (200 / 800) + (100 / 900) + (150 / 1150) divided by three. This equals 0.16, or 16%.

**Listing 7.21. Calculating MAPE**

```
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
```

[Listing 7.22](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07ex22) is the standard SageMaker function for creating a DeepAR model. You do not need to modify this function. You just need to run it as is by clicking ![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ctrl\_ent.jpg) while in the notebook cell.

**Listing 7.22. The DeepAR predictor function used in** [**chapter 6**](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06)

```
class DeepARPredictor(sagemaker.predictor.RealTimePredictor):

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            content_type=sagemaker.content_types.CONTENT_TYPE_JSON,
            **kwargs)

    def predict(
            self,
            ts,
            cat=None,
            dynamic_feat=None,
            num_samples=100,
            return_samples=False,
            quantiles=["0.1", "0.5", "0.9"]):x
        prediction_time = ts.index[-1] + 1
        quantiles = [str(q) for q in quantiles]
        req = self.__encode_request(
            ts,
            cat,
            dynamic_feat,
            num_samples,
            return_samples,
            quantiles)
        res = super(DeepARPredictor, self).predict(req)
        return self.__decode_response(
            res,
            ts.index.freq,
            prediction_time,
            return_samples)

    def __encode_request(
            self,
            ts,
            cat,
            dynamic_feat,
            num_samples,
            return_samples,
            quantiles):
        instance = series_to_dict(
            ts,
            cat if cat is not None else None,
            dynamic_feat if dynamic_feat else None)
        configuration = {
            "num_samples": num_samples,
            "output_types": [
                "quantiles",
                "samples"] if return_samples else ["quantiles"],
            "quantiles": quantiles
        }
        http_request_data = {
            "instances": [instance],
            "configuration": configuration
        }
        return json.dumps(http_request_data).encode('utf-8')
    def __decode_response(
            self,
            response,
            freq,
            prediction_time,
            return_samples):
        predictions = json.loads(
            response.decode('utf-8'))['predictions'][0]
        prediction_length = len(next(iter(
                predictions['quantiles'].values()
            )))
        prediction_index = pd.DatetimeIndex(
            start=prediction_time,
            freq=freq,
            periods=prediction_length)
        if return_samples:
            dict_of_samples = {
                    'sample_' + str(i): s for i, s in enumerate(
                        predictions['samples'])
                }
        else:
            dict_of_samples = {}
        return pd.DataFrame(
            data={**predictions['quantiles'],
            **dict_of_samples},
            index=prediction_index)
    def set_frequency(self, freq):
        self.freq = freq
def encode_target(ts):
    return [x if np.isfinite(x) else "NaN" for x in ts]

def series_to_dict(ts, cat=None, dynamic_feat=None):
    # Given a pandas.Series, returns a dict encoding the timeseries.
    obj = {"start": str(ts.index[0]), "target": encode_target(ts)}
    if cat is not None:
        obj["cat"] = cat
    if dynamic_feat is not None:
        obj["dynamic_feat"] = dynamic_feat
    return objple>
```

Just as in [chapter 6](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06), you now need to set up the estimator and then set the parameters for the estimator. SageMaker exposes several parameters for you. The only two that you need to change are the first two parameters shown in lines 1 and 2 of [listing 7.23](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07ex23): context\_length and prediction\_length.

The _context length_ is the minimum period of time that will be used to make a prediction. By setting this value to 90, you are saying that you want DeepAR to use 90 days of data as a minimum to make its predictions. In business settings, this is typically a good value because it allows for the capture of quarterly trends. The _prediction length_ is the period of time you are predicting. In this notebook, you are predicting February data, so you use the prediction\_length of 28 days.

**Listing 7.23. Setting up the estimator**

```
%%time
estimator = sagemaker.estimator.Estimator(
    sagemaker_session=sess,
    image_name=image_name,
    role=role,
    train_instance_count=1,
    train_instance_type='ml.c5.2xlarge', # $0.476 per hour as of Jan 2019.
    base_job_name='ch7-energy-usage-dynamic',
    output_path=s3_output_path
)

estimator.set_hyperparameters(
    context_length="90",                         1
    prediction_length=str(prediction_length),    2
    time_freq=freq,                              3
    epochs="400",                                4
    early_stopping_patience="40",                5
    mini_batch_size="64",                        6
    learning_rate="5E-4",                        7
    num_dynamic_feat=2,                          8
 )

estimator.fit(inputs=data_channels, wait=True)
```

* _1_ Sets the context length to 90 days
* _2_ Sets the prediction length to 28 days
* _3_ Sets the frequency to daily
* _4_ Sets the epochs to 400 (leave this value as is)
* _5_ Sets the early stopping to 40 (leave this value as is)
* _6_ Sets the batch size to 64 (leave this value as is)
* _7_ Sets the learning rate to 0.0005 (the decimal conversion of the exponential value 5E-4)
* _8_ Sets the number of dynamic features to 2 for holidays and temperature (leave this as is)

[Listing 7.24](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07ex24) creates the endpoint you’ll use to test the predictions. In the next chapter, you’ll learn how to expose that endpoint to the internet, but for this chapter, just like the preceding chapters, you’ll hit the endpoint using code in the notebook.

**Listing 7.24. Setting up the endpoint**

```
endpoint_name = 'energy-usage-dynamic'

try:
    sess.delete_endpoint(
        sagemaker.predictor.RealTimePredictor(
            endpoint=endpoint_name).endpoint)
    print(
        'Warning: Existing endpoint deleted to make way for new endpoint.')
    from time import sleep
    sleep(30)
except:
    passalexample>
```

Now it’s time to build the model. The following listing creates the model and assigns it to the variable predictor.

**Listing 7.25. Building and deploying the model**

```
%%time
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    predictor_cls=DeepARPredictor,
    endpoint_name=endpoint_name)
```

**7.5.6. Part 6: Making predictions and plotting results**

Once the model is built, you can run the predictions against each of the days in February. First, however, you’ll test the predictor as shown in the next listing.

**Listing 7.26. Checking the predictions from the model**

```
predictor.predict(
    cat=[cats[0]],
    ts=usage_per_site[0][start_date+30:end_training],
    dynamic_feat=[
             hols_per_site[0][start_date+30:end_training+28].tolist(),
             max_per_site[0][start_date+30:end_training+28].tolist(),
         ],
    quantiles=[0.1, 0.5, 0.9]
).head()alexample>
```

Now that you know the predictor is working as expected, you’re ready run it across each of the days in February 2019. But before you do that, to allow you to calculate the MAPE, you’ll create a list called _usages_ to store the actual power consumption for each site for each day in February 2019. When you run the predictions across each day in February, you store the result in a list called _predictions_.

**Listing 7.27. Getting predictions for all sites during February 2019**

```
usages = [
    ts[end_training+1:end_training+28].sum() for ts in usage_per_site]

predictions= []
for s in range(len(usage_per_site)):
    # call the end point to get the 28 day prediction
    predictions.append(
        predictor.predict(
            cat=[cats[s]],
            ts=usage_per_site[s][start_date+30:end_training],
            dynamic_feat=[
                hols_per_site[s][start_date+30:end_training+28].tolist(),
                max_per_site[s][start_date+30:end_training+28].tolist(),
            ]
        )['0.5'].sum()
    )

for p,u in zip(predictions,usages):
    print(f'Predicted {p} kwh but usage was {u} kwh.')
```

Once you have the usage list and the predictions list, you can calculate the MAPE by running the mape function you created in [listing 7.21](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07ex21).

**Listing 7.28. Calculating MAPE**

```
print(f'MAPE: {round(mape(usages, predictions),1)}%')
```

[Listing 7.29](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07ex29) is the same plot function you saw in [chapter 6](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_016.html#ch06). The function takes the usage list and creates predictions in the same way you did in [listing 7.27](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07ex27). The difference in the plot function here is that it also calculates the lower and upper predictions at an 80% confidence level. It then plots the actual usage as a line and shades the area within the 80% confidence threshold.

**Listing 7.29. Displaying plots of sites**

```
def plot(
    predictor,
    site_id,
    end_training=end_training,
    plot_weeks=12,
    confidence=80
):
    low_quantile = 0.5 - confidence * 0.005
    up_quantile = confidence * 0.005 + 0.5
    target_ts = usage_per_site[site_id][start_date+30:]
    dynamic_feats = [
            hols_per_site[site_id][start_date+30:].tolist(),
            max_per_site[site_id][start_date+30:].tolist(),
        ]

    plot_history = plot_weeks * 7

    fig = plt.figure(figsize=(20, 3))
    ax = plt.subplot(1,1,1)

    prediction = predictor.predict(
        cat = [cats[site_id]],
        ts=target_ts[:end_training],
        dynamic_feat=dynamic_feats,
        quantiles=[low_quantile, 0.5, up_quantile])

    target_section = target_ts[
        end_training-plot_history:end_training+prediction_length]
    target_section.plot(color="black", label='target')

    ax.fill_between(
        prediction[str(low_quantile)].index,
        prediction[str(low_quantile)].values,
        prediction[str(up_quantile)].values,
        color="b",
        alpha=0.3,
        label=f'{confidence}% confidence interval'
    )

    ax.set_ylim(target_section.min() * 0.5, target_section.max() * 1.5)
```

The following listing runs the plot function you created in [listing 7.29](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07ex29).

**Listing 7.30. Plotting several sites and the February predictions**

```
indices = [2,26,33,39,42,47,3]
for i in indices:
    plot_num = indices.index(i)
    plot(
        predictor,
        site_id=i,
        plot_weeks=6,
        confidence=80
```

[Figure 7.9](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07fig09) shows the predicted results for several sites. As you can see, the daily prediction for each time series falls within the shaded area.

**Figure 7.9. Site plots showing predicted usage for February 2019**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch07fig09\_alt.jpg)

One of the advantages of displaying the data in this manner is that it is easy to pick out sites where you haven’t predicted accurately. For example, if you look at site 3, the last site in the plot list in [figure 7.10](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07fig10), you can see that there was a period in February with almost no power usage, when you predicted it would have a fairly high usage. This provides you with an opportunity to improve your model by including additional datasets.

**Figure 7.10. Predicted usage for site 3 is incorrect in early February.**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch07fig10\_alt.jpg)

When you see a prediction that is clearly inaccurate, you can investigate what happened during that time and determine if there is some data source that you could incorporate into your predictions. If, for example, this site had a planned maintenance shutdown in early February and this shutdown was not already included in your holiday data, if you can get your hands on a schedule of planned maintenance shutdowns, then you can easily incorporate that data in your model in the same way that you incorporated the holiday data.

#### 7.6. Deleting the endpoint and shutting down your notebook instance <a href="#ch07lev1sec6__title" id="ch07lev1sec6__title"></a>

As always, when you are no longer using the notebook, remember to shut down the notebook and delete the endpoint. We don’t want you to get charged for SageMaker services that you’re not using.

**7.6.1. Deleting the endpoint**

To delete the endpoint, uncomment the code in [listing 7.31](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07ex31), then click ![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ctrl\_ent.jpg) to run the code in the cell.

**Listing 7.31. Deleting the endpoint**

```
# Remove the endpoints
# Comment out these cells if you want the endpoint to persist after Run All
# sess.delete_endpoint('energy-usage-baseline')
# sess.delete_endpoint('energy-usage-dynamic')
```

**7.6.2. Shutting down the notebook instance**

To shut down the notebook, go back to your browser tab where you have SageMaker open. Click the Notebook Instances menu item to view all of your notebook instances. Select the radio button next to the notebook instance name, as shown in [figure 7.11](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07fig11), then select Stop from the Actions menu. It takes a couple of minutes to shut down.

**Figure 7.11. Shutting down the notebook**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch07fig11\_alt.jpg)

#### 7.7. Checking to make sure the endpoint is deleted <a href="#ch07lev1sec7__title" id="ch07lev1sec7__title"></a>

If you didn’t delete the endpoint using the notebook (or if you just want to make sure it is deleted), you can do this from the SageMaker console. To delete the endpoint, click the radio button to the left of the endpoint name, then click the Actions menu item and click Delete in the menu that appears.

When you have successfully deleted the endpoint, you will no longer incur AWS charges for it. You can confirm that all of your endpoints have been deleted when you see the text “There are currently no resources” displayed on the Endpoints page ([figure 7.12](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_017.html#ch07fig12)).

**Figure 7.12. Confirm that all endpoints were deleted.**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch07fig12\_alt.jpg)

Kiara can now predict power consumption for each site with a 6.9% MAPE, even for months with a number of holidays or predicted weather fluctuations.

#### Summary <a href="#ch07lev1sec8__title" id="ch07lev1sec8__title"></a>

* Past usage is not always a good predictor of future usage.
* DeepAR is a neural network algorithm that is particularly good at incorporating several different time-series datasets into its forecasting, thereby accounting for events in your time-series forecasting that your time-series data can’t directly infer.
* The datasets used in this chapter can be classified into two types of data: categorical and dynamic. Categorical data is information about the site that doesn’t change, and dynamic data is data that changes over time.
* For each day in your prediction range, you calculate the Mean Average Prediction Error (MAPE) for the time-series data by defining the function mape.
* Once the model is built, you can run the predictions and display the results in multiple time-series charts to easily visualize the predictions.
