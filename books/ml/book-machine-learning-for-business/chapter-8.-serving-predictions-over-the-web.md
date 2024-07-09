# Chapter 8. Serving predictions over the web

_This chapter covers_

* Setting up SageMaker to serve predictions over the web
* Building and deploying a serverless API to deliver SageMaker predictions
* Sending data to the API and receiving predictions via a web browser

Until now, the machine learning models you built can be used only in SageMaker. If you wanted to provide a prediction or a decision for someone else, you would have to submit the query from a Jupyter notebook running in SageMaker and send them the results. This, of course, is not what AWS intended for SageMaker. They intended that your users would be able to access predictions and decisions over the web. In this chapter, you’ll enable your users to do just that.

Serving tweets

In [chapter 4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04), you helped Naomi identify which tweets should be escalated to her support team and which tweets could be handled by an automated bot. One of the things you didn’t do for Naomi was provide a way for her to send tweets to the machine learning model and receive a decision as to whether a tweet should be escalated. In this chapter, you will rectify that.

#### 8.1. Why is serving decisions and predictions over the web so difficult? <a href="#ch08lev1sec1__title" id="ch08lev1sec1__title"></a>

In each of the previous chapters, you created a SageMaker model and set up an endpoint for that model. In the final few cells of your Jupyter Notebook, you sent test data to the endpoint and received the results. You have only interacted with the SageMaker endpoint from within the SageMaker environment. In order to deploy the machine learning model on the internet, you need to expose that endpoint to the internet.

Until recently, this was not an easy thing to do. You first needed to set up a web server. Next, you coded the API that the web server would use, and finally, you hosted the web server and exposed the API as a web address (URL). This involved lots of moving parts and was not easy to do. Nowadays, all this is much easier.

In this chapter, you’ll tackle the problem of creating a web server and hosting the API in a way that builds on many of the skills relating to Python and AWS that you’ve learned in previous chapters. At present, you can serve web applications without worrying about the complexities of setting up a web server. In this chapter, you’ll use AWS Lambda as your web server ([figure 8.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08fig01)).

**Figure 8.1. Sending a tweet from a browser to SageMaker**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch08fig01\_alt.jpg)

AWS Lambda is a server that boots on demand. Every tweet you send to the SageMaker endpoint creates a server that sends the tweet and receives the response, and then shuts down once it’s finished. This sounds like it might be slow from reading that description, but it’s not. AWS Lambda can start and shut down in a few milliseconds. The advantage when serving your API is that you’re paying for the Lambda server only when it is serving decisions from your API. For many APIs, this is a much more cost-effective model than having a permanent, dedicated web server waiting to serve predictions from your API.

Serverless computing

Services like AWS Lambda are often called _serverless_. The term serverless is a misnomer. When you are serving an API on the internet, by definition it cannot be serverless. What serverless refers to is the fact that somebody else has the headache of running your server.

#### 8.2. Overview of steps for this chapter <a href="#ch08lev1sec2__title" id="ch08lev1sec2__title"></a>

This chapter contains very little new code. It’s mostly configurations. To help follow along throughout the chapter, you’ll see a list of steps and where you are in the steps. The steps are divided into several sections:

1. Set up the SageMaker endpoint.
2. Configure AWS on your local computer.
3. Create a web endpoint.
4. Serve decisions.

With that as an introduction, let’s get started.

#### 8.3. The SageMaker endpoint <a href="#ch08lev1sec3__title" id="ch08lev1sec3__title"></a>

Up to this point, you have interacted with your machine learning models using a Jupyter notebook and the SageMaker endpoint. When you interact with your models in this manner, it hides some of the distinctions between the parts of the system.

The SageMaker endpoint can also serve predictions to an API, which can then be used to serve predictions and decisions to users over the web. This configuration works because it is a safe environment. You can’t access the SageMaker endpoint unless you’re logged into Jupyter Notebook, and anyone who is logged into the Notebook server has permission to access the endpoint.

When you move to the web, however, things are a little more wild. You don’t want just anyone hitting on your SageMaker endpoint, so you need to be able to secure the endpoint and make it available only to those who have permission to access it.

Why do you need an API endpoint in addition to the SageMaker endpoint? SageMaker endpoints don’t have any of the required components to allow them to be safely exposed to the wilds of the internet. Fortunately, there are lots of systems that can handle this for you. In this chapter, you’ll use AWS’s infrastructure to create a serverless web application configured to serve predictions and decisions from the SageMaker endpoint you set up in [chapter 4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04). To do so, you’ll follow these steps:

1. Set up the SageMaker endpoint by
   1. Starting SageMaker
   2. Uploading a notebook
   3. Running a notebook
2. Configure AWS on your local computer.
3. Create a web endpoint.
4. Serve decisions.

To begin, you need to start SageMaker and create an endpoint for the notebook. The notebook you’ll use is the same as the notebook you used for [chapter 4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04) (customer \_support.ipynb), except it uses a different method for normalizing the tweet text. Don’t worry if you didn’t work through that chapter or don’t have the notebook on SageMaker anymore, we’ll walk you through how to set it up.

#### 8.4. Setting up the SageMaker endpoint <a href="#ch08lev1sec4__title" id="ch08lev1sec4__title"></a>

Like each of the other chapters, you’ll need to start SageMaker (detailed in [appendix C](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_028.html#app03)). For your convenience, that’s summarized here. First, go to the SageMaker service on AWS by clicking this link:

[https://console.aws.amazon.com/sagemaker/home](https://console.aws.amazon.com/sagemaker/home)

Then, start your notebook instance. [Figure 8.2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08fig02) shows the AWS Notebook instances page. Click the Start action.

**Figure 8.2. Starting a SageMaker instance**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch08fig02\_alt.jpg)

In a few minutes, the page refreshes and a link to Open Jupyter appears, along with an InService status message. [Figure 8.3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08fig03) shows the AWS Notebook Instances page after the notebook instance is started.

**Figure 8.3. Opening Jupyter**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch08fig03\_alt.jpg)

The next section shows you how to upload the notebook and data for this chapter. But what are the differences between the notebook used in [chapter 4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04) and the notebook used in [chapter 8](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08)?

In this chapter, even though you are deciding which tweets to escalate as you did in [chapter 4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04), you will create a new notebook rather than reuse the notebook for [chapter 4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04). The reason for this is that we want to be able to pass the text of the tweet as a URL in the address bar of the browser (so we don’t have to build a web form to enter the text of the tweet). This means that the text of the tweet can’t contain any characters that are not permitted to be typed in the address bar of the browser. As this is not how we built the model in [chapter 4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04), we need to train a new model in this chapter.

The notebook we create in this chapter is exactly the same as the notebook in [chapter 4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04), except that it uses a library called _slugify_ to preprocess the tweet rather than NLTK. Slugify is commonly used to turn text into website URLs. In addition to providing a lightweight mechanism to normalize text, it also allows the tweets to be accessed as URLs.

**8.4.1. Uploading the notebook**

Start by downloading the Jupyter notebook to your computer from this link:

[https://s3.amazonaws.com/mlforbusiness/ch08/customer\_support\_slugify.ipynb](https://s3.amazonaws.com/mlforbusiness/ch08/customer\_support\_slugify.ipynb)

Now, in the notebook instance shown in [figure 8.4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08fig04), create a folder to store the notebook by clicking New on the Files page and selecting the Folder menu item as shown in [figure 8.5](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08fig05). The new folder will hold all of your code for this chapter.

**Figure 8.4. Creating a new notebook folder: Step 1**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch08fig04\_alt.jpg)

**Figure 8.5. Creating a new notebook folder: Step 2**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch08fig05\_alt.jpg)

[Figure 8.6](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08fig06) shows the new folder after you click it. Once in the folder, you see an Upload button on the top right of the page. Clicking this button opens a file selection window. Navigate to the location where you downloaded the Jupyter notebook, and upload it to the notebook instance.

**Figure 8.6. Uploading the notebook to a new notebook foldre**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch08fig06\_alt.jpg)

[Figure 8.7](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08fig07) shows the notebook you have uploaded to SageMaker.

**Figure 8.7. Verifying that the notebook customer\_support\_slugify.ipynb was uploaded to your SageMaker folder**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch08fig07\_alt.jpg)

**8.4.2. Uploading the data**

Even though you can’t reuse the notebook from [chapter 4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04), you can reuse the data. If you set up the notebook and the data for [chapter 4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04), you can use that dataset; skip ahead to section 8.4.3, “Running the notebook and creating the endpoint.” If you didn’t do that, follow the steps in this section.

If you didn’t set up the notebook and the data for [chapter 4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04), download the dataset from this location:

[https://s3.amazonaws.com/mlforbusiness/ch04/inbound.csv](https://s3.amazonaws.com/mlforbusiness/ch04/inbound.csv)

Save this file to a location on your computer. You won’t do anything with this file other than upload it to S3, so you can use your downloads directory or some other temporary folder.

Now, head to AWS S3, the AWS file storage service, by clicking this link:

[https://s3.console.aws.amazon.com/s3/home](https://s3.console.aws.amazon.com/s3/home)

Once there, create or navigate to the S3 bucket where you are keeping the data for this book (see [appendix B](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_024.html#app02) if you haven’t created an S3 bucket yet).

In your bucket, you can see any folders you have created. If you haven’t done so already, create a folder to hold the data for [chapter 4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04) by clicking Create Folder. (We are setting up the data for [chapter 4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04) because this chapter uses the same data as that chapter. You may as well store the data as in [chapter 4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04), even if you haven’t worked through the content of that chapter.) [Figure 8.8](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08fig08) shows the folder structure you might have if you have followed all the chapters in this book.

**Figure 8.8. Example of what your S3 folder structure might look like**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch08fig08\_alt.jpg)

Inside the folder, click Upload on the top left of the page, find the CSV data file you just saved, and upload it. After you have done so, you’ll see the inbound.csv file listed in the folder ([figure 8.9](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08fig09)). Keep this page open because you’ll need to get the location of the file when you run the notebook.

**Figure 8.9. Example of what your S3 bucket might look like once you have uploaded the CSV data**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch08fig09\_alt.jpg)

You now have a Jupyter notebook set up on SageMaker and data loaded onto S3. You are ready to begin to build and deploy your model in preparation for serving predictions over the web.

**8.4.3. Running the notebook and creating the endpoint**

Now that you have a Jupyter notebook instance running and uploaded your data to S3, run the notebook and create the endpoint. You do this by selecting Cell from the menu and then clicking Run All ([figure 8.10](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08fig10)).

**Figure 8.10. Running all cells in the notebook**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch08fig10\_alt.jpg)

After 5 minutes or so, all the cells in the notebook will have run, and you will have created an endpoint. You can see that all the cells have run by scrolling to the bottom of the notebook and checking for a value in the second-to-last cell (below the Test the Model heading) as shown in [figure 8.11](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08fig11).

**Figure 8.11. Confirming that all cells in the notebook have run**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch08fig11\_alt.jpg)

Once you have run the notebook, you can view the endpoint by clicking the Endpoints link as shown in [figure 8.12](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08fig12).

**Figure 8.12. Navigating to Endpoints to view your current endpoints**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch08fig12\_alt.jpg)

Here you will see the ARN (Amazon Resource Name) of the endpoint you have created. You will need this when you set up the API endpoint. [Figure 8.13](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08fig13) shows an example of what your endpoint might look like.

**Figure 8.13. Example of what your endpoint ARN might look like**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch08fig13\_alt.jpg)

Now, you can set up the serverless API endpoint. The API endpoint is the URL or web address that you will send the tweets to. You will set up the endpoint on AWS serverless technology, which is called AWS Lambda. In order for AWS Lambda to know what to do, you will install the Chalice Python library.

Chalice is a library built by AWS to make it easy to use Python to serve an API endpoint. You can read more about Chalice here: [https://chalice.readthedocs.io/en/latest/](https://chalice.readthedocs.io/en/latest/).

#### 8.5. Setting up the serverless API endpoint <a href="#ch08lev1sec5__title" id="ch08lev1sec5__title"></a>

To review where you are at in the process, you have just set up a SageMaker endpoint that is ready to serve decisions about whether to escalate a tweet to your support team. Next, you’ll set Chalice, the serverless API endpoint as follows:

1. Set up the SageMaker endpoint.
2. Configure AWS on your local computer by
   1. Creating credentials
   2. Installing the credentials on your local computer
   3. Configuring the credentials
3. Create a web endpoint.
4. Serve decisions.

It’s somewhat ironic that the first thing you need to do to set up a serverless API endpoint is set up software on your computer. The two applications you need are Python (version 3.6 or higher) and a text editor.

Instructions for installing Python are in appendix E. Although installing Python used to be tricky, it’s become much easier for Windows operating systems with the inclusion of Python in the Microsoft Windows Store. And installing Python on Apple computers has been made easier for some time now by the Homebrew package manager.

As we mentioned, you’ll also need a text editor. One of the easiest editors to set up is Microsoft’s Visual Studio Code (VS Code). It runs on Windows, macOS, and Linux. You can download VS Code here: [https://code.visualstudio.com/](https://code.visualstudio.com/).

Now that you are set up to run Python on your computer, and you have a text editor, you can start setting up the serverless endpoint.

**8.5.1. Setting up your AWS credentials on your AWS account**

To access the SageMaker endpoint, your serverless API needs to have permission to do so. And because you are writing code on your local computer rather than in a SageMaker notebook (as you have done for each of the previous chapters in the book), your local computer also needs permission to access the SageMaker endpoint and your AWS account. Fortunately, AWS provides a simple way to do both.

First, you need to create credentials in your AWS account. To set up credentials, click the AWS username in the top right of the browser from any AWS page ([figure 8.14](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08fig14)).

**Figure 8.14. Creating AWS credentials**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch08fig14\_alt.jpg)

In the page that opens, there’s a Create access key button that allows you to create an _access key_, which is one of the types of credentials you can use to access your AWS account. Click this button.

[Figure 8.15](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08fig15) shows the AWS user interface for creating an access key. After clicking this button, you will be able to download your security credentials as a CSV file.

**Figure 8.15. Creating an AWS access key**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch08fig15\_alt.jpg)

**Note**

You are presented with your one and only opportunity to download your keys as a CSV file.

Download the CSV file and save it somewhere on your computer where only you have access ([figure 8.16](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08fig16)). Anyone who gets this key can use your AWS account.

**Figure 8.16. Downloading the AWS access key**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch08fig16\_alt.jpg)

With the access key downloaded to your computer, you can set up your local computer to access AWS. We’ll cover that next.

**8.5.2. Setting up your AWS credentials on your local computer**

To set up your local computer to access AWS, you need to install two AWS Python libraries on your local computer. This section will walk you through how to install these libraries from VS Code, but you can use any terminal application such as Bash on Unix or macOS, or PowerShell on Windows.

First, create a folder on your computer that you will use for saving your code. Then open VS Code and click the Open Folder button as shown in [figure 8.17](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08fig17).

**Figure 8.17. Opening a folder in VS Code**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch08fig17\_alt.jpg)

Create a new folder on your computer to hold the files for this chapter. Once you have done so, you can start installing the Python libraries you’ll need.

The code you’ll write on your local computer needs Python libraries in the same way SageMaker needs Python libraries to run. The difference between your local computer and SageMaker is that SageMaker has the libraries you need already installed, whereas on your computer, you may need to install the libraries yourself.

In order to install the Python libraries on your computer, you need to open a terminal shell. This is a way to enter commands into your computer using only the keyboard. Opening a terminal window in VS Code is done by pressing ![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ctrl\_shift.jpg). Alternatively, you can open a terminal window from VS Code by selecting Terminal from the menu bar and then selecting New Terminal.

A terminal window appears at the bottom of VS Code, ready for you to type into. You can now install the Python libraries you need to access SageMaker.

The first library you will install is called boto3. This library helps you interact with AWS services. SageMaker itself uses boto3 to interact with services such as S3. To install boto3, in the terminal window, type

```
pip install boto3
```

Next, you’ll need to install the command-line interface (CLI) library that lets you stop and start an AWS service from your computer. It also allows you to set up credentials that you have created in AWS. To install the AWS CLI library, type

```
pip install awscli
```

With both boto3 and the CLI library installed, you can now configure your credentials.

**8.5.3. Configuring your credentials**

To configure your AWS credentials, run the following command at the prompt in the terminal window:

```
aws configure
```

You are asked for your AWS Access Key ID and the AWS Secret Access Key you downloaded earlier. You are also asked for your AWS region. [Figure 8.18](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08fig18) shows how to locate your SageMaker region.

**Figure 8.18. Locating your SageMaker region**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch08fig18\_alt.jpg)

[Figure 8.18](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08fig18) shows the address bar of a web browser when you are logged into SageMaker. The address shows which region your SageMaker service is located in. Use this region when you configure the AWS credentials. Note that you can leave the Default output format blank.

```
AWS Access Key ID:                  1
AWS Secret Access Key:              2
Default region name [us-east-1]:    3
Default output format [None]:
```

* _1_ Enter the access key ID you downloaded earlier.
* _2_ Enter the secret access key you downloaded earlier.
* _3_ Enter the AWS region you use with SageMaker.

You’ve completed the configuration of AWS on your local computer. To recap, you set up the SageMaker endpoint, then you configured AWS on your local computer. Now you will create the web endpoint that allows you to serve decisions regarding which tweets to escalate to Naomi’s support team. Let’s update where we are in the process:

1. Set up the SageMaker endpoint.
2. Configure AWS on your local computer.
3. Create a web endpoint by
   1. Installing Chalice
   2. Writing endpoint code
   3. Configuring permissions
   4. Updating requirements.txt
   5. Deploying Chalice
4. Serve decisions

#### 8.6. Creating the web endpoint <a href="#ch08lev1sec6__title" id="ch08lev1sec6__title"></a>

You are at the point in the chapter that amazed us when we first used AWS to serve an API endpoint. You are going to create a serverless function using an AWS technology called a Lambda function ([https://aws.amazon.com/lambda/](https://aws.amazon.com/lambda/)), and configure the API using an AWS technology called the Amazon API Gateway ([https://aws.amazon.com/api-gateway/](https://aws.amazon.com/api-gateway/)). Then you’ll deploy the SageMaker endpoint so it can used by anyone, anywhere. And you will do it in only a few lines of code. Amazing!

**8.6.1. Installing Chalice**

Chalice ([https://github.com/aws/chalice](https://github.com/aws/chalice)) is open source software from Amazon that automatically creates and deploys a Lambda function and configures an API gateway endpoint for you. During configuration, you will create a folder on your computer to store the Chalice code. Chalice will take care of packaging the code and installing it in your AWS account. It can do this because, in the previous section, you configured your AWS credentials using the AWS CLI.

The easiest way to get started is to navigate to an empty folder on your computer. Right-click the folder to open a menu, and then click Open with Code as shown in [figure 8.19](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08fig19). Alternatively, you can open this folder from VS Code in the same way you did in [figure 8.17](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08fig17). Neither way is better than the other—use whichever approach you prefer.

**Figure 8.19. Opening the VS Code editor in a folder**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch08fig19\_alt.jpg)

To install Chalice, once you have opened VS Code, go to the terminal window like you did when you configured the AWS CLI, and type this command:

```
pip install chalice
```

Depending on the permissions you have on your computer, if this produces an error, you might need to type this command:

```
pip install --user chalice
```

Just like the AWS CLI you used earlier, this command creates a CLI application on your system. Now you’re all set to use Chalice.

Using Chalice is straightforward. There are two main commands:

* new-project
* deploy

To create a new project named tweet\_escalator, run the following command at the prompt:

```
chalice new-project tweet_escalator
```

If you look in the folder you opened VS Code from, you will see a folder called tweet\_escalator that contains some files that Chalice automatically created. We’ll discuss these files shortly, but first, let’s deploy a Hello World application.

In the terminal window, you’ll see that after running chalice new-project tweet\_escalator, you’re still in the folder you opened VS Code from. To navigate to the tweet\_escalator folder, type

```
cd tweet_escalator
```

You’ll see that you are now in the folder tweet\_escalator:

```
c:\\mlforbusiness\ch08\tweet_escalator
```

Now that you are in the tweet\_escalator folder, you can type chalice deploy to create a Hello World application:

```
c:\\mlforbusiness\ch08\tweet_escalator chalice deploy
```

Chalice will then automatically create a Lambda function on AWS, set up the permissions to run the application (known as an _IAM role_), and configure a Rest endpoint using AWS Gateway. Here’s Chalice’s process:

* Create a deployment package
* Create an IAM role (tweet\_escalator-dev)
* Create a Lambda function (tweet\_escalator-dev)
* Create a Rest API

The resources deployed by Chalice are

* Lambda ARN (arn:aws:lambda\_us-east-1:3839393993:function:tweet\_escalator-dv)
* Rest API URL ([https://eyeueiwwo.execute-api.us-east-1.amazonaws.com/api/](https://eyeueiwwo.execute-api.us-east-1.amazonaws.com/api/))

You can run the Hello World application by clicking the Rest API URL shown in the terminal. Doing so opens a web browser and displays {"hello":"world"} in JSON, as shown in [figure 8.20](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08fig20).

**Figure 8.20. Hello World**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch08fig20\_alt.jpg)

Congratulations! Your API is now up and running and you can see the output in your web browser.

**8.6.2. Creating a Hello World API**

Now that you have the Hello World application working, it’s time to configure Chalice to return decisions from your endpoint. [Figure 8.21](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08fig21) shows the files that Chalice automatically created when you typed chalice new-project tweet\_escalator. Three important components are created:

**Figure 8.21. Screenshot of the Chalice folder on your computer**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch08fig21\_alt.jpg)

* A .chalice folder that contains the configuration files. The only file in this folder that you will need to modify is the policy-dev.json file, which sets the permissions that allow the Lambda function to call the SageMaker endpoint.
* An app.py file that contains the code that runs when the endpoint is accessed (such as when you view it in your web browser).
* A requirements.txt file that lists any Python libraries that your application needs to run.

[Listing 8.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08ex1) shows the code in the app.py file that Chalice creates automatically. The app only needs a name, a route, and a function to work.

**Listing 8.1. Chalice’s default app.py code**

```
from chalice import Chalice                 1

app = Chalice(app_name='tweet_escalator')   2

@app.route('/')                             3
def index():                                4
    return {'hello': 'world'}               5
```

* _1_ Imports Chalice, the library that creates the Lambda function and API gateway
* _2_ Imports Chalice, the library that creates the Lambda function and API gateway
* _3_ Sets the default route
* _4_ Defines the function that runs when the default route is hit
* _5_ Sets the value that gets returned by the function and displayed in the web browser

In [listing 8.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08ex1), the name of the app (line 2) is the name that is used to identify the Lambda function and API gateway on AWS. The route (line 3) identifies the URL location that runs the function. And the function (line 4) is the code that is run when the URL location is accessed.

Accessing URLs

Building this application is beyond the scope of this book. In this chapter, you will just set up the URL location that invokes the SageMaker endpoint and displays an escalation recommendation in a web browser.

Building this application is beyond the scope of this book. In this chapter, you will just set up the URL location that invokes the SageMaker endpoint and displays an escalation recommendation in a web browser.

**8.6.3. Adding the code that serves the SageMaker endpoint**

You can keep the Hello World code you just created and use it as the basis for your code that will serve the SageMaker endpoint. For now, at the bottom of the Hello World code, add two blank lines and enter the code shown in [listing 8.2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08ex2). The full code listing can be downloaded from this link: [https://s3.amazonaws.com/mlforbusiness/ch08/app.py](https://s3.amazonaws.com/mlforbusiness/ch08/app.py).

**Listing 8.2. The default app.py code**

```
@app.route('/tweet/{tweet}')                     1
def return_tweet(tweet):                         2
    tokenized_tweet = [
        slugify(tweet, separator=' ')]           3
    payload = json.dumps(
        {"instances" : tokenized_tweet})         4

    endpoint_name = 'customer-support-slugify'   5

    runtime = boto3.Session().client(
        service_name='sagemaker-runtime',
        region_name='us-east-1')                 6

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=payload)                            7

    response_list = json.loads(
        response['Body'].read().decode())        8
    response = response_list[0]                  9

    if '1' in response['label'][0]:             10
        escalate = 'Yes'
    else:
        escalate = 'No'

    full_response = {                           11
        'Tweet': tweet,
        'Tokenised tweet': tokenized_tweet,
        'Escalate': escalate,
        'Confidence': response['prob'][0]
    }
    return full_response                        12
```

* _1_ Defines the route
* _2_ Sets up the function
* _3_ Tokenizes the tweet
* _4_ Sets up the payload
* _5_ Identifies the SageMaker endpoint
* _6_ Prepares the endpoint
* _7_ Invokes the endpoint and gets the response
* _8_ Converts the response to a list
* _9_ Gets the first item in the list
* _10_ Sets escalate decision to Yes or No
* _11_ Sets the full response format
* _12_ Returns the response

Just like the @app.route you set in line 3 of [listing 8.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08ex1), you start your code by defining the route that will be used. Instead of defining the route as / as you did earlier, in line 1 of [listing 8.2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08ex2), you set the route as /tweet/{tweet}/. This tells the Lambda function to watch for anything that hits the URL path /tweet/ and submit anything it sees after that to the SageMaker endpoint, for example, if Chalice creates an endpoint for you at

[https://ifs1qanztg.execute-api.us-east-1.amazonaws.com/api/](https://ifs1qanztg.execute-api.us-east-1.amazonaws.com/api/)

When you go to this endpoint, it returns {"hello": "world"}. Similarly, the code in line 1 of [listing 8.2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08ex2) would send I am angry to the SageMaker endpoint when you access this endpoint:

[https://ifs1qanztg.execute-api.us-east-1.amazonaws.com/api/tweet/i-am-angry](https://ifs1qanztg.execute-api.us-east-1.amazonaws.com/api/tweet/i-am-angry)

The code {tweet} tells Chalice to put everything it sees at the end of the URL into a variable called tweet. In the function you see in line 2, you are using the variable tweet from line 1 as your input to the function.

Line 3 slugifies the tweet using the same function that the Jupyter notebook uses. This ensures that the tweets you send to the SageMaker endpoint are normalized using the same approach that was used to train the model. Line 4 reflects the code in the Jupyter notebook to create the payload that gets sent to the SageMaker endpoint. Line 5 is the name of the SageMaker endpoint you invoke. Line 6 ensures that the endpoint is ready to respond to a tweet sent to it, and line 7 sends the tweet to the SageMaker endpoint.

Line 8 receives the response. The SageMaker endpoint is designed to take in a list of tweets and return a list of responses. For our application in this chapter, you are only sending a single tweet, so line 9 returns just the first result. Line 10 converts the escalate decision from 0 or 1 to No or Yes, respectively. And finally, line 11 defines the response format, and line 12 returns the response to the web browser.

**8.6.4. Configuring permissions**

At this point, your Chalice API still cannot access your AWS Lambda function. You need to give the AWS Lambda function permission to access your endpoint. Your Hello World Lambda function worked without configuring permissions because it did not use any other AWS resources. The updated function needs access to AWS SageMaker, or it will give you an error.

Chalice provides a file called policy-dev.json, which sets permissions. You’ll find it in the .chalice folder that’s located in the same folder as the app.py file you’ve just worked on. Once you navigate into the .chalice folder, you’ll see the policy-dev.json file. Open it in VS Code and replace the contents with the contents of [listing 8.3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08ex3).

**Note**

If you don’t want to type or copy and paste, you can download the policy-dev.json file here: [https://s3.amazonaws.com/mlforbusiness/ch08/policy-dev.json](https://s3.amazonaws.com/mlforbusiness/ch08/policy-dev.json).

**Listing 8.3. Contents of policy-dev.json**

```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "VisualEditor0",
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogStream",
                "logs:PutLogEvents",
                "logs:CreateLogGroup"
            ],
            "Resource": "arn:aws:logs:*:*:*"
        },
        {
            "Sid": "VisualEditor1",
            "Effect": "Allow",
            "Action": "sagemaker:InvokeEndpoint",     1
            "Resource": "*"
        }
    ]
}
```

* _1_ Adds permission to invoke a SageMaker endpoint

Your API now has permission to invoke the SageMaker endpoint. There is still one more step to do before you can deploy the code to AWS.

**8.6.5. Updating requirements.txt**

You need to instruct the Lambda function to install the slugify so it can be used by the application. To do this, you add the line in [listing 8.4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08ex4) to the requirements.txt file located in the same folder as the app.py file.

**Note**

You can download the file here: [https://s3.amazonaws.com/mlforbusiness/ch08/requirements.txt](https://s3.amazonaws.com/mlforbusiness/ch08/requirements.txt).

**Listing 8.4. Contents of requirements.txt**

```
python-slugify
```

The requirements.txt update is the final step you need to do before you are ready to deploy Chalice.

**8.6.6. Deploying Chalice**

At last, it’s time to deploy your code so that you can access your endpoint. In the terminal window in VS Code, from the tweet\_escalator folder, type:

```
chalice deploy
```

This regenerates your Lambda function on AWS with a few additions:

* The Lambda function now has permission to invoke the SageMaker endpoint.
* The Lambda function has installed the slugify library so it can be used by the function.

#### 8.7. Serving decisions <a href="#ch08lev1sec7__title" id="ch08lev1sec7__title"></a>

To recap, in this chapter, you have set up the SageMaker endpoint, configured AWS on your computer, and created and deployed the web endpoint. Now you can start using it. We’re finally at the last step in the process:

1. Set up the SageMaker endpoint.
2. Configure AWS on your local computer.
3. Create a web endpoint.
4. Serve decisions.

To view your API, click the Rest API URL link that is shown in your terminal window after you run chalice deploy ([figure 8.22](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08fig22)). This still brings up the Hello World page because we didn’t change the output ([figure 8.23](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08fig23)).

**Figure 8.22. The Rest API URL used to access the endpoint in a web browser.**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch08fig22\_alt.jpg)

**Figure 8.23. Hello World, again**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch08fig23\_alt.jpg)

To view the response to a tweet, you need to enter the route in the address bar of your browser. An example of the route you need to add is shown in [figure 8.24](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08fig24). At the end of the URL in the address bar of your browser (after the final /), you type tweet/the-text-of-the-tweet-with-dashes-instead-of-spaces and press ![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/enter.jpg).

**Figure 8.24. Tweet response: I am very angry**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch08fig24\_alt.jpg)

The response displayed on the web page now changes from {"hello": "world"} to

```
{"Tweet":"I-am-very-angry","Tokenized tweet":["i am very angry"],
     "Escalate":"Yes","Confidence":1.0000098943710327}
```

The response shows the tweet it pulled from the address bar, the tokenized tweet after running it through slugify, the recommendation on whether to escalate the tweet or not (in this case the answer is Yes), and the confidence of the recommendation.

To test additional phrases, simply type them into the address bar. For example, entering thanks-i-am-happy-with-your-service generates the response shown in [figure 8.25](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08fig25). As expected, the recommendation is to not escalate this tweet.

**Figure 8.25. Tweet response: I am happy with your service**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch08fig25\_alt.jpg)

It is interesting to see the results for negating a tweet such as turning “I am very angry” to “I am not angry.” You might expect that the API would recommend not escalating this, but that is often not the case. [Figure 8.26](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_019.html#ch08fig26) shows the response to this tweet. You can see it still recommends escalation, but its confidence is much lower—down to 52%.

**Figure 8.26. Tweet response: I am not angry**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch08fig26\_alt.jpg)

To see why it was escalated, you need to look at the data source for the tweets. When you look at the negated tweets, you see that many of the tweets were labeled Escalate because the negated phrase was part of a longer tweet that expressed frustration. For example, a common tweet pattern was for a person to tweet “I’m not angry, I’m just disappointed.”

#### Summary <a href="#ch08lev1sec8__title" id="ch08lev1sec8__title"></a>

* In order to deploy a machine learning model on the internet, you need to expose that endpoint to the internet. Nowadays, you can serve web applications containing your model without worrying about the complexities of setting up a web server.
* AWS Lambda is a web server that boots on demand and is a much more cost-effective way to serve predictions from your API.
* The SageMaker endpoint can also serve predictions to an API, which can then be used to serve predictions and decisions to users over the web, and you can secure the endpoint and make it available only to those who have permission to access it.
* To pass the text of the tweet as a URL, the text of the tweet can’t contain any characters that are not permitted to be typed in the address bar of the browser.
* You set up a SageMaker endpoint with slugify (rather than NLTK) to normalize tweets. Slugify is commonly used to turn text into website URLs.
* You set up the serverless API SageMaker endpoint on AWS serverless technology, which is called AWS Lambda. In order for AWS Lambda to know what to do, you install the Chalice Python library.
* To access the SageMaker endpoint, your serverless API needs to have permission to do so. Using Microsoft’s Visual Studio Code (VS Code), you set up credentials in your AWS account by creating an access key, then setting up your AWS credentials on your local computer.
* You set up the AWS command-line interface (CLI) and boto3 libraries on your local computer so you can work with AWS resources from your local machine.
* To create a serverless function, you learned about AWS Lambda functions and the AWS API Gateway services and how easy it is to use them with Chalice.
* You deployed an API that returns recommendations about whether to escalate a tweet to Naomi’s support team.
