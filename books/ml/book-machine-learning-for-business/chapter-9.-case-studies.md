# Chapter 9. Case studies

_This chapter covers_

* Review of the topics in this book
* How two companies using machine learning improved their business
  * Case study 1: Implementing a single machine learning project in your company
  * Case study 2: Implementing machine learning at the heart of everything your company does

Throughout the book, you have used AWS SageMaker to build solutions to common business problems. The solutions have covered a broad range of scenarios and approaches:

* Using XGBoost supervised learning to solve an approval routing challenge
* Reformatting data so that you could use XGBoost again, but this time to predict customer churn
* Using BlazingText and Natural Language Processing (NLP) to identify whether a tweet should be escalated to your support team
* Using unsupervised Random Cut Forest to decide whether to query a supplier’s invoice
* Using DeepAR to predict power consumption based on historical trends
* Adding datasets such as weather forecasts and scheduled holidays to improve DeepAR’s predictions

In the previous chapter, you learned how to serve your predictions and decisions over the web using AWS’s serverless technology. Now, we’ll wrap it all up with a look at how two different companies are implementing machine learning in their business.

In [chapter 1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_010.html#ch01), we put forward our view that we are on the cusp of a massive surge in business productivity and that this surge is going to be due in part to machine learning. Every company wants to be more productive, but they find it’s difficult to achieve this goal. Until the advent of machine learning, if a company wanted to become more productive, they needed to implement and integrate a range of best-in-class software or change their business practices to conform exactly to how their ERP (Enterprise Resource Planning) system worked. This greatly slows the pace of change in your company because your business either comprises a number of disparate systems, or it’s fitted with a straightjacket (also known as your ERP system). With machine learning, a company can keep many of their operations in their core systems and use machine learning to assist with automating decisions at key points in the process. Using this approach, a company can maintain a solid core of systems yet still take advantage of the best available technology.

In each of the chapters from 2 through 7, we looked at how machine learning could be used to make a decision at a particular point in a process (approving purchase orders, reconnecting with customers at risk of churning, escalating tweets, and reviewing invoices), and how machine learning can be used to generate predictions based on historical data combined with other relevant datasets (power consumption prediction based on past usage and other information such as forecasted weather and upcoming holidays).

The two case studies we look at in this chapter show several different perspectives when adopting machine learning in business. The first case study follows a labor-hire company as it uses machine learning to automate a time-consuming part of its candidate interview process. This company is experimenting with machine learning to see how it can solve various challenges in their business. The second case study follows a software company that already has machine learning at its core but wants to apply it to speed up more of its workflow. Let’s jump in and look at how the companies in these case studies use machine learning to enhance their business practices.

#### 9.1. Case study 1: WorkPac <a href="#ch09lev1sec1__title" id="ch09lev1sec1__title"></a>

WorkPac is Australia’s largest privately held labor-hire company. Every day, tens of thousands of workers are contracted out across thousands of clients. And every day, to maintain a suitable pool of candidates, WorkPac has teams of people interviewing candidates.

The interview process can be thought of as a pipeline where candidates go into a funnel at the top and are categorized into broad categories as they progress down the funnel. Recruiters who are experts in a particular category apply metadata to the candidates so they can be filtered based on skills, experience, aptitude, and interest. Applying these filters allows the right pool of candidates to be identified for each open position.

[Figure 9.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_020.html#ch09fig01) shows a simplified view of the categorization process. Candidate resumes go into the top of the funnel and are classified into different job categories.

**Figure 9.1. Funnel to categorize candidates into different types of jobs**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch09fig01\_alt.jpg)

The anticipated benefit of automating the categorization funnel is that it frees up time for recruiters to focus on gathering metadata about candidates rather than classifying candidates. An additional benefit of using a machine learning model to perform the classification is that, as a subsequent phase, some of the metadata gathering can also be automated.

Before implementing the machine learning application, when a candidate submits their resume through WorkPac’s candidate portal, it’s categorized by a generalist recruiter and potentially passed on to a specialist recruiter for additional metadata. After passing through this process, the candidate would then be available for other recruiters to find. For example, if the candidate was classified as a truck driver, then recruiters looking to fill a truck driver role would be able to find this candidate.

Now that WorkPac has implemented the machine learning application, the initial classification is performed by a machine learning algorithm. The next phase of this project is to implement a chat bot that can elicit some of the metadata, further freeing up valuable recruiter time.

**9.1.1. Designing the project**

WorkPac considered two approaches to automating the classification of candidates:

* A simple keyword classification system for candidates
* A machine learning approach to classify candidates

The keyword classification system was perceived as low risk but also low reward. Like the approval-routing scenario you looked at in [chapter 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02), _keyword classification_ requires ongoing time and effort to identify new keywords. For example, if Caterpillar releases a new mining truck called a 797F, WorkPac has to update their keyword list to associate that term with truck driver. Adopting a machine learning approach ensures that as new vehicles are released by manufacturers, for example, the machine learning model learns to associate 797F vehicles with truck drivers.

The machine learning approach was perceived as higher reward but also higher risk because it would be WorkPac’s first time delivering a machine learning project. A machine learning project is a different beast than a standard IT project. With a typical IT project, there are standard methodologies to define the project and the outcomes. When you run an IT project, you know in advance what the final outcome will look like. You have a map, and you follow the route to the end. But with a machine learning project, you’re more like an explorer. Your route changes as you learn more about the terrain. Machine learning projects are more iterative and less predetermined.

To help overcome these challenges, WorkPac retained the services of Blackbook.ai to assist them. Blackbook.ai is an automation and machine learning software company that services other businesses. WorkPac and Blackbook.ai put together a project plan that allowed them to build trust in the machine learning approach by delivering the solution in stages. The stages this project progressed through are typical of machine learning automation projects in general:

* _Stage 1_—Prepare and test the model to validate that decisions can be made using machine learning.
* _Stage 2_—Implement proof of concept (POC) around the workflow.
* _Stage 3_—Embed the process into the company’s operations.

**9.1.2. Stage 1: Preparing and testing the model**

Stage 1 involved building a machine learning model to classify existing resumes. WorkPac had more than 20 years of categorized resume data, so they had plenty of data to use for training. Blackbook.ai used OCR technology to extract text from the resumes and trained the model on this text. Blackbook.ai had enough data that they were able to balance out the classes by selecting equal numbers of resumes across each of the job categories. After training and tuning the model, the model was able to hit an F Score of 0.7, which was deemed to be suitable for this activity.

F scores

An _F score_ (also known as an F1 score) is a measure of the performance of a machine learning model. In [chapter 3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03), you learned how to create a confusion matrix showing the number of false positive and false negative predictions. An F score is another way of summarizing the results of a machine learning model. An example is the best way to see how an F score is calculated.

The following table summarizes the results of a machine learning algorithm that made 50 predictions. The algorithm was attempting to predict whether a particular candidate should be classified as a truck driver, assayer, or engineer.

**Table 9.1. Data table showing candidate predictions**

|                       | Prediction (truck driver) | Prediction (assayer) | Prediction (engineer) | Total |
| --------------------- | ------------------------- | -------------------- | --------------------- | ----- |
| Actual (truck driver) | 11                        | 4                    | 0                     | 15    |
| Actual (assayer)      | 4                         | 9                    | 2                     | 15    |
| Actual (engineer)     | 3                         | 3                    | 14                    | 20    |
| Total                 | 18                        | 16                   | 16                    | 50    |

The top row of the table indicates that out of 15 candidates who were actually truck drivers, the algorithm correctly predicted that 11 were truck drivers and incorrectly predicted that 4 were assayers. The algorithm did not incorrectly predict that any truck drivers were engineers. Likewise for the second row (Assayer), the algorithm correctly predicted that nine assayer candidates were indeed assayers, but it incorrectly predicted that four truck drivers were assayers and two engineers were assayers. If you look at the top row (the actual truck driver), you would say that 11 out of 15 predictions were correct. This is known as the _precision_ of the algorithm. A result of 11/15 means the algorithm has a precision of 73% for truck drivers.

You can also look at each column of data. If you look at the first column, Prediction (truck driver), you can see that the algorithm predicted 18 of the candidates were truck drivers. Out of 18 predictions, it got 11 right and 7 wrong. It predicted 4 assayers were truck drivers and 3 engineers were assayers. This is known as the _recall_ of the algorithm. The algorithm correctly recalls 11 out of 18 predictions (61%).

From this example, the importance of both precision and recall can be seen. The precision result of 73% looks pretty good, but the results look less favorable when you consider that only 61% of its truck driver predictions are correct. The F score reduces this number to a single value using the following formula:

((Precision \* Recall) / (Precision + Recall)) \* 2

Using the values from the table, the calculation is

((.73 \* .61) / (.73 + .61)) \* 2 = 0.66

so the F score of the first row is 0.66. Note that if you average the F scores across a table for a multiclass algorithm (as in this example), the result will typically be close to the precision. But it’s useful to look at the F score for each class to see if any of these have wildly different recall results.

During this stage, Blackbook.ai developed and honed their approach for transforming resumes into data that could be fed into their machine learning model. In the model development phase, a number of the steps in this process were manual, but Blackbook.ai had a plan to automate each of these steps. After achieving an F score in excess of 0.7 and armed with a plan for automating the process, Blackbook.ai and WorkPac moved on to stage 2 of the project.

**9.1.3. Stage 2: Implementing proof of concept (POC)**

The second stage involves building a POC that incorporates the machine learning model into WorkPac’s workflow. Like many business process-improvement projects involving machine learning, this part of the project took longer than the machine learning component. From a risk perspective, this part of the project was a standard IT project.

In this stage, Blackbook.ai built a workflow that took resumes uploaded from candidates, classified the resumes, and presented the resumes and the results of the classification to a small number of recruiters in the business. Blackbook.ai then took feedback from the recruiters and incorporated recommendations into the workflow. Once the workflow was approved, they moved on to the final stage of the project—implementation and rollout.

**9.1.4. Stage 3: Embedding the process into the company’s operations**

The final stage of the project was to roll out the process across all of WorkPac. This is typically time consuming as it involves building error-catching routines that allow the process to function in production, and training staff in the new process. Although time consuming, this stage can be low risk, providing the feedback from the stage 2 users is positive.

**9.1.5. Next steps**

Now that resumes are being automatically classified, WorkPac can build and roll out chatbots that are trained on a particular job type to get metadata from candidates (such as work history and experience). This allows their recruiters to focus their efforts on the highest-value aspects of their jobs, rather than spending time gathering information about the candidates.

**9.1.6. Lessons learned**

One of the time-consuming aspects of a machine learning project is getting the data to feed the model. In this case, the data was locked away in resume documents in PDF format. Rather than spending time building their own OCR data-extraction service, Blackbook.ai solved this problem by using a commercial resume data-extraction service. This allowed them to get started right away at a low cost. If the cost of this service becomes too high down the track, a separate business case can be prepared to replace the OCR service with an in-house application.

To train the machine learning model, Blackbook.ai also required metadata about the existing documents. Getting this metadata required information to be extracted from WorkPac’s systems using SQL queries, and it was time consuming to get this data from WorkPac’s internal teams. Both WorkPac and Blackbook.ai agreed this should have been done in a single workshop rather than as a series of requests over time.

#### 9.2. Case study 2: Faethm <a href="#ch09lev1sec2__title" id="ch09lev1sec2__title"></a>

Faethm is a software company with artificial intelligence (AI) at its core. At the heart of Faethm’s software is a system that predicts what a company (or country) could look like several years from now, based on the current structure of its workforce and the advent of emerging technologies like machine learning, robotics, and automation. Faethm’s data science team accounts for more than a quarter of their staff.

**9.2.1. AI at the core**

What does it mean to have AI at the core of the company? [Figure 9.2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_020.html#ch09fig02) shows how Faethm’s platform is constructed. Notice how every aspect of the platform is designed to drive data to Faethm’s AI engine.

Faethm combines their two main data models—Technology Adoption Model and Workforce Attribution Model—with client data in their AI engine to predict how a company will change over the coming years.

**9.2.2. Using machine learning to improve processes at Faethm**

This case study doesn’t focus on how Faethm’s AI engine predicts how a company will change over the coming years. Instead, it focuses on a more operational aspect of their business: how can it onboard new customers faster and more accurately? Specifically, how can it more accurately match their customers’ workforce to Faethm’s job categorization? This process fits section 4, Contextual Client Data, which is shown in Faethm’s Platform Construct ([figure 9.2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_020.html#ch09fig02)).

**Figure 9.2. Every aspect of Faethm’s operating model drives data toward its AI engine.**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch09fig02\_alt.jpg)

[Figure 9.3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_020.html#ch09fig03) shows a company’s organizational structure being converted to Faethm’s job classification. Correctly classifying jobs is important because the classified jobs serve as a starting point for Faethm’s modeling application. If the jobs do not reflect their customer’s current workforce, the end result will not be correct.

**Figure 9.3. Job description categorization funnel**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch09fig03\_alt.jpg)

At first glance, this looks like a challenge similar to what WorkPac faced, in that both Faethm and WorkPac are classifying jobs. The key difference is the incoming data: WorkPac has 20 years of labeled resume data, whereas Faethm has only a few years of job title data. So Faethm broke the project down into four stages:

* _Stage 1_—Get the data
* _Stage 2_—Identify features
* _Stage 3_—Validate the results
* _Stage 4_—Implement in production

**9.2.3. Stage 1: Getting the data**

When Faethm started operations in 2017, the team manually categorized their customers’ job title data. Over time, it developed several utility tools to speed up the process, but categorizing the job titles for incoming clients still required manual effort from expert staff. Faethm wanted to use its considerable machine learning expertise to automate this process.

Faethm decided to use SageMaker’s BlazingText algorithm. This was due, in part, to the fact that BlazingText handles out-of-vocabulary words by creating vectors from sub-words.

What are out-of-vocabulary words?

As discussed in [chapter 4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_014.html#ch04), BlazingText turns words into a string of numbers called a _vector_. The vector represents not only a word, but also the different contexts it appears in. If the machine learning model only creates vectors from whole words, then it cannot do anything with a word that it hasn’t been specifically trained on.

With job titles, there are lots of words that might not appear in training data. For example, the model might be trained to recognize gastroenterologist and neuroradiologist, but it can stumble when it comes across a gastrointestinal radiologist. BlazingText’s sub-word vectors allow the model to handle words like gastrointestinal radiologist, for example, because it creates vectors from _gas_, _tro_, _radio_, and _logist_, even though these terms are only sub-words of any of the words the model is trained on.

The first problem Faethm needed to surmount was getting sufficient training data. Instead of waiting until it had manually classified a sufficient number of clients, Faethm used its utility tools to create a large number of classified job titles similar to, but not exactly the same as, existing companies. This pool of companies formed the training dataset.

Training data

You might not have to worry about labeling your data. WorkPac was able to use undersampling and oversampling to balance their classes because they had 20 years of labeled data. When you are looking at machine learning opportunities in your business, you might find yourself in a similar position in that the processes most amenable to implementing machine learning are those that have been done by a person for a long time, and you have their historical decisions to use as training data.

An additional complication with Faethm’s data was that their classes were imbalanced. Some of the jobs they classified into titles had hundreds of samples (Operations Manager, for example). Others had only one. To address this imbalance, Faethm adjusted the weighting of each category (like you did using XGBoost in [chapter 3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_013.html#ch03)). Now armed with a large labeled dataset, Faethm could begin building the model.

**9.2.4. Stage 2: Identifying the features**

Once they had the data, Faethm looked at other features that might be relevant for classifying job titles into roles. Two features found to be important in the model were industry and salary. For example, an analyst in a consulting firm or bank is usually a different role than an analyst in a mining company, and an operations manager earning $50,000 per year is likewise a different role than an operations manager earning $250,000 per year.

By requesting the anonymised employee\_id of each employee’s manager, Faethm was able to construct two additional features: first, the ratio of employees with each title who have direct reports; and second, the ratio of employees with each title who have managers reporting to them. The addition of these two features resulted in a further significant improvement in accuracy.

**9.2.5. Stage 3: Validating the results**

After building the model in SageMaker, Faethm was able to automatically categorize a customer’s workforce into jobs that serve as inputs into Faethm’s predictive model. Faethm then classified the workforce using its human classifiers and identified the anomalies. After several rounds of tuning and validation, Faethm was able to move the process into production.

**9.2.6. Stage 4: Implementing in production**

Implementing the algorithm in production was simply a matter of replacing the human decision point with the machine learning algorithm. Instead of making the decision, Faethm’s expert staff spend their time validating the results. As it takes less time to validate than it does to classify, their throughput is greatly improved.

#### 9.3. Conclusion <a href="#ch09lev1sec3__title" id="ch09lev1sec3__title"></a>

In the case studies, you progressed from a company taking its first steps in machine learning to a company with machine learning incorporated into everything it does. The goal of this book has been to provide you with the context and skills to use machine learning in your business.

Throughout the book, we provided examples of how machine learning can be applied at decision points in your business activities so that a person doesn’t have to be involved in those processes. By using a machine learning application, rather than a human, to make decisions, you get the dual benefits of a more consistent and a more robust result than when using rules-based programming.

In this chapter, we have shown different perspectives on machine learning from companies using machine learning today. In your company, each of the following perspectives is helpful in evaluating which problems you should tackle and why.

**9.3.1. Perspective 1: Building trust**

WorkPac and Blackbook.ai made sure that the projects had achievable and measurable outcomes, delivered in bite-sized chunks throughout the project. These companies also made sure they reported progress regularly, and they weren’t overpromising during each phase. This approach allowed the project to get started without requiring a leap of faith from WorkPac’s executive team.

**9.3.2. Perspective 2: Geting the data right**

There are two ways to read the phrase _Getting the data right_. Both are important. The first way to read the phrase is that the data needs to be as accurate and complete as possible. The second way to read the phrase is that you need to correctly build the process for extracting the data and feeding it into the machine learning process.

When you move into production, you need to be able to seamlessly feed data into your model. Think about how you are going to do this and, if possible, set this up in your training and testing processes. If you automatically pull your data from source systems during development, that process will be well tested and robust when you move to production.

**9.3.3. Perspective 3: Designing your operating model to make the most of- f your machine learning capability**

Once you have the ability to use machine learning in your company, you should think about how to you can use this functionality in as many places as possible and how you can get as many transactions as possible flowing through the models. Faethm’s first question when considering a new initiative was probably, “How can this feed our AI engine?” In your company, when looking at a new business opportunity, you’ll want to ask, “How can this new opportunity fit into our existing models or be used to bolster our current capability?”

**9.3.4. Perspective 4: What does your company look like once you are usin- ng machine learning everywhere?**

As you move from your first machine learning projects to using machine learning everywhere, your company will look very different. In particular, the shape of your workforce will change. Preparing your workforce for this change is key to your success.

Armed with these perspectives and the skills you picked up as you’ve worked your way through the chapters in this book, we hope you are ready to tackle processes within your own company. If you are, then we’ve achieved what we set out to do, and we wish you every success.

#### Summary <a href="#ch09lev1sec4__title" id="ch09lev1sec4__title"></a>

* You followed WorkPac as they embarked on their first machine learning project.
* You saw how Faethm, an experienced machine learning company, incorporated machine learning into yet another of its processes.
