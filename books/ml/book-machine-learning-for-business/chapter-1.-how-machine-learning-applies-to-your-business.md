# Chapter 1. How machine learning applies to your business

_This chapter covers_

* Why our business systems are so terrible
* What machine learning is
* Machine learning as a key to productivity
* Fitting machine learning with business automation
* Setting up machine learning within your company

Technologists have been predicting for decades that companies are on the cusp of a surge in productivity, but so far, this has not happened. Most companies still use people to perform repetitive tasks in accounts payable, billing, payroll, claims management, customer support, facilities management, and more. For example, all of the following small decisions create delays that make you (and your colleagues) less responsive than you want to be and less effective than your company needs you to be:

* To submit a leave request, you have to click through a dozen steps, each one requiring you to enter information that the system should already know or to make a decision that the system should be able to figure out from your objective.
* To determine why your budget took a hit this month, you have to scroll through a hundred rows in a spreadsheet that you’ve manually extracted from your finance system. Your systems should be able to determine which rows are anomalous and present them to you.
* When you submit a purchase order for a new chair, you know that Bob in procurement has to manually make a bunch of small decisions to process the form, such as whether your order needs to be sent to HR for ergonomics approval or whether it can be sent straight to the financial approver.

We believe that you will soon have much better systems at work—machine learning applications will automate all of the small decisions that currently hold up processes. It is an important topic because, over the coming decade, companies that are able to become more automated and more productive will overtake those that cannot. And machine learning will be one of the key enablers of this transition.

This book shows you how to implement machine learning, decision-making systems in your company to speed up your business processes. “But how can I do that?” you say. “I’m technically minded and I’m pretty comfortable using Excel, and I’ve never done any programming.” Fortunately for you, we are at a point in time where any technically minded person can learn how to help their company become dramatically more productive. This book takes you on that journey. On that journey, you’ll learn

* How to identify where machine learning will create the greatest benefits within your company in areas such as
  * Back-office financials (accounts payable and billing)
  * Customer support and retention
  * Sales and marketing
  * Payroll and human resources
* How to build machine learning applications that you can implement in your company

#### 1.1. Why are our business systems so terrible? <a href="#ch01lev1sec1__title" id="ch01lev1sec1__title"></a>

> _“The man who goes alone can start today; but he who travels with another must wait till that other is ready.”_
>
> _Henry David Thoreau_

Before we get into how machine learning can make your company more productive, let’s look at why implementing systems in your company is more difficult than adopting systems in your personal life. Take your personal finances as an example. You might use a money management app to track your spending. The app tells you how much you spend and what you spend it on, and it makes recommendations on how you could increase your savings. It even automatically rounds up purchases to the nearest dollar and puts the spare change into your savings account. At work, expense management is a very different experience. To see how your team is tracking against their budget, you send a request to the finance team, and they get back to you the following week. If you want to drill down into particular line items in your budget, you’re out of luck.

There are two reasons why our business systems are so terrible. First, although changing our own behavior is not easy, changing the behavior of a group of people is really hard. In your personal life, if you want to use a new money management app, you just start using it. It’s a bit painful because you need to learn how the new app works and get your profile configured, but still, it can be done without too much effort. However, when your company wants to start using an expense management system, everyone in the company needs to make the shift to the new way of doing things. This is a much bigger challenge. Second, managing multiple business systems is really hard. In your personal life, you might use a few dozen systems, such as a banking system, email, calendar, maps, and others. Your company, however, uses hundreds or even thousands of systems. Although managing the interactions between all these systems is hard for your IT department, they encourage you to use their _end-to-end enterprise software system_ for as many tasks as possible.

The end-to-end enterprise software systems from software companies like SAP and Oracle are designed to run your entire company. These end-to-end systems handle your inventory, pay staff, manage the finance department, and handle most other aspects of your business. The advantage of an end-to-end system is that everything is integrated. When you buy something from your company’s IT catalog, the catalog uses your employee record to identify you. This is the same employee record that HR uses to store your leave request and send you paychecks. The problem with end-to-end systems is that, because they do everything, there are better systems available for each thing that they do. Those systems are called _best-of-breed systems_.

Best-of-breed systems do one task particularly well. For example, your company might use an expense management system that rivals your personal money management application for ease of use. The problem is that this expense management system doesn’t fit neatly with the other systems your company uses. Some functions duplicate existing functions in other systems ([figure 1.1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_010.html#ch01fig01)). For example, the expense management system has a built-in approval process. This approval process duplicates the approval process you use in other aspects of your work, such as approving employee leave. When your company implements the best-of-breed expense management system, it has to make a choice: does it use the expense management approval workflow and train you to use two different approval processes? Or does it integrate the expense management system with the end-to-end system so you can approve expenses in the end-to-end system and then pass the approval back into the expense management system?

**Figure 1.1. Best-of-breed approval function overlaps the end-to-end system approval function.**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch01fig01\_alt.jpg)

To get a feel for the pros and cons of going with an end-to-end versus a best-of-breed system, imagine you’re a driver in a car rally that starts on paved roads, then goes through desert, and finally goes through mud. You have to choose between putting all-terrain tires on your car or changing your tires when you move from pavement to sand and from sand to mud. If you choose to change your tires, you can go faster through each of the sections, but you lose time when you stop and change the tires with each change of terrain. Which would you choose? If you could change tires quickly, and it helped you go much faster through each section, you’d change tires with each change of terrain.

Now imagine that, instead of being the driver, your job is to support the drivers by providing them with tires during the race. You’re the Chief Tire Officer (CTO). And imagine that instead of three different types of terrain, you have hundreds, and instead of a few drivers in the race, you have thousands. As CTO, the decision is easy: you’ll choose the all-terrain tires for all but the most specialized terrains, where you’ll reluctantly concede that you need to provide specialty tires. As a driver, the CTO’s decision sometimes leaves you dissatisfied because you end up with a system that is clunkier than the systems you use in your personal life.

We believe that over the coming decade, machine learning will solve these types of problems. Going back to our metaphor about the race, a machine learning application would automatically change the characteristics of your tires as you travel through different terrains. It would give you the best of both worlds by rivaling best-of-breed performance while utilizing the functionality in your company’s end-to-end solution.

As another example, instead of implementing a best-of-breed expense management system, your company could implement a machine learning application to

* Identify information about the expense, such as the amount spent and the vendor name
* Decide which employee the expense belongs to
* Decide which approver to submit the expense claim to

Returning to the example of overlapping approval functions, by using machine learning in conjunction with your end-to-end systems, you can automate and improve your company’s processes without implementing a patchwork of best-of-breed systems ([figure 1.2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_010.html#ch01fig02)).

**Figure 1.2. Machine learning enhances the functionality of end-to-end systems.**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch01fig02\_alt.jpg)

Is there no role for best-of-breed systems in the enterprise?

There is a role for best-of-breed systems in the enterprise, but it is probably different than the role these systems have filled over the past 20 years or so. As you’ll see in the next section, the computer era (1970 to the present) has been unsuccessful in improving the productivity of businesses. If best-of-breed systems were successful at improving business productivity, we should have seen some impact on the performance of businesses that use best-of-breed systems. But we haven’t.

So what will happen to the best-of-breed systems? In our view, the best-of-breed systems will become

* More integrated into a company’s end-to-end system
* More modular so that a company can adopt some of the functions, but not others

Vendors of these best-of-breed systems will base their business cases on the use of problem-specific machine learning applications to differentiate their offerings from those of their competitors or on solutions built in-house by their customers. Conversely, their profit margins will get squeezed as more companies develop the skills to build machine learning applications themselves rather than buying a best-of-breed solution.

#### 1.2. Why is automation important now? <a href="#ch01lev1sec2__title" id="ch01lev1sec2__title"></a>

We are on the cusp of a dramatic improvement in business productivity. Since 1970, business productivity in mature economies such as the US and Europe has barely moved, compared to the change in the processing power of computers, and this trend has been clearly visible for decades now. Over that period of time, business productivity has merely doubled, whereas the processing power of computers is 20 million times greater!

If computers were really helping us become more productive, why is it that much faster computers don’t lead to much greater productivity? This is one of mysteries of modern economics. Economists call this mystery the _Solow Paradox_. In 1987, Robert Solow, an American economist, quipped:

> _“You can see the computer age everywhere but in the productivity statistics.”_

Is the failure of businesses to become more productive just a feature of business? Are businesses at maximum productivity now? We don’t think so. Some companies have found a solution to the Solow Paradox and are rapidly improving their productivity. And we think that they will be joined by many others—hopefully, yours as well.

[Figure 1.3](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_010.html#ch01fig03) is from a 2017 speech on productivity given by Andy Haldane, Chief Economist for the Bank of England.\[[1](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_010.html#ch01fn01)] It shows that since 2002, the top 5% of companies have increased productivity by 40%, while the other 95% of companies have barely increased productivity at all.\[[2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_010.html#ch01fn02)] This low-growth trend is found across nearly all countries with mature economies.

> 1
>
> Andy Haldane, “Productivity Puzzles,” [https://www.bis.org/review/r170322b.pdf](https://www.bis.org/review/r170322b.pdf).

> 2
>
> Andy Haldane dubbed the top 5% of companies frontier firms.

**Figure 1.3. Comparison of productivity across frontier firms (the top 5%) versus all companies**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch01fig03\_alt.jpg)

**1.2.1. What is productivity?**

Productivity is measured at a country level by dividing the annual Gross Domestic Product (GDP) by the number of hours worked in a year. The GDP per hour worked in the UK and the US is currently just over US$100. In 1970, it was between US$45 and US$50. But the GDP per hour worked by the top 5% of firms (the frontier firms) is over US$700 and rising.

The frontier firms were able to hit such a high GDP per hour by minimizing human effort to generate each dollar of revenue. Or, to put it another way, these firms _automate_ everything that can be automated. We predict that productivity growth will improve rapidly as more companies figure out how to replicate what the top companies are doing and will make the jump from their current level of productivity to the top levels of productivity.

We believe that we’re at the end of the Solow Paradox; that machine learning will enable many companies to hit the productivity levels we see in the top 5% of companies. And we believe that those companies that do not join them, that don’t dramatically improve their productivity, will wither and die.

**1.2.2. How will machine learning improve productivity?**

In the preceding sections, we looked at why companies struggle to become more automated and the evidence showing that, while company productivity has not improved much over the past 50 years, there is a group of frontier firms becoming more productive by automating everything that can be automated. Next we’ll look at how machine learning can help your company become a frontier firm before showing you how you can help your company make the shift.

For our purposes, _automation_ is the use of software to perform a repetitive task. In the business world, repetitive tasks are everywhere. A typical retail business, for example, places orders with suppliers, sends marketing material to customers, manages products in inventory, creates entries in their accounting system, makes payments to their staff, and hundreds of other things.

Why is it so hard to automate these processes? From a higher level, these processes look pretty simple. Sending marketing material is just preparing content and emailing it to customers. Placing orders is simply selecting product from a catalog, getting it approved, and sending the order to a supplier. How hard can it be?

The reason automation is hard to implement is because, even though these processes look repetitive, there are small decisions that need to be made at several steps along the way. This is where machine learning fits in. You can use machine learning to make these decisions at each point in the process in much the same way a human currently does.

#### 1.3. How do machines make decisions? <a href="#ch01lev1sec3__title" id="ch01lev1sec3__title"></a>

For the purposes of this book, think of machine learning as a way to arrive at a decision, based on patterns in a dataset. We’ll call this _pattern-based decision making_. This is in contrast to most software development these days, which is _rules-based decision making_--where programmers write code that employs a series of rules to perform a task.

When your marketing staff sends out an email newsletter, the marketing software contains code that queries a database and pulls out only those customers selected by the query (for example, males younger than 25 who live within 20 kilometers of a certain clothing outlet store). Each person in the marketing database can be identified as being in this group or not in this group.

Contrast this with machine learning where the query for your database might be to pull out all users who have a purchasing history similar to that of a specific 23-year-old male who happens to live close to one of your outlet stores. This query will get a lot of the same people that the rules-based query gets, but it will also return those who have a similar purchasing pattern and are willing to drive further to get to your store.

**1.3.1. People: Rules-based or not?**

Many businesses rely on people rather than software to perform routine tasks like sending marketing material and placing orders with suppliers. They do so for a number of reasons, but the most prevalent is that it’s easier to teach a person how to do a task than it is to program a computer with the rules required to perform the same task.

Let’s take Karen, for example. Her job is to review purchase orders, send them to an approver, and then email the approved purchase orders to the supplier. Karen’s job is both boring and tricky. Every day, Karen makes dozens of decisions about who should approve which orders. Karen has been doing this job for several years, so she knows the simple rules, like IT products must be approved by the IT department. But she also knows the exceptions. For example, she knows that when Jim orders toner from the stationery catalog, she needs to send the order to IT for approval, but when Jim orders a new mouse from the IT catalog, she does not.

The reason Karen’s role hasn’t been automated is because programming all of these rules is hard. But harder still is maintaining these rules. Karen doesn’t often apply her “fax machine” rule anymore, but she is increasingly applying her “tablet stylus” rule, which she has developed over the past several years. She considers a tablet stylus to be more like a mouse than a laptop computer, so she doesn’t send stylus orders to IT for approval. If Karen really doesn’t know how to classify a particular product, she’ll call IT to discuss it; but for most things, she makes up her own mind.

Using our concepts of rules-based decision making versus pattern-based decision making, you can see that Karen incorporates a bit of both. Karen applies rules most of the time but occasionally makes decisions based on patterns. It’s the pattern-based part of Karen’s work that makes it hard to automate using a rules-based system. That’s why, in the past, it has been easier to have Karen perform these tasks than to program a computer with the rules to perform the same tasks.

**1.3.2. Can you trust a pattern-based answer?**

Lots of companies have manual processes. Often this is the case because there’s enough variation in the process to make automation difficult. This is where machine learning comes in.

Any point in a process where a person needs to make a decision is an opportunity to use machine learning to automate the decision or to present a restricted choice of options for the person to consider. Unlike rules-based programming, machine learning uses examples rather than rules to determine how to respond in a given situation. This allows it to be more flexible than rules-based systems. Instead of breaking when faced with a novel situation, machine learning simply makes a decision with a lower level of confidence.

Let’s look at the example of a new product coming into Karen’s catalog. The product is a voice-controlled device like Amazon Echo or Google Home. The device looks somewhat like an IT product, which means the purchase requires IT approval. But, because it’s also a way to get information into a computer, it kind of looks like an accessory such as a stylus or a mouse, which means the purchase doesn’t require IT approval.

In a rules-based system, this product would be unknown, and when asked to determine which approver to send the product to, the system could break. In a machine learning system, a new product won’t break the system. Instead, the system provides an answer with a lower level of confidence than it does for products it has seen before. And just like Karen could get it wrong, the machine learning application could get it wrong too. Accepting this level of uncertainty might be challenging for your company’s management and risk teams, but it’s no different than having Karen make those same decisions when a new product comes across her desk.

In fact, a machine learning system for business automation workflow can be designed to perform better than a human acting on their own. The optimal workflow often involves both systems and people. The system can be configured to cater to the vast majority of cases but have a mechanism where, when it has a low confidence level, it passes the case to a human operator for a decision. Ideally, this decision is fed back into the machine learning application so that, in the future, the application has a higher level of confidence in its decision.

It’s all well and good for you to say you’re comfortable with the result. In many instances, in order to make pattern-based decisions in your company, you’ll need the approval of your risk and management teams. In a subsequent section, once we take a look at the output of a pattern-based decision, you’ll see some potential ways of getting this approval.

**1.3.3. How can machine learning improve your business systems?**

So far in this chapter, we have been referring to the system that can perform multiple functions in your company as an end-to-end system. Commonly, these systems are referred to as ERP (Enterprise Resource Planning) systems.

ERP systems rose to prominence in the 1980s and 1990s. An _ERP system_ is used by many medium and large enterprises to manage most of their business functions like payroll, purchasing, inventory management, capital depreciation, and others. SAP and Oracle dominate the ERP market, but there are several smaller players as well.

In a perfect world, all of your business processes would be incorporated into your ERP system. But we don’t live in a perfect world. Your company likely does things slightly differently than your ERP’s default configuration, which creates a problem. You have to get someone to program your ERP to work the way your business does. This is expensive and time consuming, and can make your company less able to adjust to new opportunities as they arise. And, if ERP systems were the answer to all enterprise problems, then we should have seen productivity improvements during the uptake of ERP systems in the 1980s and 1990s. But there was little uptake in productivity during this period.

When you implement machine learning to support Karen’s decisions, there’s little change in the management process involved for your internal customers. They continue to place orders in the same ways they always have. The machine learning algorithms simply make some of the decisions automatically, and the orders get sent to approvers and suppliers appropriately and automatically. In our view, unless the process can be cleanly separated from the other processes in your company, the optimal approach is to first implement a machine learning automation solution and then, over time, migrate these processes to your ERP systems.

**Tip**

Automation is not the only way to become more productive. Before automating, you should ask whether you need to do the process at all. Can you create the required business value without automating?

#### 1.4. Can a machine help Karen make decisions? <a href="#ch01lev1sec4__title" id="ch01lev1sec4__title"></a>

Machine learning concepts are difficult to get one’s head around. This is, in part, due to the breadth of topics encompassed by the term _machine learning_. For the purposes of this book, think of machine learning as a tool that identifies patterns in data and, when you provide it with new data, it tells you which pattern the new data most closely fits.

As you read through other resources on machine learning, you will see that machine learning can cover many other things. But most of these things can be broken down into a series of decisions. Take machine learning systems for autonomous cars, for example. On the face of it, this sounds very different from the machine learning we are looking at. But it is really just a series of decisions. One machine learning algorithm looks at a scene and decides how to draw boxes around each of the objects in the scene. Another machine learning algorithm decides whether these boxes are things that need to be driven around. And, if so, a third algorithm decides the best way to drive around them.

To determine whether you can use machine learning to help out Karen, let’s look at the decisions made in Karen’s process. When an order comes in, Karen needs to decide whether to send it straight to the requester’s financial approver or whether she should send it to a technical approver first. She needs to send an order to a technical approver if the order is for a technical product like a computer or a laptop. She does not need to send it to a technical approver if it is not a technical product. And she does not need to send the order for technical approval if the requester is from the IT department. Let’s assess whether Karen’s example is suitable for machine learning.

In Karen’s case, the question she asks for every order is, “Should I send this for technical approval?” Her decision will either be yes or no. The things she needs to consider when making her decision are

* Is the product a technical product?
* Is the requester from the IT department?

In machine learning lingo, Karen’s decision is called the _target variable_, and the types of things she considers when making the decision are called _features_. When you have a target variable and features, you can use machine learning to make a decision.

**1.4.1. Target variables**

Target variables come in two flavors:

* Categorical
* Continuous

_Categorical variables_ include things like yes or no; and north, south, east, or west. An important distinction in our machine learning work in this book is whether the categorical variable has only two categories or has more than two categories. If it has only two categories, it is called a _binary target variable_. If it has more than two categories, it is called a _multiclass target variable_. You will set different parameters in your machine learning applications, depending on whether the variable is binary or multiclass. This will be covered in more detail later in the book.

_Continuous variables_ are numbers. For example, if your machine learning application predicts house prices based on features such as neighborhood, number of rooms, distance from schools, and so on, your target variable (the predicted price of the house) is a continuous variable. The price of a house could be any value from tens of thousands of dollars to tens of millions of dollars.

**1.4.2. Features**

In this book, features are perhaps the most important machine learning concept to understand. We use features all the time in our own decision making. In fact, the things you’ll learn in this book about features can help you better understand your own decision-making process.

As an example, let’s return to Karen as she makes a decision about whether to send a purchase order to IT for approval. The things that Karen considers when making this decision are its _features_. One thing Karen can consider when she comes across a product she hasn’t seen before is who manufactured the product. If a product is from a manufacturer that only produces IT products, then, even though she has never seen that product before, she considers it likely to be an IT product.

Other types of features might be harder for a human to consider but are easier for a machine learning application to incorporate into its decision making. For example, you might want to find out which customers are likely to be more receptive to receiving a sales call from your sales team. One feature that can be important for your repeat customers is whether the sales call would fit in with their regular buying schedule. For example, if the customer normally makes a purchase every two months, is it approximately two months since their last purchase? Using machine learning to assist your decision making allows these kinds of patterns to be incorporated into the decision to call or not call; whereas, it would be difficult for a human to identify such patterns.

Note that there can be several levels to the things (features) Karen considers when making her decision. For example, if she doesn’t know whether a product is a technical product or not, then she might consider other information such as who the manufacturer is and what other products are included on the requisition. One of the great things about machine learning is that you don’t need to know all the features; you’ll see which features are the most important as you put together the machine learning system. If you think it might be relevant, include it in your dataset.

#### 1.5. How does a machine learn? <a href="#ch01lev1sec5__title" id="ch01lev1sec5__title"></a>

A machine learns the same way you do. It is trained. But how? Machine learning is a process of rewarding a mathematical function for getting answers right and punishing the function for getting answers wrong. But what does it mean to reward or punish a function?

You can think of a _function_ as a set of directions on how to get from one place to another. In [figure 1.4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_010.html#ch01fig04), to get from point A to point B, the directions might read thus:

1. Go right.
2. Go a bit up.
3. Go a bit down.
4. Go down sharply.
5. Go up!
6. Go right.

**Figure 1.4. Machine learning function to identify a pattern in the data**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch01fig04\_alt.jpg)

A machine learning application is a tool that can determine when the function gets it right (and tells the function to do more of that) or gets it wrong (and tells the function to do less of that). The function knows it got it right because it becomes more successful at predicting the target variable based on the features.

Let’s pull a dataset out of [figure 1.4](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_010.html#ch01fig04) to look at a bigger sample in [figure 1.5](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_010.html#ch01fig05). You can see that the dataset comprises two types of circles: dark circles and light circles. In [figure 1.5](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_010.html#ch01fig05), there is a pattern that we can see in the data. There are lots of light circles at the edges of the dataset and lots of dark circles near the middle. This means that our function, which provides the directions on how to separate the dark circles from light circles, will start at the left of the diagram and do a big loop around the dark circles before returning to its starting point.

**Figure 1.5. Machine learning functions to identify a group of similar items in a dataset**

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295836/files/ch01fig05\_alt.jpg)

When we are training the process to reward the function for getting it right, we could think of this as a process that rewards a function for having a dark circle on the right and punishes it for having a dark circle on the left. You could train it even faster if you also reward the function for having a light circle on the left and punish it for having a light circle on the right.

So, with this as a background, when you’re training a machine learning application, what you’re doing is showing a bunch of examples to a system that builds a mathematical function to separate certain things in the data. The thing it is separating in the data is the _target variable_. When the function separates more of the target variables, it gets a reward, and when it separates fewer target variables, it gets punished.

Machine learning problems can be broken down into two types:

* Supervised machine learning
* Unsupervised machine learning

In addition to features, the other important concept in machine learning as far as this book is concerned is the distinction between supervised and unsupervised machine learning.

Like its name suggests, _unsupervised_ machine learning is where we point a machine learning application at a bunch of data and tell it to do its thing. Clustering is an example of unsupervised machine learning. We provide the machine learning application with some customer data, for example, and it determines how to group that customer data into clusters of similar customers. In contrast, classification is an example of _supervised_ machine learning. For example, you could use your sales team’s historical success rate for calling customers as a way of training a machine learning application how to recognize customers who are most likely to be receptive to receiving a sales call.

**Note**

In most of the chapters in this book, you’ll focus on supervised machine learning where, instead of letting the machine learning application pick out the patterns, you provide the application with a historical dataset containing samples that show the right decision.

One of the big advantages of tackling business automation projects using machine learning is that you can usually get your hands on a good dataset fairly easy. In Karen’s case, she has thousands of previous orders to draw from, and for each order, she knows whether it was sent to a technical approver or not. In machine learning lingo, you say that the dataset is _labeled_, which means that each sample shows what the target variable should be for that sample. In Karen’s case, the historical dataset she needs is a dataset that shows what product was purchased, whether it was purchased by someone from the IT department or not, and whether Karen sent it to a technical approver or not.

#### 1.6. Getting approval in your company to use machine learning to make decisions <a href="#ch01lev1sec6__title" id="ch01lev1sec6__title"></a>

Earlier in the chapter, we described how you could learn enough about decision making using machine learning to help your company. But what does your company need in order to take full advantage of your good work? In theory, it’s not that hard. Your company just needs four things:

* It needs a person who can identify opportunities to automate and use machine learning, and someone who can put together a proof of concept that shows the opportunity is worth pursuing. That’s you, by the way.
* You need to be able to access the data required to feed your machine learning applications. Your company will likely require you to complete a number of internal forms describing why you want access to that data.
* Your risk and management teams need to be comfortable with using pattern-based approaches to making decisions.
* Your company needs a way to turn your work into an operational system.

In many organizations, the third of these four points is the most difficult. One way to tackle this is to involve your risk team in the process and provide them with the ability to set a threshold on when a decision needs to be reviewed by Karen.

For example, some orders that cross Karen’s desk very clearly need to be sent to a technical approver, and the machine learning application must be 100% confident that it should go to a technical approver. Other orders are less clear cut, and instead of returning a 1 (100% confidence), the application might return a 0.72 (a lower level of confidence). You could implement a rule that if the application has less than 75% confidence that the decision is correct, then route the request to Karen for a decision.

If your risk team is involved in setting the confidence level whereby orders must be reviewed by a human, this provides them with a way to establish clear guidelines for which pattern-based decisions can be managed in your company. In [chapter 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02), you’ll read more about Karen and will help her with her work.

#### 1.7. The tools <a href="#ch01lev1sec7__title" id="ch01lev1sec7__title"></a>

In the old days (a.k.a. 2017), setting up a scalable machine learning system was very challenging. In addition to identifying features and creating a labeled dataset, you needed to have a wide range of skills, encompassing those of an IT infrastructure administrator, a data scientist, and a back-end web developer. Here are the steps that _used_ to be involved in setting up your machine learning system. (In this book, you’ll see how to set up your machine learning systems without doing all these steps.)

1. Set up your development environment to build and run a machine learning application (IT infrastructure administrator)
2. Train the machine learning application on your data (data scientist)
3. Validate the machine learning application (data scientist)
4. Host the machine learning application (IT infrastructure administrator)
5. Set up an endpoint that takes your new data and returns a prediction (back-end web developer)

It’s little wonder that machine learning is not yet in common use in most companies! Fortunately, nowadays some of these steps can be carried out using cloud-based servers. So although you need to understand how it all fits together, you don’t need to know how to set up a development environment, build a server, or create secure endpoints.

In each of the following seven chapters, you’ll set up (from scratch) a machine learning system that solves a common business problem. This might sound daunting, but it’s not because you’ll use a service from Amazon called AWS SageMaker.

**1.7.1. What are AWS and SageMaker, and how can they help you?**

AWS is Amazon’s cloud service. It lets companies of all sizes set up servers and interact with services in the cloud rather than building their own data centers. AWS has dozens of services available to you. These range from compute services such as cloud-based servers (EC2), to messaging and integration services such as SNS (Simple Notification Service) messaging, to domain-specific machine learning services such as Amazon Transcribe (for converting voice to text) and AWS DeepLens (for machine learning from video feeds).

SageMaker is Amazon’s environment for building and deploying machine learning applications. Let’s look at the functionality it provides using the same five steps discussed earlier (section 1.7). SageMaker is revolutionary because it

* Serves as your development environment in the cloud so you don’t have to set up a development environment on your computer
* Uses a preconfigured machine learning application on your data
* Uses inbuilt tools to validate the results from your machine learning application
* Hosts your machine learning application
* Automatically sets up an endpoint that takes in new data and returns predictions

One of the best aspects of SageMaker, aside from the fact that it handles all of the infrastructure for you, is that the development environment it uses is a tool called the Jupyter Notebook, which uses Python as one of its programming languages. But the things you’ll learn in this book working with SageMaker will serve you well in whatever machine learning environment you work in. Jupyter notebooks are the de facto standard for data scientists when interacting with machine learning applications, and Python is the fastest growing programming language for data scientists.

Amazon’s decision to use Jupyter notebooks and Python to interact with machine learning applications benefits both experienced practitioners as well as people new to data science and machine learning. It’s good for experienced machine learning practitioners because it enables them to be immediately productive in SageMaker, and it’s good for new practitioners because the skills you learn using SageMaker are applicable everywhere in the fields of machine learning and data science.

**1.7.2. What is a Jupyter notebook?**

Jupyter notebooks are one of the most popular tools for data science. These combine text, code, and charts in a single document that allows a user to consistently repeat data analysis, from loading and preparing the data to analyzing and displaying the results.

The Jupyter Project started in 2014. In 2017, the Jupyter Project steering committee members were awarded the prestigious ACM Software System award “for developing a software system that has had a lasting influence, reflected in contributions to concepts, in commercial acceptance, or both.” This award is a big deal because previous awards were for things like the internet.

In our view, Jupyter notebooks will become nearly as ubiquitous as Excel for business analysis. In fact, one of the main reasons we selected SageMaker as our tool of choice for this book is because when you’re learning SageMaker, you’re learning Jupyter.

#### 1.8. Setting up SageMaker in preparation for tackling the scenarios in- n chapters 2 through 7 <a href="#ch01lev1sec8__title" id="ch01lev1sec8__title"></a>

The workflow that you’ll follow in each chapter is as follows:

1. Download the prepared Jupyter notebook and dataset from the links listed in the chapter. Each chapter has one Jupyter notebook and one or more datasets.
2. Upload the dataset to S3, your AWS file storage bucket.
3. Upload the Jupyter notebook to SageMaker.

At this point, you can run the entire notebook, and your machine learning model will be built. The remainder of each chapter takes you through each cell in the notebook and explains how it works.

If you already have an AWS account, you are ready to go. Setting up SageMaker for each chapter should only take a few minutes. Appendixes B and C show you how to do the setup for [chapter 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02).

If you don’t have an AWS account, start with [appendix A](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_021.html#app01) and progress through to [appendix C](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_028.html#app03). These appendixes will step you through signing up for AWS, setting up and uploading your data to the S3 bucket, and creating your notebook in SageMaker. The topics are as follows:

* [Appendix A](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_021.html#app01): How to sign up for AWS
* [Appendix B](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_024.html#app02): How to set up S3 to store files
* [Appendix C](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_028.html#app03): How to set up and run SageMaker

After working your way through these appendixes (to the end of [appendix C](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_028.html#app03)), you’ll have your dataset stored in S3 and a Jupyter notebook set up and running on SageMaker. Now you’re ready to tackle the scenarios in [chapter 2](https://learning.oreilly.com/library/view/machine-learning-for/9781617295836/kindle\_split\_012.html#ch02) and beyond.

#### 1.9. The time to act is now <a href="#ch01lev1sec9__title" id="ch01lev1sec9__title"></a>

You saw earlier in this chapter that there is a group of frontier firms that are rapidly increasing their productivity. Right now these firms are few and far between, and your company might not be competing with any of them. However, it’s inevitable that other firms will learn to use techniques like machine learning for business automation to dramatically improve their productivity, and it’s inevitable that your company will eventually compete with them. We believe it is a case of eat or be eaten.

The next section of the book consists of six chapters that take you through six scenarios that will equip you for tackling many of the scenarios you might face in your own company, including the following:

* Should you send a purchase order to a technical approver?
* Should you call a customer because they are at risk of churning?
* Should a customer support ticket be handled by a senior support person?
* Should you query an invoice sent to you by a supplier?
* How much power will your company use next month based on historical trends?
* Should you add additional data such as planned holidays and weather forecasts to your power consumption prediction to improve your company’s monthly power usage forecast?

After working your way through these chapters, you should be equipped to tackle many of the machine learning decision-making scenarios you’ll face in your work and in your company. This book takes you on the journey from being a technically minded non-developer to someone who can set up a machine learning application within your own company.

#### Summary <a href="#ch01lev1sec10__title" id="ch01lev1sec10__title"></a>

* Companies that don’t become more productive will be left behind by those that do.
* Machine learning is the key to your company becoming more productive because it automates all of the little decisions that hold your company back.
* Machine learning is simply a way of creating a mathematical function that best fits previous decisions and that can be used to guide current decisions.
* Amazon SageMaker is a service that lets you set up a machine learning application that you can use in your business.
* Jupyter Notebook is one of the most popular tools for data science and machine learning.
