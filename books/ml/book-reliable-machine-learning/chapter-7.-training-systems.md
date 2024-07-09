# Chapter 7. Training Systems

_ML training_ is the process by which we transform input data into models. We take a set of input data, almost always preprocessed and stored in an efficient way, and process it through a set of ML algorithms. The output is a representation of that data, called a _model_, that we can integrate into other applications. For more details on what a model is, see [Chapter 3](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch03.html#basic\_introduction\_to\_models).

A _training algorithm_ describes the specific steps by which software reads data and updates a model to try to represent that data. A _training system_, on the other hand, describes the entire set of software surrounding that algorithm. The simplest implementation of an ML training system is on a single computer running in a single process that reads data, performs some cleaning and imposes some consistency on that data, applies an ML algorithm to it, and creates a representation of the data in a model with new values as a result of what it learns from the data. Training on a single computer is by far the simplest way to build a model, and the large cloud providers do rent powerful configurations of individual machines. Note, though, that many interesting uses of ML in production process a significant amount of data and as a result might benefit from significantly more than one computer. Distributing processing brings scale but also complexity.

In part, because of our broad conception of what an ML training system is, ML training systems may have less in common with one another across different organizations and model builders than any other part of the end-to-end ML system. In [Chapter 8](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch08.html#serving-id0000021), you will see that even across different use cases, many of the basic requirements of a serving system are broadly similar: they take a representation of the model, load it into RAM, and answer queries about the contents of that model sent from an application. In serving systems, sometimes that serving is for very small models (on phones, for example). Sometimes it is for huge models that don’t even all fit on a single computer. But the structure of the problem is similar.

In contrast, training systems do not even necessarily live in the same part of our ML lifecycle (see [Figure 1-1](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch01.html#ml\_lifecycle)). Some training systems are closest to the input data, performing their function almost completely offline from the serving system. Other training systems are embedded in the serving platform and are tightly integrated with the serving function. Additional differences appear when we look at the way that training systems maintain and represent the state of the model. Because of this significant variety of differences across legitimate and well-structured ML training systems, it is not reasonable to cover all of the ways that organizations train models.

Instead, this chapter covers a somewhat idealized version of a simple, distributed ML training system. We’ll describe a system that lives in a distinct part of the ML loop, next to the data and producing artifacts bound for the model quality evaluation system and serving system. Although most ML training systems that you will encounter in the real world will have significant differences from this architecture, separating it out will allow us to focus on the particularities of training itself. We will describe the required elements for a functional and maintainable training system and will also describe how to evaluate the costs and benefits of additional desirable characteristics.

## Requirements

A training system requires the following elements, although they might appear in a different order or combined with one another:

Data to train onThis includes human labels and annotations if we have them. This data should be preprocessed and standardized by the time we use it. It will usually be stored in a format that is optimized for efficient access during training. Note that “efficient access during training” might mean different things depending on our model. The data should also be stored in an access-protected and policy-enforcing environment.Model configuration systemMany training systems have a means of representing the configuration of an individual model separate from the configuration of the training system as a whole.[1](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#ch01fn70) These should store model configurations in a versioned system with some metadata about the teams creating the models and the data used by the models. This will come in extremely handy later.Model-training framework

Most model creators will not be writing model-training frameworks by hand. It seems likely that most ML engineers and modelers will eventually be exclusively using a training systems framework and customizing it as necessary. These frameworks generally come with the following:

Orchestration

Different parts of the system need to run at different times and need to be informed about one another. We call this _orchestration_. Some of the systems that do this include the following two elements as well, but these functions can be assembled separately, so they are broken out here.

Job/work scheduling

Sometimes part of orchestration, job scheduling refers to actually starting the binaries on the computers and keeping track of them.

Training or model development software

This software handles the ordinary boilerplate tasks usually associated with building an ML model. Common examples right now include TensorFlow, PyTorch, and many others. Disagreements rivaling religious wars start over which of these is best, but all of them accomplish the same job of helping model developers build models more quickly and more consistently.

Model quality evaluation system

Some engineers don’t think of this as part of the training system, but it has to be. The process of model building is iterative and exploratory. Model builders try out ideas and discard most of them. The model quality evaluation system provides rapid and consistent feedback on the performance of models and allows model builders to make decisions quickly.

**WARNING**

This is the most commonly skipped portion of a training system but really is mandatory.

If we do not have a model quality evaluation system, each of our model developers will build a more ad hoc and less reliable one for themselves and will do so at a higher cost to the organization. This topic is covered much more extensively in [Chapter 5](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch05.html#evaluating\_model\_validity\_and\_quality).

Syncing models to servingThe last thing we do with a model is send it to the next stage, usually the serving system but possibly another kind of analysis system.

If we have a system that provides for these basic requirements, we will be able to offer a minimally productive technical environment to model developers. In addition to these basic elements, though, we will want to add infrastructure specifically geared toward reliability and manageability. Among these elements, we should include some careful thoughts about monitoring this multistage pipeline, metadata about team ownership of features and models, and a full-fledged feature storage system.

## Basic Training System Implementation

[Figure 7-1](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#basic\_ml\_training\_system\_architecture) depicts a proposed architecture for a simple but relatively complete and manageable ML training system.[2](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#ch01fn71)

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098106218/files/assets/reml_0701.png" alt="Basic ML training system architecture" height="335" width="600"><figcaption></figcaption></figure>

**Figure 7-1. Basic ML training system architecture**

In this simplified training system, the data flows in from the left, and models emerge on the right. In between we clean up, transform, and read the data. We use an ML framework that applies a training algorithm to turn the data into a model. We evaluate the model that we just produced. Is it well formed? Is it useful? Finally, we copy a servable version of that model into our serving system so we can integrate it into our application. All the while, we keep track of our models and data in a metadata system, we make sure the pipeline continues to work, and we monitor the whole thing. Next, we’ll go into detail about the roles of each of these components.

### Features

_Training data_ is data about events or facts in the world that we think will be relevant to our model. A _feature_ is a specific, measurable aspect of that data. Specifically, features are those aspects of the data that we believe are most likely to be useful in modeling, categorizing, and predicting future events given similar circumstances. To be useful, features need a consistent definition, consistent normalization, and consistent semantics across the whole ML system, including the feature store, training, quality evaluation, and serving. For significantly more detail, see [Chapter 4](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch04.html#feature\_and\_training\_data).

If we think of a feature for YarnIt purchase data like “purchase price,” we can easily understand how this can go badly if we’re not careful. First of all, we probably need to standardize purchases in currency, or at least not mix currencies in our model. So let’s say we convert everything to US dollars. We need to guarantee that we do that conversion based on the exchange rate at a particular point in time—say, the closing rate at the end of the trading day in London for the day that we are viewing data. We then have to store the conversion values used in case we ever need to reconvert the raw data. We probably should normalize the data, or put it into larger buckets or categories. We might have everything under $1 in the 0th bucket, and $1–$5 in the next bucket, and so on in $5 increments. This makes certain kinds of training more efficient. It also means that we need to ensure we have standard normalization between training and serving and that if we ever change normalization, we update it everywhere carefully in sync.

Features and feature development are a critical part of how we experiment when we are making a model. We are trying to figure out which aspects of the data matter to our task and which are not relevant. In other words, we want to know which features make our model better. As we develop the model, we need easy ways to add new features, produce new features on old data when we get a new idea for what to look for in existing logs, and remove features that turned out to not be important. Features can be complicated.

### Feature Store

We need to store the features and, not surprisingly, the most common name for the system where we do that is the _feature store_. The characteristics of a feature store will exist, even if our model training system reads raw data and extracts features each time. Most people will find it convenient, and an especially important reliability bonus, to store extracted features in a dedicated system of some kind. This topic is covered extensively in [Chapter 4](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch04.html#feature\_and\_training\_data).

One common data architecture for this is a bunch of files in a directory (or a bunch of buckets in an object storage system). This is obviously not the most sophisticated data storage environment but has a huge advantage, giving us a fast way to start training, and it appears to facilitate experimentation with new features. Longer term, though, this unstructured approach has two huge disadvantages.

First, it is extremely difficult to ensure that the system as a whole is consistently functioning correctly. Systems like this with unstructured feature-engineering environments frequently suffer from training-serving feature skew (whereby features are defined differently in the training and serving environments) as well as problems related to inconsistent feature semantics over time, even in the training system alone.

The second problem is that unstructured feature-engineering environments can actually hinder collaboration and innovation. They make it more difficult to understand the provenance of features in a model and more difficult to understand who added them, when, and why.[3](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#ch01fn72) In a collaborative environment, most new model engineers will benefit enormously from understanding the work of their predecessors. This is made easier by being able to trace backward from a model that works well, to the model definition, and ultimately to the features that are used. A feature store gives a consistent place to understand the definition and authorship of features and can significantly improve innovation in model development.

### Model Management System

A _model management system_ can provide at least three sets of functionality:

Metadata about modelsConfiguration, hyperparameters, and developer authorshipSnapshots of trained modelsUseful for bootstrapping new variants of the same model more efficiently by using transfer learning, and tremendously useful for disaster recovery when we accidentally delete a modelMetadata about featuresAuthorship and usage of each feature by specific models

While these functions are theoretically separable, and are often separate in the software offerings, together they form the system that allows engineers to understand the models in production, how they are built, who built them, and what features they are built upon.

Just as with feature stores, everyone has a rudimentary form of a model management system, but if it amounts to “whatever configuration files and scripts are in the lead model developer’s home directory,” it may be appropriate to check whether that level of flexibility is still appropriate for the needs of the organization. There are reasonable ways to get started with model management systems so that the burden is low. It does not need to be complicated but can be a key source of information, tying serving all the way back through training to storage. Without this data, it’s not always possible to figure out what is going wrong in production.

### Orchestration

_Orchestration_ is the process by which we coordinate and keep track of all the other parts of the training system. This typically includes scheduling and configuring the various jobs associated with training the model as well and tracking when training is complete. Orchestration is often provided by a system that is tightly coupled with our ML framework and job/process scheduling system, but does not have to be.

For orchestration here, think of Apache Airflow as an example. Many systems are technically workflow orchestration systems but are focused on building data analytics pipelines (such as Apache Beam or Spark, or Google Cloud Dataflow). These typically come with significant assumptions about the structure of your tasks, have additional integrations, and have many restrictions built in. Note that Kubernetes is not a pipeline orchestration system: Kubernetes has a means of orchestrating containers and tasks that run in them, but generally does not by itself provide the kinds of semantics that help us specify how data moves through a pipeline.

#### Job/process/resource scheduling system

Everyone who runs ML training pipelines in a distributed environment will have a way of starting processes, keeping track of them, and noticing when they finish or stop. Some people are fortunate enough to work at an organization that provides centralized services, either locally or on a cloud provider, for scheduling jobs and tasks. Otherwise, it is best to use one of the popular compute resource management systems, either open source or commercial.

Examples of resource scheduling and management systems include software such as the previously mentioned Kubernetes, although it also includes many other features such as setting up networking among containers and handling requests to and from containers. More generally and more traditionally, Docker could be regarded as a resource scheduling system by providing a means of configuring and distributing virtual machine (VM) images to VMs.

#### ML framework

The _ML framework_ is where the algorithmic action is. The point of ML training is to transform the input data into a representation of that data, called a _model_. The ML framework we use will provide an API to build the model that we need and will take care of all of the boilerplate code to read the features and convert them into the data structures appropriate for the model. ML frameworks are typically fairly low level and, although they are much discussed and debated, are ultimately quite a small part of the overall ML loop in an organization.

### Quality Evaluation

The ML model development process can be thought of as continuous partial failure followed by modest success. It is essentially unheard of for the first model that anyone tries to be the best, or even a reasonably adequate, solution to a particular ML problem. It necessarily follows, therefore, that one of the essential elements of a model-training environment is a systematic way to evaluate the model that we just trained.

At some level, model quality evaluation has to be extremely specific to the purposes of a particular model. Vision models correctly categorize pictures. Language models interpret and predict text. At the most basic level, a model quality evaluation system offers a means of performing a quality evaluation, usually authored by the model developer, and storing the results in a way that they can be compared to previous versions of the same model. The operational role of such a system is ultimately to be sufficiently reliable that it can be an automatic gate to prevent “bad” models from being sent to our production serving environment.

Evaluation starts with factors as simple as verifying that we are loading the right version of the right model and ensuring that the model loads in our model server. Evaluation also must include performance aspects to make sure that the model can be served in the memory and computational resources that we have available. But also, we have to care about how the model performs on requests that we believe to be representative of the requests we will get. For significantly more detail on this topic, see [Chapter 5](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch05.html#evaluating\_model\_validity\_and\_quality).

### Monitoring

Monitoring distributed data processing pipelines is difficult. The kinds of straightforward things that a production engineer might care about, such as whether the pipeline is making sufficiently fast progress working through the data, are quite difficult to produce accurately and meaningfully in a distributed system. Looking at the oldest unprocessed data might not be meaningful because there could be a single old bit of data that’s stuck in a system that is otherwise done processing everything. Likewise, looking at the data processing rate might not be useful by itself if some kinds of data are markedly more expensive to process than others.

This book has an entire monitoring chapter ([Chapter 9](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch09.html#monitoring\_and\_observability\_for\_models)). Harder questions will be tackled there. For this section, the single most important metric to track and alert on is training system throughput. If we have a meaningful long-term trend of how quickly our training system is able to process training data under a variety of conditions, we should be able to set thresholds to alert us when things are not going well.

## General Reliability Principles

Given this simple, but workable, overall architecture, let’s look at how this training system should work. If we keep several general principles in mind during the construction and operationalization of the system, things will generally go much better.

### Most Failures Will Not Be ML Failures

ML training systems are complex data processing pipelines that happen to be tremendously sensitive to the data that they are processing. They are also most commonly distributed across many computers, although in the simplest case they might be on a single computer. This is not a base state likely to lead to long-term reliability, and production engineers generally look at this data sensitivity for the most common failures. However, when experienced practitioners look at the experienced failures in ML systems over time, they find that most of the failures are not ML specific.[4](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#ch01fn73) They are software and systems failures that occur commonly in this kind of distributed system. These failures often have impact and detection challenges that are ML specific, but the underlying causes are most commonly not ML specific.

Amusing examples of these failures include such straightforward things as “the training system lost permission to read the data so the model trained on nothing,” and “the version that we copied to serving wasn’t the version we thought it was.” Most of them are of the form of incorrectly monitored and managed data pipelines. For many more examples, see Chapters [11](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch11.html#incident\_response) and [15](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch15.html#case\_studies\_mlops\_in\_practice).

To make ML training systems reliable, look to systems and software errors and mitigate those first. Look at software versioning and deployment, permissions and data access requirements, data updating policies and data organization, replication systems, and verification. Essentially, do all of the work to make a general distributed system reliable before beginning any ML-specific work.

### Models Will Be Retrained

Perhaps this section should be titled something even stronger: models must be _retrained_. Some model developers will train a model from a dataset once, check the results, deploy the model into production, and claim to be done. They will note that if the dataset isn’t changing and the purpose of the model is achieved, the model is good enough and there is no good reason to ever train it again.

Do not believe this. Eventually, whether in a day or a year, that model developer or their successor will get a new idea and want to train a different version of the same model. Perhaps a better dataset covering similar cases will be identified or created, and then the model developers will want to train on that one. Perhaps just for disaster-recovery reasons you’d like to prove that if you delete every copy of the model by mistake, you can re-create it. You might simply want to verify that the toolchain for training and validation is intact.

For all of these reasons, assume every model will be retrained and plan accordingly—store configs and version them, store snapshots, and keep versioned data and metadata. This approach has tremendous value: most of the debates about so-called “offline” and “online” models are actually debates about retraining models in the presence of new data. By creating a hard production requirement that models can be retrained, the technical environment is much of the way to facilitating periodic retraining of all models (including rapid retraining).[5](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#ch01fn74)

**TIP**

Assume every model will be retrained.

### Models Will Have Multiple Versions (at the Same Time!)

Models are almost always developed in cohorts, most obviously because we will want to train different versions with different hyperparameters. One common approach is to have a named model with multiple minor changes to it. Sometimes those changes arrive in a rapid cluster at the beginning, and other times they arrive over time. But just as models will be retrained, it is true that they will be changed and developed over time. In many environments, we will want to serve two or more versions of the same model at the same time in order to determine how the different versions of the model work for different conditions (for those familiar with traditional web development for user experience, this is essentially A/B testing for models).

Hosting simultaneous versions of the same model requires specific infrastructure. We need to use our model management infrastructure to keep track of model metadata (including things like the model family, model name, model version, and model creation date). We also need to have a system to route a subset of lookups to one version of the model versus another version.

### Good Models Will Become Bad

We should assume that the models we produce will be hard to reproduce and will have subtle and large reliability problems in the future. Even when a model works well when we launch it, we have to assume that either the model or the world might change in some difficult-to-predict way that causes us enormous trouble in future years. Make backup plans.

The very first backup plan is to make a non-ML (or at least “simpler-ML”) fallback path or “fail-safe” implementation for our model. This is going to be a heuristic or algorithm or default that ensures that at least some basic functionality is provided by our application when the ML model is unable to provide sophisticated predictions, categorizations, and insights. Common algorithms that accomplish this goal are simplistic and extremely general but at least slightly better than nothing. One example we’ve mentioned earlier is that for recommendations on the _yarnit.ai_ storefront, we might simply default to showing popular items when we don’t have a customized recommendation model available.

This approach has a tremendous problem, however: it limits how good you can let your ML models be. If the models become so much better than the heuristics or defaults, you will come to depend upon them so thoroughly that no backup will be sufficient to accomplish the same goals. Depending on defaults and heuristics is a completely appropriate path for most organizations that are early in the ML adoption lifecycle. But it is a dependency that you should wean yourself off of if you’d like to actually take advantage of ML in your organization.

The other backup plan is to keep multiple versions of the same model and plan to revert to an older one if you need to. This will cover cases where a newer version of the model is significantly worse for some reason, but it will not help when the world as a whole has changed and therefore all versions of this model are not very good.

Ultimately, the second backup plan, combined with the ability to serve multiple models at the same time and quickly develop new variations of existing models, provides a path to understanding and resolving future model quality problems when the world has changed in a way that makes the model perform poorly. It is important for production traditionalists to note that no fixed or defensible barrier exists between model development and production in this case. Model quality is both a production engineering problem and a model development problem (which might occur urgently in production).

### Data Will Be Unavailable

Some of the data used to train new models will not be available when we try to read it. Data storage systems, especially distributed data storage systems, have failure modes that include small amounts of actual data loss, but much higher amounts of data unavailability. This is a problem worth thinking through in advance of its occurrence because it will definitely occur.

Most ML training datasets are already sampled from another, larger dataset, or simply a subset of all possible data by virtue of when or how we started collecting the data in the first place. For example, if we train on a dataset of customer purchase behavior at _yarnit.ai_, this dataset is already incomplete from the start, in at least two obvious ways.

First, there was some date before which we were not collecting this data (or some date before which we choose not to use the data for whatever reason). Second, this is really only customer purchase behavior on our site and does not include any customer purchase behavior of similar products on any other sites. This is unsurprising because our competitors don’t share data with us, but it does mean that we’re already seeing only a subset of what is almost certainly relevant training data. For very high-volume systems (web-browsing logs, for example), many organizations subsample this data before training automatically as well, simply to reduce the cost of data processing and model training.

Given that our training data is already subsampled, probably in several ways, when we lose data, we should answer this question: is the loss of data biased in some way? If we were to drop one out of every 1,000 training records in a completely random way, this is almost certainly safe to ignore for the model. On the other hand, losing all of the data from people in Spain, or from people who shop in the mornings, or from the day before a large knitting conference—these are not ignorable. They are likely to create new biases in the data that we train on versus the data that we do not.[6](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#ch01fn75)

Some training systems will try to pre-solve the problem of missing data for the entire system in advance of it occurring. This will work only if all of your models have a similar set of constraints and goals. This is because the impact of missing data is something that matters to each model and can be understood only in the context of its impact on each model.

Missing data can also have security properties that are worth considering. Some systems, especially those designed to prevent fraud and abuse, will be under constant observation and attack from outside malicious parties. Attacks on these systems consist of trying different kinds of behaviors to determine the response of the system and taking advantage of gaps or weaknesses that appear. In these cases, training system reliability teams need to be certain that there is no way for an outside party to systematically bias which particular time periods are skipped during training. It is not at all unheard of for attackers to find ways to, for example, create very large volumes of duplicate transactions for short periods of time in order to overwhelm data processing systems and try to hit a high-rate discard heuristic in the system. This is the kind of scenario that everyone working on data loss scenarios needs to consider in advance.

### Models Should Be Improvable

Models will change over time, not just by adding new data. They will also see larger, structural changes. The requirements to change come from several directions. Sometimes we will add a feature to the application or implementation that provides new data but also requires new features of the model. Sometimes our user behavior will change substantially enough that we need to alter the model to accommodate. Procedurally, the most challenging change to model training in the training system we’re describing here is adding a completely new feature.

### Features Will Be Added and Changed

Most production ML training systems will have some form of a feature store to organize the ML training data. (Feature stores offer many advantages and are discussed in more detail in [Chapter 4](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch04.html#feature\_and\_training\_data).) From the perspective of a training system, what we need to note is that a significant part of model development over time is often adding new features to the model. This happens when a model developer has a new idea about some data that might, in combination with our existing data, usefully improve the model.

Adding features will require a change to the feature store schema that is implementation specific, but it might also benefit from a process to “backfill” the feature store by reprocessing raw logs or data from the past to add the new feature for prior examples. For instance, if we decide that the local weather in the city that we believe our customers to be shopping from is a salient way to predict what they might buy, we’ll have to add `customer_temperature` and `customer_precipitation` columns to the feature store.[7](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#ch01fn76) We might also reprocess browsing and purchasing data for the last year to add these two columns in the past so that we can validate our assumption that this is an important signal. Adding new columns to the feature store and changing to schema and content of data in the past are both activities that can significantly impact reliability of all our models in the training system if the changes are not managed and coordinated carefully.

### Models Can Train Too Fast

ML production engineers are sometimes surprised to learn that in some learning systems, models can train too fast.[8](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#ch01fn77) This can depend a bit on the exact ML algorithm, model structure, and parallelism of the system in which it’s being implemented. But it is entirely possible to have a model that, when trained too quickly, produces garbage results, but when trained more slowly, produces much more accurate results.

Here is one way this can happen: there is a distributed representation of the state of the model used by a distributed set of learning processes. The learning processes read new data, consult the state of the model, and then update the state of part of the model to reflect the piece of data they just read.[9](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#ch01fn78) As long as there are either locks (slow!) or no updates to the particular key we are updating (unlikely at scale) in the middle of that process, everything is fine.

The problem is that multiple race conditions can exist, where two or more learner tasks consult the model, read some data, and queue and update to the model at the same time. One really common occurrence then is that the updates can stack on top of each other, and that portion of the model can move too far in a certain direction. The next time a bunch of learner tasks consult the model, they find it skewed in one direction by a lot, compared to the data that they are reading, so they queue up changes to move it substantially in the other direction. Over time, this part of the model (and other parts of the model) can diverge from the correct value rather than converge.

**WARNING**

For distributed training setups, multiple race conditions is an extremely common source of failure.

Unfortunately for the discipline of ML production engineering, there is no simple way to determine when a model is being trained “too fast.” There’s a real-world test that is inconvenient and frustrating: if you train the same model faster and slower (typically, with more and fewer learner tasks), and the slower model is “better” in some set of quality metrics, then you might be training too fast.

The main approaches to mitigating this problem are to structurally limit the number of updates “in flight” by making the state of the model as seen by any learning processes closely synchronized with the state of the model as stored. This can be done by storing the current state of the model in very fast storage (RAM) and by limiting the rate at which multiple processes update the model. It is possible, of course, to use locking data structures for each key or each part of the model, but the performance penalties imposed by these are usually too high to seriously consider.

### Resource Utilization Matters

This should be stated simply and clearly: ML training and serving is computationally expensive. One basic reason that we care about resource efficiency for ML training is that without an efficient implementation, ML may not make business sense. Consider that an ML model offers some business or organizational value proportional to the value that it provides, divided by the cost to create the model. While the biggest costs at the beginning are people and opportunity costs, as we collect more data, train more models, and use them more, computer infrastructure costs will grow to an increasingly large share of our expenditure. So it makes sense to pay attention to it early.

Specifically and concretely, utilization describes the following:

portion of compute resources usedtotal compute resources paid for

This is the converse of wastefulness and measures how well we’re using the resources we pay for. In an increasingly cloud world, this is an important metric to track early.

Resource utilization is also a reliability issue. The more headroom we have to retrain models compared to the resources we have available, the more resilient the overall system will be. This is because we will be able to recover from outages more quickly. This is true if we’re training the model on a single computer or on a huge cluster. Furthermore, utilization is also an innovation issue. The cheaper models are to train, the more ideas we will be able to explore in a given amount of time and budget. This markedly increases the likelihood that we will find a good idea among all of the bad ones. It is easy to lose track of this down here in the details, but we are not really here to train models—we’re here to make some kind of difference in our organization and for our customers.

So it’s clear we care about efficient use of our resources. Here are some simple ways we can make ML training systems work well in this respect:

Process data in batchesWhen possible (algorithm-dependent), train on chunks of data at the same time.Rebuild existing models

Early-stage ML training systems often rebuild models from scratch on all of the data when new data arrives. This is simpler from a configuration and software perspective but ultimately can become enormously inefficient. This idea of incrementally updating models has a few other variants:

* Use transfer learning to build a model by starting with an existing model.[10](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#ch01fn79)
* Use a multimodel architecture with a long-term model that is large but seldom retrained, and a smaller, short-term model that is cheaply updated frequently.
* Use online learning, whereby the model is incrementally updated as each new data point arrives.

Simple steps such as these can significantly impact an organization’s computational costs for training and retraining models.

### Utilization != Efficiency

To know whether our ML efforts are useful, we have to measure the value of the process rather than the CPU cycles spent to deliver it. Efficiency measures the following:

value producedcost

Cost can be calculated two ways, each of which provides a different view of our efforts. _Money-indexed cost_ is the dollars we spend on resources. Here we just calculate the total amount of money spent on training models. This has the advantage of being a very real figure for most organizations. It has the disadvantage that changes in pricing of resources can make it hard to see changes that are due to modeling and system work versus exogenous changes from our resource providers. For example, if our cloud provider starts charging much more for a GPU that we currently use, our efficiency will go down through no change we made. This is important to know, of course, but it doesn’t help us build a more efficient training system. In other words, money cost is the most important, long-term measure of efficiency but is ironically not always the best way to identify projects to improve efficiency.

Conversely, _resource-indexed cost_ is measured in dollar-constant terms. One way to do this is to identify the most expensive and most constrained resource and use that as the sole element of the resource-indexed cost. For example, we might measure cost as _CPU seconds_ or _GPU seconds_. This has the advantage that when we make our training system more efficient, we will be able to see it immediately, regardless of current pricing details.

This raises the difficult question of what, exactly, is the value of our ML efforts. Again, there are two kinds of value we might measure: _per model_ and _overall_. At the per model level of granularity, _value_ is less grandiose than we might expect. We don’t need to measure the actual business impact of every single trained model. Instead, for simplicity, let’s assume that our training system is worthwhile. In that case, we need a metric that helps us compare the value being created across different implementations training the same model. One that works well is something like

number of features trained

or even

number of examples processed

or possibly

number of experimental models trained

So for a per model, resource-indexed, cost-efficiency metric, we might have this:

millions of examplesGPU second

This will help us easily see efforts to make reading and training more efficient without requiring us to know anything at all about what the model actually does.

Conversely, _overall_ value attempts to measure value across the entire program of ML model training, considering how much it costs us to add value to the organization as a whole. This will include the cost of the staff, the test models, and the production model training and serving. It should also attempt to measure the overall value of the model to our organization. Are our customers happier? Are we making more money? Overall efficiency of the ML training system is measured at a whole-program or whole-group basis and is measured over months rather than seconds.

Organizations that do not have a notion of efficiency will ultimately misallocate time and effort. It is far better to have a somewhat inaccurate measure that can be improved than to not measure efficiency at all.

### Outages Include Recovery

This is somewhat obvious but still worth stating clearly: ML outages include the time it takes to recover from the outage. This has a huge and direct implication for monitoring, service levels, and incident response. For example, if a system can tolerate a 24-hour outage of your training system, but it takes you 18 hours to detect any problems and 12 hours to train a new model after the problems are detected, we cannot reliably stay within the 24 hours. Many people modeling production-engineering response to training-system outages utterly neglect to include model recovery time.

## Common Training Reliability Problems

Now that you understand the basic architecture of a training system and have looked at general reliability principles about training systems, let’s look at some specific scenarios where training systems fail. This section covers three of the most common reliability problems for ML training systems: data sensitivity, reproducibility, and capacity shortfalls. For each, we will describe the failure and then give a concrete example of how that might occur in the context of YarnIt, our fictional online knitting and crochet supply store.

### Data Sensitivity

As has been repeatedly mentioned, ML training systems can be extremely sensitive to small changes in the input data and to changes in the distribution of that data. Specifically, we can have the same volume of training data but have significant gaps in the way that the data covers various subsets of the data. Think about a model that is trying to predict things about worldwide purchases but has data from only US and Canadian transactions. Or consider an image-categorization algorithm that has no pictures of cats but many pictures of dogs. In each of these scenarios, the model will have a biased view of reality by virtue of training on only a biased set of data. These gaps in training data coverage can be present from the very beginning or can occur over time as we experience gaps or shifts in the training data.

Lack of representativeness in the input data is one common source of bias in ML models; here, we are using _bias_ in both the technical sense of the difference between the predicted value and the correct value in a model, but also in the social sense of being prejudiced against or damaging for a population in society. Strange distributions in the data can also cause a wide variety of other much more mundane problems. For some subtle and interesting cases, see [Chapter 11](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch11.html#incident\_response), but for now let’s consider a straightforward data sensitivity problem at YarnIt.

### Example Data Problem at YarnIt

YarnIt uses an ML model to rank the results of end-user searches. Customers come to the website and type some words for a product they are looking for. We generate a simple and broad list of candidate products that might match that search and then rank them with an ML model designed to predict how likely each product is to be useful to the user who is doing this query right now.

The model will have features like “words in the product name,” “product type,” “price,” “country of origin of the query,” and “price sensitivity of the user.” These will help us rank a set of candidate products for this user. And we retrain this model every day in order to ensure that we’re correctly ranking new products and adapting to changes in purchase patterns.

In one case, our pricing team at YarnIt creates a series of promotions to sell off overstocked products. The modeling team wants to capture the pre-discount price and the sale price separately, as these might be different signals to user purchase behavior. But because of the change in data formatting, they mistakenly exclude all discounted purchases from the training set after they add the discounted price to the dataset. Until they notice this, the model will train entirely on full-price purchases. From the perspective of the ML system, discounted items are simply no longer ever purchased by anyone, ever. As a result, the model will eventually stop recommending the discounted products, since there is no longer any evidence from our logging, data, and training system that anyone is ever purchasing them! This kind of very small error in data handling during training can lead to significant errors in the model.

### Reproducibility

ML training is often not strictly reproducible; it is almost impossible to use exactly the same binaries on exactly the same training data and produce exactly the same model, given the way most modern ML training frameworks work. Even more disconcerting, it may not even be possible to get approximately the same model. Note that while _reproducibility_ in academic ML refers to reproducing the results in a published paper, here we are referring to something more straightforward and more concerning: reproducing our own results from this same model on this same dataset.

ML reproducibility challenges come from several sources, some of them fixable and others not. It is important to address the solvable problems first. Here are some of the most common causes of model irreproducibility:

Model configuration, including hyperparametersSmall changes in the precise configuration of the model, especially in the hyperparameters chosen, can have big effects on the resulting model. The solution here is clear: use versioning for the model configurations, including the hyperparameters, and ensure that you’re using exactly the same values.Data differencesAs obvious as it may sound, most ML training feature storage systems are frequently updated, and it is difficult to guarantee that there are no changes at all to data between two runs of the same model. If you’re having reproducibility challenges, eliminating the possibility of differences in the training data is a critical step.Binary changesEven minor version updates to your ML training framework, learning binaries, or orchestration or scheduling system can result in changes to the resulting models. Hold these constant across training runs while you’re debugging reproducibility problems.[11](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#ch01fn80)

Aside from those fixable causes for irreproducibility, at least three causes are not easily fixed:

Random initializationsMany ML algorithms and most ML systems use random initializations, random shuffling, or random starting-point selection as a core part of the way they work. This can contribute to differences across training runs. In some cases, this difference can be mitigated by using the same random seed across runs.System parallelismIn a distributed system (or even a single-computer training system with multiple threads), jobs will be scheduled on lots of processors, and they will learn in a somewhat different order each time. There will be ordering effects depending on which keys are updated in what order. Without sacrificing the throughput and speed advantages of distributed computing, there’s no obvious way to avoid this. Note that some modern hardware accelerator architectures offer custom, high-speed interconnections among chips much faster than other networking technologies. NVIDIA’s NVLink or the interconnections among Google’s Cloud TPUs are examples of this. These interconnections reduce, but do not eliminate, the lag in propagating state among compute nodes.[12](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#ch01fn81)Data parallelismJust as the learning jobs are distributed, so is the data, assuming we have a lot of it. Most distributed data systems do not have strong ordering guarantees without imposing significant performance constraints. We have to assume that we will end up reading the training data in somewhat different order even if we do so from a limited number of training jobs.

Addressing these three causes is costly and challenging to the point of being almost impossible. Some level of inability to reproduce precisely the same model is a necessary feature of the ML training process.

### Example Reproducibility Problem at YarnIt

At YarnIt, we retrain our search and recommendations models nightly to ensure that we regularly adjust them for changes in products and customer behavior. Typically, we take a snapshot of the previous day’s model and then train the new events since then on top of that model. This is cheaper to do but ultimately does mean that each model is really dozens or hundreds of incremental training runs on top of a model trained quite some time ago.

Periodically, we have small changes to the training dataset over time. The most common changes are charges that are due to fraud. Detecting that a transaction is fraudulent may take up to several days, and by that point we may have already trained a new model with that transaction included as an authorized purchase. The most thorough way to fix that would be to recategorize the original transaction as fraud and then retrain every model that had ever included that transaction from an older snapshot. That would be extremely expensive to do every time we have a fraudulent transaction. We could conceivably end up retraining the last couple of weeks’ models constantly. The other approach is to attempt to reverse the fraud from the model. This is complicated because there is no foolproof or exact way to revert a transaction in most ML models.[13](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#ch01fn82) We can approximate the change by treating the fraud detected as a new negative event, but the resulting model won’t be quite the same.[14](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#ch01fn83)

This is all for the models that are currently in production. At the same time, model developers at YarnIt are constantly developing new models to try to improve their ability to predict and rank. When they develop a new model, they train it from scratch on all the data with the new model structure and then compare it to the existing model to see if it is materially better or worse. It may be obvious where this is going: the problem is that if we retrain the _current_ production model from scratch on the current data, that model may well be significantly different from the current production model that is in production (which was trained iteratively on the same data over time). The fraudulent transactions listed previously will just never be trained on rather than be trained on, left for a while, and then deleted later. The situation is actually even less deterministic than that: even if we train the exact same model on the exact same data with no changes, we might have nontrivial differences, with one model trained all at once and another trained incrementally over several updates.[15](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#ch01fn84)

This kind of really unnerving problem is why model quality must be jointly owned by model, infrastructure, and production engineers together. The only real reliability solution to this problem is to treat each model as it is trained as a slightly different variant of the Platonic ideal of that model and fully renounce the idea of equality between trained models, even when they are the same model configuration trained by the same computers on the same data twice in a row. This, of course, may tend to massively increase the cost and complexity of regression testing. If we absolutely need them to be more stable (note that this is not “the same” since we cannot achieve that in most cases), then we may have to start thinking about ensembles of copies of the same model so that we minimize the change over time.[16](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#ch01fn85)

### Compute Resource Capacity

Just as it is a common cause of outages in non-ML systems, lack of sufficient capacity to train is a common cause of outages for ML systems. The basic capacity that we need to train a new model includes the following:

I/O capacityThis is capacity at the feature store so that we can read the input data quickly.Compute capacityThis is the CPU or accelerator of the training jobs so that we can learn from the input data. This requires a pretty significant number of compute operations.Memory read/write capacityThe state of the model at any given time is stored somewhere, but most commonly in RAM, so when the training system updates the state, the system requires memory bandwidth to do so.

One of the tricky and troubling aspects of ML training system capacity problems is that changes in the distribution of input data, and not just its size, can create compute and storage capacity problems. Planning for capacity problems in ML training systems requires thoughtful architectures as well as careful and consistent monitoring.

### Example Capacity Problem at YarnIt

YarnIt updates many models each day. These are typically trained during the lowest usage period for the website, which is whatever is nighttime for the largest number of users, and are expected to be updated before the start of the busy period the following day. Timing the training in this way gives us the possibility to reuse some resources between the online serving and the ML training system. At the very least, we will need to read the logs produced by the serving system, since the models that YarnIt trains daily read the searches, purchases, and browsing history from the website the day before.

As with most ML models, some types of events are less computationally complicated to process than others. Some of the input data in our feature store requires connections to other data sources in order to complete the input for some training operations. For example, when we show purchases for products listed by our partners rather than by YarnIt directly, we need to look up details about that partner in order to continue to build models that accurately predict customer preferences about those products from that partner. If, for whatever reason, the portion of our purchases from partners increases over time, we might see a significant capacity shortfall in the ability to read from the partner information datasets. Furthermore, this might appear as if we have run out of compute capacity, when actually the CPUs are all waiting on responses from the partner data storage system.

Additionally, some models might be more important than others, and we probably want a system for prioritization of training jobs in those cases where we are resource constrained and need to focus more resources on those important models. This commonly occurs after an outage. Imagine we have a 48-hour outage of some part of the training system. At that point, we have stale models representing our best view of the world over two days ago. Since we were down for so long, it is reasonable to expect that we will take time to catch up, even using all of the machine resources that we have available. In this case, knowing which models are most important to refresh quickly is extremely useful.

## Structural Reliability

Some of the reliability problems for an ML training system come not from the code or the implementation of the training system, but instead from the broader context in which these are implemented. These challenges are sometimes invisible to systems and reliability engineers because they do not show up in the models or the systems. These challenges show up in the organization and the people.

### Organizational Challenges

Many organizations adding ML capabilities start by hiring someone to develop a model. Only later do they add ML systems and reliability engineers. This is reasonable to a point, but to be fully productive, model developers need a stable, efficient, reliable, and well-instrumented environment to run in. While the industry has relatively few people who have experience as production engineers or SREs on ML systems, it turns out that almost all the problems with ML systems are distributed systems problems. Anyone who has built and cared for a distributed system of similar scale should be able to be an effective production engineer on our ML system with some time and experience.

That will be enough for us to get started adding ML to our organization. But if we learned anything from the preceding failure examples, it is that some are extremely straightforward but others really do involve understanding the basics of how the models are structured and how the learning algorithms work. To be successful over the long term, we do not need ML production engineers who are experts in ML, but we do need people who are actively interested in it and are committed to learning more details about how it works. We will not be able to simply delegate all model quality problems to the modeling team.

Finally, we will also have a seniority and visibility problem. ML teams are more likely to get more senior attention than many other similarly sized or scoped teams. This is at least in part because when ML works, it is applied to some of the most valuable parts of our business: making money, making customers happy, and so on. When we fail at those things, senior leaders notice. ML engineers across the whole ecosystem need to learn to be comfortable communicating at a more senior level of the organization and with nontechnical leaders who have an interest in their work, which can have serious reputational and legal consequences when it goes wrong. This is uncomfortable for some of these engineers, but managers building ML teams should prepare them for this eventuality.

For a more in-depth discussion about organizational considerations beyond just the training system, see Chapters [13](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch13.html#integrating\_ml\_into\_your\_organization) and [14](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch14.html#practical\_ml\_org\_implementation\_example).

### Ethics and Fairness Considerations

ML can be powerful but also can cause powerful damage. If no one in our organization is responsible for ensuring that we’re using ML properly, we are likely to eventually run into trouble. The ML training system is one place where we can have visibility into problems (model quality monitoring) and can enforce governance standards.

For organizations that are newer to implementing ML, the model developers and ML training system engineers may be jointly responsible for implementing minimal privacy, fairness, and ethics checks. At a minimum, these must ensure that we are compliant with local laws regarding data privacy and use in every jurisdiction in which we operate. They must also ensure that datasets are fairly created and curated and that models are checked for the most common kinds of bias.

One common and effective approach is for an organization to adopt a set of Responsible AI principles and then, over time, build the systems and organizational capacity to ensure that those principles are consistently and successfully applied to all uses of ML at the organization. Think about how to be consistent at the model level ([Chapter 5](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch05.html#evaluating\_model\_validity\_and\_quality)), policy level ([Chapter 6](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch06.html#fairnesscomma\_privacycomma\_and\_ethical)), but also apply principles to data ([Chapter 4](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch04.html#feature\_and\_training\_data)), monitoring ([Chapter 9](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch09.html#monitoring\_and\_observability\_for\_models)), and incident response ([Chapter 11](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch11.html#incident\_response)).

## Conclusion

Although ML training system implementers will still need to make many choices, this chapter should provide a clear sense of the context, structure, and consequences of those choices. We have outlined the major components of a training system as well as many of the practical reliability principles that affect our use of those systems. With this perspective on how trained models are created, we can now turn our attention to the following steps in the ML lifecycle.

[1](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#ch01fn70-marker) In many modern frameworks (notably, TensorFlow, PyTorch, and JAX), the configuration language used is most commonly actual code, usually Python. This is a significant source of headaches for newcomers to the ML training system world, but does offer advantages of flexibility and familiarity (for some).

[2](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#ch01fn71-marker) Several types of ML systems (notably, reinforcement learning systems) are quite different from this architecture. They often have additional components, like an agent and simulation, and also put prediction in what we call “training” here. We’re not ignorant of these differences, but chose the most common components to simplify this discussion. Your system may have these components in a different order or might have additional ones.

[3](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#ch01fn72-marker) Worse, still, when provenance cannot be tracked, we will have governance problems (compliance, ethics, legal). For example, if we cannot prove that the data that we trained on is owned by us or licensed for this use, we are open to claims that we misused it. If we cannot demonstrate the chain of connection that created a dataset, we cannot show compliance with privacy rules and laws.

[4](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#ch01fn73-marker) For example, see [“How ML Breaks: A Decade of Breaks for One Large ML Pipeline”](https://oreil.ly/Y1tk8) by Daniel Papasian and Todd Underwood.

[5](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#ch01fn74-marker) This recommendation has one interesting exception, but one that will not apply to the vast majority of practitioners: huge language models. Multiple very large language models are being trained by large ML-centric organizations in order to provide answers to complex queries across a variety of languages and data types. These models are so expensive to train that the production model for them is explicitly to train them once and use them (either directly or via transfer learning) “forever.” Of course, if the cost of training these models is significantly reduced or other algorithmic advances arise, these organizations may find themselves training new versions of these models anyway.

[6](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#ch01fn75-marker) Statisticians refer to these various properties of the data as _missing completely at random_ (the propensity for the data point to be missing is completely random); _missing at random_, or _MAR_ (the propensity of the data to be missing is not related to the data but is related to another variable—a truly unfortunate name for a statistical term); and _not missing at random_ (the likelihood of the data to be missing is correlated with something in the data). In this case, we’re describing MAR data because the propensity for any given data point to be missing is correlated with another variable (in this case, geography or time of day, for example).

[7](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#ch01fn76-marker) Adding these features might have significant privacy implications. These are discussed briefly in [Chapter 4](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch04.html#feature\_and\_training\_data) and much more extensively in [Chapter 6](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch06.html#fairnesscomma\_privacycomma\_and\_ethical).

[8](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#ch01fn77-marker) This is most common in model architectures that use gradient descent with significant amounts of parallelism in learning. But this is an extremely common setup for large ML training systems. One example of a model architecture that does not suffer from this problem is random forests.

[9](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#ch01fn78-marker) An architecture in which the parameters of the model are stored is described well in [“Scaling Distributed Learning with the Parameter Server”](https://oreil.ly/dSXst) by Mu Li et al. and in [Chapter 12](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch12.html#how\_product\_and\_ml\_interact) of _Dive Into Deep Learning_ by Joanne Quinn et al. (Corwin, 2019). Variants of this architecture have become the most common way that large-scale ML training stacks distribute their learning.

[10](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#ch01fn79-marker) Transfer learning most generally involves taking learning from one task and applying it to a different, but related task. Most commonly in production ML environments, transfer learning involves starting learning with the snapshot of an already trained, earlier version of our model. We will either train only on new features, not included in the snapshot, or train only on new data that has appeared since the training of the snapshot. This can speed up learning significantly and thereby reduces costs significantly as well.

[11](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#ch01fn80-marker) The astute reader might note how terrifying this is. Another way to read this is “my models could change any time I happen to update TensorFlow or PyTorch, even for a new minor version.” This is essentially true but not common, and the differences often aren’t pronounced.

[12](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#ch01fn81-marker) As long as the speed at which processors (whether CPUs or GPUs/accelerators) and their local memory operate is significantly higher than the speed at which we can access that state from across a network connection, there will always be lag in propagating that state. When processors update a portion of the model based on input data that they have learned from, there can always be other processors using an older version of those keys in the model.

[13](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#ch01fn82-marker) A large and growing set of work exists on the topic of deleting data from ML models. Readers should consult some of this research to understand more about the various approaches and consequences of deleting previously learned data from a model. One paper summarizing some recent work on this topic is [“Making AI Forget You: Data Deletion in Machine Learning”](https://oreil.ly/GWShn) by Antonio A. Ginart et al., but be aware that this is an active area of work.

[14](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#ch01fn83-marker) [Chapter 6](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch06.html#fairnesscomma\_privacycomma\_and\_ethical) covers some cases where we want to delete private data from an existing model trained on that data. The short version is that if the data is truly private and is included in our model, unless we used differential privacy during model construction and provide careful guarantees on how the model can be queried, we probably have to retrain the model from scratch. Indeed, we have to do this every single time someone requests that their data be removed. This, alone, is a powerful argument for ensuring that our models do not include private data.

[15](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#ch01fn84-marker) Details of why this is the case are really specific to the model and ML framework, and beyond the scope of this book. But it often boils down to nondeterminism in the ML framework exacerbated by nondeterminism in the parallel processing of data. Reproducing this nondeterminism in your own environment is tremendously educational and more than a tiny bit terrifying. And yes, this footnote did just encourage readers to reproduce irreproducibility.

[16](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#ch01fn85-marker) _Ensemble models_ are just models that are collections of other models. Their most common use is to combine multiple very different models for a single purpose. In this case, we would combine multiple copies of the same model.
