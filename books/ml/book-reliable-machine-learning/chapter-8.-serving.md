# Chapter 8. Serving

You’ve made a model; now you have to get it out there into the world and start predicting things. This is a process often known as _serving the model_. That’s a common shorthand for “creating a structure to ensure our system can ask the model to make predictions on new examples, and return those predictions to the people or systems that need them” (so you can see why the shorthand was invented).

In our _yarnit.ai_ online store example, we can imagine that our team has just created a model that is great at predicting the likelihood that a given user will purchase a given product. We need to have a way for the model to share its predictions with our overall system. But how, exactly, should we set this up?

We have a range of possibilities, each with different architectures and trade-offs. They are sufficiently different in approach that it might not be obvious looking at the list that these are all attempts to solve the same problem: how can we integrate our predictions with the overall system? We could do any of the following:

* Load the model into 1,000 servers in Des Moines, Iowa, and feed all incoming traffic to these servers.
* Precompute the model’s predictions for the 100,000,000 most commonly seen combinations of yarn products and user queries using a big offline batch-processing job. Write those to a shared database once a day that is read by our system, and use a default score of _p_ = 0.01 for anything not in that list.
* Create a JavaScript version of the model and load it into the web page so that predictions are made in the user’s browser.
* Create a mobile app that has the model embedded into it so that predictions are made on the user’s mobile device.
* Have different versions of the model with different trade-offs of computational cost and accuracy. Create a tiered system in which versions of the model are available in the cloud, using different hardware with different costs. Send the easy queries to a cheaper (less accurate) model and send the more difficult queries to a more expensive (more accurate) model.

This chapter is devoted to helping us map out the criteria for selecting from choices like this. Along the way, we will also discuss critical practicalities like ensuring that the feature pipeline used in serving is compatible with that used in training, and strategies for updating a model in serving.

## Key Questions for Model Serving

There are a lot of ways that we can think about creating structures around a model that support serving, each with very different sets of trade-offs. To help navigate this space, it’s useful to think through some specific questions about the needs for our system.

### What Will Be the Load to Our Model?

The first thing to understand about our serving environment is the level of traffic that our model will be asked to handle—often referred to as _queries per second_ (_QPS_) when queries are made on demand. A model serving predictions to millions of daily users may be asked to handle tens of thousands of queries per second. A model that runs an audio recognizer that listens for a _wake word_ on a mobile device, like “Hey YarnIt,” may run at a few QPS. A model that predicts housing prices for a real estate service might not be served on demand at all, and may instead be run as part of a large batch-processing pipeline.

A few basic strategies can address large traffic load. The first is to replicate the model across many machines and run these in parallel—perhaps using a cloud-based platform to allow a combination of traffic distribution and easy scaling up as demand grows. The second is to use more powerful hardware, such as hardware accelerators like GPUs or other specialized chips. These often require batching requests together to maximize efficiency, because the chips are so powerful that they can be bottlenecked more on input and output rather than on computing the model predictions themselves.[1](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch08.html#ch01fn86)

We could also tune the computation cost of the model itself, perhaps by using fewer features, or a deep learning model with fewer layers or parameters, or approaches such as quantization and sparsification to make the internal mathematical operations less expensive. Model cascades can also be effective at cost reduction—this is where a cheap model is used to make first-guess decisions on easy examples, and only the more difficult examples are sent to a more expensive model.

### What Are the Prediction Latency Needs of Our Model?

_Prediction latency_ is the time between the moment we make a request and the moment we get an answer back. Acceptable prediction latency can vary dramatically among applications, and is a major determiner of serving architecture choices.

For an online web store like _yarnit.ai_, we might have a total time budget of only half a second between the time that the user types in a query like “merino wool yarn” and the time they expect to see a full page of suggested products. Factoring in network delays and other processing necessary to build and load the page, this might mean that we have only a few milliseconds for the model to make all of its predictions on candidate products. Other very low-latency applications might include models that are used in high-frequency trading platforms, or that do real-time guidance of autonomous vehicles.

On the other end of the spectrum, we might have a model that is being used to determine the optimal spot to drill for oil, or that is trying to guide the design of protein sequences to be used to create new antibody treatments. For applications like these, latency is not a major concern because using these predictions (such as actually creating an oil rig or actually testing a candidate protein sequence in the wet lab) is likely to take weeks or months. Other application modalities have implicit delays built in. For example, an email spam-filtering model might not need to have millisecond response time if a user checks their inbox only every morning.

Taken together, latency and traffic load define the overall computational needs of our ML system. If prediction latency is too high, we can mitigate the issue by using more powerful hardware, or by making our model less expensive to compute. However, it is important to note that parallelization by creating a larger number of model replicas is usually not a solution to prediction latency, as the end-to-end time it takes to send a single example through the model isn’t affected by simply having more versions of the model available.

Real systems often produce a distribution of latency values, due to network effects and overall system load. It can be useful to look at tail latency, such as the worst few percent, rather than average latency, so that we do not miss noticing if a percentage of requests are getting dropped.[2](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch08.html#ch01fn87)

### Where Does the Model Need to Live?

In our modern world, defined as it is by flows of information and concepts like virtual machines and cloud computing, it can be easy to forget that computers are physical devices, and that a model needs to be stored on a physical device in a specific location. We need to determine the home (or homes) for our model, and this choice has significant implications on the overall serving system architecture. Here are some options to consider.

#### On a local machine

Although this is not really a production-level solution, in some cases a model developer may have a model running on their local machine, perhaps invoking small batch jobs to process data when needed. This is not recommended for anything beyond small-scale prototyping or bespoke uses. Even in these cases, it is easy to come to rely on this in early stages and create more trouble than expected when we need to migrate to a production-level environment.

#### On servers owned or managed by our organization

If our organization owns or operates its own servers, we likely will run our models on this same platform. This may be especially important when specific privacy or security concerns are in place. It may also be the right option if latency is a hypercritical concern, or if specialty hardware is needed to run our models. However, this choice can limit flexibility in terms of ability to scale up or down, and will likely require special attention to monitoring.

#### In the cloud

Serving our models by using a cloud-based provider can allow for easily scaling our overall computational footprint up or down, and may also allow us to choose from several hardware options. This may be done in two ways. The first, running model servers on our own virtual servers and controlling how many of them we use, is essentially indistinguishable from the preceding option of using servers owned or managed by our organization. In this case, it might be slightly easier to scale up or scale down the number of servers, but the management overhead is otherwise similar. Here we’re more interested in the second case: using a managed inference service.

In a managed inference service, some monitoring needs may be automatically addressed—although we will still likely need to independently verify and monitor overall model quality and predictive performance. Round-trip latency will likely be higher because of network costs. Depending on the geographical location of the actual datacenters, these costs may be higher or lower, and we may be able to mitigate some of these issues by using datacenters in multiple major geographical locations if we will be fielding requests globally. Privacy and security needs are also highlighted here, as we will be sending information across the network and will need to ensure that appropriate safeguards are in place. Finally, in addition to privacy and security concerns, we may have governance reasons for being cautious about using particular cloud providers: some online activities are regulated by national governments in a way that requires certain data to be kept in particular jurisdictions. Make sure you know about these factors before making a serving layout plan.

#### On-device

Today’s world is filled with computational devices that are part of our daily lives. Everything from mobile phones to smart watches, digital assistants, automobiles, thermostats, printers, home security systems, and even exercise equipment have a surprising amount of computational capacity, and developers are finding ML applications in nearly all of them. When a model is needed in these settings, it is much more likely that it will need to be stored on the device itself, because the alternative is to access a model in the cloud, which requires constant network connection and may also have complicated privacy concerns. These settings of serving “on the edge” typically have strict constraints on model size, because memory is limited, as well as the amount of power that may be consumed by model predictions.

Updating the model in such settings typically requires a push across the network, and is unlikely to happen for all such devices in a timely fashion; some devices may never receive any updates at all. Because of the difficulty of making fixes to push updates, testing and verification take on a whole new level of importance in these settings. In some critical use cases, such as a model that continually needs to scan input audio for certain commands, it may even be necessary to encode the model at the hardware level rather than the software levels. This can yield huge efficiency improvements, but at the cost of making updates more difficult—or even impossible.

### What Are the Hardware Needs for Our Model?

In recent years, a range of computational hardware and chip options has emerged that has enabled dramatic improvements in serving efficiency for various model types. Understanding these options is important for informing our overall serving architecture.

The main thing to know about deep models in serving is that they rely on dense matrix multiplications, which basically means that we need to do a lot of multiplication and addition operations in a way that is compute intensive, but that is also highly predictable in terms of memory access patterns.[3](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch08.html#ch01fn88) The little multiplication and addition operations that make up one dense matrix multiplication operation parallelize beautifully. This means that traditional CPUs will struggle to perform well. At the time of writing, a typical CPU has around eight cores, each with one or at most a small number of algorithmic logic units (ALUs), which are the pieces of the chip that know how to do multiplication and addition operations. Thus, CPUs can typically parallelize only a handful of these operations at once, and their strengths in handling branching, memory access, and a wide variety of computational operations don’t really come into play. This makes running inference for deep learning models slow on CPUs.

A much better choice for serving deep learning models are chips called _hardware accelerators_. The most common ones are GPUs because these chips were first developed for processing graphics, which also rely on doing fast dense matrix multiplications.[4](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch08.html#ch01fn89) The main insight in a GPU is that if a few ALUs are good, thousands must be better. GPUs are thus great at the special-purpose task of dense matrix multiplications, but typically are not well suited to other tasks.[5](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch08.html#ch01fn90)

Of course, GPUs have drawbacks, the most obvious of which is that these are specialized hardware. This typically means that either we need to invest organizationally in serving deep models using GPUs, or we are using a cloud service that supplies GPUs (and may charge a premium accordingly), or that we are serving in an on-device setting where a GPU is locally available.

The other main drawback of GPUs is that they’re not well suited to operations not involving large amounts of dense matrix multiplications. Sparse models are one such example. Sparse models are most useful when we need to use only a small number of important pieces of information out of a large universe of possibilities, such as the specific words that show up in a given sentence or search query out of the large universe of all possible words. With appropriate modeling, sparsity can be used in these settings to dramatically reduce computational cost, but GPUs can’t easily benefit from this and CPUs may be much more appropriate. Sparse models may include nondeep methods such as sparse linear models or random forests. They can also appear in deep learning models as sparse embeddings, which can be thought of as a learned input adapter that converts sparse input data (such as text) into a dense representation that is more easily used within a deep model.

### How Will the Serving Model Be Stored, Loaded, Versioned, and Updated?

As a physical object, our serving model has a specific size that needs to be stored. A model that is serving offline in an environment might be stored on disk and loaded in by specific binaries in batch jobs whenever a new set of predictions needs to be made. The main storage requirements are thus the disk space needed to keep the model, as well as the I/O capacity to load the model from disk, and the RAM needed to load the model into memory for use—and of these costs, the RAM is likely more expensive or more limited in capacity.

A model that is used in live online serving needs to be stored in RAM in dedicated machines, and for high-throughput services in latency-critical settings, copies of this model likely will be stored and served in many replica machines in parallel. As we discuss in [Chapter 10](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch10.html#continuous\_ml), most models will need to be updated eventually by retraining on new data, and some are updated weekly, daily, hourly, or even more frequently. This means that we will need to swap the version of a model currently used in serving on a given machine with a new version.

If we want to avoid production outages while this happens, we have two main strategies. The first is to allocate twice as much RAM for the serving jobs, so that the new version of the model can be loaded into the machine while the old one is still serving, and then hot-swapping which one is used once the new version is fully ready. This works well but is wasteful of RAM for the majority of the time when a model is not being loaded or swapped. The second is to overprovision in terms of the number of replica machines by a certain percentage and then to progressively take a proportion (e.g., 10%) offline in turn to update the model. This more gradual approach also allows for more graceful error checking and canarying.

It is also important to remember that if we want our system to support A/B testing, which most developers will want to use, then it will be important to create an architecture that allows both an A and a B version of the model to be served—and indeed, developers may want to have many kinds of Bs running in A/B tests at the same time. Deciding exactly how many versions will be supported and at what capacity is an important architectural choice that requires balancing resourcing, system complexity, and organizational requirements together.

### What Will Our Feature Pipeline for Serving Look Like?

Features need to be processed at serving time as well as at training time. Any feature processing or other data manipulation that is done to our data at training time will almost certainly need to be repeated for all examples sent to our model at serving time, and the computational requirements for this may be considerable. In some cases, this is as simple as converting the raw pixel values of an image into a dense vector to be fed to an image model. In more typical production settings, it may require joining several sources of information together in real time.

For example, for our _yarnit.ai_ store, we might need to supply a product recommendation model with the following:

* Tokenized normalized text from a user query, drawn from the search box entry
* Information about past purchase history, drawn from a stored database of user information
* Information about product prices and descriptions, drawn from a stored product database
* Information about geography, language, and time of day, drawn from a localization system

Each kind of information comes from a different source, and may have different opportunities for precomputation or reuse from query to query or session to session. In many cases, this means that the actual code used to turn these pieces of information into features for our ML model to use may be different at serving time from the code used for similar tasks at training time. This distinction is one of the main sources of classic _training-serving skew_ errors and bugs, which are notoriously difficult to detect and debug. For a much more in-depth discussion of this kind of skew, and others, see [Chapter 9](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch09.html#monitoring\_and\_observability\_for\_models).

One of the promises of modern feature stores is that they handle both training and serving together in a single logical package. The reality of this promise may differ by system and use case, so it is well worth ensuring robust monitoring in any case.

It is also worth noting that creating features for our model to use at serving time is a key source of latency, and in many systems will be the dominating factor. This means that the serving feature pipeline is far from an afterthought, and is indeed often the most production-critical part of the entire serving stack.

## Model Serving Architectures

With the preceding questions in mind, we will now examine four broad categories of serving architectures. Obviously, each needs to be tailored to specific use cases, and some serving systems may use a combination of approaches. With that said, we observe that most of the architecture and deployment approaches fall into the following four broad categories:

* Offline
* Online
* Model as a service
* Serving at the edge

We look at each in detail now.

### Offline Serving (Batch Inference)

_Offline serving_ is often the simplest and fastest architecture to implement. The application serving the end user is not exposed to the models directly. Models are trained ahead of time, often referred to as _batch inference_.

Batch inference is a way to avoid the problem of hosting a model to be reachable for predictions on demand when you don’t need that. It works by loading the model and executing its predictions offline against a predefined set of input data. As a result, the model’s predictions are stored as a simple dataset, perhaps in a database or a _.csv_ file or another resource that stores data. Once these predictions are needed, the problem is identical to any other problem for which static data resources need to be loaded from data storage. In essence, by computing the model predictions offline, you convert on-demand model predictions into a more standard problem of simple data lookup ([Figure 8-1](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch08.html#offline\_model\_serving\_via\_data\_store)).

For example, the _popularity_ of each product on _yarnit.ai_ for a given subset of users can be computed offline—perhaps at a convenient low-load time, if doing so is expensive in some way—and used as a sort function helper when displaying arbitrary items at any point as required to render the page.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098106218/files/assets/reml_0801.png" alt="Offline model serving via data store" height="207" width="600"><figcaption></figcaption></figure>

**Figure 8-1. Offline model serving via data store**

If the use case is less demanding, we might even be able to avoid the complexity of storing and serving model predictions via a database and write the predictions to a flat file, or in-memory data structure, and use them directly within the application ([Figure 8-2](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch08.html#offline\_model\_serving\_via\_in\_memory\_dat)). As an example, our web store could use a _search query intent classifier_ (specific product versus broad category) to help the query engine rewrite the query for retrieving the search results efficiently. (You could, of course, build an approximation to the same structure by, say, indexing yarn by fiber content as a hash containing wool, cotton, acrylic, blends, etc., and/or a reverse hash.)

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098106218/files/assets/reml_0802.png" alt="Offline model serving via in-memory data structures" height="264" width="600"><figcaption></figcaption></figure>

**Figure 8-2. Offline model serving via in-memory data structures**

#### Advantages

The advantages of offline serving are as follows:

Less complicatedThis approach requires no special infrastructure. Often you can reuse something you already have, or start something small and simple. The runtime system has fewer moving parts.Easy accessApplications facilitating the use case can perform simple key-value lookups or SQL queries based on the data store.Better performancePredictions are provided quickly, since they have been precomputed. This might be an overriding consideration for certain mission-critical applications.FlexibleBy using separate tables or records based on an identifier, this approach provides a flexible and easy way to roll out and roll back various models.VerificationThe ability to verify all model predictions before use is a significant benefit for establishing correct operation.

#### Disadvantages

The disadvantages of offline serving are listed here:

Availability of data (training)Training data needs to be available ahead of time. Hence, model enhancements will take longer to deploy into production systems. Also, a critical upstream data outage could lead to stale models, days’ worth of delays, permanently lost data, and expensive backfill processes to “catch up” the offline job to a current state.Availability of data (serving)Effectively, the serving data needs to be available ahead of time; for fully correct operation, the system needs to know in advance every possible query that will be made of it. This is simply impossible in many use cases.ScalingScaling is difficult, especially for use cases dependent on large datasets or large query spaces. For example, you can’t handle a search query space with a long tail—i.e., many different queries, the bulk of which aren’t commonly used—with high accuracy and low latency.Capacity limitsStoring multiple model outputs in memory or in application databases will have storage limitations and/or cause performance problems. This will impact the ability to run multiple A/B tests at the same time. This may not cause a real problem, provided that the database and query resource requirements scale at similar rates and provided we have enough resources to provision.Less selectivitySince models and predictions are precomputed, we won’t be able to influence the predictions by using online context.

### Online Serving (Online Inference)

In contrast to the preceding approach, _online serving_ does not rely on precomputed outputs from a fixed query space. Instead, we provide predictions in real time by ingesting/streaming samples of real-time data, generally from user activity. In our web store example, we could build a more personalized shopping experience by having the model constantly learn the real-time user behavior by using the current context along with historical information to make the predictions. The current context might include location, views/impressions on precomputed recommendations, recent search sessions, items viewed, or items added to and removed from the basket.

Because all of this activity can be taken into account at prediction time, this allows significantly more flexibility in how to respond. Applications powered by inferences generated by offline models plus training the supplemental models for additional parameters in real time ([Figure 8-3](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch08.html#online\_model\_serving\_in\_combination\_wit)) provides huge benefits and significant business impacts.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098106218/files/assets/reml_0803.png" alt="Hybrid online model serving in combination with predictions generated offline" height="292" width="600"><figcaption></figcaption></figure>

**Figure 8-3. Hybrid online model serving in combination with predictions generated offline**

#### Advantages

Advantages of online serving include the following:

AdaptabilityOnline models learn as they go, and so greatly reduce the cadence with which model retraining and redeployment are required. Instead of adapting to concept drift at deployment time, the model adapts to concept drift at inference time, improving the performance of the model for customers.Amenable to supplemental modelsInstead of training and changing one global model, we can tune more situation-specific models with a small subset of real-time data (for example, user- or location-specific models).

#### Disadvantages

Here are some disadvantages of the online-serving approach:

Latency budget requiredThe model needs access to all relevant features. It will need quick access to new queries so that it can convert them into features and in turn look up relevant features stored elsewhere. If all the data we need for a single training example can’t be sent to the server as part of the payload on the API call, we need to grab that data from somewhere else in milliseconds. Typically, that means using an in-memory store of some kind (for example, Redis).Deployment complexitiesAs the predictions are made in real time, rolling out the model changes is highly challenging, especially in a container-orchestration environment like Kubernetes.Scalability constrainedSince a model can and will change from time to time, it’s not horizontally scalable. Instead, we might need to build a cluster of single-model instances that can consume new data as quickly as possible, and return the sets of learned parameters as part of the API response.Higher oversight requirementsThis approach needs more advanced monitoring and adjustment/rollback mechanisms in place, since real-time changes could well include fraudulent behaviors caused by the bad actors in the ecosystem, and they could interact with, or influence, model behavior in some way.Higher management requirementsIn addition to strong monitoring and rollback mechanisms, doing this correctly requires nontrivial expertise and fine-tuning to get right, both for data science and product engineering. Therefore, this approach is probably worth it for only critical line-of-business applications, usually with high monetary impact for the business.

Serving models online is more powerful when it’s combined with the model-as-a-service approach we discuss in the following section. We note in passing that real-time predictions can be served either synchronously or asynchronously. While synchronous mode is more straightforward and simpler to reason about, asynchronous mode gives us a lot more flexibility to handle the way results are passed around, and enable approaches like sending predictions via push _or_ pull mechanisms, depending on the application and the end client (browser, app, device, internal service, etc.).

### Model as a Service

The _Model-as-a-Service (MaaS)_ approach is similar to software as a service and inherently favors a microservice architecture. With MaaS, models are stored in a dedicated cluster and served results via well-defined APIs. Regardless of the transport or serialization methods (e.g., gRPC or REST),[6](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch08.html#ch01fn91) because models are served as a microservice, they’re relatively isolated from the main application ([Figure 8-4](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch08.html#models\_served\_as\_a\_separate\_microservic)). This is therefore the most flexible and scalable deployment/serving strategy, since no in-process interaction or tight coupling are necessarily required.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098106218/files/assets/reml_0804.png" alt="Models served as a separate microservice" height="264" width="600"><figcaption></figcaption></figure>

**Figure 8-4. Models served as a separate microservice**

Given the wide popularity of X-as-a-service approaches throughout the industry, we will focus on this particular method more than others, and will examine in detail various aspects of serving model predictions via APIs later in the chapter.

#### Advantages

The following are advantages of MaaS:

Leveraging contextBy definition in the MaaS context, we have the ability to serve predictions in real time by using real-time context and new features.Separation of concernsA separate service approach allows ML engineering to make model adjustments in a stabler way, and apply well-known techniques for managing operational problems. Most models of the MaaS type can be organized in a stateless way without any shared configuration dependencies. In these cases, adding a new model-serving capacity is as simple as adding new instances to the serving architecture, also known as _horizontal scaling_.Deployment isolationAs per any development architecture in which RPCs are the sole method of communication, the choice of technical stack could vary between application and model service layers, allowing respective teams to develop very differently if required. Independent deployment cycles could follow too, and make it a little easier to deploy versions on different timescales, or multiple environments: QA, staging, canaries, etc.Version managementVersioning is easy to proliferate, since we can store multiple versions of the models in the same cluster and point to them as required; this is extremely convenient for A/B testing, for example. The version-identifying information about the model being used can often be designed as a part of the service’s response data as well. Among other benefits, this allows for rolling redeployments because stakeholder systems can rely on a model identifier to track, route, and collate any event data that may be generated as a result of using the ML model, such as tracking which model was used to serve a particular result in an A/B test.Centralization facilitates monitoringBecause the model architecture is centralized, it’s comparatively easier to monitor system health, capacity/throughout, latency, and resource consumption, as well as per model business metrics like impressions, clicks, conversions, and so on. If we design architectural components that wrap inputs/outputs and standardize the process of identifying models and loading them from configs, many of the SRE “golden four” types of observability metrics can be obtained “for free” just by plugging into other predefined tools that provide these for other general microservices.

#### Disadvantages

MaaS disadvantages are as follows:

Management overheadWhen you climb aboard the microservice train, getting off is difficult, and a lot of overhead is required to stay onboard safely and well. However, this overhead does at least have the advantage of being somewhat well documented and understood.Organizational complianceWhen we reply on the standard framework for deploying microservices, we might initially get a lot of things “for free,” such as log aggregation, metrics scraping and dashboarding, tracking metadata for containers and compute usage, and managed delivery software that converts a code build or release into a real deployment. But we will also get change requests to comply with privacy, security standards, authentication, auditing, resource limitations, and various migrations.Latency budgets requiredIn any kind of microservice architecture that effectively externalizes your call stack, latency becomes a critical and unignorable constraint. Since user-perceived latency needs to be kept within reasonably tight constraints (subsecond, ideally), this imposes performance-related constraints on all the other systems you’ll communicate with. It also potentially creates an organizational blind spot around user-perceived performance, since (by default in siloed enterprises) no one team will own that performance as a whole. As a result, the choice of underlying data stores, languages, and organizational structure and patterns becomes important.Distributed availabilityArchitectures built on distributed microservices must be able to tolerate partial failures robustly. The calling service must have reasonable fallbacks when the model service is down.

### Serving at the Edge

A slightly less commonly understood serving architecture is used when a model is deployed onto edge devices ([Figure 8-5](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch08.html#models\_served\_at\_the\_edge\_and\_as\_a\_sepa)). An _edge device_ might be anything from an Internet of Things (IoT) doorbell to self-driving vehicles, or anything in between. Today, the bulk of edge devices with an internet connection are modern smartphones.

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098106218/files/assets/reml_0805.png" alt="Models served at the edge and as a separate microservice on the server" height="264" width="600"><figcaption></figcaption></figure>

**Figure 8-5. Models served at the edge and as a separate microservice on the server**

Usually these models don’t exist on their own: a server-side supplemental model of some kind helps fill the gaps. It’s also common that most edge applications primarily rely on on-device inferences. That might change in the future, with emerging techniques like federated/collaborative learning.[7](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch08.html#ch01fn92) The closeness to the user is a great advantage for some applications, but we often face severe resource limits in this architecture.

#### Advantages

Serving at the edge has these advantages:

Low latencyPutting the model on the device allows it to be quicker. Near-instantaneous response (and no risk of dropped packets, etc.) to predict things can be absolutely critical for some applications: high latency or jitter in self-driving vehicles could cause accidents, injuries, or even deaths. Running models on edge devices is essentially compulsory here.More-efficient network usageThe more queries you can answer locally, the fewer you have to send over the network.Improved privacy and securityMaking inferences locally means that the user data and the predictions made on that data are much harder to compromise. This is really useful for personalized search, or recommendations that might require PII such as user profile, location, or transaction history.More reliableWhen the network connection is not consistently stable, being able to execute certain operations locally that were previously executed remotely becomes desirable and sometimes even necessary.Energy efficiencyOne key design requirement of edge devices is energy efficiency. In certain cases, local computing consumes less energy than network transmission.

#### Disadvantages

The following are disadvantages of serving at the edge:

Resource constraints (specialization)With limited computing power, edge devices can perform only a few tasks. Non-edge infrastructure should still handle training, building, and serving of large models, while edge devices can perform local inferences with smaller models.Resource constraints (accuracy)ML models can consume lots of RAM and be computationally expensive; fitting these on memory-constrained edge devices can be difficult or impossible. The good news is a lot of research is ongoing to find alternative ways to address this; for example, parameter-efficient neural networks like SqueezeNet and MobileNet are both attempts to keep the models small and efficient without sacrificing too much accuracy.Device heterogeneity (device-specific programming languages)Coming up with a way to ensure that edge serving and on-device training happen precisely the same on both iOS and Android, for example, is a significant challenge. Doing it efficiently within the context of mobile development best practices also involves the intersections of two highly domain-specific groups of people (ML engineers and mobile engineers), which can create organizational strain on scarce shared specialty teams or prevent embracing standardized team models for full-stack development. A similar set of interactions exists whenever software is deployed into public-facing use as a service. For example, an accounting service available on the web will require that software engineers expert in building accounting systems deal with production engineers experienced at running software in production. The difference here is mostly of degree: ML engineers and mobile engineers come from extremely different worlds and technical contexts and are unlikely to communicate well without effort.Device software versions are in the user’s controlUnless you use a backend-for-frontend proxy service design pattern to route various calls off-device into a server-side backend, the owner of the edge device controls the software update cycles. We might push out a critical improvement to an on-device ML model in an iOS app, but that doesn’t mean that millions of existing users have to update the version on their iPhones. They might wait around and continue using the outdated version for as long as they would like. Because any ML model deployed to the edge device might need to robustly keep operating and have its prediction and on-device learning setup continue working for a long time, it’s a huge architectural commitment that might carry a lot of future-looking tech debt and legacy support with it, and should be chosen carefully.

**NOTE**

One of the important attributes you need to track when serving ML models in production is versioning. Feedback loop data, backups, disaster recovery, and performance measurement all rely on it. We discuss these ideas in more detail in [Chapter 9](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch09.html#monitoring\_and\_observability\_for\_models). In particular, we will look at suggested measurements in two sections: serving and SLOs.

### Choosing an Architecture

Having talked about the various architecture options, we now need to choose the right one! Depending on the use case, that could be a complex affair; the differences between the model lifecycles, formats, and so on are one axis of consideration, never mind the vast implementation landscape that exists.

Our recommended approach is to first consider the _amount_ of data and _speed_ of the data required for your application: if extremely low latency is the priority, use offline/in-memory serving. Otherwise, use MaaS, except when you’re running on an edge device, in which case serving at the edge is (obviously) the most appropriate.

The rest of this chapter focuses on MaaS, since it’s more flexible, and pedagogically better since it suffers from fewer constraints.

## Model API Design

Production-scale ML models are usually built using a wide variety of programming languages, toolkits, frameworks, and custom-built software. When trying to integrate with other production systems, such differences limit their _accessibility_, since ML and software engineers may have to learn a new programming language or write a parser for a new data format, and their _interoperability_, requiring data format converters and multiple language platforms.

One way of improving the accessibility and interoperability is to provide an abstracted interface via web services. Resource-oriented architectures (ROAs) in the REST style appear well suited to this task, given the natural alignment of REST’s design philosophy with the desire to hide implementation-specific details.[8](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch08.html#ch01fn93) Partially in support of this view, we’ve seen rapid growth in the area of ML web services in recent years: for example, Google Prediction/Vision APIs, Microsoft Azure Machine Learning, and many more.

Most service-oriented architecture (SOA) best practices apply to ML model/inference APIs too.[9](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch08.html#ch01fn94) But you’ll want to take note of the following points for models:

Data science versus engineering skillsMany organizations have a pure data science team, with little or no experience running services in production. To gain all the benefits of DevOps, however, you will want to empower the data science team to take full ownership of releasing its models to production. Instead of “handing over” models to another team, they will collaborate with operations teams and co-own that process from start to finish.Representations and modelsEven the slightest change in distribution of a feature may cause models to drift. For complex-enough models, creating this representation may mean numerous data pipelines, databases, and even upstream models. Handling this relationship is nontrivial for many ML teams.Scale/performance characteristicsIn general, the _predict_ part of the pipeline is purely compute-bound, something that is rather unique in a service environment. In many cases, the _representation_ part of the workflow is more I/O bound, especially when we need to enrich the input, by loading data/features, or retrieve the image/video we’re trying to predict on.

We believe the overwhelming factor that drives many design patterns in inference service design is—perhaps surprisingly—_organizational support and skill set_. A fundamental tension exists between requiring a data science team to fully own all end-to-end components of a production deployment and fully separating production concerns away from the data science team so it may focus fully on its domain specialization of model training and model optimization.

Too far in either direction may prove unhealthy. If the data science team is asked to own too much, or doesn’t have close collaborative partnership with operations support teams, it can become overwhelmed dealing with production concerns for which it has no training. If the data science team owns too little, it may be disconnected from the constraints or realities of the production system its models must fit into, and will be unable to remediate errors, assist in critical bug fixes, or contribute to architectural planning.

So, when we are ready to deploy models in production, we actually deploy two different things: the model itself and the APIs that go and query the model to fetch the predictions for a given input. Those two things also generate a lot of telemetry and a lot of information that’ll later be used to help us monitor the models in production, try to detect drift or other anomalies, and feed back into the training phase of the ML lifecycle.

### Testing

Testing model APIs, before deploying and serving in production, is extremely critical because models can be have a significant memory footprint and require significant computational resources to provide fast answers. Data scientists and ML engineers need to work closely with the software and QA engineers, and product and business teams, to estimate API usage. At a minimum, we need to perform the following tests:

* Functional testing (e.g., expected output for given input)
* Statistical testing (e.g., test the API on 1,000 unseen requests, and the distribution of the predicted class should match the trained distribution)
* Error handling (e.g., data type validation in the request)
* Load testing (e.g., _n_ simultaneous users calling _x_ times/second)
* End-to-end testing (e.g., validate that all the subsystems are working and/or logging as expected)

## Serving for Accuracy or Resilience?

When serving ML models, a performance increase doesn’t always mean business growth. Monitoring and correlating the model metrics with the business key performance indicators (KPIs) help bridge the gap between performance analysis and business impact, integrating the whole organization to function more efficiently toward a common goal. It is important to view every improvement in the ML pipeline through business KPIs; this helps in quantifying which factors matter the most.

Model performance is an assessment of the model’s ability to perform a task accurately, not only with sample data but also with actual user data in real time in a production setup. It is necessary to evaluate performance to spot any erroneous predictions like drift in detection, bias, and increased data inconsistency. Detection is followed by mitigation of these errors by debugging, based on its behavior to ensure that the deployed model is making accurate predictions at the user’s end and is resilient to data fluctuations. ML model metrics are measured and evaluated based on the type of model that the users are served by (for example, binary classification, linear regression, etc.), to yield a statistical report that enlists all the KPIs and becomes the basis of model performance.

Even though improvements in these metrics, such as minimizing log loss or improving recall, will lead to better statistical performance for the model, we find that business owners tend to care less about these statistical metrics and more about business KPIs. We will be looking for KPIs that provide a detailed view of how well a particular organization is performing, and create an analytical basis for optimized decision making. In our _yarnit.ai_ web store example, a couple of main KPIs could be as follows:

Page views per visitThis measures the average number of pages a user visits during a single site visit. A high value might indicate an unsatisfactory user experience due to the enormous digging the user had to do to reach what they want. Alternatively, a very low value might indicate boredom or frustration with the site and point to abandonment.Returning customer orderThis measures the orders of an existing customer, and is essential for keeping track of brand value and growth.

A resilient model, while not the best model with respect to data science measures like accuracy or AUC, will perform well on a wide range of datasets beyond just the training set. It will also perform better for a longer period of time, as it’s more robust and less overfitted. This means that we don’t need to constantly monitor and retrain the model, which can disrupt model use in production and potentially even create losses for the organization. While no single KPI measures model resilience, here are a few ways we can evaluate the resiliency of models:

* Smaller standard deviations in a cross-validation run
* Similar error rates for longer times in production models
* Less discrepancy between error rates of test and validation datasets
* How much the model is impacted by input drift

We discuss more details about model quality and evaluation in [Chapter 5](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch05.html#evaluating\_model\_validity\_and\_quality), and API/system-level KPIs like latencies and resource utilization in [Chapter 9](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch09.html#monitoring\_and\_observability\_for\_models).

## Scaling

We’ve exposed the models via API endpoints so they can deliver value to the business and customers. This is good, but it’s just the beginning. If all goes well, the model endpoints might see significantly higher workloads in the near future. If the organization starts to serve many more users, these increased demands can quickly bring down the ML services/infrastructure.

ML models deployed as API endpoints need to respond to such changes in demand. The number of API instances serving the models should increase when requests rise. When workload decreases, the number of instances should be reduced so that we don’t end up in a state of underutilization of resources in the cluster, and we could potentially save a significant amount of operational expenses. This is similar to the autoscaling in any cloud computing environment in modern software architectures. Caching can also be efficient in ML environments, just as in traditional software architectures. Let’s discuss these briefly.

### Autoscaling

_Autoscaling_ dynamically adjusts the number of instances provisioned for a model in response to changes in the workload. Autoscaling works by monitoring a target metric (e.g., CPU or memory usage) and comparing it to a target value we monitor for. Additionally, we can configure the minimum and maximum scaling capacity and a cool-down period to control scaling behavior and price. In our _yarnit.ai_ web store example, the per language spelling-correction module used for powering search use cases can be scaled independently from scaling the personalized recommendations module for sending periodic emails recommending new/similar products based on customer purchase history.

### Caching

Consider the problem of predicting categories and subcategories within our _yarnit.ai_ online store. A user might search for “cable needles,” and we might predict their intended shopping area is Equipment → Needles coming from an internal taxonomy of our store category layout. In a case like this, rather than repeatedly invoke the expensive ML model each time a repeat query like “cable needles” is encountered, we could leverage a cache.

For simple cases that have a small number of queries in the cache, this can usually be solved with a simple in-memory cache, possibly defined directly in the application logic or in the model’s API server. But if we are dealing with a huge number of customer queries to fit in the cache, we may want to expand our cache into a separate API/service that can be independently scaled and monitored.

## Disaster Recovery

ML serving via MaaS has all the same failure-recovery requirements as other software as a service (SaaS) platforms: surviving the loss of a datacenter and diversifying infrastructure risks, avoiding vendor lock-in, rolling back bad code changes quickly, and ensuring good circuit breaking to avoid contributing to failure cascades. Separate from these standard service failure considerations, the deep reliance of ML systems on training and data pipelines (whether online or offline) creates additional requirements, including accommodating data schema changes and database upgrades, onboarding new data sources, durable recovery of stateful data resources (like the state of online learning, or the state of on-device retraining in an edge serving use case after an app crashes), and graceful failure in the face of missing data or upstream data ETL job outages—to name but a few.

Data is constantly changing and growing in data warehouses, data lakes, and streaming data sources: adding new and/or enhancing existing features in the product/service creates new telemetry, a new data source may be added to supplement a new model, an existing database goes through a migration, someone accidentally begins initializing a counter at 1 instead of 0 in the last version of the model, and the list can go on. Any one of such changes brings more challenges to ML systems in production.

We discussed the challenges around data availability in [Chapter 7](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch07.html#training\_systems). Without proper care for failure recovery, ML models that experience unexplained data changes or data disruptions may need to be taken out of production and iterated offline, sometimes for months or longer. During the early stages of architecture review, be sure to ask many questions about how the system will react to unusual data changes and how the system can be made robust to allow it to continue operating in production. Additionally, we will inevitably want to expand a successful model’s scope or optimize a poorly performing model by adding additional data features. It is critical to factor in this data extensibility as an early architectural consideration to avoid failure scenarios where we are blocked from being able to ingest a new feature for the model because of the logistics of accommodating the new data in production.

Also for high availability, we may want to run the model API clusters in multiple datacenters and/or availability zones/regions in the cloud computing world. This will allow us to quickly route the traffic when an outage occurs in a specific cluster. Such deployment architecture decisions are fundamentally driven by the SLOs of the organization.[10](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch08.html#ch01fn95) We discuss SLOs in more detail in [Chapter 9](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch09.html#monitoring\_and\_observability\_for\_models).

Just as with application data, we need to have backup strategies in place to constantly take snapshots of the current model data and use the last known good copies when needed. These backups could be used offline for further analysis and could potentially feed into training pipelines to enhance the existing models by deriving new features.

## Ethics and Fairness Considerations

The general topic of fairness and ethics (along with privacy) is covered in depth in [Chapter 6](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch06.html#fairnesscomma\_privacycomma\_and\_ethical). This is a broad area that can be overwhelming for system implementers to consider. We strongly encourage you to read that chapter for a general introduction along with some concrete suggestions.

We should consider the following specific points for serving, however:

Organizational support and transparencyWhen it comes to ethics and fairness while serving the ML models in production, we need to establish checks and balances as part of the development and deployment framework and be transparent with both internal stakeholders and customers about the data being collected and how it will be used.Minimize privacy attack surfaceWhen we process a request through the model APIs, request and response schemas should try to avoid or at least minimize the need for user personal, demographic information. If it’s part of the request, we need to make sure that data is not logged anywhere while serving the predictions. Even for serving personalized predictions, organizations that are extremely committed to ethics and privacy often interact with serving infrastructure with short-lived user identifiers/tokens instead of tracking the unique identifiers like user ID, device ID, and so on.Secure endpointsAlong with data privacy, especially when dealing with PII, product/business owners and ML/software engineers should invest more time and resources to secure the model API endpoints even though they are accessible only within the internal network (i.e., user requests are first processed by the application server before calling model APIs).Everyone’s responsibilityFairness and ethics are a responsibility for everyone, not just ethicists, and it is critical that implementers and users of an ML serving system be educated about these topics. Governance of these critical issues is not just the domain of ML engineers, and must be informed holistically by other members of the organization, including legal counsel, governance and risk management, operations and budget planning, and all members of the engineering team.

## Conclusion

Serving reliably is hard. Making the models available to millions of users with millisecond latencies and 99.99% uptime is extremely challenging. Setting up the backend infrastructure so the right people can be notified when something goes wrong and then figuring out what went wrong is also hard. But we can successfully tackle that complexity in multiple ways, including asking the right questions about your system at the start, picking the right architecture, and paying specific attention to the APIs you might implement.

Serving isn’t a one-time activity either, of course. Once we’re serving, we then need to monitor and measure success (and availability) constantly. There are multiple ways to measure the ML model and product impact over the business, including input from key stakeholders, customers, and employees, and actual ROI as measured in revenue, or some other organizationally relevant metric. Hints on this, and other topics in deployment, logging, debugging, and experimentation are to be found in [Chapter 9](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch09.html#monitoring\_and\_observability\_for\_models), while there is a much more complete coverage of measuring models in [Chapter 5](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch05.html#evaluating\_model\_validity\_and\_quality).

[1](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch08.html#ch01fn86-marker) In some cases, however, your choices about how to arrange computation are fixed and cannot be changed. Models that must be served on a device are one such example. Say we are deploying a model within a mobile app that uses image recognition to identify knitting patterns for sweaters from pictures taken with a mobile camera phone. We might choose to implement that image recognition directly on the mobile device, and by avoiding sending pictures to servers elsewhere, we’ll improve latency and reliability, and potentially even privacy—though for mobile devices, ML computation is generally battery-expensive.

[2](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch08.html#ch01fn87-marker) _Tail latency_ refers to the longest latencies of the total distribution of latencies observed when querying a model. If we query a model many times and order the latency it takes to get a response from shortest to longest, we might find a distribution for which the median response time is quite fast. But in some cases we have a long tail of much, much longer responses. This is the tail, and the durations are the tail latencies. See [“The Tail at Scale”](https://research.google/pubs/pub40801) by Jeffrey Dean and Luiz André Barroso for more.

[3](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch08.html#ch01fn88-marker) Think millions or billions of individual arithmetic operations for one prediction from a deep neural network.

[4](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch08.html#ch01fn89-marker) While GPUs are by far the most common type of ML hardware accelerator used in training and serving, many other specialized accelerator architectures are designed specifically for ML. Companies like Google, Apple, Facebook, Amazon, Qualcomm, and Samsung Electronics all have ML accelerator products or projects. This space is changing rapidly.

[5](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch08.html#ch01fn90-marker) Indeed, GPUs are _so_ good at computation that they are often bottlenecked not on their ability to do the matrix multiplications, but instead on bandwidth for getting data in and out of the chip. Batching requests together to amortize the input and output costs can be an extremely effective strategy, in many cases allowing us to process hundreds of requests with the same wall-clock latency as a single request. The only problem with batching is that we may be slower waiting for enough requests to come in to create a batch of sufficient size, but in environments with high load, this is not usually an issue.

[6](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch08.html#ch01fn91-marker) [gRPC](https://grpc.io/) is an open source RPC system initially developed by Google. Representational State Transfer (REST) is a widely used pattern for APIs that developers follow when they create web APIs.

[7](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch08.html#ch01fn92-marker) Federated learning is an approach that trains a model across multiple disconnected edge devices. Read more at [TensorFlow](https://oreil.ly/dYqeC).

[8](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch08.html#ch01fn93-marker) Resource-oriented architectures (as compared to service-oriented architecture) extend the REST pattern for web API building. A resource is an entity that has a state that can be assigned to a uniform resource locator (URL). See [“An Overview of Resource-Oriented Architectures”](https://oreil.ly/qzVwx) by Joydip Kanjilal for an overview.

[9](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch08.html#ch01fn94-marker) Similarly, service-oriented architecture is an approach whereby an application is decomposed into a series of services. It’s a somewhat overused term that often means different things to different people in the industry (as is reflected in [“Service-Oriented Architecture”](https://oreil.ly/e5GzU) by Cesar de la Torre et al.)

[10](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch08.html#ch01fn95-marker) SLOs are thoroughly introduced in _Site Reliability Engineering: How Google Runs Production Systems_, and even more thoroughly covered in _Implementing Service Level Objectives_ by Alex Hidalgo (O’Reilly, 2020).
