# Chapter 7: LLM for Data Analysis

## 7 LLMs for Data Science

### Join our book community on Discord

[https://packt.link/EarlyAccessCommunity](https://packt.link/EarlyAccessCommunity)

![Qr code Description automatically generated](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file49.png)

This chapter is about how generative AI can automate data science. Generative AI, in particular large language models (LLMs) have the potential to accelerate scientific progress across various domains, especially by providing efficient analysis of research data and aiding in literature review processes. A lot of current approaches that fall within the domain of AutoML can help data scientists increase their productivity and help make the data science more repeatable. I’ll first give an overview over automation in data science and then we’ll discuss how data science is affected by generative AI.Next, we’ll discuss how we can use code generation and tools in different ways to answer questions related to data science. This can come in the form of doing a simulation or of enriching out dataset with additional information. Finally, we put the focus on exploratory analysis of structured datasets. We can set up agents to run SQL or tabular data in Pandas. We’ll see how we can ask questions about the dataset, statistical questions about the data, or ask for visualizations.Throughout the chapter, we’ll work on different approaches to doing data science with LLMs, which you can find in the `data_science` directory in the Github repository for the book at [https://github.com/benman1/generative\_ai\_with\_langchain](https://github.com/benman1/generative\_ai\_with\_langchain)The main sections are:

* Automated data science
* Agents can answer data science questions
* Data exploration with LLMs

Let’s start by discussing how data science can be automated and which parts of it, and how generative AI will impact data science.

### Automated data science

Data science is a field that combines computer science, statistics, and business analytics to extract knowledge and insights from data. Data scientists use a variety of tools and techniques to collect, clean, analyze, and visualize data. They then use this information to help businesses make better decisions.The work of a data scientist can vary depending on the specific role and industry. However, some common tasks that data scientists might perform include:

* Collecting data: Data scientists need to collect data from a variety of sources, such as databases, social media, and sensors.
* Cleaning data: Data scientists need to clean data to remove errors and inconsistencies.
* Analyzing data: Data scientists use a variety of statistical and machine learning techniques to analyze data.
* Visualizing data: Data scientists use data visualizations to communicate insights to stakeholders.
* Building models: Data scientists build models to predict future outcomes or make recommendations.

Data analysis is a subset of data science that focuses on extracting insights from data. Data analysts use a variety of tools and techniques to analyze data, but they typically do not build models.The overlap between data science and data analysis is that both fields involve working with data to extract insights. However, data scientists typically have a more technical skillset than data analysts. Data scientists are also more likely to build models and sometimes deploy models into production. Data scientists sometimes deploy models into production so that they can be used to make decisions in real time, however, we’ll avoid automatic deployment of models in this discussion.Here is a table that summarizes the key differences between data science and data analysis:

| **Feature**        | **Data Science**                        | **Data Analysis**   |
| ------------------ | --------------------------------------- | ------------------- |
| Technical skillset | More technical                          | Less technical      |
| Machine learning   | Yes                                     | No                  |
| Model deployment   | Sometimes                               | No                  |
| Focus              | Extracting insights and building models | Extracting insights |

Figure 7.1: Comparison of Data Science and Data Analysis.

The common denominator between the two is collecting data, cleaning data, analyzing data, visualizing data, all of which fall into the category of extracting insights. Data science, additionally is about training machine learning models and usually it has a stronger focus on statistics. In some cases, depending on the setup in the company and industry practices, deploying models and writing software can be added to the list for data science. Automatic data analysis and data science aims to automate many of the tedious, repetitive tasks involved in working with data. This includes data cleaning, feature engineering, model training, tuning, and deployment. The goal is to make data scientists and analysts more productive by enabling faster iterations and less manual coding for common workflows.A lot of these tasks can be automated to some degree. Some of the tasks for data science are similar to those of a software developer that we talked about in chapter 6, _Developing Software_, namely writing and deploying software although with a narrower focus, on models.Data science platforms such as Weka, H2O, KNIME, RapidMiner, and Alteryx are unified machine learning and analytics engines that can be used for a variety of tasks, including preprocessing of large volumes of data and feature extraction. All of these come with a graphical user interface (GUI), have the capability to integrate 3rd party data source, and write custom plug-ins. KNIME is mostly open-source, however also offers a commercial product called KNIME Server. Apache Spark is a versatile tool that can be used for a variety of tasks involved in data science. It can be used to to clean, transform, extract features, and prepare high-volume data for analysis and also to train and deploy machine learning models, both in streaming scenarios, when it’s about real-time decisions or monitoring events.Further, at its most fundamental, libraries for scientific computing such as NumPy can be serve for all tasks involved in automated data science. Deep learning and machine learning libraries such as TensorFlow, Pytorch, and Scikit-Learn can be used for a variety of tasks beyond creating complex machine learning models, including data preprocessing and feature extraction. Orchestration tools such as Airflow, Kedro, or others can help in all these tasks, and include a lot of integrations with specific tools related to all steps in data science.Several data science tools have generative AI support. In _Chapter 6_, _Developing Software_, we’ve already mentioned GitHub Copilot, but there are others such as the PyCharm AI Assistant, and even more to the point, Jupyter AI, which is a subproject of Project Jupyter that brings generative artificial intelligence to Jupyter notebooks. Jupyter AI allows users to generate code, fix errors, summarize content, and even create entire notebooks using natural language prompts. The tool connects Jupyter with LLMs from various providers, allowing users to choose their preferred model and embedding.Jupyter AI prioritizes responsible AI and data privacy. The underlying prompts, chains, and components are open source, ensuring transparency. It saves metadata about model-generated content, making it easy to track AI-generated code within the workflow. Jupyter AI respects user data privacy and only contacts LLMs when explicitly requested, which is done through LangChain integrations.To use Jupyter AI, users can install the appropriate version for JupyterLab and access it through a chat UI or the magic command interface. The chat interface features Jupyternaut, an AI assistant that can answer questions, explain code, modify code, and identify errors. Users can also generate entire notebooks from text prompts.The software allows users to teach Jupyternaut about local files and interact with LLMs using magic commands in notebook environments. It supports multiple providers and offers customization options for the output format. This screenshot from the documentation shows the chat feature, the Jupyternaut chat:

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file50.png" alt="Figure 7.2: Jupyter AI – Jupyternaut chat." height="684" width="375"><figcaption><p>Figure 7.2: Jupyter AI – Jupyternaut chat.</p></figcaption></figure>

It should be plain to see that having a chat like that at your fingertips to ask questions, create simple functions, or change existing functions can be a boon to data scientists. The benefits of using these tools include improved efficiency, reduced manual effort in tasks like model building or feature selection, enhanced interpretability of models, identification and fixing of data quality issues, integration with other scikit-learn pipelines (pandas\_dq), and overall improvement in the reliability of results.Overall, automated data science can greatly accelerate analytics and ML application development. It allows data scientists to focus on higher value and creative aspects of the process. Democratizing data science for business analysts is also a key motivation behind automating these workflows. In the following sections, we’ll look into these steps in turn, and we’ll discuss automating them, and we’ll highlight how generative AI can make a contribution to improving the workflow and create efficiency gains.

#### Data collection

Automated data collection is the process of collecting data without human intervention. Automatic data collection can be a valuable tool for businesses. It can help businesses to collect data more quickly and efficiently, and it can free up human resources to focus on other tasks. Generally, in the context of data science or analytics we refer to ETL (Extract, Transform, and Load) as the process that not only takes data from one or more sources (the data collection), but also prepares it for specific use cases. The ETL process typically follows these steps:

1. Extract: The data is extracted from the source systems. This can be done using a variety of methods, such as web scraping, API integration, or database queries.
2. Transform: The data is transformed into a format that can be used by the data warehouse or data lake. This may involve cleaning the data, removing duplicates, and standardizing the data format.
3. Load: The data is loaded into the data warehouse or data lake. This can be done using a variety of methods, such as bulk loading or incremental loading.

ETL and data collection can be done using a variety of tools and techniques, such as:

* Web scraping: Web scraping is the process of extracting data from websites. This can be done using a variety of tools, such as Beautiful Soup, Scrapy, Octoparse, .
* APIs (Application Programming Interfaces): These are a way for software applications to talk to each other. Businesses can use APIs to collect data from other companies without having to build their own systems.
* Query languages: Any database can serve as data source including of the SQL (Structured Query Language) or the no-SQL variety.
* Machine learning: Machine learning can be used to automate the process of data collection. For example, businesses can use machine learning to identify patterns in data and then collect data based on those patterns.

Once the data has been collected, it can be processed to prepare it for use in a data warehouse or data lake. The ETL process will typically clean the data, remove duplicates, and standardize the data format. The data will then be loaded into the data warehouse or data lake, where it can be used by data analysts or data scientists to gain insights into the business.There are many ETL tools including commercial ones such as AWS glue, Google Dataflow, Amazon Simple Workflow Service (SWF), dbt, Fivetran, Microsoft SSIS, IBM InfoSphere DataStage, Talend Open Studio or open-source tools such as Airflow, Kafka, and Spark. In Python are many more tools, too many to list all, such as Pandas for data extraction and processing, and even celery and joblib, which can serve as ETL orchestration tools. In LangChain, there’s an integration with Zapier, which is an automation tool that can be used to connect different applications and services. This can be used to automate the process of data collection from a variety of sources.Here are some of the benefits of using automated ETL tools:

* Increased accuracy: Automated ETL tools can help to improve the accuracy of the data extraction, transformation, and loading process. This is because the tools can be programmed to follow a set of rules and procedures, which can help to reduce human error.
* Reduced time to market: Automated ETL tools can help to reduce the time it takes to get data into a data warehouse or data lake. This is because the tools can automate the repetitive tasks involved in the ETL process, such as data extraction and loading.
* Improved scalability: Automated ETL tools can help to improve the scalability of the ETL process. This is because the tools can be used to process large volumes of data, and they can be easily scaled up or down to meet the needs of the business.
* Improved compliance: Automated ETL tools can help to improve compliance with regulations such as GDPR and CCPA. This is because the tools can be programmed to follow a set of rules and procedures, which can help to ensure that data is processed in a compliant manner.

The best tool for automatic data collection will depend on the specific needs of the business. Businesses should consider the type of data they need to collect, the volume of data they need to collect, and the budget they have available.

#### Visualization and EDA

Automated EDA (Exploratory Data Analysis) and visualization refer to the process of using software tools and algorithms to automatically analyze and visualize data, without significant manual intervention. Traditional EDA involves manually exploring and summarizing data to understand its various aspects before performing machine learning or deep learning tasks. It helps in identifying patterns, detecting inconsistencies, testing assumptions, and gaining insights. However, with the advent of large datasets and the need for efficient analysis, automated EDA has become important.Automated EDA and visualization tools provide several benefits. They can speed up the data analysis process, reducing the time spent on tasks like data cleaning, handling missing values, outlier detection, and feature engineering. These tools also enable a more efficient exploration of complex datasets by generating interactive visualizations that provide a comprehensive overview of the data.Several tools are available for automated EDA and visualization, including:

* D-Tale: A library that facilitates easy visualization of pandas data frames. It supports interactive plots, 3D plots, heatmaps, correlation analysis, custom column creation.
* ydata-profiling (previously pandas profiling): An open-source library that generates interactive HTML reports (`ProfileReport`) summarizing different aspects of the dataset such as missing values statistics, variable types distribution profiles, correlations between variables. It works with Pandas as well as Spark DataFrames.
* Sweetviz: A Python library that provides visualization capabilities for exploratory data analysis with minimal code required. It allows for comparisons between variables or datasets.
* Autoviz: This library automatically generates visualizations for datasets regardless of their size with just a few lines of code.
* DataPrep: With just a few lines you can collect data from common data sources do EDA and data cleaning such as standardization of column names or entries.
* Lux: Displays a set of visualizations with interesting trends and patterns in the dataset displayed via an interactive widget that users can quickly browse in order to gain insights.

The use of generative AI in data visualization adds another dimension to automated EDA by allowing algorithms to generate new visualizations based on existing ones or specific user prompts. Generative AI has the potential to enhance creativity by automating part of the design process while maintaining human control over the final output.Overall, automated EDA and visualization tools offer significant advantages in terms of time efficiency, comprehensive analysis, and the generation of meaningful visual representations of data. Generative AI has the potential to revolutionize data visualization in a number of ways. For example, it can be used to create more realistic and engaging visualizations, which can help in business communication and to communicate data more effectively to stakeholders to provide each user with the information they need to gain insights and make informed decisions.Generative AI can enhance and extend the creation that traditional tools are capable of by making personalized visualizations tailored to the individual needs of each user. Further, Generative AI can be used to create interactive visualizations that allow users to explore data in new and innovative ways.

#### Pre-processing and feature extraction

Automated data preprocessing is the process of automating the tasks involved in data preprocessing. This can include tasks such as data cleaning, data integration, data transformation, and feature extraction. It is related to the transform step in ETL, so there’s a lot of overlap in tools and techniques.Data preprocessing is important because it ensures that data is in a format that can be used by data analysts and machine learning models. This includes removing errors and inconsistencies from the data, as well as converting it into a format that is compatible with the analytical tools that will be used.Manually engineering features can be tedious and time consuming, so automating this process is valuable. Recently, several open source Python libraries have emerged to help auto-generate useful features from raw data as we’ll see.Featuretools offers a general-purpose framework that can synthesize many new features from transactional and relational data. It integrates across multiple ML frameworks making it flexible. Feature Engine provides a simpler set of transformers focused on common data transformations like handling missing data. For optimizing feature engineering specifically for tree-based models, ta from Microsoft shows strong performance through techniques like automatic crossing.AutoGluon Features applies neural network style automatic feature generation and selection to boost model accuracy. It is tightly integrated with the AutoGluon autoML capabilities. Finally, TensorFlow Transform operates directly on Tensorflow pipelines to prepare data for models during training. It has progressed rapidly with diverse open source options now available. Featuretools provides the most automation and flexibility while integrating across ML frameworks. For tabular data, ta and Feature Engine offer easy-to-use transformers optimized for different models. Tf.transform is ideal for TensorFlow users, while AutoGluon specializes in the Apache MXNet deep learning software framework.As for time series data, Tsfel is a library that extracts features from time series data. It allows users to specify the window size for feature extraction and can analyze the temporal complexity of the features. It computes statistical, spectral, and temporal features.On the other hand, tsflex is a time series feature extraction toolkit that is flexible and efficient for sequence data. It makes few assumptions about the data structure and can handle missing data and unequal lengths. It also computes rolling features.Both libraries offer more modern options for automated time series feature engineering compared to tsfresh. Tsfel is more full-featured, while tsflex emphasizes flexibility on complex sequence data.There are a few tools that focus on data quality for machine learning and data science that come with data profiling and automatic data transformations. For example, the pandas-dq library, which can be integrated with scikit-learn pipelines, offers a range of useful features for data profiling, train-test comparison, data cleaning, data imputation (filling missing values), and data transformation (e.g., skewness correction). It helps improve the quality of data analysis by addressing potential issues before modeling.More focused on improved reliability through early identification of potential issues or errors are tools like Great Expectations and Deequ. Great Expectations is a tool for validating, documenting, and profiling data to maintain quality and improve communication between teams. It allows users to assert expectations on the data, catch issues quickly through unit tests for data, create documentation and reports based on expectations. Deequ is built on top of Apache Spark for defining unit tests for data quality in large datasets. It lets users explicitly state assumptions about the dataset and verifies them through checks or constraints on attributes. By ensuring adherence to these assumptions, it prevents crashes or wrong outputs in downstream applications.All these libraries allow data scientists to shorten feature preparation and expand the feature space to improve model quality. Automated feature engineering is becoming essential to leveraging the full power of ML algorithms on complex real-world data.

#### AutoML

Automated Machine Learning (AutoML) frameworks are tools that automate the process of machine learning model development. They can be used to automate tasks such as data cleaning, feature selection, model training, and hyperparameter tuning. This can save data scientists a lot of time and effort, and it can also help to improve the quality of machine learning models.The basic idea of AutoML is illustrated in this diagram from the Github repo of the mljar autoML library (source: https://github.com/mljar/mljar-supervised):

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file51.png" alt="Figure 7.3: How AutoML works." height="1587" width="2646"><figcaption><p>Figure 7.3: How AutoML works.</p></figcaption></figure>

Load some data, try different combinations of preprocessing methods, ML algorithms, training and model parameters, create explanations, compare results in a leaderboard together with visualizations. The main value proposition of an AutoML framework is the ease-of-use of and an increased developer productivity in finding a machine learning model, understanding it, and getting it to production. AutoML tools have been around for a long time. One of the first broader frameworks was AutoWeka, written in Java, and it was designed to automate the process of machine learning model development for tabular data in the Weka (Waikato Environment for Knowledge Analysis) machine learning suite, which is developed at the University of Waikato.In the years since AutoWeka was released, there have been many other AutoML frameworks developed. Some of the most popular AutoML frameworks today include auto-sklearn, autokeras, NASLib, Auto-Pytorch, tpot, optuna, autogluon, and ray (tune). These frameworks are written in a variety of programming languages, and they support a variety of machine learning tasks.Recent advances in autoML and neural architecture search have allowed tools to automate large parts of the machine learning pipeline. Leading solutions like Google AutoML, Azure AutoML, and H2O AutoML/Driverless AI can automatically handle data prep, feature engineering, model selection, hyperparameter tuning, and deployment based on the dataset and problem type. These make machine learning more accessible to non-experts as well.Current autoML solutions can handle structured data like tables and time series data very effectively. They can automatically generate relevant features, select algorithms like tree ensembles, neural networks or SVMs, and tune hyperparameters. Performance is often on par or better than manual process due to massive hyperparameter search. AutoML for unstructured data like images, video and audio is also advancing rapidly with neural architecture search techniques.Open source libraries like AutoKeras, AutoGluon, and AutoSklearn provide accessible autoML capabilities as well. However, most autoML tools still require some coding and data science expertise. Fully automating data science remains challenging and autoML does have limitations in flexibility and controllability. But rapid progress is being made with more user-friendly and performant solutions coming to market.Here’s a tabular summary of frameworks:

| **Framework**           | **Language** | **ML Frameworks**         | **First Release** | **Key Features**                                                                | **Data Types**                    | **Maintainer**                     | **Github stars** |
| ----------------------- | ------------ | ------------------------- | ----------------- | ------------------------------------------------------------------------------- | --------------------------------- | ---------------------------------- | ---------------- |
| Auto-Keras              | Python       | Keras                     | 2017              | Neural architecture search, easy to use                                         | Images, text, tabular             | Keras Team (DATA Lab, Texas A\&M)  | 8896             |
| Auto-PyTorch            | Python       | PyTorch                   | 2019              | Neural architecture search, hyperparameter tuning                               | Tabular, text, image, time series | AutoML Group, Univ. of Freiburg    | 2105             |
| Auto-Sklearn            | Python       | Scikit-learn              | 2015              | Automated scikit-learn workflows                                                | Tabular                           | AutoML Group, Univ. of Freiburg    | 7077             |
| Auto-WEKA               | Java\*       | WEKA                      | 2012              | Bayesian optimization                                                           | Tabular                           | University of British Columbia     | 315              |
| AutoGluon               | Python       | MXNet, PyTorch            | 2019              | Optimized for deep learning                                                     | Text, image, tabular              | Amazon                             | 6032             |
| AWS SageMaker Autopilot | Python       | XGBoost, sklearn          | 2020              | Cloud-based, simple                                                             | Tabular                           | Amazon                             | -                |
| Azure AutoML            | Python       | Scikit-learn, PyTorch     | 2018              | Explainable models                                                              | Tabular                           | Microsoft                          | -                |
| DataRobot               | Python, R    | Multiple                  | 2012              | Monitoring, explainability                                                      | Text, image, tabular              | DataRobot                          | -                |
| Google AutoML           | Python       | TensorFlow                | 2018              | Easy to use, cloud-based                                                        | Text, image, video, tabular       | Google                             | -                |
| H2O AutoML              | Python, R    | XGBoost, GBMs             | 2017              | Automatic workflow, ensembling                                                  | Tabular, time series, images      | h2o.ai                             | 6430             |
| hyperopt-sklearn        | Python       | Scikit-learn              | 2014              | Hyperparameter tuning                                                           | Tabular                           | Hyperopt team                      | 1451             |
| Ludwig                  | Python       | Transformers/Pytorch      | 2019              | Low-code framework for building and tuning custom LLMs and deep neural networks | Multiple                          | Linux Foundation                   | 9083             |
| MLJar                   | Python       | Multiple                  | 2019              | Explainable, customizable                                                       | Tabular                           | MLJar                              | 2714             |
| NASLib                  | Python       | PyTorch, TensorFlow/Keras | 2020              | Neural architecture search                                                      | Images, text                      | AutoML Group, Univ. of Freiburg    | 421              |
| Optuna                  | Python       | Agnostic                  | 2019              | Hyperparameter tuning                                                           | Agnostic                          | Preferred Networks Inc             | 8456             |
| Ray (Tune)              | Python       | Agnostic                  | 2018              | Distributed hyperparameter tuning; Accelerating ML workloads                    | Agnostic                          | University of California, Berkeley | 26906            |
| TPOT                    | Python       | Scikit-learn, XGBoost     | 2016              | Genetic programming, pipelines                                                  | Tabular                           | Epistasis Lab, Penn State          | 9180             |
| TransmogrifAI           | Scala        | Spark ML                  | 2018              | AutoML on Spark                                                                 | Text, tabular                     | Salesforce                         | 2203             |

Figure 7.4: Comparison of open-source AutoML frameworks. Weka can be accessed from Python as pyautoweka. Stars for Ray Tune and H2O concern the whole project rather than only the automl part. The H2O commercial product related to AutoML is Driverless AI. Most projects are maintained by a community of contributors not affiliated with just one company

I have only included the biggest frameworks, libraries or products – omitting a few. Although the focus is on open-source frameworks in Python, I’ve included a few big commercial products. Github stars aim to show the popularity of the framework – they are not relevant to proprietary products. Pycaret is another big project (7562 stars) that gives the option to train several models simultaneously and compare them with relatively low amounts of code. Projects like Nixtla’s Statsforecast and MLForecast, or Darts have similar functionality specific to time series data.Libraries like Auto-ViML and deep-autoviml handle various types of variables and are built on scikit-learn and keras, respectively. They aim to make it easy for both novices and experts to experiment with different kinds of models and deep learning. However, users are advised to exercise their own judgement for accurate and explainable results.Important features of AutoML frameworks include the following:

* Deployment: Some solutions, especially those in the cloud, can be directly deployed to production. Others export to tensorflow or other formats.
* Types of data: Most solutions focus on tabular datasets; deep learning automl frameworks often work with different types of data. For example, autogluon facilitates rapid comparison and prototyping of ml solutions for images, text, time series in addition to tabular data. A few that focus on hyperparameter optimization such as optuna and ray tune, are totally agnostic to the format.
* Explainability: This can be very important depending on the industry, related to regulation (for example, healthcare or insurance) or reliability (finance). For a few solutions, this is a unique selling point.
* Monitoring: After deployment, the model performance can deteriorate (drift). A few providers provide monitoring of performance.
* Accessibility: Some providers require coding or at least basic data science understanding, others are turnkey solutions that require very little to no code. Typically, low and no-code solutions are less customizable.
* Open Source: The advantage of open-source platforms is that they are fully transparent about the implementation and the availability of methods and their parameters, and that they are fully extensible.
* Transfer Learning: This capability means being able to extend or customize existing foundation models.

There is a lot more to cover here that would go beyond the scope of this chapter such as the number of available methods. Less-well supported are features such as self-supervised learning, reinforcement learning, or generative image and audio models. For deep learning, a few libraries focus on the backend being specialized in Tensorflow, Pytorch, or MXNet. Auto-Keras, NASLib, and Ludwig have broader support, especially because they work with Keras. Starting with version 3.0, which is scheduled for release in fall 2023, Keras supports the three major backends TensorFlow, JAX, and PyTorch. Sklearn has its own hyperparameter optimization tools such as grid search, random search, successive halving. More specialized libraries such as auto-sklearn and hyperopt-sklearn go beyond this by offering methods for Bayesian Optimization. Optuna can integrate with a broad variety of ML frameworks such as AllenNLP, Catalyst, Catboost, Chainer, FastAI, Keras, LightGBM, MXNet, PyTorch, PyTorch, Ignite, PyTorch, Lightning, TensorFlow, and XGBoost. Ray Tune comes with its own integrations among which is optuna. Both of them come with cutting edge parameter optimization algorithms and mechanisms for scaling (distributed training).In addition to the features listed above, some of these frameworks can automatically perform feature engineering tasks, such as data cleaning and feature selection, for example removing highly correlated features, and generating performance results graphically. Each of the listed tools has their own implementations for each step of the process such as feature selection and feature transformations – what differs is the extent to which this is automated. More specifically, the advantages of using AutoML frameworks include:

* Time savings: AutoML frameworks can save data scientists a lot of time by automating the process of machine learning model development.
* Improved accuracy: AutoML frameworks can help to improve the accuracy of machine learning models by automating the process of hyperparameter tuning.
* Increased accessibility: AutoML frameworks make machine learning more accessible to people who do not have a lot of experience with machine learning.

However, there are also some disadvantages to using AutoML frameworks:

* Black box: AutoML frameworks can be "black boxes," meaning that it can be difficult to understand how they work. This can make it difficult to debug problems with AutoML models.
* Limited flexibility: AutoML frameworks can be limited in terms of the types of machine learning tasks that they can automate.

A lot of the above tools have at least some kind of automatic feature engineering or preprocessing functionality, however, there are a few more specialized tools for this.

#### The impact of generative models

Generative AI and LLMs like GPT-3 have brought about significant changes to the field of data science and analysis. These models, particularly LLMs have the potential to revolutionize all steps involved in data science in a number of ways offering exciting opportunities for researchers and analysts. Generative AI models, such as ChatGPT, have the ability to understand and generate human-like responses, making them valuable tools for enhancing research productivity.Generative AI can play a crucial role in analyzing and interpreting research data. These models can assist in data exploration, uncover hidden patterns or correlations, and provide insights that may not be apparent through traditional methods. By automating certain aspects of data analysis, generative AI saves time and resources, allowing researchers to focus on higher-level tasks.Another area where generative AI can benefit researchers is in performing literature reviews and identifying research gaps. ChatGPT and similar models can summarize vast amounts of information from academic papers or articles, providing a concise overview of existing knowledge. This helps researchers identify gaps in the literature and guide their own investigations more efficiently. We’ve looked at this aspect of using generative AI models in _Chapter 4_, _Question Answering_.Other use cases for generative AI can be:

* Automatically generate synthetic data: Generative AI can be used to automatically generate synthetic data that can be used to train machine learning models. This can be helpful for businesses that do not have access to large amounts of real-world data.
* Identify patterns in data: Generative AI can be used to identify patterns in data that would not be visible to human analysts. This can be helpful for businesses that are looking to gain new insights from their data.
* Create new features from existing data: Generative AI can be used to create new features from existing data. This can be helpful for businesses that are looking to improve the accuracy of their machine learning models.

According to recent reports by the likes of McKinsey and KPMG, the consequences of AI relate to what data scientists will work on, how they will work, and who can work on data science tasks. The main areas of key impact include:

* Democratization of AI: Generative models allow many more people to leverage AI by generating text, code, and data from simple prompts. This expands the use of AI beyond data scientists.
* Increased productivity: By auto-generating code, data, and text, generative AI can accelerate development and analysis workflows. This allows data scientists and analysts to focus on higher-value tasks.
* Innovation in data science: Generative AI is bringing about is the ability to explore data in new and more creative ways, and generate new hypotheses and insights that would not have been possible with traditional methods
* Disruption of industries: New applications of generative AI could disrupt industries by automating tasks or enhancing products and services. Data teams will need to identify high-impact use cases.
* Limitations remain: Current models still have accuracy limitations, bias issues, and lack of controllability. Data experts are needed to oversee responsible development.
* Importance of governance: Rigorous governance over development and ethical use of generative AI models will be critical to maintaining stakeholder trust.
* Need for partnerships - Companies will need to build ecosystems with partners, communities and platform providers to effectively leverage generative AI capabilities.
* Changes to data science skills - Demand may shift from coding expertise to abilities in data governance, ethics, translating business problems, and overseeing AI systems.

Regarding democratization and innovation of data science, more specifically, generative AI is also having an impact on the way that data is visualized. In the past, data visualizations were often static and two-dimensional. However, generative AI can be used to create interactive and three-dimensional visualizations that can help to make data more accessible and understandable. This is making it easier for people to understand and interpret data, which can lead to better decision-making.Again, one of the biggest changes that generative AI is bringing about is the democratization of data science. In the past, data science was a very specialized field that required a deep understanding of statistics and machine learning. However, generative AI is making it possible for people with less technical expertise to create and use data models. This is opening up the field of data science to a much wider range of people.LLMs and generative AI can play a crucial role in automated data science by offering several benefits:

* Natural Language Interaction: LLMs allow for natural language interaction, enabling users to communicate with the model using plain English or other languages. This makes it easier for non-technical users to interact with and explore the data using everyday language, without requiring expertise in coding or data analysis.
* Code Generation: Generative AI can automatically generate code snippets to perform specific analysis tasks during EDA. For example, it can generate code to retrieve data (for example, SQL) clean data, handle missing values, or create visualizations (for example in Python). This feature saves time and reduces the need for manual coding.
* Automated Report Generation: LLMs can generate automated reports summarizing the key findings of EDA. These reports provide insights into various aspects of the dataset such as statistical summary, correlation analysis, feature importance, etc., making it easier for users to understand and present their findings.
* Data Exploration and Visualization: Generative AI algorithms can explore large datasets comprehensively and generate visualizations that reveal underlying patterns, relationships between variables, outliers or anomalies in the data automatically. This helps users gain a holistic understanding of the dataset without manually creating each visualization.

Further, we could think that generative AI algorithms should be able to learn from user interactions and adapt their recommendations based on individual preferences or past behaviors. They improve over time through continuous adaptive learning and user feedback, providing more personalized and useful insights during automated EDA.Finally, generative AI models can identify errors or anomalies in the data during EDA by learning patterns from existing datasets (Intelligent Error Identification). They can detect inconsistencies and highlight potential issues quickly and accurately.Overall, LLMs and generative AI can enhance automated EDA by simplifying user interaction, generating code snippets, identifying errors/ anomalies efficiently, automating report generation, facilitating comprehensive data exploration, visualization creation, and adapting to user preferences for more effective analysis of large and complex datasets.However, while these models offer immense potential to enhance research and aiding in literature review processes, they should not be treated as infallible sources. As we’ve seen earlier, LLMs work by analogy and struggle with reasoning and math. Their strength is creativity, not accuracy, and therefore, researchers must exercise critical thinking and ensure that the outputs generated by these models are accurate, unbiased, and aligned with rigorous scientific standards.One notable example is Microsoft's Fabric, which incorporates a chat interface powered by generative AI. This allows users to ask data-related questions using natural language and receive instant answers without having to wait in a data request queue. By leveraging LLMs like OpenAI models, Fabric enables real-time access to valuable insights.Fabric stands out among other analytics products due to its comprehensive approach. It addresses various aspects of an organization's analytics needs and provides role-specific experiences for different teams involved in the analytics process, such as data engineers, warehousing professionals, scientists, analysts, and business users.With the integration of Azure OpenAI Service at every layer, Fabric harnesses generative AI's power to unlock the full potential of data. Features like Copilot in Microsoft Fabric provide conversational language experiences, allowing users to create dataflows, generate code or entire functions, build machine learning models, visualize results, and even develop custom conversational language experiences.Anecdotally, ChatGPT is (and Fabric in extension) often produces incorrect SQL queries. This is fine when used by analysts who can check the validity of the output, but a total disaster as a self-service analytics tool for non-technical business users. Therefore, organizations must ensure that they have reliable data pipelines in place and employ data quality management practices while using Fabric for analysis.While the possibilities of generative AI in data analytics are promising, caution must be exercised. The reliability and accuracy of LLMs should be verified using first-principled reasoning and rigorous analysis. While these models have shown their potential in ad-hoc analysis, idea generation during research, and summarizing complex analyses, they may not always be suitable for self-service analytical tools for non-technical users due to the need for validation by domain experts.Let’s start to use agents to run code or call other tools to answer questions!

### Agents can answer data science questions

As we’ve seen with Jupyter AI (Jupyternaut chat) – and in chapter 6, _Developing Software_ – there’s a lot of potential to increase efficiency creating and creating software with generative AI (code LLMs). This is a good starting point for the practical part of this chapter as we look into the use of generative AI in data science.We’ve already seen different agents with tools before. For example, the LLMMathChain can execute Python to answer math questions as illustrated here:

```
from langchain import OpenAI, LLMMathChain
llm = OpenAI(temperature=0)
llm_math = LLMMathChain.from_llm(llm, verbose=True)
llm_math.run("What is 2 raised to the 10th power?")
```

While this is useful to extract information and feed it back, it’s less obvious to see how to plug this into a traditional EDA process. Similarly, the CPAL (`CPALChain`) and PAL (`PALChain`) chains can answer more complex reasoning questions while keeping hallucinations in check, but it’s hard to come up with real-life use cases for them.With the `PythonREPLTool` we can create simple visualizations of toy data or train with synthetic data, which can be nice for illustration or bootstrapping of a project. This is an example from the LangChain documentation:

```
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.llms.openai import OpenAI
from langchain.agents.agent_types import AgentType
agent_executor = create_python_agent(
    llm=OpenAI(temperature=0, max_tokens=1000),
    tool=PythonREPLTool(),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)
agent_executor.run(
    """Understand, write a single neuron neural network in PyTorch.
Take synthetic data for y=2x. Train for 1000 epochs and print every 100 epochs.
Return prediction for x = 5"""
)
```

Please note that this should be executed with caution since the Python code is executed directly on the machine without any safeguards in place. This actually works, and creates a dataset, trains a model, and we get a prediction back:

```
Entering new AgentExecutor chain...
I need to write a neural network in PyTorch and train it on the given data
Action: Python_REPL
Action Input: 
import torch
model = torch.nn.Sequential(
    torch.nn.Linear(1, 1)
)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# Define the data
x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0], [8.0]])
for epoch in range(1000):  # Train the model
    y_pred = model(x_data)
    loss = loss_fn(y_pred, y_data)
    if (epoch+1) % 100 == 0:
        print(f'Epoch {epoch+1}: {loss.item():.4f}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
# Make a prediction
x_pred = torch.tensor([[5.0]])
y_pred = model(x_pred)
Observation: Epoch 100: 0.0043
Epoch 200: 0.0023
Epoch 300: 0.0013
Epoch 400: 0.0007
Epoch 500: 0.0004
Epoch 600: 0.0002
Epoch 700: 0.0001
Epoch 800: 0.0001
Epoch 900: 0.0000
Epoch 1000: 0.0000
Thought: I now know the final answer
Final Answer: The prediction for x = 5 is y = 10.00.
```

Again, this is very cool, but it’s hard to see how that would scale up without more serious engineering similar to what we did in chapter 6, _Developing Software_.LLMs and tools can be useful if we want to enrich our data with category or geographic information. For example, if our company offers flights from Tokyo, and we want to know the distances of our customers from Tokyo, we can use Wolfram Alpha as a tool. Here’s a simplistic example:

```
from langchain.agents import load_tools, initialize_agent
from langchain.llms import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
llm = OpenAI(temperature=0)
tools = load_tools(['wolfram-alpha'])
memory = ConversationBufferMemory(memory_key="chat_history")
agent = initialize_agent(tools, llm, agent="conversational-react-description", memory=memory, verbose=True)
agent.run(
    """How far are these cities to Tokyo?
* New York City
* Madrid, Spain
* Berlin
""")
```

Please make sure you’ve set the OPENAI\_API\_KEY and WOLFRAM\_ALPHA\_APPID environment variables as discussed in chapter 3, _Getting Started with LangChain_. Here’s the output:

```
> Entering new AgentExecutor chain...
AI: The distance from New York City to Tokyo is 6760 miles. The distance from Madrid, Spain to Tokyo is 8,845 miles. The distance from Berlin, Germany to Tokyo is 6,845 miles.
> Finished chain.
'
The distance from New York City to Tokyo is 6760 miles. The distance from Madrid, Spain to Tokyo is 8,845 miles. The distance from Berlin, Germany to Tokyo is 6,845 miles.
```

Now, a lot of these questions are very simple. However, we can give agents datasets to work with and here’s where it can get very powerful when we connect more tools. Let’s start with asking and answering questions about structured datasets!

### Data exploration with LLMs

Data exploration is a crucial and foundational step in data analysis, allowing researchers to gain a comprehensive understanding of their datasets and uncover significant insights. With the emergence of LLMs like ChatGPT, researchers can harness the power of natural language processing to facilitate data exploration.As we’ve mentioned earlier Generative AI models, such as ChatGPT, have the ability to understand and generate human-like responses, making them valuable tools for enhancing research productivity. Asking our questions in natural language and getting responses in digestible pieces and shape can be a great boost to analysis.LLMs can assist in exploring not only textual data but also other forms of data such as numerical datasets or multimedia content. Researchers can leverage ChatGPT's capabilities to ask questions about statistical trends in numerical datasets or even query visualizations for image classification tasks.Let’s load up a dataset and work with that. We can quickly get a dataset from scikit-learn:

```
from sklearn.datasets import load_iris
df = load_iris(as_frame=True)["data"]
```

The Iris dataset is well-known – it’s a toy dataset, but it will help us illustrate the capabilities of using generative AI for data exploration. We’ll use the DataFrame in the following. We can create a Pandas dataframe agent now and we’ll see how easy it is to get simple stuff done!

```
from langchain.agents import create_pandas_dataframe_agent
from langchain import PromptTemplate
from langchain.llms.openai import OpenAI
PROMPT = (
    "If you do not know the answer, say you don't know.\n"
    "Think step by step.\n"
    "\n"
    "Below is the query.\n"
    "Query: {query}\n"
)
prompt = PromptTemplate(template=PROMPT, input_variables=["query"])
llm = OpenAI()
agent = create_pandas_dataframe_agent(llm, df, verbose=True)
```

I’ve put the instruction for the model to say it doesn’t know when in doubt and thinking step by step, both to reduce hallucinations. Now we can query our agent against the DataFrame:

```
agent.run(prompt.format(query="What's this dataset about?"))
```

We get the answer `'This dataset is about the measurements of some type of flower.`' which is correct. Let’s show how to get a visualization:

```
agent.run(prompt.format(query="Plot each column as a barplot!"))
```

It’s not perfect, but we are getting a nice-looking plot:

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file52.png" alt="Figure 7.5: Iris dataset barplots." height="564" width="1004"><figcaption><p>Figure 7.5: Iris dataset barplots.</p></figcaption></figure>

We can also ask to see the distributions of the columns visually, which will give us this neat plot:

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file53.png" alt="Figure 7.6: Iris dataset boxplots." height="517" width="973"><figcaption><p>Figure 7.6: Iris dataset boxplots.</p></figcaption></figure>

We can request the plot to use other plotting backends such as seaborn, however, please note that these have to be installed. We can also ask more questions about the dataset like which row has the biggest difference between petal length and petal width. We get the answer with the intermediate steps as follows (shortened):

```
df['difference'] = df['petal length (cm)'] - df['petal width (cm)']
df.loc[df['difference'].idxmax()]
Observation: sepal length (cm)    7.7
sepal width (cm)     2.8
petal length (cm)    6.7
petal width (cm)     2.0
difference           4.7
Name: 122, dtype: float64
Thought: I now know the final answer
Final Answer: Row 122 has the biggest difference between petal length and petal width.
```

I think that’s worth a pat on the back, LLM!Next steps could be adding more instructions to the prompt about plotting such about the sizes of plots. It’s a bit harder to implement the same plotting logic in a streamlit app, because we need to use the plotting functionality in corresponding streamlit functions, for example, `st.bar_chart()`, however, this can be done as well. You can find explanations for this on the Streamlit blog (“Building a Streamlit and scikit-learn app with ChatGPT”).What about statistical tests?

```
agent.run(prompt.format(query="Validate the following hypothesis statistically: petal width and petal length come from the same distribution."))
```

We get this response:

```
Thought: I should use a statistical test to answer this question.
Action: python_repl_ast 
Action Input: from scipy.stats import ks_2samp
Observation: 
Thought: I now have the necessary tools to answer this question.
Action: python_repl_ast
Action Input: ks_2samp(df['petal width (cm)'], df['petal length (cm)'])
Observation: KstestResult(statistic=0.6666666666666666, pvalue=6.639808432803654e-32, statistic_location=2.5, statistic_sign=1)
Thought: I now know the final answer
Final Answer: The p-value of 6.639808432803654e-32 indicates that the two variables come from different distributions.
```

'The p-value of 6.639808432803654e-32 indicates that the two variables come from different distributions.'That’s check for statistical test! That’s cool. We can ask fairly complex questions about the dataset with simple prompts in plain English. There’s also the pandas-ai library, which uses LangChain under the hood and provides similar functionality. Here’s an example from the documentation with an example dataset:

```
import pandas as pd
from pandasai import PandasAI
df = pd.DataFrame({
    "country": ["United States", "United Kingdom", "France", "Germany", "Italy", "Spain", "Canada", "Australia", "Japan", "China"],
    "gdp": [19294482071552, 2891615567872, 2411255037952, 3435817336832, 1745433788416, 1181205135360, 1607402389504, 1490967855104, 4380756541440, 14631844184064],
    "happiness_index": [6.94, 7.16, 6.66, 7.07, 6.38, 6.4, 7.23, 7.22, 5.87, 5.12]
})
from pandasai.llm.openai import OpenAI
llm = OpenAI(api_token="YOUR_API_TOKEN")
pandas_ai = PandasAI(llm)
pandas_ai(df, prompt='Which are the 5 happiest countries?') 
```

This will give us the requested result similarly to before when we were using LangChain directly. Please note that pandas-ai is not part of the setup for the book, so you’ll have to install it separately if you want to use it.For data in SQL-databases, we can connect with a `SQLDatabaseChain`. The LangChain documentation shows this example:

```
from langchain.llms import OpenAI
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
db = SQLDatabase.from_uri("sqlite:///../../../../notebooks/Chinook.db")
llm = OpenAI(temperature=0, verbose=True)
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
db_chain.run("How many employees are there?")
```

We are connecting to a database first. Then we can ask questions about the data in natural language. This can also be quite powerful. An LLM will create the queries for us. I would expect this to be particularly useful when we don’t know about the schema of the database.The `SQLDatabaseChain` can also check queries and autocorrect them if the `use_query_checker` option is set.Let’s summarize!

### Summary

In this chapter, we’ve explored the state-of-the-art in automated data analysis and data science. There are quite a few areas, where LLMs can benefit data science, mostly as coding assistants or in data exploration.We’ve started off with an overview over frameworks that cover each step in the data science process such as AutoML methods, and we’ve discussed how LLMs can help us further increasing productivity and making data science and data analysis more accessible, both to stakeholders and to developers or users. We’ve then investigated how code generation and tools, similar to code LLMs _Chapter 6_, _Developing Software_, can help in data science tasks by creating functions or models that we can query, or how we can enrich data using LLMs or third-party tools like Wolfram Alpha.We then had a look at using LLMs in data exploration. In _Chapter 4_, _Question Answering_, we looked at ingesting large amounts of textual data for analysis. In this chapter, we focused on exploratory analysis of structured datasets in SQL or tabular form. In conclusion, AI technology has the potential to revolutionize the way we can analyze data, and ChatGPT plugins or Microsoft Fabric are examples of this. However, at the current state-of-affairs, AI can’t replace data scientists, only help enable them.Let’s see if you remember some of the key takeaways from this chapter!

### Questions

Please have a look to see if you can come up with the answers to these questions from memory. I’d recommend you go back to the corresponding sections of this chapter, if you are unsure about any of them:

1. What’s the difference between data science and data analysis?
2. What steps are involved in data science?
3. Why would we want to automate data science/analysis?
4. What frameworks exist for automating data science tasks and what can they do?
5. How can generative AI help data scientists?
6. What kind of agents and tools can we use to answer simple questions?
7. How can we get an LLM to work with data?
