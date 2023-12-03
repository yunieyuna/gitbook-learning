# Chapter 6: Developing Software with LangChain Coder

## 6 Developing Software

### Join our book community on Discord

[https://packt.link/EarlyAccessCommunity](https://packt.link/EarlyAccessCommunity)

![Qr code Description automatically generated](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file42.png)

While this book is about integrating generative AI and, in particular, large language models (LLMs) into software applications, in this chapter, we’ll talk about how we can leverage LLMs to help in software development. This is a big topic and software development was highlighted in reports by several consultancies such as KPMG and McKinsey as one of the domains impacted most by generative AI.We’ll first discuss how LLMs could help in coding tasks, and we’ll go through a lot of literature as an overview to see how far we have come in automating software engineers. We’ll also discuss a lot of the recent progress and new models. Then, we’ll play around with a few models evaluating the generated code qualitatively. Next, we’ll implement a fully-automated agent for software development tasks. We go through the design choices and show a bit of the results that we got in an agent implementation of only a few lines of Python with LangChain. There are a lot of possible extensions to this approach, which we’ll also go through.Throughout the chapter, we’ll work on different approaches to software development, which you can find in the `software_development` directory in the Github repository for the book at [https://github.com/benman1/generative\_ai\_with\_langchain](https://github.com/benman1/generative\_ai\_with\_langchain)The main sections are:

* Software development and AI
* Writing code with LLMs
* Automated software development

We'll begin the chapter by giving a broad overview over the state-of-the-art of using AI for software development.

### Software development and AI

The emergence of powerful AI systems like ChatGPT has sparked great interest in using AI as a tool to assist software developers. A June 2023 report by KPMG estimated that about 25% of software development tasks could be automated away. A McKinsey report from the same month highlighted software development as a function, where generative AI can have a significant impact in terms of cost reduction and efficiency gain. The idea of utilizing artificial intelligence to aid programming is not new, but has rapidly evolved alongside advances in computing and AI. The two areas are intertwined as we’ll see.Early efforts in language and compiler design the 1950s and 60s sought to make it easier to write software. Data processing languages like **FLOW-MATIC** (also known as: **Business Language version 0**), designed under Grace Hopper at Remington Rand in 1955, generated code from English-like statements. Similarly, programming languages such as **BASIC** (**Beginners’ All-purpose Symbolic Instruction Code**), created at Dartmouth College in 1963, aimed to make it easier to write software in an interpreted environment.Other efforts further simplified and standardized the programming syntax and interfaces. The **flow-based programming** (**FBP**) paradigm, invented by J. Paul Morrison in the early 1970s, allows to define applications as connected black box processes, which exchange data by message passing. Visual low-code or no-code platforms followed in the same mold with popular proponents such as LabVIEW, extensively used for system design in Electronical Engineering, and the KNIME extract, transform, load tool for data science.Some of the earliest efforts to automate coding itself through AI were **expert systems**, which emerged in the 1980s. As a form of narrow AI, they focused on encoding domain knowledge and rules to provide guidance. These would be formulated in a very specific syntax and executed in rule engines. These encoded best practices for programming tasks like debugging, though their usefulness was constrained by the need for meticulous rule-based programming.For software development, from command line editors such as ed (1969), to vim and emacs (1970s), to today’s integrated development environment (IDEs) such as Visual Studio (first released in 1997) and PyCharm (since 2010), these tools have helped developers write code, navigate in complex projects, refactor, get highlighting and setup and run tests. IDE’s also integrated and provide feedback from code validation tools, some of which have been around since the 1970s. Prominently, Lint, written by Stephen Curtis Johnson in 1978 at Bell Labs can flag bugs, stylistic errors and suspicious constructs. Many tools apply formal methods; however, machine learning has been applied including genetic programming and neural network based approaches for at least 20 years. In this chapter, we’ll how far we’ve come with analyzing code using deep neural networks, especially transformers.This brings us to the present day, where models have been trained to produce full or partial programs based on natural language descriptions (in coding assistants or chatbots) or some code inputs (completion).

#### Present day

Researchers at DeepMind published two papers in the journals Nature and Science, respectively, that represent important milestones in using AI to transform foundational computing, in particular using reinforcement learning to discover optimized algorithms. In October 2022, they released algorithms discovered by their model **AlphaTensor** for matrix multiplication problems, which can speed up this essential computation required by deep learning models, but also in many other applications. **AlphaDev** uncovered novel sorting algorithms that were integrated into widely used C++ libraries, improving performance for millions of developers. It also generalized its capabilities, discovering a 30% faster hashing algorithm now used billions of times daily. These discoveries demonstrate AlphaDev's ability to surpass human-refined algorithms and unlock optimizations difficult at higher programming levels.Their model **AlphaCode**, published as a paper in February 2022, showcases an AI-powered coding engine that creates computer programs at a rate comparable to that of an average programmer. They report results on different datasets including HumanEval and others, which we’ll come to in the next section. The DeepMind researchers highlight the large-scale sampling of candidate pool of algorithms and a filtering step to select from it. The model was celebrated as a breakthrough achievement; however, the practicality and scalability of their approach is unclear.Today, new code LLMs such as ChatGPT and Microsoft's Copilot are highly popular generative AI models, with millions of users and significant productivity-boosting capabilities. There are different tasks related to programming that LLMs can tackle such as these:

1. Code completion: This task involves predicting the next code element based on the surrounding code. It is commonly used in integrated development environments (IDEs) to assist developers in writing code.
2. Code summarization/documentation: This task aims to generate a natural language summary or documentation for a given block of source code. This summary helps developers understand the purpose and function of the code without having to read the actual code.
3. Code search: The objective of code search is to find the most relevant code snippets based on a given natural language query. This task involves learning the joint embeddings of the query and code snippets to return the expected ranking order of code snippets. Neural code search is specifically focused on in the experiment mentioned in the text.
4. Bug finding/fixing: AI systems can reduce manual debugging efforts and enhance software reliability and security. Many bugs and vulnerabilities are hard to find for programmers, although there are typical patterns for which code validation tools exist. As an alternative, LLMs can spot problems with a code and (when prompted) correct them. Thus, these systems can reduce manual debugging efforts and help improve software reliability and security.
5. Test generation: Similar to code completion, LLMs can generate unit tests (compare Bei Chen and others, 2022) and other types of tests enhancing the maintainability of a code base.

AI programming assistants combine the interactivity of earlier systems with cutting-edge natural language processing. Developers can query bugs in plain English or describe desired functions, receiving generated code or debugging tips. However, risks remain around code quality, security, and excessive dependence. Striking the right balance of computer augmentation while maintaining human oversight is an ongoing challenge.Let’s look at the current performance of AI systems for coding, particularly code LLMs.

#### Code LLMs

Quite a few AI models have emerged, each with their own strengths and weaknesses, which are continuously competing with each other to improve and deliver better results. This comparison should give an overview over some of the largest and most popular models:

| **Model**                 | **Reads files** | **Runs code** | **Tokens** |
| ------------------------- | --------------- | ------------- | ---------- |
| ChatGPT; GPT 3.5/4        | No              | No            | up to 32k  |
| ChatGPT: Code interpreter | Yes             | Yes           | up to 32k  |
| Claude 2                  | Yes             | No            | 100k       |
| Bard                      | No              | Yes           | 1k         |
| Bing                      | Yes             | No            | 32k        |

Figure 6.1: Public chat interfaces for software development.

While this competition benefits users by providing a wider range of options, it also means that relying solely on ChatGPT may no longer be the optimal choice. Users now face the decision of selecting the most suitable model for each specific task.The latest wave leverages machine learning and neural networks for more flexible intelligence. Powerful pre-trained models like GPT-3 enable context-aware, conversational support. Deep learning approaches also empower bug detection, repair recommendations, automated testing tools, and code search.Microsoft's GitHub Copilot, which is based on OpenAI’s Codex, draws on open source code to suggest full code blocks in real-time. According to a Github report in June 2023, developers accepted the AI assistant’s suggestions about 30 percent of the time, which suggests that the tool can provide useful suggestions, with less experienced developers profiting the most.

> **Codex** is a model, developed by OpenAI. It is capable of parsing natural language and generating code and powers GitHub Copilot. A descendant of the GPT-3 model, it has been fine-tuned on publicly available code from GitHub, 159 gigabytes of Python code from 54 million GitHub repositories, for programming applications.

To illustrate the progress made in creating software, let’s look at quantitative results in a benchmark: the **HumanEval dataset**, introduced in the Codex paper (“_Evaluating Large Language Models Trained on Code_”, 2021) is designed to test the ability of large language models to complete functions based on their signature and docstring. It evaluates the functional correctness of synthesizing programs from docstrings. The dataset includes 164 programming problems that cover various aspects such as language comprehension, algorithms, and simple mathematics. Some of the problems are comparable to simple software interview questions. A common metric on HumanEval is pass@k (pass@1) – this refers to the fraction of correct samples when generating k code samples per problem.This table summarizes the progress of AI models on the HumanEval task (source: Suriya Gunasekar and others, “_Textbooks Are All You Need_”, 2023; [https://arxiv.org/pdf/2306.11644.pdf](https://arxiv.org/pdf/2306.11644.pdf)):

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file43.png" alt="Figure 6.2: Model comparison on coding task benchmarks (HumanEval and MBPP). The performance metrics are self-reported. This table only includes models as opposed to other approaches, for example reasoning strategies. Llama2’s self-reported performance on HumanEval is 29.9%." height="934" width="1476"><figcaption><p>Figure 6.2: Model comparison on coding task benchmarks (HumanEval and MBPP). The performance metrics are self-reported. This table only includes models as opposed to other approaches, for example reasoning strategies. Llama2’s self-reported performance on HumanEval is 29.9%.</p></figcaption></figure>

Please note that the data used in training most LLM models includes some amount of source code. For example, The Pile dataset, which was curated by EleutherAI's GPT-Neo for training open-source alternatives of the GPT models, GPT-Neo, includes at least about 11% of code from Github (102.18GB). The Pile was used in training of Meta’s Llama, Yandex's YaLM 100B, and many others.Although, HumanEval has been broadly used as a benchmark for code LLMs, there are a multitude of benchmarks for programming. Here’s an example question and the response from an advanced computer science test given to Codex:

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file44.png" alt="Figure 6.3: A question given in a CS2 exam (left) and the Codex response (source “My AI Wants to Know if This Will Be on the Exam: Testing OpenAI’s Codex on CS2 Programming Exercises” James Finnie-Ansley and others, 2023)." height="426" width="1684"><figcaption><p>Figure 6.3: A question given in a CS2 exam (left) and the Codex response (source “My AI Wants to Know if This Will Be on the Exam: Testing OpenAI’s Codex on CS2 Programming Exercises” James Finnie-Ansley and others, 2023).</p></figcaption></figure>

There are many interesting studies that shed a light on AI’s capability to help software developers or that expand on that capability as summarized in this table:

| **Authors**                         | **Publication Date** | **Conclusions**                                                                                                                                             | **Task**                                            | **Model/Strategy Analyzed**  |
| ----------------------------------- | -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | ---------------------------- |
| Abdullah Al Ishtiaq and others      | April 2021           | Pre-trained language models like BERT can enhance code search through improved semantic understanding.                                                      | Code search                                         | BERT                         |
| Mark Chen et al. (OpenAI)           | July 2021            | Evaluates Codex on code generation, shows potential to advance program synthesis                                                                            | Code generation                                     | Codex                        |
| Ankita Sontakke and others          | March 2022           | Even state-of-the-art models produce poor quality code summaries, indicating they may not understand code.                                                  | Code summarization                                  | Transformer models           |
| Bei Chen et al. (Microsoft)         | July 2022            | CODE-T leverages LLMs to auto-generate test cases, reducing human effort and improving code evaluation. It achieves 65.8% HumanEval pass@1.                 | Code generation, testing                            | CODET                        |
| Eric Zelikman et al. (Stanford)     | December 2022        | Parsel framework enables LLMs to decompose problems and leverage strengths, improving performance on hierarchical reasoning                                 | Program synthesis, planning                         | Codex                        |
| James Finnie-Ansley and others      | January 2023         | Codex outperforms most students on advanced CS2 programming exams.                                                                                          | CS2 programming                                     | Codex                        |
| Yue Liu and others                  | February 2023        | Existing automated code generation has limitations in robustness and reliability.                                                                           | Code generation                                     | 5 NMT models                 |
| Mingyang Geng and others            | February 2023        | A two-stage approach significantly increased effectiveness of code summarization.                                                                           | Code summarization                                  | LLM + reinforcement learning |
| Noah Shinn et al.                   | March 2023           | Reflexion enables trial-and-error learning via verbal reflection, achieving 91% HumanEval pass@1                                                            | Coding, reasoning                                   | Reflexion                    |
| Haoye Tian and others               | April 2023           | ChatGPT shows promise for programming assistance but has limitations in robustness, generalization, and attention span.                                     | Code generation, program repair, code summarization | ChatGPT                      |
| Chuqin Geng and others              | April 2023           | ChatGPT demonstrates impressive capabilities for intro programming education but would only get a B- grade as a student.                                    | Intro functional programming course                 | ChatGPT                      |
| Xinyun Chen and others              | April 2023           | Self-debugging technique enables language models to identify and correct mistakes in generated code.                                                        | Code generation                                     | Self-Debugging               |
| Masum Hasan and others              | April 2023           | Transforming text to an intermediate formal language enabled more efficient app code generation from descriptions.                                          | App code generation                                 | Seq2seq networks             |
| Anis Koubaa and others              | May 2023             | ChatGPT struggles with complex programming problems and is not yet suitable for fully automated programming. It performs much worse than human programmers. | Programming problem solving                         | ChatGPT                      |
| Wei Ma and others                   | May 2023             | ChatGPT understands code syntax but is limited in analyzing dynamic code behavior.                                                                          | Complex code analysis                               | ChatGPT                      |
| Raymond Li et al. (BigCode)         | May 2023             | Introduces 15.5B parameter StarCoder trained on 1 trillion GitHub tokens, achieves 40% HumanEval pass@1                                                     | Code generation, multiple languages                 | StarCoder                    |
| Amos Azaria and others              | June 2023            | ChatGPT has errors and limitations, so outputs should be independently verified. It is best used by experts well-versed in the domain.                      | General capabilities and limitations                | ChatGPT                      |
| Adam Hörnemalm                      | June 2023            | ChatGPT increased efficiency for coding and planning but struggled with communication. Developers wanted more integrated tooling.                           | Software development                                | ChatGPT                      |
| Suriya Gunasekar et al. (Microsoft) | June 2023            | High-quality data enables smaller models to match larger models, altering scaling laws                                                                      | Code generation                                     | Phi-1                        |

Figure 6.2: Literature review of AI for programming tasks. The publication dates refer mostly to the preprint releases.

This is just a small subset of studies, but hopefully this helps to shed a light on some of the developments in the field. Recent research explored how ChatGPT can support programmers’ daily work activities like coding, communication, and planning. Other research describes new models (such as Codex, StarCoder, or Phi-1) or approaches for planning or reasoning to execute these models.Most recently, the paper “_Textbooks Are All You Need_” by Suriya Gunasekar and others at Microsoft Research (2023) introduced phi-1, a 1.3B parameter Transformer-based language model for code. The paper demonstrates how high-quality data can enable smaller models to match larger models for code tasks. The authors start with a 3 TB corpus of code from The Stack and StackOverflow. A large language model (LLM) filters this to select 6B high-quality tokens. Separately, GPT-3.5 generates 1B tokens mimicking textbook style. A small 1.3B parameter model phi-1 is trained on this filtered data. Phi-1 is then fine-tuned on exercises synthesized by GPT-3.5. Results show phi-1 matches or exceeds the performance of models over 10x its size on benchmarks like HumanEval and MBPP.The core conclusion is that high-quality data significantly impacts model performance, potentially altering scaling laws. Instead of brute force scaling, data quality should take precedence. The authors reduce costs by using a smaller LLM to select data, rather than expensive full evaluation. Recursively filtering and retraining on selected data could enable further improvements. It’s important to appreciate that there’s a massive step change in difficulty between short code snippets, where task specifications are translated directly into code and the right API calls have to be issued in a sequence specific to the task, and generating complete programs, which relies on a much deeper understanding and reasoning about the task, the concepts behind, and planning how to accomplish it. However, reasoning strategies can make a big difference for short snippets as well as the paper “_Reflexion: Language Agents with Verbal Reinforcement Learning_” by Noah Shinn and others (2023) shows. The authors propose a framework called Reflexion that enables LLM agents (implemented in LangChain) to learn quickly and efficiently from trial-and-error using verbal reinforcement. The agents verbally reflect on task feedback signals and store their reflective text in an episodic memory buffer, which helps the agents make better decisions in subsequent trials. The authors demonstrate the effectiveness of Reflexion in improving decision-making in diverse tasks such as sequential decision-making, coding, and language reasoning. Reflexion has the potential to outperform previous state-of-the-art models, such as GPT-4, in specific tasks, as shown by its 91% pass@1 accuracy on the HumanEval coding benchmark, which beats any approach previously published including GPT-4’s 67% (as reported by OpenAI).

#### Outlook

Looking forward, advances in multimodal AI may further evolve programming tools. Systems capable of processing code, documentation, images, and more could enable a more natural workflow. The future of AI as a programming partner is bright, but requires thoughtful coordination of human creativity and computer-enabled productivity.While promising, effectively leveraging AI programming assistants requires establishing standards through workshops to create useful prompts and pre-prompts for tasks. Focused training ensures proper validation of generated code. Integrating AI into existing environments rather than stand-alone browsers improves developer experience. As research continues, AI programming assistants present opportunities to increase productivity if thoughtfully implemented with an understanding of limitations. With careful oversight, AI stands to automate tedious tasks, freeing developers to focus on complex programming problems.Legal and ethical concerns arise during the pre-training phase, specifically regarding the rights of content creators whose data is used to train the models. Copyright laws and fair use exemptions are debated in relation to the use of copyrighted data by machine learning models.For example, the Free Software Foundation has raised concerns about potential copyright violations associated with code snippets generated by Copilot and Codex. They question whether training on public repositories falls within fair use, how developers can identify infringing code, the nature of machine learning models as modifiable source code or compilations of training data, and the copyrightability of machine learning models. Further, an internal GitHub study found that a small percentage of generated code contained direct copies from the training data, including incorrect copyright notices. OpenAI recognizes the legal uncertainty surrounding these copyright implications and calls for authoritative resolution. The situation has been compared to the Authors Guild, Inc. v. Google, Inc. court case regarding fair use of text snippets in Google Books. Ideally, we want to be able to do this without relying on a cloud-based service that charges us for a request and that may force us to give up the ownership of our data. However, it’s very convenient to outsource the AI so that all we have to implement are the prompts and the strategies of how to issue calls with our client. Many of the open-source models have made impressive progress on coding tasks, and they have the advantage of full transparency and openness about their development process. Most of them have been trained on code that’s been released under permissive licenses, therefore they are not coming with the same legal concerns as other commercial products.There is a broader impact of these systems beyond coding itself on education and the ecosystem around software development. For example, the emergence of ChatGPT resulted in a massive traffic decline for the popular Stack Overflow question-and-answer forum for programmers. After initially blocking any contributions generated using large language models (LLMs), Stack Overflow launched Overflow AI to bring enhanced search, knowledge ingestion, and other AI features to Stack products. New semantic search is to provide intelligent, conversational results using Stack’s knowledge base.Large language models like Codex and ChatGPT excel in code generation for common problems, but struggle with new ones and long prompts. Most importantly, ChatGPT understands syntax well but has limitations in analyzing dynamic code behavior. In programming education, AI models surpass many students but have a lot of room for improvement, however, they haven’t yet reached the level of being able to replace programmers and human intelligence. Scrutiny is necessary as mistakes can occur, making expert supervision crucial. The potential of AI tools in coding is encouraging but challenges remain in robustness, generalization, attention span, and true semantic understanding. Further development is needed to ensure reliable and transparent AI programming tools that can augment developers, allowing them to write code faster with fewer bugs.In the next section, we’ll see how we can generate software code with LLMs and how we can execute this from within LangChain.

### Writing code with LLMs

Let’s start off by applying a model to write code for us. We can use one of the publicly available models for generating code. I’ve listed a few examples before such as ChatGPT or Bard. From LangChain, we can call OpenAI’s LLMs, PaLM’s code-bison, or a variety of open-source models for example through Replicate, HuggingFace Hub, or – for local models – Llama.cpp, GPT4All, or HuggingFace Pipeline integrations.Let’s have a look at StarCoder This screenshot shows the model in a playground on HuggingFace Spaces:

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file45.png" alt="Figure 6.3: StarCoder Models Playground. We can choose between different models: StarCoder, StarCoderPlus, StarCoderBase.This is available at https://huggingface.co/spaces/bigcode/bigcode-playground" height="1018" width="2842"><figcaption><p>Figure 6.3: StarCoder Models Playground. We can choose between different models: StarCoder, StarCoderPlus, StarCoderBase.This is available at https://huggingface.co/spaces/bigcode/bigcode-playground</p></figcaption></figure>

Please note that, as the description says, the StarCoder model is not instruction-tuned, which means that we can’t give it tasks to do. We can’t say “write a class that…” but we can ask it to complete a text as shown in the screenshot, where we prompt the model with “`# dataclass of customer including an alphanumeric id, a name, and a birthday`” – let’s try this!We can toggle settings for temperature, max new tokens, top-n, and a repetition penalty. For anything non-trivial, we need to get the max new tokens setting up.I am getting this code, which gives us a useful data model for our hypothetical customer:

```
@dataclass(frozen=True)  # frozen means that the class cannot be modified after creation. This is useful for storing data in databases or other persistent storages (e.g., files).
class Customer:
    cust_id : str = field()   # this defines how to create instances from the class; it's called "field" because we're creating fields within our object/instance
    firstname : str = ""      # these are default values if no value was provided when instantiating objects using this class
    lastname : str = ""       # note that there can only ever exist one instance variable per line - you'll get errors otherwise!
    def __post_init__(self):
        """This method runs automatically once upon instantiation"""
        self._validate_cust_id()    # validate input before saving anything into memory
        self.__set_fullname__()     # set full name based on given inputs
    @property        # getter function used by Python interpreter whenever trying to access property directly instead of through. notation e.g.: myobj.firstname vs just myobj.firstname()
    def fullname(self)->str:
        return f"{self.lastname}, {self.firstname}"
    ## private methods below here
    ### validation functions go here
    def _validate_cust_id(self):
        assert len(self.cust_id)==8, \
            ValueError("Customer ID must have exactly eight characters")
        try:
            int(self.cust_id[0]) == True
        except Exception as err:
            raise TypeError('First character of CustID should be numeric') from None
    #### setter functions go here
    def __set_fullname__(self):
        '''sets the customers' full name'''
        names = [x.strip().title() for x in re.split("[\W]+", self.firstname)] + [''] * max((1,-len([y.strip().title() for y in re.split("[\W]+", self.lastname)])) )
        print(names)
        self.firstname=' '.join(filter(None,[n[:3]for n in reversed(names)])+[''])
        self.lastname =''.join([' '*(4-len(l))+' '+ l[-2:]if not l=='I' else '' for l in filter(None,[n[:-2].upper()+','*bool(re.search('[aeiou]',n))+n[-2:].upper() for n in reversed(names)])]).rstrip(',') 
```

This is missing the imports, which would usually come before the comment prompt, so I can’t fault the model for it, we need to add these two lines to the top:

```
import re
from dataclasses import dataclass, field
```

This doesn’t throw an error when we run it as is, which is impressive, however, there are a few things not ideal or even wrong with it when we look a bit more in detail.On the first glance, I personally don’t like the commenting style and the lack of docstrings. There are some `print()` statements left in the code; often this is done for debugging purposes. This shouldn’t be in production code. It’s probably not ideal to have empty strings as defaults for `firstname` and `lastname`. I would expect the customer id to be assigned based on a hashing function – I can’t see why the id should have exactly 8 characters as enforced in the validation function for the property. There are more problems with this code that affect its correctness and readability, but there are a few more crucial problems in the code including attempting to write read-only attributes. `firstname` and `lastname` in the dataclass – `frozen=True` in a dataclass means that attributes can’t be changed after creation. The logic about creating a variable name from `firstname` and `lastname` using regular expressions in a non-standard syntax is strange to say the least – and incorrect. The last name gets dropped in the process. The filters on the reversed name is also highly suspect. I leave it here. This is giving us some good ideas and a structure to start with, but it’s not production code. It doesn’t even work. You can see this code as `customer.py` in the book’s Github repo.Let’s try this again. Perhaps we started off on a bad foot. We started a code snippet in bad syntax expected for beginners and expected code that works. That’s not realistic. Let’s try again, and start with a prompt that is more up to standard:

```
"""Customer data model.
Here we implement our data model for customers, a dataclass with
fields firstname, lastname, customer_id (a hash).
"""
```

We get the following result:

```
@dataclass(frozen=True)  # frozen means that the object cannot be modified after creation!
class Customer:
    """A class representing one single customer."""
    firstname: str = ""
    lastname: str = ""
    id_: int = field(default=-1024, init=False)
    
    def __post_init__(self):
        self.__hash__()
        
    @property    
    def fullname(self)->str: 
        return f"{self.firstname} {self.lastname}"    
            
    def __eq__(self, other:"Customer")->bool:        
        if isinstance(other, type(self)):
            return all([
                getattr(self, attr)==getattr(other,attr) 
                for attr in ["firstname", "lastname"]])            
        else: 
            raise TypeError("Cannot compare different types")        
                
    def __lt__(self, other:"Customer"):  
        try:            
            assert len({type(x) for x in [self, other]})==1             
            
            if not hasattr(self,"fullname"):                
                setattr(self, 'fullname',f'{self.firstname}{self.lastname}')
                
            if not hasattr(other,'fullname'):               
                setattr(other, 'fullname',f'{other.firstname}{other.lastname}')
            return self.fullname < other.fullname
            
        except AssertionError as e:  
            print('Error:',e )
      
    def __repr__(self):    
        return f"Customer('{self.firstname}', '{self.lastname}','{hex(abs(self._customer_id))[-6:]})"   
    
    def __hash__(self):       
        hsh = abs(hash((self.firstname+self.lastname))) % ((sys.maxsize + 1)*2)+ sys.maxsize*3      
        self.id_=hsh         
        
        return hsh
```

It’s good to see that the customer id is created using a hash as expected. There’s some boilerplate code for comparing two different customer objects. However, again there are problems, similar ones to before. First, it’s missing the imports, which I don’t understand after our prompt which would be a module docstring to be found at the start of a file, and the imports would come right after. Second, it’s again attempting to set an attribute after initialization of the class that’s supposed to be frozen. After fixing these two problems, we get our first `Customer()`, then there’s a problem, where the customer id is referenced with the wrong name. After fixing this, we can initialize our customer, look at the attributes, and compare one customer to another. I can see how this approach is starting to become useful for writing boilerplate code. You can see this code as `customer2.py` in the book’s Github repo. Let’s try an instruction-tuned model so we can give it tasks! StarChat, which is based on StarCoder, is available on HuggingFace under [https://huggingface.co/spaces/HuggingFaceH4/starchat-playground](https://huggingface.co/spaces/HuggingFaceH4/starchat-playground) This screenshot shows an example with StarChat:

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file46.png" alt="Figure 6.4: StarChat implementing a function in Python for calculating prime numbers. Please note that not all the code is visible in the screenshot." height="968" width="1900"><figcaption><p>Figure 6.4: StarChat implementing a function in Python for calculating prime numbers. Please note that not all the code is visible in the screenshot.</p></figcaption></figure>

You can find the complete code listing on Github.For this very example that should be well-known in first year Computer Science courses, no imports are needed. The algorithm’s implementation is straightforward. It executes right away and gives the expected result. Within LangChain, we can use the `HuggingFaceHub` integration like this:

```
from langchain import HuggingFaceHub
llm = HuggingFaceHub(
    task="text-generation",
    repo_id="HuggingFaceH4/starchat-alpha",
    model_kwargs={
        "temperature": 0.5,
        "max_length": 1000
    }
)
print(llm(text))
```

As of August 2023, this LangChain integration has some issues with timeouts – hopefully, this is going to get fixed soon. We are not going to use it here.Llama2 is not one of the best models for coding with a pass@1 of about 29 as mentioned earlier, however, we can try it out on HuggingFace chat:

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file47.png" alt="Figure 6.5: HuggingFace chat with Llama2 at https://huggingface.co/chat/" height="884" width="1616"><figcaption><p>Figure 6.5: HuggingFace chat with Llama2 at https://huggingface.co/chat/</p></figcaption></figure>

Please note that this is only the beginning of the output. Llama2 finds a good implementation and the explanations are spot on. Well done, StarCoder and Llama2! – Or perhaps, this was just too easy? There so many ways to get code completion or generation. We can run even try a small local model:

```
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
checkpoint = "Salesforce/codegen-350M-mono"
model = AutoModelForCausalLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=500
)
text = """
def calculate_primes(n):
    \"\"\"Create a list of consecutive integers from 2 up to N.
    For example:
    >>> calculate_primes(20)
    Output: [2, 3, 5, 7, 11, 13, 17, 19]
    \"\"\"
"""
```

CodeGen is a model by Salesforce AI Research. CodeGen 350 Mono performed 12.76% pass@1 in HumanEval. As of July 2023, new versions of CodeGen was released with only 6B parameters that are very competitive, which clocks in at a performance of 26.13%. This last model was trained on the BigQuery dataset containing C, C++, Go, Java, Javascript, and Python, as well as the BigPython dataset, which consists of 5.5TB of Python code. Another interesting, small model is Microsoft’s CodeBERT (2020), a model for program synthesis that has been trained and tested on Ruby, Javascript, Go, Python, Java, and PHP.. Since this model was released before the HumanEval benchmark, the performance statistics for the benchmark were not part of the initial publication.We can now get the output from the pipeline directly like this:

```
completion = pipe(text)
print(completion[0]["generated_text"])
```

Alternatively, we can wrap this pipeline via the LangChain integration:

```
llm = HuggingFacePipeline(pipeline=pipe)
llm(text)
```

This is a bit verbose. There’s also the more convenient constructor method `HuggingFacePipeline.from_model_id()`.I am getting something similar to the StarCoder output. I had to add an `import math`, but the function works. This pipeline we could use in a LangChain agent, however, please note that this model is not instruction-tuned, so you cannot give it tasks, only completion tasks. You can also use all of these models for code embeddings. Other models that have been instruction-tuned and are available for chat, can act as your techie assistant to help with advice, document and explain existing code, or translate code into other programming languages – for the last task they need to have been trained on enough samples in these languages.Please note that the approach taken here is a bit naïve. For example, we could be taking more samples and choose between them such as in the a few of the papers we’ve discussed.Let’s now try to implement a feedback cycle for code development, where we validate and run the code and change it based on feedback.

### Automated software development

We’ll now going to write a fully-automated agent that will write code for us and fix any problems responding to feedback.In LangChain, we have several integrations for code execution like these the `LLMMathChain`, which executes Python code to solve math questions, and the `BashChain` that executes Bash terminal commands, which can help with system administration tasks. However, these are for problem solving with code rather than creating a software.This can, however, work quite well.

```
from langchain.llms.openai import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
llm = OpenAI()
tools = load_tools(["python_repl"])
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
result = agent("What are the prime numbers until 20?")
print(result)
```

We can see how the prime number calculations get processed quite well under the hood between OpenAI’s LLM and the Python interpreter:

```
Entering new AgentExecutor chain...
I need to find a way to check if a number is prime
Action: Python_REPL
Action Input: 
def is_prime(n):
    for i in range(2, n):
        if n % i == 0:
            return False
    return True
Observation: 
Thought: I need to loop through the numbers to check if they are prime
Action: Python_REPL
Action Input: 
prime_numbers = []
for i in range(2, 21):
    if is_prime(i):
        prime_numbers.append(i)
Observation:
Thought: I now know the prime numbers until 20
Final Answer: 2, 3, 5, 7, 11, 13, 17, 19
Finished chain.
{'input': 'What are the prime numbers until 20?', 'output': '2, 3, 5, 7, 11, 13, 17, 19'}
```

We get to the right answer about the prime numbers, however, it’s not entirely clear how this approach would scale for building software products, where it is about modules, abstractions, separation of concerns, and maintainable code. There are a few interesting implementations for this around. The MetaGPT library approaches this with an agent simulation, where different agents represent job roles in a company or IT department:

```
from metagpt.software_company import SoftwareCompany
from metagpt.roles import ProjectManager, ProductManager, Architect, Engineer
async def startup(idea: str, investment: float = 3.0, n_round: int = 5):
    """Run a startup. Be a boss."""
    company = SoftwareCompany()
    company.hire([ProductManager(), Architect(), ProjectManager(), Engineer()])
    company.invest(investment)
    company.start_project(idea)
    await company.run(n_round=n_round)
```

This is a really inspiring use case of an agent simulation. The llm-strategy library by Andreas Kirsch generates code for dataclasses using decorator patterns. Other examples for automatic software development include AutoGPT and BabyGPT although these often tend to get stuck in loops or stop because of failures. A simple planning and feedback loop like this can be implemented in LangChain with a ZeroShot Agent and a planner. The Code-It project by Paolo Rechia and Gpt-Engineer by AntonOsika both follows such as pattern as illustrated in this graph for Code-It:

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781835083468/files/media/file48.jpg" alt="Figure 6.6: Code-It control flow (source: https://github.com/ChuloAI/code-it)." height="586" width="691"><figcaption><p>Figure 6.6: Code-It control flow (source: https://github.com/ChuloAI/code-it).</p></figcaption></figure>

Many of these steps consist of specific prompts that are sent to LLMs with instructions to break down the project or to set up the environment. It’s quite impressive to implement the full feedback loop with all the tools. We can implement a relatively simple feedback loop in different ways in LangChain, for example using `PlanAndExecute` chain, a `ZeroShotAgent`, or `BabyAGI`. Let’s go with `PlanAndExecute`!The main idea is to set up a chain and execute it with the objective of writing a software, like this:

```
llm = OpenAI()
planner = load_chat_planner(llm)
executor = load_agent_executor(
    llm,
    tools,
    verbose=True,
)
agent_executor = PlanAndExecute(
    planner=planner,
    executor=executor,
    verbose=True,
    handle_parsing_errors="Check your output and make sure it conforms!",
    return_intermediate_steps=True
)
agent_executor.run("Write a tetris game in python!")
```

I am omitting the imports here, but you can find the full implementation in the Github repo of the book. The other options can be found there as well. There are a few more pieces to this, but this could already write some code, depending on the instructions that we give. One thing we need is clear instructions for a language model to write Python code in a certain form:

```
DEV_PROMPT = (
    "You are a software engineer who writes Python code given tasks or objectives. "
    "Come up with a python code for this task: {task}"
    "Please use PEP8 syntax and comments!"
)
software_prompt = PromptTemplate.from_template(DEV_PROMPT)
software_llm = LLMChain(
    llm=OpenAI(
        temperature=0,
        max_tokens=1000
    ),
    prompt=software_prompt
)
```

We need to make sure, we take a model that is able to come up with code. We’ve discussed already the models we can choose between for this. I’ve chosen a longer context so we don’t get cut off in the middle of a function, and a low temperature, so it doesn’t get too wild.However, on its own this model wouldn’t be able to store it to file, do anything meaningful with it, and act on the feedback from the execution. We need to come up with code and then test it, and see if it works. Let’s see how we can implement this – that’s in the `tools` argument to the agent executor, let’s see how this is defined!

```
software_dev = PythonDeveloper(llm_chain=software_llm)
code_tool = Tool.from_function(
    func=software_dev.run,
    name="PythonREPL",
    description=(
        "You are a software engineer who writes Python code given a function description or task."
    ),
    args_schema=PythonExecutorInput
)
```

The `PythonDeveloper` class has all the logic about taking tasks given in any form and translating them into code. I won’t go into all the detail here, however, the main idea is here:

```
class PythonDeveloper():
    """Execution environment for Python code."""
    def __init__(
            self,
            llm_chain: Chain,
    ):
        self. llm_chain = llm_chain
    def write_code(self, task: str) -> str:
        return self.llm_chain.run(task)
    def run(
            self,
            task: str,
    ) -> str:
        """Generate and Execute Python code."""
        code = self.write_code(task)
        try:
            return self.execute_code(code, "main.py")
        except Exception as ex:
            return str(ex)
    def execute_code(self, code: str, filename: str) -> str:
        """Execute a python code."""
        try:
            with set_directory(Path(self.path)):
                ns = dict(__file__=filename, __name__="__main__")
                function = compile(code, "<>", "exec")
                with redirect_stdout(io.StringIO()) as f:
                    exec(function, ns)
                    return f.getvalue()
```

I am again leaving out a few pieces. The error handling if very simplistic here. In the implementation on Github, we can distinguish different kinds of errors we are getting, such as these:

* `ModuleNotFoundError`: this means that the code tries to work with packages that we don’t have installed. I’ve implemented logic to install these packages.
* `NameError`: using variable names that don’t exist.
* `SyntaxError`: the code often doesn’t close parentheses or is not even code
* `FileNotFoundError`: the code relies on files that don’t exist. I’ve found a few times that the code tried showing images that were made up.
* `SystemExit`: if something more dramatic happens and Python crashes.

I’ve implemented logic to install packages for `ModuleNotFoundError`, and clearer messages for some of these problems. In the case of missing images, we could add a generative image model to create these. Returning all this as enriched feedback to the code generation, results in increasingly specific output such as this:

```
Write a basic tetris game in Python with no syntax errors, properly closed strings, brackets, parentheses, quotes, commas, colons, semi-colons, and braces, no other potential syntax errors, and including the necessary imports for the game
```

The Python code itself gets compiled and executed in a subdirectory and we take redirect the output of the Python execution in order to capture it – both of this is implemented as Python contexts. Please be cautious with executing code on your system, because some of these approaches are quite sensitive to security, because they lack a sandboxed environment, although tools and frameworks exists such as codebox-api, RestrictedPython, pychroot, or setuptools’ DirectorySandbox to just name a few of these for Python.So let’s set tools:

```
ddg_search = DuckDuckGoSearchResults()
tools = [
    codetool,
    Tool(
        name="DDGSearch",
        func=ddg_search.run,
        description=(
            "Useful for research and understanding background of objectives. "
            "Input: an objective. "
            "Output: background information about the objective. "
        )
    )
]
```

An internet search is definitely worth adding to make sure we are implementing something that has to do with our objective. I’ve seen a few implementations of Rock, Paper, Scissors instead of Tetris.We can define additional tools such as a planner that breaks down the tasks into functions. You can see this in the repo.Running our agent executor with the objective to implement tetris, every time the results are a bit different. I see several searches for requirements and game mechanics, and several times a code is produced and run. The pygame library is installed. The final code snippet is not the final product, but it brings up a window:

```
# This code is written in PEP8 syntax and includes comments to explain the code
# Import the necessary modules
import pygame
import sys
# Initialize pygame
pygame.init()
# Set the window size
window_width = 800
window_height = 600
# Create the window
window = pygame.display.set_mode((window_width, window_height))
# Set the window title
pygame.display.set_caption('My Game')
# Set the background color
background_color = (255, 255, 255)
# Main game loop
while True:
    # Check for events
    for event in pygame.event.get():
        # Quit if the user closes the window
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    # Fill the background with the background color
    window.fill(background_color)
    # Update the display
    pygame.display.update()
```

The code is not too bad in terms of syntax – I guess the prompt must have helped. However, in terms of functionality it’s very far from Tetris. This implementation of a fully-automated agent for software development is still quite experimental. It’s also very simple and basic, consisting only of about 340 lines of Python including the imports, which you can find on Github.I think a better approach could be to break down all the functionality into functions and maintain a list of functions to call, which can be used in all subsequent generations of code. We could also try a test-driven development approach or have a human give feedback rather than a fully automated process.An advantage to our approach is however that it’s easy to debug, since all steps including searches and generated code are written to a log file in the implementation.Let’s summarize!

### Summary

In this chapter, we’ve discussed LLMs for source code, and how they can help in developing software. There are quite a few areas, where LLMs can benefit software development, mostly as coding assistants.We’ve applied a few models for code generation using naïve approaches and we’ve evaluated them qualitatively. We’ve seen that the suggested solutions seem superficially correct but don’t actually perform the task or are full of bugs. This could particularly affect beginners, and could have significant implications regarding safety and reliability.In previous chapters, we’ve seen LLMs used as goal-driven agents to interact with external environments. In coding, compiler errors, results of code execution can be used to provide feedback as we’ve seen. Alternatively, we could have used human feedback or implemented tests. Let’s see if you remember some of the key takeaways from this chapter!

### Questions

Please have a look to see if you can come up with the answers to these questions from memory. I’d recommend you go back to the corresponding sections of this chapter, if you are unsure about any of them:

1. What can LLMs do to help in software development?
2. How do you measure a code LLM’s performance on coding tasks?
3. Which code LLM models are available, both open- and closed-source?
4. How does the Reflexion strategy work?
5. What options do we have available to establish a feedback loop for writing code?
6. What do you think is the impact of generative AI on software development?
