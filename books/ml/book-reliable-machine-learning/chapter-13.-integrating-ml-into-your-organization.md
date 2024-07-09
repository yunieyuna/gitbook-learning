# Chapter 13. Integrating ML into Your Organization

Integrating any significant new discipline into an organization often looks more like an exercise in irregular gardening than anything else: you spread the seeds around, regardless of whether the ground is fertile or not, and every so often come back to see what has managed to flourish. You might be lucky and see a riot of color in the spring, but without more structure and discipline, you’ll more likely be greeted by something barren.

Getting organizational change right is so hard for plenty of _general_ reasons. For a start, an effectively infinite amount of material is available on how to change organizations and cultures. Even choosing from this plethora of options is daunting, never mind figuring out how best to implement whatever you settle on.

In the case of ML, though, we have a few _domain-specific_ reasons this is true, and arguably these are more relevant. As is rapidly becoming a cliché, the thing that is fundamentally different about ML is its tight coupling with the nature and expression of _data_. As a result, anywhere there is data in your organization, there is something potentially relevant to ML. Even trying to enumerate all the areas of the business that have or process data in some way helps to make this point—data is everywhere, and ML follows too. Thus ML is not just a mysterious, separate thing that can be isolated from other development activities. For ML to be successful, _leaders need a holistic view of what’s going on, and a way to influence what’s being done with it—at every level._

What is particularly frustrating and counterintuitive about this situation is that almost every change management methodology recommends starting out small, in order to manage the risk of trying to do too much at one time. Though this is greatly sensible in most cases, and your first ad hoc experiments can generally be done without too much overhead, success in a small pilot guarantees nothing about how ML implementation on a larger scale might work well. Avoiding siloization is hard enough in most organizations, but is particularly crucial for ML.

However, there is good news: it is absolutely practical to start small, to grow a successful pilot, and to make sure the opportunities and risks of ML are handled correctly as you grow. But you have to be deliberate about the way you do it, which is why we should talk about our frameworks and assumptions first.

## Chapter Assumptions

We have written this chapter with assumptions that we would like to make clear before we begin. Each is detailed in this section.

### Leader-Based Viewpoint

Our first assumption—which might already be clear—is that this chapter and [Chapter 14](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch14.html#practical\_ml\_org\_implementation\_example) are unapologetically written for the organizational leader. Though there are points of relevance to data scientists, ML engineers, SREs, and so on, this chapter is most urgently addressed to those responsible for the health, structure, and outcomes of their organization, on a scale from a team (two or more people) to a business unit or company (hundreds or thousands of people).

### Detail Matters

Generally speaking, organizational leaders don’t engage in detail and directly with implementation and management of _any_ change project, except in cases of strong need. ML needs to be a little different. As per the preceding assumption, since doing ML well involves understanding the principles of how it works, what use it makes of data, what counts as data, and so on, leaders need to know this before the decisions they make are going to be sensible.

Our main observation here is that by default, leaders are not going to pick up ML-relevant knowledge as part of their regular management activity, and so there needs to be an explicit mechanism for doing so. This is in opposition to the bulk of conventional managerial theory, which asserts that most teams can be managed effectively with a handful of representative KPIs and a good understanding of team dynamics. We currently believe that ML is sufficiently complex, new, and potentially impactful that being aware of the details matters—though we expect this will change as time goes on.[1](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch13.html#ch01fn139) For the moment, though, leaders need to understand ML basics and be able to access practitioners, which in turn will help them assess the likelihood of success, and make outcomes better.

Outside of training programs, our main recommendation to achieve this level of understanding is that ML should not be siloed, either as a standalone technology-driven effort, or in any other way. Given it can touch everything, a strong siloization would invert the flow of information and control, and could introduce big risk management problems. ML is naturally a horizontal activity.

### ML Needs to Know About the Business

Our third assumption is that the complexity of the _business_ is a direct input to how ML is conceived and implemented. It’s not just about including ML folks in department circulars about goals and performance—it is much more pervasive and holistic than that. ML practitioners need to be more aware of broad business-level concerns and state than the average product developer.

A couple of examples will make this clearer:

Example 1

An ML developer at YarnIt wants to make a business impact, specifically on the web sales part of the business. They work to build a model identifying products that are underperforming in terms of sales. The model recommends these products in particular contexts in order to increase their sales. A model like this might be successful in a variety of ways. It might identify new purchasers for these products or simply remind people who used to purchase these products that they might want to purchase them again.

But now there’s a problem in our case: it manages to find cashmere wool yarn, a prestige item that YarnIt has in very low supply. Since this model does not have features for (can’t effectively represent or understand) margin or inventory, it manages to sell all of it. Cashmere yarn is in short supply in the industry, and replacing it will take weeks, or even months. Although very few customers purchase this yarn in large quantities, many of them may buy a little bit from time to time as part of larger orders. Another effect that you sometimes see in ecommerce is that when a web shop doesn’t have a particular thing in stock, customers sometimes take their whole order elsewhere—and YarnIt experiences this too. So now YarnIt is losing sales on _other_ products because of their lack of stock on _this_ product.

Example 2

Another case exposes a cross-organizational privacy and compliance problem. The recommendations and discovery teams train models to help customers get the most out of the _yarnit.ai_ site and help YarnIt make the most money from customers. Among the features that the models use is information about the browser the customers are using to access the website, so the team uses a limited set of information from the `User-Agent` string provided by the browser—just enough to determine the browser and platform.

Meanwhile, the web design team has been working on new interactions and wants a better sense of what browser configurations customers are using. To get all of the information required, the team decides, _without_ talking to the ML folks, to just log and track the full browser information. As a result, the modeling teams start using the full browser `User-Agent` in their models without knowing it.

The problem is that the full contents of the `User-Agent` plus location information (that the model also has) often uniquely identify a single person. So now we have a model that has the ability to target individual people. This violates some privacy governance policies of YarnIt and compliance requirements in the countries where YarnIt operates as well, posing a serious risk for the whole organization.

In both of these cases, the individual teams did nothing wrong _in their context_. But acting on incomplete information about the effect of their choices on other teams led to bad outcomes. As a result, leaders need to be aware of how ML functions in their organization, so they can provide the vital coordination and broad oversight that would otherwise be missing. The big question is really how best to provide this.

Here is one structural way to think about it: you need to be able to centralize the portions of the ML work where oversight and control are most important, to liberate those portions of the work where domain-specific concerns are most important, and to provide an _integration point_ where these workstreams can meet. It is at that integration point that oversight, steering, and communication should take place.

The good news is that most organizations already have some venues where cross-cutting conversations take place. If you are lucky, it will be natural for such conversations to take place within (for example) product management, where customer lifecycle management is a normal matter of concern. If it will not fit into an existing meeting, workstream, or venue of some kind, you will have to create a new one. But, however it is implemented, you will need to have these kinds of conversations going on.

### The Most Important Assumption You Make

We base our material on all of the preceding assumptions. However, your organizational change effort also has assumptions—even, or especially, if you think it doesn’t—and those assumptions are close to you and therefore may ultimately determine the success of your ML effort.

The most important one, the one that’s hugely valuable to examine before you embark on your ML journey, is strongly related to the question of what you’re trying to achieve with your ML project. It is what you assume ML can do for you.

### The Value of ML

ML can do more for your business than just make you more money.[2](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch13.html#ch01fn140) Implementing ML could mean that you could improve civic engagement, raise more funds for disaster relief, or figure out which bridges most urgently need maintenance.[3](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch13.html#ch01fn141) But for business leaders, it usually means making more money and making customers happier (and happy customers generally lead to more money for the business), or sometimes reducing costs via automation.

As an example, let’s go back to YarnIt. The CEO started hearing about ML some years ago. Some of the early ideas for how to put ML to use included the following:

* Power the search results on the website by including a model of which products a given customer (or web session owner if they weren’t logged in) was likely to purchase, given their previous behavior and interest.
* Help customers discover new products, including devoting a substantial portion of the front page to those selected by a model purpose-built to do so.
* Manage inventory. ML can model the supply chain constraints and inventory levels and propose optimal reordering for products to ensure that YarnIt has stock of the appropriate mix of products, given financial, sales, storage, and supply constraints.
* Improve profitability by adding a product and order margin as a feature to many of the other models.

Notice that these ideas have different timescales and different levels of organizational intrusiveness. Some of them can be easily tacked on to what is already being done and how it’s done. Changing the ranking of the search results will probably have a measurable impact on customer satisfaction and sales, but can be done relatively quickly (weeks or months, not quarters or years) and will not require particular participation from the broader organization to implement. Changing the way inventory and supply chain work will take longer and will require much broader participation from other departments. If done well, it has the possibility of really transforming the overall efficiency of YarnIt as a company, perhaps akin to just-in-time supply chains. But it is not a change that can be led by an ML engineer.

Even facing these implementation challenges, organizational leaders are usually broadly sold on the value of ML. Indeed, ML has the potential to realize additional value based on data already collected by the organization. It is the kind of once-in-a-generation technological change that really can transform the way organizations function—hence the necessity to examine carefully what you think ML can do for you, figuring out what subset of that you want to achieve, and writing all that down before you start.

## Significant Organizational Risks

Suppose you have assessed what ML can do for you, decided on the specific form it will take, written down your assumptions, and are eagerly rubbing your hands together, anticipating the wonderful changes that are going to bring you fame and fortune right now. “What next?” is the obvious question. Unfortunately, before we get started, it’s as important to understand the risks as it is the value. Otherwise, you won’t be in a position to make a well-founded decision that involves prioritizing one over the other.

### ML Is Not Magic

While most business leaders have some appreciation of the value and potential of ML, they do not necessarily understand the risks equally well. As a result, you can sometimes see a leadership perspective emerging that treats ML practitioners as almost magical in what they can achieve. Yet no one—at the leadership level, anyway—understands how or why. By misunderstanding the scope and mechanisms of ML, leadership also overlooks the scope of the impact of those projects across the organization. The greatest danger in this case is that the risks become invisible, or effectively become externalities—in other words, someone else’s problem. That is a recipe for inevitable, though perhaps arbitrarily deferred disaster.

### Mental (Way of Thinking) Model Inertia

Transforming the way an organization works is never a simple proposition, and thousands of pages have been written on that topic. Here, we confine ourselves to saying that implementing ML is just like any other change, in that it requires stakeholder management and obtaining buy-in from those affected, but also unlike other changes in that the total set of stakeholders is likely to be much larger.

As a result, that component of the problem that is a function of the number of stakeholders (for example, the pure logistics of figuring out who needs to be involved) obviously gets larger. More importantly, though, any component that involves persuasion, communication, and, in particular, understanding the way people _model_ the change is also dramatically increased in importance.

When driving a significant change, just showing up to all your meetings with the talking points of the senior leader involved is rarely going to work. You are not going to persuade people to change their behavior based on a characterization of the situation that is only from a senior leader’s point of view. In particular, the key issue is of _mindset_, and what _mental models_ are used by both leaders and practitioners throughout the organization to represent what’s going on and how to react to it. If the plan assumes that everyone will just move to understanding things the way you (or a certain individual) do, the plan will probably be short-lived.

But sometimes the new way of doing things is genuinely the correct way of responding to a particular situation. If it _is_ the correct way to proceed, and yet the mental models are not changing, you, the changer, have to undergo the burden of persuasion. To be most effective, that persuasion has to be accompanied by a motivation or set of motivations, and those need to speak to the mental models of the audience. For example, perhaps the audience believes there’s nothing real to be gained by spending the effort to make the data of their teams readable by others; or perhaps they’re scared by the prospect of being shown to be worse than other teams; or perhaps they profoundly believe that the only thing that matters is getting product features out to the public as fast as possible, and spending effort on literally anything else doesn’t matter. Either way, your proposal for change will require the mental models of your audience to be solicited, understood, and addressed.

Ultimately, for most practical concerns, implementing ML requires serious stakeholder management and a large concerted effort to shift mental (way-of-thinking) models.

### Surfacing Risk Correctly in Different Cultures

Obviously, if benefits are clear at the leadership level, but risks are invisible, this leads to risk management taking place in an ad hoc, underfunded fashion, or potentially even deliberately not taking place. Those risks could even be misrepresented, particularly in negatively oriented cultures. It may be useful to review the organization typology suggested by Dr. Ron Westrum in the context of software engineering organizations to understand the implications of this situation in more detail.[4](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch13.html#ch01fn142)

If we could simplify somewhat, Westrum suggests that organizations can be broadly characterized as power-oriented, rule-oriented, or performance-oriented. Of these, the only organizational culture that experiences _structural_ problems when implementing ML is the rule-oriented, bureaucratic culture. Why? On one hand, _power-oriented_ organizations tend to crush novelty, and as a result are much less likely to implement ML on their own in any serious way. On the other hand, _performance-oriented_ cultures have an openness to novelty, cooperation, communication, and risk sharing. These environments are likely to tolerate the kinds of open coordination that successful ML implementations require.

On the contrary, _rule-oriented_ organizations tolerate novelty but punish people when it goes wrong. Failure leads to negative consequences for those seen to have failed, organizations have narrow (and thoroughly defended) responsibilities, and coordination is minimal. In these organizations, we expect that ML will be able to gain a foothold, but when anything goes wrong or becomes difficult, those “at fault” will be punished and the innovation will promptly cease. Unfortunately, such behavior makes it very hard to adequately model and respond to the cross-cutting risks that go with ML; significant losses may well result.

### Siloed Teams Don’t Solve All Problems

Another common risk is for ML teams and projects to be treated equivalently to the way other new kinds of work are treated, and a common instinct is to start a new siloed team to do that work, leading to its separation in the organization. This is a common way to reduce startup friction in order to build something and demonstrate results. However, it does present a problem for ML, since implementing it at all usually requires help from multiple departments or divisions. But more importantly, because of the broad scope of impact that ML projects can have, successfully deploying ML requires organizational change to support structure, processes, and the people needed to keep it reliable. It is definitely possible to keep the scope too narrow for success.

## Implementation Models

Having discussed some risks involved in introducing ML to an organization, let’s focus on the nuts and bolts—how to actually get it done.

A small implementation project probably starts with applying ML to something that is integral to your organization’s success. This involves creating and curating data sources, assembling teams with the right expertise in the problem space _and_ in ML, and making the horizontal regulation mechanisms you’ll need to track progress and steer the ship. Throughout this process, it will probably be advantageous to proceed with the blissful optimism that’s required when you know trouble will come find you, eventually—though you don’t know quite when or what sort.

Start out by picking a metric of some kind: ideally, it will be something _useful_ but not _critical_ in your system. That way, you can get valuable experience from the implementation, and the work will involve assembling the cross-connections between teams that you’ll need for future expansion, but if things get messed up, the likelihood of significant disaster is lowered.

Let’s consider an example at YarnIt. The implementation team will probably consider a few options. Using ML to help with search ranking is one appealing place to start. But the team notices that sales coming directly from the search results page represent significant revenue. This makes this an appealing place to apply ML eventually, but a risky place to get started. After looking for other parts of the site and seeing that all of them are revenue sensitive or revenue critical, the team takes a different approach: what if we _add_ ML-produced results where there is nothing right now? The team members look around the _yarnit.ai_ site and notice that several pages do not show recommendations to end users but could. They decide to add recommended products to the cart-add confirmation page that users see when they add an item to their shopping cart—or to put it another way, we take that moment where the users have demonstrated their interest in one product to recommend other products.

This is a good place to start: purely additive, with low risk and at least a chance of a reasonable return. Purchase intent is already present, and the change is not too intrusive to existing customer workflows. So the team pursues this “People who bought _X_ also bought _Y_” model and decide to measure it by collecting click-through rates on those recommendations and comparing them to click-through rates on search results. Of course, once the team knows more about how to do this, another possible option is to adopt a more traditional approach that combines looking at what is achievable/feasible and what the expected yield would be, rather than focusing on minimizing intervention risk.

### Remembering the Goal

Though it is important to preserve flexibility, particularly in the conduct of the implementation, we also have to remember the goal—to experiment with ML in order to build capacity in the organization. Hopefully, you’ll achieve the business metric improvement you selected, but even if you don’t, the project can still be successful overall: you can still learn a lot and try another approach if things don’t work out.

But it is important to walk that delicate line between not becoming distracted by what you encounter along the way, and not being too rigid about what you’d decided previously.

**TIP**

Writing down your strategic goals, and the context that led to those goals, can be useful to refer back to when you are amidst troubleshooting tactical issues or handling an incident.

### Greenfield Versus Brownfield

A fundamental question that often arises when you’re starting a new set of activities in your organization is whether you’re doing it from nothing (also known as _greenfield_) or you have an existing system, team, or business process to handle (also known as _brownfield_). In practice, almost all implementations are brownfield, because most organizations get most value out of improving a system they already have. In general, though simplifying considerably, transformation projects go easier the more of a greenfield situation it is or can be made to be.

A common intuition is that it is easier to build on something that already exists and is (somewhat) functional. But in fact, a crucial measure of success for new initiatives is how much opposition it attracts. Opposition is more commonly encountered in brownfield situations, where someone else’s career success can often depend on nothing changing.

For just those reasons, most implementation projects that expect to meet significant opposition usually try to start a new team or function that covers previously uncovered responsibilities. Our view is that because of the strong interconnected nature of ML, it is not realistic to expect that relative isolation to continue for long. Eventually—and probably sooner than you think—you’ll talk to someone else you need to ask permission from.

Our best guidance here, as previously, is to start with a metric that makes sense, since that’s the easiest story to tell—successful transformations almost always require good stories. Then use that to determine how greenfield or brownfield your project needs to be, while acknowledging that most things are brownfield.

### ML Roles and Responsibilities

Doing ML work well involves a dizzying array of skills, focus areas, and business concerns. We find one useful way to structure this knowledge is by thinking, as always, of the flow of data within your organization. For example:

Business analysts or business managersThese roles are responsible for the operations of a particular line of business as well as the financial results from that line of business. They have the data and desire needed to make ML successful, but if it goes badly, their ability to do their job will suffer as a result of bad information.Product managersThese roles set the direction for the product and determine how ML will be incorporated into existing products. They help us decide what, if anything, we will do with the data. There may also be ML-specific product managers who guide what we implement as well.Data engineers or data scientistsThese people understand how to extract, curate, manage, and track data as well as how to extract value from it.ML engineersThey build and manage models and the systems that produce them.Product engineersThey develop the products that we are trying to improve with ML. They help us understand how to add ML to the product.SREs for ML or MLOps staffThey lead the overall reliability and safety for the deployment of ML models. They improve existing processes for building and deploying models, propose and manage the metrics to track our performance over time, and develop new software infrastructure to enforce model reliability. These roles wrap around the entire process and are some of the only engineers looking at the process from end to end.

Each of these roles may be combined with others in a smaller organization. They are functions to think about filling.

### How to Hire ML Folks

Hiring talented ML staff is difficult right now and is likely to stay difficult for the foreseeable future. The growth of demand for ML skills has far outstripped the supply of educated, experienced staff. This affects all employers, but the most prestigious of ML companies (generally large tech organizations) continue to hire most of the new graduates and experienced staff. This leaves other organizations in difficult circumstances.

The usual recommendations for how to proceed in this case include options like attempting to reach potentially qualified candidates earlier in the cycle, making sure that the operation of the recruitment process is generally strong, communicating with the candidate regularly, selling the candidate on the advantages that the company in question has, and so on. While all of those are true, useful, require effort, and might well work for you, they are standard approaches. If the market is particularly hot, doing all of those well _still_ might not work.

We recommend another approach. Reframe the problem as dividing staffing between those who really require ML knowledge and experience immediately, and those who can learn it on the job as they go. Most situations, and indeed startup programs, require only one or two experienced ML researchers or practitioners. These experienced employees can help design models to meet the organization’s goals and can also specify the systems needed to build and deploy those models. But the staff needed to manage the data, integrate ML into the product, and maintain the models in production can all be folks who are talented in other ways, but are learning ML deeply on the job. (You can even buy books like this to help bring those employees up to speed more quickly!)

So, having partitioned the problem, we still have the question of how most organizations can hire those first few experienced ML researchers and engineers to seed this process. The standard playbook involves a mix of hiring contractors or consultants from experienced firms, paying for a single superstar with real experience and credentials who is willing to teach, and betting on junior rising stars, while understanding that the path will be bumpy. These are practical options when your organization has a desire to produce ML but does not have prestige or money to compete against bigger firms.

Now that we’ve considered some of the concrete challenges that organizations face adapting ML specifically, let’s take a step back and consider the problem from the perspective of traditional organizational design.

## Organizational Design and Incentives

Making an organization function well, given what it is supposed to do—often called _organizational design_—is a difficult art that involves a mixture of strategy, structure, and process. The key point for leaders is that reporting structures are often the least important part of successful organizational designs. Other much more powerful aspects and determinants affect behavior.

Before we dive in even deeper, it is worth acknowledging that organizational design is a technical and often jargon-filled topic. For some leaders, especially those at smaller organizations, it may be difficult to see the forest for the trees in the proceeding sections. Talk of strategy and process and structure can be difficult to map onto the main actual tasks: hiring the right people and getting ML added to your application. Ultimately, though, the main lesson is that thinking about the way your organization currently works, and how that will change, hugely improves your chances of doing ML successfully.

We can choose from numerous models to understand how to change an organization in order to achieve a certain goal. This section is not designed to provide a complete review of all of them. Rather, we will select one common approach to thinking about organizations, the [Star Model](https://oreil.ly/y1xts) by Jay R. Galbraith, and apply it specifically to the challenge of implementing ML in an organization ([Figure 13-1](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch13.html#the\_star\_model\_left\_parenthesishashone)).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098106218/files/assets/reml_1301.png" alt="The Star Model (© Jay R. Galbraith. Reprinted with permission.)" height="350" width="600"><figcaption></figcaption></figure>

**Figure 13-1. The Star Model (© Jay R. Galbraith. Reprinted with permission.)**

In this model, strategy, structure, processes, rewards, and people are all design policies or choices that can be set by management and that influence the behavior of the employees in the organization.

This model is useful because it goes beyond the reporting structure or organization chart, where most leaders tend to start and end their change efforts. Galbraith points out that “most design efforts invest far too much time drawing the organization chart and far too little on processes and rewards.” This model allows you to take that observation and then think about whether all of the interconnected aspects are affected or can be changed to support the requirements better. Policies, processes, people, and rewards policies can then be adjusted to support your structure and strategy.

Let’s review each of these in the context of an organization trying to implement ML.

### Strategy

The _strategy_ is the direction that your organization is trying to go. It drives your business or organizational success model. It affects which parts of the organization are given attention or funded, and how the organization is measured or considered successful.

“Best-in-class machine learning for the yarn distribution industry” could be a strategy that identifies ML as a primary focus for YarnIt, but also might limit where ML is deployed if we insist on only “best-in-class” ML.[5](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch13.html#ch01fn143) Another strategy of “machine learning in all aspects of the product” might mean the organization funding new and innovative ways of using ML everywhere, with more tolerance for lower-quality results to start with. On the other hand, if we set a strategy to “increase sales by diversifying approaches including the use of ML,” we might consider it as more experimental or less important than other more traditional ways of increasing sales.

### Structure

_Structure_ describes who has power in an organization. You may also think of it as the organization chart or reporting structure because it identifies formal oversight authority. (It can be very different, of course; in other places, the authority may lie within the team, where certain technical leaders must support a decision before it is implemented.)

One way to think about choices for organizational structure, and the one that Galbraith identifies, is that it includes functional, product, market, geographic, and process structures:

FunctionalThis structure organizes the company around a specific function or specialty (for example, centralizing ML implementation in a single team).ProductThis structure divides staff into separate product lines. In this case, the ML teams would be distributed into the individual product teams.MarketThe company is organized by the customer market segment or industry they sell to. For YarnIt, this might be by type of crafter (knitter, weaver, or crocheter).GeographicalThis structure organizes by territory: the product has a dependency on region, location, or even distribution economics (such as where the food comes from). The only obvious reason to consider this structural approach for ML would be governance and compliance with local laws. This is probably not how we would structure an ML implementation.ProcessAlso sometimes known as a _horizontal organization_, this structure aggregates power in all of the people who develop and deploy processes in an organization. This may be a good model for ML teams that work across various product lines but need to create standards and processes for the organization.

Leaders will generally have a mental model for the way the organization works and the approach they should use to effectuate change. For example, think of a mental model that, to start a new function, the senior leader must be hired first. With this mental model, a leader will tend to centralize ML functions around a specific senior leader—and this has obvious drawbacks if the right leader is not to be found, or centralization doesn’t fit well with (say) the existing engineering culture. Similarly, a siloed ML function might work better for senior leaders to maintain control of, but would inhibit progress of ML on other engineering teams. Ultimately, leaders will probably need to shift their mental model of the way things work, depending on the chosen ML strategy. (No one-size-fits-all structure exists, though for those desiring a this-size-fits-some approach, we cover structure implementation choices in detail in [Chapter 14](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch14.html#practical\_ml\_org\_implementation\_example).)

### Processes

_Processes_ constrain the flow of information and decisions through an organization, and hence are critical to the way ML will work. They can be used to address issues in the structure as necessary. The Galbraith framework defines two types of processes: _vertical processes_ allocate scarce resources (e.g., budget), and _horizontal processes_ manage workflows (e.g., customer order entry and fulfillment end to end).

One potential way to begin adding ML to your organization is to treat the introduction as a vertical process, with decisions made centrally but implemented throughout the organization. That works well if the leaders have mastered their dependencies and connections. If that’s not the case, you can get disconnected decisions. For example, if we fund an ML training and serving team to add a new ML feature to our application, do we also fund teams to curate all of the data, or to handle model quality measurement over time or fairness? If we do, we might end up duplicating centralized functions in our local scope, which is inefficient and potentially friction-increasing.

Once the organization has several ML projects implemented, centralizing the infrastructure from those projects to fulfill specific workflows may add robustness and reliability. For example, many model teams will start by providing their own infrastructure end to end, but eventually we might have many modeling teams providing models that integrate into our application. At that point, we could centralize serving for some of those models, think about building a central feature store, and so start establishing common aspects of the ML organizational infrastructure regardless of the model team.

### Rewards

_Rewards_ are both financial and nonmonetary. While most organizations will find it difficult to compete for ML talent on a financial basis alone, it might make more sense for an organization to compete on mission, culture, or growth. Most employees value recognition, status, or career opportunity. They also value applying their skills autonomously to create something of value. The autonomous part is tricky because ML staff need to be independent, but it is also critical that they be subject to the kind of governance that the organization needs to ensure that the ML it deploys is fair, ethical, effective, and compliant with relevant laws. Aligning rewards to not just raw execution of business goals, but also reliability, fairness, and robustness will help create the right incentives to not overlook these areas.

One other surprising point should be noted about rewards for ML skill and knowledge. Recall that ML is likely to impact most parts of our organization. One thing that should be considered is rewarding staff throughout the organization for learning more about ML. If the sales staff, accounting staff, buyers, and product managers all have a basic education in ML, the organization may well be much more effective in the long run.

We expect that ML expertise will continue to be scarce indefinitely into the future, with consequent effects on compensation, difficulty of hiring, etc.—see [“How to Hire ML Folks”](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch13.html#how\_to\_hire\_ml\_folks) for suggestions for active approaches here.

### People

Finally, we need to consider the collection of factors that influence the human beings in our organization. This includes the mindsets and skills that those people need. It also includes human resource policies of recruiting, selection, rotation, training, and development of people. For example, flexible organizations need flexible people. Cross-functional teams require people who can cooperate with each other and are “generalists” that understand multiple aspects of the organization.

Given how rare ML education and skills are at present, most organizations should consider hiring staff who can learn on the job rather than only those already qualified. While this is true across the map of ML staffing, it is especially true in the territory of SRE. An ML production engineer benefits much more from solid reliability and distributed systems skills than they do from ML skills. In these roles, ML is the _context_ in which the work happens, but not always the _content_ of that work.

Finally, the organization will need people who can work through the ambiguity of problems caused by ML without stopping at a root cause of “the ML model said so.” That’s a fine place to start, but people will need to be able to think creatively about the way ML models are built, how changes in the world and in the data impact them, as well as how those models impact the rest of their organization. Some of that perspective and approach will come from ML education and skills, but some comes from a curious and persistent approach to problem-solving that not everyone starts with.

### A Note on Sequencing

The preceding topics were separated out for clarity of explanation and ease of illustration. Though separation of concerns is a powerful technique much used in computer science, in real-world organizational work, everything is entangled and intertwined in a way that can often make it practically impossible to change things by just exerting control over a single dimension. The good news is, it often turns out that’s what you want.

Changing one single dimension of the preceding Star Model elements is unlikely to result in success by itself. A strategy change divorced from process change will almost certainly result in just effectively the same output. Swapping in a new set of people who will learn the old culture has a good chance of developing a new set of workers who will behave as the old ones did. Financially rewarding new behavior while still allowing old behavior to be easy to accomplish (because all the processes are optimized for that) won’t change anything in and of itself. And so on; the grim reality is, successful change often relies on pushing across many fronts in parallel.

However, you don’t have to move forward at the same pace across all of these fronts at the same time, or with the same intensity. That’s the second piece of good news—you can _sequence_ this. Announce you’re changing strategy, then processes, then rewards. Deal with them one at a time, but touch them all—at least, the ones that matter for your organization. Tell everyone the timescale you’re following and the criteria you’re using to evaluate success. Communicate your intentions, but acknowledge that not everything is going to change at once—it won’t—but loudly and publicly commit to the overall goal. That increases credibility for the change and gets you supporters inside the organization.

## Conclusion

We can’t provide context-free recommendations for which precise dimensions of change to push on, since so much depends on your local situation. However, we can recommend at least thinking about the following:

* What does the organization care about? Driving change by trying to accomplish something the organization doesn’t care about is more likely to be ignored, but also much less likely to get resources. Your work is overall more likely to be successful if you’re aligned with those concerns.
* What are people doing today, and how does it need to change? Many plans for change originate from high-level staff disconnected from the day-to-day experiences of other staff members. Your change plan will have a much better chance of success if you take a step back and look at what they do and why they do it.
* How easy will it be to do the new thing rather than the old thing? If the new thing is harder to do than the old thing, everyone may well agree it’s vitally important to do it, but the change will be slower and more difficult, if it happens at all. Make it easier to do the right thing and harder to do the wrong thing.
* Finally, acknowledge that change will take time. As we’ve said, displaying organizational vulnerability not only gets you more support from people who appreciate realism, but also allows people to manage their own reaction to the change better. Just don’t forget to keep a regular communications cadence up: a big-bang announcement followed by nothing for ages often causes people to wonder whether the momentum has stalled.

If you want to see worked examples of the preceding points, see [Chapter 14](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch14.html#practical\_ml\_org\_implementation\_example).

[1](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch13.html#ch01fn139-marker) For what it’s worth, we don’t venerate leaders who are continually involved in the details—sometimes it’s better and sometimes it’s worse—but we do believe that it’s necessary for leaders to understand the trade-offs at this point in ML’s evolution. At the very least, organizational leaders need to know the business metric being optimized and need to have a means of measuring whether the ML system is optimizing that metric effectively. Understanding some of the details of the implementation as well as the process for measuring the effectiveness will allow leaders to do so with confidence.

[2](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch13.html#ch01fn140-marker) This observation isn’t limited to for-profit businesses. Leaders looking to implement ML often hope for it to improve the thing they already do: make more money, give out more food, pay for more housing. ML can do these things, but it can also transform the way you think about running the organization as a whole.

[3](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch13.html#ch01fn141-marker) During the writing of this book, a prominent bridge collapsed in Pittsburgh, Pennsylvania, where some of us live. Although the main problem with physical infrastructure in the US is simply that the country doesn’t spend enough money, it is also true that prioritizing where to spend limited resources might be amenable to an ML application.

[4](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch13.html#ch01fn142-marker) Westrum’s original paper is [“A Typology of Organizational Cultures”](https://oreil.ly/T9Jje). The [Google Cloud Architecture Center](https://oreil.ly/POd9Y) reviews this in the context of DevOps, but most of the points are relevant to ML production engineering as well.

[5](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch13.html#ch01fn143-marker) “Kind of reasonable most of the time” ML can actually be an improvement on existing algorithmic deployments in a variety of situations.
