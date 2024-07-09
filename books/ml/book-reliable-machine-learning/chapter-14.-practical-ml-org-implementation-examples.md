# Chapter 14. Practical ML Org Implementation Examples

Organizations are complex entities, and all of their different aspects are connected. Organizational leaders will face new challenges and changes in their organization as a result of adopting ML. To consider these in practice, let’s look at three common organizational adoption structures and how they apply to the organizational design questions we have been considering.

For each of these scenarios, we will describe how the organizational leader has chosen to integrate ML into the organization and the impact of that choice. Overall, we will consider the advantages and likely pitfalls each choice has, but in particular, we’ll consider the way that each choice affects the process, rewards, and people aspects (from the Star Model introduced in [Chapter 13](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch13.html#integrating\_ml\_into\_your\_organization)). Organizational leaders should be able to see enough details in these implementation scenarios to recognize aspects of their own organizations, and be able to map these into their own organizational circumstances and strategies.

## Scenario 1: A New Centralized ML Team

Let’s say that YarnIt decides to incorporate ML into its stack by hiring a single ML expert who develops a model to produce shopping recommendations. The pilot is successful, and sales increase as a result of the launch. The company now needs to make some decisions about how to expand on this success and how (and how much!) to invest in ML. The YarnIt CEO decides to hire a new VP to build and run the ML Center of Excellence team as a new, centralized capability for the organization.

### Background and Organizational Description

This organizational choice has significant advantages. The team can specialize, ample opportunities for collaboration and working together exist, and the leadership has clear scope to prioritize the work on ML across the company. Reliability experts can be in the same organization if their scope is limited to ML systems. The centralization also creates a significant nexus of influence: the leaders of the ML organization have more standing to advocate for their priorities across all of YarnIt.

As the group grows and the projects diversify, more of YarnIt will need to interact with the ML organization. This is where the centralization becomes a disadvantage. The ML team cannot be too distant from the rest of the business, as it will take the team longer to see opportunity, to deeply understand the raw data, and to build good models. An ML team siloed away from individual product teams is unlikely to be successful if it doesn’t have the support of those product teams. Even worse, placing these two functions (ML and product development) completely separately in the organizational chart might encourage the teams to be competitive instead of cooperative.

Finally, a centralized organization may not be usefully responsive to the needs of the business units requesting help to add ML to their products. When it comes to productionizing ML, the business units likely will not understand the reliability needs of the ML teams and not understand why reliability processes are being followed (thus slowing delivery).

While these pitfalls exist for a solely centralized ML team, the organization can always evolve. This scenario goes through the Star Model as if it uses only a centralized team, but we will also illustrate an evolution of a centralized team doing infrastructure in scenario 3. Another possible evolution is that the centralized team educates and enables others to increase ML literacy within the rest of the organization.

### Process

As we have mentioned, the impact of introducing ML in an organization tends to be pervasive. To address some of the disadvantages of the centralized organization, introducing processes can help distribute (or decentralize) decisions or knowledge. These processes include the following:

Reviews by key stakeholdersThese reviews, presented by the ML team on a regular basis, should ensure approval of the current modeling results as well as an understanding by business leaders of the way the system is adapting to the business. A separate science of exactly which metrics to include in these reviews is necessary, but the metrics should be complete and need to include the improvements as well as the costs of any given model implementation. Key business stakeholders may also review the priority of the various ML team efforts as well as the ROI and use cases for those efforts.Independent evaluation of changesOne issue that can crop up in a centralized ML team is that all the changes become dependent on one another and may be held up by other changes. If the team instead evaluates the changes independently, ensuring accuracy of each model as it changes production by itself, then changes can be available more quickly. Often a model may improve performance on average but may hurt performance on specific subgroups. Judging whether these trade-offs are worth it can often require significant analysis and digging, and may require judgment based on business goals that are not easily reflected by simple metrics like predictive accuracy.De-risked testing of combinations of changesIn large model development teams, several improvements often are developed in parallel and these then might be launched together. The question is whether the updates play well together. This creates a need to test candidate changes in combination, in addition to individually.[1](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch14.html#ch01fn144) The caveat is that it might not be possible to test all combinations (because of resource/opportunity cost). It is important to create a process to vet candidate changes, and to determine which are useful to test in combination. This may be in the form of a go/no-go meeting to discuss model launches and how the teams are testing in combination with other changes. The meeting may then facilitate further testing or a decision to launch.

While this is easier in a centralized ML model team, introduce processes so that people from the various business units can also evaluate the changes. For the centralized ML team, this may be the product/business team that requested a change or feature, or the support team that may be affected by the changes.

### Rewards

To ensure a successful implementation of the ML program, we need to _reward interaction between business and model builders_. Staff members need to be evaluated on the effectiveness of their cross-organizational collaboration and alignment to business outcomes rather than simply the effectiveness of accomplishing the narrow mission of their own department or division or team. Seeking input and formal reviews by other organizations, providing feedback, and communicating plans should all be rewarded behaviors. The way these are rewarded is culturally dependent per organization, but they should be recognized.

Rewards may be monetary (bonuses, time off) but can also include promotions and career mobility. At YarnIt, we might add evaluative characteristics like “effective influencer” to the characteristics we look for in successful employees.

### People

An _experimentation mindset_ in product leaders is required, both to appreciate the value created by ML as well as to tolerate some of the risks. Product leaders need to understand that tuning the ML model for the organizational objectives may take some experimentation and that negative impacts will almost certainly occur along the way.

**BIGGER ISN’T ALWAYS BETTER**

The YarnIt ML team begins work on a project whose success definition is to create _bigger carts_ (shopping carts containing more products worth more money). The idea is to increase the number of large purchases, since customers who make large purchases are more profitable to the organization. The team measures its success by the size of the shopping cart created per user impacted by the new model, since that’s the metric the team has easily available.

A problem arises: the new larger carts are abandoned at a much higher rate than expected. Users seem to be creating large carts but then never checking out and purchasing the products. This cart abandonment rate is escalated to a senior leader in sales who starts troubleshooting the problem immediately, first with the web and payments teams (might be a UI problem or payments processing problem) and then ultimately with the ML team.

Together they develop a theory of what might be happening: the ML model is successful at convincing users to put more in their carts, but some users balk when they see the total size of the purchase and decide to abandon the cart instead of purchasing some of the products they can afford. At this point, everyone can collaborate on a solution.

Sales can generate an acceptable target for cart abandonment for the ML team to include in its model optimization, the web UI team can think of ways to make it easy to check out parts of a cart rather than a whole cart, and the product team can think about remarketing to users who abandoned carts, asking whether they might want to purchase just some of the products. And overall the ML team can evaluate whether the “large carts”’ effort did, in fact, optimize revenue.

The end result will be happier users and a more profitable YarnIt, but only by working through this set of challenges rather than simply rejecting the poorly performing ML model right away.

YarnIt, and all organizations implementing ML, need to hire for a mindset of _nuance_ in order to be successful. Leaders need to be tolerant of complexity and comfortable working across organizational boundaries even outside their own scope of authority. Notably, leaders also must have the confidence to represent the complexity of these impacts, without overly simplifying or sanitizing, upward to their leadership. YarnIt’s CEO does not need to hear that ML is magic and will solve all problems, but instead how business objectives are being achieved with ML as a tool. The goal is not to do ML but to move the needle for the business. While ML is powerful, the nuance of ML is to derive value by minimizing the negative impacts.

The people in this centralized team need to be trained about quality, fairness, ethics, and privacy issues if they do not already have expertise in these areas.

### Default Implementation

An oversimplified default implementation of the centralized model is as follows:

* Hire a new leader with ML modeling and production skills.
  * Hire ML engineering staff to build models.
  * Hire software engineering staff to build ML infrastructure.
  * Hire ML production engineering staff to run the infrastructure.
* Establish implementation plans with product teams (source of data and integration point for new models).
* Establish regular executive reviews of the whole program.
* Plan compensation by successful implementation and compensate both ML staff and product area staff.
* Start a privacy, quality, fairness, and ethics program to establish standards and compliance monitoring for those standards.

## Scenario 2: Decentralized ML Infrastructure and Expertise

YarnIt might decide to invest in several experts across the organization, rather than a single senior leader. Each department will have to hire its own data scientists, including the shopping recommendations and inventory management teams. Essentially YarnIt will allow data science and simple implementations of ML to appear wherever there is sufficient demand and a department is willing to pay for it.

### Background and Organizational Description

This approach is much faster, or at least it is faster to get started. Every team can hire and staff projects according to its own priorities. The ML experts as they are hired will be close to the business and products, and will thereby have a great understanding of the requirements, goals, and even politics of each group.

There are risks. Without a central place for ML expertise, especially in management, developing a deeper understanding of what YarnIt needs to do to be successful at ML will be harder. Management will not understand what specialized tools and infrastructure will be needed. There is likely to be a bias for trying to solve ML problems with existing tools. It will be hard to understand when the ML team is advocating for something it really needs (model-specific quality-tracking tools like TensorBoard), as opposed to something that might be nice to have but may not be required (GPUs for some model types and sizes or cloud training services that offer huge scale but also large costs). Additionally, each team will repeat some of the same work: creating a robust and easy-to-use serving system that can share resources across multiple models, and monitoring systems to keep track of training progress for models and to ensure they complete training. All of this duplication can be expensive if it is avoidable.

If some of these teams are doing work that spans the other products, and they probably will be, troubleshooting and debugging become much harder. When product or production problems arise, YarnIt will need to figure out which team’s model is responsible or, worse yet, get multiple teams together to debug an interaction among their models. A proliferation of dashboards and monitoring will make this exponentially more difficult. Uncertainty about the impact of any given model’s change will go up.

Finally, YarnIt will struggle to ensure that it has a consistent approach. In terms of ML fairness, ethics, and privacy, just a single bad model can harm their users and damage our reputation in public. YarnIt also may be duplicating authentication, integration into IT, logging, and other DevOps tasks with this organizational structure.

While real trade-offs exist, this decentralized approach is exactly the right one for many organizations. It reduces startup costs while ensuring that the organization gets targeted value out of ML immediately.

### Process

To make this structure effective, organizations should focus on processes that can introduce consistency without introducing too much overhead. These processes include the following:

Reviews by senior stakeholdersModel developers should still participate in reviews by senior stakeholders. It is a really useful practice for model developers to create write-ups of each proposed model development objective and finding that they come up with. These internal reports or whitepapers, while short, can record in some detail the ideas that have been tried and what the organization has learned from them. Over time, these create organizational memory and enforce rigor in evaluation similar to the ML equivalent of the rigor in a code review for software engineers. YarnIt should create a template for these reports, possibly generated by collaboration among some of the first groups to start using ML, and a standard schedule for reviews by a small group with representatives beyond just the organization implementing ML.Triage or ML production meetingsML model developers should meet weekly with production engineering staff and stakeholders from the product development group to review any changes or unexpected effects of the ML deployments. Like everything, this can be done well or poorly. A bad version of this meeting may not have all the relevant points of view, might be based on incidental problems rather than well-understood systematic ones, might delve too much into problem-solving, or simply might last too long. Good production meetings are short, focus on triage and prioritization, assign ownership of problems, and review past assignments for updates and progress.Minimum standards for technical infrastructureYarnIt should establish these minimum standards to ensure that models all pass certain tests before launching into production. These tests should include baseline tests, such as “Can the model serve a single query?” as well as more sophisticated ones involving model quality. Even simple changes such as standardized URLs could help drive consistency internally (and anything that helps to make things easier to remember and behave the same is useful in the complex, quickly changing world of ML).

### Rewards

To balance the decentralizing effects of this approach, YarnIt senior management will need to _reward consistency and published quality standards_. For example, leaders should reward employees for timely write-ups and careful reviews that are published in a widely available corpus. Each team will have local priorities that it will tend to prioritize, so it is necessary to reward behaviors that balance increased velocity with consistency, technical rigor, and communication.

One specific factor to note is that in this scenario, YarnIt is less likely to have staff with significant ML experience. One useful reward is to encourage staff to attend (and present at) conferences related to their work.

### People

In this deployment, YarnIt should look for people who can _think both locally and globally_, balancing the local benefits against possible disadvantages to the company (or vice versa). Skills such as influencing without authority and collaborating across organizational lines may be useful to explicitly call out and reward.

The organization will still need people who _care about quality, fairness, ethics, and privacy issues_ and can influence the organization—this is true in every deployment scenario. The difference here is that in this case, these staff members will have to develop local implementations to achieve quality, fairness, ethics, and privacy while also developing broad standards and advocating for their implementation across the company.

### Default Implementation

Here’s an oversimplified default implementation of the decentralized structure scenario:

* Each team hires experts in their own business units:
  * Hire ML engineering staff to build models directly with the product teams.
  * Hire or shift software engineering staff to build ML infrastructure.
  * Hire ML staff or shift production engineering staff to run the infrastructure.
* Develop a practice of internal reports of findings for review by senior stakeholders.
* Establish company-wide technical infrastructure standards.
* Run weekly triage or ML production meetings to review changes.
* Start a privacy, quality, fairness, and ethics program to establish standards and compliance monitoring for those standards.

## Scenario 3: Hybrid with Centralized Infrastructure/Decentralized Modeling

YarnIt started its implementation via the centralized models, but as the organization matures and ML adoption spreads throughout the company, the company decides to revisit that model and consider a hybrid structure. In this case, the organization will maintain some centralized infrastructure teams and some ML model consulting teams in the central organization, but individual business units are free to hire and develop their own ML modeling experts as well.

These distributed ML staff members might start by relying heavily on the central modeling consultants but over time will grow more independent. All of the teams will be expected to use and contribute to the central ML production implementation, however.

By centralizing the investment and use of infrastructure, YarnIt will continue to benefit from efficiency and consistency. But decentralizing at least some of the ML expertise will increase the speed of adoption and improve alignment between the ML models and the business needs.

Note that many organizations evolve into this hybrid model. As a leader, it might be wise to plan for this evolution.

### Background and Organizational Description

The disadvantages of this hybrid implementation draw from each of the centralized and decentralized ML organizational structures. Inefficiencies might exist in decentralizing staffing and implementing ML throughout the organization. The business units might not understand ML well and might have particularly bad implementations. This can be especially problematic if the failures relate to privacy or ethics. Meanwhile, the centralized infrastructure might create friction for the decentralized modeling teams. And, the centralized infrastructure may feel more complex and be costlier. However, the longer the company is around, the more that the centralized infrastructure model will pay off.

### Process

One way to think about the impact of this hybrid implementation is to reconsider the cart abandonment example in [“Bigger Isn’t Always Better”](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch14.html#bigger\_isnapostrophet\_always\_better). In this case, provided the modeling team lives in the web store product team, these team members are more likely to notice the problem quickly and to realign the metrics of the model with sales instead of simply maximizing cart size. They will also think of possible _new features_ like “purchase half this yarn now, and set aside half of it for later in one month.” In all of these cases, ML has triggered the conversation about how to be more customer friendly.

But say the problem occurs across organizational divisions: a web store model causes a problem for purchasing. In those cases, resolution is likely to be much slower, as the purchasing team tries to convince the web team that the model is causing problems. In these cases, organizational culture will have to support cross-team model troubleshooting and even development.

Consider the processes recommended for scenario 1 and 2 to see if they may help alleviate possible disadvantages to this implementation:

* De-risked testing of combinations of changes
* Independent evaluation of changes
* Reviews:
  * ML team(s) findings documentation and review
  * Business reviews of model outcomes
  * Triage or ML production meetings
* Minimum standards for technical infrastructure
* Training or teams for quality, fairness, ethics, and privacy issues

### Rewards

In the hybrid scenario, YarnIt senior management should reward business units for utilizing the centralized infrastructure, to prevent them from developing their own, duplicative infrastructure. Centralized infrastructure teams should be rewarded for meeting the needs of the other business units. Measuring and rewarding adoption, while also encouraging use of central infrastructure in almost all cases, makes sense.

Central infrastructure teams should have a plan to identify key technology developed in the business units and extend its use to the rest of the company. And from a career development perspective, ML modelers from the business units should be able to rotate onto the central infrastructure team for a period of time to understand the services available and their constraints, as well as to provide an end-user perspective to those teams.

### People

To function well across YarnIt, all of these teams will need to have a company-wide perspective of their work. The infrastructure teams need to build infrastructure that works and that is genuinely useful and desirable for the rest of the company. The ML teams embedded with the business need to have a mindset that cooperation is best, so they should be looking for opportunities to collaborate across divisions.

### Default Implementation

Here is an oversimplified default implementation of the centralized infrastructure / decentralized modeling model:

* Hire a centralized team (leader) with ML infrastructure and production skills:
  * Hire software engineering staff to build ML infrastructure.
  * Hire ML production engineering staff to run the infrastructure.
* Each product team hires experts in their own business units:
  * Hire ML engineering staff to build models directly with the product teams.
* Develop a practice of internal reports of findings for review by senior stakeholders.
* Establish company-wide technical infrastructure standards.
* Plan compensation by successful implementation and compensate not only for meeting business goals but also for efficiency of utilizing the central infrastructure.
* Select processes that will aid in cross-organizational collaboration such as cross-team ML findings reviews.
* Start a privacy, quality, fairness, and ethics program to establish standards and compliance monitoring for those standards.

## Conclusion

Introducing ML technologies into an organization for the first time is difficult, and the best path will necessarily be different from organization to organization. Success will require thinking, in advance, about the organizational changes necessary to ensure success. This includes being honest about the missing skills and roles, process changes, and even entire missing suborganizations. Sometimes this can be solved by hiring or promoting the right new senior leader and charging them with the implementation. But often the necessary organizational changes will span the whole company.

[Table 14-1](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch14.html#summary\_of\_structures\_and\_requirements) summarizes various organizational structures, and their impacts and requirements around people, process, and rewards.

Some teams, such as the production engineering or software engineering teams, will not require significant ML skills to start being effective. But they will benefit from study groups, conference attendance, and other professional development activities. We also need to build ML skills among business leaders. Identify a few key leaders who can understand the benefits and complexities of adding ML to your infrastructure.

The biggest barrier to success, though, is often the tolerance of the organization’s leadership for risk, change, and details, as well as for sticking with ML because it might take a while to manifest returns. To make progress with ML, we have to take risks and change what we do. And ensuring that the teams are focused on business results will help adoption. This involves altering processes that work well and changing the behavior of successful teams and leaders. It often means risking successful lines of business as well. To understand the risks being taken, leaders have to care about some of the details of the implementation. Leaders need to be tolerant of risk but plainly care about the ultimate reliability of their ML implementation.

|             | Centralized ML infrastructure and expertise                                                                                                                                                                                                                                                                                                                                                                                                          | Decentralized ML infrastructure and expertise                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Hybrid with centralized infrastructure and decentralized modeling                                                                                                                                                                                                                                                                                                                                                       |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **People**  | <p>Specialized teams with clear focus, nexus of influence on ML priorities and investments.<br><br>Experimentation mindset is required.<br><br>Leaders/senior members need to be effective collaborators and influencers outside of their own organization.<br><br>Teams need to be the champions of ML quality, fairness, ethics, and privacy across the company.</p>                                                                               | <p>ML expertise is spread across various teams and is often both duplicated and sparse.<br><br>Leaders/senior members need to encourage and enforce internal communities.<br><br>Siloed decisions will cause bad/inconsistent customer experiences and thereby significant business impacts.<br><br>Teams across all product areas need to gain expertise on ML quality, fairness, ethics, and privacy.</p>                                                                                                                 | <p>Centralize ML infrastructure and modeling for common/core business use cases but encourage individual model development for specific needs.<br><br>Avoids duplication and improves team efficiencies and consistency, especially at scale.<br><br>ML quality, fairness, ethics, and privacy need to be in the DNA across all departments.</p>                                                                        |
| **Process** | <p>Needs a lot of cross-functional collaboration for making decisions and sharing knowledge.<br><br>Key stakeholders across business units need to review proposals and results, and launch plans collectively.<br><br>Decentralized/independent model evaluation is needed to ensure and measure business goals and impacts.<br><br>Validate changes in combinations to avoid unintentional regressions and establish go/no-go review meetings.</p> | <p>Needs a lot of documentation around best practices, knowledge, evaluation, and launch criteria to maintain consistency, <em>or</em> a deliberate decision not to maintain any outside of local team scope (which is problematic for ML).<br><br>Key stakeholders across business units need to review proposals and results, and launch plans collectively.<br>Well-structured and moderated go/no-go meetings are needed to avoid delays.<br><br>Establish standards for technical infrastructure for ML pipelines.</p> | <p>Needs cross-functional collaboration and decent documentation between infrastructure and individual product teams on a project/program basis.<br><br>Establish cross-functional teams with clear accountability.<br><br>Regular cross-functional syncs should occur at project/program level.<br><br>Key stakeholders across business units need to review proposals and results, and launch plans collectively.</p> |
| **Rewards** | <p>On top of overall quality and meeting business goals, individual/team performance needs to be measured based on the effectiveness of cross-functional collaboration.<br><br>Establish mechanisms to compensate both ML and product teams together for successful AI feature launches.</p>                                                                                                                                                         | On top of overall quality and meeting business goals, individual/team performance needs to be measured based on consistency, published quality standards, and operating internal ML communities.                                                                                                                                                                                                                                                                                                                            | On top of overall quality and meeting business goals, individual/team performance needs to be measured based on reusability, evolution of common infrastructure, and speed of execution.                                                                                                                                                                                                                                |

[1](https://learning.oreilly.com/library/view/reliable-machine-learning/9781098106218/ch14.html#ch01fn144-marker) In most of the literature, this is presented as an algorithmic or technical problem. Of course, it is those things but it is also very much an organizational problem. If we don’t have the decision and management framework to evaluate changes separately and strategize about how to deploy them, we will not be able to correctly prioritize that work.
