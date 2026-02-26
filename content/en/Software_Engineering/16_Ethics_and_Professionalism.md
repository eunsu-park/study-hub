# Lesson 16: Ethics and Professionalism

**Previous**: [Team Dynamics and Communication](./15_Team_Dynamics_and_Communication.md) | **Next**: [Overview](./00_Overview.md)

---

Software engineering has produced systems that land spacecraft on Mars, sequence the human genome, and connect billions of people. It has also produced systems that discriminate against loan applicants based on race, that failed the safety-critical controls of passenger aircraft, and that harvest personal data without meaningful consent. The power of software to affect human lives at scale — for good and ill — creates genuine ethical obligations for the people who build it. This lesson examines those obligations: the professional codes that define them, the classic failures that illustrate what happens when they are ignored, and the practical questions every working engineer eventually faces.

**Difficulty**: ⭐⭐

**Prerequisites**:
- [What Is Software Engineering](./01_What_Is_Software_Engineering.md) — the scope and responsibility of the discipline
- No prior ethics background required

## Learning Objectives

After completing this lesson, you will be able to:

1. Summarize the ACM/IEEE Software Engineering Code of Ethics and its eight principles
2. Analyze historical software failures through an ethical lens
3. Identify ethical dimensions in common software engineering decisions
4. Explain the principles of responsible AI: fairness, transparency, and accountability
5. Describe intellectual property concepts relevant to software engineers
6. Discuss privacy regulations (GDPR, CCPA) at a conceptual level
7. Outline career paths and professional development options in software engineering
8. Explain the concept of sustainable software engineering

---

## 1. Software Engineering as a Profession

What distinguishes a *profession* from an *occupation*? Traditionally, professions are characterized by:
- A specialized body of knowledge requiring formal education
- Ethical obligations to clients, the public, and the profession itself
- Self-regulation through professional bodies and codes of conduct
- Certification or licensure to practice

Medicine, law, and civil engineering are the paradigmatic professions. Software engineering has most of these characteristics but remains incompletely professionalized: there is no universal licensure requirement, no mandatory professional membership, and ethical obligations are unevenly understood and enforced.

This incompleteness is both a freedom and a risk. Software engineers rarely need a professional license to write code that runs on safety-critical systems, yet their decisions can have life-or-death consequences. The aerospace and medical device industries have some regulatory requirements (FAA DO-178C for aviation software, IEC 62304 for medical devices), but most software is written with no regulatory oversight at all.

The argument for treating software engineering as a profession — even where it is not legally required — is that the societal impact of software has grown to the point where informal, good-faith effort is insufficient. Systematic ethical frameworks and professional standards are necessary.

---

## 2. ACM/IEEE Software Engineering Code of Ethics

The most widely cited ethical framework for software engineering is the **ACM/IEEE-CS Software Engineering Code of Ethics and Professional Practice** (1999), developed jointly by the Association for Computing Machinery and the IEEE Computer Society.

The code has eight principles, each elaborating obligations to different stakeholders:

### Principle 1: Public

> Software engineers shall act consistently with the public interest.

This is the foundational principle. It requires that engineers consider impacts beyond the immediate client or employer: safety, privacy, non-discrimination, and the honest representation of software capabilities. When organizational pressure conflicts with public safety, public safety takes precedence.

### Principle 2: Client and Employer

> Software engineers shall act in a manner that is in the best interests of their client and employer, consistent with the public interest.

Engineers owe competent, honest service to those who hire them. This includes: using only honest, legal means; disclosing conflicts of interest; keeping confidences; and not deceiving clients about the state of a project.

### Principle 3: Product

> Software engineers shall ensure that their products and related modifications meet the highest professional standards possible.

This means: striving for high quality, not knowingly releasing software with defects that could harm users, providing adequate documentation, and respecting the privacy and security of data.

### Principle 4: Judgment

> Software engineers shall maintain integrity and independence in their professional judgment.

Engineers should not subordinate their professional judgment to financial pressure, career pressure, or the preferences of powerful colleagues. If you believe an architectural decision is wrong, document and communicate your concern — do not simply defer to whoever has seniority.

### Principle 5: Management

> Software engineering managers and leaders shall subscribe to and promote an ethical approach to the management of software development and maintenance.

Those with authority over others have additional responsibilities: fair treatment, reasonable expectations, honest communication, and not asking subordinates to behave unethically.

### Principle 6: Profession

> Software engineers shall advance the integrity and reputation of the profession consistent with the public interest.

This includes supporting educational initiatives, not misrepresenting capabilities, sharing knowledge, and participating in professional communities.

### Principle 7: Colleagues

> Software engineers shall be fair to and supportive of their colleagues.

Fair credit for contributions, honest assessment of colleagues' work, mentoring junior engineers, and not sabotaging others' professional reputations.

### Principle 8: Self

> Software engineers shall participate in lifelong learning regarding the practice of their profession and shall promote an ethical approach to the practice of the profession.

Staying current with the field, recognizing the limits of one's knowledge, and being honest about uncertainty.

---

## 3. Ethical Dilemmas in Software Engineering

Ethics becomes difficult precisely because it involves genuine conflicts between legitimate values. Common ethical tensions in software engineering:

### Safety vs Schedule

A ship-it culture that rewards speed can suppress safety concerns. Engineers who believe a product is not safe to release face pressure: raise the concern and be labeled a blocker, or stay quiet and ship something that might harm users.

The Therac-25 (see Section 5) is the most cited example: safety concerns about race conditions were known and dismissed due to schedule and budget pressure.

**Ethical obligation**: An engineer who believes a product poses safety risks to users has a professional obligation to escalate that concern through all available channels, document it in writing, and — if organizational channels fail — consider whistleblowing or refusing the work.

### Privacy vs Functionality

Many features can be built more easily by collecting and retaining user data. Search history improves recommendations. Location data enables better logistics. But every piece of data collected is a liability: it can be stolen, misused, subpoenaed, or used to manipulate users.

The ethical question is not "can we collect this data?" but "should we collect this data, and if so, under what constraints and for how long?"

### Bias vs Business Logic

Machine learning models trained on historical data encode historical patterns — including historical discrimination. A resume screening model trained on historical hiring data may learn to discriminate based on gender or ethnicity if those patterns existed in past decisions. The model is technically "accurate" (it predicts outcomes similar to historical decisions) while being discriminatory.

Engineers who deploy such systems have an obligation to audit for disparate impact, design for fairness, and not hide behind "the algorithm decided."

### Environmental Impact

Training large AI models consumes electricity equivalent to hundreds of transatlantic flights. Data centers account for 1–2% of global electricity consumption. As engineers, we make decisions about computational efficiency that have environmental consequences.

---

## 4. Responsible AI

Artificial intelligence systems raise ethical concerns that are distinct in scale and opacity from traditional software.

### Fairness

An AI system is *fair* in a technical sense if its error rates and outcomes do not systematically differ across protected groups (race, gender, age, disability status). Multiple mathematical definitions of fairness exist and are mutually incompatible — you cannot simultaneously achieve equal error rates across groups and equal selection rates in a general case (Chouldechova, 2017). This is not a technical problem to be solved; it is a policy choice about which fairness criterion matters most for a given use case.

Practical steps:
- Audit training data for underrepresentation and historical bias
- Measure model performance disaggregated by demographic group, not just overall accuracy
- Include domain experts and affected communities in fairness criteria design

### Transparency and Explainability

Complex models (deep neural networks, ensemble methods) are difficult or impossible to explain at the individual prediction level. When an AI system denies a loan application, rejects a resume, or flags a medical image, the affected person deserves an explanation.

Explainability methods (LIME, SHAP, attention visualization) provide approximate explanations. The EU AI Act (2024) requires high-risk AI systems to be interpretable, pushing toward simpler, more explainable designs in regulated domains.

### Accountability

When an AI system causes harm, who is responsible? The data scientists who trained it? The product managers who deployed it? The company that sold it? The regulator who approved it?

Accountability structures must be deliberately designed:
- Assign a responsible human for every consequential AI decision
- Document model behavior, training data, and known limitations (model cards, datasheets for datasets)
- Maintain the ability to audit and reverse AI decisions

### Automation Bias

People tend to over-trust automated systems — even when those systems are wrong. Medical professionals who receive an AI diagnosis tend to agree with it at higher rates than warranted by its accuracy. Pilots who receive autopilot inputs tend to trust them even when warning signs suggest malfunction.

Designing AI systems responsibly means designing interfaces that support, not replace, human judgment — especially for safety-critical decisions.

---

## 5. Historical Case Studies

History offers vivid examples of what happens when software engineering ethics fail. These cases are studied in professional curricula because their lessons generalize.

### Therac-25 (1985–1987)

The Therac-25 was a radiation therapy machine that, between 1985 and 1987, delivered lethal radiation overdoses to at least six patients, killing several. The root cause was a race condition in the control software — a class of bug that had been masked in earlier hardware-based safety interlocks that the Therac-25 removed without replacing with software equivalents.

**What went wrong ethically:**
- The manufacturer (AECL) was slow to respond to incident reports from hospitals
- When told of accidents, AECL initially denied the software could be at fault
- Regulators had limited capacity to audit embedded software
- The software engineer who discovered the race condition had documented it, but the documentation was not acted upon

**Lessons:** Safety-critical software requires independent verification, not just developer testing. Incident reports must be investigated seriously, not dismissed. Removing hardware safety mechanisms requires equivalent software replacements.

### Boeing 737 MAX MCAS (2018–2019)

The Maneuvering Characteristics Augmentation System (MCAS) was software added to the Boeing 737 MAX to compensate for the aerodynamic effects of larger, repositioned engines. MCAS relied on a single angle-of-attack sensor. When that sensor failed, MCAS repeatedly pushed the nose of the aircraft down. Two crashes killed 346 people.

**What went wrong ethically:**
- MCAS was not disclosed to pilots in training materials to avoid costly simulator requirements
- The system's dependence on a single sensor without redundancy violated standard aviation safety principles
- Internal Boeing communications revealed that employees had known about concerns and were pressured not to raise them with regulators
- Regulatory oversight had been delegated to Boeing itself, creating a conflict of interest

**Lessons:** Safety-critical design decisions should not be made on the basis of cost and schedule. Regulators must maintain independent competence to audit, not just rubber-stamp manufacturer self-certification. Engineers who raise safety concerns must be protected, not silenced.

### Volkswagen Emissions Defeat Device (2015)

Volkswagen programmed its diesel vehicles to detect when they were being tested for emissions compliance and switch to a low-emissions mode. Under real driving conditions, the vehicles emitted up to 40 times the legal limit of nitrogen oxides.

**What went wrong ethically:**
- Engineers and managers deliberately designed software to deceive regulators
- The program was known internally and involved multiple levels of the organization
- The software was presented as a technological achievement in clean diesel efficiency

**Lessons:** Writing code designed to deceive regulators is fraud, not engineering. The existence of organizational pressure does not eliminate individual ethical responsibility. When directed to implement something deceptive, the appropriate response is refusal and escalation, not compliance.

---

## 6. Intellectual Property in Software

Software engineers regularly encounter intellectual property (IP) considerations that have legal and ethical dimensions.

### Copyright

Software source code is automatically protected by copyright upon creation. The copyright holder (typically the employer for work created within the scope of employment) has exclusive rights to reproduce, distribute, and create derivative works.

**Implications for engineers:**
- Code you write at work is generally owned by your employer (check your employment agreement)
- Copying code from a non-compatible open-source license into your project may create legal exposure
- Submitting employer code to open-source projects without authorization may violate your employment agreement

### Open Source Licenses

Open source licenses grant users the right to use, study, modify, and distribute software, subject to conditions. Understanding license compatibility is part of a software engineer's professional responsibility.

Key license categories:

| Category | Example Licenses | Key Condition |
|----------|-----------------|--------------|
| Permissive | MIT, Apache 2.0, BSD | Keep copyright notice; few other restrictions |
| Weak copyleft | LGPL, MPL | Modifications to the licensed code must be shared; proprietary code can link to it |
| Strong copyleft | GPL, AGPL | Any work that includes the licensed code must be distributed under the same license |

The AGPL (Affero GPL) extends strong copyleft to network use — software running over a network must release its source under AGPL conditions, which is why many companies prohibit AGPL dependencies in commercial products.

### Patents

Software patents are controversial and vary by jurisdiction. In the US, software algorithms can be patented; in Europe, they generally cannot (though patents on software's technical implementation are possible). Engineers should be aware that implementing an algorithm described in a patent without a license may create legal exposure, even if the algorithm is published in a paper.

### Trade Secrets

Code and algorithms that provide competitive advantage can be protected as trade secrets, provided the company takes reasonable steps to keep them confidential (access controls, NDAs). Unlike patents, trade secrets do not require disclosure and do not expire — but they lose protection if the information becomes public.

---

## 7. Privacy and Regulations

### GDPR (General Data Protection Regulation)

The GDPR, effective 2018 in the EU, established rights for individuals over their personal data and obligations for organizations that process it. Key principles relevant to software engineers:

- **Data minimization**: Collect only what is necessary for the specified purpose
- **Purpose limitation**: Data collected for one purpose cannot be used for another without consent
- **Storage limitation**: Data should not be retained longer than necessary
- **Security**: Appropriate technical and organizational measures must protect personal data
- **Privacy by design**: Privacy protections must be built into systems from the start, not added as an afterthought

Engineering implications: GDPR compliance must be considered during system design, not just as a legal checkbox. This means designing deletion workflows, data access logs, consent management systems, and data retention policies.

### CCPA (California Consumer Privacy Act)

The CCPA (2020) gives California residents rights over their personal information: the right to know what data is collected, the right to delete, and the right to opt out of data sale. Similar laws have been enacted in other US states.

### Privacy by Design

Privacy by Design (PbD), formalized by Ann Cavoukian, articulates seven foundational principles:
1. Proactive, not reactive — anticipate privacy risks before they occur
2. Privacy as the default setting — maximum privacy without user action
3. Privacy embedded into design — not bolted on
4. Full functionality — privacy AND functionality, not a zero-sum trade
5. End-to-end security — full lifecycle protection
6. Visibility and transparency
7. Respect for user privacy

---

## 8. Whistleblowing and Organizational Ethics

Sometimes an organization directs engineers to do something they believe is harmful, illegal, or unethical. What are the options?

**Internal escalation**: Raise the concern with management, legal, compliance, or ethics hotlines. Document every communication in writing. This is almost always the right first step.

**Legal whistleblowing**: Many jurisdictions offer legal protections for employees who report illegal activity to regulators. In the US, the SEC Whistleblower Program offers financial rewards for reporting securities fraud. The Sarbanes-Oxley Act protects employees of public companies who report fraud.

**Public disclosure**: Leaking information to journalists or the public is the option of last resort, with significant legal and career risk. It may be the only effective option when internal channels are captured or suppressed.

**Refusal**: An engineer can refuse to implement something they believe is unethical or illegal. This may cost them their job but preserves their professional integrity.

The ACM Code of Ethics makes clear: "Computing professionals should not allow organizational pressures to override their ethical obligations." This is easy to say and difficult to live — it requires organizational support, legal protection, and personal courage.

---

## 9. Sustainable Software Engineering

The environmental impact of software systems is a growing professional concern.

### The Scale of the Problem

- Data centers consumed approximately 200–250 TWh of electricity globally in 2022 — roughly 1% of total global electricity demand
- Training a single large language model can emit as much CO2 as five cars over their lifetimes (Strubell et al., 2019)
- Cryptocurrency mining is energy-intensive by design
- The manufacturing of hardware (chips, servers, devices) is itself energy- and resource-intensive

### Principles of Sustainable Software Engineering

The Green Software Foundation has defined eight principles for sustainable software engineering:

1. **Carbon**: Build applications that are carbon-efficient
2. **Electricity**: Build applications that are energy-efficient
3. **Carbon Intensity**: Consume electricity with the lowest carbon intensity — schedule work during times of high renewable energy availability
4. **Embodied Carbon**: Consider the carbon cost of hardware manufacturing, not just runtime electricity
5. **Energy Proportionality**: Run hardware at high utilization rather than idle servers
6. **Networking**: Reduce the amount of data transmitted across networks
7. **Demand Shaping**: Instead of always meeting demand, sometimes shift or reduce demand
8. **Optimization**: Improve system efficiency continuously

### Practical Actions

- **Profile and optimize**: Many systems consume 10x more resources than necessary due to inefficient algorithms or unnecessary computation
- **Right-size infrastructure**: Avoid massively over-provisioned servers
- **Use efficient algorithms**: An O(n²) algorithm consuming server resources has a carbon cost as well as a financial one
- **Design for device longevity**: Web and mobile applications that require the latest hardware contribute to electronic waste by making older devices unusable

---

## 10. Professional Development and Career Paths

Software engineering offers remarkable career diversity. Understanding the landscape helps engineers make informed choices.

### Career Paths

**Individual Contributor (IC) track**: Engineers who prefer to remain deeply technical can follow a career that progresses from junior engineer → software engineer → senior engineer → staff engineer → principal engineer → distinguished engineer/fellow. Senior IC roles are equivalent in seniority and compensation to senior management roles.

**Engineering management track**: Engineers who prefer organizational leadership move to engineering manager → senior manager → director → VP → CTO. Management requires different skills: hiring, performance management, organizational design, and executive communication.

**Specialist paths**: Security engineering, site reliability engineering, data engineering, machine learning engineering, DevRel (developer relations), technical writing, and engineering program management are specialized roles that blend engineering skill with domain expertise or other disciplines.

### Professional Development

**Certifications**: While software engineering does not generally require licensure, certifications can signal specific knowledge areas. Examples: AWS/GCP/Azure cloud certifications, CKAD/CKA for Kubernetes, CISSP for security, PMP for project management. Certifications are most valuable in fields where clients or regulators need assurance of competence.

**Conferences**: Academic conferences (ICSE, FSE) and industry conferences (KubeCon, PyCon, QCon, Strange Loop) are venues for learning, networking, and professional development. Contributing talks is a significant professional milestone.

**Open source participation**: Contributing to open source projects builds a public portfolio, develops collaboration skills, and contributes to the commons. Starting small — documentation fixes, bug reports, small code contributions — is the right path for most contributors.

**Writing and speaking**: Technical blog posts, conference talks, and published papers communicate ideas, build reputation, and force the clarity of thought that imprecise understanding cannot produce.

### Learning Mindset

Technology changes fast. A language or framework that is dominant today may be legacy in ten years. Professional longevity in software engineering requires:
- Willingness to continuously learn new languages, tools, and paradigms
- Strong fundamentals (algorithms, data structures, systems thinking, security) that transfer across tools
- Understanding of the domain your software serves (healthcare, finance, logistics) deepens over time and becomes a competitive advantage
- Soft skills (communication, influence, conflict resolution) become increasingly important as seniority grows

---

## Summary

Software engineering ethics is not a theoretical abstraction — it is the set of practical principles that govern decisions made every day in the industry. From whether to ship software that has known safety defects to how personal data is stored, from how AI systems are audited for bias to how open-source licenses are respected, ethical questions are embedded in engineering practice.

Key concepts:
- The **ACM/IEEE Code of Ethics** defines obligations to the public, clients, the profession, and colleagues — with public interest as the foundational principle
- Historical failures (Therac-25, Boeing 737 MAX, VW emissions) illustrate the life-and-death consequences when ethical obligations are overridden by commercial pressure
- **Responsible AI** requires attention to fairness, explainability, accountability, and the risk of automation bias
- **Intellectual property** — copyright, open-source licenses, patents, trade secrets — creates legal and ethical obligations that engineers must understand
- **Privacy by design** embeds privacy protection into systems from the start, not as an afterthought
- **Sustainable software engineering** recognizes that the environmental impact of computation is a professional responsibility
- Professional development is a lifelong process; the most durable skills are fundamentals, communication, and domain expertise

---

## Practice Exercises

1. **Ethical Analysis — Airline Scheduling**: A team is building an automated airline crew scheduling system that reassigns crews after irregular operations (delays, cancellations). Management wants to optimize for cost minimization. A safety analyst notes that fatigued crew scheduling correlates with incidents. Apply the ACM/IEEE Code of Ethics principles to analyze this scenario. Which principles are most relevant? What obligations do the engineers have? What would you do if management overrode your safety concerns?

2. **Bias Audit Design**: A company uses a machine learning model to score job applicants. You are asked to audit the model for fairness. Describe your audit plan: what data would you analyze, what fairness metrics you would compute, what you would do if you found statistically significant disparate impact against a protected group, and what you would document in the model card.

3. **Open Source License Compatibility**: Your team wants to incorporate three open-source libraries into a commercial product: Library A (MIT), Library B (GPL v2), and Library C (Apache 2.0). Assess whether each library can be incorporated into a closed-source commercial product without open-sourcing your own code. What restrictions does each impose? Which library creates a potential conflict?

4. **Case Study — Volkswagen Defeat Device**: Assume you are a software engineer at Volkswagen in 2010 and are asked to implement the emissions defeat device logic as a routine feature. You understand what it does. Walk through your ethical decision-making process: Who are the stakeholders? What ethical principles apply? What options do you have? What would you actually do, and why?

5. **Career Planning**: Map out a 10-year career plan for a software engineer interested in becoming a principal engineer focused on distributed systems. Include: skills to develop at each stage, certifications or credentials worth pursuing, open-source projects or communities to engage with, and how you would build a public reputation in the field.

---

## Further Reading

- **Codes of Ethics**:
  - ACM/IEEE Software Engineering Code of Ethics — acm.org/code-of-ethics (full text, freely available)
  - ACM Code of Ethics and Professional Conduct (2018) — acm.org

- **Books**:
  - *A Hippocratic Oath for Computer Scientists* — concept discussed broadly in professional computing literature
  - *Data Feminism* — Catherine D'Ignazio, Lauren F. Klein (power, bias, and data)
  - *Weapons of Math Destruction* — Cathy O'Neil (algorithmic harm in high-stakes decisions)
  - *The Alignment Problem* — Brian Christian (AI safety and alignment research)
  - *Sustainable Software Engineering* — Green Software Foundation (greensoftware.foundation)

- **Case Studies**:
  - "An Investigation of the Therac-25 Accidents" — Nancy Leveson, Clark Turner, IEEE Computer 1993 (the definitive account)
  - US Department of Justice investigation reports on Boeing 737 MAX (publicly available)
  - FTC reports on VW emissions settlement

- **Regulations**:
  - GDPR full text — gdpr.eu
  - EU AI Act summary — artificialintelligenceact.eu
  - Green Software Foundation Principles — greensoftware.foundation/articles/software-carbon-intensity

---

**Previous**: [Team Dynamics and Communication](./15_Team_Dynamics_and_Communication.md) | **Next**: [Overview](./00_Overview.md)
