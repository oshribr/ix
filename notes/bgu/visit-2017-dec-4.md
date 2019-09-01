# University visit on Dec 4th, 2017

I went to BGU to talk to faculty members at CS and ISE
departments who make give insights on problem statement
and solution approaches for deterioration prediction.

I met with 

* **Kobi Gal**,
http://www.ise.bgu.ac.il/engineering/PersonalWebSite1main.aspx?id=djueVuMe, and his PhD student Reuth Mirsky. They work on computer
interventions to human behaviors. Kobi is a former student of
Sarit Krauss from Bar-Ilan University.

* **Ronen Brafman**, https://www.cs.bgu.ac.il/~brafman/. Ronen is a
professor with the CS department and a world-level expert in
planning and decision making under uncertainty.

I also planned to meet with **Eitan Bachmat**,
https://www.cs.bgu.ac.il/~ebachmat/, who has recently started
working on healthcare, but will have to do it at a later time
although with a better introduction.

I also had a few minor meetings which other faculty members.

## The story

During the meetings, I told the story on improving ICU by 
issuing earlier more relevant alerts. I briefly told basically
what Maor told during the meetup last August --- here is [the
narrative] I wrote for myself before going to Beer Sheva.

I asked the counterparts to discuss with me both the problem
statement (that is, **what** formally we want to find/optimize)
and, given that we agreed on some form of problem statement, 
the methods to solve the problem. In what follows  I compile
all discussions. I first give a possible problem statement, and
then briefly address methods which were mentioned.

## Problem Statement

We want to _select_ _times_ at which
alerts are issued as well as _types_ of alerts, that is, to what
possible future deterioration the alerts are related. We want to
select times in such a way that the stay duration of the patient
is _minimized_ provided the patient is going to survive, and the
probability of survival is _maximized_. This is known as a
problem of reinforcement learning or probabilistic planning:
there are actions with uncertain outcome and a value function on
the finl state. 

An example value function may be `Pr(survival)^a * exp(- bT)`
(or `a * log Pr(survival) - b * T`, which is the same function
in a more convenient way). Such formulation is made under
assumption  that the _doctor_ takes a perfect action given an
alert. Since in practice the ability of a doctor to respond to
alerts is limited by the number of doctors and frequency of
alerts, we can model that by introducing _alert cost_ as a
function of number of alerts of each type `Ca(N_1, ..., N_k)`.
Just for illustration
for the case when the function is linear and the cost is the
same for any alert, the value function becomes:
`a * log Pr(survival) - b * T - Ca * N` (`N` is the number of
alerts).  This is what we want to _optimize_ by deciding _when
and whether_ to raise alerts.

To complete the problem statement, we must define relative to
what to select alert times and types, as we obviously cannot
select them during the algorithm execution relative to future
deteriorations. Therefore, we identify

1. Certain critical points in the patient's temporal process
   (changepoints or highlights).
2. Interventions that follow the highlight (the first
   significant intervention in the simplest case).

Then, we finally get an optimization problem of decision making
under uncertainty which _maximizes the value `V = a*log
Pr(survival) - b * T - Ca * N` by making a decision `Highlight ->
(Time, Type)` that is by accepting features of a highlight as
input and providing optimal time and type of alert _relative_ to
the highlight as the output. In a more complicated non-Markovian
setting, we can make the decision depend on all past highlights
rather than just the most recent one.

Finally, since we do not know the probability of survival, only
the outcome, we can just introduce the cost of death `Cd` and
penalize by this cost the patient death.

_Formally_:

Find policy `Py`:

    Py(Highlights) -> (Time, Type)

Such that  expectation of V

    V(T, N) = - b * T      - Ca * N     if survived
            =         - Cd - Ca * N     otherwise

is maximized. (Note that `-b * T` term is only for the
survivals, as we do not want the policy to kill hopeless
patients as soon as possible by silencing alerts --- or do we?)

## Solution approaches

I am describing this only briefly and will expand later.

### POMDP planning

One way to solve this problem is to represent it as a partially
observable Markov decision process (POMDP), learn the process
parameters from the data, and then find an optimal policy using
one of policy optimization algorithms. A POMDP is defined by
actions, observations, and costs. At every time point we either
issue an alert of a particular time, or remain silent --- and
these are our actions (one for each alert type and silence).
Observations are whatever data we have about the patient at each
time step. 

A problem with straightforward application of this approach is
that given the problem size (number of features and number of
steps) the POMDP solution will be too computationally expensive. 

### Regression + optimization

Let us assume that we are able to identify interesting points
(changepoints/highlights) independently of the rest of the
problem, using unlabeled data. Then, we can use the available
labeled data to learn regression which predicts stay duration
and outcome based on (changepoint, following-intervention)
pairs, independently for each pair in the simplest case.

Then, having learned such regression function we can optimize
the function _input_ (namely the intervention time and type) to
maximize the reward `V`. The intervention time and type given
changepoint resulting in maximum predicted value is the desired
alert time and type.

This can practically be implemented using unsupervised learning
on time series for changepoint detection, followed by supervised
learning where only immediate interventions (such as those
following from an alert) are tagged.  Changepoint detection can
be improved by filtering of false positives (explaining away
and learning).
