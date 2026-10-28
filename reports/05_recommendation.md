# 5. Recommendation for marketing

## The one-paragraph version

Keep the Purr-sonal Perks campaign, but **only for members in their first 12
months**, and **never send retention discounts to members with 12+ months of
tenure**. Sent to everyone, the campaign cuts cancellations by 24% and still
loses money on average (-$1.50 per member over 12 months; negative in
90% of re-simulations), because most people who get
the 20% discount were never going to leave, and long-tenured members leave
*more* when we remind them we exist. Targeted by tenure, the same campaign is
worth about **$57k over 12 months** on today's base. That is roughly
6% of the $1,013k revenue at risk - so the campaign is a useful lever, not the
retention strategy. The bigger levers are onboarding, capacity and the
moment a member adopts a cat.

## What to do, in priority order

**1. Re-target the campaign by tenure, starting next month.**
Send to members under 12 months (65% of the base). Expected value
$7.79 per eligible member (95% CI $3.54 to $12.04). Do not use the churn
score alone to pick recipients: the top-30%-risk list is worth $3.25 per member and
lost to the tenure rule in 80% of re-simulations. **Risk tells you who might
leave; only the experiment tells you who will respond.** Tell finance the
campaign is cash-negative for the first quarter (-$2.93 per member over 90 days).

**2. Put 12+ month members on a "do not discount" list.**
The campaign raised their cancellations by 4.3 pp (Holm p = 0.0011), replicated in
both halves of the data. For this group, test recognition without a price
message (a members' night, naming a kitten after them) - but as a hypothesis
with its own test, not a rollout.

**3. Make the first 90 days the product.**
Members under 3 months are 27% of the base and 41% of revenue at risk, and they
respond to the campaign most strongly (12.7 pp fewer cancellations). The model's top signal
is visits per month of tenure; build an onboarding journey that triggers when a
new member is below ~3 visits a month (personal cat intro, a booked second visit,
a first event invitation).

**4. Treat crowding as an operations problem, and test it.**
The model barely uses booking failures (they move with engagement), but
"too crowded" is the reason on 25% of cancellation forms since September. That
disagreement is the point: SHAP ranks predictive signal, the forms report a
cause. Test priority booking for at-risk members, or peak-time capacity, with
a holdout - do not assume either the model or the forms is right.

**5. Turn cat adoption from an exit into a tier.**
Adopting a cafe cat multiplies predicted cancellation odds by about
1.9x. That is the mission working. Offer an "Alumni" membership at the moment of
adoption (lower fee, events, vet and supplies partner perks) and measure it
against a holdout.

**6. Test annual billing conversion - carefully.**
Monthly-billed members hold 85% of revenue at risk, but annual members look
safer partly because committed people choose annual. Only a randomised
annual-upgrade offer (around month 3-4) can tell whether converting someone
changes their behaviour.

## Next experiment

Three arms, members under 12 months only: control / guest pass + personalised
message, **no discount** / current offer. Keep the control arm untreated for
six months afterwards so it doubles as a long-term holdout (does the retention
effect last beyond 60 days?).
4,757 eligible members gives 1,585 per arm; at alpha = 0.025 per comparison
(two comparisons) the MDE is 26% relative on a 14.0% baseline. The current offer
creates $21.64 of retention value per treated member against a $9.66 cost; the
question is how much of that value the 20% discount is actually buying.
If guest-pass-only retains nearly as well, it is the better business.

## What not to do

* Don't report the +33% week-1 visit lift as the campaign's impact. It is gone by week 3.
* Don't read the experiment before day 60 and stop on a good number. Daily
  peeking turns a 5% false-positive rate into 27%.
* Don't use SHAP drivers as a to-do list. They say where the model sees risk,
  not what will change it. Every lever above is phrased as a test for that reason.
* Don't trust a churn model with AUC above ~0.9 until someone has run the
  leakage checklist.

## Limitations

The data is simulated with known ground truth so the traps can be verified;
the effect sizes, prices and costs are illustrative. Revenue at risk assumes
constant hazard over 12 months. Policy values are estimated on the same
experiment used to choose the policy (mitigated by cross-fitting and
pre-registration, not eliminated) - the holdout in the next test is the real check.
