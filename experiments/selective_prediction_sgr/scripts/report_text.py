"""Narrative text of the results report (``build_report.py``), one HTML template per section.

Every template is filled with ``str.format`` from a dictionary of numbers that ``build_report.py`` reads from the result
csv files, so the text stays in sync with the data. Display names only; raw ids belong to the appendix.

References: Geifman and El-Yaniv 2017 (SGR, arXiv 1705.08500), Traub et al. 2024 (AUGRC), Angelopoulos et al.
(Learn-then-Test).
"""

from __future__ import annotations

TITLE = "Selective prediction with a risk guarantee on CIFAR-10"
SUBTITLE = (
    "We reproduce the SGR method of Geifman and El-Yaniv (2017), compare uncertainty methods and threshold rules, and "
    "show that a deep ensemble gives more guaranteed coverage than the original model."
)

SECTIONS: dict[str, str] = {}

SECTIONS["glance"] = """
<ul>
  <li>The method of the paper is reproduced: the coverage ranking of Table 1 holds, and at the risk levels the paper
  reports we accept {repro_gain_lo} to {repro_gain_hi} points more images than the paper.</li>
  <li>The guarantee is real. With the plain SGR rule (delta = 0.001) the test risk exceeded the target in
  {sgr_viol_max} of the {n_pairs} seed x split pairs for the methods in this report, at every target. Over all 38
  scores in the appendix the worst case is {sgr_viol_all} (one or two pairs).</li>
  <li>Our improvement: a deep ensemble of five models with the same SGR rule accepts {ens_cov_001} of the images at
  r* = 0.01 with a valid guarantee, against {paper_cov_001} reported by the paper (which uses a weaker bound).</li>
  <li>Ensemble members already help at size two,
  which costs less than {mc_passes} MC dropout passes.</li>
  <li>The empirical threshold accepts more images, but breaks the risk target in about half of the splits.</li>
</ul>
"""

SECTIONS["setup"] = """
<p>The 10,000 CIFAR-10 test images are split 10 times at random into 5,000 <em>selection</em> images (used to choose
the threshold) and 5,000 <em>test</em> images (used to check it). Every number is a mean over 5 training seeds x 10
splits = {n_pairs} pairs. The main model is a VGG-16 with dropout; ResNet-18 is a replication. The confidence level of
the guarantee is delta = 0.001.</p>
<dl class="gloss">
  <dt>Selective risk</dt><dd>error rate on the images the model chooses to answer.</dd>
  <dt>Coverage</dt><dd>share of images the model answers. Higher is better.</dd>
  <dt>r*</dt><dd>the target risk we want to stay below (for example 0.01 = at most 1% errors among the answers).</dd>
  <dt>Certified</dt><dd>a split is certified if the rule found a threshold whose bound is below r*. Otherwise the
  model answers nothing on that split.</dd>
  <dt>Violation</dt><dd>the share of the {n_pairs} pairs whose test risk is above r*.</dd>
  <dt>AURC / AUGRC</dt><dd>area under the risk-coverage curve (and its generalized version). Lower is better; they
  judge the whole ranking of the images rather than one threshold.</dd>
</dl>
<h3>Methods compared</h3>
{methods_table}
<h3>Threshold rules compared</h3>
{rules_table}
"""

SECTIONS["repro"] = """
<p>Table 1 of the paper reports, for several target risks r*, the test coverage that SGR reaches with a VGG-16. The
figure and table compare it with our runs of the same model (the paper model, softmax score).</p>
{fig}
{table}
<p>The ranking of the paper is reproduced. When we cut the threshold at the risk level the paper reports, we accept
{repro_gain_lo} to {repro_gain_hi} points more than the paper. With the SGR rule at delta = 0.001 we end up
{sgr_gap_lo} to {sgr_gap_hi} points below the paper for r* of 0.02 and above. The reason is the confidence level:
the numbers in the paper are much closer to a delta of 0.1 (see the third line in the figure, {d01_002} at
r* = 0.02), while delta = 0.001 is a much stricter bound on only 5,000 selection images. At r* = 0.01 the bound is
met on only {cert_001} of the splits, so the mean coverage ({sgr_cov_001}) is low and has a large spread.</p>
"""

SECTIONS["methods"] = """
<p>Which uncertainty method ranks the images best? The left plots use the whole risk-coverage curve (AURC, AUGRC),
the table adds what the methods deliver with the guarantee.</p>
{fig}
{table}
<p>The deep ensemble is clearly best: AUGRC of {a_ens} against {a_soft} for the plain softmax, and the highest
guaranteed coverage. MC dropout ({a_mc}) is about as good as the softmax of the same dropout model ({a_drop}), so the
{mc_passes} extra passes buy nothing. The ensemble's epistemic score has the lowest AURC, but the AUGRC and the SGR
coverage both prefer its plain max-probability score. Other methods (SWA, SWAG, DDU, VBLL, SNGP and Laplace, see
the appendix) are not better than the softmax. Note that these were only fine-tuned for 20 epochs at learning rate
0.01 from the base model and end below its accuracy ({acc_ft} against {acc_base}), so this comparison is partly a
training-budget effect.</p>
"""

SECTIONS["rules"] = """
<p>Given a score, how should the threshold be chosen? The figure shows the deep ensemble's coverage per rule and
target, with the share of violating splits written next to rules that are not guaranteed.</p>
{fig}
{table}
<p>Among the rules with a guarantee (green), SGR, Learn-then-Test with a fixed sequence and the split fixed-sequence
rule give nearly the same coverage: the limit is the number of selection images, not the search. The
Bonferroni variant of Learn-then-Test is a little more conservative (about one point lower). The empirical threshold
accepts the most images, but it violates the target in about half of the splits (40 to 50% for the ensemble). The
conformal rule gets more coverage than SGR and violated in only a few splits for the ensemble, but it has no
guarantee and its behavior depends on the score. The softmax cutoff 1 - r* was safe in these runs, but it gives up coverage at larger r* (for example {chow_005} at r* = 0.05
against {sgr_005} for SGR). Only the rules in green carry a guarantee.</p>
"""

SECTIONS["improvement"] = """
<p>Replacing the single model by a deep ensemble of five, and keeping the plain SGR rule with delta = 0.001, gives more
coverage than the original paper at every target in the table, although our bound is stricter than the one behind the
paper's numbers.</p>
{fig}
{table}
<p>On VGG-16 the ensemble accepts {ens_cov_001} of the images at r* = 0.01 (paper: {paper_cov_001}) and the guarantee
is certified on {ens_cert_001} of the splits; the single softmax model manages {soft_cov_001} and certifies only
{soft_cert_001}. The same picture appears for ResNet-18 ({res_ens_001} against {res_soft_001} at r* = 0.01),
so it is not tied to one architecture.</p>
<h3>What does it cost?</h3>
{fig_cost}
<p>At r* = 0.01 an ensemble of two models already reaches {cost_e2} coverage, more than {mc_passes} MC dropout passes
({cost_mc}). Five members reach {cost_e5}. Cost is counted in forward passes per image. The ResNet-18 cost sweep could
not be run because its saved predictions are not available.</p>
"""

SECTIONS["other"] = """
<p>These experiments are secondary; the full tables are in the folded blocks.</p>
<h3>Calibration</h3>
{fig_cal}
<p>Temperature scaling lowers the expected calibration error (for the base model from {ece_raw} to {ece_ts}). Under
corruption the calibration error grows with severity.</p>
{tab_cal}
<h3>Out-of-distribution inputs</h3>
{fig_ood}
<p>As a detector for SVHN images the ensemble reaches an AUROC of {auroc_ens}, the base model {auroc_base}.</p>
{tab_ood}
<h3>Covariate shift</h3>
{fig_shift}
<p>With corrupted test images the risk guarantee is no longer valid, since the threshold was chosen on clean images.
The risk rises with severity for all methods shown.</p>
"""

SECTIONS["next"] = """
<ul>
  <li><strong>Label-budget sweep</strong> (most direct): how much coverage is left if fewer than 5,000 selection
  labels are available? The coverage gap to the paper is a question of n, so this is the natural next measurement.</li>
  <li><strong>Logit-based scores</strong>, which keep information that the softmax throws away.</li>
  <li><strong>Cheap ensembles</strong> (snapshot or shared-trunk ensembles) to keep most of the gain at a fraction of
  the cost.</li>
</ul>
"""
