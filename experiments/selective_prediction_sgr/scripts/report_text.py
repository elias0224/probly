"""Narrative text of the results report (``build_report.py``), one HTML template per section.

Every template is filled with ``str.format`` from a dictionary of numbers that ``build_report.py`` reads from the result
csv files, so the text stays in sync with the data. Display names only; raw ids belong to the appendix.

References: Geifman and El-Yaniv 2017 (SGR, arXiv 1705.08500), Traub et al. 2024 (AUGRC), Angelopoulos et al.
(Learn-then-Test).
"""

from __future__ import annotations

TITLE = "Replicating SGR with probly: CIFAR-10 and ImageNet"
SUBTITLE = (
    "Status of the replication of Geifman and El-Yaniv (2017), Sec. 5: CIFAR-10 (Table 1) and ImageNet (Tables 3-6, "
    "Fig. 2c), implementation differences, and further comparisons on CIFAR-10."
)

SECTIONS: dict[str, str] = {}

_GLANCE_CIFAR = """
  <li><strong>CIFAR-10 (Table 1).</strong> The coverage ranking of the paper holds. Cutting the threshold at the paper's
  test risk gives {repro_gain_lo} to {repro_gain_hi} points more coverage than the paper; SGR at delta = 0.001 gives
  {sgr_gap_lo} to {sgr_gap_hi} points less for r* of 0.02 and above. At r* = 0.01 SGR certifies {cert_001} of the
  splits.</li>
  <li><strong>Printed bounds.</strong> Recomputed from the paper's own train numbers, the bound B* of Lemma 3.1 does
  not match the printed one in any CIFAR-10 row (implied delta {cif_d_lo} to {cif_d_hi}).</li>
  <li><strong>Guarantee.</strong> SGR at delta = 0.001 exceeded the target in {sgr_viol_max} of the {n_pairs}
  seed x split pairs for the methods in this report. The empirical threshold exceeds it in about half of the
  splits.</li>
  <li><strong>Deep ensemble.</strong> Five members with the same SGR rule reach {ens_cov_001} coverage at r* = 0.01
  (paper: {paper_cov_001}; softmax: {soft_cov_001}).</li>
"""

SECTIONS["glance"] = (
    """
<ul>"""
    + _GLANCE_CIFAR
    + """
  <li><strong>ImageNet (Tables 3-6).</strong> Accuracies match torchvision's published ones (VGG-16 {acc_vgg1} top-1, ResNet-50
  {acc_res1}). The VGG-16 risk-coverage curve lies close to the paper's test points (mean |risk difference| {lu_vgg1}
  top-1, {lu_vgg5} top-5). Our ResNet-50 curve gives {res_cd_lo} to {res_cd_hi} points more coverage at the paper's risk,
  so it is likely not the paper's model and its tables are not a like-for-like comparison.</li>
  <li><strong>ImageNet, bounds.</strong> The printed bounds do not follow from Lemma 3.1 either (implied delta
  {im_d_lo} to {im_d_hi}), and the paper's train risks lie at or above our delta = 0.1 values in {tr_ge} of {tr_n}
  rows. SGR at delta = 0.001 accepts more than the paper in {im_more} of {im_rows} rows and exceeded the target on the
  test half in at most {im_viol_sgr} of the splits (test-set noise, see section 5), the empirical threshold in
  {im_viol_emp_lo} to {im_viol_emp_hi} at the worst r* per table.</li>
  <li><strong>Fig. 2c reproduces.</strong> Risk at 60% coverage (VGG-16 top-1): SR {r60_sr1}, MC variance {r60_mcv1}
  (paper: about 0.10 and above 0.20); MC max-prob, not in the paper, {r60_mcm1}.</li>
  <li><strong>Not done.</strong> CIFAR-100 (Table 2), Keras weights and Keras preprocessing.</li>
</ul>
"""
)
SECTIONS["glance_cifar"] = "<ul>" + _GLANCE_CIFAR + "  <li>ImageNet results are not available in this build.</li>\n</ul>"

SECTIONS["setup"] = """
<p>CIFAR-10: the 10,000 test images are split 10 times at random into 5,000 selection and 5,000 test images; every
number is a mean over 5 training seeds x 10 splits = {n_pairs} pairs. ImageNet: 100 random halvings of the 50,000
validation images (25,000 / 25,000), pretrained torchvision models. The paper used a single split. delta = 0.001
throughout, as in the paper. A split is <em>certified</em> if the rule found a threshold with bound at most r*;
otherwise nothing is accepted. <em>Violation</em> is the share of certified splits whose test risk exceeds r*. It is
measured on the finite test half, while the guarantee concerns the true risk, so test-set noise alone can produce a
few violations.</p>
<p>probly components: <code>SGRSelector</code>, <code>CoverageSelector</code> and <code>ThresholdSelector</code> for
the thresholds; <code>risk_coverage_curve</code>, <code>aurc</code> and <code>coverage_at_risk</code> from
<code>probly.metrics.selective_prediction</code> (CIFAR-10 additionally AUGRC).</p>
<details><summary>Glossary</summary><dl class="gloss">
<dt>SR</dt><dd>Softmax response: the highest softmax probability, the paper's score. Called softmax in the CIFAR-10
sections.</dd>
<dt>MC variance</dt><dd>Variance of the predicted class's probability over the MC-dropout passes (the paper's
MC-dropout score).</dd>
<dt>MC max-prob</dt><dd>Highest probability of the mean MC-dropout prediction.</dd>
<dt>1 - top-5 mass</dt><dd>One minus the summed probability of the five most likely classes (score for the top-5
loss).</dd>
<dt>r*, delta</dt><dd>Target risk, and the allowed probability that the true risk of the returned rule exceeds it.</dd>
<dt>B*</dt><dd>The SGR bound of Lemma 3.1 at the returned threshold.</dd>
<dt>Implied delta</dt><dd>The delta at which B*, recomputed from the paper's printed train numbers, equals the
printed bound.</dd>
<dt>Coverage at the paper's risk</dt><dd>Coverage of our risk-coverage curve at the paper's printed test risk,
independent of any threshold rule.</dd>
<dt>Empirical threshold</dt><dd>Largest coverage whose selection-set risk is below r*, without a bound.</dd>
<dt>AURC, AUGRC</dt><dd>Area under the risk-coverage curve and under the generalized risk-coverage curve (Traub et
al., 2024); lower is better.</dd>
</dl></details>
<h3>Methods compared (CIFAR-10)</h3>
{methods_table}
<h3>Threshold rules compared (CIFAR-10)</h3>
{rules_table}
"""

SECTIONS["diff"] = """
<p>Differences that can change a number in the comparison with the paper:</p>
{table}
"""

SECTIONS["repro"] = """
<p>Table 1 of the paper gives, per target risk r*, the test coverage of SGR with a VGG-16. The comparison uses the
paper model (VGG-16 with dropout, softmax score).</p>
{fig}
{table}
<p>Cut at the paper's test risk, our curve accepts {repro_gain_lo} to {repro_gain_hi} points more than the paper. SGR at
delta = 0.001 ends {sgr_gap_lo} to {sgr_gap_hi} points below it for r* of 0.02 and above. SGR at delta = 0.1 gives
{d01_002} at r* = 0.02, close to the paper (see section 6). At r* = 0.01 the bound is met on {cert_001} of the splits,
so the mean coverage ({sgr_cov_001}) has a large spread.</p>
<p class="d">Returning the last tested iterate as in Algorithm 1 (<code>sgr_last_tested</code>) instead of the last
certified one returns a threshold on every split, but on the splits where <code>SGRSelector</code> abstains that
threshold accepts almost nothing, so the mean coverage at r* = 0.01 is {c01_cmp}.</p>
"""

SECTIONS["im_repro"] = """
<p>Protocol: {n_pairs_im} halvings of the validation set; the first half selects the threshold at each r* of the
paper's table, the second half measures test risk and coverage. Losses are top-1 and top-5 error. Both models are
torchvision weights (VGG-16, ResNet-50). The score is SR (softmax response) unless stated otherwise.</p>
{acc_table}
<p>The line-up compares our risk-coverage curve with the paper's test points independently of SGR: risk at the
paper's coverage, and coverage at the paper's risk. VGG-16 is the relevant comparison. ResNet-50 (V1 weights) gives
{res_cd_lo} to {res_cd_hi} points more coverage than the paper at the same risk in every row, so it is likely not the
paper's model. Its accuracy ({acc_res1}) does not settle this: Table 5 only bounds the paper's top-1 accuracy to
{pacc_lo} to {pacc_hi}.</p>
{fig_curves}
<h3>SGR at delta = 0.001 against the paper</h3>
{sgr_table}
{fig_cov}
<p>SGR at delta = 0.001 accepts more than the paper in {im_more} of {im_rows} rows, in line with our curves lying at or
above the paper's points. {im_less_text} At most {im_viol_sgr} of the splits exceed r* on the test half. This does not
break the guarantee, which concerns the true risk: the test risk of a split varies with a standard deviation of
{im_sd_lo} to {im_sd_hi} over splits, so a threshold close to r* can exceed it on the test half. The empirical threshold
accepts more still and exceeds r* in {im_viol_emp_lo} to {im_viol_emp_hi} of the splits at the worst r* per table.</p>
<h3>Fig. 2c: SR against MC dropout</h3>
{fig2c_table}
<p>The paper reports for VGG-16 top-1 at 60% coverage about 10% error with SR and more than 20% with MC dropout, and
calls SR significantly better on both losses. This reproduces: with the variance of the MC samples as score (the
paper's MC-dropout criterion), risk at 60% coverage is {r60_mcv1} against {r60_sr1} for top-1 and {r60_mcv5} against
{r60_sr5} for top-5 (VGG-16). The max probability of the mean MC prediction, which the paper does not use, is about as
good as SR ({r60_mcm1}).</p>
<h3>Score for the top-5 loss</h3>
<p>The paper uses SR for both losses. The score 1 - top-5 mass matches the top-5 loss and is better on the whole
curve; SGR coverage at r* = 0.01 is {t5m_vgg01} against {t5sr_vgg01} (VGG-16) and {t5m_res01} against {t5sr_res01}
(ResNet-50); at r* = 0.02 {t5m_vgg02} against {t5sr_vgg02} and {t5m_res02} against {t5sr_res02}.</p>
{top5_table}
"""

SECTIONS["bounds"] = """
<p>The table recomputes B* (Lemma 3.1) from the paper's printed train risk and train coverage (m = 5,000 for CIFAR-10,
about 25,000 for ImageNet) at delta = 0.001 with and without the division by ceil(log2 m), and gives the range of
B* minus the printed bound per paper table. The last column is the delta at which B* equals the printed bound. In no
row is the printed bound reached at delta = 0.001. The implied delta is {cif_d_lo} to {cif_d_hi} for CIFAR-10 and
{im_d_lo} to {im_d_hi} for ImageNet.</p>
{table}
<p>The same is visible in the train risk where the rule stops: the paper's value lies at or above ours for delta = 0.1
in {tr_ge} of {tr_n} ImageNet rows. Further, the train risks of the VGG-16 and ResNet-50 tables are identical to
within 0.0001 in {tr_same} of {tr_total} rows, although the models differ.</p>
{train_table}
"""

SECTIONS["methods"] = """
<p>Ranking quality by AURC and AUGRC (left) and what the methods deliver under SGR (table).</p>
{fig}
{table}
<p>The deep ensemble has the lowest AUGRC ({a_ens}, softmax {a_soft}) and the highest guaranteed coverage. MC dropout
({a_mc}, {mc_passes} passes) samples from the base model fine-tuned with dropout layers; a single deterministic pass of
that model, not in the table, gives {a_drop}, so the gain over the softmax comes from the fine-tuning, not from the
sampling. The paper model (trained from scratch with dropout) is a different model. The ensemble's
epistemic score has the lowest AURC, while AUGRC and SGR coverage favor its max-probability score. SWA, SWAG, DDU,
VBLL, SNGP and Laplace (appendix) are not better than the softmax; these were fine-tuned for 20 epochs at learning rate
0.01 from the base model and end below its accuracy ({acc_ft} against {acc_base}), so the comparison also reflects
training budget.</p>
"""

SECTIONS["rules"] = """
<p>Coverage per threshold rule for the deep ensemble; the share of violating splits is written next to rules without a
guarantee.</p>
{fig}
{table}
<p>SGR, Learn-then-Test with a fixed sequence and the split fixed-sequence rule give nearly the same coverage; the
Bonferroni variant is {bonf_gap_lo} to {bonf_gap_hi} points below SGR. The empirical threshold accepts the most and
violates the target in {emp_viol_lo} to {emp_viol_hi} of the splits. The conformal rule accepts more than SGR and
violated in at most {conf_viol_hi} of the splits for the ensemble, without a guarantee. The softmax cutoff 1 - r* was safe in these runs but accepts less at larger r* ({chow_005} at r* = 0.05
against {sgr_005} for SGR).</p>
"""

SECTIONS["improvement"] = """
<p>Deep ensemble of five with the plain SGR rule at delta = 0.001, against the paper's reported coverage (which rests
on a weaker bound, section 6).</p>
{fig}
{table}
<p>VGG-16, r* = 0.01: ensemble {ens_cov_001} coverage, certified on {ens_cert_001} of the splits (paper:
{paper_cov_001}; softmax {soft_cov_001}, certified on {soft_cert_001}). ResNet-18: {res_ens_001} against
{res_soft_001}.</p>
<h3>Cost</h3>
{fig_cost}
<p>At r* = 0.01 two members reach {cost_e2} coverage, {mc_passes} MC dropout passes {cost_mc}, five members
{cost_e5}. The ResNet-18 cost sweep was not run (its saved predictions are not available).</p>
"""

SECTIONS["other"] = """
<p>Secondary experiments; full tables are folded.</p>
<h3>Calibration</h3>
{fig_cal}
<p>Temperature scaling lowers the expected calibration error of the base model from {ece_raw} to {ece_ts}. Under
corruption the calibration error grows with severity.</p>
{tab_cal}
<h3>Out-of-distribution inputs</h3>
{fig_ood}
<p>AUROC for SVHN: ensemble {auroc_ens}, base model {auroc_base}.</p>
{tab_ood}
<h3>Covariate shift</h3>
{fig_shift}
<p>With corrupted test images the guarantee does not hold, since the threshold was chosen on clean images. The risk
rises with severity for all methods shown.</p>
"""
