"""Narrative text of the results report (``build_report.py``), one HTML fragment per section.

The numbers quoted here were read off the result csv files of the full run (5 seeds x 10 splits). The tables in the
report are rendered from the csv files themselves, so they stay correct if the run is repeated; this text does not.
"""

from __future__ import annotations

TITLE = "Selective prediction with risk guarantees on CIFAR-10"
SUBTITLE = (
    "SGR and other accept/abstain rules on top of 12 uncertainty methods: clean data, OOD inputs, corruptions and "
    "calibration. VGG-16, 5 seeds x 10 random 5k/5k splits."
)

SECTIONS: dict[str, str] = {}

SECTIONS["overview"] = """
<p>A selective classifier answers only when its uncertainty score is below a threshold, and abstains otherwise.
The threshold is chosen on a labelled <em>selection</em> half (5k images) so that the <em>selective risk</em> (error
rate on the accepted inputs) stays below a target r*. The other 5k images, the <em>test</em> half, measure what that
threshold actually delivers. SGR (selection with guaranteed risk) adds a high-probability bound: with probability
1 - delta (delta = 0.001) the risk of the chosen threshold is at most r*.</p>

<p><strong>In short</strong></p>
<ul>
  <li><strong>The guarantee holds.</strong> On clean data SGR and both Learn-then-Test variants violate r* in
  0% of the 50 seed x split pairs, for every score and every r* from 0.01 to 0.06. Rules without a guarantee
  (empirical threshold, raw Chow, conformal) violate r* in up to 100% of the splits.</li>
  <li><strong>The price is coverage, and it is steep at r* = 0.01.</strong> At r* = 0.02 nearly every score
  certifies 0.73-0.87 coverage. At r* = 0.01 certification becomes a coin flip for many scores: a split is either
  certified at about 0.7 coverage or certifies nothing. The mean coverage therefore drops from 0.79 (ensemble) to
  0.24 (plain softmax) to 0 (SNGP).</li>
  <li><strong>Deep ensembles win almost everywhere in distribution</strong> (best AURC, best certified coverage,
  best OOD AUROC). MC dropout and the dropout model's plain softmax come next, at one model's cost.</li>
  <li><strong>The guarantee does not survive shift.</strong> It holds for the distribution it was certified on.
  Under corruption the realized risk grows with severity for every method. The corruption type matters far more
  than the method.</li>
  <li><strong>Two expectations do not hold.</strong> Distance-aware density scores (GDA, DDU) do not detect OOD
  better than softmax scores. The epistemic part of the entropy decomposition is a worse selection and OOD score
  than total or aleatoric uncertainty for most methods.</li>
  <li><strong>Confound to fix before the benchmark.</strong> SWA, SWAG, DDU, VBLL and SNGP were fine-tuned for
  only 20 epochs at lr 0.01 from a base model that had annealed to lr of about 1e-4. They end below the base
  model's accuracy (0.916-0.926 vs 0.9375). Their low ranks are at least partly a training-budget effect.</li>
</ul>
"""

SECTIONS["setup"] = """
<h3>Protocol</h3>
<ul>
  <li><strong>Model and data:</strong> CIFAR-10, VGG-16 with conv-block dropout. The base model is trained for 250
  epochs (SGD, lr 0.1 halved every 25 epochs), once per seed 0-4.</li>
  <li><strong>Splits:</strong> the 10k test set is split 10 times at random into a 5k selection half and a 5k test
  half. Every number is a mean (+- std) over the 5 x 10 = 50 seed x split pairs. The deep ensemble uses the five
  base models as members, so there is only one ensemble and its spread comes from the splits alone.</li>
  <li><strong>Violation:</strong> the share of the 50 pairs whose test-half risk exceeds r*. A rule with a delta =
  0.001 guarantee should essentially never violate.</li>
  <li><strong>Shift and OOD:</strong> thresholds fitted on clean selection data are applied unchanged to corrupted
  versions of the test half (contrast, gaussian blur, gaussian noise, pixelate at severities 1, 3, 5). For OOD, the
  same thresholds are applied to SVHN and CIFAR-100 images.</li>
</ul>

<h3>Methods and their scores</h3>
<p>A <em>criterion</em> is a method plus one score derived from it: maxprob (1 - max mean probability), the entropy
decomposition (total = aleatoric + epistemic) or a density. In total there are 30 criteria.</p>
<table class="static">
<thead><tr><th>method</th><th>what it is</th><th>training on top of the base</th><th>forward passes</th>
<th>criteria</th></tr></thead>
<tbody>
<tr><td>sr_base</td><td>plain softmax of the base model</td><td>none</td><td>1</td><td>maxprob</td></tr>
<tr><td>sr_dropout</td><td>softmax of the dropout model (eval mode)</td><td>dropout layers inserted before the two
Linear layers, 50 epochs at lr 0.01</td><td>1</td><td>maxprob</td></tr>
<tr><td>mc</td><td>MC dropout on the dropout model</td><td>same as sr_dropout</td><td>100</td><td>maxprob, total,
aleatoric, epistemic, variance of the predicted class</td></tr>
<tr><td>ens</td><td>deep ensemble of the 5 base models</td><td>5 x base</td><td>5</td><td>maxprob, total,
aleatoric, epistemic</td></tr>
<tr><td>finetune</td><td>base fine-tuned without dropout (budget control for sr_dropout)</td><td>50 epochs at lr
0.01</td><td>1</td><td>maxprob</td></tr>
<tr><td>laplace</td><td>last-layer Laplace approximation (post hoc)</td><td>none</td><td>100 samples of the last
layer</td><td>maxprob, total, aleatoric, epistemic</td></tr>
<tr><td>gda</td><td>Gaussian discriminant on the base features (post hoc)</td><td>none</td><td>1</td>
<td>density</td></tr>
<tr><td>ddu</td><td>DDU-style: spectral normalization + feature density</td><td>20 epochs at lr 0.01</td><td>1</td>
<td>maxprob, density</td></tr>
<tr><td>vbll</td><td>variational Bayesian last layer</td><td>20 epochs at lr 0.01</td><td>samples of the last
layer</td><td>maxprob, total, aleatoric, epistemic</td></tr>
<tr><td>swa</td><td>SWA mean weights</td><td>20 epochs at constant lr 0.01</td><td>1</td><td>maxprob</td></tr>
<tr><td>swag</td><td>SWAG weight posterior</td><td>same run as swa</td><td>one per weight sample</td><td>maxprob,
total, aleatoric, epistemic</td></tr>
<tr><td>sngp</td><td>spectral norm + Gaussian process output layer</td><td>20 epochs at lr 0.01</td><td>1</td>
<td>maxprob, Dempster-Shafer (ds)</td></tr>
</tbody></table>

<h3>Accept/abstain rules (modes)</h3>
<table class="static">
<thead><tr><th>mode</th><th>rule</th><th>guarantee</th></tr></thead>
<tbody>
<tr><td>emp</td><td>largest coverage whose empirical selection risk is at most r*</td><td>none</td></tr>
<tr><td>sgr</td><td>SGR binary search with a Clopper-Pearson bound at level delta / 13</td><td>risk &le; r* with
probability 1 - delta</td></tr>
<tr><td>cov</td><td>label-free coverage target (conformal quantile) set to emp's coverage</td><td>coverage only</td></tr>
<tr><td>LTT Bonf</td><td>Learn-then-Test, 100 candidate thresholds, Bonferroni</td><td>risk &le; r* w.p. 1 -
delta</td></tr>
<tr><td>LTT FS</td><td>Learn-then-Test, fixed-sequence testing from 10 starts</td><td>risk &le; r* w.p. 1 -
delta</td></tr>
<tr><td>Chow raw</td><td>accept if max probability &ge; 1 - r* (no fitting)</td><td>none (assumes calibration)</td></tr>
<tr><td>Chow TS</td><td>the same after temperature scaling</td><td>none</td></tr>
<tr><td>Conf LAC</td><td>accept if the conformal LAC set at level alpha = r* is a singleton</td><td>marginal
coverage of the set only</td></tr>
<tr><td>Conf APS</td><td>the same with APS sets</td><td>marginal coverage of the set only</td></tr>
</tbody></table>

<h3>Metrics</h3>
<ul>
  <li><strong>AURC</strong> (x1000): area under the risk-coverage curve on the full test set, lower is better.
  <strong>E-AURC</strong> subtracts the AURC of a perfect ranking with the same accuracy, so it measures the
  ranking alone.</li>
  <li><strong>coverage / risk:</strong> share of accepted test-half inputs and their error rate.</li>
  <li><strong>OOD AUROC:</strong> how well the score separates clean CIFAR-10 test images from OOD images.
  <strong>accepted share:</strong> the share of OOD images that the clean-fitted threshold accepts, lower is
  better.</li>
  <li><strong>ECE / NLL / Brier:</strong> calibration and proper scoring rules, raw and after temperature scaling
  (T fitted on the clean selection half).</li>
</ul>
"""

SECTIONS["reproduction"] = """
<p>First check: does the pipeline reproduce the CIFAR-10 numbers of the original SGR paper (Geifman and El-Yaniv,
2017, Table 1)? The paper reports one test risk and coverage per target. Our risk-coverage curve, read at the
paper's risk, gives slightly <em>higher</em> coverage at every point (for example 0.808 vs 0.786 at r* = 0.01 for
the base softmax). The reproduction holds, with a slightly stronger base model.</p>
<p>With SGR (probly's <code>SGRSelector</code>) the guarantee holds at every r* (0% violations), and from r* = 0.02
on the coverage is within a few points of the paper. At r* = 0.01 the plain softmax (sr_base) certifies on only
34% of the splits (see the next sections). The dropout model and MC dropout certify on all splits there, at about
0.73 coverage.</p>
<p>The label-free coverage target (probly's <code>CoverageSelector</code>) hits the requested coverage to within
+-0.006 and gives risks at or below the paper's.</p>
"""

SECTIONS["clean"] = """
<p>The ranking quality on clean data decides everything downstream. A better ranking gives more coverage at the
same risk under every threshold rule.</p>
<ul>
  <li><strong>The ensemble leads clearly:</strong> AURC 5.4-6.2 for all four of its scores, accuracy 0.947 vs
  about 0.937 for a single model. Its epistemic score (mutual information) is its best. The ensemble is the only
  method where that holds.</li>
  <li><strong>Dropout helps, but partly through training.</strong> The dropout model's plain softmax (6.85) is
  nearly as good as 100-sample MC dropout (6.70). The no-dropout fine-tune control already improves the base from
  8.95 to 7.60, so about half of the gain is the extra 50 epochs of training.</li>
  <li><strong>Post-hoc last-layer methods add nothing to the ranking.</strong> Laplace (9.08) and the DDU/VBLL
  maxprob scores (9.2, 10.3) do not beat the plain softmax of the base they start from (8.95). GDA density (8.46)
  is slightly better.</li>
  <li><strong>The bottom group is confounded.</strong> SWA (11.2), SWAG (11.9), SNGP (12.3) and VBLL lose 1-2
  points of accuracy relative to the base during their short 20-epoch restart at lr 0.01. Their E-AURC is also worse, so
  the ranking itself suffers, not just accuracy, but this has to be rerun with a fair budget before it counts
  against the methods.</li>
  <li><strong>Maxprob of the mean prediction is the best score for most methods.</strong> The decomposition
  scores are close behind (within 0.2-0.3), except epistemic, which is worst for SWAG (15.0) and Laplace. The
  density scores rank badly in distribution (ddu_density 17.7, the worst criterion).</li>
</ul>
<p><strong>Main score per method</strong> (used in the summary figures): the criterion with the lowest clean AURC
in that method. That is maxprob for 9 of 12 methods, ens_epistemic for the ensemble, gda_density for GDA, and the
single score for sr_base and sr_dropout.</p>
"""

SECTIONS["rules"] = """
<p>Given a score, which threshold rule should be used? The heatmap shows the test coverage of every rule per
criterion. Red outlines mark rules that violate r* in more than 1% of the splits.</p>
<ul>
  <li><strong>SGR, LTT Bonf, LTT FS:</strong> 0% violations everywhere on clean data. The only exception is
  swag_aleatoric/epistemic, at 2% (1 of 50 splits) at r* = 0.01, which is consistent with the delta-level
  guarantee. LTT fixed-sequence matches SGR's coverage. LTT Bonferroni costs about 1 point of coverage. LTT does
  not rescue r* = 0.01: for sr_base it reaches 0.32 vs SGR's 0.24.</li>
  <li><strong>emp and cov:</strong> the best coverage, and violations in 20-60% of the splits, as expected for a
  threshold fitted to hit r* exactly.</li>
  <li><strong>Chow raw</strong> (accept if max probability &ge; 1 - r*) trusts the raw softmax and violates in
  96-100% of the splits at r* = 0.01 for the overconfident single models (sr_base, sr_dropout, finetune,
  laplace). It is fine at r* = 0.03. <strong>Chow TS</strong> after temperature scaling swings to the other side
  and is far too conservative (sr_base coverage 0.0004 at r* = 0.01).</li>
  <li><strong>Conf LAC</strong> (singleton conformal sets) gives the highest coverage at r* = 0.01 (sr_base 0.78).
  Its guarantee is only marginal, though: it bounds the risk by alpha / coverage, not by alpha. It violates in
  2-10% of the splits (38% for SNGP). <strong>Conf APS</strong> singletons are useless (coverage below 0.3).</li>
</ul>
<p><strong>Recommendation:</strong> keep SGR and LTT FS as the guaranteed rules, emp as the unguaranteed
reference, and LAC as the conformal comparison. Drop APS and Chow TS. Keep Chow raw only as a cautionary
example.</p>
"""

SECTIONS["cliff"] = """
<p>The most important caveat of the guarantee: at r* = 0.01 the certified coverage is <em>bimodal</em>. On each
split, SGR either certifies a threshold near the empirical one (about 0.7 coverage) or certifies nothing and
accepts 0%. The mean coverage in the tables is therefore mostly a <em>certification probability</em> times about
0.7. The "uncertified" table shows that share directly:</p>
<ul>
  <li>never uncertified at r* = 0.01: ensemble, MC dropout, sr_dropout (and ddu_maxprob at 2%);</li>
  <li>sometimes: finetune 10%, swa/swag about 20%, gda 44%, laplace 50-80%, vbll 56-58%, sr_base 66%;</li>
  <li>always: sngp_maxprob, sngp_ds and ddu_density (100%). ddu_density is still uncertified on 40% of the splits
  at r* = 0.02 and 10% at r* = 0.03.</li>
</ul>
<p><strong>Why.</strong> The bound has to be computed at level delta / 13 = 7.7e-5, because the binary search tests
up to 13 thresholds, and with about 3500 accepted points it has a slack of roughly 0.005. So a threshold is
certifiable at r* = 0.01 only if the true risk among the accepted points is about 0.005 or less. Whether a split
gets below that depends on how many errors happen to sit among its most confident points. Scores that put a few
errors among their most confident points fail on most splits. This is a sample-size limit of any finite-sample
guarantee, not a search problem: LTT, which tests many thresholds with a different correction, fails on the same
splits. A larger selection set (ImageNet) or r* &ge; 0.02 avoids it.</p>
<p><strong>Confirmed by the diagnostic</strong> (<code>scripts/check_sgr_path.py</code>). For each seed it prints
the smallest bound over <em>all</em> acceptable prefixes of the selection half, averaged over the 10 splits. If that
value is above r*, no threshold of any search can be certified.</p>
<table class="static">
<thead><tr><th>criterion</th><th>min bound per seed 0-4</th><th>uncertified splits per seed</th></tr></thead>
<tbody>
<tr><td>sngp_maxprob</td><td>0.0148, 0.0130, 0.0129, 0.0115, 0.0121</td><td>10, 10, 10, 10, 10</td></tr>
<tr><td>sngp_ds</td><td>0.0146, 0.0131, 0.0142, 0.0120, 0.0121</td><td>10, 10, 10, 10, 10</td></tr>
<tr><td>ddu_density</td><td>0.0192, 0.0170, 0.0136, 0.0150, 0.0284</td><td>10, 10, 10, 10, 10</td></tr>
<tr><td>sr_base</td><td>0.0107, 0.0102, 0.0086, 0.0073, 0.0111</td><td>9, 9, 5, 0, 10</td></tr>
<tr><td>ddu_maxprob</td><td>0.0079, 0.0072, 0.0062, 0.0055, 0.0055</td><td>0, 1, 0, 0, 0</td></tr>
</tbody></table>
<ul>
  <li><strong>SNGP and ddu_density:</strong> the min bound is above 0.01 for every seed. Their most confident
  predictions carry a risk of about 0.004-0.007 (SNGP) and 0.005-0.02 (DDU density), which is too high at n = 5k.
  Their coverage of 0 at r* = 0.01 is genuine for this training budget and not a selector bug. It fits the
  training-budget confound: SNGP ends at 0.924 accuracy after 20 epochs.</li>
  <li><strong>sr_base is bimodal across seeds, not across splits.</strong> Seed 3 (min bound 0.0073) certifies on
  every split. Seeds 0, 1 and 4 (min bound 0.010-0.011, just above r*) almost never do, and seed 2 sits at the edge
  (5 of 10). So the 0.24 +- 0.33 coverage is a property of which base model was trained, and r* = 0.01 lies exactly
  on the edge for a single softmax network.</li>
  <li><strong>ddu_maxprob</strong> has the lowest min bounds of all (0.0055-0.0079), because its most confident
  points are almost error-free (0-2 errors among the top 1000). Its low certified coverage is set by the bound's
  slack, not by errors.</li>
</ul>
"""

SECTIONS["ood"] = """
<p>How well does each score separate CIFAR-10 test images from OOD images (SVHN: far OOD, CIFAR-100: near OOD), and
how much OOD data would a clean-fitted threshold let through?</p>
<ul>
  <li><strong>The ensemble is best on both.</strong> SVHN AUROC 0.915-0.926 with a tiny std (one ensemble, only the
  splits vary). On CIFAR-100, ens_epistemic is best at 0.895. swag_aleatoric also reaches 0.925 on SVHN, but with
  a large spread (+-0.020).</li>
  <li><strong>Density scores do not win.</strong> GDA density (SVHN 0.895, CIFAR-100 0.874) is middle of the pack.
  DDU density (0.852, 0.834) is among the worst. With a VGG without residual connections and only 20 epochs of
  spectral-norm fine-tuning, the feature space is not distance-aware enough. This is the main expectation that
  breaks.</li>
  <li><strong>Epistemic is not the OOD score.</strong> For MC dropout the epistemic part (SVHN 0.868) and the
  predicted-class variance (0.859) are worse than total (0.899) and aleatoric (0.916). The same holds for SWAG
  (0.802 vs 0.925) and Laplace. Epistemic only helps for the ensemble on near OOD.</li>
  <li><strong>sr_base and laplace vary a lot</strong> across seeds on SVHN (+-0.05-0.07). Some base seeds are much
  more overconfident on SVHN than others.</li>
  <li><strong>Accepted OOD share at SGR r* = 0.03:</strong> 0.1-0.46 of SVHN and 0.16-0.42 of CIFAR-100 images are
  accepted. The lowest shares (ddu_density 0.10, swag_aleatoric 0.05) come with much lower in-distribution
  coverage, so read them next to the coverage, not alone. A risk guarantee on clean data does not stop OOD inputs.
  With CIFAR-100 mixed in, the accepted risk rises far above r* (see the mixed tables).</li>
</ul>
"""

SECTIONS["shift"] = """
<p>The thresholds are fitted on clean selection data and applied to corrupted test images. At r* = 0.03 the clean
SGR risk is about 0.02.</p>
<ul>
  <li><strong>The corruption type dominates.</strong>
    <ul>
      <li>Contrast and blur at severity 1 keep the risk at about 0.02: the guarantee still holds.</li>
      <li>Pixelate at severity 1 is at about 0.027-0.031, borderline.</li>
      <li>Gaussian noise breaks everything at once: accuracy falls from 0.94 to about 0.71 at severity 1, and the
      SGR risk is 0.08-0.18.</li>
    </ul></li>
  <li><strong>By severity 3</strong> every method and corruption is above r* (contrast and blur 0.03-0.05,
  pixelate 0.05-0.07, noise 0.39-0.59). <strong>At severity 5</strong> contrast drops accuracy to about 0.2 and
  the risk to 0.28-0.67. No method holds anything there.</li>
  <li><strong>Scores that degrade least</strong> do so mainly by accepting less:
    <ul>
      <li>ddu_density: noise severity 1 at 0.082 vs 0.15 for maxprob, at coverage 0.39 vs 0.67;</li>
      <li>gda_density: 0.114;</li>
      <li>mc_aleatoric and swag_aleatoric: contrast severity 3 at 0.036-0.037 vs 0.048, and the lowest severity 5
      risks.</li>
    </ul>
  A good shift score lowers its coverage when the inputs change; maxprob scores keep accepting.</li>
  <li><strong>The ensemble is not robust here.</strong> It keeps the highest coverage under shift, and so has some
  of the highest severity-5 risks (contrast 0.66-0.67).</li>
</ul>
<p><strong>Takeaway for the benchmark:</strong> a clean-data guarantee says nothing under shift. The useful shift
metric is the <em>risk inflation</em> (realized risk / r*) together with the coverage drop, per corruption type.
Mild contrast/blur shifts are a sensible "holds" case, and gaussian noise is a "breaks" case.</p>
"""

SECTIONS["calibration"] = """
<p>Calibration does not decide selection quality (only the ranking does), but it matters for rules that read the
probabilities directly (Chow, conformal) and for reporting.</p>
<ul>
  <li><strong>Temperature scaling fixes the single models:</strong> ECE sr_base 0.041 -> 0.019 (T = 1.47),
  finetune 0.038 -> 0.018, laplace 0.035 -> 0.016, NLL down by about 0.04. All of them are overconfident (T &gt;
  1).</li>
  <li><strong>MC dropout and SWA are calibrated out of the box</strong> (ECE 0.007 and 0.012, T = 1.00). The
  ensemble is slightly underconfident-looking at 0.014 and goes to 0.007 with T = 1.23. SWAG is
  <em>underconfident</em> (T = 0.78) and goes to 0.009. SNGP stays the worst after TS (0.024).</li>
  <li><strong>TS does not improve the ranking.</strong> A single temperature cannot reorder points by maxprob, but
  it can by entropy. msr AURC barely moves or gets worse (sr_base 8.95 -> 9.67), so TS is a calibration fix, not a
  selection fix.</li>
  <li><strong>Under shift</strong>, with T fitted on clean data: TS still helps the overconfident sr and ens, and
  hurts SWAG and SNGP (T &lt; 1 makes them even less confident on data where they should already abstain).</li>
</ul>
<p><em>Not done yet:</em> vector or Dirichlet scaling, and AU/EU AUROC for OOD.</p>
"""

SECTIONS["verdict"] = """
<table class="static">
<thead><tr><th>question</th><th>winner</th><th>runner-up / note</th></tr></thead>
<tbody>
<tr><td>clean ranking (AURC)</td><td>deep ensemble (5.4)</td><td>MC dropout 6.7, dropout softmax 6.85</td></tr>
<tr><td>certified coverage, r* = 0.01</td><td>deep ensemble (0.79)</td><td>MC dropout, dropout softmax 0.73; many
others certify only on some splits</td></tr>
<tr><td>certified coverage, r* &ge; 0.02</td><td>deep ensemble</td><td>most single models within 1-5 points</td></tr>
<tr><td>best threshold rule with a guarantee</td><td>SGR = LTT FS</td><td>LTT Bonf about 1 point less coverage</td></tr>
<tr><td>best coverage without a guarantee</td><td>Conf LAC</td><td>2-10% violations, marginal guarantee
only</td></tr>
<tr><td>far OOD (SVHN AUROC)</td><td>ensemble aleatoric/total (0.926/0.923)</td><td>swag_aleatoric 0.925 with high
variance</td></tr>
<tr><td>near OOD (CIFAR-100 AUROC)</td><td>ensemble epistemic (0.895)</td><td>mc_aleatoric 0.893</td></tr>
<tr><td>mild shift (risk kept lowest)</td><td>density and aleatoric scores</td><td>by accepting less, not by ranking
better</td></tr>
<tr><td>calibration</td><td>MC dropout, SWA (no TS needed)</td><td>TS fixes all single models</td></tr>
<tr><td>best value for one model</td><td>dropout model, softmax or MC</td><td>1 forward pass gets almost all of MC's
benefit</td></tr>
</tbody></table>
"""

SECTIONS["next"] = """
<p>Decisions to make before this moves into the benchmark package:</p>
<ol>
  <li><strong>Fair training budget.</strong> Rerun SWA/SWAG, DDU, VBLL and SNGP with the same 50-epoch decayed
  fine-tune as finetune/dropout (or from a common checkpoint before the base anneals), so that no method ends
  below the base's accuracy. Until then, the bottom of the clean ranking and the SNGP risk floor are provisional:
  they hold for this budget, not for the methods in general.</li>
  <li><strong>Report the certification probability.</strong> At small r* a mean coverage hides a 0-or-0.7 mixture.
  The benchmark should report the "certified share" next to the coverage, and treat r* = 0.01 at n = 5k as
  outside the regime where guarantees are cheap.</li>
  <li><strong>Main score per method</strong> = lowest clean AURC (maxprob for most). Alternative: a fixed score per
  family (always maxprob, always total entropy) for a cleaner comparison.</li>
  <li><strong>Threshold rules to keep:</strong> SGR, LTT FS, emp, Conf LAC. Drop APS and Chow TS.</li>
  <li><strong>Shift reporting:</strong> risk inflation and coverage drop per corruption type. Keep contrast/blur
  (holds) and gaussian noise (breaks) as the representative cases.</li>
  <li><strong>OOD reporting:</strong> AUROC plus the accepted OOD share at a fixed r*, always shown next to the
  in-distribution coverage.</li>
  <li><strong>Calibration extras</strong> still open: vector/Dirichlet scaling, AU/EU AUROC.</li>
  <li><strong>Figures:</strong> the older figures under <code>results/</code>, <code>results/shift/</code> and
  <code>results/calibration/</code> were drawn with a font bug that made their tick labels hairline-thin. They
  are fixed on the next evaluation run.</li>
</ol>
"""
