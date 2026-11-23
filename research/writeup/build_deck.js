// Results deck: gaitpdb -> WearGait-PD external validation. Style matches MSCAF-Gait-Major-Project-Review.pptx.
// Build: npm install pptxgenjs@3.12.0 (outside the repo), then
//   DECK_AUTHORS="names · guide · institute" NODE_PATH=<that>/node_modules node research/writeup/build_deck.js out.pptx
// Names are passed at build time so they are not committed to the public repo. The built .pptx is git-ignored.
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/Users/gurkaranbirsingh/.claude/skills/synced/36fc8711-05a0-4cd8-9bda-c545137cf2e4_d484b6be-6c9d-4837-86b4-cf35ad00881a/pptx/scripts/apply_theme.js");

const FIG = "/Users/gurkaranbirsingh/Foot-Pressure-Based-Abnormal-Gait-Recognition/research/writeup/figures/";
const OUT = process.argv[2] || "deck.pptx";

const THEME = {
  name: "Gait Transport", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "1A1228", lt1: "FFFFFF", dk2: "241640", lt2: "F3F1F8",
            accent1: "1F8E7D", accent2: "453781", accent3: "B5DE2B", accent4: "9FE0D4",
            accent5: "C9C2DE", accent6: "D9534F", hlink: "1F8E7D", folHlink: "453781" },
};
const HEX = { ink: "1A1228", dark: "241640", muted: "6E6885", teal: "1F8E7D", purple: "453781",
              tint: "F3F1F8", lime: "B5DE2B", mint: "9FE0D4", lav: "C9C2DE", red: "D9534F", white: "FFFFFF" };

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";            // 10 x 5.625 in, as the existing deck
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Does a Parkinson's gait model travel?";
const C = pres.SchemeColor;

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "DARK", background: { color: HEX.dark },
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: 0.7, y: 0.95, w: 8.6, h: 0.3, fontSize: 11.5,
        bold: true, color: C.accent4, fontFace: "Calibri", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.35, w: 8.6, h: 1.1, fontSize: 36,
        bold: true, color: C.background1, fontFace: "Cambria", valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: HEX.white },
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: 0.55, y: 0.32, w: 9.0, h: 0.24, fontSize: 10.5,
        bold: true, color: C.accent1, fontFace: "Calibri", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.58, w: 9.0, h: 0.62, fontSize: 24,
        bold: true, color: C.text1, fontFace: "Cambria", valign: "top", align: "left", margin: [0, 0, 0, 4] }, text: "" } },
  ],
  slideNumber: { x: 9.25, y: 5.16, w: 0.5, h: 0.28, fontSize: 10, color: HEX.muted, fontFace: "Calibri" },
});

let sec = null;
function slide(master, kicker, title, notes) {
  const s = pres.addSlide({ masterName: master, sectionTitle: sec });
  s.addText(kicker, { placeholder: "kicker" });
  s.addText(title, { placeholder: "title" });
  s.addNotes(notes);
  return s;
}
function section(t) { sec = t; pres.addSection({ title: t }); }
const T = (text, o) => Object.assign({ isTextBox: true, fontFace: "Calibri", fontSize: 13, color: C.text1,
                                         margin: 0, valign: "top" }, o, { text });
function text(s, t, o) { const x = T(t, o); const txt = x.text; delete x.text; s.addText(txt, x); }
function bullets(s, items, o) {
  s.addText(items.map((it, i) => ({ text: it, options: { bullet: { indent: 14 }, breakLine: i < items.length - 1,
    paraSpaceAfter: 6 } })), Object.assign({ isTextBox: true, fontFace: "Calibri", fontSize: 13, color: C.text1,
    margin: 0, valign: "top" }, o));
}
function card(s, x, y, w, h, fill, name) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill },
    line: { color: fill }, objectName: name });
}
function stat(s, x, y, w, value, label, color, name) {
  card(s, x, y, w, 1.55, HEX.tint, name + " card");
  text(s, value, { x: x + 0.2, y: y + 0.15, w: w - 0.4, h: 0.75, fontSize: 40, bold: true, fontFace: "Cambria",
                   color, objectName: name + " value" });
  text(s, label, { x: x + 0.2, y: y + 0.92, w: w - 0.4, h: 0.55, fontSize: 11.5, color: C.text2,
                   objectName: name + " label" });
}
function caption(s, t, x, y, w) { text(s, t, { x, y, w, h: 0.3, fontSize: 10, italic: true, color: HEX.muted }); }

// ---------- 1. title ----------
section("Opening");
{
  const s = slide("DARK", "MAJOR PROJECT  ·  RESULTS", "Does a Parkinson's gait model travel?",
    "[0:00-0:40] Good morning. Last time we reviewed MSCAF-Gait and showed how much of the published accuracy on gaitpdb comes from leakage. This time we asked the next question: if we train an honest model on gaitpdb, does it still work on a completely independent dataset? We ran that test as a pre-registered, frozen external validation on WearGait-PD.");
  text(s, "Frozen, pre-registered external validation of gaitpdb-trained plantar-pressure models on WearGait-PD",
       { x: 0.7, y: 2.6, w: 8.6, h: 0.65, fontSize: 16, color: C.accent5 });
  text(s, process.env.DECK_AUTHORS || "Team names · Guide · Department · Institute",
       { x: 0.7, y: 4.0, w: 8.6, h: 0.95, fontSize: 12, color: C.background1 });
}

// ---------- 2. answer up front ----------
{
  const s = slide("CONTENT", "THE ANSWER IN ONE SLIDE", "It works inside gaitpdb and fails on WearGait-PD",
    "[0:40-1:40] The answer first. Inside gaitpdb our primary model reaches an AUC of 0.67, and still 0.67 when we cut the walks down to WearGait-PD's short format. On WearGait-PD, frozen before we saw any diagnosis, it reaches 0.47, which is chance. The gap of 0.20 is statistically significant after Holm correction. The rest of the talk explains how we got there and why we trust it.");
  stat(s, 0.55, 1.45, 2.85, "0.67", "AUC inside gaitpdb\n(subject-level CV)", HEX.teal, "internal");
  stat(s, 3.575, 1.45, 2.85, "0.47", "AUC on WearGait-PD\n(frozen model, n = 113)", HEX.red, "external");
  stat(s, 6.6, 1.45, 2.85, "0.20", "residual gap after matching\nwalk length (p = 0.007)", HEX.purple, "gap");
  bullets(s, [
    "Primary model M1-fb: logistic regression on stride time mean and stride time variability",
    "Holds in every sensitivity check: other trials, walkway instead of insoles, stricter QC, patient subgroups",
    "Likely reason: in WearGait-PD the controls are 11.5 years older and from a different hospital site",
  ], { x: 0.6, y: 3.3, w: 8.8, h: 1.6, fontSize: 13.5 });
}

// ---------- 3. why ----------
section("Why");
{
  const s = slide("CONTENT", "THE PROBLEM", "Published accuracy rarely survives a new dataset",
    "[1:40-3:00] Two problems. First, leakage: cut walks into windows and split them randomly, and the same person is in train and test. Our base paper itself drops from 99.6 to 82.3 percent when tested on new people. Second, even honest testing inside one dataset only tells you about new people from the same source. A 2026 review found that only 6 percent of 102 plantar-pressure PD studies tested on a different dataset. External validation is what clinical prediction research treats as the real test.");
  card(s, 0.55, 1.45, 4.3, 3.35, HEX.tint, "leakage card");
  text(s, "1  Leakage", { x: 0.8, y: 1.6, w: 3.9, h: 0.35, fontSize: 15, bold: true, fontFace: "Cambria", color: C.accent2 });
  text(s, "99.6% → 82.3%", { x: 0.8, y: 2.0, w: 3.9, h: 0.6, fontSize: 30, bold: true, fontFace: "Cambria", color: C.text1 });
  text(s, "MSCAF-Gait accuracy with random window splits vs new, unseen people (same paper, same data). Gait identifies the person, so random splits reward recognising people, not disease.",
       { x: 0.8, y: 2.7, w: 3.85, h: 1.9, fontSize: 12.5, color: C.text2 });
  card(s, 5.15, 1.45, 4.3, 3.35, HEX.tint, "single dataset card");
  text(s, "2  One dataset", { x: 5.4, y: 1.6, w: 3.9, h: 0.35, fontSize: 15, bold: true, fontFace: "Cambria", color: C.accent2 });
  text(s, "Only 6%", { x: 5.4, y: 2.0, w: 3.9, h: 0.6, fontSize: 30, bold: true, fontFace: "Cambria", color: C.text1 });
  text(s, "test a plantar-pressure PD model on a dataset other than the one it was trained on (2026 review of 102 studies; 82 use gaitpdb). New device, site or population is never tested.",
       { x: 5.4, y: 2.7, w: 3.85, h: 1.9, fontSize: 12.5, color: C.text2 });
  caption(s, "Yuan et al., IEEE TNSRE 2025  ·  Wang, Zhao et al., npj Parkinson's Disease 2026", 0.55, 4.95, 8.5);
}

// ---------- 4. datasets ----------
{
  const s = slide("CONTENT", "THE TWO DATASETS", "Same signal, different device, protocol and people",
    "[3:00-4:00] gaitpdb is three Israeli studies with 8-sensor insoles and long level walks, about 90 strides per foot. WearGait-PD, released in 2026, uses Moticon insoles and four short passes over a 16-foot walkway, about 17 valid strides per foot. Both give total vertical force per foot at 100 Hz, so we can run exactly the same pipeline. The important differences are the device, the walk length, and who was recruited.");
  const rows = [
    ["", "gaitpdb (training)", "WearGait-PD (external)"],
    ["Participants used", "165: 93 PD, 72 controls", "123: 72 PD, 51 controls (113 with insoles)"],
    ["Device", "8 force sensors per foot, 100 Hz", "Moticon OpenGo, 16 sensors, 100 Hz"],
    ["Walk", "Long level walk (~2 min)", "4 passes over a 16-ft walkway"],
    ["Valid strides / foot", "≈ 90", "median 17"],
    ["Control age (median)", "64", "78.5"],
    ["Sites", "3 studies, Israel", "2 Johns Hopkins sites (one for PD, one for controls)"],
  ];
  s.addTable(rows.map((r, i) => r.map((c, j) => ({ text: c, options: {
      bold: i === 0 || j === 0, color: i === 0 ? HEX.white : HEX.ink, fill: { color: i === 0 ? HEX.dark : (i % 2 ? HEX.tint : HEX.white) },
      fontSize: 12, fontFace: "Calibri", valign: "middle", margin: [0.04, 0.08, 0.04, 0.08] } }))),
    { x: 0.55, y: 1.45, w: 8.9, colW: [2.1, 3.0, 3.8], rowH: 0.42, border: { type: "solid", pt: 0.5, color: HEX.lav },
      objectName: "dataset table" });
  caption(s, "Goldberger et al. 2000 (PhysioNet)  ·  Anderson et al., Scientific Data 2026", 0.55, 4.95, 8.5);
}

// ---------- 5. design ----------
section("Method");
{
  const s = slide("CONTENT", "STUDY DESIGN", "We locked everything before looking at the answer",
    "[4:00-5:30] The order matters. We downloaded WearGait-PD and immediately renamed every file to a random code, keeping the diagnosis key separate. Then features, the insole-walkway check, and all decisions were made blind. We froze the model, the training data and the test code with a git tag and SHA-256 fingerprints, scored WearGait-PD, saved the score fingerprints, and only then unblinded. Nothing could be tuned after seeing the result.");
  const steps = [["Blind", "random file codes; diagnosis key kept apart"], ["Features", "same pipeline both datasets"],
                 ["Device check", "insole vs walkway, 2,947 strides"], ["Freeze", "git tag + SHA-256 of every input"],
                 ["Score", "frozen model, label-free"], ["Unblind", "H1–H3 with Holm"]];
  const w = 1.38, gap = 0.124;
  steps.forEach(([h, d], i) => {
    const x = 0.55 + i * (w + gap);
    s.addShape(pres.shapes.OVAL, { x: x + w / 2 - 0.32, y: 1.55, w: 0.64, h: 0.64,
      fill: { color: i === 3 ? HEX.purple : HEX.teal }, line: { color: HEX.white }, objectName: `step ${i + 1} circle` });
    text(s, String(i + 1), { x: x + w / 2 - 0.32, y: 1.55, w: 0.64, h: 0.64, fontSize: 18, bold: true,
      color: C.background1, align: "center", valign: "middle", objectName: `step ${i + 1} number` });
    text(s, h, { x, y: 2.35, w, h: 0.35, fontSize: 14, bold: true, align: "center", fontFace: "Cambria", objectName: `step ${i + 1} head` });
    text(s, d, { x, y: 2.72, w, h: 0.85, fontSize: 11.5, color: C.text2, align: "center", objectName: `step ${i + 1} text` });
  });
  card(s, 0.55, 3.75, 8.9, 1.05, HEX.tint, "hypotheses card");
  text(s, [
    { text: "H1 ", options: { bold: true, color: HEX.purple } }, { text: "external AUC > 0.5     ", options: {} },
    { text: "H2 ", options: { bold: true, color: HEX.purple } }, { text: "still > 0.5 at equal walking speed     ", options: {} },
    { text: "H3 ", options: { bold: true, color: HEX.purple } }, { text: "a gap remains after matching walk length", options: { breakLine: true } },
    { text: "One-sided tests, Holm correction over H1–H3 (family α = 0.05)", options: { color: HEX.muted, fontSize: 11.5 } },
  ].map(r => r), { x: 0.8, y: 3.88, w: 8.4, h: 0.8, isTextBox: true, fontFace: "Calibri", fontSize: 13, color: C.text1, margin: 0, valign: "top" });
}

// ---------- 6. features and models ----------
{
  const s = slide("CONTENT", "FEATURES AND MODEL", "Simple, interpretable stride-timing features",
    "[5:30-6:30] From total force per foot we detect heel strike and toe-off, keep only clean strides inside straight walking, and compute five features. The full model M1 uses all five. M1-fb uses only the two that need heel strikes alone. Models are L2 logistic regression with all scaling and tuning fitted inside training folds, and splits are always by subject.");
  card(s, 0.55, 1.45, 4.3, 3.35, HEX.tint, "M1 card");
  text(s, "M1  (T1-core, 5 features)", { x: 0.8, y: 1.6, w: 3.9, h: 0.35, fontSize: 15, bold: true, fontFace: "Cambria", color: C.accent2 });
  bullets(s, ["Stride time mean", "Stride time variability (CV)", "Swing time variability", "Stance % of stride", "Left–right swing asymmetry"],
          { x: 0.85, y: 2.05, w: 3.8, h: 1.9, fontSize: 13 });
  text(s, "needs heel strike and toe-off", { x: 0.8, y: 4.25, w: 3.9, h: 0.3, fontSize: 11.5, italic: true, color: HEX.muted });
  card(s, 5.15, 1.45, 4.3, 3.35, HEX.tint, "M1-fb card");
  text(s, "M1-fb  (2 features)", { x: 5.4, y: 1.6, w: 3.9, h: 0.35, fontSize: 15, bold: true, fontFace: "Cambria", color: C.accent1 });
  bullets(s, ["Stride time mean", "Stride time variability (CV)"], { x: 5.45, y: 2.05, w: 3.8, h: 0.8, fontSize: 13 });
  text(s, "needs heel strike only", { x: 5.4, y: 2.95, w: 3.9, h: 0.3, fontSize: 11.5, italic: true, color: HEX.muted });
  text(s, "L2 logistic regression · subject-level 20 × 5-fold CV · scaling and C tuned inside training folds",
       { x: 5.4, y: 3.45, w: 3.85, h: 1.1, fontSize: 12.5, color: C.text2 });
}

// ---------- 7. device check ----------
{
  const s = slide("CONTENT", "DEVICE CHECK (BEFORE THE FREEZE)", "Insoles time strides perfectly but not toe-off",
    "[6:30-7:45] Before testing, we compared insole events with the walkway on 2,947 strides, without labels. Whole stride time agrees to 0.4 ms. But toe-off is detected about 35 ms differently, and that varies from person to person, so stance and swing features are not trustworthy on WearGait-PD. Our pre-declared rule then said: use only the heel-strike model, M1-fb, as the primary model. That made the primary model weaker, but honest.");
  const rows = [["Insole − walkway", "Bias", "Tolerance", "Result"],
                ["Stride duration", "+0.4 ms", "≤ 10 ms", "PASS"],
                ["Stance duration", "−35.3 ms", "≤ 20 ms", "FAIL"],
                ["Swing duration (SD)", "39.5 ms", "≤ 15 ms", "FAIL"]];
  s.addTable(rows.map((r, i) => r.map((c, j) => ({ text: c, options: {
      bold: i === 0 || j === 3, fontSize: 13, fontFace: "Calibri", valign: "middle", margin: [0.05, 0.1, 0.05, 0.1],
      color: i === 0 ? HEX.white : (j === 3 ? (c === "PASS" ? HEX.teal : HEX.red) : HEX.ink),
      fill: { color: i === 0 ? HEX.dark : (i % 2 ? HEX.tint : HEX.white) } } }))),
    { x: 0.55, y: 1.5, w: 5.2, colW: [2.0, 1.1, 1.1, 1.0], rowH: 0.45, border: { type: "solid", pt: 0.5, color: HEX.lav },
      objectName: "agreement table" });
  card(s, 6.1, 1.5, 3.35, 3.3, HEX.dark, "rule card");
  text(s, "Pre-declared rule", { x: 6.35, y: 1.65, w: 2.9, h: 0.35, fontSize: 14, bold: true, fontFace: "Cambria", color: C.accent4 });
  text(s, "If toe-off fails, the primary model uses heel-strike features only.\n\nPrimary: M1-fb (internal AUC 0.665)\nSecondary: M1 (0.824)",
       { x: 6.35, y: 2.1, w: 2.9, h: 2.5, fontSize: 13, color: C.background1 });
  text(s, "2,947 paired strides, 113 trials, no labels read. Bland–Altman (2007) repeated-measures limits.",
       { x: 0.55, y: 3.65, w: 5.2, h: 0.7, fontSize: 12, color: C.text2 });
}

// ---------- 8. internal results ----------
{
  const s = slide("CONTENT", "INTERNAL RESULTS (gaitpdb)", "Short walks cost almost nothing internally",
    "[7:45-9:00] Inside gaitpdb, the full model reaches 0.82 and the two-feature model 0.67. The key number for H3: when we cut gaitpdb walks into WearGait-PD's structure, four segments of four or five strides, M1-fb still reaches 0.673. So walk length alone should not cost much. Training on short walks did not help either, so the frozen model is trained on full walks.");
  s.addChart(pres.charts.BAR, [{ name: "AUC", labels: ["M1, full walks", "M1-fb, full walks", "M1-fb, emulated SelfPace walks", "M1-fb, 15 strides per foot"],
    values: [0.824, 0.665, 0.673, 0.646] }], {
    x: 0.45, y: 1.35, w: 5.6, h: 3.5, barDir: "bar", chartColors: [HEX.teal], showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0.00", dataLabelFontSize: 11, dataLabelColor: HEX.ink, dataLabelFontFace: "+mn-lt",
    valAxisMinVal: 0.5, valAxisMaxVal: 0.9, valAxisMajorUnit: 0.1, valAxisLabelFontSize: 10, catAxisLabelFontSize: 11,
    valAxisLabelColor: HEX.muted, catAxisLabelColor: HEX.ink, valAxisLabelFontFace: "+mn-lt", catAxisLabelFontFace: "+mn-lt",
    valGridLine: { color: "E4E3DF", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
    showTitle: true, title: "Internal AUC (subject-level CV)", titleFontSize: 12, titleColor: HEX.ink, titleFontFace: "+mn-lt",
    catAxisOrientation: "maxMin", barGapWidthPct: 60, objectName: "internal AUC chart" });
  card(s, 6.35, 1.45, 3.1, 3.35, HEX.tint, "D15 card");
  text(s, "Walk-structure emulation", { x: 6.55, y: 1.6, w: 2.75, h: 0.35, fontSize: 14, bold: true, fontFace: "Cambria", color: C.accent2 });
  text(s, "gaitpdb walks cut into 4 × 4–5 straight strides (20 draws).\n\nH3 reference: 0.673\n\nShort-walk training gained only +0.005, so the model trains on full walks (D15).",
       { x: 6.55, y: 2.0, w: 2.75, h: 2.7, fontSize: 12, color: C.text1 });
}

// ---------- 9. results ----------
section("Results");
{
  const s = slide("CONTENT", "CONFIRMATORY RESULTS", "The frozen model is at chance on WearGait-PD",
    "[9:00-10:45] Now the frozen test on 113 people, 67 PD and 46 controls. H1: AUC 0.47, confidence interval 0.36 to 0.58, not supported. H2: speed-adjusted AUC 0.55, not supported, and by our rule it only counts if H1 passes. H3: the gap of 0.20 is supported, p 0.007 against a Holm threshold of 0.017. The secondary five-feature model is worse, 0.385, which means it ranks patients lower than controls. The 10 excluded people were 5 and 5, so the exclusion is balanced.");
  s.addImage({ path: FIG + "fig3_external_roc.png", x: 0.45, y: 1.3, w: 3.75, h: 3.58, objectName: "external ROC figure",
               altText: "ROC curves of frozen M1-fb (AUC 0.470) and M1 (AUC 0.385) on WearGait-PD, both near the chance diagonal" });
  const rows = [["Test", "Estimate [95% CI]", "p", "Decision"],
                ["H1  AUC > 0.5", "0.470 [0.357, 0.582]", "0.703", "not supported"],
                ["H2  speed-adjusted", "0.553 [0.444, 0.665]", "0.171", "not supported"],
                ["H3  gap > 0", "0.204 [0.034, 0.342]", "0.007", "supported"]];
  s.addTable(rows.map((r, i) => r.map((c, j) => ({ text: c, options: {
      bold: i === 0 || j === 0 || (j === 3 && c === "supported"), fontSize: 12, fontFace: "Calibri", valign: "middle",
      margin: [0.04, 0.07, 0.04, 0.07], color: i === 0 ? HEX.white : (j === 3 && c === "supported" ? HEX.purple : HEX.ink),
      fill: { color: i === 0 ? HEX.dark : (i % 2 ? HEX.tint : HEX.white) } } }))),
    { x: 4.45, y: 1.45, w: 5.0, colW: [1.45, 1.65, 0.6, 1.3], rowH: 0.45, border: { type: "solid", pt: 0.5, color: HEX.lav },
      objectName: "hypothesis table" });
  bullets(s, ["n = 113 (67 PD, 46 controls); Holm thresholds 0.017 / 0.025 / 0.05",
              "Secondary M1: AUC 0.385, ranking reversed",
              "Recalibration cannot help: it fixes probabilities, not ranking"],
          { x: 4.5, y: 3.5, w: 4.95, h: 1.4, fontSize: 12.5 });
}

// ---------- 10. robustness ----------
{
  const s = slide("CONTENT", "IS IT A BUG?", "The failure holds in every sensitivity check",
    "[10:45-12:15] The first question anyone asks: is it your code? After unblinding we ran about twenty checks, all labelled exploratory. Walkway strides instead of insoles: 0.39. Average of all trials: 0.47. Stricter quality rules, no DBS patients, medication on or off: all near chance. And a model trained inside WearGait-PD itself reaches only 0.45, so stride timing carries no PD signal in this dataset at all. Only age matching moves the estimate up, to 0.57, still uncertain.");
  s.addImage({ path: FIG + "fig1_auc_forest.png", x: 0.45, y: 1.3, w: 5.95, h: 3.6, objectName: "AUC forest figure",
               altText: "Forest plot: internal AUC 0.67, emulated 0.67, external 0.47; sensitivity analyses 0.39 to 0.57, all intervals include 0.5" });
  card(s, 6.65, 1.45, 2.8, 3.35, HEX.tint, "within card");
  text(s, "Trained inside WearGait-PD", { x: 6.85, y: 1.6, w: 2.45, h: 0.6, fontSize: 13.5, bold: true, fontFace: "Cambria", color: C.accent2 });
  text(s, "0.45", { x: 6.85, y: 2.25, w: 2.45, h: 0.7, fontSize: 36, bold: true, fontFace: "Cambria", color: HEX.red });
  text(s, "Same two features, 20 × 5-fold CV inside WearGait-PD. Stride timing carries no PD signal there at all.",
       { x: 6.85, y: 3.0, w: 2.45, h: 1.6, fontSize: 12, color: C.text2 });
}

// ---------- 11. why ----------
{
  const s = slide("CONTENT", "WHY IT FAILED (EXPLORATORY)", "Older controls, from a different site",
    "[12:15-14:00] Why? In gaitpdb the controls are slightly younger than the patients. In WearGait-PD the controls are 11.5 years older, slower and mostly women, and older people walk more irregularly, so they look Parkinson-like to the model. Then we found that the WearGait-PD ID prefixes are hospital sites: every patient came from Johns Hopkins Outpatient Center and every control from Bayview. Another group's 2026 preprint shows site alone predicts diagnosis with AUC 0.955 in this dataset. So we cannot fully separate disease from age or site. We say this plainly in the report.");
  s.addImage({ path: FIG + "fig2_age_by_group.png", x: 0.45, y: 1.3, w: 6.0, h: 2.37, objectName: "age figure",
               altText: "Age histograms: gaitpdb controls median 64 and PD 68; WearGait-PD controls median 78.5 and PD 67" });
  bullets(s, ["Controls older (78.5 vs 67), slower, 70% women",
              "Variability rises with age: old controls look 'PD-like' (ρ = 0.29)",
              "Age-matched (n = 52): AUC 0.57 [0.41, 0.73], still uncertain"],
          { x: 0.5, y: 3.8, w: 5.95, h: 1.15, fontSize: 12 });
  card(s, 6.7, 1.45, 2.75, 3.35, HEX.dark, "site card");
  text(s, "Diagnosis = site", { x: 6.9, y: 1.6, w: 2.4, h: 0.35, fontSize: 14, bold: true, fontFace: "Cambria", color: C.accent4 });
  text(s, "All 72 PD: Johns Hopkins Outpatient Center\nAll 51 controls: Johns Hopkins Bayview\n\nSite alone predicts diagnosis: AUC 0.955 (Sifat et al., 2026 preprint)",
       { x: 6.9, y: 2.05, w: 2.4, h: 2.6, fontSize: 12, color: C.background1 });
}

// ---------- 12. conclusions ----------
section("Close");
{
  const s = slide("CONTENT", "WHAT WE LEARNED", "An honest negative result, and why it matters",
    "[14:00-15:30] Three messages. One: a gaitpdb stride-timing model does not identify mild, treated PD in an independent cohort. Two: walk length and stride measurement are not the cause; case-mix and site are the leading explanation, which we cannot fully resolve. Three: the frozen design is what makes this believable. Without it, someone could have picked the age-matched 0.57 or the walkway turning result and reported success.");
  const items = [["1", "No transport", "Internal AUC 0.67–0.82 did not carry over: external 0.47 (M1-fb), 0.385 (M1)."],
                 ["2", "Not length, not measurement", "Emulated short walks keep 0.67; insoles match the walkway on stride time (0.4 ms)."],
                 ["3", "Case-mix and site", "Older controls from a different site; disease vs age vs site cannot be fully separated."],
                 ["4", "Why freezing matters", "Exploratory variants reach 0.57–0.66. Only the frozen test can be trusted."]];
  items.forEach(([n, h, d], i) => {
    const x = 0.55 + (i % 2) * 4.5, y = 1.45 + Math.floor(i / 2) * 1.7;
    card(s, x, y, 4.35, 1.55, HEX.tint, `lesson ${n} card`);
    s.addShape(pres.shapes.OVAL, { x: x + 0.2, y: y + 0.2, w: 0.5, h: 0.5, fill: { color: HEX.teal }, line: { color: HEX.teal },
      objectName: `lesson ${n} circle` });
    text(s, n, { x: x + 0.2, y: y + 0.2, w: 0.5, h: 0.5, fontSize: 15, bold: true, color: C.background1, align: "center", valign: "middle" });
    text(s, h, { x: x + 0.85, y: y + 0.2, w: 3.3, h: 0.35, fontSize: 14, bold: true, fontFace: "Cambria", color: C.accent2 });
    text(s, d, { x: x + 0.85, y: y + 0.6, w: 3.3, h: 0.85, fontSize: 12, color: C.text1 });
  });
}

// ---------- 13. limitations and future ----------
{
  const s = slide("CONTENT", "LIMITATIONS AND NEXT STEPS", "What we cannot say, and what would settle it",
    "[15:30-17:00] Limitations: 113 people gives an AUC interval of about plus or minus 0.11; the device check limited us to two features; the age and site explanation is exploratory. What would settle it: the third WearGait-PD site, VA Puget Sound, recruited both patients and controls, so a release including it would allow a within-site test. Richer pressure features such as centre of pressure, and turning, which looked promising on the walkway, are the next things to test, again with a frozen protocol.");
  card(s, 0.55, 1.45, 4.3, 3.35, HEX.tint, "limits card");
  text(s, "Limitations", { x: 0.8, y: 1.6, w: 3.9, h: 0.35, fontSize: 15, bold: true, fontFace: "Cambria", color: C.accent2 });
  bullets(s, ["n = 113: AUC CI about ± 0.11", "Two features only (toe-off failed the device check)",
              "Age and site explanation is exploratory", "Different speed measures, so speed is used within a dataset only"],
          { x: 0.85, y: 2.05, w: 3.8, h: 2.6, fontSize: 12.5 });
  card(s, 5.15, 1.45, 4.3, 3.35, HEX.tint, "next card");
  text(s, "Next steps", { x: 5.4, y: 1.6, w: 3.9, h: 0.35, fontSize: 15, bold: true, fontFace: "Cambria", color: C.accent1 });
  bullets(s, ["Within-site test with VA Puget Sound (PD and controls at one site)", "Centre-of-pressure and per-sensor features",
              "Turning on the walkway (hint: AUC 0.61–0.66)", "Same frozen protocol for every new claim"],
          { x: 5.45, y: 2.05, w: 3.8, h: 2.6, fontSize: 12.5 });
}

// ---------- 14. thanks ----------
{
  const s = slide("DARK", "THANK YOU", "Questions",
    "[17:00] Thank you. Everything is reproducible from the public repository: the freeze tag, the fingerprints, and the project log that records every decision with its date. Happy to take questions.");
  text(s, "Code, freeze tag `model-frozen`, SHA-256 fingerprints and full decision log:\ngithub.com/Gurkaran18/Foot-Pressure-Based-Abnormal-Gait-Recognition",
       { x: 0.7, y: 2.75, w: 8.6, h: 0.75, fontSize: 13, color: C.accent5 });
  text(s, "Data: gaitpdb (PhysioNet) and WearGait-PD (Synapse); not redistributed.",
       { x: 0.7, y: 4.3, w: 8.6, h: 0.4, fontSize: 11.5, color: C.background1 });
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
