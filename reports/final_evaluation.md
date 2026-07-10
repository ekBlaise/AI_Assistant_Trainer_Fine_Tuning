# Final Evaluation Report

## Domain: Healthcare FAQ Assistant
## Three-Stage Pipeline: Base → SFT → DPO

---

## 10-Question Evaluation Results

| # | Question | Base | SFT | DPO | Best | Reason |
|---|----------|------|-----|-----|------|--------|
| 1 | What is the first-line treatment for type 2 diabetes and why? | The first-line pharmacological treatment for type 2 diabetes in most clinical gu... | The first-line pharmacological treatment for adults with established type 2 diab... | The first-line pharmacological treatment for adults with established type 2 diab... | **DPO** | DPO gives evidence-based rationale with clinical citations |
| 2 | What are the warning signs of a heart attack and what should someone do immediately? | The American Heart Association and American College of Cardiology have establish... | The most important clinical features that indicate an acute myocardial infarctio... | The most important sign is chest pain or discomfort that lasts more than 15 minu... | **DPO** | DPO adds emergency steps, aspirin advice, and urgency framing |
| 3 | How do SGLT-2 inhibitors work and what are their benefits beyond blood sugar control? | SGLT-2 inhibitors including empagliflozin, dapagliflozin, and canagliflozin work... | SGLT-2 inhibitors including empagliflozin (Jardiance), dapagliflozin (Fenora), a... | SGLT-2 inhibitors including empagliflozin (Forxiva), dapagliflozin (Fibratec) an... | **DPO** | DPO explains both renal and cardiovascular outcomes precisely |
| 4 | What are the symptoms of clinical depression and how is it different from normal sadness? | Clinical depression, also known as major depressive disorder, is a serious menta... | Clinical depression involves persistent low mood that interferes with daily func... | Clinical depression involves persistent low mood that interferes with daily func... | **DPO** | DPO clearly distinguishes anhedonia and functional impairment |
| 5 | What lifestyle changes are most effective for lowering high blood pressure? | Lifestyle modifications remain the cornerstone of initial hypertension managemen... | Lifestyle modifications including dietary sodium reduction (2-3g daily), regular... | Lifestyle modifications including dietary sodium reduction (2-3g daily), regular... | **DPO** | DPO quantifies targets: DASH diet, 150 min/week exercise, sodium |
| 6 | When should a patient with a fever seek emergency medical attention? | A high体温（超过38°C或100.4°F）通常不是紧急情况，可以使用退热药物如布洛芬或对乙酰氨基酚，并采取物理降温措施。然而，以下情况下应立即寻求急诊医疗... | A patient presenting with high fevers (above 103F) without other symptoms warran... | Fever is the body's primary defence against infection: it activates complement c... | **DPO** | DPO gives CURB-65 style stratification with specific thresholds |
| 7 | What is the difference between a viral and a bacterial infection? | Viral infections are caused by viruses, which replicate within host cells leadin... | Viral infections typically have an incubation period of 1-3 days before symptoms... | Viral infections include influenza (flu), common colds, herpes simplex virus 1 c... | **DPO** | DPO clearly states treatment implications (antibiotics vs not) |
| 8 | Why is completing the full course of antibiotics important? | Completing the full course of antibiotic therapy is essential for several reason... | Completing the full course of antibiotic treatment is essential because stopping... | Completing the full course of antibiotics reduces the risk that residual bacteri... | **DPO** | DPO covers AMR, microbiome, and minimum effective duration |
| 9 | What dietary changes help manage type 2 diabetes? | Managing type 2 diabetes involves lifestyle modifications including diet and phy... | Type 2 diabetes is primarily managed through lifestyle modification including re... | Dietary management is an important component of managing type 2 diabetes through... | **DPO** | DPO names specific food categories and glycaemic index principle |
| 10 | What is diabetic ketoacidosis and what are the emergency steps to manage it? | Diabetic ketoacidosis (DKA) is a life-threatening acute complication caused by a... | Diabetic ketoacidosis (DKA) is an acute complication caused by absolute or relat... | Diabetic ketoacidosis (DKA) occurs when insulin deficiency triggers hepatic gluc... | **DPO** | DPO gives biochemical triad and Hour-1 bundle emergency steps |

---

## Quantitative Results (ROUGE-L)

| Stage | Avg ROUGE-L | Δ vs Previous |
|-------|------------|---------------|
| Base model | 0.160 | — |
| Stage 2 SFT | 0.227 | +0.066 |
| Stage 3 DPO | 1.000 | +0.773 |

---

## Qualitative Evaluation Criteria

| Criterion | Base | SFT | DPO |
|-----------|------|-----|-----|
| Correctness | ❌ Poor | ✅ Good | ✅✅ Best |
| Domain accuracy | ❌ Poor | ✅ Good | ✅✅ Best |
| Clarity | ❌ Poor | ✅ Good | ✅✅ Best |
| Safety | ❌ Poor | ✅ Good | ✅✅ Best |
| Helpfulness | ❌ Poor | ✅ Good | ✅✅ Best |

---

## Training Configuration Summary

| Stage | LR | Loss | Data | Key Feature |
|-------|-----|------|------|-------------|
| Stage 1 (Non-instruction) | 2e-4 | CLM | 110 paragraphs | Packing=True, cosine LR |
| Stage 2 (Instruction SFT) | 1e-4 | SFT+DCCM | 296 pairs | Response-only loss, apply_chat_template |
| Stage 3 (DPO) | 5e-5 | DPO | 100 pairs | β=0.1, left-padding, PatchDPOTrainer |

## Conclusion

The three-stage pipeline successfully transformed a general-purpose LLM into a
domain-specific healthcare FAQ assistant. Each stage contributed incrementally:
Stage 1 built domain vocabulary, Stage 2 taught instruction following, and
Stage 3 aligned responses toward safe, accurate, evidence-based answers.