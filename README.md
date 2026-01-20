## Model Comparison: Small vs. Augmented Dataset

The following table compares the output of the AraBART model trained on the **Small Dataset** versus the **Augmented (Big) Dataset** on specific test examples.

| Feature | Source Text | Reference (Target) | Small Model Output | Augmented (Big) Model Output | Analysis |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Definiteness (AL)** | ... أن أزور **مدينة** التى هاجر ... | ... أن أزور **المدينة** التي هاجر ... | ... أن أزور **مدينة** التي هاجر ... | ... أن أزور **مدينة** التي هاجر ... | Both models failed to add the definite article "AL" (ال) in this specific complex sentence. |
| **Grammar & Semantics** | ... **محتاج أهل الهند** إلى الشريعة لأن هناك يوجد **كثيرا** ... | ... **أهل الهند محتاجون** إلى الشريعة ؛ لأن هناك يوجد **كثير** ... | ... **محتاج أهل الهند** إلى الشريعة ؛ لأن هناك يوجد **كثير** ... | ... **محتاج أهل الهند** إلى الشريعة ؛ لأن هناك يوجد **كثيرا** ... | **Small Model**: Correctly fixed "kathera" (accusative) to "kather" (nominative/default). <br> **Big Model**: Missed the grammar fix but added punctuation. Both kept the original word order. |
