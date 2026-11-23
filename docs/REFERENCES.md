# مراجع المشاريع الستة

ملف واحد نرجعله بدل ما نفتح المشاريع كل مرة. لكل مشروع: إيه هو، فين الحاجات المهمة، نقاط القوة، اللي ناخده، والضعف.

**قاعدة الترخيص (قرار Nader):** المشروع للاستخدام الخاص، مش تجاري ومش هيتوزّع. فالتراخيص GPL وCC BY-NC-SA **مش عائق**، وبنستخدم أي كود أو موديل مفيد. الترخيص بيتسجّل هنا وفي `NOTICE.md` كمصدر بس. **لو المشروع اتشارك أو اتباع في يوم، الحاجات المعلّمة بـ ⚠ لازم تتراجع أو تتشال.**

**درجة التحقق:** ✔ اتقرأ الكود فعلاً · ◐ اتقرأ README أو docstring أو اسم الدالة بس · ✘ لسه ما اتقراش. كل الأرقام المنقولة هي قياسات أصحاب المشاريع على بياناتهم، مش قياساتنا.

## 0. الجدول السريع

| المشروع | اللغة | الترخيص | يستحق إيه |
|---|---|---|---|
| Koharu | Rust | MIT / Apache-2.0 | معمارية المراحل، ترتيب القراءة، قواعد الكشف، عتبات التنظيف، كتابة النص |
| KOMA Studio | Python + TS | MIT | موديلات AOT/LaMa وعقدها، تقطيع الأشرطة، القاموس، فحوص الجودة، إدارة الـ GPU، Comic Text Detector ⚠ GPL |
| المحرك القديم | Python | MIT (نفس صاحب المشروع) | خبرة التنظيف (14 ألف سطر)، RT-DETR، OCR، تحكم المعدل، طبقة Discord |
| OverTranslate | C# | MIT | قراءتين بحجمين، CRAFT وتقطيع الأعمدة، الفوريغانا، مفاوضة الـ prompt، أوزان CPT-flex ⚠ NC |
| BallonsTranslator | Python | GPL-3.0 ⚠ | عدة فرضيات للماسك، Comic Text Detector، canny flood، panel_finder |
| Manga-Translator-OCR | Python | MIT | تجميع حسب الفقاعة، مرحلة إصلاح الترجمة، أفكار الرسم |

## 1. Koharu (koharu-rs/koharu)

**إيه هو:** تطبيق ديسكتوب بـ Rust لترجمة المانجا بالكامل، workspace من crates صغيرة. ◐✔

**أماكن مهمة:**
- `crates/koharu-scene`: الـ scene graph (صفحات وعقد).
- `crates/koharu-pipeline`: المراحل.
- `crates/koharu-ml/src/koharu_layout_highres/` (`suppress.rs`, `layout.rs`): كشف التخطيط وقواعد التكرار ✔.
- `crates/koharu-ml/src/lama`, `manga_ocr_ar`, `baberu_ocr`, `hayai_ocr`, `qwen_image`, `flux3_klein`, `moss_image`: موديلات.
- `crates/koharu-renderer`: كتابة النص. `crates/koharu-psd`: تصدير PSD. `crates/koharu-agent`: واجهة MCP.

**نقاط القوة:**
1. كل مرحلة تكتب حقولها بس في الـ Block، فتبديل الموديل ما يكسرش حاجة.
2. قواعد الكشف بأرقام مقاسة: تكرار IoU ≥ 0.5، نص متداخل ≥ 0.8، نص داخل فقاعة ≥ 0.5 ✔ (منقولة في `detect.py`).
3. عتبات اللون المسطح للتنظيف: mean ≤ 10 وmax < 64 ✔ (منقولة).
4. ترجمة بمحادثة واحدة للفصل، JSON بمفاتيح.
5. كتابة بـ HarfBuzz وحجم خط موحد للصفحة.

**ناخده:** المعمارية (تم)، قواعد الكشف (تم)، الترجمة (مصمّمة)، الكتابة (لسه)، PSD (دفعة 4).

**ضعف / تحفظات:**
- تصحيح: ما فيش "Koharu Model License" في الريبو (اتفحص بـ grep). الكود MIT/Apache-2.0. رخصة أوزان موديل التخطيط على Hugging Face لسه ما اتقراتش، فنقراها قبل ما نحمّله. نكمل بـ RT-DETR لحد وقتها.
- Qwen-Image 2.1 عندهم تحت Qwen Research License (بحث وتقييم بس).
- OCR في Koharu كله ياباني (manga_ocr_ar وBaberu) أو ja/zh/ko/en (Hayai)؛ مفيش OCR عربي. العربي هدف ترجمة وكتابة بس، بـ unicode-bidi وHarfRust من غير كود خاص.
- ⚠ manga-ocr-ar: CC BY-NC-SA، وهو لمانجا يابانية رغم اسمه.
- إرسال صورة الصفحة للـ LLM: قاسوا 16 صفحة ما لقوش تحسن وزاد الوقت.

## 2. KOMA Studio (klaus-2/koma-studio)

**إيه هو:** تطبيق Electron/Tauri بباك-إند بايثون (`packages/mini-backend`) وبوت دعم Discord (`apps/discord-support-bot`). ◐✔

**أماكن مهمة:**
- `packages/mini-backend/models/inpainting`: عقد AOT وLaMa ONNX ✔ (منقولة في `model_tiers.py`).
- `services/splitter_analysis.py`: تقطيع الأشرطة ✔ (منقول في `strips.py`).
- `core/`, `mini_backend_tools/`: إدارة العمليات، الكاش، الجدولة ◐.
- Comic Text Detector: GPL-3.0 حسب `docs/THIRD-PARTY-NOTICES.md` عندهم، والأوزان بتتنزّل وقت التشغيل. تصحيح: الـ adapter عندهم بيستعمل رأس الصناديق بس وبيتجاهل ماسك النص؛ الماسك عندهم من Otsu + connected components (`baka_segmenter.py`) أو من تحويل الصناديق لماسك.

**نقاط القوة:** جودة الترجمة (قاموس، never-translate، فحوص بقايا اللغة والتكرار والفراغ)، تحويل OOM على الـ GPU لـ CPU، كتالوج موديلات بحد VRAM، طابور مهام.

**ناخده:** عقد AOT/LaMa (تم)، الأشرطة (تم)، القاموس والفحوص (دفعة 2)، إدارة الـ GPU (دفعة 3). ماسك Comic Text Detector نفسه مش موجود في كود KOMA (بيتجاهلوه)، فلو عايزينه نلاقي الوصفة في BallonsTranslator (وكيل 5). المفيد فعلاً من KOMA: عقد `BaseInpainter` (عتبة ماسك 128، دمج مخفّف الحواف محصور بالماسك)، تمرير تعبئة لون مسطح من إحصاءات الحلقة، فحوص QA (echo وresidue ونسبة الطول وidentical)، وذاكرة ترجمة sqlite بمطابقة تامة، وفحص VRAM حر قبل OCR VLM (8/10/14GB)، وفترة cooldown للمزوّد 300 ثانية. أوزان AOT وLaMa-manga (ogkalu) ملهاش رخصة مذكورة.

**ضعف:** معظم الكود TypeScript/واجهة، مش محتاجينه؛ الجرد مبني على القراءة ما اتشغلش.

## 3. المحرك القديم (Nader-Geek/Manga-AutoTranslate--Arabic، فرع new-engine)

**إيه هو:** محرك بايثون كبير، `manga_sections/` حوالي 36 ألف سطر. ✔ (أجزاء)

| الملف | الأسطر | فيه إيه |
|---|---|---|
| `20_models.py` | 2693 | تحميل ONNX، خيارات ORT، فحص تطابق GPU/CPU ✔ |
| `30_ocr.py` | 1330 | OCR ◐ |
| `70_extract.py` | 5445 | الكشف والاستخراج ◐ |
| `80_translate.py`, `50_translator.py` | 2087 / 1940 | الترجمة، تحكم المعدل، تدوير المفاتيح ◐ |
| `85_cleanup.py` | **14284** | التنظيف: `_bubble_interior_mask`, `_absorb_text_halo`, `_scrub_*_residuals` ✔ (منقولة) |
| `90_render.py` | — | الرسم ◐ |
| `95_run.py` | 3355 | التشغيل ◐ |
| `bot/` | — | طبقة Discord: أوامر، حدود، فحوص رفع ◐ |

**قوته:** خبرة تنظيف متراكمة من صور حقيقية. **نقاطه الجاهزة للنقل لما صورة تفشل:** قاعدة حالة بحالة من `85_cleanup.py`؛ عنده نسخة `lama-manga-nofft.onnx` (ما اتقراش بالتفصيل).

**ضعف:** ملف واحد 14 ألف سطر، حالات حافة متشابكة؛ ننقل حالة بحالة بس.

## 4. OverTranslate (Hon-Lu/OverTranslate) — MIT

**إيه هو:** تطبيق C# (.NET) للترجمة الفورية والمانجا، بتركيز على OCR عمودي ياباني. ✔ (الملفات اللي تحت)

**أماكن مهمة:**
- `src/OverTranslate/Services/Ocr/Manga/`: `MangaVerticalReader`, `MangaPatchInput`, `MangaColumnCuts`, `MangaColumnAngles`.
- `Services/Ocr/Vertical*`: `SecondLook`, `RepeatedColumns`, `RubyColumns`.
- `Services/Realtime/RealtimeNaturalBackground.cs`، `Imaging/Inpaint.cs` (Navier-Stokes من OpenCV).
- `ocrmodels/manga-vertical.json`: وصف جرافات الـ ONNX (encoder، decoder-cross، decoder-step مع KV cache، `vocab.txt`).
- `OverTranslate.Translation/OpenAi/OpenAiPrompt.cs`.

**نقاط القوة (أرقامهم):**
1. قراءتين بحجمين للكاشف: 98 → 111-119 من 132 فقاعة، 15 إطار.
2. CRAFT (MIT, craft_mlt_25k) يقطع الأعمدة قبل الـ OCR: manga-ocr بيضيّع أعمدة بعد حوالي 40 حرف؛ الخطأ 21.6% ← 3.6% على 301 فقاعة.
3. قواعد الفوريغانا: عمود ضيق كانا بس لاصق يمين عمود الكانجي يتشال قبل التجميع؛ التكرار جوه صندوق تاني يتحسم بالنص.
4. تبطين المسح غير متماثل (فوق 8، تحت 4).
5. الـ prompt: تسميات "日文：… / 繁體中文：…" أحسن من XML بنسبة 2:1 (اختبار أعمى)، الأمر آخر الرسالة، السياق ترجمات سابقة بس، system/user ما فرقش.
6. أوزان CPT-flex ONNX fp16 (`hon-lu/manga-ocr-base-CPT-flex-onnx-fp16`) ⚠ CC BY-NC-SA 4.0 (الأصل muscgab).

**ناخده:** الأرقام 1 و2 و3 و4 و6 (الآن مسموحة)، الـ prompt كتجربة A/B ضد تصميمنا مش بدل منه.

**ضعف:** C#، فنأخد الخوارزميات مش الكود؛ مفيش قياسات على مانهوا ولا ويبتون.

## 5. BallonsTranslator (dmMaze/BallonsTranslator) — GPL-3.0 ⚠

**إيه هو:** محرر ديسكتوب (PyQt) لترجمة المانجا بوحدات قابلة للتبديل: كاشف، OCR، مُنظّف، مترجم. ◐✔

**أماكن مهمة:**
- `modules/textdetector/ctd/textmask.py`: `minxor_thresh`, `get_topk_masklist` ✔.
- `utils/textblock_mask.py`: `canny_flood`, `connected_canny_flood`, `extract_ballon_mask` ✔.
- `utils/bubble_shape.py` ✔، `modules/textdetector/panel_finder.py` ✔، `modules/inpaint/inpaint_default.py` ✔.

**نقاط القوة:**
1. عدة فرضيات للماسك: Otsu لكل قناة، وألوان أعلى 3 تجمعات ±30، ويتختار الأقل XOR مع ماسك متوقع.
2. استخراج ماسك الفقاعة بـ canny + flood fill.
3. Comic Text Detector (كاشف + ماسك نص) جاهز.
4. panel_finder: عتبة ← contours ← ترتيب (للمانجا التقليدية مش الويبتون).

**ناخده (مسموح للاستخدام الخاص):** `get_topk_masklist` كأساس لماسك متعدد الفرضيات مع خريطة CRAFT كماسك متوقع؛ `extract_ballon_mask` لتقسيم الفقاعات؛ Comic Text Detector كخيار ماسك؛ panel_finder للمانجا العادية.

**ضعف:** معماريته ملتصقة بالواجهة؛ نستخرج الدوال مش الإطار.

## 6. Manga-Translator-OCR (Kaushik-Paul) — MIT

**إيه هو:** خط أنابيب بايثون صغير: layout، OCR، ترجمة، رسم. ◐ (docstrings وقوائم الدوال بس)

**أماكن مهمة:** `main/layout.py`, `translator.py`, `renderer.py`, `README.md`, `AGNETS.md`.

**نقاط القوة:**
1. تجميع حسب الفقاعة بـ flood fill على الورق: نص في فقاعتين ينقسم، كتلتين في فقاعة تتدمجان، الباقي مسار SFX.
2. مرحلة إصلاح الترجمة: تكتشف الياباني المتبقي والروماجي وكلام الرفض وتعيد طلب الفاشل بس، بحسب عدد متوقع للأسطر.
3. أفكار الرسم: تنسيق السطور على أشرطة عرض الفقاعة، تنظيف النص من حروف ناقصة في الخط، لون الحد من لون النص، تشطير في الفقاعات الضيقة.

**ناخده:** 1 و2 (الكود MIT)، 3 لما نوصل لـ typeset.

**ضعف:** ما اتقراش كود كامل، ولا اتشغّل؛ مشروع صغير ومفيش قياسات جودة.

## 7. الباكلوج المرتّب (ما يُنقل، بالأولوية)

مربوط بالفجوات اللي ظهرت على فصل الـ 13 صفحة (T4):

| # | البند | المصدر | بيعالج |
|---|---|---|---|
| 1 | **Comic Text Detector (ماسك نص)** يتجرّب كبديل/مكمّل لـ `textmask` | KOMA/Ballons ⚠ GPL | أشباح النص على الفقاعات الداكنة، المستطيل الأبيض في صفحة 07 |
| 2 | قراءتين بحجمين للكاشف | OverTranslate | النص الصغير الفايت (صفحة 10، 03) |
| 3 | CRAFT: خريطة الحروف ← تقطيع أعمدة قبل OCR + ماسك متوقع | OverTranslate | OCR للكتل الطويلة، ماسك أدق |
| 4 | ماسك متعدد الفرضيات + min-XOR | Ballons ⚠ | الماسك على خلفيات مخططة |
| 5 | CPT-flex OCR (ONNX fp16 + KV cache) كبديل لـ manga-ocr-base ويتقارن | OverTranslate ⚠ NC | دقة القراءة، السرعة |
| 6 | فوريغانا: حذف قبل التجميع | OverTranslate | دمج غلط في الجُمل |
| 7 | تبطين غير متماثل للماسك | OverTranslate | علامات الداكوتن |
| 8 | تقسيم الصندوق على فقاعتين / دمج في فقاعة | Manga-Translator-OCR + Ballons | تجميع النص |
| 9 | مرحلة إصلاح الترجمة | Manga-Translator-OCR | رفض وبقايا لغة |
| 10 | قاموس بالمداخل الظاهرة في الصفحة + تاريخ بحد توكنز | KOMA + Ballons + OverTranslate | جودة وتكلفة الترجمة |
| 11 | فلتر ثبات عرض الخط | Ballons | زخارف تُكشف كنص (صفحة 12) |
| 12 | SFX: كشف + تصنيف | KOMA + القديم | المؤثرات الصوتية الفايتة |
| 13 | panel_finder | Ballons | مانجا تقليدية فقط |

**الترتيب المقترح:** 1 ثم 2 ثم 3 (أكبر أثر على الجودة المرئية)، وكل واحد يتقاس على Colab قبل التالي.

## 8. اللي لسه مجهول

- مش مثبت إن أي بند أعلاه يحسّن عندنا؛ كلها أرقام أصحابها.
- الأوزان من HuggingFace مش قابلة للتجربة من الـ sandbox، التجريب على Colab.
- ✘ ما قريتش كود كامل للـ OCR والترجمة في المحرك القديم، ولا `renderer.py` ولا `translator.py` في Manga-Translator-OCR.

## ملحق: تفاصيل Koharu المقروءة بالكود

الملخص الكامل (97 سطر، كل ادعاء موسوم read code/inferred) في ملف الـ scratchpad `01-koharu.md`. أهم ما فيه: عتبات الكشف (onomatopoeia 0.4، bubble 0.4)، ترتيب القراءة XY-cut، bubble holder عند 0.5، "غرف" الفقاعة، قص الـ OCR مع تفريغ الجيران بلون الوسيط، سلم التنظيف (لون مسطح ← plane fit ← Telea ← LaMa بـ 800px وهامش 128)، prompt الترجمة بـ JSON مرقّم مع echo guard، وكاسر السطور DP بحد تصغير 0.5 وارتفاع سطر 1.2. الـ typeset عندهم مش مرحلة منفصلة (التموضع بيتكتب وقت الترجمة).

ملخص KOMA الكامل (112 سطر): `docs/refs-koma.md`. وملخص Koharu: `docs/refs-koharu.md`.
