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
1. تجميع حسب الفقاعة بـ flood fill على الورق: نص في فقاعتين ينقسم، كتلتين في فقاعة تتدمجان، الباقي يرجع كماسك متبقٍّ (تصحيح: `layout.py` ما فيهوش مسار SFX خاص؛ `pipeline.py` هو اللي بيتعامل مع النص خارج الفقاعات).
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

ملخص OverTranslate الكامل (183 سطر): `docs/refs-overtranslate.md`. تفاصيل: كاشف RT-DETR (640 squash، حد 0.3، دمج 0.7)، OCR بـ CPT-flex (شبكة 196 patch، decode جشع بحد 160 توكن)، CRAFT بدخل 384 وقطع عند أقل حبر، SecondLook بمقياس 0.77، وقاموس طويل-أولاً بحد 20 مدخل/600 حرف، وتاريخ 16 زوج بقص 2000 توكن. Inpaint.cs منفذ OpenCV (Apache-2.0) فنستخدم `cv2.inpaint` بـ INPAINT_NS. رخصة أوزان RT-DETR لسه ما اتفحصتش، وما اتقراش: MangaLineGeometry وLumaPage وVerticalColumnGrouping وCpuAdaptiveBackground.

ملخص BallonsTranslator الكامل (266 سطر): `docs/refs-ballons.md`. القسم 1.9 فيه وصفة تشغيل Comic Text Detector لوحده: إدخال ONNX ثابت 1024 عبر cv2.dnn، حشو أسفل ويمين بس، ترتيب مخرجات mask/lines معكوس، فك DBNet (thresh 0.3، unclip 1.5، قطع 0.6)، وتغذية BGR لـ ONNX (مسار torch بيقلب القنوات). أفخاخ: كتل YOLO بتتحسب وتتجاهل، و`panel_finder.closest_text_to_panel_index` فيه خطأ (ما بيسجّلش أقرب panel). التنظيف: AOT بتحجيم 2048 وحشو 128، LaMa بحشو مضاعفات 64، وطريق مختصر للفقاعة المسطحة (std 7/10). مستخرج Qt-free لسياق الترجمة (تاريخ/قاموس/ذاكرة) يستاهل النقل. `bubble_shape` مربوط بـ Qt.

ملخص Manga-Translator-OCR الكامل (169 سطر): `docs/refs-manga-translator-ocr.md`. الترخيص MIT (Kaushik Paul 2026) متأكد بالـ grep؛ الأوزان والخطوط ليها تراخيصها ولسه ما اتفحصتش. `layout.py` شبه جاهز للنقل (flood fill على الورق مع تحقق بالألوان والتشبع ونسبة الثقوب والمساحة، ودمج union-find). `translator.py`: دفعات 10، parser مرقّم، إصلاح الدفعة ثم الأسطر الفاشلة لوحدها. `renderer.py` قريت دوال محددة بس؛ أفضل ما فيه `render_text_in_bubble` (كلمات على أشرطة صفوف من distance transform لماسك الفقاعة)، لكن `_sanitize_for_font` بيعتمد على `fc-query` وثوابت بكسلات مطلقة، وترتيب القراءة عندهم 3 أشرطة خام.

## ملحق: المحرك القديم (ملخص كامل في `docs/refs-old-engine.md`، حوالي 190 سطر)

- **تصحيح مهم:** المحرك القديم بيترجم **فارسي**، والعربي بيشتغل بس عبر البوت من `legacy/manga_ar.py` (10,410 سطر، ما اتقراش) بـ Ollama.
- **OCR:** RapidOCR على ONNX (PP-OCRv6)، بإعادة بناء على CPU بعد خطأ GPU. لكل قصة سلم توقف مبكر: خام ثم CLAHE ثم Otsu ثم مقلوب، ومسار سريع لسطر واحد بيتخطى الكشف.
- **الاستخراج:** تقييم نوافذ النظام بحوالي 120 كلمة مفتاحية (EN/JA/KO/ZH)، مسح العلامات المائية والبانرات (بلاطات مع ميزانية OCR)، نص منحني (مطابقة تربيعية ثم تدوير الحروف). `_discover_free_text_regions` مستدعاة من كود SFX ومش معرّفة في النسخة دي، فاكتشاف SFX عندهم كود ميت.
- **الترجمة:** سلالم إعادة المحاولة وتدوير المفاتيح وتوقيت RPM لكل مفتاح وتبريد `retryDelay` وترتيب شلال الموديلات.
- **الرسم:** arabic_reshaper + bidi بـ Layout.BASIC وفحص تغطية الخط (عندنا HarfBuzz أحسن).
- **التنظيف:** جدول حوالي 55 دالة حالات خاصة مع أرقام الأسطر. أولوية النقل مستنتجة من تاريخ الأخطاء في PROGRESS.md.
- **Discord:** 12 أمر slash، طابور asyncpg بـ `FOR UPDATE SKIP LOCKED`، عامل subprocess يقرأ التقدم بـ regex من السجل، وتسليم حسب الحجم (سلم JPEG وتقطيع وبديل Gofile).
- **فجوات:** أعماق الثوابت داخل دوال التنظيف الطويلة، وتنسيق `20_models`/`95_run` اتقروا جزئياً.

## ملحق: clean-bot (Master-V00/clean-bot، ملخص كامل في `docs/refs-clean-bot.md`، 166 سطر)

- **الترخيص:** ملف LICENSE هو MIT (omar soliman 2025)، لكن الوكيل بيقول (من الذاكرة، **مش متحقق منه**) إن الكود فورك من pcleaner/PanelCleaner المظنون GPL-3. أوزان YOLO عبر Ultralytics (AGPL). Hi-SAM Apache-2. لاستخدامك الخاص مش عائق، بس نسجّله.
- **الكشف:** Hi-SAM (vit_h/vit_b، 900 نقطة foreground، nms 0.42، درجة 0.5) أو ComicTextDetector أو YOLO11s. كله بيرجّع صناديق بس والماسك بيتحوّل لمستطيلات معبّاة. مسارات ONNX الاحتياطية لـ CTD/YOLO شكلها مكسورة.
- **التنظيف:** لون مسطح بمعيار انحراف حواف (تضخيم 2 بكسل × 11 خطوة، يقبل ماسك أكبر لو أحسن 10%، يفشل فوق std 15)، وبعدين LaMa على الصفحة كلها من غير تقسيم. نموهم للماسك: 10+0.2·std بحد 20، عزل +8، تلاشي 4. الضغط المسبق لـ 3000px JPEG q85 بيخسّر جودة. سلم OOM للكاشف (GPU ثم CPU ثم vit_b) يستاهل النقل.
- **الطابور والبوت:** عامل asyncio تسلسلي يحجز صفوف Supabase بـ UPDATE شرطي من غير تنظيف للمهام العالقة. بوت واحد `index.js`: نموذج بيستلم رابط Drive، استطلاع كل 10 ثواني، تسليم في القناة أو خاص، مستويات رصيد، حذف Drive بعد 20 دقيقة (والحذف غالباً بمعرّف غلط). مفيش cooldown أو حد تزامن لكل مستخدم.
- **المقارنة:** سلمنا (cost-routed) أدق من تصميمهم (مسطح ثم LaMa كامل). اللي يستاهل النقل: درجة انتظام الحواف للون المسطح، ووصفة دمج LaMa (ماسك ثنائي داخل، دمج ألفا ناعم خارج).
