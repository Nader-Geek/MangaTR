# MangaTR

نواة لترجمة المانجا والمانهوا والكوميكس لعربي وإنجليزي: كشف، OCR، ترجمة، تنظيف، وكتابة النص داخل الحبابة. البوت طبقة رفيعة فوقها.

**الحالة:** خط الأنابيب كامل ومختبَر بنماذج وهمية (181 اختبار): كشف ← OCR ← تنظيف ← ترجمة ← كتابة ← تصدير، وبوت Discord فوقه. **اللي لسه ما اتجربش على صور حقيقية:** الـ OCR (manga-ocr)، والترجمة بمفتاح حقيقي، والكتابة على صفحات حقيقية. دول بيتجربوا على Colab بنوتبوك الفصل.

- التصميم: [docs/DESIGN.md](docs/DESIGN.md). مراجع المشاريع الستة وباكلوج الأفكار: [docs/REFERENCES.md](docs/REFERENCES.md).
- الاستخدام خاص ومش تجاري ومش بيتوزّع، فالتراخيص (GPL وCC-NC لبعض المكونات) مسجّلة كمصدر في `NOTICE.md`؛ لو المشروع اتشارك لازم تتراجع.

## ترجمة فصل (سطر الأوامر)

```bash
pip install -e ".[gpu,ocr]"
python -m mangatr.gpu_check --models-dir models      # ينزّل الموديلات ويفحصها
export GEMINI_API_KEY=...                            # أو OPENAI_API_KEY / ANTHROPIC_API_KEY
python -m mangatr.cli translate --images pages/ --to ar --provider gemini --out out/
```

`--to en` للإنجليزية، `--provider gemini,openai` سلسلة بدائل، `--glossary terms.json` قاموس، `--format zip|cbz|pdf|images`. بيطلّع `out/report.json` (لكل صفحة: نصوص، مترجم، أخطاء، أزمنة). الكتابة بتحتاج Pillow بدعم Raqm (`apt-get install libraqm0`) وخط عربي (Noto Sans Arabic أو Cairo في `fonts/`، أو `MANGATR_FONT`).

## بوت Discord

```bash
pip install -e ".[gpu,ocr,bot]"
export DISCORD_TOKEN=... GEMINI_API_KEY=...
python -m bot.app
```

أمر `/translate` بيستقبل صور أو zip/cbz. حدود لكل مستخدم (فاصل بين المهام، حصة صفحات يومية، مهمة واحدة في نفس الوقت)، فحص الرفع (zip bomb، أبعاد الصور، مسارات)، طابور بمهلة، وتسليم بتقسيم لأجزاء تحت حد Discord.

## Colab

- `notebooks/colab_translate_chapter.ipynb`: الفصل كامل.
- `notebooks/colab_compare_clean.ipynb`: أوراق مقارنة التنظيف.
- `notebooks/colab_gpu_check.ipynb`: فحص الـ GPU والموديلات.

## فحص الـ GPU على Colab

افتح `notebooks/colab_gpu_check.ipynb` في Colab بـ runtime من نوع T4. بينزّل الموديلات، ويقيس زمنها على الـ GPU والـ CPU، ويطلّع تقرير تنسخه. الشرح الكامل جوه الـ notebook.

## تشغيل الاختبارات

```bash
pip install -e ".[dev]"
ruff check . && pytest
```

الاختبارات اللي محتاجة GPU عليها علامة `gpu` وبتتخطى افتراضياً.
