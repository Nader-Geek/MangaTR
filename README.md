# MangaTR

نواة لترجمة المانجا والمانهوا والكوميكس لعربي وإنجليزي: كشف، OCR، ترجمة، تنظيف، وكتابة النص داخل الحبابة. البوت طبقة رفيعة فوقها.

**الحالة:** تصميم وهيكل. المراحل لسه عقود موثقة (docstrings) مش تنفيذ.

- التصميم الكامل: [docs/DESIGN.md](docs/DESIGN.md)
- الكود الحالي: `mangatr/scene.py` (Page وBlock)، `mangatr/languages.py`، `mangatr/pipeline.py`

## تشغيل الاختبارات

```bash
pip install -e ".[dev]"
ruff check . && pytest
```

الاختبارات اللي محتاجة GPU عليها علامة `gpu` وبتتخطى افتراضياً.
