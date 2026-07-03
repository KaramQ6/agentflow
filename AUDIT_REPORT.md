# 🔍 Audit Report — `agentflowkit` v0.4.0

> **الوضع:** Pass 1 (تقرير فقط — ما تم تعديل أي سطر).
> **التاريخ:** 2026-07-03
> **الفرع:** `main` @ `03a370d`
> **المدقّق:** Claude (Opus 4.8) — Two-Pass Security & Health Audit

---

## 0. الملخّص التنفيذي

| البند | النتيجة |
|-------|---------|
| **Stack** | Python ≥3.10 · packaging: `hatchling` · quality: `ruff` + `mypy --strict` |
| **Core deps** | `openai>=1.0.0`, `pydantic>=2.0.0` (باقي backends: docker/redis/chromadb/aiomqtt = optional extras) |
| **Test baseline** | ✅ `202 passed, 11 skipped` بـ ~22s — كلها خضراء (التقرير الأصلي حكى 99؛ الواقع أكبر وأصحّ) |
| **Ruff (src)** | ❌ 2 errors بـ `swarm.py` |
| **Mypy (src)** | ❌ 1 error بـ `swarm.py` |
| **pip-audit** | ✅ ولا CVE بأي dependency حقيقي لـ agentflow |
| **Secrets / eval / exec** | ✅ نظيف |

**أهم 5 أولويات:**
1. **D1** — تصليح lint/type بـ `swarm.py` (بوابة الجودة حمرا حالياً، إصلاح آمن 100%).
2. **A1** — لفّ استدعاءات ChromaDB المتزامنة بـ `asyncio.to_thread` (blocking داخل الـ event loop).
3. **S1** — قرار حول الـ sandbox fallback الصامت (يغيّر سلوك خارجي).
4. **H1** — سقف عدد الجلسات بـ `InMemoryContext` (نمو ذاكرة غير محدود).
5. **H2/A2/A3/H3/H4/H5** — تحصينات صغيرة آمنة.

**تنبيه على البيئة:** الـ `venv` المفحوص بيئة مشتركة (torch+cuda، transformers، streamlit، rembg، langchain…) مش معزولة لـ agentflow — بيأثّر على قراءة الـ CVEs (شوف القسم 3).

---

## 1. 🔒 Security & AI Risks (أولوية قصوى)

| # | Sev | Issue | File+Line | Impact | Suggested Fix |
|---|-----|-------|-----------|--------|---------------|
| S1 | 🟠 Med-High | **Fallback صامت لـ `SubprocessSandbox`** — لما Docker مش متوفر، `create_sandbox(prefer_docker=True)` بيرجع `SubprocessSandbox` اللي بشغّل كود الـ LLM مباشرة على الهوست (`sys.executable -c code`) بدون عزل. الاسم `sandboxed_tool` بيوحي بالأمان، والـ fallback بصير بصمت وقت الإعداد (في warning بس وقت التنفيذ). | `sandbox.py:389-415` | كود مولّد من LLM (يحتمل injection) ينفّذ على جهاز المستخدم بكامل صلاحياته. | opt-in صريح `allow_insecure_fallback=False`؛ إذا Docker مفقود وما في سماح → `raise`. **⚠️ يغيّر السلوك الخارجي — بدّه قرار.** |
| S2 | 🟡 Low | **Heredoc breakout بالـ C++** — الكود بينحطّ داخل `sh -c` heredoc بفاصل ثابت `AGENTFLOW_EOF`؛ كود فيه هاض السطر بيكسر الـ heredoc. | `sandbox.py:64-79` | منخفض: الكسر بيضل جوّا نفس الـ container المعزول (network none, cap_drop ALL, read_only). مش هروب من الحدود. | تمرير الكود عبر stdin أو file mount بدل heredoc. |
| S3 | 🟡 Low-Med | **Trust elevation بالـ prompt** — مخرجات الوكلاء/الأدوات السابقة بتنحقن بالـ **system** prompt (مقصوصة 300 حرف)؛ محتوى غير موثوق (نتيجة أداة web/MQTT) بيترفّع لمستوى system. | `agent.py:106-115` | منخفض-متوسط: بيضخّم prompt-injection؛ متأصّل بأطر الوكلاء. | نقل ذاكرة الجلسة لرسالة `user`/`assistant` مش `system`. informational. |
| S4 | 🟢 OK | **No hardcoded secrets / no eval / no exec** — بس placeholders بالـ tests/docs (`"test-key"`, `"sk-or-..."`). `subprocess` محصور بـ `sandbox.py` بالتصميم. | — | نظيف. | لا شيء. |
| S5 | 🟢 OK | **No path traversal بالـ loggers** — كتابة على stdout بس (`StreamHandler`)، ما في file paths. | `logging.py` | نظيف. | لا شيء. |

---

## 2. ⚙️ Async & Concurrency Health

| # | Sev | Issue | File+Line | Impact | Suggested Fix |
|---|-----|-------|-----------|--------|---------------|
| A1 | 🟠 Med | **استدعاءات ChromaDB متزامنة (blocking) جوّا `async def`** — كل دوال `VectorContext` معرّفة `async` بس بتنادي `._collection.upsert/.query/.get/.delete` المتزامنة مباشرة. مع `PersistentClient` (disk IO) أو حساب embeddings بيتجمّد الـ event loop. باقي الكود بيلفّ الـ blocking بـ `asyncio.to_thread` (DockerSandbox / tools.py) — هون غير متسق. | `memory.py:258-323` | تجميد الـ loop تحت الحمل. | `await asyncio.to_thread(self._collection.method, …)`. |
| A2 | 🟡 Low-Med | **RateLimiter ماسك الـ lock عبر `await sleep`** — `_wait_for_window` ماسك `self._lock` وهو نايم بالـ `asyncio.sleep`، فكل الكوروتينات الباقية بتتسكّر (تسلسل الإنتاجية)، والـ semaphore slot محجوز طول الانتظار. | `rate_limiter.py:47-59` | خنق الـ throughput، مش deadlock. | احسب مدة النوم تحت الـ lock، حرّر الـ lock، بعدها نام. |
| A3 | 🟡 Low | **تسرّب Semaphore عند الإلغاء** — `acquire()` بياخد الـ semaphore بعدها `_wait_for_window`؛ إلغاء أثناء النوم ما بيحرّر الـ slot. وبـ `llm.py` الـ `acquire()` برّا الـ try/finally (121 مقابل try 123). | `rate_limiter.py:36-39`, `llm.py:120-121` | حالة حافة ضيّقة (cancellation). | خلّي `acquire` يحرّر الـ semaphore إذا `_wait_for_window` رمى؛ أو انقل `acquire` جوّا try. |
| A4 | 🟢 OK | **`asyncio.gather`** — كلها `return_exceptions=True` مع معالجة، أو await متسلسل للـ tasks. ما في gather exceptions مهملة. | `pipeline.py:248,440,558`; `agent.py:283-286`; `swarm.py:162-163` | نظيف. | لا شيء. |
| A5 | 🟢 OK | **`InMemoryContext` locking** — قفل واحد متّسق، ما في await بين check/act يسبّب race، ولا nesting. | `memory.py:48-104` | نظيف. | لا شيء. |

---

## 3. 📦 Dependencies

### النُّسخ (Installed vs Latest)

| Package | Installed | Latest | نوع | ملاحظة |
|---------|-----------|--------|-----|--------|
| `openai` | 1.99.9 | **2.44.0** | **MAJOR** | الكود بستورد `APIError, RateLimitError, AsyncOpenAI` + `openai.types.chat` — 2.x محتمل يكسر. **توصية بس.** |
| `pydantic-core` | 2.46.4 | 2.47.0 | minor | آمن (patch/minor). |
| `anyio` | 4.14.0 | 4.14.1 | patch | آمن. |
| `ruff` (dev) | 0.1.14 | 0.15.20 | — | ⚠️ مثبّت **أقل** من floor المعلن `>=0.4` بالـ pyproject. |
| `pytest-asyncio` (dev) | 0.23.3 | 1.4.0 | major | dev بس. |
| `pytest-cov` (dev) | 7.0.0 | 7.1.0 | minor | dev بس. |

### CVEs (pip-audit)

✅ **ولا CVE بأي من dependencies الحقيقية لـ agentflow** (`openai`, `pydantic`).

كل الثغرات المكتشفة بحزم **مش تابعة** لـ agentflow، موجودة بالـ venv المشترك:

```
setuptools  65.5.0  CVE-2024-6345 (RCE), PYSEC-2025-49 (path traversal), PYSEC-2022-43012
starlette   0.37.2  عدة CVEs (2024-2026)
tornado     6.5.2   عدة CVEs (2026)
werkzeug    3.1.3   CVE-2025-66221, CVE-2026-21860, CVE-2026-27199
transformers 4.57.6 PYSEC-2025-217, CVE-2026-1839, CVE-2026-4372
streamlit   1.53.1  CVE-2026-33682
pyarrow / rembg / wheel  ثغرات إضافية
```

- كل هدول **مش** dependencies لـ agentflow.
- تنبيه: الـ extra الاختياري `chromadb` بيجرّ transitively `fastapi/starlette/uvicorn`.
- **توصية:** شغّل الـ audit بـ venv معزول فيه agentflow + extras بس عشان قراءة دقيقة.

---

## 4. 🔁 Duplicated Logic & Refactors

| # | Sev | Issue | File+Line | Impact | Suggested Fix |
|---|-----|-------|-----------|--------|---------------|
| D1 | 🟠 Med | **Lint/type فاشلة بـ `swarm.py`** — `ruff`: B007 (`iteration` unused @104)، F841 (`arguments` unused @148). `mypy`: no-untyped-def @184 (`_make_delegate_fn`). بوابة الجودة **حالياً حمرا**. | `swarm.py:104,148,184` | إصلاحات تافهة وآمنة. | `for _ in range(...)`، احذف `arguments` المكرّر، ضيف return type annotation. **✅ أسهل مكسب.** |
| D2 | 🟠 Med | **بلوك HITL pause/persist مكرّر 3×** حرفياً. | `pipeline.py:250-296, 442-485, 560-595` | صيانة مؤلمة. | استخرج helper `_persist_pause_state(...)`. **⚠️ يلمس تدفق pipeline — بدّه قرار.** |
| D3 | 🟡 Low | **حلقة ReAct مكرّرة** بين الوكيل والـ supervisor. | `agent.py:195-307` vs `swarm.py:104-182` | تكرار كبير بس **core logic**. | **توصية بس — ممنوع لمسها حسب القيود.** |
| D4 | 🟡 Low | **guard استيراد redis مكرّر** بنمطين مختلفين. | `cache.py:80-84` vs `memory.py:108-135` | بسيط. | توحيد النمط. |

---

## 5. 🩺 General Health

| # | Sev | Issue | File+Line | Impact | Suggested Fix |
|---|-----|-------|-----------|--------|---------------|
| H1 | 🟠 Med | **نمو `InMemoryContext` غير محدود بالجلسات** — `max_entries` بحدّ الإدخالات **لكل جلسة**، بس عدد الجلسات (`self._store` keys) غير محدود. الجلسات المنتهية بتتنظّف بس لما تتقرا هي بالذات عبر `load_context`. workload بيولّد session_id فريد لكل طلب وما بيرجع يقراه = تسرّب ذاكرة. | `memory.py:60, 78-91` | نمو ذاكرة غير محدود. | سقف max-sessions/sweep دوري، أو توثيق إن الـ caller لازم `clear()`. |
| H2 | 🟡 Low | **`getattr` برّا الـ try بـ tools** — `kwargs = {k: getattr(validated, k) for k in arguments}` قبل الـ try؛ إذا الـ LLM بعت مفتاح زيادة (pydantic بتجاهله)، `getattr` بترمي `AttributeError` غير ملفوفة بـ `ToolError`. | `tools.py:94` | منخفض. | كرّر على حقول الموديل مش مفاتيح `arguments`، أو انقل جوّا try. |
| H3 | 🟡 Low | **ما في لفّ لأخطاء Redis** — استثناءات redis الخام بتنتشر بدل framework error (غير متّسق مع لفّ `LLMError`/`ToolError`). | `memory.py:186-201`, `cache.py:119-131` | منخفض. | لفّ استدعاءات redis. |
| H4 | 🟡 Low | **`InMemoryCache` موصوف "Thread-safe" بدون قفل** — dict عادي بلا lock (بعكس `InMemoryContext`). آمن ضمن الـ event loop بس، مش thread-safe فعلياً؛ وكمان FIFO مش LRU رغم التسمية. | `cache.py:36-70` | docstring مضلّل. | صحّح الـ docstring أو ضيف قفل. |
| H5 | 🟡 Low | **DockerSandbox `read_only=True` مع كتابة `/tmp`** — مسار الـ C++ بيكتب `/tmp/code.cpp` بس الـ container read-only بلا tmpfs → تنفيذ C++ بالـ Docker بيفشل runtime. | `sandbox.py:201, 73-76` | خلل وظيفي (مش أمني). | ضيف `tmpfs={"/tmp": ""}` أو `read_only=False`. |

---

## 6. القرارات المعلّقة (بدّها موافقتك)

| القرار | الوصف | الخيار |
|--------|-------|--------|
| **S1** | الـ sandbox fallback الصامت | نضيف `allow_insecure_fallback` ونمنع الـ fallback الصامت؟ (يغيّر سلوك) |
| **D2** | HITL persist helper | نستخرج helper (يلمس pipeline flow)؟ |
| **D3** | ReAct dedup | توصية فقط — ممنوع اللمس حسب القيود |
| **openai 2.x** | ترقية major | نتركها توصية أم نجرّبها بفرع منفصل؟ |

---

## 7. خطة Pass 2 المقترحة (بعد الموافقة)

**المجموعة الآمنة (ما بتغيّر سلوك — بتنفّذ مباشرة بعد الموافقة):**
`D1` (lint/type) → `A1` (to_thread) → `H1` (session cap) → `H2` (try scope) → `A2`+`A3` (rate limiter) → `H3` (redis wrap) → `H4` (docstring) → `H5` (tmpfs) → dependency patches (anyio, pydantic-core).

**بعد كل مجموعة:** `pytest` + `ruff` + `mypy` — والـ 202 لازم تضل خضرا.

**تُترك كتوصيات فقط:** `S1`, `D2`, `D3`, `openai 2.x`.

---

*انتهى Pass 1 — ما تم تعديل أي سطر بالكود.*
