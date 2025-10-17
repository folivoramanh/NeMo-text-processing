# Vietnamese Audio-Based Text Normalization Guide

Hướng dẫn sử dụng Audio-based Text Normalization cho tiếng Việt.

## Mục lục
- [Non-Deterministic Mode (Không có ASR)](#non-deterministic-mode-không-có-asr)
- [Audio-based Mode (Có ASR Prediction)](#audio-based-mode-có-asr-prediction)
- [Batch Processing](#batch-processing)
- [Whitelist Custom](#whitelist-custom)
- [Advanced Options](#advanced-options)

---

## Non-Deterministic Mode (Không có ASR)

### 1. Single Sentence - Python API

Trả về **nhiều options** normalized cho 1 câu input.

```python
from nemo_text_processing.text_normalization.normalize_with_audio import NormalizerWithAudio

# Khởi tạo normalizer
normalizer = NormalizerWithAudio(
    lang='vi',
    input_case='cased',
    cache_dir=None,          # Set path để cache .far files
    overwrite_cache=True     # Set False để reuse cache
)

# Input text
text = "Hôm nay là 15/10/2025, tôi có 100đ"

# Normalize - trả về set of options
options = normalizer.normalize_non_deterministic(
    text=text,
    n_tagged=10,                    # Số lượng options tối đa
    punct_post_process=True,        # Post-process punctuation
    verbose=False
)

# Output
print(f"Input: {text}")
print(f"Got {len(options)} options:")
for i, option in enumerate(sorted(options), 1):
    print(f"{i}. {option}")
```

**Output:**
```
Input: Hôm nay là 15/10/2025, tôi có 100đ
Got 8 options:
1. Hôm nay là ngày mười lăm tháng mười năm hai nghìn hai mươi lăm, tôi có một trăm đồng
2. Hôm nay là ngày mười lăm tháng mười năm hai không hai năm, tôi có một trăm đồng
3. Hôm nay là mùng mười lăm tháng mười năm hai nghìn hai mươi lăm, tôi có một trăm đồng
...
```

### 2. Single Sentence - Command Line

```bash
cd nemo_text_processing/text_normalization

python normalize_with_audio.py \
    --text "Hôm nay là 15/10/2025" \
    --language vi \
    --input_case cased \
    --n_tagged 10 \
    --verbose
```

**Output:** Danh sách các options normalized

---

## Audio-based Mode (Có ASR Prediction)

Sử dụng ASR prediction để chọn option tốt nhất dựa trên **CER** (Character Error Rate).

### 1. Single Sentence with ASR Prediction

```python
from nemo_text_processing.text_normalization.normalize_with_audio import NormalizerWithAudio

normalizer = NormalizerWithAudio(
    lang='vi',
    input_case='cased'
)

# Input text và ASR prediction
text = "10:30 ngày 15/10"
asr_pred = "mười giờ ba mươi phút ngày mười lăm tháng mười"

# Normalize với ASR guidance
result = normalizer.normalize(
    text=text,
    n_tagged=20,
    pred_text=asr_pred,           # ASR prediction
    cer_threshold=100,            # CER threshold (%), -1 to disable
    punct_post_process=True,
    verbose=True
)

print(f"Input:  {text}")
print(f"ASR:    {asr_pred}")
print(f"Result: {result}")
```

**Output:**
```
Input:  10:30 ngày 15/10
ASR:    mười giờ ba mươi phút ngày mười lăm tháng mười
Result: mười giờ ba mươi phút ngày mười lăm tháng mười
```

### 2. Test với nhiều cases

```python
test_cases = [
    {
        'text': '10:30',
        'asr': 'mười giờ ba mươi phút'
    },
    {
        'text': '100đ',
        'asr': 'một trăm đồng'
    },
    {
        'text': 'covid-19',
        'asr': 'covid mười chín'
    }
]

normalizer = NormalizerWithAudio(lang='vi', input_case='cased')

for case in test_cases:
    result = normalizer.normalize(
        text=case['text'],
        n_tagged=10,
        pred_text=case['asr'],
        punct_post_process=True
    )
    print(f"{case['text']:15s} -> {result}")
```

---

## Batch Processing

### 1. Process List of Sentences

```python
from nemo_text_processing.text_normalization.normalize_with_audio import NormalizerWithAudio

normalizer = NormalizerWithAudio(lang='vi', input_case='cased')

sentences = [
    "Hôm nay là 15/10/2025",
    "Giá 100đ",
    "10:30 sáng",
    "covid-19 pandemic"
]

# Non-deterministic: trả về nhiều options
for text in sentences:
    options = normalizer.normalize_non_deterministic(
        text=text,
        n_tagged=5,
        punct_post_process=True
    )
    print(f"\nInput: {text}")
    print(f"Options: {len(options)}")
    for i, opt in enumerate(sorted(options)[:3], 1):
        print(f"  {i}. {opt}")
```

### 2. Process Manifest File (JSON)

Manifest format (`.json`):
```json
{"audio_filepath": "audio1.wav", "text": "10:30 ngày 15/10", "pred_text": "mười giờ ba mươi phút ngày mười lăm tháng mười", "duration": 3.5}
{"audio_filepath": "audio2.wav", "text": "100đ", "pred_text": "một trăm đồng", "duration": 2.1}
```

#### Command Line:

```bash
python normalize_with_audio.py \
    --manifest input_manifest.json \
    --output_filename output_manifest.json \
    --language vi \
    --input_case cased \
    --n_tagged 20 \
    --batch_size 100 \
    --n_jobs 4 \
    --manifest_text_field text \
    --manifest_asr_pred_field pred_text \
    --cer_threshold 100
```

**Parameters:**
- `--manifest`: Input manifest file path
- `--output_filename`: Output manifest file path
- `--n_tagged`: Number of options to generate
- `--batch_size`: Batch size for parallel processing
- `--n_jobs`: Number of parallel workers
- `--manifest_text_field`: Field name for text (default: "text")
- `--manifest_asr_pred_field`: Field name for ASR prediction (default: "pred_text")
- `--cer_threshold`: CER threshold in % (use -1 to disable)

#### Python API:

```python
from nemo_text_processing.text_normalization.normalize_with_audio import NormalizerWithAudio

normalizer = NormalizerWithAudio(lang='vi', input_case='cased')

normalizer.normalize_manifest(
    manifest='input_manifest.json',
    output_filename='output_manifest.json',
    n_tagged=20,
    batch_size=100,
    n_jobs=4,
    text_field='text',
    asr_pred_field='pred_text',
    cer_threshold=100,
    punct_post_process=True
)
```

**Output manifest:**
```json
{"audio_filepath": "audio1.wav", "text": "10:30 ngày 15/10", "pred_text": "mười giờ ba mươi phút ngày mười lăm tháng mười", "normalized": "mười giờ ba mươi phút ngày mười lăm tháng mười", "duration": 3.5}
{"audio_filepath": "audio2.wav", "text": "100đ", "pred_text": "một trăm đồng", "normalized": "một trăm đồng", "duration": 2.1}
```

---

## Whitelist Custom

Whitelist cho phép bạn thêm các từ hoặc cụm từ có cách đọc đặc biệt.

### 1. Tạo Whitelist File

Tạo file TSV: `my_whitelist.tsv`

Format: `written_form<tab>spoken_form`

```tsv
NeMo	en i em ô
NVIDIA	en ví đi a
COVID-19	cô vít mười chín
Dr.	bác sĩ
Mr.	ông
Ms.	bà
km/h	ki lô mét trên giờ
m²	mét vuông
```

**Lưu ý:**
- Dùng tab (`\t`) để phân cách, KHÔNG dùng spaces
- Không có header row
- Case-sensitive

### 2. Sử dụng Whitelist

#### Python API:

```python
from nemo_text_processing.text_normalization.normalize_with_audio import NormalizerWithAudio

normalizer = NormalizerWithAudio(
    lang='vi',
    input_case='cased',
    whitelist='/path/to/my_whitelist.tsv'  # Absolute path
)

# Test
text = ""
result = normalizer.normalize(text, n_tagged=10, punct_post_process=True)
print(result)
```

#### Command Line:

```bash
python normalize_with_audio.py \
    --text "NVIDIA phát triển NeMo" \
    --language vi \
    --whitelist /path/to/my_whitelist.tsv \
    --input_case cased
```

### 3. Whitelist có sẵn

Default whitelist tại: `nemo_text_processing/text_normalization/vi/data/whitelist/`

Các files:
- `default.tsv`: Common words
- `abbreviation.tsv`: Abbreviations
- Có thể thêm entries mới vào đây

---

## Advanced Options

### 1. Cache Management

Cache `.far` files để tăng tốc độ load:

```python
normalizer = NormalizerWithAudio(
    lang='vi',
    input_case='cased',
    cache_dir='./cache',        # Cache directory
    overwrite_cache=False       # False = reuse cache
)
```

**Lợi ích:**
- Lần đầu: ~10-15s to build grammars
- Lần sau: ~1-2s to load from cache

### 2. Number of Options Control

```python
# Nhiều options (chậm hơn nhưng có nhiều choices)
options = normalizer.normalize_non_deterministic(
    text=text,
    n_tagged=50  # Generate up to 50 options
)

# Ít options (nhanh hơn)
options = normalizer.normalize_non_deterministic(
    text=text,
    n_tagged=5   # Only 5 options
)
```

### 3. Punctuation Post-Processing

```python
# Với post-processing (recommended)
result = normalizer.normalize(
    text=text,
    punct_post_process=True  # Remove non-ending punctuation, clean spaces
)

# Không post-processing
result = normalizer.normalize(
    text=text,
    punct_post_process=False  # Keep all punctuation as-is
)
```

**Post-processing rules:**
- Chỉ giữ: `,` `.` `?` `!`
- Loại bỏ: `:` `;` `(` `)` `+` `-` `=` `/` etc.
- Collapse multiple spaces
- Strip leading/trailing spaces

### 4. CER Threshold

```python
# Strict: chỉ chấp nhận perfect match
result = normalizer.normalize(
    text=text,
    pred_text=asr_pred,
    cer_threshold=10  # CER <= 10%
)

# Relaxed: accept imperfect matches
result = normalizer.normalize(
    text=text,
    pred_text=asr_pred,
    cer_threshold=100  # Accept any match
)

# Disabled: không dùng CER filtering
result = normalizer.normalize(
    text=text,
    pred_text=asr_pred,
    cer_threshold=-1  # Disabled
)
```

---

## Examples

### Example 1: Basic Usage

```python
from nemo_text_processing.text_normalization.normalize_with_audio import NormalizerWithAudio

# Setup
normalizer = NormalizerWithAudio(lang='vi', input_case='cased')

# Test
text = "Hôm nay 15/10 tôi có 100đ"
options = normalizer.normalize_non_deterministic(text, n_tagged=5)

for opt in options:
    print(opt)
```

### Example 2: With ASR

```python
from nemo_text_processing.text_normalization.normalize_with_audio import NormalizerWithAudio

normalizer = NormalizerWithAudio(lang='vi', input_case='cased')

text = "10:30"
asr = "mười giờ ba mươi phút"

result = normalizer.normalize(
    text=text,
    n_tagged=10,
    pred_text=asr,
    punct_post_process=True
)

print(f"{text} -> {result}")
```

### Example 3: Batch with Custom Whitelist

```python
from nemo_text_processing.text_normalization.normalize_with_audio import NormalizerWithAudio

normalizer = NormalizerWithAudio(
    lang='vi',
    input_case='cased',
    whitelist='./my_whitelist.tsv'
)

sentences = ["NVIDIA NeMo", "COVID-19", "Dr. Nguyễn"]

for text in sentences:
    result = normalizer.normalize(text, n_tagged=5, punct_post_process=True)
    print(f"{text:20s} -> {result}")
```

### Example 4: Manifest Processing

```bash
# Create test manifest
cat > test_manifest.json << EOF
{"text": "10:30", "pred_text": "mười giờ ba mươi phút"}
{"text": "100đ", "pred_text": "một trăm đồng"}
EOF

# Process
python normalize_with_audio.py \
    --manifest test_manifest.json \
    --output_filename output.json \
    --language vi \
    --n_tagged 10 \
    --batch_size 10
```

---

## Troubleshooting

### 1. Lỗi Import

```
ModuleNotFoundError: No module named 'pynini'
```

**Fix:** Install pynini
```bash
pip install pynini
```

### 2. Cache Issues

Nếu gặp lỗi khi load cache:

```python
normalizer = NormalizerWithAudio(
    lang='vi',
    input_case='cased',
    cache_dir=None,         # Disable cache
    overwrite_cache=True
)
```

Hoặc xóa cache cũ:
```bash
rm -rf ./cache/*.far
```

### 3. Out of Memory

Nếu xử lý file lớn bị OOM:

```python
normalizer.normalize_manifest(
    manifest='large_file.json',
    batch_size=10,      # Giảm batch size
    n_jobs=1,          # Giảm parallel workers
    n_tagged=5         # Giảm số options
)
```

---

## Performance Tips

1. **Use cache:** Set `cache_dir` và `overwrite_cache=False`
2. **Reduce n_tagged:** Dùng 5-10 thay vì 50
3. **Batch processing:** Dùng `n_jobs` > 1 cho manifest files
4. **Pre-filter:** Loại bỏ các câu không cần normalize trước
