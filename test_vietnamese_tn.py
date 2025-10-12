from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo_text_processing.text_normalization.normalize_with_audio import NormalizerWithAudio

CACHE_DIR = None

# Format: (input_text, expected_output, asr_prediction)

TEST_CASES = [
    # Ví dụ về cardinal
    # ("100", "một trăm", "một trăm"),
    # ("1000", "một nghìn", "một nghìn"),
    
    # # Ví dụ về money
    # ("100 đồng", "một trăm đồng", "một trăm đồng"),
    # ("100 USD", "một trăm đô la mỹ", "một trăm đô la mỹ"),
    
    # # Ví dụ về date
    # ("01/01/2024", "ngày một tháng một năm hai nghìn không trăm hai mươi bốn", None),
    
    # # Ví dụ về time
    # ("10:30", "mười giờ ba mươi phút", "mười giờ ba mươi phút"),
    
    # # Ví dụ về ordinal
    # ("thứ 1", "thứ nhất", "thứ nhất"),
    
    # Thêm test cases của bạn vào đây...
    # ("input", "expected", "asr_pred_or_None"),
    
    ("Hôm nay là thứ 5, mùng 5 tháng 10 (5/10/2025), tôi có nhiều task cần làm, cụ thể là task chương IV", "Hôm nay là thứ năm, mùng năm tháng mười (năm tháng mười năm hai ngàn linh hai mươi lăm), tôi có nhiều task cần làm, cụ thể là task chương bốn", "Hôm nay là thứ năm, mùng năm tháng mười (năm tháng mười năm hai ngàn linh hai mươi lăm), tôi có nhiều task cần làm, cụ thể là task chương bốn" )
]

# Set TEST_MODE = "non_det_only" để dùng mode này
TEST_CASES_NON_DET_ONLY = [
    # Ví dụ:
    # "Hôm nay là thứ 5, mùng 5 tháng 10 (5/10/2025), tôi có nhiều task cần làm, cụ thể là task chương IV",
    # "101",
    "1101",
    "2024",
    "2004",
    # "1,12",
    "1,123",
    "thứ 1",
    # "thứ 2",
    # "thứ 3",
    "100 đồng",
    "10:30",
    # "01/01/2024",
    
    # Thêm input của bạn vào đây...
]

# Chọn test mode:
#   "full" = Test cả 3 modes (dùng TEST_CASES)
#   "non_det_only" = Chỉ test non-deterministic (dùng TEST_CASES_NON_DET_ONLY)
TEST_MODE = "non_det_only"

# Số lượng normalization options cho non-deterministic mode
N_TAGGED = 30

# In thông tin chi tiết khi test
VERBOSE = False


def init_normalizers():

    # Deterministic normalizer
    normalizer_det = Normalizer(
        input_case='cased',
        lang='vi',
        deterministic=True,
        cache_dir=CACHE_DIR,
        overwrite_cache=False,
        post_process=True,
    )
    
    # Audio-based normalizer (hỗ trợ cả non-deterministic và audio mode)
    normalizer_audio = NormalizerWithAudio(
        input_case='cased',
        lang='vi',
        cache_dir=CACHE_DIR,
        overwrite_cache=False,
    )
    
    return normalizer_det, normalizer_audio


def test_single_case(text, expected, asr_pred, normalizer_det, normalizer_audio):
    print("="*80)
    print(f"INPUT: {text}")
    if expected:
        print(f"EXPECTED: {expected}")
    if asr_pred:
        print(f"ASR PRED: {asr_pred}")
    print("-"*80)
    
    results = {}
    
    # Test mode 1: Deterministic
    try:
        result_det = normalizer_det.normalize(text, verbose=VERBOSE)
        results['deterministic'] = result_det
        status_det = "✓" if (not expected or result_det == expected) else "✗"
        print(f"{status_det} [DETERMINISTIC]: {result_det}")
    except Exception as e:
        print(f"✗ [DETERMINISTIC]: ERROR - {e}")
        results['deterministic'] = None
    
    # Test mode 2: Non-deterministic
    try:
        result_non_det = normalizer_audio.normalize_non_deterministic(
            text,
            n_tagged=N_TAGGED,
            verbose=VERBOSE,
        )
        results['non_deterministic'] = result_non_det
        status_non = "✓" if (not expected or expected in result_non_det) else "✗"
        print(f"{status_non} [NON-DETERMINISTIC]: {len(result_non_det)} options")
        for i, opt in enumerate(sorted(result_non_det), 1):
            marker = "→" if opt == expected else " "
            print(f"  {marker} {i}. {opt}")
    except Exception as e:
        print(f"✗ [NON-DETERMINISTIC]: ERROR - {e}")
        results['non_deterministic'] = None
    
    # Test mode 3: Audio-based (chỉ khi có ASR prediction)
    if asr_pred:
        try:
            result_audio = normalizer_audio.normalize(
                text,
                n_tagged=N_TAGGED,
                pred_text=asr_pred,
                verbose=VERBOSE,
            )
            results['audio_based'] = result_audio
            status_audio = "✓" if (not expected or result_audio == expected) else "✗"
            print(f"{status_audio} [AUDIO-BASED]: {result_audio}")
        except Exception as e:
            print(f"✗ [AUDIO-BASED]: ERROR - {e}")
            results['audio_based'] = None
    
    print()
    return results


def test_non_deterministic_only():
    """Test chỉ non-deterministic mode để xem có bao nhiêu options"""
    if not TEST_CASES_NON_DET_ONLY:
        print("⚠ Không có test case nào!")
        print("Vui lòng thêm test cases vào biến TEST_CASES_NON_DET_ONLY trong script")
        return
    
    print("\n" + "="*80)
    print(f"TEST NON-DETERMINISTIC MODE - Tổng cộng {len(TEST_CASES_NON_DET_ONLY)} test cases")
    print("="*80 + "\n")
    
    # Chỉ cần audio normalizer cho non-deterministic mode
    print("Đang khởi tạo normalizer...")
    normalizer_audio = NormalizerWithAudio(
        input_case='cased',
        lang='vi',
        cache_dir=CACHE_DIR,
        overwrite_cache=False,
    )
    print("✓ Khởi tạo xong!\n")
    
    # Test từng case
    for i, text in enumerate(TEST_CASES_NON_DET_ONLY, 1):
        print(f"\n[TEST {i}/{len(TEST_CASES_NON_DET_ONLY)}]")
        print("="*80)
        print(f"INPUT: {text}")
        print("-"*80)
        
        try:
            results = normalizer_audio.normalize_non_deterministic(
                text,
                n_tagged=N_TAGGED,
                verbose=VERBOSE,
            )
            
            print(f"✓ Tạo được {len(results)} options:\n")
            
            # Sort và in ra từng option
            sorted_results = sorted(results)
            for j, option in enumerate(sorted_results, 1):
                print(f"  {j:2d}. {option}")
            
            print()
            
        except Exception as e:
            print(f"✗ ERROR: {e}\n")
            if VERBOSE:
                import traceback
                traceback.print_exc()
    
    print("="*80)
    print("HOÀN THÀNH!")
    print("="*80)


def run_tests():
    """Chạy tất cả test cases"""
    if not TEST_CASES:
        print("⚠ Không có test case nào!")
        print("Vui lòng thêm test cases vào biến TEST_CASES trong script")
        return
    
    print("\n" + "="*80)
    print(f"BẮT ĐẦU TEST - Tổng cộng {len(TEST_CASES)} test cases")
    print("="*80 + "\n")
    
    # Khởi tạo normalizers
    normalizer_det, normalizer_audio = init_normalizers()
    
    # Statistics
    stats = {
        'total': len(TEST_CASES),
        'deterministic': {'pass': 0, 'fail': 0, 'error': 0},
        'non_deterministic': {'pass': 0, 'fail': 0, 'error': 0},
        'audio_based': {'pass': 0, 'fail': 0, 'error': 0},
    }
    
    # Chạy từng test case
    for i, (text, expected, asr_pred) in enumerate(TEST_CASES, 1):
        print(f"\n[TEST {i}/{len(TEST_CASES)}]")
        results = test_single_case(text, expected, asr_pred, normalizer_det, normalizer_audio)
        
        # Update statistics
        if expected:
            # Deterministic
            if results['deterministic'] is None:
                stats['deterministic']['error'] += 1
            elif results['deterministic'] == expected:
                stats['deterministic']['pass'] += 1
            else:
                stats['deterministic']['fail'] += 1
            
            # Non-deterministic
            if results['non_deterministic'] is None:
                stats['non_deterministic']['error'] += 1
            elif expected in results['non_deterministic']:
                stats['non_deterministic']['pass'] += 1
            else:
                stats['non_deterministic']['fail'] += 1
            
            # Audio-based
            if asr_pred:
                if results['audio_based'] is None:
                    stats['audio_based']['error'] += 1
                elif results['audio_based'] == expected:
                    stats['audio_based']['pass'] += 1
                else:
                    stats['audio_based']['fail'] += 1
    
    # In kết quả tổng hợp
    print("\n" + "="*80)
    print("KẾT QUẢ TỔNG HỢP")
    print("="*80)
    
    for mode in ['deterministic', 'non_deterministic', 'audio_based']:
        mode_stats = stats[mode]
        total_tested = mode_stats['pass'] + mode_stats['fail'] + mode_stats['error']
        
        if total_tested > 0:
            pass_rate = 100 * mode_stats['pass'] / total_tested if total_tested > 0 else 0
            print(f"\n{mode.upper().replace('_', ' ')}:")
            print(f"  ✓ Pass:  {mode_stats['pass']}/{total_tested} ({pass_rate:.1f}%)")
            print(f"  ✗ Fail:  {mode_stats['fail']}/{total_tested}")
            print(f"  ! Error: {mode_stats['error']}/{total_tested}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    if TEST_MODE == "non_det_only":
        test_non_deterministic_only()
    elif TEST_MODE == "full":
        run_tests()
    else:
        print(f"⚠ TEST_MODE không hợp lệ: {TEST_MODE}")
        print("Chọn 'full' hoặc 'non_det_only'")

