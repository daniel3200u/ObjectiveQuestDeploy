import re
import math
from datetime import datetime
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

stop_factory = StopWordRemoverFactory()
stop_words = set(stop_factory.get_stop_words())

stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in stop_words]
    words = [stemmer.stem(w) for w in words]
    return " ".join(words)

def parse_hukuman(text: str):
    text = re.sub(r"\s+", " ", text).strip()
    keywords = [
        "Menjatuhkan pidana",
        "Menghukum terdakwa",
        "atuhkan pidana",
        "Menjatukan pidana",
        "Menjatuhkan Hukuman",
        "Dijatuhkan pidana",
        "dengan pidana penjara"
    ]
    candidates = []
    for kw in keywords:
        for m in re.finditer(kw, text, re.IGNORECASE):
            candidates.append((m.start(), kw))
    candidates.sort(key=lambda x: x[0])
    hukuman_text = ""
    for start_idx, kw in candidates:
        substring = text[start_idx:start_idx+300].strip()
        if "penjara" in substring.lower() or "(" in substring:
            hukuman_text = substring
            break
    if not hukuman_text:
        return None
    tahun, bulan = 0, 0
    match_tahun = re.search(r"(\d+)(?:\s*\([^)]+\))?\s*tahun", hukuman_text, re.IGNORECASE)
    if match_tahun:
        try:
            tahun = int(match_tahun.group(1))
        except:
            tahun = 0
    match_bulan = re.search(r"(\d+)(?:\s*\([^)]+\))?\s*bulan", hukuman_text, re.IGNORECASE)
    if match_bulan:
        try:
            bulan = int(match_bulan.group(1))
        except:
            bulan = 0
    total_bulan = tahun * 12 + bulan
    return total_bulan if total_bulan > 0 else None

def extract_detention_duration(text: str):
    text = re.sub(r"\s+", " ", text).strip()
    matches = list(re.finditer(r"(oleh.*?)(?=Terdakwa|Perempuan|Halaman|Anak|$)", text, re.DOTALL | re.IGNORECASE))
    if not matches:
        return 0 if re.search(r"(tidak\s+ditahan|ditahan\s+dalam\s+perkara\s+lain)", text, re.IGNORECASE) else None
    def extract_dates(block):
        dates = re.findall(r"(\d{1,2}\s+[A-Za-z]+(?:\s+\d{4})|\d{1,2}-\d{1,2}-\d{4})", block)
        month_map = {
            "Januari": "January", "Februari": "February", "Maret": "March", "April": "April",
            "Mei": "May", "Juni": "June", "Juli": "July", "Agustus": "August", "September": "September",
            "Oktober": "October", "November": "November", "Desember": "December"
        }
        parsed_dates = []
        for d in dates:
            d_norm = d
            for indo, eng in month_map.items():
                d_norm = d_norm.replace(indo, eng)
            for fmt in ["%d %B %Y", "%d-%m-%Y"]:
                try:
                    parsed = datetime.strptime(d_norm.strip(), fmt)
                    parsed_dates.append(parsed)
                    break
                except ValueError:
                    continue
        return parsed_dates
    parsed_dates = []
    for m in matches:
        block = m.group(1)
        if re.search(r"penangkapan", block, re.IGNORECASE):
            continue
        parsed_dates = extract_dates(block)
        if len(parsed_dates) >= 2:
            break
    if len(parsed_dates) < 2:
        return 0 if re.search(r"(tidak\s+ditahan|ditahan\s+dalam\s+perkara\s+lain)", text, re.IGNORECASE) else None
    start_date, end_date = parsed_dates[0], parsed_dates[-1]
    duration_days = (end_date - start_date).days
    duration_months = math.ceil(duration_days / 30)
    return duration_months if duration_months > 0 else None

def extract_page_count(text: str):
    matches = re.findall(r"^Halaman\s+(\d+)", text, re.IGNORECASE | re.MULTILINE)
    if not matches:
        return None
    numbers = [int(m) for m in matches]
    return max(numbers) if numbers else None

def extract_features(text: str) -> dict:
    features = {
        "num_unique_pasal": None,
        "hukuman_bulan": None,
        "num_pages": None,
        "len_kronologi": None,
        "pengurangan_tahanan": 0
    }
    pasal_nums = re.findall(r"\bpasal\s+(\d+)", text)
    if pasal_nums:
        features["num_unique_pasal"] = len(set(pasal_nums))
    hukuman_bulan = parse_hukuman(text)
    if hukuman_bulan is not None:
        features["hukuman_bulan"] = hukuman_bulan
    num_pages = extract_page_count(text)
    if num_pages is not None:
        features["num_pages"] = num_pages
    menimbang_match = re.search(r"menimbang[,:]?\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if menimbang_match:
        kronologi_text = menimbang_match.group(1)
        words = re.findall(r"\w+", kronologi_text)
        features["len_kronologi"] = len(words)
    durasi_tahanan = extract_detention_duration(text)
    if durasi_tahanan is not None:
        features["pengurangan_tahanan"] = durasi_tahanan
    return features
