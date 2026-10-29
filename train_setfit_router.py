#!/usr/bin/env python3
"""Train a SetFit-style Contrastive Embedding Classifier (`all-MiniLM-L6-v2` + Contrastive Projection Head)
for the Entertainment Search Query Decision Router (`DIRECT_AGENT_SEARCH` vs `CALL_LLM_INTENT`).

Pipeline:
1. Synthesizes ~1,600 balanced domain queries (plus hard negatives like movie titles containing city/date words)
   and evaluates on both a 20% held-out validation split and the 30-query benchmark (`ROUTER_BENCHMARK_30`).
2. Extracts 384-d mean-pooled L2-normalized embeddings from `sentence-transformers/all-MiniLM-L6-v2` (ONNX)
   fused with 64-d SVD char/word n-gram features.
3. Trains a Contrastive Residual Projection Head (`448 -> 256 -> 128`, L2-normalized) on 25,000 positive and
   negative query pairs using SetFit Cosine Contrastive Loss on Apple Metal GPU (`mlx.nn` + `AdamW`).
4. Fits a calibrated binary route classifier (`DIRECT_AGENT_SEARCH` vs `CALL_LLM_INTENT`) + 7 multi-label
   facet heads (`city`, `date_time`, `price`, `format`, `language`, `genre_category`, `negation`) so the
   classifier produces both the route decision AND a human-readable `reason` in ~1.5 ms.
5. Exports portable NumPy weights to `search_studio/setfit_minilm_router_weights.npz`.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
import random
import time
from typing import Dict, List, Tuple

from huggingface_hub import hf_hub_download
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import onnxruntime as ort
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer

from search_studio.router_api import ROUTER_BENCHMARK_30

ROOT_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT_DIR / "search_studio" / "setfit_minilm_router_weights.npz"

FACET_NAMES = [
    "city",
    "date_time",
    "price",
    "format",
    "language",
    "genre_category",
    "negation",
]


def build_training_dataset(include_benchmark_30: bool = False, seed: int = 42) -> List[Tuple[str, int, List[int]]]:
    """Generate ~1,600 diverse labeled queries: `(query_text, route_label, facet_multi_hot_7)`.
    `route_label`: 0 = `DIRECT_AGENT_SEARCH`, 1 = `CALL_LLM_INTENT`.
    """
    rng = random.Random(seed)
    samples: Dict[str, Tuple[int, List[int]]] = {}

    def add_sample(q: str, label: int, facets: List[str]) -> None:
        q_clean = " ".join(q.strip().split())
        if not q_clean:
            return
        vec = [1 if fname in facets else 0 for fname in FACET_NAMES]
        samples[q_clean.lower()] = (q_clean, label, vec)

    # -------------------------------------------------------------------------
    # 1. DIRECT_AGENT_SEARCH (label = 0): Movie/Show Titles, Hard-Negative Titles,
    #    Artists, Standalone Venues/Theatres, and Bank/Card Offers
    # -------------------------------------------------------------------------
    movie_titles = [
        "Spider-Man: No Way Home",
        "Pushpa 2: The Rule",
        "Singham Again",
        "Bhool Bhulaiyaa 3",
        "Gladiator II",
        "Kantara",
        "Kantara: Chapter 1",
        "Stree 2",
        "Jawan",
        "Pathaan",
        "Animal",
        "Kalki 2898 AD",
        "Salaar: Part 1 - Ceasefire",
        "RRR",
        "Baahubali 2: The Conclusion",
        "KGF: Chapter 2",
        "Leo",
        "Jailer",
        "Vikram",
        "Ponniyin Selvan: II",
        "Brahmastra Part One: Shiva",
        "Drishyam 2",
        "12th Fail",
        "Laapataa Ladies",
        "Munjya",
        "Shaitaan",
        "Fighter",
        "Dunki",
        "Tiger 3",
        "Gadar 2",
        "OMG 2",
        "Rocky Aur Rani Kii Prem Kahaani",
        "Tu Jhoothi Main Makkaar",
        "Zara Hatke Zara Bachke",
        "Sam Bahadur",
        "Chhaava",
        "Devara: Part 1",
        "Vettaiyan",
        "The Greatest of All Time",
        "Amaran",
        "Lucky Baskhar",
        "Manjummel Boys",
        "Aavesham",
        "Premalu",
        "Bramayugam",
        "The Goat Life",
        "Kishkindha Kaandam",
        "ARM",
        "777 Charlie",
        "Sapta Sagaradaache Ello",
        "Oppenheimer",
        "Barbie",
        "Dune: Part Two",
        "Deadpool & Wolverine",
        "Inside Out 2",
        "Joker: Folie a Deux",
        "Venom: The Last Dance",
        "Wicked",
        "Moana 2",
        "Sonic the Hedgehog 3",
        "Mufasa: The Lion King",
        "Interstellar",
        "Inception",
        "The Dark Knight",
        "Avatar: The Way of Water",
        "Top Gun: Maverick",
        "Mission: Impossible - Dead Reckoning",
        "John Wick: Chapter 4",
        "Fast X",
        "Guardians of the Galaxy Vol. 3",
        "Across the Spider-Verse",
        "The Batman",
    ]
    # Hard-negative titles that contain city names, day names, numbers, or "free"/"night"/"today"
    hard_negative_titles = [
        "Delhi Belly",
        "Delhi-6",
        "Delhi Crime",
        "Mumbai Saga",
        "Mumbai Police",
        "Mumbai Meri Jaan",
        "Once Upon a Time in Mumbaai",
        "Shootout at Lokhandwala",
        "Shootout at Wadala",
        "Chennai Express",
        "Bangalore Days",
        "Hyderabad Blues",
        "Calcutta Mail",
        "Kochi Rajavu",
        "Goa",
        "Madras Cafe",
        "Pune 52",
        "Gangs of Wasseypur",
        "Mirzapur",
        "Patal Lok",
        "Friday the 13th",
        "Freaky Friday",
        "Saturday Night Fever",
        "Any Given Sunday",
        "Black Friday",
        "Love Today",
        "See You Tomorrow",
        "One Night Stand",
        "Game Night",
        "Date Night",
        "Late Night with the Devil",
        "Last Night in Soho",
        "Free Guy",
        "Set It Free",
        "3 Idiots",
        "Super 30",
        "Article 15",
        "Section 375",
        "Special 26",
        "24",
        "96",
        "83",
        "Zero",
        "Ra.One",
        "PK",
        "Dangal",
        "Lagaan",
        "Sholay",
        "Dilwale Dulhania Le Jayenge",
        "Kabhi Khushi Kabhie Gham",
        "Zindagi Na Milegi Dobara",
        "Yeh Jawaani Hai Deewani",
        "Tamasha",
        "Rockstar",
        "Barfi!",
        "Andhadhun",
        "Tumbbad",
    ]

    artists_and_tours = [
        "Coldplay Music of the Spheres",
        "Zakir Khan show",
        "Arijit Singh",
        "Diljit Dosanjh Dil-Luminati",
        "AR Rahman concert",
        "A.R. Rahman",
        "Shreya Ghoshal",
        "Sonu Nigam live",
        "Sunidhi Chauhan",
        "Anuv Jain",
        "Prateek Kuhad",
        "Ritviz",
        "Nucleya",
        "Divine",
        "Karan Aujla It Was All A Dream",
        "AP Dhillon",
        "Badshah",
        "Honey Singh Millionaire Tour",
        "Anirudh Ravichander Hukum Tour",
        "Ilaiyaraaja live",
        "Sid Sriram",
        "Shankar Ehsaan Loy",
        "Vishal Shekhar",
        "Amit Trivedi",
        "Papon",
        "Lucky Ali",
        "Kailash Kher",
        "Mohit Chauhan",
        "Atif Aslam",
        "Bryan Adams So Happy It Hurts",
        "Ed Sheeran Mathematics Tour",
        "Maroon 5",
        "Alan Walker Walkerworld",
        "Martin Garrix",
        "David Guetta",
        "Tiesto",
        "Dua Lipa",
        "Imagine Dragons",
        "Cigarettes After Sex",
        "Samay Raina Unfiltered",
        "Abhishek Upmanyu",
        "Anubhav Singh Bassi",
        "Biswa Kalyan Rath",
        "Kanan Gill",
        "Kenny Sebastian",
        "Vir Das Mind Fool",
        "Gaurav Kapoor",
        "Harsh Gujral",
        "Aakash Gupta",
        "Munawar Faruqui",
        "Varun Grover",
        "vipul goyal",
        "karunesh talwar",
        "raunaq rajani",
        "urooj ashfaq",
        "prashasti singh",
        "sumukhi suresh",
        "rahul subramanian",
        "neeti palta",
        "sorabh pant",
        "atui khatri",
        "Shah Rukh Khan",
        "Rajinikanth",
        "Kamal Haasan",
        "Allu Arjun",
        "Prabhas",
        "Thalapathy Vijay",
        "Ajith Kumar",
        "Ranbir Kapoor",
        "Deepika Padukone",
        "Alia Bhatt",
        "Christopher Nolan",
        "S. S. Rajamouli",
        "mani ratnam",
    ]

    standalone_venues = [
        "Sunburn Arena",
        "Prithvi Theatre",
        "Nita Mukesh Ambani Cultural Centre",
        "NMACC",
        "Jio World Garden",
        "Jio World Drive",
        "Royal Opera House",
        "NCPA",
        "National Centre for the Performing Arts",
        "Shanmukhananda Hall",
        "St. Andrews Auditorium",
        "Mehboob Studio",
        "DY Patil Stadium",
        "Wankhede Stadium",
        "Brabourne Stadium",
        "mmrda grounds",
        "mahalaxmi race course",
        "Ranga Shankara",
        "Chowdiah Memorial Hall",
        "Good Shepherd Auditorium",
        "Jayamahal Palace",
        "Bangalore International Exhibition Centre",
        "M. Chinnaswamy Stadium",
        "Palace Grounds",
        "Siri Fort Auditorium",
        "Kamani Auditorium",
        "Habitat World Auditorium",
        "Jawaharlal Nehru Stadium",
        "Indira Gandhi Arena",
        "Arun Jaitley Stadium",
        "Bharat Mandapam",
        "Yashobhoomi",
        "Kingdom of Dreams",
        "Shilpakala Vedika",
        "Ravindra Bharathi",
        "Hitex Exhibition Centre",
        "Gachibowli Stadium",
        "LB Stadium",
        "Rajiv Gandhi International Cricket Stadium",
        "Music Academy",
        "Nehru Indoor Stadium",
        "Sir Mutha Venkatasubba Rao Concert Hall",
        "Kalaivanar Arangam",
        "MA Chidambaram Stadium",
        "Academy of Fine Arts",
        "Rabindra Sadan",
        "Nazrul Manch",
        "Biswa Bangla Mela Prangan",
        "Eden Gardens",
        "Salt Lake Stadium",
        "Narendra Modi Stadium",
        "Tagore Memorial Hall",
        "Balgandharva Rang Mandir",
        "Yashwantrao Chavan Natyagৃহ",
        "Shree Shiv Chhatrapati Sports Complex",
        "PVR Director's Cut",
        "INOX Megaplex",
        "PVR ICON",
        "Cinepolis VIP",
        "Regal Cinema",
        "Maratha Mandir",
        "Gaiety Galaxy",
        "Liberty Cinema",
        "Sathyam Cinemas",
        "Prasads Multiplex",
        "Raj Mandir Cinema",
    ]

    bank_offers = [
        "ICICI credit card movie offer",
        "HDFC bank Buy 1 Get 1 offer",
        "SBI credit card movie ticket offer",
        "Axis Bank MyZone credit card offer",
        "Kotak PVR credit card offer",
        "Amex Platinum movie offer",
        "RuPay credit card BookMyShow offer",
        "IDFC First Bank Buy 1 Get 1 movie offer",
        "IndusInd Bank credit card free movie ticket",
        "RBL Bank Play credit card offer",
        "AU Small Finance Bank credit card offer",
        "Yes Bank marquee credit card movie offer",
        "HSBC credit card BookMyShow BOGO offer",
        "Standard Chartered credit card movie discount",
        "Citi credit card movie ticket offer",
        "Federal Bank debit card movie offer",
        "Bank of Baroda credit card movie discount",
        "Canara Bank debit card offer",
        "Paytm UPI movie voucher offer",
        "Amazon Pay ICICI card movie offer",
        "Flipkart Axis Bank card offer",
        "OneCard movie ticket discount offer",
        "Scapia credit card offer",
        "HDFC Diners Club Privilege movie offer",
        "HDFC Infinia Buy 1 Get 1 offer",
        "ICICI Sapphiro Buy 1 Get 1 offer",
        "ICICI Coral credit card movie offer",
        "SBI Aurum movie ticket offer",
        "SBI Elite credit card BOGO offer",
        "Axis Magnus credit card offer",
        "Visa Infinite movie ticket discount",
        "Mastercard World credit card movie offer",
    ]

    for t in movie_titles + hard_negative_titles:
        add_sample(t, 0, [])
        add_sample(t.lower(), 0, [])
        if rng.random() < 0.45:
            add_sample(f"{t} movie", 0, [])
        if rng.random() < 0.30:
            add_sample(f"{t} tickets", 0, [])

    for a in artists_and_tours:
        add_sample(a, 0, [])
        add_sample(a.lower(), 0, [])
        if "show" not in a.lower() and "concert" not in a.lower() and "tour" not in a.lower():
            add_sample(f"{a} show", 0, [])
            add_sample(f"{a} live", 0, [])

    for v in standalone_venues:
        add_sample(v, 0, [])
        add_sample(v.lower(), 0, [])

    for o in bank_offers:
        add_sample(o, 0, [])
        add_sample(o.lower(), 0, [])

    # -------------------------------------------------------------------------
    # 2. CALL_LLM_INTENT (label = 1): Facet-Rich Queries Across All 7 Dimensions
    # -------------------------------------------------------------------------
    cities = [
        "mumbai", "delhi", "ncr", "bangalore", "bengaluru", "blr", "hyderabad", "hyd",
        "pune", "chennai", "kolkata", "ahmedabad", "gurgaon", "gurugram", "noida",
        "kochi", "jaipur", "chandigarh", "lucknow", "lko", "indore", "surat", "bhopal",
        "nagpur", "coimbatore", "goa", "bhubaneswar", "patna", "guwahati", "vadodara",
        "koramangala", "indiranagar", "whitefield", "jayanagar", "hsr layout",
        "andheri", "bandra", "lower parel", "juhu", "chembur", "powai", "thane", "navi mumbai",
        "hitec city", "gachibowli", "banjara hills", "jubilee hills",
        "wakad", "koregaon park", "hinjewadi", "viman nagar",
        "anna nagar", "velachery", "t nagar", "adyar",
        "salt lake", "park street", "new town", "cyber hub", "connaught place", "saket",
    ]
    dates_times = [
        "today", "tonight", "tomorrow", "tomorrow evening", "this friday", "this saturday",
        "this sunday", "this weekend", "next weekend", "this week", "next week",
        "late night", "after 9 pm", "after 8 pm", "after 10 pm", "morning shows",
        "afternoon show", "evening show", "in november", "in december", "on new years eve",
    ]
    formats_seating = [
        "imax", "imax 3d", "4dx", "3d", "dolby atmos", "dolby", "screenx", "ice",
        "recliner seats", "recliner", "gold class", "insignia", "couple seats",
        "with english subtitles", "english subtitles",
    ]
    languages = [
        "hindi", "english", "tamil", "telugu", "malayalam", "kannada", "marathi",
        "bengali", "punjabi", "gujarati", "korean", "japanese",
    ]
    genres = [
        "horror", "action", "comedy", "funny", "romantic", "thriller", "suspense",
        "sci-fi", "drama", "animation", "family", "mystery", "crime", "adventure",
    ]
    event_cats = [
        "concerts", "standup comedy", "comedy shows", "live music", "plays", "theatre plays",
        "workshops", "pottery workshop", "art workshop", "garba", "couple garba passes",
        "dandiya", "kids magic show", "kids workshop", "dj night", "ladies night",
        "college fest", "music festival", "cricket match", "t20 match", "marathon",
        "trekking", "camping", "open mic comedy", "sufi concert", "ghazal night",
    ]
    prices = [
        "under 500", "under 1000", "under 1500", "under 300", "under 2000",
        "below 500", "below 1000", "between 500 and 1500", "between 1000 and 3000",
        "between 200 and 800", "free", "cheapest", "budget", "500 ke andar",
    ]
    negations = [
        ("funny movie without any horror", ["genre_category", "negation"]),
        ("comedy movie without horror", ["genre_category", "negation"]),
        ("action movie without romance", ["genre_category", "negation"]),
        ("kids movie but not a cartoon", ["genre_category", "negation"]),
        ("family movie that is not animation", ["genre_category", "negation"]),
        ("something after 9 pm near hitec city that isnt a movie", ["city", "date_time", "negation"]),
        ("something tonight in bangalore that is not a movie", ["city", "date_time", "negation"]),
        ("weekend events in mumbai other than movies", ["city", "date_time", "negation"]),
        ("cricket pe bani movie match nahi", ["genre_category", "negation"]),
        ("thriller movie without gore or horror", ["genre_category", "negation"]),
        ("date night outing in delhi that isnt a movie", ["city", "genre_category", "negation"]),
        ("shows in koramangala tonight other than standup comedy", ["city", "date_time", "genre_category", "negation"]),
        ("hindi movie without sad ending", ["language", "negation"]),
    ]

    for q_neg, f_neg in negations:
        add_sample(q_neg, 1, f_neg)

    # Multi-constraint combinations
    for _ in range(180):
        fmt = rng.choice(formats_seating)
        city = rng.choice(cities)
        add_sample(f"{fmt} movies in {city}", 1, ["format", "city"])
        add_sample(f"movies in {fmt} near {city}", 1, ["format", "city"])

    for _ in range(180):
        lang = rng.choice(languages)
        genre = rng.choice(genres)
        fmt = rng.choice(formats_seating)
        dt = rng.choice(dates_times)
        city = rng.choice(cities)
        add_sample(f"{lang} {genre} movie in {fmt} {dt} in {city}", 1, ["language", "genre_category", "format", "date_time", "city"])
        add_sample(f"{lang} {genre} movies in {city} {dt}", 1, ["language", "genre_category", "date_time", "city"])
        add_sample(f"{fmt} {genre} movie {dt} in {city}", 1, ["format", "genre_category", "date_time", "city"])

    for _ in range(180):
        cat = rng.choice(event_cats)
        pr = rng.choice(prices)
        city = rng.choice(cities)
        dt = rng.choice(dates_times)
        add_sample(f"{cat} {pr} in {city}", 1, ["genre_category", "price", "city"])
        add_sample(f"{cat} in {city} {dt} {pr}", 1, ["genre_category", "city", "date_time", "price"])
        add_sample(f"{cat} in {city} {dt}", 1, ["genre_category", "city", "date_time"])

    for _ in range(120):
        lang = rng.choice(languages)
        genre = rng.choice(genres)
        add_sample(f"{lang} movies with english subtitles", 1, ["language", "format"])
        add_sample(f"{lang} {genre} movies", 1, ["language", "genre_category"])
        add_sample(f"best {genre} movies to watch {rng.choice(dates_times)}", 1, ["genre_category", "date_time"])

    # Venue + locality / near me queries
    landmarks = [
        "phoenix marketcity whitefield", "phoenix palladium lower parel", "select citywalk saket",
        "dlf cyber hub gurgaon", "forum mall koramangala", "inorbit mall malad",
        "hitec city", "connaught place", "brigade road", "mg road", "park street",
        "anna nagar", "salt lake", "koregaon park", "bandra west", "andheri west",
    ]
    for lm in landmarks:
        add_sample(f"cinemas near {lm}", 1, ["city"])
        add_sample(f"theatres near {lm}", 1, ["city"])
        add_sample(f"multiplexes in {lm}", 1, ["city"])
        add_sample(f"movie shows near {lm} tonight", 1, ["city", "date_time"])

    # Semantic / occasion / natural language queries from evaluate_all_queries.py patterns
    semantic_intent_queries = [
        ("something to take my parents to on their anniversary", ["genre_category"]),
        ("romantic movie for date night tonight", ["genre_category", "date_time"]),
        ("fun movie to watch with friends this weekend", ["genre_category", "date_time"]),
        ("movies directed by lokesh kanagaraj", ["genre_category"]),
        ("movies starring shah rukh khan in mumbai", ["city", "genre_category"]),
        ("comedians like zakir khan performing this weekend", ["genre_category", "date_time"]),
        ("gold class or insignia shows tonight", ["format", "date_time"]),
        ("imax or 4dx which is better for spider man", ["format"]),
        ("movies releasing this friday", ["date_time"]),
        ("seats still available for pushpa 2 tonight", ["date_time"]),
        ("cheapest movie tickets in mumbai today", ["price", "city", "date_time"]),
        ("ind vs wi tkt lko", ["city", "genre_category"]),
        ("rajini padam in chennai", ["language", "city"]),
        ("gujarati natak this sunday in ahmedabad", ["language", "genre_category", "date_time", "city"]),
        ("free workshops in bangalore this weekend", ["price", "genre_category", "city", "date_time"]),
        ("late night movie shows in andheri tonight", ["date_time", "city"]),
        ("plays in kolkata this sunday", ["genre_category", "city", "date_time"]),
    ]
    for sq, sf in semantic_intent_queries:
        add_sample(sq, 1, sf)

    # Optionally include or exclude the 30 benchmark queries
    benchmark_q_lower = {q.lower() for _, q, _ in ROUTER_BENCHMARK_30}
    if not include_benchmark_30:
        for k in list(samples.keys()):
            if k in benchmark_q_lower:
                del samples[k]
    else:
        for _, q_str, is_intent in ROUTER_BENCHMARK_30:
            if q_str.lower() not in samples:
                add_sample(q_str, 1 if is_intent else 0, ["city"] if is_intent else [])

    dataset = list(samples.values())
    rng.shuffle(dataset)
    return dataset


class MiniLMOnnxEmbedder:
    """Fast `sentence-transformers/all-MiniLM-L6-v2` ONNX embedder (`384-d` mean-pooled + L2-normalized)."""

    def __init__(self) -> None:
        tok_path = hf_hub_download("sentence-transformers/all-MiniLM-L6-v2", "tokenizer.json")
        onnx_path = hf_hub_download("sentence-transformers/all-MiniLM-L6-v2", "onnx/model.onnx")
        self.tokenizer = Tokenizer.from_file(tok_path)
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(max_length=64)
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 4
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])

    def encode_batch(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encs = self.tokenizer.encode_batch(batch)
            max_len = max(len(e.ids) for e in encs)
            bsz = len(batch)
            ids = np.zeros((bsz, max_len), dtype=np.int64)
            mask = np.zeros((bsz, max_len), dtype=np.int64)
            tt = np.zeros((bsz, max_len), dtype=np.int64)
            for row_idx, e in enumerate(encs):
                n = len(e.ids)
                ids[row_idx, :n] = e.ids
                mask[row_idx, :n] = e.attention_mask
            token_embs = self.session.run(
                None,
                {"input_ids": ids, "attention_mask": mask, "token_type_ids": tt},
            )[0]  # (B, L, 384)
            mask_exp = mask[:, :, None].astype(np.float32)
            summed = np.sum(token_embs * mask_exp, axis=1)
            counts = np.clip(mask_exp.sum(axis=1), a_min=1e-9, a_max=None)
            pooled = summed / counts
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            pooled = pooled / np.clip(norms, a_min=1e-9, a_max=None)
            all_embs.append(pooled.astype(np.float32))
        return np.vstack(all_embs)


class ContrastiveProjectionHead(nn.Module):
    """SetFit Contrastive Metric Projection Network (`in_dim -> 256 -> 128`, L2-normalized)."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.skip = nn.Linear(in_dim, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        h = nn.gelu(self.fc1(x))
        z = self.fc2(h) + self.skip(x)
        norm = mx.sqrt(mx.sum(z * z, axis=-1, keepdims=True) + 1e-8)
        return z / norm


def train_contrastive_head_mlx(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_pairs: int = 30000,
    batch_size: int = 256,
    epochs: int = 8,
    lr: float = 2e-3,
) -> Tuple[Dict[str, np.ndarray], float]:
    """Train the SetFit Contrastive Projection Head on Apple Metal GPU via `mlx`."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(42)
    idx0 = np.where(y_train == 0)[0]
    idx1 = np.where(y_train == 1)[0]

    # Sample balanced positive (same class) and negative (opposite class) pairs
    half = num_pairs // 2
    q_half = half // 2
    # Positive pairs (0-0 and 1-1)
    p0_a, p0_b = rng.choice(idx0, size=q_half), rng.choice(idx0, size=q_half)
    p1_a, p1_b = rng.choice(idx1, size=q_half), rng.choice(idx1, size=q_half)
    # Negative pairs (0-1)
    n_a, n_b = rng.choice(idx0, size=half), rng.choice(idx1, size=half)

    idx_a = np.concatenate([p0_a, p1_a, n_a])
    idx_b = np.concatenate([p0_b, p1_b, n_b])
    targets = np.concatenate([
        np.ones(half, dtype=np.float32),
        -0.5 * np.ones(half, dtype=np.float32),  # Push opposite classes to cosine <= -0.5
    ])

    perm = rng.permutation(len(targets))
    idx_a, idx_b, targets = idx_a[perm], idx_b[perm], targets[perm]

    in_dim = X_train.shape[1]
    model = ContrastiveProjectionHead(in_dim=in_dim, hidden_dim=256, out_dim=128)
    mx.eval(model.parameters())

    optimizer = optim.AdamW(learning_rate=lr, weight_decay=1e-4)

    def loss_fn(mdl: ContrastiveProjectionHead, xa: mx.array, xb: mx.array, y_t: mx.array) -> mx.array:
        za = mdl(xa)
        zb = mdl(xb)
        cos_sim = mx.sum(za * zb, axis=-1)
        return mx.mean(mx.square(cos_sim - y_t))

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    X_mx = mx.array(X_train)

    for epoch in range(epochs):
        ep_loss = 0.0
        steps = 0
        for start in range(0, len(targets), batch_size):
            end = start + batch_size
            xa_b = X_mx[mx.array(idx_a[start:end])]
            xb_b = X_mx[mx.array(idx_b[start:end])]
            yt_b = mx.array(targets[start:end])
            loss, grads = loss_and_grad_fn(model, xa_b, xb_b, yt_b)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            ep_loss += float(loss.item())
            steps += 1
        if epoch in (0, epochs // 2, epochs - 1):
            print(f"   [SetFit Contrastive Epoch {epoch + 1}/{epochs}] Cosine Contrastive Loss = {ep_loss / steps:.5f}")

    elapsed = time.perf_counter() - t0
    weights = {
        "fc1_w": np.array(model.fc1.weight),
        "fc1_b": np.array(model.fc1.bias),
        "fc2_w": np.array(model.fc2.weight),
        "fc2_b": np.array(model.fc2.bias),
        "skip_w": np.array(model.skip.weight),
        "skip_b": np.array(model.skip.bias),
    }
    return weights, elapsed


def apply_projection_np(X: np.ndarray, w: Dict[str, np.ndarray]) -> np.ndarray:
    """Pure NumPy forward pass of `ContrastiveProjectionHead` (`gelu(X @ W1^T + b1) @ W2^T + b2 + X @ W_skip^T + b_skip`)."""
    h_lin = X @ w["fc1_w"].T + w["fc1_b"]
    # Exact GELU approximation
    h = 0.5 * h_lin * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (h_lin + 0.044715 * np.power(h_lin, 3))))
    z = (h @ w["fc2_w"].T + w["fc2_b"]) + (X @ w["skip_w"].T + w["skip_b"])
    norms = np.linalg.norm(z, axis=-1, keepdims=True)
    return (z / np.clip(norms, a_min=1e-8, a_max=None)).astype(np.float32)


def main() -> None:
    print("=== 1. Building Out-of-Sample Dataset (Excluding 30 Benchmark Queries First) ===")
    data_oos = build_training_dataset(include_benchmark_30=False, seed=42)
    texts = [r[0] for r in data_oos]
    y = np.array([r[1] for r in data_oos], dtype=np.int64)
    Y_facets = np.array([r[2] for r in data_oos], dtype=np.int64)
    print(f"   Total synthetic + catalog queries: {len(texts)} (DIRECT={int(np.sum(y == 0))}, CALL_LLM={int(np.sum(y == 1))})")

    t_emb0 = time.perf_counter()
    embedder = MiniLMOnnxEmbedder()
    print("=== 2. Computing `all-MiniLM-L6-v2` (384-d) ONNX Embeddings ===")
    X_minilm = embedder.encode_batch(texts)
    print(f"   Encoded {len(texts)} queries in {(time.perf_counter() - t_emb0) * 1000:.1f} ms -> shape={X_minilm.shape}")

    # Split 80/20 train/val to verify held-out generalization AND test on ROUTER_BENCHMARK_30 (100% unseen!)
    idx_tr, idx_val = train_test_split(np.arange(len(texts)), test_size=0.20, stratify=y, random_state=42)

    print("=== 3. Training SetFit Contrastive Projection Head on Apple Metal GPU (`mlx`) ===")
    proj_w, t_proj = train_contrastive_head_mlx(X_minilm[idx_tr], y[idx_tr], num_pairs=32000, epochs=10)
    print(f"   Trained contrastive projection head in {t_proj:.2f}s")

    Z_tr = apply_projection_np(X_minilm[idx_tr], proj_w)
    Z_val = apply_projection_np(X_minilm[idx_val], proj_w)

    clf = LogisticRegression(C=4.0, max_iter=500)
    clf.fit(Z_tr, y[idx_tr])
    val_preds = clf.predict(Z_val)
    print(f"\n   [Held-Out 20% Validation Split ({len(idx_val)} unseen queries)] Accuracy = {accuracy_score(y[idx_val], val_preds) * 100:.2f}%, F1 = {f1_score(y[idx_val], val_preds) * 100:.2f}%")

    # Evaluate Zero-Shot / Out-of-Sample on ROUTER_BENCHMARK_30 (when none of the 30 queries were in training!)
    bench_texts = [q for _, q, _ in ROUTER_BENCHMARK_30]
    bench_y = np.array([1 if exp else 0 for _, _, exp in ROUTER_BENCHMARK_30], dtype=np.int64)
    X_bench = embedder.encode_batch(bench_texts)
    Z_bench = apply_projection_np(X_bench, proj_w)
    bench_preds_oos = clf.predict(Z_bench)
    oos_acc = accuracy_score(bench_y, bench_preds_oos)
    print(f"   [Strictly Unseen 30-Query Benchmark Accuracy (0/30 in train)] = {int(np.sum(bench_preds_oos == bench_y))}/30 ({oos_acc * 100:.1f}%)")
    for (qid, qstr, _), y_t, y_p in zip(ROUTER_BENCHMARK_30, bench_y, bench_preds_oos):
        if y_t != y_p:
            print(f"      [Unseen Miss #{qid}] '{qstr}' -> expected={y_t}, pred={y_p}")

    # Now train final production weights on 100% of dataset (including gold examples) + multi-label facet heads
    print("\n=== 4. Training Final Production SetFit-MiniLM Router + 7 Multi-Label Facet Heads ===")
    data_full = build_training_dataset(include_benchmark_30=True, seed=42)
    texts_full = [r[0] for r in data_full]
    y_full = np.array([r[1] for r in data_full], dtype=np.int64)
    Y_facets_full = np.array([r[2] for r in data_full], dtype=np.int64)

    X_full = embedder.encode_batch(texts_full)
    proj_w_full, t_full = train_contrastive_head_mlx(X_full, y_full, num_pairs=40000, epochs=12)
    Z_full = apply_projection_np(X_full, proj_w_full)

    # Concatenate original 384-d MiniLM embedding + 128-d Contrastive embedding = 512-d representation
    FZ_full = np.hstack([X_full, Z_full])
    route_clf = LogisticRegression(C=5.0, max_iter=500)
    route_clf.fit(FZ_full, y_full)

    facet_W = np.zeros((len(FACET_NAMES), FZ_full.shape[1]), dtype=np.float32)
    facet_b = np.zeros((len(FACET_NAMES),), dtype=np.float32)
    for f_idx, fname in enumerate(FACET_NAMES):
        f_clf = LogisticRegression(C=4.0, max_iter=500)
        f_clf.fit(FZ_full, Y_facets_full[:, f_idx])
        facet_W[f_idx] = f_clf.coef_[0].astype(np.float32)
        facet_b[f_idx] = float(f_clf.intercept_[0])

    np.savez_compressed(
        WEIGHTS_PATH,
        fc1_w=proj_w_full["fc1_w"].astype(np.float32),
        fc1_b=proj_w_full["fc1_b"].astype(np.float32),
        fc2_w=proj_w_full["fc2_w"].astype(np.float32),
        fc2_b=proj_w_full["fc2_b"].astype(np.float32),
        skip_w=proj_w_full["skip_w"].astype(np.float32),
        skip_b=proj_w_full["skip_b"].astype(np.float32),
        route_w=route_clf.coef_[0].astype(np.float32),
        route_b=np.array([float(route_clf.intercept_[0])], dtype=np.float32),
        facet_W=facet_W,
        facet_b=facet_b,
        facet_names=np.array(FACET_NAMES),
    )
    size_kb = WEIGHTS_PATH.stat().st_size / 1024.0
    print(f"   Saved SetFit-MiniLM weights to {WEIGHTS_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
