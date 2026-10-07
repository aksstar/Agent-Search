import json
import random
import re
from datetime import datetime, timedelta, timezone
from config import LOCAL_JSON_FILE

random.seed(42)  # same output on every run; remove this line to get new data each time

IST = timezone(timedelta(hours=5, minutes=30))
TODAY = datetime(2026, 10, 6, tzinfo=IST)
SHOW_WINDOW_DAYS = 21
LISTINGS_PER_TITLE = 5  # 20 movies x 5 + 20 events x 5 = 200 records


def iso(dt):
    return dt.isoformat(timespec="seconds")


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def at(day, hhmm):
    h, m = map(int, hhmm.split(":"))
    return day.replace(hour=h, minute=m, second=0, microsecond=0)


def pick_show_day(not_before):
    days = [TODAY + timedelta(days=d) for d in range(SHOW_WINDOW_DAYS)]
    days = [d for d in days if d >= not_before] or days
    weights = [3 if d.weekday() >= 4 else 1 for d in days]  # Fri-Sun weighted for "this weekend"
    return random.choices(days, weights=weights, k=1)[0]


# City -> (state, [cinema venue, event venue]) with approximate coordinates
CITIES = {
    "Mumbai":    ("Maharashtra", [("PVR ICON Phoenix Palladium, Lower Parel", 18.9946, 72.8245),
                                  ("NSCI Dome, Worli", 19.0090, 72.8175)]),
    "Delhi NCR": ("Delhi",       [("PVR Select Citywalk, Saket", 28.5286, 77.2193),
                                  ("Jawaharlal Nehru Stadium", 28.5829, 77.2344)]),
    "Bengaluru": ("Karnataka",   [("PVR Orion Mall, Rajajinagar", 13.0110, 77.5550),
                                  ("Phoenix Marketcity, Whitefield", 12.9975, 77.6964)]),
    "Hyderabad": ("Telangana",   [("Prasads Multiplex, Khairatabad", 17.4126, 78.4664),
                                  ("HITEX Exhibition Centre", 17.4709, 78.3727)]),
    "Chennai":   ("Tamil Nadu",  [("Sathyam Cinemas, Royapettah", 13.0556, 80.2580),
                                  ("YMCA Grounds, Nandanam", 13.0285, 80.2390)]),
    "Pune":      ("Maharashtra", [("INOX Phoenix Marketcity, Viman Nagar", 18.5622, 73.9167),
                                  ("Mahalaxmi Lawns, Karve Nagar", 18.4892, 73.8205)]),
    "Kolkata":   ("West Bengal", [("INOX Quest Mall, Ballygunge", 22.5390, 88.3654),
                                  ("Science City Auditorium", 22.5397, 88.3960)]),
    "Goa":       ("Goa",         [("INOX Panaji", 15.4989, 73.8278),
                                  ("Vagator Beach Grounds", 15.6029, 73.7339)]),
}


def build_location(city, venue_idx):
    state, venues = CITIES[city]
    name, lat, lng = venues[venue_idx]
    return {
        "city": city,
        "venue": name,
        "location_city": {"address": f"{city}, {state}, India"},  # type 1: city
        "location_latlng": {"lat": lat, "lng": lng},              # type 2: lat/long
    }


# ---------------- MOVIES ----------------
# (title, genres, artists, runtime_minutes, languages)
MOVIES = [
    ("Drishyam: The Conclusion", ["Drama", "Mystery", "Thriller"], ["Ajay Devgn", "Tabu", "Shriya Saran"], 155, ["Hindi"]),
    ("Kalki 2898 AD Part 2", ["Sci-Fi", "Action"], ["Prabhas", "Amitabh Bachchan", "Deepika Padukone"], 180, ["Telugu", "Hindi"]),
    ("Jawan 2", ["Action", "Thriller"], ["Shah Rukh Khan", "Nayanthara", "Vijay Sethupathi"], 165, ["Hindi", "Tamil", "Telugu"]),
    ("Bramayugam 2", ["Horror", "Mystery"], ["Mammootty", "Arjun Ashokan"], 140, ["Malayalam"]),
    ("Shaitaan Returns", ["Horror", "Thriller"], ["Ajay Devgn", "R. Madhavan", "Jyotika"], 135, ["Hindi"]),
    ("War 3", ["Action", "Thriller"], ["Hrithik Roshan", "NTR Jr", "Kiara Advani"], 160, ["Hindi", "Telugu"]),
    ("Pushpa 3: The Rampage", ["Action", "Drama"], ["Allu Arjun", "Rashmika Mandanna", "Fahadh Faasil"], 175, ["Telugu", "Hindi"]),
    ("Ramayana: Part 1", ["Fantasy", "Drama"], ["Ranbir Kapoor", "Sai Pallavi", "Yash"], 170, ["Hindi", "Telugu", "Tamil"]),
    ("Coolie", ["Action", "Crime"], ["Rajinikanth", "Nagarjuna", "Shruti Haasan"], 168, ["Tamil", "Telugu"]),
    ("Dhoom 4", ["Action", "Thriller"], ["Ranveer Singh", "Abhishek Bachchan"], 150, ["Hindi"]),
    ("Housefull 6", ["Comedy"], ["Akshay Kumar", "Riteish Deshmukh", "Jacqueline Fernandez"], 145, ["Hindi"]),
    ("Stree 3", ["Horror", "Comedy"], ["Rajkummar Rao", "Shraddha Kapoor", "Pankaj Tripathi"], 140, ["Hindi"]),
    ("Toxic", ["Action", "Crime"], ["Yash", "Nayanthara", "Kiara Advani"], 160, ["Kannada", "Hindi", "Telugu"]),
    ("Game Changer 2", ["Action", "Drama"], ["Ram Charan", "Kiara Advani"], 158, ["Telugu", "Tamil"]),
    ("L3: The Beginning", ["Action", "Thriller"], ["Mohanlal", "Prithviraj Sukumaran"], 170, ["Malayalam", "Telugu"]),
    ("Metro In Dino 2", ["Romance", "Drama"], ["Aditya Roy Kapur", "Sara Ali Khan"], 150, ["Hindi"]),
    ("Kantara: Chapter 2", ["Action", "Drama"], ["Rishab Shetty", "Rukmini Vasanth"], 165, ["Kannada", "Telugu", "Hindi"]),
    ("Animal Park", ["Action", "Crime", "Drama"], ["Ranbir Kapoor", "Triptii Dimri"], 190, ["Hindi", "Telugu"]),
    ("Krrish 4", ["Sci-Fi", "Action"], ["Hrithik Roshan", "Priyanka Chopra"], 155, ["Hindi"]),
    ("Welcome to the Jungle", ["Comedy", "Adventure"], ["Akshay Kumar", "Suniel Shetty", "Arshad Warsi"], 150, ["Hindi"]),
]
FORMATS = ["2D", "3D", "IMAX 2D", "IMAX 3D", "4DX", "DOLBY CINEMA 2D", "EPIQ", "ICE"]
HASHTAGS = ["LIVE_IN_CINEMAS", "MUST_WATCH", "BLOCKBUSTER", "CRITICALLY_ACCLAIMED", "FAMILY_ENTERTAINER"]
MOVIE_SLOTS = ["09:30", "12:45", "16:00", "19:15", "22:30"]

movies_data = []
rec = 1
for m_idx, (title, genres, artists, runtime, langs) in enumerate(MOVIES):
    code = f"ET{477900 + m_idx * 137:08d}"
    slug = slugify(title)
    release = TODAY - timedelta(days=random.randint(0, 40))
    formats = random.sample(FORMATS, k=random.randint(1, 3))
    for city in random.sample(list(CITIES), LISTINGS_PER_TITLE):
        slot_str = random.choice(MOVIE_SLOTS)
        start = at(pick_show_day(release), slot_str)
        end = start + timedelta(minutes=runtime)
        tags = random.sample(HASHTAGS, k=random.randint(1, 2))
        if (TODAY - release).days <= 7:
            tags.insert(0, "NEW_RELEASE")
        if slot_str == "09:30":
            tags.append("MORNING_SHOW")
        elif slot_str == "22:30":
            tags.append("LATE_NIGHT_SHOW")
        movies_data.append({
            "id": f"MV{rec:05d}",
            "defaultVariantCode": code,
            "title": title,
            "categories": ["movies"],
            "slug": slug,
            "available_time": iso(release),
            "release_date": iso(release),
            "showtime_start": iso(start),
            "showtime_end": iso(end),
            "hash_tags": tags,
            "genres": genres,
            "languages": langs,
            "searchKeywords": ["movie", "cinema", *[g.lower() for g in genres], *[l.lower() for l in langs]],
            "formats": formats,
            "uri": f"https://events.example.com/{slugify(city)}/movies/{slug}/{code}",
            "media_type": "movie",
            "artists": artists,
            **build_location(city, 0),
        })
        rec += 1


# ---------------- TARGETED MOVIES FOR BENCHMARK QUERIES ----------------
# Reference dates (relative to Oct 7, 2026):
# - Tonight:       2026-10-07 (after 10 PM: 22:30, 23:15)
# - Tomorrow:      2026-10-08 (Morning before 11 AM: 09:00, 10:15)
# - This Friday:   2026-10-09
# - Next Month:    2026-11-01 to 2026-11-30 (Upcoming Telugu movies)
TARGETED_MOVIES = [
    # Query 3: "movies releasing this friday" (2026-10-09 vs older releases)
    {
        "title": "Singham Again",
        "genres": ["Action", "Drama"],
        "artists": ["Ajay Devgn", "Ranveer Singh", "Akshay Kumar"],
        "runtime": 165,
        "languages": ["Hindi"],
        "release": datetime(2026, 9, 11, 0, 0, tzinfo=IST),  # Distractor: released last month (Sep 11)
        "showtime": datetime(2026, 10, 9, 19, 30, tzinfo=IST),
        "city": "Mumbai",
        "formats": ["IMAX 2D"],
        "hash_tags": ["BLOCKBUSTER"],
        "searchKeywords": ["movies", "releasing", "friday", "new movie", "cinema"],
    },
    {
        "title": "Bhool Bhulaiyaa 4",
        "genres": ["Horror", "Comedy"],
        "artists": ["Kartik Aaryan", "Vidya Balan", "Madhuri Dixit"],
        "runtime": 150,
        "languages": ["Hindi"],
        "release": datetime(2026, 10, 9, 0, 0, tzinfo=IST),  # Target: Releasing THIS Friday (Oct 9)
        "showtime": datetime(2026, 10, 9, 20, 0, tzinfo=IST),
        "city": "Delhi NCR",
        "formats": ["2D", "4DX"],
        "hash_tags": ["NEW_RELEASE", "FRIDAY_RELEASE", "FAMILY_ENTERTAINER"],
        "searchKeywords": ["movies", "releasing", "friday", "new movie", "cinema"],
    },
    {
        "title": "Vettaiyan: The Hunter",
        "genres": ["Action", "Crime"],
        "artists": ["Rajinikanth", "Amitabh Bachchan", "Fahadh Faasil"],
        "runtime": 162,
        "languages": ["Tamil", "Telugu", "Hindi"],
        "release": datetime(2026, 10, 9, 0, 0, tzinfo=IST),  # Target: Releasing THIS Friday (Oct 9)
        "showtime": datetime(2026, 10, 9, 18, 30, tzinfo=IST),
        "city": "Chennai",
        "formats": ["IMAX 2D", "EPIQ"],
        "hash_tags": ["NEW_RELEASE", "FRIDAY_RELEASE", "MUST_WATCH"],
        "searchKeywords": ["movies", "releasing", "friday", "new movie", "cinema"],
    },

    # Query 5: "late night shows after 10 pm"
    {
        "title": "Deadpool & Wolverine (Afternoon Show)",
        "genres": ["Action", "Comedy", "Sci-Fi"],
        "artists": ["Ryan Reynolds", "Hugh Jackman"],
        "runtime": 135,
        "languages": ["English", "Hindi"],
        "release": datetime(2026, 10, 2, 0, 0, tzinfo=IST),
        "showtime": datetime(2026, 10, 7, 15, 30, tzinfo=IST),  # Distractor: 3:30 PM afternoon show
        "city": "Mumbai",
        "formats": ["IMAX 3D", "4DX"],
        "hash_tags": ["BLOCKBUSTER"],
        "searchKeywords": ["movie", "late night shows", "after 10 pm", "late night shows after 10 pm", "shows"],
    },
    {
        "title": "John Wick: Chapter 5",
        "genres": ["Action", "Thriller"],
        "artists": ["Keanu Reeves", "Donnie Yen"],
        "runtime": 160,
        "languages": ["English", "Hindi"],
        "release": datetime(2026, 10, 2, 0, 0, tzinfo=IST),
        "showtime": datetime(2026, 10, 7, 22, 45, tzinfo=IST),  # Target: 10:45 PM (after 10 PM)
        "city": "Mumbai",
        "formats": ["IMAX 2D", "DOLBY CINEMA 2D"],
        "hash_tags": ["LATE_NIGHT_SHOW", "MUST_WATCH"],
        "searchKeywords": ["movie", "late night shows", "after 10 pm", "late night shows after 10 pm", "shows"],
    },
    {
        "title": "The Batman Part II",
        "genres": ["Action", "Crime", "Thriller"],
        "artists": ["Robert Pattinson", "Zoe Kravitz"],
        "runtime": 165,
        "languages": ["English", "Hindi"],
        "release": datetime(2026, 10, 2, 0, 0, tzinfo=IST),
        "showtime": datetime(2026, 10, 7, 23, 15, tzinfo=IST),  # Target: 11:15 PM (after 10 PM)
        "city": "Bengaluru",
        "formats": ["IMAX 2D", "DOLBY CINEMA 2D"],
        "hash_tags": ["LATE_NIGHT_SHOW", "BLOCKBUSTER"],
        "searchKeywords": ["movie", "late night shows", "after 10 pm", "late night shows after 10 pm", "shows"],
    },

    # Query 9: "upcoming telugu movies next month" (Nov 2026 vs Oct 2026 / Non-Telugu)
    {
        "title": "OG: They Call Him OG",
        "genres": ["Action", "Crime"],
        "artists": ["Pawan Kalyan", "Emraan Hashmi"],
        "runtime": 165,
        "languages": ["Telugu"],
        "release": datetime(2026, 10, 1, 0, 0, tzinfo=IST),  # Distractor: Oct release (not next month)
        "showtime": datetime(2026, 10, 8, 19, 30, tzinfo=IST),
        "city": "Hyderabad",
        "formats": ["2D"],
        "hash_tags": ["LIVE_IN_CINEMAS"],
        "searchKeywords": ["movie", "upcoming", "telugu", "movies", "next month", "upcoming telugu movies next month"],
    },
    {
        "title": "Devara Part 2: Red Sea",
        "genres": ["Action", "Drama"],
        "artists": ["NTR Jr", "Saif Ali Khan", "Janhvi Kapoor"],
        "runtime": 175,
        "languages": ["Telugu"],
        "release": datetime(2026, 11, 6, 0, 0, tzinfo=IST),  # Target: Nov 6, 2026 (Next month + Telugu)
        "showtime": datetime(2026, 11, 6, 19, 0, tzinfo=IST),
        "city": "Hyderabad",
        "formats": ["IMAX 2D", "DOLBY CINEMA 2D"],
        "hash_tags": ["UPCOMING_MOVIE", "MUST_WATCH", "BLOCKBUSTER"],
        "searchKeywords": ["movie", "upcoming", "telugu", "movies", "next month", "upcoming telugu movies next month"],
    },
    {
        "title": "Spirit: The Cop Story",
        "genres": ["Action", "Crime"],
        "artists": ["Prabhas", "Sandeep Reddy Vanga"],
        "runtime": 180,
        "languages": ["Telugu"],
        "release": datetime(2026, 11, 14, 0, 0, tzinfo=IST),  # Target: Nov 14, 2026 (Next month + Telugu)
        "showtime": datetime(2026, 11, 14, 18, 30, tzinfo=IST),
        "city": "Hyderabad",
        "formats": ["IMAX 2D", "EPIQ"],
        "hash_tags": ["UPCOMING_MOVIE", "BLOCKBUSTER"],
        "searchKeywords": ["movie", "upcoming", "telugu", "movies", "next month", "upcoming telugu movies next month"],
    },
    {
        "title": "SSMB29: Globetrotter",
        "genres": ["Adventure", "Action"],
        "artists": ["Mahesh Babu", "S. S. Rajamouli"],
        "runtime": 185,
        "languages": ["Telugu"],
        "release": datetime(2026, 11, 20, 0, 0, tzinfo=IST),  # Target: Nov 20, 2026 (Next month + Telugu)
        "showtime": datetime(2026, 11, 20, 19, 30, tzinfo=IST),
        "city": "Bengaluru",
        "formats": ["IMAX 3D", "DOLBY CINEMA 2D"],
        "hash_tags": ["UPCOMING_MOVIE", "MUST_WATCH"],
        "searchKeywords": ["movie", "upcoming", "telugu", "movies", "next month", "upcoming telugu movies next month"],
    },

    # Query 10: "morning shows tomorrow before 11" (Tomorrow = 2026-10-08 before 11:00)
    {
        "title": "Sitare Zameen Par (Evening Show)",
        "genres": ["Drama", "Comedy"],
        "artists": ["Aamir Khan", "Genelia Deshmukh"],
        "runtime": 155,
        "languages": ["Hindi"],
        "release": datetime(2026, 10, 2, 0, 0, tzinfo=IST),
        "showtime": datetime(2026, 10, 8, 19, 30, tzinfo=IST),  # Distractor: 7:30 PM evening show tomorrow
        "city": "Delhi NCR",
        "formats": ["2D"],
        "hash_tags": ["FAMILY_ENTERTAINER"],
        "searchKeywords": ["movie", "morning shows", "tomorrow", "before 11", "morning shows tomorrow before 11"],
    },
    {
        "title": "Chhaava: The Great Warrior",
        "genres": ["Action", "Drama"],
        "artists": ["Vicky Kaushal", "Rashmika Mandanna"],
        "runtime": 160,
        "languages": ["Hindi"],
        "release": datetime(2026, 10, 2, 0, 0, tzinfo=IST),
        "showtime": datetime(2026, 10, 8, 9, 0, tzinfo=IST),  # Target: 09:00 AM tomorrow (before 11)
        "city": "Mumbai",
        "formats": ["IMAX 2D"],
        "hash_tags": ["MORNING_SHOW", "MUST_WATCH"],
        "searchKeywords": ["movie", "morning shows", "tomorrow", "before 11", "morning shows tomorrow before 11"],
    },
    {
        "title": "Alpha: Spy Universe",
        "genres": ["Action", "Thriller"],
        "artists": ["Alia Bhatt", "Sharvari"],
        "runtime": 150,
        "languages": ["Hindi"],
        "release": datetime(2026, 10, 2, 0, 0, tzinfo=IST),
        "showtime": datetime(2026, 10, 8, 10, 15, tzinfo=IST),  # Target: 10:15 AM tomorrow (before 11)
        "city": "Delhi NCR",
        "formats": ["2D", "DOLBY CINEMA 2D"],
        "hash_tags": ["MORNING_SHOW", "BLOCKBUSTER"],
        "searchKeywords": ["movie", "morning shows", "tomorrow", "before 11", "morning shows tomorrow before 11"],
    },
]

for t_idx, tm in enumerate(TARGETED_MOVIES):
    code = f"ET{490000 + t_idx * 111:08d}"
    slug = slugify(tm["title"])
    start = tm["showtime"]
    end = start + timedelta(minutes=tm["runtime"])
    movies_data.append({
        "id": f"MV{rec:05d}",
        "defaultVariantCode": code,
        "title": tm["title"],
        "categories": ["movies"],
        "slug": slug,
        "available_time": iso(tm["release"]),
        "release_date": iso(tm["release"]),
        "showtime_start": iso(start),
        "showtime_end": iso(end),
        "hash_tags": tm["hash_tags"],
        "genres": tm["genres"],
        "languages": tm["languages"],
        "searchKeywords": tm["searchKeywords"],
        "formats": tm["formats"],
        "uri": f"https://events.example.com/{slugify(tm['city'])}/movies/{slug}/{code}",
        "media_type": "movie",
        "artists": tm["artists"],
        **build_location(tm["city"], 0),
    })
    rec += 1


# ---------------- EVENTS ----------------
# (title, persons, languages, duration_min, ageLimit, ageLimitDisplay, keywords)
EVENTS = [
    ("Ajay Atul Live", ["Ajay Gogavale", "Atul Gogavale"], ["Marathi", "Hindi"], 180, 5, "5yrs +", ["music", "live concert"]),
    ("Sunburn Festival", ["Martin Garrix", "David Guetta"], ["English"], 480, 18, "18yrs +", ["edm", "festival"]),
    ("Zakir Khan Live", ["Zakir Khan"], ["Hindi"], 90, 16, "16yrs +", ["standup comedy"]),
    ("Arijit Singh Live in Concert", ["Arijit Singh"], ["Hindi", "Bengali"], 150, None, "All Ages", ["music", "live concert"]),
    ("Comic Con India", ["Various Artists", "Cosplayers"], ["English", "Hindi"], 600, None, "All Ages", ["cosplay", "comics", "anime"]),
    ("Diljit Dosanjh Dil-Luminati", ["Diljit Dosanjh"], ["Punjabi", "Hindi"], 180, 5, "5yrs +", ["music", "punjabi"]),
    ("Anubhav Singh Bassi Live", ["Anubhav Singh Bassi"], ["Hindi"], 90, 16, "16yrs +", ["standup comedy"]),
    ("Prateek Kuhad Silhouettes Tour", ["Prateek Kuhad"], ["English", "Hindi"], 120, 5, "5yrs +", ["indie", "music"]),
    ("Abhishek Upmanyu Live", ["Abhishek Upmanyu"], ["Hindi", "English"], 90, 16, "16yrs +", ["standup comedy"]),
    ("Lollapalooza India", ["Various Artists"], ["English"], 600, 18, "18yrs +", ["festival", "music"]),
    ("NH7 Weekender", ["Various Artists"], ["English", "Hindi"], 540, 18, "18yrs +", ["festival", "indie"]),
    ("Shreya Ghoshal All Hearts Tour", ["Shreya Ghoshal"], ["Hindi", "Bengali"], 150, None, "All Ages", ["music", "bollywood"]),
    ("Kailash Kher Sufi Night", ["Kailash Kher"], ["Hindi"], 150, None, "All Ages", ["sufi", "music"]),
    ("Nucleya Bass Yatra", ["Nucleya"], ["Hindi"], 180, 18, "18yrs +", ["edm", "bass"]),
    ("Samay Raina Unfiltered", ["Samay Raina"], ["Hindi", "English"], 90, 18, "18yrs +", ["standup comedy"]),
    ("Mughal-e-Azam The Musical", ["Feroz Abbas Khan"], ["Hindi", "Urdu"], 165, 5, "5yrs +", ["theatre", "musical"]),
    ("The Local Train Reunion", ["The Local Train"], ["Hindi"], 150, 12, "12yrs +", ["rock", "music"]),
    ("Aladdin The Broadway Musical", ["Various Artists"], ["English"], 150, None, "All Ages", ["theatre", "kids"]),
    ("Divine Gunehgar Tour", ["Divine"], ["Hindi"], 120, 16, "16yrs +", ["hip hop", "rap"]),
    ("Kids Science Carnival", ["Science Educators"], ["English"], 240, None, "All Ages", ["kids", "workshop"]),
]
EVENT_SLOTS = ["11:00", "16:00", "18:30", "19:30", "21:00"]
FESTIVAL_SLOTS = ["12:00", "14:00"]

events_data = []
rec = 0
for e_idx, (name, persons, langs, duration, age, age_disp, kws) in enumerate(EVENTS):
    group_code = f"EG{50000000 + e_idx:08d}"
    release = TODAY - timedelta(days=random.randint(0, 60))  # listing went live
    for city in random.sample(list(CITIES), LISTINGS_PER_TITLE):
        slots = FESTIVAL_SLOTS if duration >= 480 else EVENT_SLOTS
        start = at(pick_show_day(TODAY), random.choice(slots))
        end = start + timedelta(minutes=duration)
        created = release - timedelta(hours=random.randint(1, 72), minutes=random.randint(0, 59))
        events_data.append({
            "_id": f"etm{100000 + rec:06d}z",
            "ageLimit": age,
            "ageLimitDisplay": age_disp,
            "censorRating": None,
            "createdAt": iso(created),
            "displayRegions": None,
            "duration": duration,
            "eventGroupCode": group_code,
            "format": None,
            "images": None,
            "isActive": True,
            "isDefault": True,
            "isGlobal": random.choice([True, False]),
            "isSearchable": True,
            "isTentativeRelease": False,
            "isWebView": True,
            "languages": langs,
            "persons": persons,
            "release_date": iso(release),
            "showtime_start": iso(start),
            "showtime_end": iso(end),
            "searchKeywords": kws,
            "slug": f"{slugify(name)}-{slugify(city)}",
            "subTitleLanguages": [],
            "tentativeBy": None,
            "title": f"{name.upper()} - {city.upper()}",
            "trailerUrl": random.choice([None, "https://youtube.com/watch?v=eventtrailer"]),
            **build_location(city, 1),
        })
        rec += 1


# ---------------- TARGETED EVENTS FOR BENCHMARK QUERIES ----------------
TARGETED_EVENTS = [
    # Query 1: "open mic next week" (Next week: Mon Oct 12 - Sun Oct 18, 2026)
    {
        "title": "UNSCRIPTED OPEN MIC JAM - DELHI NCR",
        "persons": ["Delhi Poets Collective"],
        "languages": ["Hindi", "Urdu"],
        "duration": 120,
        "ageLimit": 16,
        "ageLimitDisplay": "16yrs +",
        "showtime": datetime(2026, 11, 22, 19, 0, tzinfo=IST),  # Distractor: Nov 22 (not next week)
        "city": "Delhi NCR",
        "searchKeywords": ["open mic", "open mic next week", "comedy", "poetry", "jam"],
        "hash_tags": ["OPEN_MIC"],
    },
    {
        "title": "THE HABITAT COMEDY & POETRY OPEN MIC - MUMBAI",
        "persons": ["Indie Poets", "Up-and-Coming Comedians"],
        "languages": ["Hindi", "English"],
        "duration": 120,
        "ageLimit": 16,
        "ageLimitDisplay": "16yrs +",
        "showtime": datetime(2026, 10, 14, 20, 0, tzinfo=IST),  # Target: Next week (Wed Oct 14)
        "city": "Mumbai",
        "searchKeywords": ["open mic", "open mic next week", "comedy", "poetry", "storytelling"],
        "hash_tags": ["OPEN_MIC", "NEXT_WEEK"],
    },
    {
        "title": "BLR BREWING ACOUSTIC & STANDUP OPEN MIC - BENGALURU",
        "persons": ["Local Artists", "Standup Newcomers"],
        "languages": ["English", "Hindi"],
        "duration": 120,
        "ageLimit": 16,
        "ageLimitDisplay": "16yrs +",
        "showtime": datetime(2026, 10, 16, 19, 30, tzinfo=IST),  # Target: Next week (Fri Oct 16)
        "city": "Bengaluru",
        "searchKeywords": ["open mic", "open mic next week", "standup", "acoustic music"],
        "hash_tags": ["OPEN_MIC", "NEXT_WEEK"],
    },

    # Query 2: "stand up comedy tonight in pune" (Tonight = 2026-10-07 in Pune)
    {
        "title": "AAKASH GUPTA STAND UP COMEDY SPECIAL - PUNE",
        "persons": ["Aakash Gupta"],
        "languages": ["Hindi", "English"],
        "duration": 90,
        "ageLimit": 18,
        "ageLimitDisplay": "18yrs +",
        "showtime": datetime(2026, 10, 7, 20, 30, tzinfo=IST),  # Target: Tonight (Oct 7) in Pune
        "city": "Pune",
        "searchKeywords": ["stand up comedy", "standup comedy", "comedy", "tonight"],
        "hash_tags": ["TONIGHT", "STANDUP_COMEDY"],
    },
    {
        "title": "BASSI KISI KO BATANA MAT STAND UP COMEDY - PUNE",
        "persons": ["Anubhav Singh Bassi"],
        "languages": ["Hindi"],
        "duration": 100,
        "ageLimit": 16,
        "ageLimitDisplay": "16yrs +",
        "showtime": datetime(2026, 10, 7, 21, 0, tzinfo=IST),  # Target: Tonight (Oct 7) in Pune
        "city": "Pune",
        "searchKeywords": ["stand up comedy", "standup comedy", "comedy", "tonight"],
        "hash_tags": ["TONIGHT", "STANDUP_COMEDY"],
    },

    # Query 4: "kids events this sunday morning" (Sunday morning = 2026-10-11 06:00-12:00)
    {
        "title": "LITTLE EINSTEINS KIDS MAGIC & PUPPET CARNIVAL - MUMBAI",
        "persons": ["Magician Zenia", "Puppet Theatre Troupe"],
        "languages": ["English", "Hindi"],
        "duration": 120,
        "ageLimit": 3,
        "ageLimitDisplay": "All Ages",
        "showtime": datetime(2026, 10, 11, 9, 30, tzinfo=IST),  # Target: Sunday Oct 11 at 9:30 AM
        "city": "Mumbai",
        "searchKeywords": ["events", "kids", "kids events", "children", "sunday morning", "magic show"],
        "hash_tags": ["KIDS_EVENT", "MORNING_SHOW", "SUNDAY_MORNING"],
    },
    {
        "title": "JUNIOR ROBOTICS & SCIENCE KIDS WORKSHOP - BENGALURU",
        "persons": ["Science Educators"],
        "languages": ["English"],
        "duration": 150,
        "ageLimit": 5,
        "ageLimitDisplay": "5yrs +",
        "showtime": datetime(2026, 10, 11, 10, 0, tzinfo=IST),  # Target: Sunday Oct 11 at 10:00 AM
        "city": "Bengaluru",
        "searchKeywords": ["events", "kids", "kids events", "children", "sunday morning", "workshop"],
        "hash_tags": ["KIDS_EVENT", "MORNING_SHOW", "SUNDAY_MORNING"],
    },

    # Query 6: "garba nights between 10th and 20th october" (2026-10-10 to 2026-10-20)
    {
        "title": "SHARAD POORNIMA GARBA NIGHTS FINALE - DELHI NCR",
        "persons": ["Osman Mir"],
        "languages": ["Gujarati", "Hindi"],
        "duration": 210,
        "ageLimit": 5,
        "ageLimitDisplay": "All Ages",
        "showtime": datetime(2026, 10, 28, 20, 0, tzinfo=IST),  # Distractor: Oct 28 (outside 10th-20th)
        "city": "Delhi NCR",
        "searchKeywords": ["garba", "garba night", "garba nights", "dandiya", "october"],
        "hash_tags": ["GARBA_NIGHT"],
    },
    {
        "title": "FALGUNI PATHAK NAVRATRI GARBA NIGHTS - MUMBAI",
        "persons": ["Falguni Pathak", "Ta Thaiya Group"],
        "languages": ["Gujarati", "Hindi"],
        "duration": 240,
        "ageLimit": 5,
        "ageLimitDisplay": "All Ages",
        "showtime": datetime(2026, 10, 12, 20, 0, tzinfo=IST),  # Target: Oct 12 (in Oct 10-20 window)
        "city": "Mumbai",
        "searchKeywords": ["garba", "garba night", "garba nights", "dandiya", "navratri", "october"],
        "hash_tags": ["GARBA_NIGHT", "NAVRATRI"],
    },
    {
        "title": "ROYAL RAAS GARBA & DANDIYA NIGHTS - PUNE",
        "persons": ["Kinjal Dave", "DJ Chetas"],
        "languages": ["Gujarati", "Hindi"],
        "duration": 240,
        "ageLimit": 5,
        "ageLimitDisplay": "All Ages",
        "showtime": datetime(2026, 10, 17, 19, 30, tzinfo=IST),  # Target: Oct 17 (in Oct 10-20 window)
        "city": "Pune",
        "searchKeywords": ["garba", "garba night", "garba nights", "dandiya", "navratri", "october"],
        "hash_tags": ["GARBA_NIGHT", "NAVRATRI"],
    },

    # Query 7: "new year eve parties in goa" (2026-12-31 in Goa)
    {
        "title": "MARINE DRIVE ROOFTOP NEW YEAR EVE PARTY - MUMBAI",
        "persons": ["DJ Aqeel"],
        "languages": ["Hindi", "English"],
        "duration": 360,
        "ageLimit": 21,
        "ageLimitDisplay": "21yrs +",
        "showtime": datetime(2026, 12, 31, 21, 0, tzinfo=IST),  # Distractor: NYE in Mumbai, not Goa
        "city": "Mumbai",
        "searchKeywords": ["new year eve", "new year eve parties", "party", "nye"],
        "hash_tags": ["NEW_YEAR_EVE", "PARTY"],
    },
    {
        "title": "VAGATOR BEACH NEW YEAR EVE MEGA PARTY - GOA",
        "persons": ["DJ Snake", "Nucleya", "Anjunabeats"],
        "languages": ["English", "Hindi"],
        "duration": 420,
        "ageLimit": 18,
        "ageLimitDisplay": "18yrs +",
        "showtime": datetime(2026, 12, 31, 20, 0, tzinfo=IST),  # Target: NYE in Goa
        "city": "Goa",
        "searchKeywords": ["new year eve", "new year eve parties", "party", "nye", "beach party"],
        "hash_tags": ["NEW_YEAR_EVE", "PARTY"],
    },
    {
        "title": "THALASSA SUNSET TO SUNRISE NEW YEAR EVE PARTY - GOA",
        "persons": ["International DJs", "Fire Dancers"],
        "languages": ["English"],
        "duration": 480,
        "ageLimit": 21,
        "ageLimitDisplay": "21yrs +",
        "showtime": datetime(2026, 12, 31, 19, 0, tzinfo=IST),  # Target: NYE in Goa
        "city": "Goa",
        "searchKeywords": ["new year eve", "new year eve parties", "party", "nye"],
        "hash_tags": ["NEW_YEAR_EVE", "PARTY"],
    },

    # Query 8: "is weekend delhi mein kya chal raha hai" (This weekend Oct 10-11, 2026 in Delhi NCR)
    {
        "title": "DELHI SUFI & STREET FOOD FESTIVAL - DELHI NCR",
        "persons": ["Nizami Brothers", "Bismil"],
        "languages": ["Hindi", "Urdu", "Punjabi"],
        "duration": 300,
        "ageLimit": None,
        "ageLimitDisplay": "All Ages",
        "showtime": datetime(2026, 10, 10, 18, 0, tzinfo=IST),  # Target: Saturday Oct 10 in Delhi NCR
        "city": "Delhi NCR",
        "searchKeywords": ["delhi", "weekend", "events", "is weekend delhi mein kya chal raha hai", "sufi", "festival"],
        "hash_tags": ["THIS_WEEKEND", "DELHI_EVENTS"],
    },
    {
        "title": "JASHN-E-DILLI CARNIVAL & COMEDY NIGHT - DELHI NCR",
        "persons": ["Gaurav Kapoor", "Indian Ocean"],
        "languages": ["Hindi", "English"],
        "duration": 240,
        "ageLimit": 12,
        "ageLimitDisplay": "12yrs +",
        "showtime": datetime(2026, 10, 11, 17, 30, tzinfo=IST),  # Target: Sunday Oct 11 in Delhi NCR
        "city": "Delhi NCR",
        "searchKeywords": ["delhi", "weekend", "events", "is weekend delhi mein kya chal raha hai", "carnival"],
        "hash_tags": ["THIS_WEEKEND", "DELHI_EVENTS"],
    },

    # Query 11: "plays on gandhi jayanti holiday" (2nd October = 2026-10-02)
    {
        "title": "ANDHA YUG: CLASSIC HINDI STAGE PLAY - MUMBAI",
        "persons": ["Prithvi Theatre Ensemble"],
        "languages": ["Hindi"],
        "duration": 140,
        "ageLimit": 12,
        "ageLimitDisplay": "12yrs +",
        "showtime": datetime(2026, 10, 25, 19, 0, tzinfo=IST),  # Distractor: Oct 25, not Oct 2
        "city": "Mumbai",
        "searchKeywords": ["play", "plays", "theatre", "drama", "gandhi jayanti", "holiday"],
        "hash_tags": ["STAGE_PLAY"],
    },
    {
        "title": "MAHATMA: THE STAGE PLAY - DELHI NCR",
        "persons": ["National School of Drama Repertory"],
        "languages": ["Hindi", "English"],
        "duration": 135,
        "ageLimit": 5,
        "ageLimitDisplay": "All Ages",
        "showtime": datetime(2026, 10, 2, 18, 0, tzinfo=IST),  # Target: Oct 2 (Gandhi Jayanti)
        "city": "Delhi NCR",
        "searchKeywords": ["play", "plays", "theatre", "drama", "gandhi jayanti", "holiday"],
        "hash_tags": ["STAGE_PLAY", "HOLIDAY_SPECIAL"],
    },
    {
        "title": "TUGHLAQ: THEATRE PLAY - MUMBAI",
        "persons": ["Alyque Padamsee Theatre Group"],
        "languages": ["Hindi"],
        "duration": 150,
        "ageLimit": 12,
        "ageLimitDisplay": "12yrs +",
        "showtime": datetime(2026, 10, 2, 15, 30, tzinfo=IST),  # Target: Oct 2 (Gandhi Jayanti)
        "city": "Mumbai",
        "searchKeywords": ["play", "plays", "theatre", "drama", "gandhi jayanti", "holiday"],
        "hash_tags": ["STAGE_PLAY", "HOLIDAY_SPECIAL"],
    },

    # Query 12: "pottery classes every saturday in october" (Saturdays in Oct: Oct 3, 10, 17, 24, 31)
    {
        "title": "PUNE CLAY COLLECTIVE POTTERY CLASSES - PUNE",
        "persons": ["Pune Clay Collective"],
        "languages": ["English", "Marathi"],
        "duration": 120,
        "ageLimit": 8,
        "ageLimitDisplay": "8yrs +",
        "showtime": datetime(2026, 11, 11, 16, 0, tzinfo=IST),  # Distractor: Wednesday in November
        "city": "Pune",
        "searchKeywords": ["pottery", "pottery classes", "pottery workshop", "clay", "saturday", "october"],
        "hash_tags": ["POTTERY_WORKSHOP"],
    },
    {
        "title": "CLAYSTATION WHEEL POTTERY CLASSES & CERAMICS WORKSHOP - BENGALURU",
        "persons": ["ClayStation Master Potters"],
        "languages": ["English", "Hindi"],
        "duration": 150,
        "ageLimit": 8,
        "ageLimitDisplay": "8yrs +",
        "showtime": datetime(2026, 10, 10, 11, 0, tzinfo=IST),  # Target: Saturday Oct 10
        "city": "Bengaluru",
        "searchKeywords": ["pottery", "pottery classes", "pottery workshop", "clay", "ceramics", "saturday", "october"],
        "hash_tags": ["POTTERY_WORKSHOP", "SATURDAY_OCTOBER"],
    },
    {
        "title": "BANDRA STUDIO HAND-BUILDING POTTERY CLASSES - MUMBAI",
        "persons": ["Bandra Studio Potters"],
        "languages": ["English", "Hindi"],
        "duration": 120,
        "ageLimit": 8,
        "ageLimitDisplay": "8yrs +",
        "showtime": datetime(2026, 10, 17, 15, 0, tzinfo=IST),  # Target: Saturday Oct 17
        "city": "Mumbai",
        "searchKeywords": ["pottery", "pottery classes", "pottery workshop", "clay", "saturday", "october"],
        "hash_tags": ["POTTERY_WORKSHOP", "SATURDAY_OCTOBER"],
    },
]

for te_idx, te in enumerate(TARGETED_EVENTS):
    group_code = f"EG{50000100 + te_idx:08d}"
    start = te["showtime"]
    end = start + timedelta(minutes=te["duration"])
    release = TODAY - timedelta(days=10)
    events_data.append({
        "_id": f"etm{100000 + rec:06d}z",
        "ageLimit": te["ageLimit"],
        "ageLimitDisplay": te["ageLimitDisplay"],
        "censorRating": None,
        "createdAt": iso(release),
        "displayRegions": None,
        "duration": te["duration"],
        "eventGroupCode": group_code,
        "format": None,
        "images": None,
        "isActive": True,
        "isDefault": True,
        "isGlobal": False,
        "isSearchable": True,
        "isTentativeRelease": False,
        "isWebView": True,
        "languages": te["languages"],
        "persons": te["persons"],
        "release_date": iso(release),
        "showtime_start": iso(start),
        "showtime_end": iso(end),
        "searchKeywords": te["searchKeywords"],
        "hash_tags": te["hash_tags"],
        "slug": slugify(te["title"]),
        "subTitleLanguages": [],
        "tentativeBy": None,
        "title": te["title"],
        "trailerUrl": "https://youtube.com/watch?v=eventtrailer",
        **build_location(te["city"], 1),
    })
    rec += 1

output = {"movies": movies_data, "events": events_data}
with open(LOCAL_JSON_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"Generated {len(movies_data)} movies + {len(events_data)} events "
      f"= {len(movies_data) + len(events_data)} records -> {LOCAL_JSON_FILE}")

