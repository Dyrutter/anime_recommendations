import os
from bs4 import BeautifulSoup
import requests
import time
from datetime import datetime
import json
from zipfile import ZipFile
import pandas as pd
import shutil
import zipfile
import random
import logging

logging.basicConfig(
    filename='./scrape.log',
    level=logging.INFO,
    filemode='a',
    format='%(asctime)s-%(name)s - %(levelname)s - %(message)s',
    datefmt='%d %b %Y %H:%M:%S %Z',
    force=True)
logger = logging.getLogger()

BASE_PATH = os.getcwd()
HTML_PATH = BASE_PATH + "/html"

if not os.path.isdir(HTML_PATH):
    os.umask(0)
    os.makedirs(HTML_PATH)
else:
    with os.scandir(HTML_PATH) as entries:
        for entry in entries:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry.path)
            else:
                os.remove(entry.path)

USER_PATH = BASE_PATH + "/users"
if not os.path.isdir(USER_PATH):
    os.umask(0)
    os.makedirs(USER_PATH)
else:
    with os.scandir(USER_PATH) as entries:
        for entry in entries:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry.path)
            else:
                os.remove(entry.path)



####### GET CLUB IDS

def get_number(string):
    return int(string.strip().replace(",", ""))


clubs_id = set()
possibles_users = 0
page = 1

while True:
    print(f"\r{page}", end="")

    time.sleep(2)  # Wait 2 seconds per page
    data = requests.get(f"https://myanimelist.net/clubs.php?p={page}")
    soup = BeautifulSoup(data.text, "html.parser")
    rows = soup.find_all("tr", {"class": "table-data"})
    for row in rows:
        members = get_number(row.find("td", {"class": "ac"}).text)
        club_id = get_number(
            row.find("a", {"class": "fw-b"}).get("href").split("=")[-1]
        )
        if (
            club_id not in clubs_id and members > 30
        ):  # Only save groups with more than 30 members
            possibles_users += members
            clubs_id.add(club_id)

    page += 1
    if possibles_users > 1000000:  # Threshold to stop
        break

with open(f"{BASE_PATH}/clubs.txt", "w") as file:
    for club in clubs_id:
        file.write(f"{club}\n")


logger.info("clubs written")

###### GET USERNAME IN EVERY CLUB
if not os.path.exists(f"{BASE_PATH}/users_list.txt"):
    with open(f"{BASE_PATH}/users_list.txt", "w", encoding="UTF-8") as file:
        pass
    
if not os.path.exists(f"{BASE_PATH}/_revised_clubs.txt"):
    with open(f"{BASE_PATH}/_revised_clubs.txt", "w", encoding="UTF-8") as file:
        pass



with open(f"{BASE_PATH}/clubs.txt") as file:
    clubs_id = [x.strip() for x in file.readlines()]

with open(f"{BASE_PATH}/users_list.txt", encoding="UTF-8") as file:
    users = set([x.strip() for x in file.readlines()])

with open(f"{BASE_PATH}/_revised_clubs.txt", encoding="UTF-8") as file:
    revised_clubs = set([int(x.strip()) for x in file.readlines()])

len(users), len(revised_clubs), len(clubs_id)


for i, club_id in enumerate(clubs_id):
    if club_id in revised_clubs:
        continue

    page = 1
    while True:
        print(f"\r{i+1}/{len(clubs_id)} --> {str(page).zfill(2)}", end="")
        link = f"https://api.jikan.moe/v3/club/{club_id}/members/{page}"

        try:
            time.sleep(4.2)
            data = requests.get(link)
        except KeyboardInterrupt:
            raise KeyboardInterrupt()
        except:  # Other exception wait 2 min and try again
            time.sleep(120)
            continue

        if data.status_code != 200:
            break

        with open(f"{BASE_PATH}/users_list.txt", "a", encoding="UTF-8") as file:
            for user in map(lambda x: x["username"], json.loads(data.text)["members"]):
                if user not in users and user != "":
                    file.write(f"{user}\n")
                    users.add(user)
        page += 1

    revised_clubs.add(club_id)
    with open(f"{BASE_PATH}/_revised_clubs.txt", "a", encoding="UTF-8") as file:
        file.write(f"{club_id}\n")

with open(f"{BASE_PATH}/users_list.txt", encoding="UTF-8") as file:
    users = list(set([x.strip() for x in file.readlines()]))[1:]
    random.shuffle(users)

with open(f"{BASE_PATH}/users.csv", "w", encoding="UTF-8") as file:
    file.write("user_id,username\n")
    for i, user in enumerate(users):
        file.write(f"{i},{user}\n")

logger.info("part one complete")
######## GET ANIMELIST PER USER
with open(f"{BASE_PATH}/users.csv", "r", encoding="UTF-8") as file:
    file.readline()
    users = [x.strip().split(",") for x in file.readlines()]
    users = [(int(x[0]), x[1]) for x in users]

last_revised_users = -1
if os.path.exists(f"{BASE_PATH}/_last_revised_users.txt"):
    with open(f"{BASE_PATH}/_last_revised_users.txt", "r", encoding="UTF-8") as file:
        last_revised_users = int(file.readline())

len(users), last_revised_users

for i, (user_id, username) in enumerate(users):
    if user_id <= last_revised_users:
        continue

    now = datetime.now()
    print(f'\r{str(now).split(".")[0]} --> {i+1}/{len(users)}', end="")
    page = 1
    all_animes = []

    while True:
        link = f"https://api.jikan.moe/v3/user/{username}/animelist/all?page={page}"
        try:
            time.sleep(4.2)
            data = requests.get(link, timeout=15)
        except KeyboardInterrupt:
            raise KeyboardInterrupt()
        except:  # Other exception wait 2 min and try again
            time.sleep(120)
            continue

        if data.status_code != 200:
            break

        data = json.loads(data.text)
        for anime in data["anime"]:
            all_animes.append((anime["mal_id"], anime["score"], anime["watching_status"], anime["watched_episodes"]))

        page += 1
        if len(data["anime"]) < 300:
            break

    if len(all_animes) != 0:
        with open(f"{USER_PATH}/{user_id}.csv", "w") as f1:
            f1.write(f"anime_id,score,watching_status,watched_episodes\n")
            for anime_id, anime_score, watching_status, watched_episodes in all_animes:
                f1.write(
                    f"{anime_id},{anime_score},{watching_status},{watched_episodes}\n"
                )

    revised_users = user_id
    with open(f"{BASE_PATH}/_last_revised_users.txt", "w", encoding="UTF-8") as file:
        file.write(f"{user_id}\n")
logger.info("part 2 complete")
######DOWNLOAD ANIME HTML
unique_anime = set()

folder = os.listdir(USER_PATH)
for i, user_file in enumerate(folder):
    if ".csv" not in user_file:
        continue

    print(f"\r{i + 1}/{len(folder)}", end="")
    with open(f"{USER_PATH}/{user_file}", "r") as file:
        file.readline()
        for line in file:
            anime = line.strip().split(",")[0]
            unique_anime.add(anime)

print("         ")
print(len(unique_anime))

MAX = 7  # MAX SECOND TO WAIT PER REQUEST
MIN = 3  # MIN SECONDS TO WAIT PER REQUEST


def sleep():
    time_to_sleep = random.random() * (MAX - MIN) + MIN
    time.sleep(time_to_sleep)


def get_link_by_text(soup, anime_id, text):
    links = list(filter(lambda x: anime_id in x["href"], soup.find_all("a", text=text)))
    return links[0]["href"]


def save(path, data):
    with open(path, "w", encoding="UTF-8") as file:
        file.write(data)


def save_link(link, anime_id, name):
    sleep()
    path = f"{HTML_PATH}/{anime_id}/{name}.html"
    data = requests.get(link)
    soup = BeautifulSoup(data.text, "html.parser")
    soup.script.decompose()
    save(path, soup.prettify())
    return soup


def save_reviews(link, anime_id):
    page = 1
    while True:
        sleep()
        actual_link = f"{link}?p={page}"
        data = requests.get(actual_link)
        soup = BeautifulSoup(data.text, "html.parser")
        reviews = soup.find_all("a", text="Overall Rating")
        if len(reviews) == 0:
            break

        path = f"{HTML_PATH}/{anime_id}/reviews_{page}.html"
        soup.script.decompose()
        save(path, soup.prettify())
        page += 1

def scrap_anime(anime_id):
    path = f"{HTML_PATH}/{anime_id}"
    os.makedirs(path, exist_ok=True)
    sleep()
    data = requests.get(f"https://myanimelist.net/anime/{anime_id}")

    anime_info = data.text
    soup = BeautifulSoup(anime_info, "html.parser")
    soup.script.decompose()
    save(f"{HTML_PATH}/{anime_id}/details.html", soup.prettify())

    link_review = get_link_by_text(soup, anime_id, "Reviews")
    link_recomendations = get_link_by_text(soup, anime_id, "Recommendations")
    link_stats = get_link_by_text(soup, anime_id, "Stats")
    link_staff = get_link_by_text(soup, anime_id, "Characters & Staff")
    link_pictures = get_link_by_text(soup, anime_id, "Pictures")

    save_link(link_pictures, anime_id, "pictures")
    save_link(link_staff, anime_id, "staff")
    save_link(link_stats, anime_id, "stats")
    save_link(link_recomendations, anime_id, "recomendations")
    save_reviews(link_review, anime_id)


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(
                os.path.join(root, file),
                os.path.relpath(os.path.join(root, file), path),
            )

for i, anime_id in enumerate(unique_anime):
    if os.path.isfile(f"{HTML_PATH}/{anime_id}.zip"):
        continue

    print(f"\r{i+1}/{len(unique_anime)}", end="")

    try:
        scrap_anime(anime_id)
    except KeyboardInterrupt:
        break
    except:  # Other exception wait 2 min and try again
        time.sleep(120)
        continue

    path = f"{HTML_PATH}/{anime_id}"
    zipf = zipfile.ZipFile(f"{path}.zip", "w", zipfile.ZIP_DEFLATED)
    zipdir(path, zipf)
    zipf.close()

    shutil.rmtree(path)

logger.info("part 4 complete")

######### SCRAPING ANIME INFO FROM HTML

def extract_zip(input_zip):
    input_zip = ZipFile(input_zip)
    return {name: input_zip.read(name) for name in input_zip.namelist()}

KEYS = ['MAL_ID', 'Name', 'Score', 'Genders', 'English name', 'Japanese name', 'Type', 'Episodes',
        'Aired', 'Premiered', 'Producers', 'Licensors', 'Studios', 'Source', 'Duration', 'Rating',
        'Ranked', 'Popularity', 'Members', 'Favorites', 'Watching', 'Completed', 'On-Hold', 'Dropped',
        'Plan to Watch', 'Score-10', 'Score-9', 'Score-8', 'Score-7', 'Score-6', 'Score-5', 'Score-4',
        'Score-3', 'Score-2', 'Score-1']

def get_name(info):
    return info.find("h1", {"class": "title-name h1_bold_none"}).text.strip()


def get_english_name(info):
    span = info.findAll("span", {"class": "dark_text"})
    return span.parent.text.strip()


def get_table(a_soup):
    return a_soup.find("div", {"class": "po-r js-statistics-info di-ib"})


def get_score(stats):
    score = stats.find("span", {"itemprop": "ratingValue"})
    if score is None:
        return "Unknown"
    return score.text.strip()


def get_gender(sum_info):
    text = ", ".join(
        [x.text.strip() for x in sum_info.findAll("span", {"itemprop": "genre"})]
    )
    return text


def get_description(sum_info):
    return sum_info.find("td", {"class": "borderClass", "width": "225"})


def get_all_stats(soup):
    return soup.find("div", {"id": "horiznav_nav"}).parent.findAll(
        "div", {"class": "spaceit_pad"}
    )
def get_info_anime(anime_id):
    data = extract_zip(f"data/html/{anime_id}.zip")
    anime_info = data["stats.html"].decode()
    soup = BeautifulSoup(anime_info, "html.parser")

    stats = get_table(soup)
    description = get_description(soup)
    anime_info = {key: "Unknown" for key in KEYS}

    anime_info["MAL_ID"] = anime_id
    anime_info["Name"] = get_name(soup)
    anime_info["Score"] = get_score(stats)
    anime_info["Genders"] = get_gender(description)

    for d in description.findAll("span", {"class": "dark_text"}):
        information = [x.strip().replace(" ", " ") for x in d.parent.text.split(":")]
        category, value = information[0], ":".join(information[1:])
        value.replace("\t", "")

        if category in ["Broadcast", "Synonyms", "Genres", "Score", "Status"]:
            continue

        if category in ["Ranked"]:
            value = value.split("\n")[0]
        if category in ["Producers", "Licensors", "Studios"]:
            value = ", ".join([x.strip() for x in value.split(",")])
        if category in ["Ranked", "Popularity"]:
            value = value.replace("#", "")
        if category in ["Members", "Favorites"]:
            value = value.replace(",", "")
        if category in ["English", "Japanese"]:
            category += " name"

        anime_info[category] = value

    # Stats (Watching, Completed, On-Hold, Dropped, Plan to Watch)
    for d in get_all_stats(soup)[:5]:
        category, value = [x.strip().replace(" ", " ") for x in d.text.split(":")]
        value = value.replace(",", "")
        anime_info[category] = value

    # Stast votes per score
    for d in get_all_stats(soup)[6:]:
        score = d.parent.parent.find("td", {"class": "score-label"}).text.strip()
        value = [x.strip().replace(" ", " ") for x in d.text.split("%")][1].strip(
            "(votes)"
        )
        label = f"Score-{score}"
        anime_info[label] = value.strip()

    for key, value in anime_info.items():
        if str(value) in ["?", "None found, add some", "None", "N/A", "Not available"]:
            anime_info[key] = "Unknown"
    return anime_info

# Generate anime.tsv
anime_revised = set()
filename = 'anime.tsv'
file_to_create = os.path.join(BASE_PATH, filename) # Adds strings of current directory and name of new file
if not os.path.exists(file_to_create): # checks if file already exists
    with open(file_to_create, 'w', encoding='utf-8') as f:
        f.write(filename)
exist_file = os.path.exists(f"{BASE_PATH}/anime.tsv")
if exist_file:
    # If the file exist, include new data.
    actual_data = pd.read_csv(f"{BASE_PATH}/anime.tsv", sep="\t")
    anime_revised = list(actual_data.MAL_ID.unique())

# actual_data.head()
total_data = []
zips = os.listdir(HTML_PATH)
for i, anime in enumerate(zips):
    if not ".zip" in anime:
        continue

    anime_id = int(anime.strip(".zip"))

    if int(anime_id) in anime_revised:
        continue

    print(f"\r{i+1}/{len(zips)} ({anime_id})", end="")

    anime_id = anime.strip(".zip")
    total_data.append(get_info_anime(anime_id))

if len(total_data):
    df = pd.DataFrame.from_dict(total_data)
    df["MAL_ID"] = pd.to_numeric(df["MAL_ID"])
    df = df.sort_values(by="MAL_ID").reset_index(drop=True)

    if exist_file:
        df = (
            pd.concat([actual_data, df]).sort_values(by="MAL_ID").reset_index(drop=True)
        )

else:
    df = actual_data

pd.set_option("display.max_columns", None)
df.head()


df.to_csv(f"{BASE_PATH}/anime.tsv", index=False, sep="\t", encoding="UTF-8")

logger.info("Part 5 complete")
##### FULL ANIMELIST
if not os.path.exists(f"{BASE_PATH}/animelist.csv"):
    with open(f"{BASE_PATH}/animelist.csv", "w", encoding="UTF-8") as file:
        file.write("user_id,anime_id,rating,watching_status,watched_episodes\n")
unique_anime = set()
all_users = sorted(os.listdir(USER_PATH), key=lambda x:int(x.split(".")[0]))

with open(f"{BASE_PATH}/animelist.csv", "a") as f1:

    for i, user_file in enumerate(all_users):
        if not user_file.endswith(".csv"):
            continue

        print(f"\r{i+1}/{len(all_users)}", end="")

        user_id = user_file.split(".")[0]
        with open(f"{USER_PATH}/{user_file}", "r") as file:
            file.readline()
            for line in file:
                anime_id, score, watching_status, watched_episodes = line.strip().split(",")
                f1.write(f"{user_id},{anime_id},{score},{watching_status},{watched_episodes}\n")

logger.info("All files created")       