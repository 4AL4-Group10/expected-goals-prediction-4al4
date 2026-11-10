# data_processing.py
#
# you can run with:  python data_processing.py
# Turns StatsBomb match files into one table where each row is a shot.
# result will be saved to data/processed/shots.csv (so it can be used for training)

import glob
import json
import math
import os

import pandas as pd

# Where raw data is and where processed file goes
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "..", "data", "open-data", "data")
OUT_DIR = os.path.join(HERE, "..", "data", "processed")

# StatsBomb pitch is 120x80 yards and team shooting always attacks towards x = 120
GOAL_CENTER = (120, 40)
LEFT_POST = (120, 36)
RIGHT_POST = (120, 44)

# These true/false shot details only show up in data when theyre true, so if 1 is missing we treat it false (0).
SHOT_FLAGS = ["first_time", "aerial_won", "one_on_one", "open_goal", "follows_dribble"]

# Columns the model will use as inputs
NUMBER_COLUMNS = ["x", "y", "distance", "angle", "minute",
                  "defenders_in_way", "nearest_opponent_dist", "keeper_dist_to_goal"]
FLAG_COLUMNS = SHOT_FLAGS + ["under_pressure", "has_freeze_frame"]

# Every option each text column can take (found by looking through all the matches)
CATEGORY_OPTIONS = {
    "shot_type": ["Open Play", "Free Kick", "Penalty", "Corner", "Kick Off"],
    "body_part": ["Right Foot", "Left Foot", "Head", "Other"],
    "technique": ["Normal", "Half Volley", "Volley", "Lob", "Overhead Kick",
                  "Backheel", "Diving Header"],
    "play_pattern": ["Regular Play", "From Free Kick", "From Throw In", "From Corner",
                     "From Counter", "From Goal Kick", "From Keeper", "From Kick Off", "Other"],
}


def distance(a, b):
    # Straight line distance between 2points
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def shot_angle(x, y):
    # How wide goal looks from where shot was taken
    # We get direction to each post and angle is the gap between them
    to_left = math.atan2(LEFT_POST[1] - y, LEFT_POST[0] - x)
    to_right = math.atan2(RIGHT_POST[1] - y, RIGHT_POST[0] - x)
    return abs(to_right - to_left)


def side_of_line(start, end, point):
    # Positive if point is on one side of the line negative if its on other and 0 if it sits right on the line
    return (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0])


def is_in_the_way(player, shooter):
    # Checks if a player is inside triangle made by the shooter and the two posts
    # A point is inside a triangle if it's on the same side of all three edges.
    edge1 = side_of_line(shooter, LEFT_POST, player)
    edge2 = side_of_line(LEFT_POST, RIGHT_POST, player)
    edge3 = side_of_line(RIGHT_POST, shooter, player)

    all_positive = edge1 >= 0 and edge2 >= 0 and edge3 >= 0
    all_negative = edge1 <= 0 and edge2 <= 0 and edge3 <= 0
    return all_positive or all_negative


def defensive_features(shooter, freeze_frame):
    # freeze frame=  list of nearby players and where they were standing
    # when shot was taken. We turn it into a few numbers the model can use

    # Penalties don't have a freeze frame so will fill them in like a normal penalty with no one in the way and keeper on goal line
    if not freeze_frame:
        return {"defenders_in_way": 0, "nearest_opponent_dist": 30,
                "keeper_dist_to_goal": 0, "has_freeze_frame": 0}

    defenders = []
    keeper = None
    for player in freeze_frame:
        if player["teammate"]:
            continue
        if player["position"]["name"] == "Goalkeeper":
            keeper = player["location"]
        else:
            defenders.append(player["location"])

    defenders_in_way = 0
    for d in defenders:
        if is_in_the_way(d, shooter):
            defenders_in_way += 1

    # If there are no defenders in the frame act like the closest one is far away
    if defenders:
        nearest = min(distance(shooter, d) for d in defenders)
    else:
        nearest = 30

    # found via audit some few frames don't show the keeper, so we going to assume they're on goal line
    if keeper:
        keeper_dist = distance(GOAL_CENTER, keeper)
    else:
        keeper_dist = 0

    return {"defenders_in_way": defenders_in_way, "nearest_opponent_dist": nearest,
            "keeper_dist_to_goal": keeper_dist, "has_freeze_frame": 1}


def get_shots(path):
    # Reads 1 match file and returns a row for every shot in it
    # The match id is the file name (eg. <#>.json)
    match_id = int(os.path.basename(path).replace(".json", ""))
    with open(path, encoding="utf-8") as f:
        events = json.load(f)

    rows = []
    for event in events:
        if event["type"]["name"] != "Shot":
            continue

        shot = event["shot"]
        x, y = event["location"][0], event["location"][1]

        row = {
            "match_id": match_id,
            "event_id": event["id"],
            "period": event["period"],
            "minute": event["minute"],
            "x": x,
            "y": y,
            "distance": distance((x, y), GOAL_CENTER),
            "angle": shot_angle(x, y),
            "shot_type": shot["type"]["name"],
            "body_part": shot["body_part"]["name"],
            "technique": shot["technique"]["name"],
            "play_pattern": event["play_pattern"]["name"],
            # under_pressure is stored on the event itself, not inside the shot since its an event not an aspect of the shot
            "under_pressure": int(event.get("under_pressure", False)),
        }
        for flag in SHOT_FLAGS:
            row[flag] = int(shot.get(flag, False))

        row.update(defensive_features((x, y), shot.get("freeze_frame")))

        # only known after shot is taken, so they are never used as inputs
        # is_goal is our label and statsbomb_xg is only kept to compare our model against
        row["outcome"] = shot["outcome"]["name"]
        row["is_goal"] = int(shot["outcome"]["name"] == "Goal")
        row["statsbomb_xg"] = shot["statsbomb_xg"]

        rows.append(row)
    return rows


def get_match_info():
    # Competition and mens/womens info is kept in a separate matches folder
    # returns lookup of match id to (competition, gender)
    info = {}
    for path in glob.glob(os.path.join(DATA_ROOT, "matches", "*", "*.json")):
        with open(path, encoding="utf-8") as f:
            for match in json.load(f):
                info[match["match_id"]] = (match["competition"]["competition_name"],
                                           match["home_team"]["home_team_gender"])
    return info


def build_feature_matrix(df):
    # gives back model inputs (X) and labels (y)
    # Text columns are turned into 0/1 columns, 1 for each option,
    # eg. body_part becomes body_part_Head, body_part_Left Foot, and so on.
    X = df[NUMBER_COLUMNS + FLAG_COLUMNS].astype(float)
    for column, options in CATEGORY_OPTIONS.items():
        for option in options:
            X[f"{column}_{option}"] = (df[column] == option).astype(float)
    y = df["is_goal"]
    return X, y


def main():
    files = sorted(glob.glob(os.path.join(DATA_ROOT, "events", "*.json")))
    print(f"Reading {len(files)} matches...")

    rows = []
    for path in files:
        rows.extend(get_shots(path))
    df = pd.DataFrame(rows)
    shots_raw = len(df)

    # Period 5 is a penalty shootout, which isn't normal play and doesn't count to the score
    shootout = df["period"] == 5
    # A few shots have locations that are off the pitch??? which has to be a recording error
    off_pitch = ~df["x"].between(0, 120) | ~df["y"].between(0, 80)
    df = df[~shootout & ~off_pitch]
    df = df.drop_duplicates("event_id")

    # add competition and gender for each match. giving unknown to the matches that arent listed as some arent listed
    match_info = get_match_info()
    df["competition"] = [match_info.get(m, ("Unknown", "unknown"))[0] for m in df["match_id"]]
    df["gender"] = [match_info.get(m, ("Unknown", "unknown"))[1] for m in df["match_id"]]

    # saving summary of what was removed so numbers in report can be checked
    report = {
        "matches": len(files),
        "shots_raw": shots_raw,
        "dropped_shootout": int(shootout.sum()),
        "dropped_off_pitch": int((off_pitch & ~shootout).sum()),
        "no_freeze_frame": int((df["has_freeze_frame"] == 0).sum()),
        "shots_final": len(df),
        "goals_final": int(df["is_goal"].sum()),
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "shots.csv"), index=False)
    with open(os.path.join(OUT_DIR, "pipeline_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()



# AI Disclosure
# 1. 1 query used to understand how the StatsBomb data is organised and what a shot event contains
# 2. 3 queries used to understand what format the data needs to be in for model training and also understanding and figuring out model features and inputs
# 3. 1 query used to help explore the data and find unusual or missing values
# 4. 2 queries used to help plan and understand how to break the preprocessing into stages
# 5. 1 query used to understand how to organise the script so teammates can import it for implementating and evaluation
# 6. 2 queries used to help with reading a match file and pulling out the shots and labels
# 7. 2 queries used to understand how to calculate shot distance and angle
# 8. 2 queries used to understand how to check whether a defender is between the shooter and goal
# 9. 2 queries used to help turn the freeze frame data into useful features
# 10. 1 query used to understand how to handle missing values without leaking info between splits
# 11. 1 query used to help add competition and gender information from the matches folder
# 12. 2 queries used to help with combining all the matches, removing bad shots, and saving the results
# 13. 2 queries used to understand one-hot encoding and how to keep the same columns across splits
# 14. 1 query used to understand data leakage and check which fields shouldn't be model inputs
# 15. 1 query used to help check that the output was correct

# 24 queries * 4.32 g c02 emission = 103.68g c02 emission