# app.py
# Messi World Cup 2022 – Streamlit App

import streamlit as st
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mplsoccer import Pitch
from tqdm import tqdm

# ===============================
# CONFIG
# ===============================
st.set_page_config(
    page_title="Messi World Cup 2022 Analysis",
    layout="wide"
)

DATA_PATH = "data/statsbomb_wc2022"
MATCHES_PATH = os.path.join(DATA_PATH, "matches/wc2022_matches.json")
EVENTS_DIR = os.path.join(DATA_PATH, "events")

MESSI_ID = 5503
MESSI_NAME = "Lionel Andrés Messi Cuccittini"
FINAL_THIRD_X = 80

# ===============================
# DATA LOADING (CACHED)
# ===============================
@st.cache_data(show_spinner=True)
def load_matches():
    with open(MATCHES_PATH, "r", encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))


@st.cache_data(show_spinner=True)
def build_clean_events(df_matches):
    OPEN_PLAY_EXCLUDE = ["Corner", "Free Kick", "Throw-in", "Kick Off"]
    event_rows = []

    for match_id in tqdm(df_matches["match_id"]):
        with open(os.path.join(EVENTS_DIR, f"{match_id}.json"), "r", encoding="utf-8") as f:
            events = json.load(f)

        for e in events:
            event_type = e.get("type", {}).get("name")
            if event_type not in ["Pass", "Shot"]:
                continue

            row = {
                "match_id": match_id,
                "player_id": e.get("player", {}).get("id"),
                "player_name": e.get("player", {}).get("name"),
                "event_type": event_type,
                "x": e.get("location", [None, None])[0],
                "y": e.get("location", [None, None])[1],
                "end_x": None,
                "end_y": None,
                "shot_assist": False,
                "xg": None,
                "minute": e.get("minute")
            }

            if event_type == "Pass":
                p = e.get("pass", {})
                if p.get("type", {}).get("name") in OPEN_PLAY_EXCLUDE:
                    continue
                row["end_x"], row["end_y"] = p.get("end_location", [None, None])
                row["shot_assist"] = p.get("shot_assist", False)

            if event_type == "Shot":
                row["xg"] = e.get("shot", {}).get("statsbomb_xg")

            event_rows.append(row)

    return pd.DataFrame(event_rows)


@st.cache_data(show_spinner=True)
def build_per90(df):
    minutes = (
        df.dropna(subset=["player_id", "minute"])
        .groupby(["match_id", "player_id"])
        .minute.agg(["min", "max"])
        .reset_index()
    )
    minutes["minutes_played"] = (minutes["max"] - minutes["min"]).clip(lower=1)
    minutes = minutes.groupby("player_id")["minutes_played"].sum().reset_index()

    passes = df[(df.event_type == "Pass") & (df.x >= 80)].groupby("player_id").size().reset_index(name="final_third_passes")
    assists = df[(df.event_type == "Pass") & (df.shot_assist)].groupby("player_id").size().reset_index(name="shot_assists")
    xg = df[df.xg.notna()].groupby("player_id")["xg"].sum().reset_index()

    out = minutes.merge(passes, how="left").merge(assists, how="left").merge(xg, how="left").fillna(0)
    out = out[out.minutes_played >= 300]

    out["final_third_passes_per90"] = out.final_third_passes / out.minutes_played * 90
    out["xg_per90"] = out.xg / out.minutes_played * 90

    return out


# ===============================
# VISUALS
# ===============================
def draw_final_third_pass_map(df_passes, title):
    pitch = Pitch(pitch_type="statsbomb", half=True)
    fig, ax = pitch.draw(figsize=(9, 6))

    ax.add_patch(Rectangle((FINAL_THIRD_X, 0), 120 - FINAL_THIRD_X, 80,
                           facecolor="yellow", alpha=0.25))

    for _, r in df_passes.iterrows():
        pitch.arrows(r.x, r.y, r.end_x, r.end_y,
                     ax=ax,
                     color="red" if r.shot_assist else "blue",
                     width=2,
                     alpha=0.7)

    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)


def plot_messi_comparison(df, messi_id):
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(df.final_third_passes_per90, df.xg_per90,
               alpha=0.7, edgecolors="black")

    messi = df[df.player_id == messi_id]
    ax.scatter(messi.final_third_passes_per90, messi.xg_per90,
               s=300, marker="*", color="gold", edgecolors="black")

    ax.set_xlabel("Final-third passes per 90")
    ax.set_ylabel("xG per 90")
    ax.set_title("Messi vs Attacking Players – World Cup 2022")

    st.pyplot(fig)
    plt.close(fig)


# ===============================
# APP
# ===============================
def main():
    st.title("Lionel Messi – Attacking Influence (World Cup 2022)")

    df_matches = load_matches()
    df_events = build_clean_events(df_matches)
    df_per90 = build_per90(df_events)

    players = df_events[["player_id", "player_name"]].drop_duplicates()
    df_per90 = df_per90.merge(players, on="player_id", how="left")

    # -----------------------
    # SIDEBAR CONTROLS
    # -----------------------
    st.sidebar.header("Controls")

    arg_matches = df_matches[
        df_matches.home_team.apply(lambda x: x["home_team_name"] == "Argentina") |
        df_matches.away_team.apply(lambda x: x["away_team_name"] == "Argentina")
    ].copy()

    arg_matches["label"] = (
        arg_matches.home_team.apply(lambda x: x["home_team_name"]) +
        " vs " +
        arg_matches.away_team.apply(lambda x: x["away_team_name"])
    )

    selected_match = st.sidebar.selectbox(
        "Select Argentina match",
        options=arg_matches.match_id,
        format_func=lambda x: arg_matches[arg_matches.match_id == x]["label"].iloc[0]
    )

    players_in_match = df_events[df_events.match_id == selected_match][["player_id", "player_name"]].drop_duplicates()
    player_dict = dict(zip(players_in_match.player_id, players_in_match.player_name))

    selected_player = st.sidebar.selectbox(
        "Select player",
        options=player_dict.keys(),
        index=list(player_dict.keys()).index(MESSI_ID),
        format_func=lambda x: player_dict[x]
    )

    # -----------------------
    # PASS MAP
    # -----------------------
    df_passes = df_events[
        (df_events.match_id == selected_match) &
        (df_events.player_id == selected_player) &
        (df_events.event_type == "Pass") &
        (df_events.x >= FINAL_THIRD_X)
    ].dropna(subset=["x", "y", "end_x", "end_y"])

    draw_final_third_pass_map(
        df_passes,
        f"{player_dict[selected_player]} – Final-third Open-play Passes"
    )

    st.divider()

    # -----------------------
    # COMPARISON
    # -----------------------
    plot_messi_comparison(df_per90, MESSI_ID)


if __name__ == "__main__":
    main()
