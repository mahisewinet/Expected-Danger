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
    page_icon="⚽",
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
def load_matches():
    with open(MATCHES_PATH, "r", encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))
        
def find_match_id(df_matches, team1, team2):
    match = df_matches[
        (
            df_matches["home_team"].apply(
                lambda x: isinstance(x, dict) and x.get("home_team_name") == team1
            ) &
            df_matches["away_team"].apply(
                lambda x: isinstance(x, dict) and x.get("away_team_name") == team2
            )
        ) |
        (
            df_matches["home_team"].apply(
                lambda x: isinstance(x, dict) and x.get("home_team_name") == team2
            ) &
            df_matches["away_team"].apply(
                lambda x: isinstance(x, dict) and x.get("away_team_name") == team1
            )
        )
    ]

    if match.empty:
        st.error(f"No match found for {team1} vs {team2}")
        st.stop()

    return int(match.iloc[0]["match_id"])



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
    fig, ax = pitch.draw(figsize=(6.5, 4))

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

def get_player_lookup(df):
    return (
        df[["player_id", "player_name"]]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )


def plot_messi_comparison(df, messi_id):

    fig, ax = plt.subplots(figsize=(12, 8))

    # Scatter all players
    ax.scatter(
        df["final_third_passes_per90"],
        df["xg_per90"],
        s=90,
        alpha=0.7,
        color="steelblue",
        edgecolors="black",
        linewidth=0.5,
        label="Other players"
    )

    # Highlight Messi
    messi = df[df["player_id"] == messi_id]

    if not messi.empty:
        ax.scatter(
            messi["final_third_passes_per90"],
            messi["xg_per90"],
            s=350,
            marker="*",
            color="gold",
            edgecolors="black",
            linewidth=2,
            label="Lionel Messi",
            zorder=10
        )

        ax.annotate(
            "MESSI",
            (
                messi["final_third_passes_per90"].iloc[0],
                messi["xg_per90"].iloc[0]
            ),
            xytext=(12, 12),
            textcoords="offset points",
            fontsize=13,
            fontweight="bold",
            color="darkred",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="gold",
                alpha=0.8
            )
        )

    # Annotate top 5 xG players (excluding Messi)
    df_others = df[df["player_id"] != messi_id]
    top_xg_players = df_others.nlargest(5, "xg_per90")

    for _, row in top_xg_players.iterrows():
        ax.annotate(
            row["player_name"],
            (row["final_third_passes_per90"], row["xg_per90"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            alpha=0.8
        )

    # Axis labels
    ax.set_xlabel("Final-Third Passes per 90", fontsize=12, fontweight="bold")
    ax.set_ylabel("Expected Goals (xG) per 90", fontsize=12, fontweight="bold")

    # Title
    ax.set_title(
        "Messi vs Comparable Attackers\n"
        "World Cup 2022 — Final-Third Creativity vs Goal Threat (per 90)",
        fontsize=14,
        fontweight="bold",
        pad=20
    )

    # Grid
    ax.grid(True, linestyle="--", alpha=0.3)

    # Reference lines (dataset averages)
    ax.axhline(
        y=df["xg_per90"].mean(),
        color="gray",
        linestyle=":",
        alpha=0.6
    )

    ax.axvline(
        x=df["final_third_passes_per90"].mean(),
        color="gray",
        linestyle=":",
        alpha=0.6
    )

    # Legend
    ax.legend(loc="upper left")

    # Streamlit render
    st.pyplot(fig)
    plt.close(fig)



# ===============================
# APP
# ===============================
def main():

    st.title("Lionel Messi – Attacking Impact at the 2022 World Cup")
    st.markdown(
        "This app analyses Lionel Messi’s **final-third creativity** and "
        "**attacking output** using open-play event data from the 2022 FIFA World Cup."
    )

    # ===============================
    # LOAD DATA
    # ===============================
    df_matches = load_matches()
    df_events_clean = build_clean_events(df_matches)

    # ===============================
    # BUILD PER-90 DATASET
    # ===============================
    df_per90 = build_per90(df_events_clean)
    df_players = get_player_lookup(df_events_clean)
    df_per90_named = df_per90.merge(df_players, on="player_id", how="left")

    # ===============================
    # CONSTANTS
    # ===============================
    MESSI_NAME = "Lionel Andrés Messi Cuccittini"
    MESSI_ID = 5503
    FINAL_THIRD_X = 80

    # ===============================
    # SIDEBAR — MATCH SELECTION
    # ===============================
    st.sidebar.header("Match selection")

    match_options = {
        "Argentina vs Saudi Arabia": ("Argentina", "Saudi Arabia"),
        "Argentina vs Mexico": ("Argentina", "Mexico")
    }

    selected_match_label = st.sidebar.selectbox(
        "Select match",
        options=list(match_options.keys())
    )

    team1, team2 = match_options[selected_match_label]
    match_id = find_match_id(df_matches, team1, team2)

    # ===============================
    # FILTER MESSI PASSES (MATCH-SPECIFIC)
    # ===============================
    df_messi_match = df_events_clean[
        (df_events_clean["match_id"] == match_id) &
        (df_events_clean["player_name"] == MESSI_NAME) &
        (df_events_clean["event_type"] == "Pass") &
        (df_events_clean["x"] >= FINAL_THIRD_X)
    ].dropna(subset=["x", "y", "end_x", "end_y"])

    # ===============================
    # VISUALISATIONS
    # ===============================
    st.subheader("Final-Third Open-Play Passing")

    draw_final_third_pass_map(
        df_messi_match,
        f"Lionel Messi – Final-Third Open-Play Passes\n{selected_match_label} (World Cup 2022)"
    )

    st.subheader("Messi vs Comparable Attackers (Tournament Context)")

    plot_messi_comparison(df_per90_named, MESSI_ID)

if __name__ == "__main__":
    main()















