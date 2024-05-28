import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import pandas as pd
import time

# Define retry strategy
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)

# Create a session with retry logic
session = requests.Session()
session.mount("https://", adapter)

# Read clan tags from the clan_members_tags.xlsx file
player_tags_df = pd.read_excel('clan_members_tags.xlsx')

# Create an empty DataFrame to store player battle logs
all_player_battle_logs = pd.DataFrame()

# List of card names
card_names = [
    "Knight", "Archers", "Goblins", "Giant", "P.E.K.K.A", "Minions", "Balloon", "Witch", "Barbarians", "Golem",
    "Skeletons", "Valkyrie", "Skeleton Army", "Bomber", "Musketeer", "Baby Dragon", "Prince", "Wizard",
    "Mini P.E.K.K.A", "Spear Goblins", "Giant Skeleton", "Hog Rider", "Minion Horde", "Ice Wizard", "Royal Giant",
    "Guards", "Princess", "Dark Prince", "Three Musketeers", "Lava Hound", "Ice Spirit", "Fire Spirit", "Miner",
    "Sparky", "Bowler", "Lumberjack", "Battle Ram", "Inferno Dragon", "Ice Golem", "Mega Minion", "Dart Goblin",
    "Goblin Gang", "Electro Wizard", "Elite Barbarians", "Hunter", "Executioner", "Bandit", "Royal Recruits",
    "Night Witch", "Bats", "Royal Ghost", "Ram Rider", "Zappies", "Rascals", "Cannon Cart", "Mega Knight",
    "Skeleton Barrel", "Flying Machine", "Wall Breakers", "Royal Hogs", "Goblin Giant", "Fisherman", "Magic Archer",
    "Electro Dragon", "Firecracker", "Mighty Miner", "Elixir Golem", "Battle Healer", "Skeleton King", "Archer Queen",
    "Golden Knight", "Monk", "Skeleton Dragons", "Mother Witch", "Electro Spirit", "Electro Giant", "Phoenix",
    "Little Prince", "Cannon", "Goblin Hut", "Mortar", "Inferno Tower", "Bomb Tower", "Barbarian Hut", "Tesla",
    "Elixir Collector", "X-Bow", "Tombstone", "Furnace", "Goblin Cage", "Goblin Drill", "Fireball", "Arrows", "Rage",
    "Rocket", "Goblin Barrel", "Freeze", "Mirror", "Lightning", "Zap", "Poison", "Graveyard", "The Log", "Tornado",
    "Clone", "Earthquake", "Barbarian Barrel", "Heal Spirit", "Giant Snowball", "Royal Delivery"
]
support_card_names = [
    "Tower Princess", "Dagger Duchess", "Cannoneer"
]

# Initialize player and opponent card columns
def initialize_card_columns():
    columns = {}
    for card in card_names:
        columns[f'player_{card}'] = 0
        columns[f'opponent_{card}'] = 0

    for card in support_card_names:
        columns[f'player_{card}'] = 0
        columns[f'opponent_{card}'] = 0
    return columns

# Iterate through each player tag
for index, row in player_tags_df.iterrows():
    # Make the API request for player battle logs using the current player tag
    player_tag = row['tag']
    url = f"https://proxy.royaleapi.dev/v1/players/%23{player_tag}/battlelog"
    headers = {
        'Authorization': "Bearer [Your API Key without the brackets]",
        'Accept': "application/json"
    }
    
    try:
        response = session.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
        battle_logs = response.json()
        print(index)
        for battle in battle_logs:
            # Check if starting trophies are empty, if true skip
            if battle['team'][0].get('startingTrophies') is None or battle['opponent'][0].get('startingTrophies') is None:
                continue
            # Check if support card IDs are empty or if there are more than 8 cards in the player's or opponent's deck
            if len(battle['team'][0]['supportCards']) == 0:
                continue
            if len(battle['team'][0]['cards']) > 8:
                continue

            # Prepare player and opponent tower hit points
            player_left_tower_hp = battle['team'][0]['princessTowersHitPoints'][0] if battle['team'][0]['princessTowersHitPoints'] else 0
            player_right_tower_hp = battle['team'][0]['princessTowersHitPoints'][1] if battle['team'][0]['princessTowersHitPoints'] and len(battle['team'][0]['princessTowersHitPoints']) > 1 else 0
            opponent_left_tower_hp = battle['opponent'][0]['princessTowersHitPoints'][0] if battle['opponent'][0]['princessTowersHitPoints'] else 0
            opponent_right_tower_hp = battle['opponent'][0]['princessTowersHitPoints'][1] if battle['opponent'][0]['princessTowersHitPoints'] and len(battle['opponent'][0]['princessTowersHitPoints']) > 1 else 0
            
            player_data = {
                'player_crowns': battle['team'][0]['crowns'],
                'opponent_crowns': battle['opponent'][0]['crowns'],
                'player_startingTrophies': battle['team'][0].get('startingTrophies', None),
                'opponent_startingTrophies': battle['opponent'][0].get('startingTrophies', None),
                'player_kingTowerHitPoints': battle['team'][0].get('kingTowerHitPoints', None),
                'player_leftPrincessTowerHitPoints': player_left_tower_hp,
                'player_rightPrincessTowerHitPoints': player_right_tower_hp,
                'opponent_kingTowerHitPoints': battle['opponent'][0].get('kingTowerHitPoints', None),
                'opponent_leftPrincessTowerHitPoints': opponent_left_tower_hp,
                'opponent_rightPrincessTowerHitPoints': opponent_right_tower_hp,
                'player_elixirLeaked': battle['team'][0].get('elixirLeaked', None),
                'opponent_elixirLeaked': battle['opponent'][0].get('elixirLeaked', None),
            }

            # Initialize card columns
            player_data.update(initialize_card_columns())

            # Extracting card information for the player and opponent
            for card in battle['team'][0]['cards']:
                card_name = next((name for name in card_names if name.lower() in card['name'].lower()), None)
                if card_name:
                    player_data[f'player_{card_name}'] = card['level']

            for card in battle['opponent'][0]['cards']:
                card_name = next((name for name in card_names if name.lower() in card['name'].lower()), None)
                if card_name:
                    player_data[f'opponent_{card_name}'] = card['level']

            for card in battle['team'][0]['supportCards']:
                support_card_name = next((name for name in support_card_names if name.lower() in card['name'].lower()), None)
                if support_card_name:
                    player_data[f'player_{support_card_name}'] = card['level']

            for card in battle['opponent'][0]['supportCards']:
                support_card_name = next((name for name in support_card_names if name.lower() in card['name'].lower()), None)
                if support_card_name:
                    player_data[f'opponent_{support_card_name}'] = card['level']

            all_player_battle_logs = pd.concat([all_player_battle_logs, pd.DataFrame([player_data])], ignore_index=True)
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching battle logs for player {player_tag}: {e}")
        
    time.sleep(0.2)  # Add a delay between requests to avoid rate limiting

# Save all player battle logs to player_battle_logs.xlsx
all_player_battle_logs.to_excel('player_battle_logs.xlsx', index=False)

print("Excel file with all player battle logs saved successfully.")
