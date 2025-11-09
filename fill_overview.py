import sys

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
df = pd.read_csv('data/10k_Poplar_Tv_Shows.csv')
df = df.drop_duplicates().reset_index(drop=True)
ids_to_fill = df[df['overview'].isnull()]['id'].tolist()
print(f"Missing overviews: {len(ids_to_fill)}")



from pathlib import Path
from dotenv import load_dotenv
import os
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)
tmdb_api_key = os.getenv('TMDB_KEY')
if not tmdb_api_key:
    raise ValueError("TMDB_KEY not found in .env")
omdb_api_key = os.getenv('OMDB_KEY')
if not omdb_api_key:
    raise ValueError("OMDB_KEY not found in .env")


# mi serve per il controllo incrociato
if 'imdb_id' not in df.columns:
    df['imdb_id'] = None


print("###### RETRIEVE DATA FROM TMDB ######")


# Reference: https://developer.themoviedb.org/reference/tv-series-details
import requests
from tqdm import tqdm
updated_overviews = {}
updated_imdb_ids = {}
for id in tqdm(ids_to_fill, desc="Processing missing entries"):
    # recupero overview da TMDB
    url = f"https://api.themoviedb.org/3/tv/{id}"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {tmdb_api_key}",
    }
    overview_found = False
    try:
        response = requests.get(url, headers=headers)
        overview = response.json()['overview']
        if overview and (overview.strip() != ""):
            updated_overviews[id] = overview.strip()
            overview_found = True
            # print(f"Updated overview for {id}: {overview}")
    except Exception as e:
        # print(f"Error for ID {id}: {type(e).__name__}")
        pass

    # se non trova l'overview dal TMDB, recupera l'id di IMDb per interrogare invece OMDB

    # Reference: https://developer.themoviedb.org/reference/tv-series-external-ids
    if not overview_found:
        external_ids_url = f"https://api.themoviedb.org/3/tv/{id}/external_ids"
        try:
            response_ids = requests.get(external_ids_url, headers=headers)
            response_ids.raise_for_status()
            imdb_id = response_ids.json().get('imdb_id')
            if imdb_id and imdb_id.startswith('tt'):
                updated_imdb_ids[id] = imdb_id
        except Exception:
            pass

print(f"Found {len(updated_overviews)}/{len(ids_to_fill)} overviews from TMDB for the missing entries.")
print(f"Found {len(updated_imdb_ids)}/{len(ids_to_fill)} IMDb IDs for entries that still need an overview.")

mask = df['id'].isin(ids_to_fill)
df.loc[mask, 'overview'] = df.loc[mask, 'id'].map(updated_overviews).fillna(df.loc[mask, 'overview'])
df.loc[mask, 'imdb_id'] = df.loc[mask, 'id'].map(updated_imdb_ids).fillna(df.loc[mask, 'imdb_id'])



# elimino le righe che non hanno ne overview ne imdb_id
df['overview'] = df['overview'].fillna('')
initial_rows = len(df)
condition_to_drop = (df['overview'].str.strip() == '') & (df['imdb_id'].isnull())
rows_to_drop = df[condition_to_drop]
df = df.drop(rows_to_drop.index)
print(f"Final rows: {len(df)}. Rows removed: {initial_rows - len(df)}")
df.to_csv('data/filled1.csv', index=False)








print("\n\n###### RETRIEVE DATA FROM OMDB ######")
df = pd.read_csv('data/filled1.csv')
# individuo righe dove manca overview ma con imdb_id
is_overview_missing = (df['overview'].str.strip() == '') | (df['overview'].isnull())
is_imdb_present = df['imdb_id'].notnull()
rows_to_fetch_omdb = df[is_overview_missing & is_imdb_present]   # condizione necessaria per cercare risorse in OMDB
ids_to_fetch_omdb = rows_to_fetch_omdb['id'].tolist()
imdb_ids_to_fetch = rows_to_fetch_omdb['imdb_id'].tolist()
print(f"Missing overviews with a valid imdb_id to check on OMDb: {len(ids_to_fetch_omdb)}")

updated_overviews_omdb = {}

# Creo un dizionario per mappare imdb_id -> tmdb_id per l'aggiornamento
# Questo è necessario perché aggiorneremo il df usando l'id originale (TMDB)
tmdb_imdb_map = dict(zip(imdb_ids_to_fetch, ids_to_fetch_omdb))

if imdb_ids_to_fetch:  # Esegui solo se ci sono ID da cercare
    for imdb_id in tqdm(imdb_ids_to_fetch, desc="Fetching from OMDb"):
        # Reference: https://www.omdbapi.com/
        omdb_url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={omdb_api_key}"
        try:
            response = requests.get(omdb_url)
            data = response.json()
            if data.get('Response') == 'True':
                plot = data.get('Plot')
                if plot and (plot.strip() != "") and (plot != 'N/A'):
                    tmdb_id = tmdb_imdb_map[imdb_id]
                    updated_overviews_omdb[tmdb_id] = plot.strip()
        except Exception:
            pass

print(f"Found {len(updated_overviews_omdb)}/{len(ids_to_fetch_omdb)} new overviews from OMDb.")

df['overview'] = df.apply(
    lambda row: updated_overviews_omdb.get(row['id'], row['overview']),
    axis=1
)


initial_rows_final = len(df)
df['overview'] = df['overview'].fillna('')
final_rows_to_drop = df[df['overview'].str.strip() == '']
if not final_rows_to_drop.empty:
    df = df.drop(final_rows_to_drop.index)
print(f"Initial rows for this phase: {initial_rows_final}. Final rows: {len(df)}. Rows removed: {initial_rows_final - len(df)}")

df.to_csv('data/filled_final.csv', index=False, encoding='utf-8')
