# Challenge for W

## Indice
- [Dataset](#dataset)
- [Struttura del Progetto](#struttura-del-progetto)
- [Metodologia](#metodologia)
- [Risultati Finali](#risultati-finali)
- [Setup](#setup)





# Dataset
Source: [10,000+ Popular TV Shows Dataset (TMDB) from Kaggle](https://www.kaggle.com/datasets/riteshswami08/10000-popular-tv-shows-dataset-tmdb).

Il dataset originale disponibile su Kaggle presentava un numero significativo di valori mancanti 
(circa il 10%) nella colonna `overview`.
Per ridurre la perdita di dati, è stata eseguita una fase di arricchimento tramite API esterne (**TMDB**, **OMDb**).
Questo processo ha permesso di recuperare la quasi totalità dei dati,
garantendo un dataset di partenza più completo e di alta qualità per l'analisi.






## Struttura del Progetto
- **`data/`**: cartella contenente dataset originale e file CSV intermedi e finali dello stesso dataset
- **`processed_data/`**: cartella contenente i set di training, validazione e test pronti per il training
- **`fill_overview.py`**: script di arricchimento dati tramite API
- **`analisi.ipynb`**: notebook di analisi esplorativa dei dati e feature engineering
- **`preprocessing.ipynb`**: notebook per data preprocessing: splitting, feature selection, scaling
- **`models.ipynb`**: notebook per addestramento e tuning dei modelli Ridge, RandomForest, SVR
- **`custom_nn.py`**: script che definisce una rete neurale custom (MLP) con PyTorch
- **`models_mlp.ipynb`**: notebook per addestramento e tuning del modello MLP (separato per avere maggiore focus)
- **`comparison.ipynb`**: notebook per il confronto finale dei modelli







## Metodologia

### 1. Data Cleaning e Arricchimento
* Rimozione duplicati.
* Arricchimento valori mancanti della colonna `overview`.
* Rimozione righe non recuperabili.

### 2. Feature Engineering (`analisi.ipynb`)
- Rimozione feature non significative (metadati) per il task di predizione.
- Trasformazioni Logaritmiche applicate a `vote_count` e `popularity` per normalizzare la loro distribuzione asimmetrica.
- Estratto l'anno (`release_year`) da `first_air_date` per ridurre la cardinalità.
- Feature categoriche:
    - `num_genres`: one-hot encoding del numero di generi.
    - `genre_ids`, `origin_country`, `original_language`: one-hot encoding delle categorie più frequenti per ciascuna feature
- Feature testuali:
    - `overview`: Utilizzo di `TfidfVectorizer` per trasformare le sinossi testuali in 100 feature numeriche basate sulla frequenza e importanza delle parole.

### 3. Preprocessing (`preprocessing.ipynb`)
- Data Splitting: suddivisione del dataset in set di training (70%), validazione (15%) e test (15%).
- Feature Selection: utilizzo di `VarianceThreshold` per rimuovere 31 feature con varianza quasi nulla.
- Scaling: utilizzo di `StandardScaler` sulle feature numeriche.

### 4. Tuning dei modelli

-   **Ridge Regression**:
    -   Tuning: È stata eseguita una grid search manuale per ottimizzare il parametro di regolarizzazione `alpha`.
    -   Analisi del Modello: Per interpretare il modello, è stata condotta un'analisi dei coefficienti per
    identificare le feature con il maggiore impatto sulla predizione.
    Inoltre, è stata effettuata un'analisi dei residui per verificare la presenza di bias.

-   **RandomForest Regressor**:
    -   Tuning: Utilizzo di `GridSearchCV` con `PredefinedSplit` per ottimizzare
    `n_estimators`, `max_depth`, `min_samples_leaf` e `max_features` sul nostro set di validazione personalizzato.
    -   Analisi del Modello: L'interpretabilità è stata affrontata tramite l'analisi della Feature Importance,
    che misura il contributo di ciascuna feature.
    È stato anche visualizzato un singolo albero decisionale di esempio
    per comprendere la logicam delle regole di split apprese dal modello.

-   **Support Vector Regression (SVR)**:
    -   Tuning: Anche in questo caso è stato utilizzato `GridSearchCV` per esplorare una griglia di iperparametri,
    inclusi `kernel`, `C`, `gamma` e `epsilon`.
    -   Analisi del Modello: Data la natura "black-box" del kernel RBF, l'importanza delle feature è stata stimata
    tramite Permutation Feature Importance ([source](https://christophm.github.io/interpretable-ml-book/feature-importance.html)).
    Questa tecnica, agnostica rispetto al modello, misura il calo di performance
    quando una singola feature viene mescolata casualmente, rivelando così il suo valore predittivo.

-   **Multi-Layer Perceptron (MLP)**:
    -   Tuning: Il tuning manuale della rete neurale personalizzata con PyTorch ha riguardato
    l'architettura della rete (`hidden_sizes`) e il `dropout_rate`.
    -   Analisi del Modello: Per monitorare il training e prevenire l'overfitting, è stato implementato un meccanismo di early stopping basato sulla performance sul set di validazione.
    Le curve di apprendimento del modello migliore sono state visualizzate per analizzare la convergenza.






## Risultati Finali
Dopo aver ri-addestrato i modelli con i migliori iperparametri sul set combinato di training e validazione,
sono state calcolate le performance finali sul test set.

| Model            | RMSE       | MAE        | R²         |
|------------------|------------|------------|------------|
| Ridge            | 1.1796     | 0.8156     | 0.1746     |
| **RandomForest** | **1.1407** | **0.7771** | **0.2281** |
| SVR              | 1.1840     | 0.8075     | 0.1684     |
| MLP              | 1.1833     | 0.8230     | 0.1693     |

Conclusioni:
- Il RandomForest Regressor è risultato il modello più performante su tutte le metriche, 
mostrando una capacità superiore nel catturare le relazioni non lineari presenti nei dati.
- Gli altri modelli (Ridge, SVR, MLP) hanno mostrato performance molto simili tra loro,
ma significativamente inferiori a quelle del RandomForest.
- Il valore di R^2 relativamente basso per tutti i modelli indica che la predizione del `vote_average` è un problema 
intrinsecamente complesso, quanto meno con le feature a disposizione.





## Setup
Assicurarsi di avere Python installato e creare un ambiente virtuale.
Le librerie necessarie sono elencate nel file `requirements.txt`. Installarle con:
```
pip install -r requirements.txt
```

Creare un file `.env` nella directory principale del progetto e aggiungere:
```
TMDB_KEY=XXXXXXX
OMDB_KEY=XXXXXXX
```