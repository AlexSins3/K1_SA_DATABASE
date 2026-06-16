"""Diagnostic: vérification de data leakage."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from prepare_data import get_model_data, ALL_FEATURES, load_raw_data, pair_matches, to_directed, engineer_features, split_chronological

raw = load_raw_data()
matches = pair_matches(raw)
directed = to_directed(matches)
featured = engineer_features(directed)
featured['y'] = featured['Win'].astype(int)

train, test = split_chronological(featured)

# CHECK 1: Les 2 rows d'un meme match sont-elles dans le meme split?
print('=== CHECK 1: Leakage entre lignes dirigées ===')
train_idx = set(train.index)
test_idx = set(test.index)

n_leak = 0
for i in range(0, len(featured)-1, 2):
    if (i in train_idx and i+1 in test_idx) or (i+1 in train_idx and i in test_idx):
        n_leak += 1
print(f'Matchs leakés (1 row train, 1 row test): {n_leak}')

# CHECK 2: Biais ceinture rouge
print('\n=== CHECK 2: Biais ceinture rouge ===')
red_train = train[train["Is_Red"]==1]["y"].mean()
blue_train = train[train["Is_Red"]==0]["y"].mean()
red_test = test[test["Is_Red"]==1]["y"].mean()
blue_test = test[test["Is_Red"]==0]["y"].mean()
print(f'WinRate rouge (train): {red_train:.3f}')
print(f'WinRate bleu (train): {blue_train:.3f}')
print(f'WinRate rouge (test): {red_test:.3f}')
print(f'WinRate bleu (test): {blue_test:.3f}')

# CHECK 3: Baselines naïves
print('\n=== CHECK 3: Baselines naïves ===')
# Prédire victoire si mieux classé
naive_ranking = (test['Ranking_Adv'] > 0).astype(int)
naive_ranking_acc = (naive_ranking == test['y']).mean()
print(f'Baseline "mieux classé gagne": {naive_ranking_acc:.3f}')

# CHECK 4: Les features cumulatives utilisent-elles des infos futures?
# Test: le H2H_Adv dans le test set inclut-il des matchs du test set?
print('\n=== CHECK 4: Vérification features cumulatives ===')
print(f'H2H_Adv moyen (train): {train["H2H_Adv"].mean():.3f}, std: {train["H2H_Adv"].std():.3f}')
print(f'H2H_Adv moyen (test): {test["H2H_Adv"].mean():.3f}, std: {test["H2H_Adv"].std():.3f}')
print(f'H2H_Adv != 0 dans train: {(train["H2H_Adv"] != 0).sum()} / {len(train)}')
print(f'H2H_Adv != 0 dans test: {(test["H2H_Adv"] != 0).sum()} / {len(test)}')

# CHECK 5: Corrélation directe — est-ce que les features sont trop corrélées à la target?
print('\n=== CHECK 5: Corrélations features → target ===')
corrs = featured[ALL_FEATURES + ['y']].corr()['y'].drop('y').abs().sort_values(ascending=False)
print(corrs.head(15).to_string())

# CHECK 6: Train sans la feature Is_Red
print('\n=== CHECK 6: Impact Is_Red (test sans) ===')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

feats_no_red = [f for f in ALL_FEATURES if f != 'Is_Red']
pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, C=0.5, class_weight='balanced'))])
pipe.fit(train[feats_no_red], train['y'])
proba = pipe.predict_proba(test[feats_no_red])[:, 1]
print(f'AUC sans Is_Red: {roc_auc_score(test["y"], proba):.4f}')

# CHECK 7: Duplicates exactes
print('\n=== CHECK 7: Paires A-B dupliquées ===')
featured['pair_key'] = featured.apply(lambda r: frozenset([str(r['Nom']), str(r['Opponent'])]), axis=1)
dup_in_both = set(train['pair_key'].unique()) & set(test['pair_key'].unique())
print(f'Paires présentes dans TRAIN et TEST: {len(dup_in_both)} / {len(test["pair_key"].unique())} test pairs')
