# Predizione di malattie cardiache (Heart Disease UCI)

## Obiettivo
Classificazione binaria (malato / sano) su 303 pazienti del dataset Cleveland.

## Metodo
- Preprocessing in Pipeline (imputazione, one-hot, scaling) per evitare data leakage
- Confronto di Logistic Regression, Random Forest, Gradient Boosting (5-fold stratificata)
- Feature engineering testata (hr_ratio, bp_chol, oldpeak_exang): nessun miglioramento significativo
- Test set (20%) usato una sola volta

## Risultati
| | ROC-AUC | Recall (malati) |
|---|---|---|
| CV, LogReg | 0.907 | 0.79 |
| Test set | 0.958 | 0.93 |

## Interpretabilità
Variabili più rilevanti: ca, cp, thal. [commento su cp=4]

## Limiti
- 303 pazienti, dataset storico (anni '80), nessuna validazione esterna
- Test set piccolo: le metriche sono stime con alta incertezza
- Non è un dispositivo medico: un software di supporto alla diagnosi
  rientrerebbe nella regola 11 dell'MDR 2017/745 (classe IIa o superiore)