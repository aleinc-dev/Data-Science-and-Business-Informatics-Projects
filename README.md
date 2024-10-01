# Data-Science-and-Business-Informatics-Projects

# BREVE DESCRIZIONE DEI PROGETTI SVOLTI

Il percorso di studi ha previsto la partecipazione a diversi progetti pratici, svolti in gruppo, sviluppando competenze di collaborazione e problem-solving. Questi progetti sono stati focalizzati sull'applicazione concreta delle conoscenze teoriche apprese durante i relativi corsi.
Segue una breve descrizione dei progetti, con l'obiettivo di fornire una panoramica generale delle competenze pratiche acquisite.

# Data Mining 1
*Dataset impiegato*: Ravdess. Contiene un insieme di dati relativi a registrazioni audio all’interno delle quali diversi attori sono stati incaricati di “recitare” determinate proposizioni in contesti emotivi differenti.

*Obiettivo*: Lo scopo della seguente indagine è quello di analizzare il dataset per riconoscere gli aspetti emotivi del discorso indipendentemente dai contenuti semantici. 

Il procedimento mostra una iniziale indagine sulle variabili che descrivono il dataset, con una conseguente fase di data preparation dove vengono gestiti missing values, outliers, inconsistenze semantiche e variabili ridondanti. A seguito di queste operazioni iniziali sono state adottate tecniche approfondite di:
-	clustering
-	classification
-	pattern mining
*Linguaggio e librerie*: linguaggio Python, principalmente attraverso le librerie Numpy, Pandas, Matplotlib e Sklearn.

# Data Mining 2
*Obiettivo*: medesimo di ProgettoDM1 sul medesimo dataset, ma attraverso l’utilizzo di ulteriori tecniche di classificazione avanzata, quali:
-	Logistic Regression
-	Support Vector Machine
-	Ensemble Methods (Random Forest, Bagging&Boosting)
-	Neural Networks
-	Gradient Boosting (XGBoost, LGBMClassifier)

E tecniche di regressione multivariata non lineare, quali:
-	SVR
-	GradientBoostingRegressor
Facendo poi riferimento alle registrazioni audio (e non solo alle misure aggregate considerate sinora), sono stati condotti degli studi a proposito delle time series, tra cui:
-	Ricerca di Motifs/Discords
-	Clustering
-	Classification (KNN, Shapelets, Rocket e MiniRocket)

# Business Process Modeling
*Obiettivo*: Analisi e modellazione di un processo organizzativo tramite Business Process Modeling Notation (BPMN). Il modello è stato successivamente convertito in WorkFlow Net (Petri Net) e sottoposto a verifica di Soundness e altre proprietà, al fine di garantirne la correttezza.

*Software utilizzati*: Bpmn-js, Woflan, Woped.

# Statistics for Data Science
*Obiettivo*: Analisi approfondita di un articolo scientifico che dimostra l’equivalenza analitica di tre metodi (Geometrico, Wilcoxon Rank-Sum Statistic e Adjusted Percent of Concordant Pairs) per il calcolo dell'Area Under the Curve (AUC), utilizzata per valutare la performance di un modello di classificazione. Successivamente, implementazione dell’analisi in linguaggio R per confermare i risultati presentati nell'articolo oppure, qualora necessario, sollevare eventuali dubbi sulla correttezza dei metodi, supportati da argomentazioni matematiche.

*Software utilizzati*: RStudio, PowerPoint.

# Algorithms and Data Structures for Data Science
*Obiettivo*: Sviluppo di notebook in linguaggio Python per risolvere problemi complessi, con particolare attenzione alla minimizzazione della complessità temporale (notazione asintotica).

Il progetto ha comportato sia la creazione di algoritmi originali quando necessario, sia l'ottimizzazione di algoritmi esistenti, per garantire soluzioni computazionalmente efficienti. Inoltre, in diversi casi è stato necessario manipolare strutture dati come Hash Tables, Heap e Binary Search Trees.

*Software utilizzati*: Jupyter.


# Machine Learning
Per questo progetto è possibile consultare le slide riassuntive di presentazione collocate all’interno della cartella relativa al progetto stesso.

# Laboratorio di Data Engineering
*Obiettivo*: Implementazione del processo ETL da fonti di dati eterogenee (CSV, JSON, XML). Creazione di un database e caricamento dei dati tramite SQL Server Management Studio. Sviluppo di query SQL complesse per l'analisi dei dati.

Costruzione e popolamento di un cubo OLAP utilizzando SSIS, con interrogazioni MDX.

Visualizzazione dei dati attraverso dashboard interattive in Power BI.

