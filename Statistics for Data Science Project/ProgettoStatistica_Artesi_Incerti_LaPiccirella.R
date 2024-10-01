#inizializzo una lista di vettori, ciascun vettore è un'istanza contenente la prorpia class label e il relativo score
#Optiamo per l'utilizzo dei dati forniti dallo studio originale al fine di condurre una riproduzione accurata dei risultati
istanze_dataset <- list(
  c(0, 150),
  c(1, 150),
  c(0, 180),
  c(0, 200),
  c(0, 205),
  c(0, 230),
  c(0, 260),
  c(0, 300),
  c(1, 190),
  c(1, 260),
  c(1, 250),
  c(1, 200),
  c(0, 280)
)


#Ordiniamo le istanze in ordine crescente sulla base del relativo score
istanze_dataset_ordinate <- istanze_dataset[order(sapply(istanze_dataset, "[[", 2))]

#Creiamo una lista di scores ordinati in ordine crescente, senza valori ripetuti
lista_scores_ordinati <- unique(sapply(istanze_dataset_ordinate, "[[", 2))



#Contiamo quanti Goods e quandi Bads sono presenti in totale nel dataset
totale_goods = 0
totale_bads = 0

for (i in 1:length(istanze_dataset)){
  if (istanze_dataset[[i]][1] == 0) {
    
    totale_goods = totale_goods + 1
  } else {
    
    totale_bads = totale_bads + 1
  }
}


#CALCOLO LE COORDINATE (X, Y) CHE COMPONGONO LA ROC CURVE (sulla X abbiamo i Goods e sulla Y abbiamo i Bads)

nGood <- 0
nBad <- 0
X_ROC <- c(0)
Y_ROC <- c(0)

for (i in lista_scores_ordinati) {
  counter_g <- 0
  counter_b <- 0
  
  for (j in istanze_dataset) {
    if (j[2] == i && j[1] == 0) {
      counter_g <- counter_g + 1
    }
    if (j[2] == i && j[1] == 1) {
      counter_b <- counter_b + 1
    }
  }
  
  nGood <- nGood + counter_g
  nBad <- nBad + counter_b
  
  X_ROC <- c(X_ROC, nGood / totale_goods)
  Y_ROC <- c(Y_ROC, nBad / totale_bads)
}


#Codice alternativo per il calcolo di X_ROC e Y_ROC, meno leggibile, ma con costo computazionale inferiore
#nGood <- 0
#nBad <- 0
#X_ROC <- numeric(0)
#Y_ROC <- numeric(0)
#
#for (i in 1:length(istanze_dataset_ordinate)) {
#  if (istanze_dataset_ordinate[[i]][1] == 0) {
#    nGood <- nGood + 1
#  }
#  if (istanze_dataset_ordinate[[i]][1] == 1) {
#    nBad <- nBad + 1
#  }
#  if (i == 1) {
#    X_ROC <- c(X_ROC, nGood / totale_goods)
#    Y_ROC <- c(Y_ROC, nBad / totale_bads)
#  }
#  if (i > 1 && istanze_dataset_ordinate[[i]][2] != istanze_dataset_ordinate[[i - 1]][2]) {
#    X_ROC <- c(X_ROC, nGood / totale_goods)
#    Y_ROC <- c(Y_ROC, nBad / totale_bads)
#  }
#  if (i > 1 && istanze_dataset_ordinate[[i]][2] == istanze_dataset_ordinate[[i - 1]][2]) {
#    if (length(X_ROC) > 0 && length(Y_ROC) > 0) {
#      X_ROC[length(X_ROC)] <- nGood / totale_goods
#      Y_ROC[length(Y_ROC)] <- nBad / totale_bads
#    }
#  }
#}
#
#X_ROC <- c(0, X_ROC)
#Y_ROC <- c(0, Y_ROC)
#
#print(X_ROC)
#print(Y_ROC)




#AUC GEOMETRICA

Geo_AUC_func <- function(X, Y) {
  
  AUC_geometric = 0
  j = 0
  
  for (i in 1:(length(Y)-1)) {
    
    j <- i + 1
    
    #AUC_geometric = AUC_geometric + (X[j] - X[i]) * Y[i] + (X[j] - X[i]) * (Y[j] - Y[i])/2
    
    AUC_geometric <- AUC_geometric + (X[j] - X[i]) * (Y[i] + Y[j])/2
    
  }
  
  return(AUC_geometric)
  
}


AUC_geometric = Geo_AUC_func(X_ROC, Y_ROC)







#PROBABILITY OF CORRECT RANKING OF A (GOOD, BAD) PAIR


Prob_AUC_func <- function(ds) {
  
  #Calcoliamo il numero totale di Concordant Pairs (N_c)
  N_c = 0
  
  for (i in 1:length(ds)) {
    
    for (j in 1:length(ds)) {
      
      if (ds[[i]][2] > ds[[j]][2] & ds[[i]][1] == 0 & ds[[j]][1] == 1) {
        
        N_c = N_c +1
        
      }
    }
  }
  
  
  #Estraiamo numero totale di TIED PAIRS nel dataset (num_tied_pairs)
  
  num_tied_pairs = 0
  
  for (i in lista_scores_ordinati) {
    num_goods_per_quel_j = 0
    num_bads_per_quel_j = 0
    
    for (j in ds) {
      if (j[[2]] == i) {
        if(j[[1]] == 0) {
          num_goods_per_quel_j = num_goods_per_quel_j + 1
          
        } else {
          num_bads_per_quel_j = num_bads_per_quel_j + 1
          
        }
      }
    }
    T_S_i = num_bads_per_quel_j * num_goods_per_quel_j
    num_tied_pairs = num_tied_pairs + T_S_i
  }
  
  
  #Estraiamo numero totale di GOODS nel dataset
  #(Nonostante avessimo già calcolato totale_goods e totale_bads precedentemente, riportiamo nuovamente la formula per presentare un procedimento più completo)
  
  totale_goods = 0
  totale_bads = 0
  
  for (i in ds) {
    
    if (i[[1]] == 0) {
      
      totale_goods = totale_goods + 1
    } else {
      totale_bads = totale_bads + 1
    }
  }
  
  AUC_probability = N_c/(totale_goods*totale_bads) + 0.5*(num_tied_pairs/(totale_goods*totale_bads))
  
  return(AUC_probability)

}

AUC_probability = Prob_AUC_func(istanze_dataset)





#WILCOXON RANK-SUM STATISTIC


wilc_AUC_func <- function(ds) {
  
  #Estraiamo per ogni istanza in ds il relativo rank
  temporanea <- numeric() #converte vettori in vettori numerici
  
  for (i in 1:length(ds)) {
    temporanea <- c(temporanea, ds[[i]][2])
  }
  
  ranked <- rank(temporanea)
  
  
  #Estraiamo solo i ranks relativi ai Goods
  rankedgood <- numeric()
  for (i in 1:length(ds)) {
    if (ds[[i]][1] == 0) {
      rankedgood <- c(rankedgood, ranked[i])
    }
  }
  
  
  #Calcoliamo ora la somma totale dei rank dei Goods (R_G)
  R_G <- sum(rankedgood)
  
  
  
  
  
  #NELLA SEZIONE QUI SOTTO MOSTRIAMO COME SAREBBE STATO IL CODICE SENZA L'UTILIZZO DELLA FUNZIONE rank()
  ##------------------------------------------------------------------------------------------------------------------------------------------------------------------
  #
  ##Ordiniamo in ordine crescente ds in base al valore degli scores
  #for (i in 1:length(ds)) {
  #  for (j in 1:length(ds)) {
  #    if (ds[[i]][2] > ds[[j]][2] & i < j) {
  #      
  #      ausiliario = ds[[i]]
  #      
  #      ds[[i]] = ds[[j]]
  #      ds[[j]] = ausiliario
  #    }
  #  }
  #}
  #print(ds)
  #
  #
  #ranked = rank(ds[2])
  #print(ranked)
  #
  #
  ##calcolo di R_G
  #R_G = 0
  #
  #for (i in lista_scores_ordinati) {
  #  counter = 0
  #  sum_dei_ranks = 0
  #  for (j in 1:length((ds))) {
  #    if (ds[[j]][2] == i) {
  #      counter = counter + 1
  #      sum_dei_ranks = sum_dei_ranks + j
  #    }
  #  }
  #  rank_dei_tied_con_i = sum_dei_ranks/counter
  #  for (k in ds) {
  #    if (k[2] == i & k[1] == 0) {
  #      R_G = R_G + rank_dei_tied_con_i
  #    }
  #  }
  #}
  #
  #
  ##------------------------------------------------------------------------------------------------------------------------------------------------------------------
  
  
  
  
  #Estriamo ora il numero TOTALE di GOODS e di BADS in ds
  #(Nonostante avessimo già calcolato totale_goods e totale_bads precedentemente, riportiamo nuovamente la formula per presentare un procedimento più completo)
  
  totale_goods = 0
  totale_bads = 0
  
  for (i in ds) {
    
    if (i[[1]] == 0) {
      
      totale_goods = totale_goods + 1
    } else {
      totale_bads = totale_bads + 1
    }
  }
  
  
  AUC_Wilcoxon = (R_G - 0.5*totale_goods*(totale_goods + 1))/(totale_goods*totale_bads)
  
  return((AUC_Wilcoxon))

}

AUC_Wilcoxon = wilc_AUC_func(istanze_dataset)





#VERIFICA FINALE
if (round(AUC_Wilcoxon, 4) == round(AUC_probability, 4) & round(AUC_probability, 4) == round(AUC_geometric, 4)) {
  print("I risultati delle 3 formule coincidono!")
}

#I risultati delle funzioni sono stati approssimati perchè presentano delle approssimazioni diverse oltre la quarta cifra decimale






