import json
import csv
import math
import xml.etree.ElementTree as ET

#NB. All'interno del seguente codice abbiamo intenzione di aumentare la presenza di (try, except) e di funzioni

# Apriamo il file CSV "Police.csv" in modalità lettura e creiamo una lista "lines" che contiene tutte le righe del file sotto forma di stringhe
with open('Police.csv', 'r') as police:
    # Leggiamo tutte le linee del file in una lista
    lines_police = police.readlines()
    #Non chiudiamo il file manualmente con police.close() perchè quando apriamo il file con "with open..." il file viene chiuso automaticamente al termine del blocco di codice



#Dal momento che "lines" è una lista di stringhe, lo trasformiamo in una lista di liste (dove ciascuna lista interna è una lista delle parole delle singole stringhe):
police_table = list()
#Dopo il ciclo for qui sotto, tutti_elements diventerà una lista di liste, dove ciascuna lista interna contiene come elementi le parole delle ex-stringhe
for line in lines_police:
    element = line.strip().split(",")
    police_table.append(element)

police_table[0][10] = "date_id" #Sostituisco date_fk con date_id semplicemente perchè nella consegna viene chiamato in questo modo, il fatto che sia una foreign key lo capiamo dalla struttura delle tabelle
#print(tutti_elements[0][10])


for i in range(10):
    print(police_table[i])




#ASSIGNEMENT 1 PUNTO 1

#PARTECIPANT_ID
#GUN_ID
#GEO_ID

#Ci siamo messi nell'ottica in cui non sappiamo quante caratteristiche riguardino PARTICIPANT e GUN, nè in quali posizioni si trovino...
#...l'unica informazione di cui abbiamo presunto di essere in possesso è che tutte le caratteristiche che riguardano PARTICIPANT e GUN abbiano "participant" e "gun" nei propri metadati...
#...pertanto attraverso il blocco di codice qui sotto andiamo a cercare tutte le caratteristiche contenenti "participant" e "gun" e incameriamo il loro valore e il loro indice in un dizionario dedicato.
#(Per GEO facciamo lo stesso ragionamento ma presumendo di conoscere che le informazioni relative alla latitudine e alla longitudine portino nei propri metadati le parole "latitude" e "longitude".

dizionario_part_id = {}
dizionario_gun_id = {}
dizionario_geo_id = {}
for i in range(len(police_table[0])):
    if "participant" in police_table[0][i]:
        dizionario_part_id[police_table[0][i]] = i
    if "gun" in police_table[0][i]:
        dizionario_gun_id[police_table[0][i]] = i
    if "latitude" in police_table[0][i] or "longitude" in police_table[0][i]:
        dizionario_geo_id[police_table[0][i]] = i
#print("dizionario_part_id: ", dizionario_part_id)
#print("dizionario_gun_id: ", dizionario_gun_id)
#print("dizionario_geo_id: ", dizionario_geo_id)

#Creiamo le liste che andranno a contenere le varie combinazioni per participant, gun e geo e inseriamo all'interno della prima lista i metadati che abbiamo incamerato nei dizionari.
part_id_table = list()###
metadata_part = list()
for i in dizionario_part_id.keys():
    metadata_part.append(i)
part_id_table.append(metadata_part)

gun_id_table = list()###
metadata_gun = list()
for i in dizionario_gun_id.keys():
    metadata_gun.append(i)
gun_id_table.append(metadata_gun)

geo_id_table = list()###
metadata_geo = list()
for i in dizionario_geo_id.keys():
    metadata_geo.append(i)
geo_id_table.append(metadata_geo)

#Aggiungiamo manualmente i metadati fornitici dalla consegna
police_table[0].append("participant_id")
police_table[0].append("gun_id")
police_table[0].append("geo_id")


for i in range(1, len(police_table)):
    current_combo_part_id = []
    current_combo_gun_id = []
    current_combo_geo_id = []
    for j in range(len(police_table[i])):
        if j in dizionario_part_id.values():
            current_combo_part_id.append(police_table[i][j])
        elif j in dizionario_gun_id.values():
            current_combo_gun_id.append(police_table[i][j])
        elif j in dizionario_geo_id.values():
            current_combo_geo_id.append(police_table[i][j])


    if current_combo_part_id in part_id_table:  # Se le coordinate in esame sono già state analizzate in precedenza...
        police_table[i].append(part_id_table.index(current_combo_part_id)-1)  # Nella lista di "police_table" che stiamo analizzando andiamo ad aggiungere come ultimo elemento il geo_id
    else:  # Se le coordinate in esame NON sono già state analizzate in precedenza...
        part_id_table.append(current_combo_part_id)  # Prima inseriamo la coppia di coordinate in esame in "set_coordinate_geo_id" e poi...
        police_table[i].append(part_id_table.index(current_combo_part_id)-1) # Nella lista di "police_table" che stiamo analizzando andiamo ad aggiungere come ultimo elemento il geo_id

    if current_combo_gun_id in gun_id_table:
        police_table[i].append(gun_id_table.index(current_combo_gun_id)-1)
    else:
        gun_id_table.append(current_combo_gun_id)
        police_table[i].append(gun_id_table.index(current_combo_gun_id)-1)

    if current_combo_geo_id in geo_id_table:
        police_table[i].append(geo_id_table.index(current_combo_geo_id)-1)
    else:
        geo_id_table.append(current_combo_geo_id)
        police_table[i].append(geo_id_table.index(current_combo_geo_id)-1)

# RISULTATO PER PARTICIPANT: Una lista di liste, dove ciascuna lista interna rappresenta una diversa combinazione dei valori di participant (una riga della tabella)
#... (essendo un set costruito senza ripetizioni, la loro posizione all'interno della lista omnicomprensiva indica il loro participant_id (la prima lista interna contiene i metadati), con un blocco di codice successivo andremo ad inserirli)
# NB: in lista_gun_id alcuni valori sono "unknown", ma diamo un id anche alle combinazioni che hanno unknown all'interno dei propri valori...
#...perchè se per esempio in una combinazione si conosce solo la gun_type, allora quella è una combinazione che vogliamo riconoscere fra le altre

    # Questo "if" qui sotto ci serve solamente per tenere traccia del progresso percentuale del blocco di codice
    if i % 5000 == 0:
        print(round((i / 170929) * 100, 2), "%")
    elif i == len(police_table)-1:
        print("100%")


#Qui sotto AGGIUNGIAMO GLI INDICI ALL'INTERNO DELLE TABELLE
#Non lo facciamo direttamente all'interno del blocco qui sopra perchè se no dovremmo per ogni if fare un ciclo for che vada a prendere le coppie di elementi all'interno delle liste per vedere se sono già presenti
#invece in questo modo sappiamo già che ogni elemento avrà come indice (la propria posizione - 1), quindi dobbiamo semplicemente andare a reiterare un'unica volta ciascuno dei 3 dataset
for i in range(len(part_id_table)):
    if i == 0:
        part_id_table[i].insert(0, "participant_id")
    else:
        part_id_table[i].insert(0, i-1)

for i in range(len(gun_id_table)):
    if i == 0:
        gun_id_table[i].insert(0, "gun_id")
    else:
        gun_id_table[i].insert(0, i-1)

for i in range(len(geo_id_table)):
    if i == 0:
        geo_id_table[i].insert(0, "geo_id")
    else:
        geo_id_table[i].insert(0, i-1)
#for i in range(10):
    #print("lista_part_id: ", lista_part_id[i])




#Qui sotto abbiamo creato una funzione che prende in input la lista dei valori da ricercare, la tabella all'interno della quale ricercarli e il valore (stringa) che si vuole aggiungere come metadato per la colonna che si aggiungerà alla tabella di riferimento
#La funzione ritorna una id_table che conterrà tutte le combinazioni di valori relativi all'aspetto che vogliamo indagare, con i relativi id
#Questa funzione è un modo di ottenere il risultato richiesto nelle task mantenendo un elevato grado di leggibiltà e riutilizzo del codice...
#... tuttavia, dovendo richiamare la funzione per ogni table che vogliamo creare, è meno efficiente rispetto al blocco di codice qui sopra che invece itera la table di riferimento (police_table) un'unica volta creando tutte e 3 le table richieste.
#Per questa ragione, ai fini della consegna abbiamo deciso di tenere il blocco di codice più efficiente.
"""
def funzione_id_table (lista_valori_ricerca, table_di_riferimento, valore_metadato): #valori di ricerca nei metadati "participant", table in cui ricercare questi valori, termine da aggiungere come metadato "participant_id"

    dizionario_id = {}

    for i in range(len(table_di_riferimento[0])):
        for j in lista_valori_ricerca:
            if j in table_di_riferimento[0][i]:
                dizionario_id[table_di_riferimento[0][i]] = i

    id_table = list()  ###
    metadata_id_table = list()
    for i in dizionario_id.keys():
        metadata_id_table.append(i)
    id_table.append(metadata_id_table)

    table_di_riferimento[0].append(valore_metadato)

    for i in range(1, len(table_di_riferimento)):
        current_combo_id = []

        for j in range(len(table_di_riferimento[i])):
            if j in dizionario_id.values():
                current_combo_id.append(table_di_riferimento[i][j])

        if current_combo_id in id_table:  # Se le coordinate in esame sono già state analizzate in precedenza...
            table_di_riferimento[i].append(id_table.index(current_combo_id)-1)  # Nella lista di "police_table" che stiamo analizzando andiamo ad aggiungere come ultimo elemento il geo_id
        else:  # Se le coordinate in esame NON sono già state analizzate in precedenza...
            id_table.append(current_combo_id)  # Prima inseriamo la coppia di coordinate in esame in "set_coordinate_geo_id" e poi...
            table_di_riferimento[i].append(id_table.index(current_combo_id)-1)

        # Questo "if" qui sotto ci serve solamente per tenere traccia del progresso percentuale del blocco di codice
        if i % 5000 == 0:
            print("Sto creando la tabella ", valore_metadato, ": ",   round((i / 170929) * 100, 2), "%")
        elif i == len(police_table) - 1:
            print("100%: TABELLA CREATA")



    for i in range(len(id_table)):
        if i == 0:
            id_table[i].insert(0, valore_metadato)
        else:
            id_table[i].insert(0, i - 1)

    for i in range(20):
        print(id_table[i])

    return id_table

#creiamo part_id_table:
lista_valori_ricerca_part = ["participant"]
#table_di_riferimento = police_table
#valore_metadato_geo = "participant_id"
part_id_table = funzione_id_table(lista_valori_ricerca_part, police_table, "participant_id")

#creiamo geo_id_table:
lista_valori_ricerca_geo = ["latitude", "longitude"]
#table_di_riferimento = police_table
#valore_metadato_geo = "geo_id"
geo_id_table = funzione_id_table(lista_valori_ricerca_geo, police_table, "geo_id")

#creiamo gun_id_table:
lista_valori_ricerca_gun = ["gun"]
#table_di_riferimento = police_table
#valore_metadato_geo = "gun_id"
gun_id_table = funzione_id_table(lista_valori_ricerca_gun, police_table, "gun_id")
"""












#ASSIGNEMENT 1 PUNTO 2

"""
with open("dict_partecipant_age.json", 'r') as part_age: # Apriamo il file "dict_partecipant_age.json" in modalità lettura
    age_dict = json.load(part_age) #Copiamo il contenuto del file json all'interno di un dizionario python

with open("dict_partecipant_status.json", 'r') as part_status:
    status_dict = json.load(part_status)

with open("dict_partecipant_type.json", 'r') as part_type:
    type_dict = json.load(part_type)
"""
with open("dict_partecipant_age.json", "r") as part_age, open("dict_partecipant_status.json", 'r') as part_status, open("dict_partecipant_type.json", 'r') as part_type:
    age_dict = json.load(part_age) #Copiamo il contenuto del file json all'interno di un dizionario python
    status_dict = json.load(part_status)
    type_dict = json.load(part_type)


#Effettuamo una verifica che le operazioni abbiano prodotto il risultato desiderato printando le informazioni a schermo
print("Type: ", type(age_dict), "\nContenuto AGE (F1): ", age_dict, "\n")           #F1
print("Type: ", type(type_dict), "\nContenuto TYPE (F2): ", type_dict, "\n")        #F2
print("Type: ", type(status_dict), "\nContenuto STATUS (F3): ", status_dict, "\n")  #F3

#Ora calcoliamo la crime_gravity per ciascuna istanza del dataset e andiamo ad inserirla al termine di ciascun elemento di "police_table"
#(Anche questo calcolo avrebbe potuto essere inserito all'interno del blocco di codice per il calcolo di participant_id, gun_id e geo_id al fine di ottimizzare ulteriormente il codice, ma...
#...abbiamo deciso di tenerelo qui per preservare un certo grado di leggibilità del codice e mantenere le task della consegna separate tra loro)
police_table[0].append("crime_gravity")
for i in range(1, len(police_table)):
    crime_gravity = age_dict[str(police_table[i][1])] * type_dict[str(police_table[i][4])] * status_dict[str(police_table[i][3])] #Formula fornita dalla consegna
    police_table[i].append(crime_gravity)





#DIVIDIAMO POLICE NELLE TABELLE RICHIESTE DALLA CONSEGNA E CREIAMO I RELATIVI FILE CSV



#CUSTODY.CSV

#Andiamo qui sotto a vedere quali indici corrispondono alle caratteristiche che vogliamo includere in custody.csv e andiamo ad incamerarli nella lista "indici_custody"
indici_custody = list()
for i in range(len(police_table[0])):
    if police_table[0][i] == "custody_id" or police_table[0][i] == "participant_id" or police_table[0][i] == "gun_id" or police_table[0][i] == "geo_id" or police_table[0][i] == "date_id" or police_table[0][i] == "crime_gravity" or police_table[0][i] == "incident_id":
        indici_custody.append(i)

print("indici_custody: ", indici_custody)

#Qui sotto:
#1- creiamo una lista vuota che andrà a contenere le righe del file custody.csv
custody_id_table = list()
#2-per ogni riga della tabella police (nel nostro caso rinominata in "tutti_elements") andiamo ad incamerare in "current_custody" i valori che ci interessano di quella determinata riga
for i in range(len(police_table)):
    current_custody = []

    for j in range(len(police_table[i])):
        if j in indici_custody:
            current_custody.append(police_table[i][j])
    #3- Terminata il riempimento di "current_custody" andiamo ad aggiungerlo come elemento all'interno di custody.csv
    custody_id_table.append(current_custody)


def writer_on_csv (nome_file, table_da_scrivere):
    with open(nome_file, mode='w', newline='') as current_file:
        writer = csv.writer(current_file, delimiter=',')
        writer.writerows(table_da_scrivere)


writer_on_csv("Custody.csv", custody_id_table)

"""
#4- Una volta creata la lista di liste che rappresenta il nostro file custody.csv, andiamo ad aprire il file custody.csv in modalità scrittura e andiamo a scrivere questi dati all'interno del file
with open("Custody.csv", mode='w', newline='') as current_file:
    writer = csv.writer(current_file, delimiter=',')
    writer.writerows(custody_csv)
"""



#PARTICIPANT.CSV

#Usiamo lista_part_id come tabella per creare participant.csv
"""
with open("Participant.csv", mode='w', newline='') as participant_file:
    writer = csv.writer(participant_file, delimiter=',')
    writer.writerows(lista_part_id)
"""

writer_on_csv("Participant.csv", part_id_table)



#GUN.CSV

#Qui usiamo lista_gun_id come tabella per creare il file gun.csv
"""
with open("Gun.csv", mode='w', newline='') as gun_file:
    writer = csv.writer(gun_file, delimiter=',')
    writer.writerows(lista_gun_id)
"""

writer_on_csv("Gun.csv", gun_id_table)



#GEOGRAPHY.CSV


with open('uscities.csv', 'r') as uscities: #uscities.csv è un file contenente le coordinate di circa 30°000 città americane e le relative informazioni, tra cui latitudine, longitudine e stato di appartenenza.
                                            # Il file è presente all'interno della cartella del progetto e maggiori informazioni a proposito del file sono presentate nel report.
    lines_uscities = uscities.readlines()

uscities_list = list()
#Dopo il ciclo for qui sotto, uscities_list diventerà una lista di liste, dove ciascuna lista interna contiene come elementi le parole delle ex-stringhe
for line in lines_uscities:
    element = line.strip().split(",") #strip() rimuove eventuali spazi bianchi e \n. split() prende la stringa line e la trasforma in una lista di valori utilizzando "," come separatore
    uscities_list.append(element)


for i in range(len(uscities_list)):
    for j in range(len(uscities_list[i])):
        uscities_list[i][j] = uscities_list[i][j].strip('"') #Metadati e dati nel file uscities.csv contengono le virgolette all'interno delle stringhe dei valori, quindi le eliminiamo qui


dizionario_uscities = {}
#Il ragionamento intuitivo alla base della ricerca degli elementi all'interno del ciclo for qui sotto è analogo a quello per "participant", "geo" e "gun" presentato sopra
for i in range(len(uscities_list[0])):
    if uscities_list[0][i] == "lat" or uscities_list[0][i] == "lng" or uscities_list[0][i] == "city" or uscities_list[0][i] == "state_name":
        dizionario_uscities[uscities_list[0][i]] = i
print("dizionario_uscities: ", dizionario_uscities)


del uscities_list[5278] #Questo elemento ci ha dato dei problemi durante l'iterazione del dataset in quanto non presentava un valore numerico ma un valore stringa. Essendo uno solo abbiamo deciso di eliminarlo.

geo_id_table[0].append("city")
geo_id_table[0].append("state_name")
for i in range(1, len(geo_id_table)):
    dist_min = float("inf")
    indice_dist_min = float("inf")
    for j in range(1, len(uscities_list)):
        try:
            attuale_dist = math.sqrt((float(geo_id_table[i][1]) - float(uscities_list[j][int(dizionario_uscities["lat"])]))**2 + (float(geo_id_table[i][2]) - float(uscities_list[j][int(dizionario_uscities["lng"])]))**2)
            if attuale_dist < dist_min:
                indice_dist_min = j
                dist_min = attuale_dist
        except:
            print("QUALCOSA NON HA FUNZIONATO:    \nindice = ", j, "\nvalore: ", uscities_list[j][int(dizionario_uscities["lng"])])

    geo_id_table[i].append(uscities_list[indice_dist_min][dizionario_uscities["city"]])
    geo_id_table[i].append(uscities_list[indice_dist_min][dizionario_uscities["state_name"]])

    # Questo "if" qui sotto ci serve solamente per tenere traccia del progresso percentuale del blocco di codice
    if i % 500 == 0:
        print("Attuale stato di completamento geo_id: ", round((i / len(geo_id_table)) * 100, 2), "%")
    elif i == len(police_table)-1:
        print("100%")

#for i in range(10):
    #print("LISTA_GEO_ID: ", lista_geo_id[i])


#Qui usiamo lista_geo_id come tabella per creare il file geo.csv
with open("Geography.csv", mode='w', newline='') as geo_file:
    writer = csv.writer(geo_file, delimiter=',')
    writer.writerows(geo_id_table)


#DATE.CSV

tree = ET.parse('dates.xml')
root = tree.getroot()

#Visualizziamo il contenuto del nostro file xml
for row in root.findall('row'):
    for element in row:
        print(f"Tag: {element.tag}, Testo: {element.text}")
        for attributo, valore in element.attrib.items():
            print(f"Attributo: {attributo}, Valore: {valore}")


date_id_table = []

#Poichè all'interno del file xml non abbiamo individuato metadati, essendo facilmente intuibile la loro natura li inseriamo noi manualmente:
#NB. Non inseriamo anche l'orario in quanto presenta lo stesso valore per tutte le istanze e sarebbe quindi non significativo e ridondante
date_id_table.append(["date_id", "day", "month", "year"])


for row in root.findall('row'):
    date_pk = row.find('date_pk').text

    data = row.find('date').text
    anno, mese, giorno = data.split(' ')[0].split('-')

    lista_interna = [int(date_pk), int(giorno), int(mese), int(anno)]
    date_id_table.append(lista_interna)


for i in range(10):
    print(date_id_table[i])

#Qui usiamo lista_date_id come tabella per creare il file date.csv
with open("Date.csv", mode='w', newline='') as date_file:
    writer = csv.writer(date_file, delimiter=',')
    writer.writerows(date_id_table)







import pyodbc
import csv

# Parametri per la connessione
server = 'tcp:131.114.72.230'
database = 'Group_ID_4_DB'
username = 'Group_ID_4'
password = 'OKYCQYAH'

# Connessione
connectionString = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password
cnxn = pyodbc.connect(connectionString)
cursor = cnxn.cursor()

# Di seguito il codice per eseguire l'upload di ogni tabella (una alla volta)

# Tabella Gun

csv_file = 'Gun.csv' #Questa stringa conterrebbe il persorso al file csv di interesse.
table_name = 'Gun'

# Apertura e lettura del file csv
with open(csv_file, 'r') as file:
    csv_reader = csv.reader(file)

    header = next(csv_reader)
    # Costruzione della query SQL per eseguire l'upload
    insert_query = f"INSERT INTO {table_name} ({', '.join(header)}) VALUES ({', '.join(['?'] * len(header))})"

    # Loop attraverso le righe del csv ed inserimento dei dati nelle tabelle su SQL Managment Studio
    for row in csv_reader:
        cursor.execute(insert_query, row)

# Procedimento come sopra
# Tabella  Date

csv_file = 'Date.csv' #Questa stringa conterrebbe il persorso al file csv di interesse.
table_name = 'Date'

with open(csv_file, 'r') as file:
    csv_reader = csv.reader(file)

    header = next(csv_reader)

    insert_query = f"INSERT INTO {table_name} ({', '.join(header)}) VALUES ({', '.join(['?'] * len(header))})"

    for row in csv_reader:
        cursor.execute(insert_query, row)

# Tabella Participant

csv_file = 'Participant.csv' #Questa stringa conterrebbe il persorso al file csv di interesse.
table_name = 'Participant'

with open(csv_file, 'r') as file:
    csv_reader = csv.reader(file)

    header = next(csv_reader)

    insert_query = f"INSERT INTO {table_name} ({', '.join(header)}) VALUES ({', '.join(['?'] * len(header))})"

    for row in csv_reader:
        cursor.execute(insert_query, row)

# Tabella Geography

csv_file = 'Geography.csv' #Questa stringa conterrebbe il persorso al file csv di interesse.
table_name = 'Geography'

with open(csv_file, 'r') as file:
    csv_reader = csv.reader(file)

    header = next(csv_reader)

    insert_query = f"INSERT INTO {table_name} ({', '.join(header)}) VALUES ({', '.join(['?'] * len(header))})"

    for row in csv_reader:
        cursor.execute(insert_query, row)

# Tabella Custody, da caricare per ultima essendo presenti chiavi esterne

csv_file = 'Custody.csv' #Questa stringa conterrebbe il persorso al file csv di interesse.
table_name = 'Custody'

with open(csv_file, 'r') as file:
    csv_reader = csv.reader(file)

    header = next(csv_reader)

    insert_query = f"INSERT INTO {table_name} ({', '.join(header)}) VALUES ({', '.join(['?'] * len(header))})"

    for row in csv_reader:
        cursor.execute(insert_query, row)

# Applicazione delle modifiche e termine della connessione
cnxn.commit()
cnxn.close()