import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from itertools import combinations
import networkx as nx

all_columns = []
N = 10
diagrams = [f"top {N}", f"worst {N}", "pareto", "promethee i", "promethee ii", "electre iv", "electre is"]


def read_csv(input_csv_path):
    df = pd.read_csv(input_csv_path, sep=",")
    draw_Diagram(df)


def read_user_column(df):
    print("Voici les colonnes disponibles : ")
    for i, col in enumerate(df.columns):
        print(f"{i + 1}. {col}")
        all_columns.append(col)

    print("Veuillez sélectionner une colonne par son nom ou son ID : ")
    user_input = input()

    try:
        # Essayer de convertir l'entrée en entier (ID)
        column_id = int(user_input)
        if 1 <= column_id <= len(df.columns):
            return df.columns[column_id - 1]
    except ValueError:
        # Si la conversion échoue, traiter l'entrée comme le nom de la colonne
        if user_input in df.columns:
            return user_input

    print("Entrée invalide. Veuillez choisir une colonne valide.")
    return read_user_column(df)


def choose_Diagram():
    print("Voici les colonnes disponibles : ")
    for i, col in enumerate(diagrams):
        print(f"{i + 1}. {col}")

    print("Veuillez sélectionner une colonne par son nom ou son ID : ")
    user_input = input()

    try:
        # Essayer de convertir l'entrée en entier (ID)
        column_id = int(user_input)
        if 1 <= column_id <= len(diagrams):
            return diagrams[column_id - 1]
    except ValueError:
        # Si la conversion échoue, traiter l'entrée comme le nom de la colonne
        if user_input in diagrams:
            return user_input

    print("Entrée invalide. Veuillez choisir une colonne valide.")
    return choose_Diagram().lower()


def draw_Diagram(df):
    next = True
    while next:
        diagram = choose_Diagram()
        if diagram in [diagrams[0], diagrams[1], diagrams[2]]:
            column = read_user_column(df)
            if diagram == diagrams[0]:
                get_Top(df, column, True)
            elif diagram == diagrams[1]:
                get_Top(df, column, False)
            elif diagram == diagrams[2]:
                pareto_frontier(df)
        if diagram == diagrams[3]:
            promethee(df, False)
        elif diagram == diagrams[4]:
            promethee(df, True)
        elif diagram == diagrams[5]:
            electre(df, False)
        elif diagram == diagrams[6]:
            electre(df, True)
        elif diagram == 'tab':
            for nameColumn in df.columns:
                print(f'{nameColumn}')
        else:
            print("Il faut choisir un diagramme pour déssiner.")
            draw_Diagram(df)
        print("Continuer ? (n/y)")
        next = True if input() == "y" else False
        plt.close()


def pareto_dominance(row1, row2, objectives, maximize=[]):
    better_in_any = False
    for obj in objectives:
        if obj in maximize:
            better_in_any |= row1[obj] >= row2[obj]
        else:
            better_in_any |= row1[obj] <= row2[obj]
    return better_in_any


def pareto_frontier(df):
    objectifs = ["Prix", "Conso_Moy", "Dis_Freinage", "Confort", "Acceleration", "Vitesse_Max", "Vol_Coffre"]
    dominated = set()
    pareto_front = []

    num_columns = len(objectifs)
    fig, axes = plt.subplots(nrows=num_columns, ncols=1, figsize=(8, 2 * num_columns))

    for i, obj in enumerate(objectifs):
        for j, row1 in df.iterrows():
            is_dominated = False
            for k, row2 in df.iterrows():
                if j != k and pareto_dominance(row2, row1, objectifs, maximize=["Vitesse_Max", "Vol_Coffre"]):
                    is_dominated = True
                    dominated.add(k)  # Mark as dominated
                    break
            if not is_dominated:
                pareto_front.append(row1)

        pareto_front_df = pd.DataFrame(pareto_front, columns=all_columns)

        # Plot non-dominated points
        axes[i].scatter(df[obj], df["Voiture"], label="Non optimal", color="blue")

        # Plot Pareto frontier
        axes[i].scatter(pareto_front_df[obj], pareto_front_df["Voiture"], label="Optimal (Pareto)", color="red")
        axes[i].set_xlabel(obj)
        axes[i].set_ylabel("Voiture")
        axes[i].legend()

    plt.tight_layout()
    plt.show()

    print("Pareto :")
    print(pareto_front_df)


def preference_functionPromethee(x, y, value, maximize=True):
    # Maximiser ou minimiser

    if maximize:
        if x > y:
            return value
    else:
        if x < y:
            return value
    return 0


def preference_functionElectre(x, y, value, maximize=True):
    if maximize:
        if x >= y:
            return value
    else:
        if x <= y:
            return value
    return 0

def preference_functionElectreIs(x, y, value, threshold, maximize=True):
    if maximize:
        if x >= y:
            return value
        elif x - y > -threshold:
            return value * (1 - abs(x - y)/threshold)
    else:
        if x <= y:
            return value
        elif y - x > -threshold:
            return value * (1 - abs(x - y)/threshold)
    return 0


def calculatePromethee(col, df, i, j, poids, preference_matrix, maximize):
    preference_matrix[i, j] += preference_functionPromethee(
        df.iloc[i][col], df.iloc[j][col], poids[col], maximize=maximize
    )
    preference_matrix[j, i] += preference_functionPromethee(
        df.iloc[j][col], df.iloc[i][col], poids[col], maximize=maximize
    )


def calculateElectre(col, df, i, j, poids, preference_matrix, maximize):
    preference_matrix[i, j] += preference_functionElectre(
        df.iloc[i][col], df.iloc[j][col], poids[col], maximize=maximize
    )
    preference_matrix[j, i] += preference_functionElectre(
        df.iloc[j][col], df.iloc[i][col], poids[col], maximize=maximize
    )


def calculateElectreIS(col, df, i, j, poids, thresholds, concordance_matrix, maximize):
    threshold = thresholds.get(col, 0)
    #print(col , i , j)
    concordance_matrix[i, j] += preference_functionElectreIs(
        df.iloc[i][col], df.iloc[j][col], poids[col], threshold, maximize=maximize
    )
    concordance_matrix[j, i] += preference_functionElectreIs(
        df.iloc[j][col], df.iloc[i][col], poids[col], threshold, maximize=maximize
    )


def veto_function(x, y, seuil_veto, maximize=True):
    if maximize:
        if y - x > seuil_veto:
            return 0
    else:
        if x - y > seuil_veto:
            return 0
    return 1


def calculateVeto(col, df, i, j, seuils_veto, non_discordance_matrix, maximize):
    non_discordance_matrix[i, j] *= veto_function(
        df.iloc[i][col], df.iloc[j][col], seuils_veto[col], maximize=maximize
    )
    non_discordance_matrix[j, i] *= veto_function(
        df.iloc[j][col], df.iloc[i][col], seuils_veto[col], maximize=maximize
    )


def calculConcordanceMatrix(i, j, concordance_matrix, non_discordance_matrix, seuil):
    concordance_matrix[i, j] = concordance_matrix[i, j] if non_discordance_matrix[i, j] == 1 and concordance_matrix[i, j] >= seuil else 0
    concordance_matrix[j, i] = concordance_matrix[j, i] if non_discordance_matrix[j, i] == 1 and concordance_matrix[j, i] >= seuil else 0


def get_weights():
    #poids = {"Prix": 0.1, "Vitesse_Max": 0.009, "Conso_Moy": 0.1, "Dis_Freinage": 0.5, "Confort": 0.2,
    #         "Vol_Coffre": 0.09, "Acceleration": 0.001} # Personne prudente
    #poids = {"Prix": 0.6, "Vitesse_Max": 0.063, "Conso_Moy": 0.2, "Dis_Freinage": 0.01, "Confort": 0.001,
    #         "Vol_Coffre": 0.063, "Acceleration": 0.063} # Etudiant peu riche
    poids = {"Prix": 0.0125, "Vitesse_Max": 0.4, "Conso_Moy": 0.0125, "Dis_Freinage": 0.0125, "Confort": 0.2,
             "Vol_Coffre": 0.0125, "Acceleration": 0.35} # riche
    #poids = {"Prix": 0.1, "Vitesse_Max": 0.02, "Conso_Moy": 0.08, "Dis_Freinage": 0.08, "Confort": 0.4,
    #         "Vol_Coffre": 0.3, "Acceleration": 0.02}  # familial
    # poids = {"C1": 0.1, "C2": 0.2, "C3": 0.2, "C4": 0.1, "C5": 0.2, "C6": 0.2}
    # Decommenter le code pour un profil different
    # for col in ["Prix", "Vitesse_Max", "Conso_Moy", "Dis_Freinage", "Confort", "Vol_Coffre", "Acceleration"]:
    #     weight = float(input(f"Enter weight for {col}: "))
    #     weights[col] = weight
    return poids


def get_vetos():
    # Définir vos seuils de veto pour chaque critère
    vetos = {"Prix": 5000, "Vitesse_Max": 50, "Conso_Moy": 2, "Dis_Freinage": 5, "Confort": 2,
             "Vol_Coffre": 100, "Acceleration": 3}
    # vetos = {"C1": 45, "C2": 29, "C3": 550, "C4": 6, "C5": 4.5, "C6": 4.5}
    # vetos = {"Prix": 6000, "Vitesse_Max": 75, "Conso_Moy": 1.3, "Dis_Freinage": 2.5, "Confort": 3,
    #          "Vol_Coffre": 50, "Acceleration": 2}
    # Decommenter le code pour un profil different
    # for col in ["Prix", "Vitesse_Max", "Conso_Moy", "Dis_Freinage", "Confort", "Vol_Coffre", "Acceleration"]:
    #     veto = float(input(f"Enter weight for {col}: "))
    #     vetos[col] = veto
    return vetos

def get_thresholds():
    #thresholds = {"C1": 20, "C2": 10, "C3": 200, "C4": 4, "C5": 2, "C6": 2}
    thresholds = {"Prix": 1500, "Vitesse_Max": 10, "Conso_Moy": 0.8, "Dis_Freinage": 1.5, "Confort": 0.8,
             "Vol_Coffre": 30, "Acceleration": 1.2}
    # Decommenter le code pour un profil different
    # for col in ["Prix", "Vitesse_Max", "Conso_Moy", "Dis_Freinage", "Confort", "Vol_Coffre", "Acceleration"]:
    #     threshold = float(input(f"Enter weight for {col}: "))
    #     thresholds[col] = threshold
    return thresholds


def promethee(df, isPrometheeII):
    minimize = ["Prix", "Conso_Moy", "Dis_Freinage", "Confort", "Acceleration"]
    maximize = ["Vitesse_Max", "Vol_Coffre"]
    alternatives = df['Voiture'].tolist()
    # minimize = ["C1", "C3", "C4", "C5"]
    # maximize = ["C2", "C6"]
    # alternatives = df['C0'].tolist()
    num_alternatives = len(alternatives)
    preference_matrix = np.zeros((num_alternatives, num_alternatives))

    poids = get_weights()

    for i, j in combinations(range(num_alternatives), 2):
        for col in minimize:
            # Les colonnes à minimiser
            calculatePromethee(col, df, i, j, poids, preference_matrix, False)

        for col in maximize:
            # Les colonnes à maximiser
            calculatePromethee(col, df, i, j, poids, preference_matrix, True)

    # Flux positif et négatif
    flux_positif = preference_matrix.sum(axis=1)
    flux_negatif = preference_matrix.sum(axis=0)

    # Flux
    flux = flux_positif - flux_negatif

    if isPrometheeII:
        # Classement
        classement = pd.Series(flux).rank(ascending=False).astype(int)
        result = pd.DataFrame(
            {'Voiture': alternatives, 'Flux positif': flux_positif, 'Flux négatif': flux_negatif,
             'Flux net': flux, 'Classement': classement})
        print("Promethee II :")
    else:
        # Classement Positif
        classement_flux_positif = pd.Series(flux_positif).rank(ascending=False).astype(int)
        # Classement Négatif
        classement_flux_negatif = pd.Series(flux_negatif).rank(ascending=True).astype(int)
        result = pd.DataFrame(
            {'Voiture': alternatives, 'Flux positif': flux_positif, 'Classement Flux positif': classement_flux_positif,
             'Flux négatif': flux_negatif
                , 'Classement flux négatif': classement_flux_negatif})
        print("Promethee I :")

    print(preference_matrix)
    print(result)
    # generateGraphe(result)


def electre(df, isElectreIS):
    minimize = ["Prix", "Conso_Moy", "Dis_Freinage", "Confort", "Acceleration"]
    maximize = ["Vitesse_Max", "Vol_Coffre"]
    alternatives = df['Voiture'].tolist()
    # minimize = ["C1", "C3", "C4", "C5"]
    # maximize = ["C2", "C6"]
    # alternatives = df['C0'].tolist()
    num_alternatives = len(alternatives)
    concordance_matrix = np.zeros((num_alternatives, num_alternatives))
    non_discordance_matrix = np.ones((num_alternatives, num_alternatives))

    poids = get_weights()
    seuils_veto = get_vetos()

    if isElectreIS:
        thresholds = get_thresholds()

    for i, j in combinations(range(num_alternatives), 2):
        for col in minimize:
            if not isElectreIS:
                calculateElectre(col, df, i, j, poids, concordance_matrix, False)
            else:
                calculateElectreIS(col, df, i, j, poids, thresholds, concordance_matrix, False)
            calculateVeto(col, df, i, j, seuils_veto, non_discordance_matrix, False)

        for col in maximize:
            if not isElectreIS:
                calculateElectre(col, df, i, j, poids, concordance_matrix, True)
            else:
                calculateElectreIS(col, df, i, j, poids, thresholds, concordance_matrix, True)
            calculateVeto(col, df, i, j, seuils_veto, non_discordance_matrix, True)

    for i in range(0, num_alternatives):
        non_discordance_matrix[i, i] *= 0

    print("Matrice de préférence :")
    print(concordance_matrix)

    for i, j in combinations(range(num_alternatives), 2):
        calculConcordanceMatrix(i, j, concordance_matrix, non_discordance_matrix, 0.6)

    if isElectreIS:
        # Classement
        # classement = pd.Series(flux).rank(ascending=False).astype(int)
        result = pd.DataFrame(
            {'Voiture': alternatives})
        print("Electre IS :")
    else:
        result = pd.DataFrame(
            {'Voiture': alternatives})
        print("Electre IV :")

    print("Tableau de veto :")
    print(non_discordance_matrix)
    print("Matrice de préférence :")
    print(concordance_matrix)
    generate_surclassement_graph(df, concordance_matrix, non_discordance_matrix, 0.6)
    print(result)

def generate_surclassement_graph(df, concordance_matrix, non_discordance_matrix, seuil_concordance):
    G = nx.DiGraph()
    alternatives = df['Voiture'].tolist()

    # Assurez-vous que tous les nœuds sont ajoutés au graphe
    for alternative in alternatives:
        G.add_node(alternative)

    num_alternatives = len(alternatives)
    for i in range(num_alternatives):
        for j in range(num_alternatives):
            if i != j:  # Pas de boucles
                # Vérifiez si l'alternative i surclasse l'alternative j
                if concordance_matrix[i, j] >= seuil_concordance and non_discordance_matrix[i, j] == 1:
                    G.add_edge(alternatives[i], alternatives[j])

    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw(G, pos, with_labels=True, node_size=7000, node_color='skyblue', font_size=10, font_weight='bold', arrowstyle='-|>', arrowsize=10)

    # Créer les labels des arcs après la création du graphe pour éviter les KeyError
    edge_labels = {(u, v): f'{concordance_matrix[alternatives.index(u), alternatives.index(v)]:.2f}'
                   for u, v in G.edges()}

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Graphe de Surclassement")
    plt.show()

# Usage example, assuming that the matrices and dataframe have been defined
# generate_surclassement_graph(df, concordance_matrix, non_discordance_matrix, 0.6)


def generateGraphe(df):
    G = nx.DiGraph()

    for voiture in df['Voiture']:
        G.add_node(voiture)

    for i, row1 in df.iterrows():
        for j, row2 in df.iterrows():
            if row1['Classement Flux positif'] < row2['Classement Flux positif'] and row1['Classement flux négatif'] < \
                    row2['Classement flux négatif']:
                G.add_edge(row1['Voiture'], row2['Voiture'])

    # Dessiner le graphe d'ordre partiel
    plt.figure(figsize=(12, 8))
    nx.draw(G, with_labels=True, node_color='lightblue', node_size=3000, edge_color='darkblue', linewidths=1,
            font_size=12, arrowsize=20)
    plt.title("Partial Order Graph based on Positive and Negative Flows Rankings")
    plt.show()


def get_Top(df, column, top):
    if top:
        data = df.nlargest(N, column)
        plt.title(f'Top {N} pour les valeurs de la colonne {column}')
    else:
        data = df.nsmallest(N, column)
        plt.title(f'Bottom {N} pour les valeurs de la colonne {column}')

    plt.barh(data['Voiture'][::-1], data[column][::-1])
    for index, value in enumerate(data[column][::-1]):
        plt.text(value, index, f' {value:.2f}', va='center', color='black')

    plt.tight_layout()
    plt.show()
    print(data)


def get_Min_Value(df, column):
    return df[column].min()


def get_Max_Value(df, column):
    return df[column].max()


def get_Mean_Value(df, column):
    return df[column].mean()


def get_Standard_deviation(df, column):
    return df[column].std()


def get_median(df, column):
    return df[column].median()


if __name__ == "__main__":
    input_csv_path = sys.argv[1]
    read_csv(input_csv_path)
