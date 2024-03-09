import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from itertools import combinations

all_columns = []
diagrams = ["top 5", "worst 5", "pareto", "promethee i", "promethee ii", "electre iv", "electre is"]
N = 10


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
        column = read_user_column(df)
        diagram = choose_Diagram()
        if diagram == diagrams[0]:
            get_Top(df, column, True)
        elif diagram == diagrams[1]:
            get_Top(df, column, False)
        elif diagram == diagrams[2]:
            pareto_frontier(df)
        elif diagram == diagrams[3]:
            promethee(df, False)
        elif diagram == diagrams[4]:
            promethee(df, True)
        elif diagram == diagrams[5]:
            continue
        elif diagram == diagrams[6]:
            continue
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
    # Check if row1 dominates row2 in terms of objectives
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

    # Create subplots
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


def preference_function(x, y, maximize=True):
    # Maximiser ou minimiser
    if x == y:
        return 0
    elif maximize:
        return x - y
    else:
        return y - x


def calculatePromethee(col, df, has_weight, i, j, poids, preference_matrix, maximize):
    if has_weight:
        preference_matrix[i, j] += poids[col] * preference_function(
            df.iloc[i][col], df.iloc[j][col], maximize=maximize
        )
        preference_matrix[j, i] += poids[col] * preference_function(
            df.iloc[j][col], df.iloc[i][col], maximize=maximize
        )
    else:
        preference_matrix[i, j] += preference_function(
            df.iloc[i][col], df.iloc[j][col], maximize=maximize
        )
        preference_matrix[j, i] += preference_function(
            df.iloc[j][col], df.iloc[i][col], maximize=maximize
        )


def get_weights():
    poids = {"Prix": 0.00, "Vitesse_Max": 0.09, "Conso_Moy": 0.1, "Dis_Freinage": 0.05, "Confort": 1,
             "Vol_Coffre": 0.2, "Acceleration": 0.00}
    # for col in ["Prix", "Vitesse_Max", "Conso_Moy", "Dis_Freinage", "Confort", "Vol_Coffre", "Acceleration"]:
    #     weight = float(input(f"Enter weight for {col}: "))
    #     weights[col] = weight
    return poids


def promethee(df, has_weight):
    minimize = ["Prix", "Conso_Moy", "Dis_Freinage", "Confort", "Acceleration"]
    maximize = ["Vitesse_Max", "Vol_Coffre"]
    alternatives = df['Voiture'].tolist()
    num_alternatives = len(alternatives)
    preference_matrix = np.zeros((num_alternatives, num_alternatives))

    poids = get_weights()

    for i, j in combinations(range(num_alternatives), 2):
        for col in minimize:
            # Les colonnes à minimiser
            calculatePromethee(col, df, has_weight, i, j, poids, preference_matrix, False)

        for col in maximize:
            # Les colonnes à maximiser
            calculatePromethee(col, df, has_weight, i, j, poids, preference_matrix, True)

    # Flux positif et négatif
    flux_positif = preference_matrix.sum(axis=1)
    flux_negatif = preference_matrix.sum(axis=0)

    # Flux
    flux = flux_positif - flux_negatif

    # Classement
    classement = np.argsort(flux)[::-1]

    result = pd.DataFrame(
        {'Voiture': alternatives, 'Flux positif': flux_positif, 'Flux négatif': flux_negatif,
         'Flux (flux positif - flux négatif)': flux, 'Classement': classement + 1})

    if has_weight:
        print("Promethee II :")
    else:
        print("Promethee I :")
    print(result)


def get_Top(df, column, index):
    top = df.nlargest(n=N, columns=[column]) if index else df.nsmallest(n=N, columns=[column])
    y = top[column].to_list()
    plt.barh(top['Voiture'], y)
    table = pd.DataFrame(data=top.values, columns=all_columns)
    print(table)
    plt.show()


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
