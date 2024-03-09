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
    print(f"Les diagrammes disponibles sont : {diagrams}")
    print("Veuillez choisir un type de diagramme : ")
    return input().lower()


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
            continue
        elif diagram == diagrams[4]:
            continue
        elif diagram == diagrams[5]:
            continue
        elif diagram == diagrams[6]:
            continue
        elif diagram == 'tab':
            for nameColumn in df.columns:
                print(f'{nameColumn}')
        else:
            print("Il faut choisir un diagramme pour déssiner.")
        print("Continuer ? (n/y)")
        next = True if input() == "y" else False


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
    objectives = ["Prix", "Conso_Moy", "Dis_Freinage", "Confort", "Acceleration", "Vitesse_Max", "Vol_Coffre"]
    dominated = set()
    pareto_front = []

    # Create subplots
    num_columns = len(objectives)
    fig, axes = plt.subplots(nrows=num_columns, ncols=1, figsize=(8, 2 * num_columns))

    for i, obj in enumerate(objectives):
        for j, row1 in df.iterrows():
            is_dominated = False
            for k, row2 in df.iterrows():
                if j != k and pareto_dominance(row2, row1, objectives, maximize=["Vitesse_Max", "Vol_Coffre"]):
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

    print("Pareto Frontier:")
    print(pareto_front_df)


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
