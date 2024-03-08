import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plotter
import math

all_columns = []
diagrams = ["top 5", "worst 5"]
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
        elif diagram == 'tab':
            for nameColumn in df.columns:
                print(f'{nameColumn}')
        else:
            print("Il faut choisir un diagramme pour déssiner.")
        print("Continuer ? (n/y)")
        next = True if input() == "y" else False


def get_Top(df, column, index):
    top = df.nlargest(n=N, columns=[column]) if index else df.nsmallest(n=N, columns=[column])
    y = top[column].to_list()
    plotter.barh(top['Voiture'], y)
    table = pd.DataFrame(data=top.values, columns=all_columns)
    print(table)
    plotter.show()


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


def generateFile(nom_fichier, filename, content):
    # Enregistrer les résultats dans un fichier CSV
    extension = 'output'
    nom_fichier_sortie = f'Output\\{nom_fichier}\\{nom_fichier}_{filename}_{extension}.csv'

    isExist = os.path.exists(sys.path[0] + f'\\Output\\{nom_fichier}')
    if not isExist:
        os.makedirs(sys.path[0] + f'\\Output\\{nom_fichier}')

    content.to_csv(sys.path[0] + f'\\{nom_fichier_sortie}', index=False)
    print(f"Résultats enregistrés dans {nom_fichier_sortie}")


if __name__ == "__main__":
    input_csv_path = sys.argv[1]
    read_csv(input_csv_path)
