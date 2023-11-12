'''
Program to read different datasets
For each class a data function to read the data and a getXy function to get the X and y values are defined
This way, different datasets can be used by only changing the class definition
Important: datasets used are from the UCI Machine Learning Repository, datasets should be downloaded from there and path to CSV files should be entered
'''

from os import get_terminal_size
import pandas as pd
import numpy as np
import calendar


class winequality_red:
    data = pd.read_csv('PATH-TO-DATASET', delimiter=';')

    def getXy(data):
        y = data['quality'].values
        X = data.drop(['quality'], axis=1).values

        return X, y


class winequality_white:
    data = pd.read_csv('PATH-TO-DATASET', delimiter=';')

    def getXy(data):
        y = data['quality'].values
        X = data.drop(['quality'], axis=1).values

        return X, y


class forestfires:

    def getData():

        days_dict = dict(zip(calendar.day_abbr, range(7)))
        days_dict = {k.lower(): v for k, v in days_dict.items()}

        months_dict = dict(zip(calendar.month_abbr, range(13)))
        months_dict = {k.lower(): v for k, v in months_dict.items()}

        data = pd.read_csv('PATH-TO-DATASET')

        data['day'] = data['day'].map(days_dict)
        data['month'] = data['month'].map(months_dict)

        return data

    data = getData()

    def getXy(data):
        y = data['area'].values
        X = data.drop(['area'], axis=1).values

        return X, y


class abalone:
    def getData():
        sex_dict = {'I': 0.0, 'F': 0.5, 'M': 1.0}

        data = pd.read_csv('PATH-TO-DATASET', header=None)

        data[0] = data[0].map(sex_dict)

        return data

    data = getData()

    def getXy(data):
        y = data[8].values
        X = data.drop([8], axis=1).values

        return X, y


class obesity:
    def getData():
        yesno_dict = {'yes': 1, 'no': 0}
        gender_dict = {'Male': 1, 'Female': 0}
        frequency_dict = {'no': 0, 'Sometimes': 1,
                          'Frequently': 2, 'Always': 3}
        transport_dict = {'Automobile': 0, 'Motorbike': 1,
                          'Public_Transportation': 2, 'Walking': 3, 'Bike': 4}

        obesity_dict = {'Insufficient_Weight': 0, 'Normal_Weight': 1, 'Overweight_Level_I': 2,
                        'Overweight_Level_II': 3, 'Obesity_Type_I': 4, 'Obesity_Type_II': 5, 'Obesity_Type_III': 6}

        data = pd.read_csv('PATH-TO-DATASET')

        data['Gender'] = data['Gender'].map(gender_dict)
        data['family_history_with_overweight'] = data['family_history_with_overweight'].map(
            yesno_dict)
        data['FAVC'] = data['FAVC'].map(yesno_dict)
        data['CAEC'] = data['CAEC'].map(frequency_dict)
        data['SMOKE'] = data['SMOKE'].map(yesno_dict)
        data['SCC'] = data['SCC'].map(yesno_dict)
        data['CALC'] = data['CALC'].map(frequency_dict)
        data['MTRANS'] = data['MTRANS'].map(transport_dict)
        data['NObeyesdad'] = data['NObeyesdad'].map(obesity_dict)

        return data

    data = getData()

    def getXy(data):
        y = data['Weight'].values
        X = data.drop(['Weight', 'NObeyesdad'], axis=1).values

        return X, y
