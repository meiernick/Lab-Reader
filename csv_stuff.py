#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import numpy as np


def write_to_csv(data, sep=';', filename=r".\my.csv",):
    """
    Write the Table, List or Value from the input data to a CSV-File
    """

    # If Data has wrong dimensions try to correct them
    if len(np.shape(data)) == 0:
        data = [[data]]
    elif len(np.shape(data)) == 1:
        data = [data]

    # Write to the File
    with open(filename, "a+") as csvFile:
        for row in data:
            if len(row) == 0:
                csvFile.write('\n')
                continue
            csvFile.write(str(row[0]))  # first cell without separator in front
            for cell in row[1:]:
                csvFile.write(sep)
                csvFile.write(str(cell).replace('\n', ''))
            csvFile.write('\n')


def saveValue(numbers, filename=r'my.csv'):
    """
    Save the Value to a csv-File
    """
    numbers = ['' if i==[] else i for i in numbers]
    write_to_csv([numbers], filename=filename)
    print('Saved the Value ({})'.format(numbers))


def removeValue(lines=1, filename=r".\my.csv"):
    """
    Remove the last row of the csv-File
    """

    # Read the hole file
    with open(filename, "r") as csvFile:
        fileContent = csvFile.readlines()

    # Write the hole file without last line
    with open(filename, 'w') as csvFile:
        csvFile.writelines(fileContent[:-lines])

    print('Deleted last line ({})'.format(
        ''.join(fileContent[-lines:]).replace('\n', '')))


def main(argv):
    """ Main program """
    pass


if __name__ == '__main__':
    sys.exit(main(sys.argv))
