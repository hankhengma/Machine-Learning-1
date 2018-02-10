# EECS 445 - Winter 2018
# Project 1 - test_output.py

import sys
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='Input data file. Must be in csv.')


def main():
    args = parser.parse_args()
    input_file = args.input

    if input_file[-4:] != '.csv':
        print('INVALID FILE: file must be of .csv format.')
        sys.exit()

    with open(input_file, newline='') as csvfile:
        filereader = csv.reader(csvfile)
        i=0
        for row in filereader:
            if i == 0:
                if row[0] != 'rating':
                    print('INVALID COLUMN NAME: column name is not rating.')
                    sys.exit()
            else:
                rating = int(row[0])
                if rating != -1 and rating != 0 and rating != 1:
                    print('INVALID VALUE: values need to be -1, 0, or 1.')
                    sys.exit()
            i += 1
        if i != 751:
            print('INVALID NUMBER OF ROWS: number of rows is not 751.')
            sys.exit()
        print('SUCCESS: csv file is valid.')
    return


if __name__ == '__main__':
    main()
