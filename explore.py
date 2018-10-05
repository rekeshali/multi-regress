#!/usr/bin/ python

# open, get, close data files
datafile = open("data/auto-mpg.data", "r") # opens for reading
namefile = open("data/auto-mpg.names", "r")

if datafile.mode == 'r' and namefile.mode == "r":
    book = datafile.read() # reads all as on string
    card = namefile.read()

datafile.close() # close for good practice
namefile.close()

# get keys for dict
i = 0
db = {} # initiate dict
key = [] # initiate list
card = card.split('\n') # split long string by new line
for line in card:
    key.append(line.split(':')[0].split(' ',5)[-1]) # get dict keys
    db[key[i]] = [] # initiate key:value pairs in dict
    i = i + 1

# macro to check for data format
def is_num(string): # returns true if number, false otherwise
    try:
        float(string)
        return True
    except ValueError:
        return False

# add values to dict
bad_entry = {}
nameidx = key.index('car name')
book = book.split('\n')
for line in book:
    for tooth in key:
        if tooth == key[nameidx]: # name is separated by \t
            db[tooth].append(line.split('\t')[1].split('"')[1])
        else:
            value = line.split('\t')[0].split()[key.index(tooth)] # get value according to key index
            if is_num(value): # paste if a number
                db[tooth].append(float(value))
            else: # non-numbers mean bad data, so replace for now and save location
                if tooth not in bad_entry.keys():
                    bad_entry[tooth] = []
                db[tooth].append(-1)
                bad_entry[tooth].insert(0, len(db[tooth]) - 1)

# # relocating all bad entries to end
# for badidx in bad_entry:
#     for tooth in key:
#         db[tooth].append(db[tooth][badidx])
#         db[tooth].pop(badidx)
# lbad = len(bad_entry) # for list navigation# macro for quickly checking car information

def car(idx):
    if isinstance(idx, str): # if given name instead of index get index
        idx = db[key[nameidx]].index(idx)

    print(key[nameidx] + ': ' + db[key[nameidx]][idx]) # print key:value pairs
    for validx in range(0, len(key) - 1):
        print(key[validx] + ': ' + str(db[key[validx]][idx]))

# it seems that the anomolous values are horsepower only. the uniformity could indicate that rather than making a mistake on entry, the values are simply unknown
