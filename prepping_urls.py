import random
import csv
Benign =[]
Malicious = []
Count = 0


with open("urls.csv") as File:
    try:
        for line in File:
            Count += 1
            print(Count)
            print(line)
            line = line.split(',')
            if int(line[1]) == -1:
                Benign.append(line)
            else:
                Malicious.append(line)
    except:
        Count+=1
        print(Count)

#random.shuffle(Benign)
random.shuffle(Malicious)
print(len(set(URL[0] for URL in Malicious)))


with open("PreppedURLs.csv", 'w') as WriteFile:
    for URL in Benign[:5000]:
        WriteFile.write(URL[0] + ',' + URL[1])

    for URL in Malicious[:5000]:
        WriteFile.write(URL[0] + ',' + URL[1])

