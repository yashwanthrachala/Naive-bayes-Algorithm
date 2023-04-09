def main():
    metafilename = ""
    trainfilename = ""
    testfilename = ""
    outputfilename = ""
    priorprobabilities = dict()
    posteriorprobability = dict()
    trainset = list()
    testset = list()
    metaset = list()
    flag=0
    
    while(1):
        print("Enter 1 to build model")
        print("Enter 2 to classify")
        print("Enter 3 to fing acc")
        print("Enter 4 to exit")
        choice = int(input())
        if(choice == 1):
            flag=flag+1
            if flag>0:
             priorprobabilities = dict()
             posteriorprobability = dict()
             trainset = list()
             testset = list()
             metaset = list()
            print("Enter meta file name: ")
            metafilename = input()
            f = open(metafilename, "r")
            for line in f:
                newline = line.replace(":", ",")
                newline = newline.replace("\n", "")
                metaset.append(list(newline.split(",")))
            f.close()
            
            print("Enter training file name: ")
            trainfilename = input()
            f = open(trainfilename, "r")
            for line in f:
                newline = line.replace("\n", "")
                trainset.append(list(newline.split(",")))
            
            classcount = dict()
            for j in metaset[-1][1:]:
                classcount[j] = 0
            for row in trainset:
                classcount[row[-1]] += 1
            for c in classcount:
                priorprobabilities[c] = (classcount[c]/len(trainset))
            
            allcounts = dict()
            for row in trainset:
                c = row[-1]
                if (c not in allcounts):
                    allcounts[c] = dict()
                for idx, i in enumerate(row[:-1]):
                    if (metaset[idx][0] not in allcounts[c]):
                        allcounts[c][metaset[idx][0]] = dict()
                    allcounts[c][metaset[idx][0]][i] = 1
                    
            for c in metaset[-1][1:]:
                if (c not in allcounts):
                    allcounts[c] = dict()
                for idx, i in enumerate(metaset[:-1]):
                    if (i[0] not in allcounts[c]):
                        allcounts[c][i[0]] = dict()
                    for j in i[1:]:
                        allcounts[c][i[0]][j] = 1
            
            
            for row in trainset:
                c = row[-1]
                for idx, i in enumerate(row[:-1]):
                    allcounts[c][metaset[idx][0]][i] += 1
            
            for c in allcounts:
                if (c not in posteriorprobability):
                    posteriorprobability[c] = dict()
                for i in allcounts[c]:
                    if (i not in posteriorprobability[c]):
                        posteriorprobability[c][i] = dict()
                    for j in allcounts[c][i]:
                        a = allcounts[c][i][j]
                        l1 = (classcount[c])
                        l2 = len(allcounts[c][i].keys())
                        b = l1 + l2
                        posteriorprobability[c][i][j] = a / b
            
            print("\nModel has been trained (calculated probabilities)\n")
                    
        if(choice == 2):
            print("Emter input file name to test")
            testfilename = input()
            print("Enter output file name to store")
            outputfilename = input()
            f = open(testfilename, "r")
            for line in f:
                newline = line.replace("\n", "")
                testset.append(list(newline.split(",")))
            
            features = list()
            for line in metaset:
                features.append(line[0])
            features = features[:-1]
            
            
            for row in testset:
                if (len(row) == len(metaset)):
                    row = row[:-1]
                
                maxProb = -1
                probabilities = dict()
                predicted = None
                for key in posteriorprobability:
                    probabilities[key] = 1
                    for i, val in enumerate(row):
                        probabilities[key] *= posteriorprobability[key][features[i]][val]
                    probabilities[key] = priorprobabilities[key] * probabilities[key]
                    if probabilities[key] > maxProb:
                        maxProb = probabilities[key]
                        predicted = key
                f = open(outputfilename, "a")
                s = ",".join(row)
                s = s + ',' + predicted + '\n'
                f.write(s)
                f.close()
                             
        if (choice == 3):
            print("Enter test file for acc")
            testfilename = input()
            testset = list()
            count = 1
            f = open(testfilename, "r")
            for line in f:
                newline = line.replace("\n", "")
                testset.append(list(newline.split(",")))
            features = list()
            for line in metaset:
                features.append(line[0])
            features = features[:-1]
            for row in testset:
                actual = row[-1]
                if (len(row) == len(metaset)):
                    row = row[:-1]
                maxProb = -1
                probabilities = dict()
                predicted = None
                for key in posteriorprobability:
                    probabilities[key] = 1
                    for i, val in enumerate(row):
                        probabilities[key] *= posteriorprobability[key][features[i]][val]
                    probabilities[key] = priorprobabilities[key] * probabilities[key]
                    if probabilities[key] > maxProb:
                        maxProb = probabilities[key]
                        predicted = key
                if (actual == predicted):
                    count += 1;
            acc = 100 * (count / len(testset))
            print("acc is :")
            print(acc)            
        if (choice == 4):
            break;
main()








