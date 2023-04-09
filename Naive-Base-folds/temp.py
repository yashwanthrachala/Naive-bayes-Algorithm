import numpy as np
import random


def main():
    metafilename = ""
    trainfilename = ""
    testfilename = ""
    outputfilename = ""
    priorprobabilities = dict()
    posteriorprobability = dict()
    trainset = list()
    trainsetdup = list()
    testset = list()
    metaset = list()
    flag=0
    actual_labels = list()
    predicted_labels = list()
    meta = list()
    
    while(1):
        print("Enter 1 to build model")
        print("Enter 2 to classify")
        print("Enter 3 to fing acc and confusion matrix")
        print("Enter 4 to k-fold cross validation")
        print("Enter 5 for stratified k-fold cross validation")
        print("Enter 6 for exit")
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
            meta = metaset[len(metaset)-1]
            meta = meta[1:]
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
                if row[-1] in classcount:
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
                if (c not in priorprobabilities):
                    continue
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
            
            count=0
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
                count=count+1
                f.write(s)
                f.close()
                             
        if (choice == 3):
            print("Enter test file for acc")
            testfilename = input()
            testset = list()
            count = 0
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
                actual_labels.append(actual)
                predicted_labels.append(predicted)
                if (actual == predicted):
                    count += 1;
            acc = 100 * (count / len(testset))
            print("acc is :")
            print(acc)
            actual_indices = [meta.index(label) for label in actual_labels]
            predicted_indices = [meta.index(label) for label in predicted_labels]
            confusion_matrix = np.zeros((len(meta), len(meta)), dtype=int)

            for a, p in zip(actual_indices, predicted_indices):
             confusion_matrix[a][p] += 1
            
            print("Confusion Matrix:")
            print("------------------")
            print("Actual:")
            for i in range(len(confusion_matrix)):
             row_str = "| "
             for j in range(len(confusion_matrix[i])):
              row_str += str(confusion_matrix[i][j]) + " | "
             print(row_str)
            print("------------------")
            actual_labels = list()
            predicted_labels = list()
           
           
            


        if (choice == 6):
            break;
        
        if (choice == 4):
            priorprobabilities = dict()
            posteriorprobability = dict()
            trainset = list()
            testset = list()
            metaset = list()
            total=0
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
            print("Enter the k value:")
            l=0
            k_string = input()
            k=int(k_string)
            length = len(trainset)
            p = int(length/k)
            d=p
            trainsetdup = list(trainset)
            for x in range(k):
                sub_train = list(trainsetdup)
                sub_test = sub_train[l:p]
                del sub_train[l:p]
                l=p
                p=p+d
                classcount = dict()
                for j in metaset[-1][1:]:
                    classcount[j] = 0
                for row in sub_train:
                    classcount[row[-1]] += 1
                for c in classcount:
                    priorprobabilities[c] = (classcount[c]/len(sub_train))
                
                allcounts = dict()
                for row in sub_train:
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
                
                
                for row in sub_train:
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
                
                
                testset = sub_test
                count = 0
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
                total = total + acc
                print("fold" +str(x+1)+" : "+ str(acc))
            print("")        
            print("Total Avg : ")
            print(total/k)
            priorprobabilities = dict()
            posteriorprobability = dict()
            trainset = list()
            testset = list()
            metaset = list()
        if(choice==5):
         total=0
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
         X = []
         y = []
         print("Enter K value")
         k= int (input())
         # iterate over each list in the data
         for lst in trainset:
          # extract the values except for the last value
          X.append(lst[:-1])
          # extract the last value and store it in the last_values list
          y.append(lst[-1])
         random_state=False
         random.seed(random_state)
         data = list(zip(X, y))
         random.shuffle(data)
         stratified_folds = [[] for _ in range(k)]

         for label in np.unique(y):
             label_data = [x for x in data if x[1] == label]
             fold_size = len(label_data) // k
             remaining = len(label_data) % k

             start = 0
             for i in range(k):
                 end = start + fold_size + (1 if remaining > 0 else 0)
                 stratified_folds[i].extend(label_data[start:end])

                 start = end
                 remaining -= 1
        
         for l in range(len(stratified_folds)):
          validation_set = stratified_folds[l]
          training_set = [item for j, fold in enumerate(stratified_folds) for item in fold if j != l]
          training_set = [attributes + [label] for attributes, label in training_set]
          validation_set = [attributes + [label] for attributes, label in validation_set]
          classcount = dict()
          for j in metaset[-1][1:]:
              classcount[j] = 0
          for row in training_set:
              classcount[row[-1]] += 1
          for c in classcount:
              priorprobabilities[c] = (classcount[c]/len(training_set))
          
          allcounts = dict()
          for row in training_set:
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
          
          
          for row in training_set:
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
          
          testset = validation_set
          count = 0
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
          print("fold "+str(l+1)+" : "+str(acc))
          total = total + acc
         print("-------------------------")
         print("average : "+str(total/k))
         print("-------------------------")
         priorprobabilities = dict()
         posteriorprobability = dict()
         trainset = list()
         trainsetdup = list()
         testset = list()
         metaset = list()
          
                  
main()









