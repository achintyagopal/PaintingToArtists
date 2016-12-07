# import os

# with open('results.txt', 'r') as f:
#     lineNum = 0
#     accuracies = [0] * 40
#     for line in f:
#         wordNum = 0
#         for word in line.split():
#             # 6 for train, 3 for test
#             if lineNum % 2 == 0:
#                 # train
#                 if wordNum == 5:
#                     index = lineNum % 40
#                     accuracies[index] += float(word)
#             else:
#                 # test
#                 if wordNum == 2:
#                     index = lineNum % 40
#                     accuracies[index] += float(word)
#             wordNum += 1
#         lineNum += 1

#     i = 0
#     for accuracy in accuracies:
#         if i % 2 == 0:
#             print "Iteration", i / 2 + 1, ":"
#             print "\tTraining:\t", accuracy / 20, "%"
#         else:
#             print "\tTesting:\t", accuracy / 20, "%"
#         i += 1

predictions = {}
with open('struct.svm.color.predictions', 'r') as f:
    for line in f:
        word_num = 0
        guess = None
        real = None
        for word in line.split():
            if word_num == 0:
                guess = word
            elif word_num == 1:
                real = word
            word_num += 1

        if predictions.get((real, guess)) is None:
            predictions[(real, guess)] = 0

        predictions[(real, guess)] += 1

for x,y in predictions.iteritems():
    if x[0] == 'Rembrant':
        print x, y/float(50) * 100, "%"
print ""
for x,y in predictions.iteritems():
    if x[0] == 'Picasso':
        print x,y/float(50) * 100, "%"
print ""
for x,y in predictions.iteritems():
    if x[0] == 'Dali':
        print x,y/float(50) * 100, "%"
print ""
for x,y in predictions.iteritems():
    if x[0] == 'VanGogh':
        print x,y/float(50) * 100, "%"
print ""
for x,y in predictions.iteritems():
    if x[0] == 'Monet':
        print x,y/float(50) * 100, "%"
print ""
for x,y in predictions.iteritems():
    if x[0] == 'Turner':
        print x,y/float(50) * 100, "%"

    # print x, y/float(50) * 100, "%"
