predictions = {}
with open('boost.1.predictions', 'r') as f:
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
