# coding: UTF-8
malenum = 4
femalenum = 4
single = []
married = {}
mprefer = [[1, 2, 3, 4],
           [3, 2, 1, 4],
           [1, 2, 4, 3],
           [3, 1, 4, 2]]
fprefer = [[1, 2, 3, 4],
           [2, 1, 4, 3],
           [2, 3, 1, 4],
           [1, 4, 3, 2]]

for i in range(malenum + femalenum):
    single.append(i)
while True:
    for i in range(malenum):
        if i in single:
            counter = 0
            while True:
                femaleid = mprefer[i][counter]-1
                if femaleid+malenum in single:
                    single.remove(i)
                    single.remove(femaleid+malenum)
                    married.update({femaleid+1: i+1})
                    break
                elif fprefer[femaleid].index(married[femaleid+1]) > fprefer[femaleid].index(i+1):
                    single.append(married[femaleid+1]-1)
                    single.remove(i)
                    married.update({femaleid+1: i+1})
                    break
                else:
                    counter += 1
    if len(single) == 0:
        break
married = married.items()
for (i, j) in married:
    print "女性No.%sは男性No.%sと結ばれました。" % (i, j)
