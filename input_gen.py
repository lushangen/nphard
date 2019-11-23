from UF import *
def gen_map(numLocs, numHouses, startLoc):
    unioner = UF(numLocs)
    j = 1
    i = 0
    while(unioner.sizeOfIndex(startLoc) < numLocs):
        #gen random shit
        #if add edge btween:
        if j <= 25:
            i = 0
            unioner.union(i,j)
        else:
            i = 1
            unioner.union(i,j)
        print("unioned ", i, " and ", j)
        print("size: ", unioner.sizeOfIndex(startLoc))
        j+=1

gen_map(50,25,0)
