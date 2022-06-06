import pprint
import random
import matplotlib.pyplot as plt  # print circle for cities |  print lines between cities
import math

# ----------Constants :

NUM_CITIES = 10
PERCENTAGE_ROADS = 0.8  # 0.8 is to create 80% of the roads | 1 for 100% roads
ALPHABET = "ABCDEFGHIJKLMOPQRSTUVWXYZ"


# generate cities
def create_cities():  # Return a list of random cities
    cities = set()
    while len(cities) < NUM_CITIES:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        cities.add((x, y))

    # print(cities)
    return list(cities)


def pathToCoord(citiesList, path):
    pathCoordList = []  # 2d array with road of a path "ABCD"
    for city in range(len(path) - 1):  # -1 because we use city + 1
        idx1 = ord(path[city]) - 65
        idx2 = ord(path[city + 1]) - 65
        pathCoordList.append([(citiesList[idx1][0], citiesList[idx2][0]), (citiesList[idx1][1], citiesList[idx2][1])])
    return pathCoordList


def roadListToCoord(citiesList, roadList):
    roadCoordList = []  # 2d array with road of a path "ABCD"
    for road in roadList:  # -1 because we use city + 1
        idx1 = ord(road[0]) - 65
        idx2 = ord(road[1]) - 65
        roadCoordList.append([(citiesList[idx1][0], citiesList[idx2][0]), (citiesList[idx1][1], citiesList[idx2][1])])
    return roadCoordList


def createPlot(citiesList, roadsList):  # create and show the graph

    # print(citiesList)   # citiesList is list of tuples with coordinates of cities
    x = [i[0] for i in citiesList]
    y = [i[1] for i in citiesList]
    # print(f"Cities coord in lists = x : {x} | y : {y}")

    # Adding the cities to the plot graph
    for i in range(len(x)):
        plt.plot(x[i], y[i], "o")
        plt.text(x[i], y[i], ALPHABET[i], fontsize=20, color='b')

    # Adding the roads in the graph
    for i in range(len(roadsList)):
        plt.plot(roadsList[i][0], roadsList[i][1])
    plt.grid()
    plt.show()


def createNameList():  # Create a list with all the names of the cities
    citiesNames = []
    for i in range(NUM_CITIES):
        citiesNames.append(ALPHABET[i])

    return citiesNames


def createRoads(citiesList):
    maxRoadsCount = (NUM_CITIES * (NUM_CITIES - 1)) // 2  # Formula for the maximum possible number of roads
    idxlist = [i for i in range(maxRoadsCount)]  # List of idx we are getting a sample
    # random.sample( listofvalues, length_of_the_sample)
    idxlist = random.sample(idxlist, round(maxRoadsCount * PERCENTAGE_ROADS))  # To not create all the possible roads
    # print(f'idxlist : {idxlist}  ')
    idx = 0
    citiesNames = createNameList()  #
    # Initialize the 3D array of the roadsCostList and roads path
    roadsCostList = [[-1 for col in range(len(citiesNames))] for row in range(len(citiesNames))]
    # if a road doesn't exist the cost will be -1 | impossible negative distance
    # pprint.pprint(roadsCostList)
    roadsCoordList = []  # 2D list containing the coord of the cites bound by the roads
    roadNamesList = []  # [AB, AC...]

    for i in range((len(citiesNames))):
        for j in range(len(citiesNames)):
            # array x, array y
            if i >= j:
                continue

            if idx in idxlist:  # To respect the 80% sample of roads
                roadsCoordList.append([(citiesList[i][0], citiesList[j][0]), (citiesList[i][1], citiesList[j][1])])
                roadCost = path_cost(roadsCoordList[-1])
                roadsCostList[i][j] = roadCost
                roadsCostList[j][i] = roadCost
                roadNamesList.append(citiesNames[i] + citiesNames[j])

            idx += 1
    # print(len(roadsCoordList))
    print("The roadNameList :")
    print(roadNamesList)
    # print("The roadsCoordList :")
    # pprint.pprint(roadsCoordList)
    # print("The 3D roadsCostList list with the costs  :")
    # pprint.pprint(roadsCostList)  # print 3D list in a fancy way

    # createPlot(citiesList, roadsCoordList)

    return roadNamesList, roadsCoordList, roadsCostList


def path_cost(roadsCoord):  # [(x1, y1), (x2, y2)]
    # Euclidian distance formula
    cost = math.pow(roadsCoord[0][0] - roadsCoord[1][0], 2) + math.pow(roadsCoord[0][1] - roadsCoord[1][1], 2)
    cost = math.sqrt(cost)
    cost = round(cost, 6)  # to make the costs number smaller
    return cost


# return a pathList of all the possible path for the salesman :
def BFSsearch(citiesNamesList, roadsNameList, pathList):  # pathList contains all the possibility paths for the
    # salesman and return the final pathList

    if len(pathList[0]) != len(citiesNamesList):  # No recursion if the length of the possible paths is equal to the
        # number of cities (because the salesman has been to every city)
        if ((pathList[0][0] != pathList[0][-1]) or (len(pathList) == 1)):
            newRoadToTake = []
            # print("begin")
            for path in pathList:  # e.g path = 'ADFE'
                nonVisitedCities = citiesNamesList.copy()
                for city in path:  # e.g city = 'A'
                    # We remove the cities already visited from the possibilities
                    nonVisitedCities.remove(city)

                for city in nonVisitedCities:
                    # path[-1] is last city visited by the path | city the cities not visited
                    if ((path[-1] + city) in roadsNameList) or (
                            (city + path[-1]) in roadsNameList):  # Check if road exist
                        newRoadToTake.append(path + city)
            # print(newRoadToTake)

            # If there is no new road to take the salesman has been to every city
            # So he need to go back to the first city to make a cycle
            if len(newRoadToTake) == 0:
                firstChar = pathList[0][0]  # the firstChar is the beginning and arrival city
                pathToRemove = set()
                for i in range(len(pathList)):

                    if ((pathList[i][-1] + firstChar) in roadsNameList) or (
                            (firstChar + pathList[i][-1]) in roadsNameList):
                        # If the final road exist the cycle is ok
                        print("oui")
                        pathList[i] = pathList[i] + firstChar
                    else:  # else we remove the path
                        pathToRemove.add(pathList[i])

                # Removing the paths :
                for path in pathToRemove:
                    pathList.remove(path)
                return pathList

            return BFSsearch(citiesNamesList, roadsNameList, newRoadToTake)
    else:
        # The paths go through all city, we need to check if they can go back to the first city
        firstChar = pathList[0][0]  # the firstChar is the beginning and arrival city
        pathToRemove = set()
        for i in range(len(pathList)):

            if ((pathList[i][-1] + firstChar) in roadsNameList) or ((firstChar + pathList[i][-1]) in roadsNameList):
                # If the final road exist the cycle is ok
                pathList[i] = pathList[i] + firstChar
            else:  # else we remove the path
                pathToRemove.add(pathList[i])

        for path in pathToRemove:
            pathList.remove(path)
        # print(pathList) #explanationBFS
        return pathList


def minPath(roadsCostList, pathList):
    minPathCost = 0;
    for path in pathList:
        # print(pathList[i])
        costSum = 0
        for city in range(len(path) - 1):  # -1 because we use city + 1
            # letter by letter
            idx1 = ord(
                path[city]) - 65  # convert letter into number (ASCII) -65 because ord(A)=65 so 'A':0 (the first idx)
            idx2 = ord(path[city + 1]) - 65
            costSum += roadsCostList[idx1][idx2]

        if (costSum < minPathCost) or (minPathCost == 0):
            minPathCost = costSum
            minPath = path
            # print(path, costSum)
    minPathList = list(minPath)
    print(f"minPath : {minPathList}, minPathCost : {minPathCost}")
    return minPath, minPathCost


def roadCost(path, roadsCostList):  # for path = ['A', 'B', 'C',...]
    pathCost = 0
    for city in range(len(path) - 1):  # -1 because we use city + 1
        # letter by letter
        idx1 = ord(path[city]) - 65
        idx2 = ord(path[city + 1]) - 65
        pathCost += roadsCostList[idx1][idx2]
    return pathCost


# remain : set | startingcity : 'A', | path : ['A'], roadscoord : dict de mort
def DFSsearch(nonVisitedCities, path, roadsNameList, roadsCostList):
    if len(nonVisitedCities) == 0:  # If all cities had been visited it's the end of recursion
        # path[0] is the first letter so the first city
        if ((path[0] + path[-1]) in roadsNameList) or ((path[-1] + path[0]) in roadsNameList):  # Check if road exist
            newPath = path.copy()
            newPath.append(path[0])
            # print(newPath) # explanationDFS
            pathCost = roadCost(newPath, roadsCostList)
            return [(pathCost, newPath)]

        else:
            # print("-1")
            return []

        # else:  # If there is no road linked to the last city

    # For each city there's a recursion for every new road so it goes through each possible path the longer it can
    pathCost = []
    for city in nonVisitedCities:
        if ((path[-1] + city) in roadsNameList) or ((city + path[-1]) in roadsNameList):  # Check if road exist
            newNonVisitedCities = nonVisitedCities.copy()
            newNonVisitedCities.remove(city)
            newPath = path.copy()
            newPath.append(city)
            # print(newPath)
            pathCost = pathCost + DFSsearch(newNonVisitedCities, newPath, roadsNameList, roadsCostList)

        # Final return here :
    return pathCost


def greedySearch(nonVisitedCities, path, roadsNameList, roadsCostList):
    if len(nonVisitedCities) == 0:  # If the salesman has been to all the cities
        # path[0] is the first city
        if ((path[0] + path[-1]) in roadsNameList) or ((path[-1] + path[0]) in roadsNameList):  # Check if road exist
            path.append(path[0])  # we add the first letter at the end if the final road exist
            return roadCost(path, roadsCostList), path  # road cost path gives us the cost of the path
        else:
            return []

    listOfPossiblePath = []  # [(23.3455, ['D', 'A',...]), ...]
    # Here we compute the cost of every road binding the last visited city and every other non visited city
    for city in nonVisitedCities:
        if ((city + path[-1]) in roadsNameList) or ((path[-1] + city) in roadsNameList):  # Check if road exist
            tempCost = roadCost(path[-1] + city, roadsCostList)
            # print(tempCost)
            tempPath = path.copy()
            tempPath.append(city)
            listOfPossiblePath.append((tempCost, tempPath))

    if len(listOfPossiblePath) == 0:  # If there is no more road possible, there is no solutions so we end the recursion
        return []

    i = 0
    result = []
    listOfPossiblePath = sorted(listOfPossiblePath)  # trie dans l'ordre croissant suivant les tuple[0]
    # print(listOfPossiblePath) # explanationgreedy
    while i < len(listOfPossiblePath) and result == []:
        # Because we sorted the list with the cost we explore the smallest road each tim
        nextCity = listOfPossiblePath[i][1][-1]
        newNonVisitedCities = nonVisitedCities.copy()
        newNonVisitedCities.remove(nextCity)
        pathTemp = path.copy()
        pathTemp.append(nextCity)
        i += 1
        result = greedySearch(newNonVisitedCities, pathTemp, roadsNameList, roadsCostList)

    return result


def minRoadCost(nonVisitedRoads, roadsCostList):
    minCost = math.inf
    minRoad = ''
    for road in nonVisitedRoads:
        i = ord(road[0]) - 65
        j = ord(road[1]) - 65
        if roadsCostList[i][j] < minCost:
            minCost = roadsCostList[i][j]
            minRoad = ALPHABET[i] + ALPHABET[j]
    return minRoad


def createNonVisitedRoads(roadNameList):
    nonVisitedRoads = []
    for road in roadNameList:
        nonVisitedRoads.append(road)
        nonVisitedRoads.append(road[1] + road[0])  # we add both direction
    return nonVisitedRoads

# return dict : {'A':2, 'B':1, ...}
def numRoadLinkToCity(visitedRoad):  # Useful function for the spanning tree
    freqCity = {}
    for i in range(NUM_CITIES):  # initialisation of the dict
        freqCity[ALPHABET[i]] = 0
    cityRoad = [road[0] for road in visitedRoad]  # e.g ['B', 'A', 'C', 'A',...]
    for city in cityRoad:
        freqCity[city] += 1
    # print(freqCity)  # {'A':2, 'B':1,...}
    return freqCity


# the spanning tree is an algo that will construct the path linking all the cities by choosing the smallest road (cost)
# No construction step by step, all the roads could be used and sometimes it's not possible to finish the path so it's not perfect
def minSpanningTree(nonVisitedRoads, firstCity, visitedRoads, roadNameList, roadsCostList):
    if len(nonVisitedRoads) == 0:  # If the salesman has been to all the cities
        # we have found a solution let's return the cost of the path and the path
        path = []
        for city in range(0, len(visitedRoads), 2):
            path.append(visitedRoads[city][0] + visitedRoads[city + 1][0])
        cost = 0
        for road in path:
            i = ord(road[0]) - 65
            j = ord(road[1]) - 65
            cost += roadsCostList[i][j]
        return cost, path

    if len(visitedRoads) == 0:  # the first call
        possibleRoads = []  # To store all the roads linked with teh first city path[0]
        for road in nonVisitedRoads:
            if road[0] == firstCity:  # Check if road exist
                possibleRoads.append(road)
        minRoadResult = minRoadCost(possibleRoads, roadsCostList)  # minRoadResult = 'AB'
        newNonVisitedRoads = nonVisitedRoads.copy()
        newNonVisitedRoads.remove(minRoadResult)  # delete the road
        newNonVisitedRoads.remove(minRoadResult[1] + minRoadResult[0])  # delete the road but in the other direction

        newVisitedRoads = visitedRoads.copy()
        newVisitedRoads.append(minRoadResult)  # We add the roads
        newVisitedRoads.append(minRoadResult[1] + minRoadResult[0])  # in both direction

        result = minSpanningTree(newNonVisitedRoads, firstCity, newVisitedRoads, roadNameList, roadsCostList)
        return result

    freqCity = numRoadLinkToCity(visitedRoads)  # {'A':2, 'D':2, ...}

    minRoadResult = minRoadCost(nonVisitedRoads, roadsCostList)

    # Verify that the next road isn't linked to cities already linked with two roads visited
    # To have a cycle at the end
    if minRoadResult == '':
        print(nonVisitedRoads)
        print(f'visitedRoads : {visitedRoads}')
        result = minSpanningTree(nonVisitedRoads, firstCity, visitedRoads, roadNameList, roadsCostList)
        return result
    if freqCity[minRoadResult[0]] < 2 and freqCity[minRoadResult[1]] < 2:
        # to make sure all the cities are linked and avoid a cycle of 3 cities
        zero = False
        # if we are taking a road between two already visited city we are closing the cycle we need to make sure that there is no more nonvisitedcity
        if freqCity[minRoadResult[0]] + 1 == 2 and freqCity[minRoadResult[1]] + 1 == 2:
            for key, values in freqCity.items():
                if values == 0:
                    zero = True
        if zero:  # if a city isn't connected we can't use the minroad to close the path
            newNonVisitedRoads = nonVisitedRoads.copy()
            newNonVisitedRoads.remove(minRoadResult)  # delete the road
            newNonVisitedRoads.remove(minRoadResult[1] + minRoadResult[0])
            result = minSpanningTree(newNonVisitedRoads, firstCity, visitedRoads, roadNameList, roadsCostList)
            return result
        # If it passed the conditons we can add the road to the path
        newNonVisitedRoads = nonVisitedRoads.copy()
        newNonVisitedRoads.remove(minRoadResult)  # delete the road
        newNonVisitedRoads.remove(minRoadResult[1] + minRoadResult[0])  # delete the road but in the other direction

        newVisitedRoads = visitedRoads.copy()
        newVisitedRoads.append(minRoadResult)  # We add the roads
        newVisitedRoads.append(minRoadResult[1] + minRoadResult[0])  # in both direction

        result = minSpanningTree(newNonVisitedRoads, firstCity, newVisitedRoads, roadNameList, roadsCostList)
        return result
    else:  # if the road is linking a city already visited (2 roads connected) it will not be used anymore so we can remove it and try with another road
        newNonVisitedRoads = nonVisitedRoads.copy()
        newNonVisitedRoads.remove(minRoadResult)  # delete the road
        newNonVisitedRoads.remove(minRoadResult[1] + minRoadResult[0])
        result = minSpanningTree(newNonVisitedRoads, firstCity, visitedRoads, roadNameList, roadsCostList)
        return result


def bidirectionalSearch(previousCity1, previousCity2, firstCity, lastCity, roadsCostList, cityCoordDict):

    path1 = []
    path2 = []

    path = (path1, path2)

    potentialCities = citiesLinkedToCity(firstCity, cityCoordDict, roadsNameList)
    potentialCities.discard(previousCity1)
    dict = {}
    j = ord(firstCity) - 65
    for city in potentialCities:
        i = ord(city) - 65
        g = roadsCostList[i][j] # We compute the cost from the first city to the new cities

        # compute the cost of path between the potential city and the endingCity
        h = path_cost([(cityCoordDict[city][0], cityCoordDict[lastCity][0]), (cityCoordDict[city][1], cityCoordDict[lastCity][1])])
        # h is a flying bird distance between a city linked to the first city
        f = h + g
        dict[city] = f   # {'A': 32.03444, ...}

    newCity = min(dict, key=dict.get)  # The min costof path (not following the roads , just in flying bird distance)

    path[0].append(newCity)


    if newCity == lastCity:   # if the first and ending cities are linked it's done
        return path  # (['A'], [])

    # we do the same thing the other way
    potentialCities = citiesLinkedToCity(lastCity, cityCoordDict, roadsNameList)
    potentialCities.discard(previousCity2)
    dict = {}
    j = ord(lastCity) - 65
    for city in potentialCities:
        i = ord(city) - 65
        g = roadsCostList[i][j]

        # compute the city (linked to the first city)
        h = path_cost([(cityCoordDict[city][0], cityCoordDict[newCity][0]),
                         (cityCoordDict[city][1], cityCoordDict[newCity][1])])

        f = h + g
        dict[city] = f

    newCity2 = min(dict, key=dict.get)

    path[1].append(newCity2)

    if newCity2 == newCity:
        return path


    # If the new cities aren't linked they became the first city and the last city
    recursionPath = bidirectionalSearch(firstCity, lastCity, newCity, newCity2, roadsCostList, cityCoordDict)

    for i in range(len(recursionPath[0])):
        path[0].append(recursionPath[0][i])

    for i in range(len(recursionPath[1])):
        path[1].append(recursionPath[1][i])

    return path


def citiesLinkedToCity(city, cityCoordDict, roadNameList):  # Return a set of cities linked to a city

    cities = set()
    for potentialCity in cityCoordDict.keys():
        if ((city + potentialCity) in roadNameList) or ((potentialCity + city) in roadNameList):  # Check if road exist
            cities.add(potentialCity)
    return cities


def createCitiesCoordDict(citiesList):  # {'A': (12,53), 'B':(36,98),...}

    dict = {}
    i = 0
    for city in citiesList:
        dict[ALPHABET[i]] = city
        i += 1
    return dict


def bidirectionalSearchAll(roadsCostList, cityCoordDict, citiesList):
    firstCity = (input('Enter letter corresponding to first city : '))
    lastCity = (input('Enter letter corresponding to last city : '))
    path = bidirectionalSearch("", "", firstCity, lastCity, roadsCostList, cityCoordDict)

    finalPath = []
    finalPath.append(firstCity)

    for i in range(len(path[0])):
        finalPath.append(path[0][i])

    for i in reversed(range(len(path[1]))):
        finalPath.append(path[1][i])

    finalPath.append(lastCity)
    distance = roadCost(finalPath, roadsCostList)

    # results :
    print(f'path : {finalPath}, costPath: {distance}')
    pathCoordList = pathToCoord(citiesList, finalPath)
    createPlot(citiesList, pathCoordList)


# Initialisation
citiesList = create_cities()
print(f"listOfCities : {citiesList}")
roadsNameList, roadsCoordList, roadsCostList = createRoads(citiesList)
citiesNameList = createNameList()

# BFS search :
# We search path that begin from 'D', go through all cities and end in 'D' :
pathList = BFSsearch(citiesNameList, roadsNameList, ["D"])
# print(f"pathList : {pathList}")
print("------------------BFS Search : ")
path, cost = minPath(roadsCostList, pathList)
minPathCoord = pathToCoord(citiesList, path)
createPlot(citiesList, roadsCoordList)
createPlot(citiesList, minPathCoord)

# DFS search :
print("----------DFS search :")
firstcity = ['D']
nonVisitedCities = citiesNameList
nonVisitedCities.remove(firstcity[0])
DFSresult = DFSsearch(nonVisitedCities, firstcity, roadsNameList, roadsCostList)
# print(DFSresult)
minCostDFS = min(DFSresult)
print(f"minCostDFS : {minCostDFS[0]}, minPathList : {minCostDFS[1]}")
minPathDFS = pathToCoord(citiesList, minCostDFS[1])
createPlot(citiesList, minPathDFS)

# greedy search :
print("----------Greedy search :")
firstcity = ['D']
# nonVisitedCities = citiesNameList
# print(nonVisitedCities)
# nonVisitedCities.remove(firstcity[0])
greedyResult = greedySearch(nonVisitedCities, firstcity, roadsNameList, roadsCostList)
print(f"minCostGreedy : {greedyResult[0]}, minPathGreedy : {greedyResult[1]}")
# print(roadCost(greedyResult[1], roadsCostList))
minPathGreedy = pathToCoord(citiesList, greedyResult[1])
createPlot(citiesList, minPathGreedy)


# min Spanning tree:
print("----------Minimum spanning Tree :")
allRoads = createNonVisitedRoads(roadsNameList)
spanTreeResult = minSpanningTree(allRoads, 'D', [], roadsNameList, roadsCostList)
print(spanTreeResult)
pathManTree = roadListToCoord(citiesList, spanTreeResult[1])
createPlot(citiesList, pathManTree)



# Bidirectional Search:
print("----------Bidirectional Search :")
cityCoordDict = createCitiesCoordDict(citiesList)
print(cityCoordDict)
bidirectionalSearchAll(roadsCostList, cityCoordDict, citiesList)
