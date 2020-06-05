import numpy as np
from scipy.integrate import odeint
import seaborn as sns
from matplotlib import pyplot as plt
import time
from timeit import default_timer as timer
# from simCoronavirusSeattle2 import corEqs1, findBeta2
import pickle
mycolors = sns.color_palette("hls", 5)

#05/13/20: this file is different than simCoronavirusSeattle2.py because we found two mistakes in the previous file:
#- the matrix multiplication used to find beta in line 50:
#       myProd = F * np.linalg.inv(V) is incorrect as this is not the correct matrix multiplication
# instead it was changed to:
# myProd = np.dot(F, np.linalg.inv(V))
# and there was an error in the matrix we used to run the simulations, we have used a matrix had each column divided by the
# population of the corresponding column, so column jth was divided by population N_j. In reality, we should have divided
# the whole matrix by the total population. Now this is included in the equations.


#define the equations for 6 age groups:
def corEqs1(y, t, params):
    # print('y:',  y)
    [beta, betaA, C, gamma, gammaE, k, N, numGroups] = params
    #beta: infection rate
    #C: matrix of contact rates, c_ij represents the contact rate between group i and group j
    temp = np.reshape(y, (7, numGroups))
    S = temp[0, :]
    E = temp[1, :]
    I = temp[2, :]
    A = temp[3, :]
    R = temp[4, :]
    RA = temp[5, :]
    Cases = temp[6, :]

    dS = - (np.multiply((1 / N) * np.dot(C, (beta*I + betaA*A)), S))
    # print('foi:' ,np.multiply(np.dot((1/N)*C, (beta*I + betaA*A)), S))
    dE = (np.multiply((1 / N) * np.dot(C, (beta*I + betaA*A)), S)) -gammaE * E
    dI =  k* gammaE * E - gamma * I
    dA = (1-k) * gammaE * E -gamma* A
    dR = gamma * I
    dRA = gamma * A
    dCases = (np.multiply((1 / N) * np.dot(C, (beta*I + betaA*A)), S))
    dydt = np.array([dS, dE, dI, dA, dR, dRA, dCases]).reshape((numGroups*7))
    return dydt


def corEqs2(y, t, params):
    # print('y:',  y)
    [beta, betaA, C, gamma, gammaE, k, N, numGroups] = params
    #beta: infection rate
    #C: matrix of contact rates, c_ij represents the contact rate between group i and group j
    temp = np.reshape(y, (5, numGroups))
    S = temp[0, :]
    E = temp[1, :]
    I = temp[2, :]
    R = temp[3, :]
    Cases = temp[4, :]

    dS = - (np.multiply((1 / N) * np.dot(C, (beta*I)), S))
    # print('foi:' ,np.multiply(np.dot((1/N)*C, (beta*I + betaA*A)), S))
    dE = (np.multiply((1 / N) * np.dot(C, (beta*I)), S)) -gammaE * E
    dI =  gammaE * E - gamma * I
    dR = gamma * I
    dCases = np.multiply(np.dot(C, (beta*I)), S)
    dydt = np.array([dS, dE, dI, dR,  dCases]).reshape((numGroups*5))
    return dydt


def findBeta1(R0, C, gamma, totalPop):
    # compute the eignevalues of F*V^(-1)
    [n1, n2] = np.shape(C)

    # create F
    F = np.zeros((n1, n1))
    for jvals in range(n1):
        F[jvals, :] = (1/np.sum(totalPop))* C[jvals, :] * totalPop[jvals]

    #create V
    V = np.diag(gamma * np.ones(10))
    # print(V)

    myProd = np.dot(F, np.linalg.inv(V))
    myEig = np.linalg.eig(myProd)
    largestEig = np.max(myEig[0])
    # print(myEig[0])
    # print(largestEig)
    if largestEig.imag == 0.0:
        beta = R0 / largestEig.real
        return beta
    else:
        print(largestEig)
        raise Exception('largest eigenvalue is not real')


def findBeta2(R0, C, gamma, totalPop):
    # compute the eignevalues of -F*V^(-1)
    [n1, n2] = np.shape(C)

    # create F and V
    F = np.zeros((n1, n1))
    V = np.diag(-gamma * np.ones(n1))
    N = np.sum(totalPop)

    for jvals in range(n1):
        F[jvals, :] = (C[jvals, :] / N ) * totalPop[jvals]

    # myProd = -F * np.linalg.inv(V)
    myProd = np.dot(-F, np.linalg.inv(V))

    myEig = np.linalg.eig(myProd)
    largestEig = np.max(myEig[0])

    if largestEig.imag == 0.0:
        beta = R0 / largestEig.real
        return beta
    else:
        print(largestEig)
        raise Exception('largest eigenvalue is not real')


def adjustMatrixChildrenOnly(C, red):
    CChildren = np.copy(C)
    CChildren[0:3, :] = (1 - red1) * C[0:3, :]
    CChildren[3:10, 0:3] = (1 - red1) * C[3:10, 0:3]
    return CChildren

def adjustMatrix60andOverOnly(C, red1):
    C60andOver = np.copy(C)
    C60andOver[7:10, :] = (1 - red1) * C[7:10, :]
    C60andOver[0:7, 7:10] = (1 - red1) * C[0:7, 7:10]
    return C60andOver

def adjustMatrix80andOverOnly(C, red1):
    C80andOver = np.copy(C)
    C80andOver[9:10, :] = (1 - red1) * C[9:10, :]
    C80andOver[0:9, 9:10] = (1 - red1) * C[0:9, 9:10]
    return C80andOver


def adjustMatrixHighRiskAndAdults(C, red1, red2, red3):
    CHighRiskAndAdults = np.copy(C)
    CHighRiskAndAdults[7:10, :] = (1 - red1) * C[7:10, :]
    CHighRiskAndAdults[0:7, 7:10] = (1 - red1) * C[0:7, 7:10]
    #
    CHighRiskAndAdults[3:7, 0:7] = (1 - red3) * C[3:7, 0:7]
    CHighRiskAndAdults[0:3, 3:7] = (1 - red3) * C[0:3, 3:7]
    return CHighRiskAndAdults

def adjustMatrixHighRiskAndChildren(C, red1, red2, red3):
    CHighRiskAndChildren = np.copy(C)
    CHighRiskAndChildren[7:10, :] = (1 - red1) * C[7:10, :]
    CHighRiskAndChildren[0:7, 7:10] = (1 - red1) * C[0:7, 7:10]
    #
    CHighRiskAndChildren[0:3, 0:7] = (1 - red2) * C[0:3, 0:7]
    CHighRiskAndChildren[3:7, 0:3] = (1 - red2) * C[3:7, 0:3]
    return CHighRiskAndChildren


def adjustMatrixAll(C, red1, red2, red3):
    CAll = np.copy(C)
    CAll[7:10, :] = (1 - red1) * C[7:10, :]
    CAll[0:7, 7:10] = (1 - red1) * C[0:7, 7:10]
    #
    CAll[0:3, 0:3] = (1 - red2) * C[0:3, 0:3]
    CAll[3:7, 3:7] = (1 - red3) * C[3:7, 3:7]
    #
    CAll[3:7, 0:3] = (1 - np.maximum(red2, red3)) * C[3:7, 0:3]
    CAll[0:3, 3:7] = (1 - np.maximum(red2, red3)) * C[0:3, 3:7]
    return CAll

def computeNewCases(y, I0, numGroups):
    [dim1, dim2] = np.shape(y)

    dailyCases = np.zeros((dim1, numGroups))
    # print 'a', np.shape(weeklyCases)
    # temp = np.sum(y[:, 10:14], 1)
    temp = y[:, 60:70]
    # print np.shape(temp)
    dailyCases[0, :] = I0
    for t in range(1, dim1):

        dailyCases[t, :] = temp[t, :] - temp[t-1, :]

    return dailyCases


def runModel(R0, gamma, gammaE, interventionDate, numWeeksIntervention, red1, red2, totalSimTime):
    #compute beta:
    beta = findBeta1(R0, C, gamma, totalPop)

    #for the time being asymptomatics are set to 0
    betaA = beta
    k = 1

    #set initial conditions
    S0 = totalPop
    S0[5] = S0[5] -1
    E0 = np.zeros(numGroups)
    I0 = np.zeros(numGroups)
    A0 = np.zeros(numGroups)
    Re0 = np.zeros(numGroups)
    RA0 = np.zeros(numGroups)

    I0[5] = 1
    Cases0 = np.copy(I0)
    y0 = np.array([S0, E0, I0, A0, Re0, RA0, Cases0]).reshape((7 * numGroups))


    #run the model up to a certain date:
    params = [beta, betaA, C, gamma, gammaE, k, N, numGroups]
    tspan = range(0, interventionDate +1)
    out1 = odeint(corEqs1, y0, tspan, args=(params,))

    cases1 = computeNewCases(out1, np.array(Cases0), numGroups)

    #run the model for the remainder of the simulation with the new matrices and store the results
    tspan2 = range(interventionDate, (interventionDate + numWeeksIntervention*7 + 1))
    # print(tspan2)
    if (interventionDate + numWeeksIntervention*7) < totalSimTime:
        tspan3 = range((interventionDate + numWeeksIntervention*7), totalSimTime+1)
    else:
        raise Exception('simulation out of bounds')
    y0new = np.copy(out1[-1,:])



    results = []

    #define the matrices
    C60andOver = adjustMatrix60andOverOnly(C, red1)
    ChighRiskAndChildren = adjustMatrixHighRiskAndChildren(C, red1, red2)
    ChighRiskAndAdults = adjustMatrixHighRiskAndAdults(C, red1, red2)
    Call = (1-red1)*C

    Cmats = [C, C60andOver, ChighRiskAndChildren, ChighRiskAndAdults, Call]
    totalNumberOfCases = np.zeros((5, numGroups))
    deaths = np.zeros((5, numGroups))
    prop_deaths_adverted = np.zeros((5, numGroups))
    prop_deaths_advertedTotal = np.zeros(5)
    prop_cases_adverted = np.zeros((5, numGroups))
    prop_cases_advertedTotal = np.zeros((5))
    for ivals in range(4,5):
        Ctemp = Cmats[ivals]
        print(Ctemp)
        params2 = [beta, betaA, Ctemp, gamma, gammaE, k, N, numGroups]
        out2 = odeint(corEqs1, y0new, tspan2, args=(params2,))
        print(out2[-1, 20:30])
        cases2 = computeNewCases(out2, out2[0, 60:70], numGroups)

        #run after intervention is lifted up
        y0new3 = np.copy(out2[-1,:])
        print(y0new3[20:30])
        for jvals in range(20, 30):
          if (y0new3[jvals]) < 1:
                y0new3[jvals] = 0

        params = [beta, betaA, C, gamma, gammaE, k, N, numGroups]
        out3 = odeint(corEqs1, y0new3, tspan3, args=(params,))

        cases3 = computeNewCases(out3, out3[0, 60:70], numGroups)

        cases = np.concatenate([cases1[:, :], cases2[1:, :], cases3[1:, ]])
        results.append(cases)
        totalNumberOfCases[ivals, :] = np.sum(cases, 0)
        deaths[ivals, :] = death_rate*np.sum(cases, 0)
    # print(np.shape(totalNumberOfCases))
    # print(deaths)
    baselineDeaths = deaths[0, :]
    baselineCases = totalNumberOfCases[0, :]
    # print(baselineCases)

    # for ivals in range(4,5):
    #     # print(cases[ivals, :])
    #     deaths_adverted = (baselineDeaths - deaths[ivals, :])
    #     deaths_advertedTotal = (np.sum(baselineDeaths) - np.sum(deaths[ivals, :]))
    #     prop_deaths_adverted[ivals, :] = np.divide(deaths_adverted, baselineDeaths)
    #     prop_deaths_advertedTotal[ivals] = deaths_advertedTotal/np.sum(baselineDeaths)
    #     cases_adverted = baselineCases - totalNumberOfCases[ivals, :]
    #     cases_advertedTotal = np.sum(baselineCases) - np.sum(totalNumberOfCases[ivals, :])
    #     # print('cases_adverted', cases_adverted)
    #     prop_cases_adverted[ivals, :] = np.divide(cases_adverted, baselineCases)
    #     prop_cases_advertedTotal[ivals] = cases_advertedTotal/np.sum(baselineCases)
        # print(prop_cases_adverted)
    # print(np.shape(baselineCases))
    tspanFull = np.concatenate([tspan[:], tspan2[1:], tspan3[1:]])
    plt.plot(tspan, np.sum(out1[:, 20:30], 1))
    plt.plot(tspan2, np.sum(out2[:, 20:30], 1))
    plt.plot(tspan3, np.sum(out3[:, 20:30], 1))
    # return [tspanFull, results, prop_cases_adverted, prop_cases_advertedTotal, prop_deaths_adverted, prop_deaths_advertedTotal]




def runModel2(C, death_rate, gamma, gammaE, interventionDate, N, numGroups, numWeeksIntervention, red1, red2, R0, totalPop, totalSimTime):
    #compute beta:
    beta = findBeta1(R0, C, gamma, totalPop)

    #for the time being asymptomatics are set to 0
    betaA = beta
    k = 1

    #set initial conditions
    S0 = totalPop
    S0[5] = S0[5] -1
    E0 = np.zeros(numGroups)
    I0 = np.zeros(numGroups)
    A0 = np.zeros(numGroups)
    Re0 = np.zeros(numGroups)
    RA0 = np.zeros(numGroups)

    I0[5] = 1
    Cases0 = np.copy(I0)
    y0 = np.array([S0, E0, I0, A0, Re0, RA0, Cases0]).reshape((7 * numGroups))


    #run the model up to a certain date:
    params = [beta, betaA, C, gamma, gammaE, k, N, numGroups]
    tspan = range(0, interventionDate +1)
    out1 = odeint(corEqs1, y0, tspan, args=(params,))

    cases1 = computeNewCases(out1, np.array(Cases0), numGroups)

    #run the model for the remainder of the simulation with the new matrices and store the results
    tspan2 = range(interventionDate, (interventionDate + numWeeksIntervention*7 + 1))
    # print(tspan2)
    if (interventionDate + numWeeksIntervention*7) < totalSimTime:
        tspan3 = range((interventionDate + numWeeksIntervention*7), totalSimTime+1)
    else:
        # myendtime = np.min()
        raise Exception('simulation out of bounds')
    y0new = np.copy(out1[-1,:])



    results = []

    #define the matrices
    C60andOver = adjustMatrix60andOverOnly(C, red1)
    ChighRiskAndChildren = adjustMatrixHighRiskAndChildren(C, red1, red2)
    ChighRiskAndAdults = adjustMatrixHighRiskAndAdults(C, red1, red2)
    Call = (1-red1)*C

    Cmats = [C, C60andOver, ChighRiskAndChildren, ChighRiskAndAdults, Call]
    totalNumberOfCases = np.zeros((5, numGroups))
    deaths = np.zeros((5, numGroups))
    prop_deaths_adverted = np.zeros((5, numGroups))
    prop_deaths_advertedTotal = np.zeros(5)
    prop_cases_adverted = np.zeros((5, numGroups))
    prop_cases_advertedTotal = np.zeros((5))
    for ivals in range(5):
        Ctemp = Cmats[ivals]
        params2 = [beta, betaA, Ctemp, gamma, gammaE, k, N, numGroups]
        out2 = odeint(corEqs1, y0new, tspan2, args=(params2,))

        cases2 = computeNewCases(out2, out2[0, 60:70], numGroups)

        #run after intervention is lifted up
        y0new3 = np.copy(out2[-1,:])
        # print(y0new3[20:30])
        for jvals in range(20, 30):
          if (y0new3[jvals]) < 1:
                y0new3[jvals] = 0
        params = [beta, betaA, C, gamma, gammaE, k, N, numGroups]
        out3 = odeint(corEqs1, y0new3, tspan3, args=(params,))

        cases3 = computeNewCases(out3, out3[0, 60:70], numGroups)

        cases = np.concatenate([cases1[:, :], cases2[1:, :], cases3[1:, ]])
        results.append(cases)
        totalNumberOfCases[ivals, :] = np.sum(cases, 0)
        deaths[ivals, :] = death_rate*np.sum(cases, 0)
    # print(np.shape(totalNumberOfCases))
    # print(deaths)
    baselineDeaths = deaths[0, :]
    baselineCases = totalNumberOfCases[0, :]
    # print(baselineCases)

    for ivals in range(5):
        # print(cases[ivals, :])
        deaths_adverted = (baselineDeaths - deaths[ivals, :])
        deaths_advertedTotal = (np.sum(baselineDeaths) - np.sum(deaths[ivals, :]))
        prop_deaths_adverted[ivals, :] = np.divide(deaths_adverted, baselineDeaths)
        prop_deaths_advertedTotal[ivals] = deaths_advertedTotal/np.sum(baselineDeaths)
        cases_adverted = baselineCases - totalNumberOfCases[ivals, :]
        cases_advertedTotal = np.sum(baselineCases) - np.sum(totalNumberOfCases[ivals, :])
        # print('cases_adverted', cases_adverted)
        prop_cases_adverted[ivals, :] = np.divide(cases_adverted, baselineCases)
        prop_cases_advertedTotal[ivals] = cases_advertedTotal/np.sum(baselineCases)
        # print(prop_cases_adverted)
    # print(np.shape(baselineCases))
    tspanFull = np.concatenate([tspan[:], tspan2[1:], tspan3[1:]])
    # print(prop_cases_adverted)
    return [tspanFull, results, prop_cases_adverted, prop_cases_advertedTotal, prop_deaths_adverted, prop_deaths_advertedTotal]

def runModel3(C, death_rate, gamma, gammaE, interventionDate, N, numGroups, numWeeksIntervention, red1, red2, red3, R0,  severeRate,totalPop, totalSimTime):

    #compute beta:
    beta = findBeta1(R0, C, gamma, totalPop)
    # print(beta)
    #for the time being asymptomatics are set to 0
    betaA = beta
    k = 1

    #set initial conditions
    S0 = totalPop
    S0[5] = S0[5] -1
    E0 = np.zeros(numGroups)
    I0 = np.zeros(numGroups)
    A0 = np.zeros(numGroups)
    Re0 = np.zeros(numGroups)
    RA0 = np.zeros(numGroups)

    I0[5] = 1
    Cases0 = np.copy(I0)
    y0 = np.array([S0, E0, I0, A0, Re0, RA0, Cases0]).reshape((7 * numGroups))


    #run the model up to a certain date:
    params = [beta, betaA, C, gamma, gammaE, k, N, numGroups]
    tspan = range(0, interventionDate +1)
    out1 = odeint(corEqs1, y0, tspan, args=(params,))
    # print(np.sum(out1[:, 40:50], 1))
    cases1 = computeNewCases(out1, np.array(Cases0), numGroups)

    #run the model for the remainder of the simulation with the new matrices and store the results
    tspan2 = range(interventionDate, (interventionDate + numWeeksIntervention*7 + 1))
    # print(tspan2)
    if (interventionDate + numWeeksIntervention*7) < totalSimTime:
        tspan3 = range((interventionDate + numWeeksIntervention*7), totalSimTime+1)
    else:
        # myendtime = np.min()
        raise Exception('simulation out of bounds')
    y0new = np.copy(out1[-1,:])



    results = []

    #define the matrices
    C60andOver = adjustMatrix60andOverOnly(C, red1)
    ChighRiskAndChildren = adjustMatrixHighRiskAndChildren(C, red1, red2, red3)
    ChighRiskAndAdults = adjustMatrixHighRiskAndAdults(C, red1, red2, red3)
    Call = adjustMatrixAll(C, red1, red2, red3)

    Cmats = [C, C60andOver, ChighRiskAndChildren, ChighRiskAndAdults, Call]
    totalNumberOfCases = np.zeros((5, numGroups))
    deaths = np.zeros((5, numGroups))
    prop_deaths_adverted = np.zeros((5, numGroups))
    prop_deaths_advertedTotal = np.zeros(5)
    prop_cases_adverted = np.zeros((5, numGroups))
    prop_cases_advertedTotal = np.zeros((5))
    for ivals in range(5):
        # print(ivals)
        Ctemp = Cmats[ivals]
        params2 = [beta, betaA, Ctemp, gamma, gammaE, k, N, numGroups]
        out2 = odeint(corEqs1, y0new, tspan2, args=(params2,))

        cases2 = computeNewCases(out2, out2[0, 60:70], numGroups)

        #run after intervention is lifted up
        y0new3 = np.copy(out2[-1,:])
        # print(y0new3[20:30])
        # #  NOTE ###############         assumes no asymptomatics
        totalInfectiousPeriod = (1/gamma) + (1/gammaE)
        # print(totalInfectiousPeriod < numWeeksIntervention*7)
        if (totalInfectiousPeriod < numWeeksIntervention*7) & (np.all(Ctemp == 0)):
            # if (np.all(Ctemp == 0)):
            #     print('mat=0')
            #ivals == 4 & (numWeeksIntervention == 5 or numWeeksIntervention == 6):
                y0new3[10:30] = 0
                y0new3[40:50] = y0new3[40:50] + np.copy(out2[-1, 10:20]) + np.copy(out2[-1, 20:30])
        if ivals == 1 or ivals == 2 or ivals ==3:
            for jvals in range(10, 30):
                if (y0new3[jvals]) < 1:
                    y0new3[jvals] = 0

        params = [beta, betaA, C, gamma, gammaE, k, N, numGroups]
        out3 = odeint(corEqs1, y0new3, tspan3, args=(params,))

        cases3 = computeNewCases(out3, out3[0, 60:70], numGroups)

        cases = np.concatenate([cases1[:, :], cases2[1:, :], cases3[1:, ]])
        results.append(cases)
        totalNumberOfCases[ivals, :] = np.sum(cases, 0)
        deaths[ivals, :] = np.multiply(death_rate, severeRate*np.sum(cases, 0))
    # print(np.shape(totalNumberOfCases))
    # print(deaths)
    baselineDeaths = deaths[0, :]
    baselineCases = totalNumberOfCases[0, :]
    # print(baselineCases)

    for ivals in range(5):
        # print(cases[ivals, :])
        deaths_adverted = (baselineDeaths - deaths[ivals, :])
        deaths_advertedTotal = (np.sum(baselineDeaths) - np.sum(deaths[ivals, :]))
        prop_deaths_adverted[ivals, :] = np.divide(deaths_adverted, baselineDeaths)
        prop_deaths_advertedTotal[ivals] = deaths_advertedTotal/np.sum(baselineDeaths)
        cases_adverted = baselineCases - totalNumberOfCases[ivals, :]
        cases_advertedTotal = np.sum(baselineCases) - np.sum(totalNumberOfCases[ivals, :])
        # print('cases_adverted', cases_adverted)
        prop_cases_adverted[ivals, :] = np.divide(cases_adverted, baselineCases)
        prop_cases_advertedTotal[ivals] = cases_advertedTotal/np.sum(baselineCases)
        # print(prop_cases_adverted)
    # print(np.shape(baselineCases))
    tspanFull = np.concatenate([tspan[:], tspan2[1:], tspan3[1:]])
    # print(prop_cases_adverted)
    return [tspanFull, results, prop_cases_adverted, prop_cases_advertedTotal, prop_deaths_adverted, prop_deaths_advertedTotal]

def computeHospitalizations(cases, hospRateMat):
    

    # hosp_rate = np.array([0.075901416, 0.075901416, 0.103669912,0.629689483,1.069285439,1.278737264,1.451075719,1.822122176,1.771616524,1.730339716])
    case_severity = [0.80, 0.14, 0.05]

    totalNumberOfCasesPerAgeGroup = np.sum(cases, 0)
    # print(totalNumberOfCasesPerAgeGroup)
    sympCases = 0.2*totalNumberOfCasesPerAgeGroup
    # print(sympCases)
    # print((1/100)*hospRateMat[:, 0])
    hosp_by_ageGroup = np.multiply(sympCases, (1/100)*hospRateMat[:, 0])
    # print(np.shape(hosp_by_ageGroup))
    return hosp_by_ageGroup

def computeDeaths(cases, death_rate):
    totalNumberOfCasesPerAgeGroup = np.sum(cases, 0)
    sympCases = 0.2 * totalNumberOfCasesPerAgeGroup
    deaths = np.multiply(death_rate, sympCases)
    return deaths


def runModel4(C, death_rate, gamma, gammaE, interventionDate, N, numGroups, numWeeksIntervention, red1, red2, red3, R0,  severeRate,totalPop, totalSimTime):
    #compute beta:
    beta = findBeta1(R0, C, gamma, totalPop)

    #for the time being asymptomatics are set to 0
    betaA = beta
    k = 1

    #set initial conditions
    S0 = totalPop
    S0[5] = S0[5] -1
    E0 = np.zeros(numGroups)
    I0 = np.zeros(numGroups)
    A0 = np.zeros(numGroups)
    Re0 = np.zeros(numGroups)
    RA0 = np.zeros(numGroups)

    I0[5] = 1
    Cases0 = np.copy(I0)
    y0 = np.array([S0, E0, I0, A0, Re0, RA0, Cases0]).reshape((7 * numGroups))


    #run the model up to a certain date:
    params = [beta, betaA, C, gamma, gammaE, k, N, numGroups]
    tspan = range(0, interventionDate +1)
    out1 = odeint(corEqs1, y0, tspan, args=(params,))

    cases1 = computeNewCases(out1, np.array(Cases0), numGroups)

    #run the model for the remainder of the simulation with the new matrices and store the results
    tspan2 = range(interventionDate, (interventionDate + numWeeksIntervention*7 + 1))

    if (interventionDate + numWeeksIntervention*7) < totalSimTime:
        tspan3 = range((interventionDate + numWeeksIntervention*7), totalSimTime+1)
    else:

        raise Exception('simulation out of bounds')
    y0new = np.copy(out1[-1,:])



    results = []

    #define the matrices
    C60andOver = adjustMatrix60andOverOnly(C, red1)
    ChighRiskAndChildren = adjustMatrixHighRiskAndChildren(C, red1, red2, red3)
    ChighRiskAndAdults = adjustMatrixHighRiskAndAdults(C, red1, red2, red3)
    Call = adjustMatrixAll(C, red1, red2, red3)

    Cmats = [C, C60andOver, ChighRiskAndChildren, ChighRiskAndAdults, Call]
    totalNumberOfCases = np.zeros((5, numGroups))
    deaths = np.zeros((5, numGroups))
    prop_deaths_adverted = np.zeros((5, numGroups))
    prop_deaths_advertedTotal = np.zeros(5)
    prop_cases_adverted = np.zeros((5, numGroups))
    prop_cases_advertedTotal = np.zeros((5))
    hospitalizations = np.zeros((5, numGroups))
    prop_hosp_adverted = np.zeros((5, numGroups))
    prop_hosp_advertedTotal = np.zeros(5)
    for ivals in range(5):
        # print(ivals)
        Ctemp = Cmats[ivals]
        params2 = [beta, betaA, Ctemp, gamma, gammaE, k, N, numGroups]
        out2 = odeint(corEqs1, y0new, tspan2, args=(params2,))

        cases2 = computeNewCases(out2, out2[0, 60:70], numGroups)

        #run after intervention is lifted up
        y0new3 = np.copy(out2[-1,:])
        # print(y0new3[20:30])
        # #  NOTE ###############         assumes no asymptomatics
        totalInfectiousPeriod = (1/gamma) + (1/gammaE)
        # print(totalInfectiousPeriod < numWeeksIntervention*7)
        if (totalInfectiousPeriod < numWeeksIntervention*7) & (np.all(Ctemp == 0)):
                y0new3[10:30] = 0
                y0new3[40:50] = y0new3[40:50] + np.copy(out2[-1, 10:20]) + np.copy(out2[-1, 20:30])
        # if ivals == 1 or ivals == 2 or ivals ==3:
        #     for jvals in range(10, 30):
        #         if (y0new3[jvals]) < 1:
        #             y0new3[jvals] = 0

        params = [beta, betaA, C, gamma, gammaE, k, N, numGroups]
        out3 = odeint(corEqs1, y0new3, tspan3, args=(params,))

        cases3 = computeNewCases(out3, out3[0, 60:70], numGroups)

        cases = np.concatenate([cases1[:, :], cases2[1:, :], cases3[1:, ]])
        results.append(cases)
        totalNumberOfCases[ivals, :] = np.sum(cases, 0)
        deaths[ivals, :] = np.multiply(death_rate, severeRate*np.sum(cases, 0))
        hospitalizations[ivals, :] = computeHospitalizations(cases, hospRateMat)
    # print(np.shape(totalNumberOfCases))
    # print(deaths)
    baselineDeaths = deaths[0, :]
    baselineCases = totalNumberOfCases[0, :]
    baselineHospitalizations = hospitalizations[0,:]
    # print(baselineCases)

    for ivals in range(5):
        # print(cases[ivals, :])
        deaths_adverted = (baselineDeaths - deaths[ivals, :])
        deaths_advertedTotal = (np.sum(baselineDeaths) - np.sum(deaths[ivals, :]))
        prop_deaths_adverted[ivals, :] = np.divide(deaths_adverted, baselineDeaths)
        prop_deaths_advertedTotal[ivals] = deaths_advertedTotal/np.sum(baselineDeaths)
        cases_adverted = baselineCases - totalNumberOfCases[ivals, :]
        cases_advertedTotal = np.sum(baselineCases) - np.sum(totalNumberOfCases[ivals, :])
        # print('cases_adverted', cases_adverted)
        prop_cases_adverted[ivals, :] = np.divide(cases_adverted, baselineCases)
        prop_cases_advertedTotal[ivals] = cases_advertedTotal/np.sum(baselineCases)
        hosp_adverted = baselineHospitalizations - hospitalizations[ivals, :]
        hosp_advertedTotal = np.sum(baselineHospitalizations) - np.sum(hospitalizations[ivals, :])
        prop_hosp_adverted[ivals, :] = np.divide(hosp_adverted, baselineHospitalizations)
        prop_hosp_advertedTotal[ivals] = hosp_advertedTotal/np.sum(baselineHospitalizations)
        # print(prop_cases_adverted)
    # print(np.shape(baselineCases))
    tspanFull = np.concatenate([tspan[:], tspan2[1:], tspan3[1:]])
    # print(prop_cases_adverted)
    return [tspanFull, results, prop_cases_adverted, prop_cases_advertedTotal, prop_deaths_adverted,
            prop_deaths_advertedTotal, prop_hosp_adverted, prop_hosp_advertedTotal]


#hosp rate per age group taken from Ferguson 2020
hospRateMat = np.array([[0.1, 5.0, 0.002 ],
[0.1, 5.0, 0.002 ],
[0.3, 5.0, 0.006 ],
[1.2, 5.0, 0.03 ],
[3.2, 5.0, 0.08 ],
[4.9, 6.3, 0.15 ],
[10.2, 12.2, 0.60 ],
[16.6, 27.4, 2.2 ],
[24.3, 43.2, 5.1 ],
[27.3, 70.9,9.3 ]])


# death_rate = (1/100)*hospRateMat[:, 2]
death_rate = (1/100)*np.array([0, 0, 0.2, 0.2, 0.2, 0.4, 1.3, 3.6, 8, 14.8])


if __name__ == '__main__':

    ########################## Seattle data ####################################################

    #### N = 3500000 #Seattle metropolitan area population 688245#
    #percentage of the population in each age group taken from https://factfinder.census.gov/faces/tableservices/jsf/pages/productview.xhtml?src=bkmk
    #binned as follows:
    # <5              32,036	5.3
    # 5 to 9 years    25,943	4.3
    # 10 to 14 years	22,091	3.6
    # 15 to 19 years	30,585	5.0
    # 20 to 24 years	54,885	9.0
    # 25 to 29 years	67,421	11.1
    # 30 to 34 years	59,195	9.7
    # 35 to 39 years	52,409	8.6
    # 40 to 44 years	47,295	7.8
    # 45 to 49 years	40,897	6.7
    # 50 to 54 years	39,646	6.5
    # 55 to 59 years	38,699	6.4
    # 60 to 64 years	32,063	5.3
    # 65 to 69 years	20,112	3.3
    # 70 to 74 years	12,957	2.1
    # 75 to 79 years	10,520	1.7
    # 80 to 84 years	9,539	1.6
    # 85 years and over
    seatlePop = np.array([[32036,	5.3], [25943,	4.3], [22091,	3.6], [30585,	5], [54885,	9],
                          [67421,	11.1], [59195,	9.7], [52409,	8.6], [47295,	7.8], [40897,	6.7],
                          [39646,	6.5], [38699,	6.4], [32063,	5.3], [20112,	3.3], [12957,	2.1],
                          [10520,	1.7], [9539,	1.6], [12367,	2]])

    #fracGroups as we want them:
    fracs = np.array([     5.3,  #0-5
                           4.3,  #5-10
                           3.6 + 5, #10-20
                           9 + 11.1, #20-29
                           9.7 + 8.6, #30-39
                           7.8 + 6.7, #40-49
                           6.5 + 6.4, #50-59
                           5.3+3.3, #60-69
                           2.1 + 1.7, #70-79
                           1.6 + 2.0 #>80
                           ])
    # print(np.sum(fracs))
    N = 3500000 #Seattle metropolitan area population 688245#
    totalPop = N*fracs/100
    print(totalPop)
    numGroups = 10
    # death_rate = (1/100)*np.array([0, 0, 0.2, 0.2, 0.2, 0.4, 1.3, 3.6, 8, 14.8])

    #hospitalization rate (look at file named hospitalizationsAdjustedRate.xls to see how I adjusted for this)
    # hosp_rate = [0.075901416, 0.075901416, 0.103669912,0.629689483,1.069285439,1.278737264,1.451075719,1.822122176,1.771616524,1.730339716]
    case_severity = [0.81, 0.14, 0.05]  #mild, severe and critical cases
    #contact matrix from Wallinga 2006
    # C =(1/7)* np.array([[169.14, 31.47, 17.76,34.50,15.83,11.47],
    #             [31.47, 274.51, 32.31, 34.86, 20.61, 11.50],
    #               [ 17.76,32.31,224.25,50.75,37.52,14.96],
    #               [34.50,34.86,50.75,75.66,49.45,25.08],
    #               [15.83,20.61,37.52,49.45,61.26,32.99],
    #               [11.47,11.50,14.96,25.08,32.99,54.23]])


    #contact matrix from Wallinga 2006 repeated groups to match our groups adjusted by the pop in each age group
    C =(1/7)* np.array([[169.14, 31.47, 17.76, 34.50, 34.50, 15.83, 15.83, 11.47, 11.47, 11.47],
                          [31.47, 274.51, 32.31, 34.86, 34.86, 20.61, 20.61, 11.50, 11.50, 11.50],
                          [ 17.76, 32.31, 224.25, 50.75, 50.75, 37.52, 37.52, 14.96, 14.96, 14.96],
                          [34.50, 34.86, 50.75, 75.66, 75.66, 49.45, 49.45, 25.08, 25.08, 25.08],
                            [34.50, 34.86, 50.75, 75.66, 75.66, 49.45, 49.45, 25.08, 25.08, 25.08],
                            [15.83, 20.61, 37.52, 49.45, 49.45, 61.26, 61.26, 32.99, 32.99, 32.99],
                            [15.83, 20.61, 37.52, 49.45, 49.45, 61.26, 61.26, 32.99, 32.99, 32.99],
                            [11.47, 11.50, 14.96, 25.08, 25.08, 32.99, 32.99, 54.23, 54.23, 54.23],
                          [11.47, 11.50, 14.96, 25.08, 25.08, 32.99, 32.99, 54.23, 54.23, 54.23],
                          [11.47, 11.50, 14.96, 25.08, 25.08, 32.99, 32.99, 54.23, 54.23, 54.23]])



    print((C))
    mydates = [50, 80, 100]


    # paramsFile = 'randomParams.py'
    # f = open(paramsFile, 'rb')
    # temp = pickle.load(f)
    # [R0vals, gammaEvals, gammaVals, myseed] = temp
    # f.close()


    # #hosp rate per age group taken from
    # hospRateMat = np.array([[0.1, 5.0, 0.002 ],
    # [0.3, 5.0, 0.006 ],
    # [1.2, 5.0, 0.03 ],
    # [3.2, 5.0, 0.08 ],
    # [4.9, 6.3, 0.15 ],
    # [10.2, 12.2, 0.60 ],
    # [16.6, 27.4, 2.2 ],
    # [24.3, 43.2, 5.1 ],
    # [27.3, 70.9,9.3 ]])

    severityRate = 0.2
    # print(findBeta2(2.2663529273080734, C, 0.1999516755614739, totalPop))
    # print(findBeta22(2.2663529273080734, C, 0.1999516755614739, totalPop))
    R0vector = [0.9, 2.6, 3, 3.5]
    for r0vals in range(1):#len(R0vector)):
    #     totalSimTime = mydates[simvals]
        R0 = 2#R0vector[r0vals]
        gamma = 1/5.02#gammavals #1/7
        gammaE = 1/5.16#5.17
        beta = findBeta1(R0, C, gamma, totalPop)
        print(beta)

        beta2 = findBeta2(R0, C, gamma, totalPop)
        print(beta2)
        one_over_totalPop = np.array(np.divide(np.ones(len(totalPop)), totalPop))

        betaA = beta
        k=1
        # set initial conditions
        S0 = totalPop
        S0[5] = S0[5] - 1
        E0 = np.zeros(numGroups)
        I0 = np.zeros(numGroups)
        A0 = np.zeros(numGroups)
        Re0 = np.zeros(numGroups)
        RA0 = np.zeros(numGroups)

        I0[4] = 1
        # I0 = np.array(range(1, 11))
        Cases0 = np.copy(I0)
        y0 = np.array([S0, E0, I0, A0, Re0, RA0, Cases0]).reshape((7 * numGroups))
        y02 = np.array([S0, E0, I0, Re0, Cases0]).reshape((5 * numGroups))
        # run the model up to a certain date:

        params = [beta, betaA, C, gamma, gammaE, k, N, numGroups]
        tspan = np.linspace(0,450,2000)

        interventionDate1 = 50
        numWeeks = 6
        red1, red2, red3 = 0.95, 0.85, 0.5
        severeRate = 0.2
        totalSimulationTime = 150
        start = time.time()

        print(hospRateMat[:, 0])
        # start = timer()
        # mainR = runModel3(C, death_rate, gamma, gammaE, interventionDate1, N, numGroups, numWeeks,
        #                                    red1, red2, red3, R0, severeRate, totalPop, totalSimulationTime)
        # # end =  time.time()
        # end = timer()
        # print(end - start)
        # print(np.multiply(np.dot((1 / N) * C, (beta * I + betaA * A)), S))
        # print(I0)
        # print(np.matmul((1 / N) * C, (beta * I0)))

        # print()
        # print((beta*C))
        # print(S0)
        # print((1/N)*np.dot(C, beta*I0))
        # print(np.multiply((1 / N) * np.dot(C, beta * I0), S0) - gammaE*E0)
        # print(gamma * I0)
        # outOld = odeint(corEqs1, y0, tspan, args=(paramsOld,))
        # out1 = odeint(corEqs2, y02, tspan, args=(params,))
        # print(np.sum(out1[:, 40:50], 1))
        #
        # # plt.plot(tspan, np.sum(outOld[:, 40:50], 1), color='r')
        # plt.plot(tspan, (np.sum(out1[:, 30:40], 1))/N, color='b')
    #
    plt.show()


