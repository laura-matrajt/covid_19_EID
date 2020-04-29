import numpy as np
from scipy.integrate import odeint
import seaborn as sns
from matplotlib import pyplot as plt
import pickle
mycolors = sns.color_palette("hls", 5)

#define the equations for 6 age groups:
def corEqs(y, t, params):
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

    dS = - np.multiply(np.dot(C, (beta*I + betaA*A)), S)
    dE = np.multiply(np.dot(C, (beta*I + betaA*A)), S) -gammaE * E
    dI =  k* gammaE * E-gamma * I
    dA = (1-k) * gammaE * E -gamma* A
    dR = gamma * I
    dRA = gamma * A
    dCases = np.multiply(np.dot(C, (beta*I + betaA*A)), S)
    dydt = np.array([dS, dE, dI, dA, dR, dRA, dCases]).reshape((numGroups*7))
    return dydt


def findBeta(R0, Cadjusted, gamma, totalPop):
    # compute the eignevalues of F*V^(-1)
    [n1, n2] = np.shape(Cadjusted)

    # create F
    F = np.zeros((n1, n1))
    for jvals in range(n1):
        F[jvals, :] = Cadjusted[jvals, :] * totalPop[jvals]

    #create V
    V = np.diag(gamma * np.ones(10))
    # print(V)

    myProd = F * np.linalg.inv(V)
    myEig = np.linalg.eig(myProd)
    largestEig = np.max(myEig[0])
    # print(myEig[0])
    # print(largestEig)
    beta = R0 / largestEig
    return beta



def findBeta2(R0, Cadjusted, gamma, gammaE, totalPop):
    # compute the eignevalues of F*V^(-1)
    [n1, n2] = np.shape(Cadjusted)

    # create F
    myzeromat = np.zeros((n1, n1))
    submat1 = np.zeros((n1, n1))
    for jvals in range(n1):
        submat1[jvals, :] = Cadjusted[jvals, :] * totalPop[jvals]

    F1 = np.concatenate((myzeromat, submat1), 1)
    F2 = np.concatenate((myzeromat, myzeromat), 1)
    F = np.concatenate((F1, F2), 0)
    # print(np.shape(F))
    #create V
    submat1V = np.diag(gammaE * np.ones(n1))
    submat2V  = np.diag(gamma * np.ones(n1))
    # print(-submat1V)
    V1 = np.concatenate((-submat1V, myzeromat), 1)
    V2 = np.concatenate((submat1V, -submat2V), 1)
    V = np.concatenate((V1, V2), 0)
    # print(np.linalg.inv(V))
    print(V[:, 10])
    myProd = -F * np.linalg.inv(V)
    # print(myProd)
    myEig = np.linalg.eig(myProd)
    # print(myEig[0])
    # largestEig = np.max(myEig[0])
    # print(largestEig)
    # beta = R0 / largestEig
    # return beta


def adjustMatrixChildrenOnly(Cadjusted, red1):
    CadjustedChildren = np.copy(Cadjusted)
    CadjustedChildren[0:3, :] = (1 - red1) * Cadjusted[0:3, :]
    CadjustedChildren[3:10, 0:3] = (1 - red1) * Cadjusted[3:10, 0:3]
    return CadjustedChildren

def adjustMatrix60andOverOnly(Cadjusted, red1):
    Cadjusted60andOver = np.copy(Cadjusted)
    Cadjusted60andOver[7:10, :] = (1 - red1) * Cadjusted[7:10, :]
    Cadjusted60andOver[0:7, 7:10] = (1 - red1) * Cadjusted[0:7, 7:10]
    return Cadjusted60andOver

def adjustMatrix80andOverOnly(Cadjusted, red1):
    Cadjusted80andOver = np.copy(Cadjusted)
    Cadjusted80andOver[9:10, :] = (1 - red1) * Cadjusted[9:10, :]
    Cadjusted80andOver[0:9, 9:10] = (1 - red1) * Cadjusted[0:9, 9:10]
    return Cadjusted80andOver


def adjustMatrixHighRiskAndAdults(Cadjusted, red1, red2, red3):
    CadjustedHighRiskAndAdults = np.copy(Cadjusted)
    CadjustedHighRiskAndAdults[7:10, :] = (1 - red1) * Cadjusted[7:10, :]
    CadjustedHighRiskAndAdults[0:7, 7:10] = (1 - red1) * Cadjusted[0:7, 7:10]
    #
    CadjustedHighRiskAndAdults[3:7, 0:7] = (1 - red3) * Cadjusted[3:7, 0:7]
    CadjustedHighRiskAndAdults[0:3, 3:7] = (1 - red3) * Cadjusted[0:3, 3:7]
    return CadjustedHighRiskAndAdults

def adjustMatrixHighRiskAndChildren(Cadjusted, red1, red2, red3):
    CadjustedHighRiskAndChildren = np.copy(Cadjusted)
    CadjustedHighRiskAndChildren[7:10, :] = (1 - red1) * Cadjusted[7:10, :]
    CadjustedHighRiskAndChildren[0:7, 7:10] = (1 - red1) * Cadjusted[0:7, 7:10]
    #
    CadjustedHighRiskAndChildren[0:3, 0:7] = (1 - red2) * Cadjusted[0:3, 0:7]
    CadjustedHighRiskAndChildren[3:7, 0:3] = (1 - red2) * Cadjusted[3:7, 0:3]
    return CadjustedHighRiskAndChildren


def adjustMatrixAll(Cadjusted, red1, red2, red3):
    CadjustedAll = np.copy(Cadjusted)
    CadjustedAll[7:10, :] = (1 - red1) * Cadjusted[7:10, :]
    CadjustedAll[0:7, 7:10] = (1 - red1) * Cadjusted[0:7, 7:10]
    #
    CadjustedAll[0:3, 0:3] = (1 - red2) * Cadjusted[0:3, 0:3]
    CadjustedAll[3:7, 3:7] = (1 - red3) * Cadjusted[3:7, 3:7]
    #
    CadjustedAll[3:7, 0:3] = (1 - np.maximum(red2, red3)) * Cadjusted[3:7, 0:3]
    CadjustedAll[0:3, 3:7] = (1 - np.maximum(red2, red3)) * Cadjusted[0:3, 3:7]
    return CadjustedAll

def computeNewCases(y, I0, numGroups):
    [dim1, dim2] = np.shape(y)

    dailyCases = np.zeros((dim1, numGroups))
    temp = y[:, 60:70]
    # print np.shape(temp)
    dailyCases[0, :] = I0
    for t in range(1, dim1):

        dailyCases[t, :] = temp[t, :] - temp[t-1, :]

    return dailyCases


# def runModel(R0, gamma, gammaE, interventionDate, numWeeksIntervention, red1, red2, totalSimTime):
#     #compute beta:
#     beta = findBeta(R0, Cadjusted, gamma, totalPop)
#
#     #for the time being asymptomatics are set to 0
#     betaA = beta
#     k = 1
#
#     #set initial conditions
#     S0 = totalPop
#     S0[5] = S0[5] -1
#     E0 = np.zeros(numGroups)
#     I0 = np.zeros(numGroups)
#     A0 = np.zeros(numGroups)
#     R0 = np.zeros(numGroups)
#     RA0 = np.zeros(numGroups)
#
#     I0[5] = 1
#     Cases0 = np.copy(I0)
#     y0 = np.array([S0, E0, I0, A0, R0, RA0, Cases0]).reshape((7 * numGroups))
#
#
#     #run the model up to a certain date:
#     params = [beta, betaA, Cadjusted, gamma, gammaE, k, N, numGroups]
#     tspan = range(0, interventionDate +1)
#     out1 = odeint(corEqs, y0, tspan, args=(params,))
#
#     cases1 = computeNewCases(out1, np.array(Cases0), numGroups)
#
#     #run the model for the remainder of the simulation with the new matrices and store the results
#     tspan2 = range(interventionDate, (interventionDate + numWeeksIntervention*7 + 1))
#     # print(tspan2)
#     if (interventionDate + numWeeksIntervention*7) < totalSimTime:
#         tspan3 = range((interventionDate + numWeeksIntervention*7), totalSimTime+1)
#     else:
#         raise Exception('simulation out of bounds')
#     y0new = np.copy(out1[-1,:])
#
#
#
#     results = []
#
#     #define the matrices
#     C60andOver = adjustMatrix60andOverOnly(Cadjusted, red1)
#     ChighRiskAndChildren = adjustMatrixHighRiskAndChildren(Cadjusted, red1, red2)
#     ChighRiskAndAdults = adjustMatrixHighRiskAndAdults(Cadjusted, red1, red2)
#     Call = (1-red1)*Cadjusted
#
#     Cmats = [Cadjusted, C60andOver, ChighRiskAndChildren, ChighRiskAndAdults, Call]
#     totalNumberOfCases = np.zeros((5, numGroups))
#     deaths = np.zeros((5, numGroups))
#     prop_deaths_adverted = np.zeros((5, numGroups))
#     prop_deaths_advertedTotal = np.zeros(5)
#     prop_cases_adverted = np.zeros((5, numGroups))
#     prop_cases_advertedTotal = np.zeros((5))
#     for ivals in range(4,5):
#         Ctemp = Cmats[ivals]
#         print(Ctemp)
#         params2 = [beta, betaA, Ctemp, gamma, gammaE, k, N, numGroups]
#         out2 = odeint(corEqs, y0new, tspan2, args=(params2,))
#         print(out2[-1, 20:30])
#         cases2 = computeNewCases(out2, out2[0, 60:70], numGroups)
#
#         #run after intervention is lifted up
#         y0new3 = np.copy(out2[-1,:])
#         print(y0new3[20:30])
#         for jvals in range(20, 30):
#           if (y0new3[jvals]) < 1:
#                 y0new3[jvals] = 0
#
#         params = [beta, betaA, Cadjusted, gamma, gammaE, k, N, numGroups]
#         out3 = odeint(corEqs, y0new3, tspan3, args=(params,))
#
#         cases3 = computeNewCases(out3, out3[0, 60:70], numGroups)
#
#         cases = np.concatenate([cases1[:, :], cases2[1:, :], cases3[1:, ]])
#         results.append(cases)
#         totalNumberOfCases[ivals, :] = np.sum(cases, 0)
#         deaths[ivals, :] = death_rate*np.sum(cases, 0)
#     # print(np.shape(totalNumberOfCases))
#     # print(deaths)
#     baselineDeaths = deaths[0, :]
#     baselineCases = totalNumberOfCases[0, :]
#     # print(baselineCases)
#
#     # for ivals in range(4,5):
#     #     # print(cases[ivals, :])
#     #     deaths_adverted = (baselineDeaths - deaths[ivals, :])
#     #     deaths_advertedTotal = (np.sum(baselineDeaths) - np.sum(deaths[ivals, :]))
#     #     prop_deaths_adverted[ivals, :] = np.divide(deaths_adverted, baselineDeaths)
#     #     prop_deaths_advertedTotal[ivals] = deaths_advertedTotal/np.sum(baselineDeaths)
#     #     cases_adverted = baselineCases - totalNumberOfCases[ivals, :]
#     #     cases_advertedTotal = np.sum(baselineCases) - np.sum(totalNumberOfCases[ivals, :])
#     #     # print('cases_adverted', cases_adverted)
#     #     prop_cases_adverted[ivals, :] = np.divide(cases_adverted, baselineCases)
#     #     prop_cases_advertedTotal[ivals] = cases_advertedTotal/np.sum(baselineCases)
#         # print(prop_cases_adverted)
#     # print(np.shape(baselineCases))
#     tspanFull = np.concatenate([tspan[:], tspan2[1:], tspan3[1:]])
#     plt.plot(tspan, np.sum(out1[:, 20:30], 1))
#     plt.plot(tspan2, np.sum(out2[:, 20:30], 1))
#     plt.plot(tspan3, np.sum(out3[:, 20:30], 1))
#     # return [tspanFull, results, prop_cases_adverted, prop_cases_advertedTotal, prop_deaths_adverted, prop_deaths_advertedTotal]
#
#
#
#
# def runModel2(Cadjusted, death_rate, gamma, gammaE, interventionDate, N, numGroups, numWeeksIntervention, red1, red2, R0, totalPop, totalSimTime):
#     #compute beta:
#     beta = findBeta(R0, Cadjusted, gamma, totalPop)
#
#     #for the time being asymptomatics are set to 0
#     betaA = beta
#     k = 1
#
#     #set initial conditions
#     S0 = totalPop
#     S0[5] = S0[5] -1
#     E0 = np.zeros(numGroups)
#     I0 = np.zeros(numGroups)
#     A0 = np.zeros(numGroups)
#     R0 = np.zeros(numGroups)
#     RA0 = np.zeros(numGroups)
#
#     I0[5] = 1
#     Cases0 = np.copy(I0)
#     y0 = np.array([S0, E0, I0, A0, R0, RA0, Cases0]).reshape((7 * numGroups))
#
#
#     #run the model up to a certain date:
#     params = [beta, betaA, Cadjusted, gamma, gammaE, k, N, numGroups]
#     tspan = range(0, interventionDate +1)
#     out1 = odeint(corEqs, y0, tspan, args=(params,))
#
#     cases1 = computeNewCases(out1, np.array(Cases0), numGroups)
#
#     #run the model for the remainder of the simulation with the new matrices and store the results
#     tspan2 = range(interventionDate, (interventionDate + numWeeksIntervention*7 + 1))
#     # print(tspan2)
#     if (interventionDate + numWeeksIntervention*7) < totalSimTime:
#         tspan3 = range((interventionDate + numWeeksIntervention*7), totalSimTime+1)
#     else:
#         # myendtime = np.min()
#         raise Exception('simulation out of bounds')
#     y0new = np.copy(out1[-1,:])
#
#
#
#     results = []
#
#     #define the matrices
#     C60andOver = adjustMatrix60andOverOnly(Cadjusted, red1)
#     ChighRiskAndChildren = adjustMatrixHighRiskAndChildren(Cadjusted, red1, red2)
#     ChighRiskAndAdults = adjustMatrixHighRiskAndAdults(Cadjusted, red1, red2)
#     Call = (1-red1)*Cadjusted
#
#     Cmats = [Cadjusted, C60andOver, ChighRiskAndChildren, ChighRiskAndAdults, Call]
#     totalNumberOfCases = np.zeros((5, numGroups))
#     deaths = np.zeros((5, numGroups))
#     prop_deaths_adverted = np.zeros((5, numGroups))
#     prop_deaths_advertedTotal = np.zeros(5)
#     prop_cases_adverted = np.zeros((5, numGroups))
#     prop_cases_advertedTotal = np.zeros((5))
#     for ivals in range(5):
#         Ctemp = Cmats[ivals]
#         params2 = [beta, betaA, Ctemp, gamma, gammaE, k, N, numGroups]
#         out2 = odeint(corEqs, y0new, tspan2, args=(params2,))
#
#         cases2 = computeNewCases(out2, out2[0, 60:70], numGroups)
#
#         #run after intervention is lifted up
#         y0new3 = np.copy(out2[-1,:])
#         # print(y0new3[20:30])
#         for jvals in range(20, 30):
#           if (y0new3[jvals]) < 1:
#                 y0new3[jvals] = 0
#         params = [beta, betaA, Cadjusted, gamma, gammaE, k, N, numGroups]
#         out3 = odeint(corEqs, y0new3, tspan3, args=(params,))
#
#         cases3 = computeNewCases(out3, out3[0, 60:70], numGroups)
#
#         cases = np.concatenate([cases1[:, :], cases2[1:, :], cases3[1:, ]])
#         results.append(cases)
#         totalNumberOfCases[ivals, :] = np.sum(cases, 0)
#         deaths[ivals, :] = death_rate*np.sum(cases, 0)
#     # print(np.shape(totalNumberOfCases))
#     # print(deaths)
#     baselineDeaths = deaths[0, :]
#     baselineCases = totalNumberOfCases[0, :]
#     # print(baselineCases)
#
#     for ivals in range(5):
#         # print(cases[ivals, :])
#         deaths_adverted = (baselineDeaths - deaths[ivals, :])
#         deaths_advertedTotal = (np.sum(baselineDeaths) - np.sum(deaths[ivals, :]))
#         prop_deaths_adverted[ivals, :] = np.divide(deaths_adverted, baselineDeaths)
#         prop_deaths_advertedTotal[ivals] = deaths_advertedTotal/np.sum(baselineDeaths)
#         cases_adverted = baselineCases - totalNumberOfCases[ivals, :]
#         cases_advertedTotal = np.sum(baselineCases) - np.sum(totalNumberOfCases[ivals, :])
#         # print('cases_adverted', cases_adverted)
#         prop_cases_adverted[ivals, :] = np.divide(cases_adverted, baselineCases)
#         prop_cases_advertedTotal[ivals] = cases_advertedTotal/np.sum(baselineCases)
#         # print(prop_cases_adverted)
#     # print(np.shape(baselineCases))
#     tspanFull = np.concatenate([tspan[:], tspan2[1:], tspan3[1:]])
#     # print(prop_cases_adverted)
#     return [tspanFull, results, prop_cases_adverted, prop_cases_advertedTotal, prop_deaths_adverted, prop_deaths_advertedTotal]

def runModel3(Cadjusted, death_rate, gamma, gammaE, interventionDate, N, numGroups, numWeeksIntervention, red1, red2, red3, R0,  severeRate, totalPop, totalSimTime):

    #compute beta:
    beta = findBeta(R0, Cadjusted, gamma, totalPop)

    #for the time being asymptomatics are set to 0
    betaA = beta
    k = 1

    #set initial conditions
    S0 = totalPop
    S0[5] = S0[5] -1
    E0 = np.zeros(numGroups)
    I0 = np.zeros(numGroups)
    A0 = np.zeros(numGroups)
    R0 = np.zeros(numGroups)
    RA0 = np.zeros(numGroups)

    I0[5] = 1
    Cases0 = np.copy(I0)
    y0 = np.array([S0, E0, I0, A0, R0, RA0, Cases0]).reshape((7 * numGroups))


    #run the model up to a certain date:
    params = [beta, betaA, Cadjusted, gamma, gammaE, k, N, numGroups]
    tspan = range(0, interventionDate +1)
    out1 = odeint(corEqs, y0, tspan, args=(params,))

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
    C60andOver = adjustMatrix60andOverOnly(Cadjusted, red1)
    ChighRiskAndChildren = adjustMatrixHighRiskAndChildren(Cadjusted, red1, red2, red3)
    ChighRiskAndAdults = adjustMatrixHighRiskAndAdults(Cadjusted, red1, red2, red3)
    Call = adjustMatrixAll(Cadjusted, red1, red2, red3)

    Cmats = [Cadjusted, C60andOver, ChighRiskAndChildren, ChighRiskAndAdults, Call]
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
        out2 = odeint(corEqs, y0new, tspan2, args=(params2,))

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

        params = [beta, betaA, Cadjusted, gamma, gammaE, k, N, numGroups]
        out3 = odeint(corEqs, y0new3, tspan3, args=(params,))

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

    totalNumberOfCasesPerAgeGroup = np.sum(cases, 0)
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


def runModel4(Cadjusted, death_rate, gamma, gammaE, interventionDate, N, numGroups, numWeeksIntervention, red1, red2, red3, R0,  severeRate,totalPop, totalSimTime):
    #compute beta:
    beta = findBeta(R0, Cadjusted, gamma, totalPop)

    #for the time being asymptomatics are set to 0
    betaA = beta
    k = 1

    #set initial conditions
    S0 = totalPop
    S0[5] = S0[5] -1
    E0 = np.zeros(numGroups)
    I0 = np.zeros(numGroups)
    A0 = np.zeros(numGroups)
    R0 = np.zeros(numGroups)
    RA0 = np.zeros(numGroups)

    I0[5] = 1
    Cases0 = np.copy(I0)
    y0 = np.array([S0, E0, I0, A0, R0, RA0, Cases0]).reshape((7 * numGroups))


    #run the model up to a certain date:
    params = [beta, betaA, Cadjusted, gamma, gammaE, k, N, numGroups]
    tspan = range(0, interventionDate +1)
    out1 = odeint(corEqs, y0, tspan, args=(params,))

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
    C60andOver = adjustMatrix60andOverOnly(Cadjusted, red1)
    ChighRiskAndChildren = adjustMatrixHighRiskAndChildren(Cadjusted, red1, red2, red3)
    ChighRiskAndAdults = adjustMatrixHighRiskAndAdults(Cadjusted, red1, red2, red3)
    Call = adjustMatrixAll(Cadjusted, red1, red2, red3)

    Cmats = [Cadjusted, C60andOver, ChighRiskAndChildren, ChighRiskAndAdults, Call]
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
        out2 = odeint(corEqs, y0new, tspan2, args=(params2,))

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
        if ivals == 1 or ivals == 2 or ivals ==3:
            for jvals in range(10, 30):
                if (y0new3[jvals]) < 1:
                    y0new3[jvals] = 0

        params = [beta, betaA, Cadjusted, gamma, gammaE, k, N, numGroups]
        out3 = odeint(corEqs, y0new3, tspan3, args=(params,))

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


#hosp rate per age group taken from Ferguson 2020 first column hospital rate
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


#death rate taken from Wu 2020 (JAMA) (assuming 20% of cases are seen, like in the paper)
death_rate = (1/100)*np.array([0, 0, 0.2, 0.2, 0.2, 0.4, 1.3, 3.6, 8, 14.8])


if __name__ == '__main__':
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
    # print(totalPop)
    numGroups = 10
    death_rate = (1/100)*np.array([0, 0, 0.2, 0.2, 0.2, 0.4, 1.3, 3.6, 8, 14.8])


    #contact matrix from Wallinga 2006 repeated groups to match our groups adjusted by the pop in each age group
    # C =(1/7)* np.array([[169.14, 31.47, 17.76,34.50,15.83,11.47],
    #             [31.47, 274.51, 32.31, 34.86, 20.61, 11.50],
    #               [ 17.76,32.31,224.25,50.75,37.52,14.96],
    #               [34.50,34.86,50.75,75.66,49.45,25.08],
    #               [15.83,20.61,37.52,49.45,61.26,32.99],
    #               [11.47,11.50,14.96,25.08,32.99,54.23]])

    C = (1/7)* np.array([[169.14, 31.47, 17.76, 34.50, 34.50, 15.83, 15.83, 11.47, 11.47, 11.47],
                          [31.47, 274.51, 32.31, 34.86, 34.86, 20.61, 20.61, 11.50, 11.50, 11.50],
                          [ 17.76, 32.31, 224.25, 50.75, 50.75, 37.52, 37.52, 14.96, 14.96, 14.96],
                          [34.50, 34.86, 50.75, 75.66, 75.66, 49.45, 49.45, 25.08, 25.08, 25.08],
                            [34.50, 34.86, 50.75, 75.66, 75.66, 49.45, 49.45, 25.08, 25.08, 25.08],
                            [15.83, 20.61, 37.52, 49.45, 49.45, 61.26, 61.26, 32.99, 32.99, 32.99],
                            [15.83, 20.61, 37.52, 49.45, 49.45, 61.26, 61.26, 32.99, 32.99, 32.99],
                            [11.47, 11.50, 14.96, 25.08, 25.08, 32.99, 32.99, 54.23, 54.23, 54.23],
                          [11.47, 11.50, 14.96, 25.08, 25.08, 32.99, 32.99, 54.23, 54.23, 54.23],
                          [11.47, 11.50, 14.96, 25.08, 25.08, 32.99, 32.99, 54.23, 54.23, 54.23]])


    one_over_totalPop = np.array(np.divide(np.ones(len(totalPop)), totalPop))
    Cadjusted = C * one_over_totalPop
    mydates = [50, 80, 100]


    paramsFile = 'randomParams18Mar2020.py'
    f = open(paramsFile, 'rb')
    temp = pickle.load(f)
    [R0vals, gammaEvals, gammaVals, myseed] = temp
    f.close()

    print(np.median(R0vals))
    print(np.median(gammaVals))
    print(np.median(gammaEvals))
    # gammaEvals, gammaVals]), 0))
    severeRate = 0.2

    red1 = 0.95   #reduction in >60
    red2 = 0.85    #reduction in children
    red2vec = [0.25, 0.75, 0.95]
    red3vec = [0.25, 0.75, 0.95]  #red3 is reduction in adults

    age_groups = ['0-4', '5-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '>80']

    R0, gamma, gammaE = [np.median(R0vals),  1/np.median(gammaVals), 1/np.median(gammaEvals)]#R0vals[simvals], 1/gammaVals[simvals], 1/gammaEvals[simvals]
    print([R0, gamma, gammaE])
    interventionDate = 50
    totalSimulationTime = 150
    myweeks = [6]
    severityRate = 0.2
    for redvals in range(3):
        plt.subplot(3,1,redvals+1)
        red3 = red3vec[redvals]
        for weeks in myweeks:
            [tspanFull, results, prop_cases_adverted, prop_cases_advertedTotal, prop_deaths_adverted,
             prop_deaths_advertedTotal] = runModel3(Cadjusted, death_rate, gamma, gammaE, interventionDate, N, numGroups,
                                                    weeks,

                                             red1, red2, red3, R0, severityRate, totalPop, totalSimulationTime)

            for jvals in range(5):
                cases = results[jvals]
                totalCases = np.sum(cases, 1)
                plt.plot(tspanFull, totalCases, color=mycolors[jvals])

    plt.show()





