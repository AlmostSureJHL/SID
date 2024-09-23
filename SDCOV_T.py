# -*- coding: utf-8 -*-
"""
Created on 2022-12-09 22:21

@author: Lijinhong
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from scipy.spatial.distance import squareform
import math


def SDCov_gauss(x, z, d, bandwidth, method, B=999, seed=1): # the pertumati
    n = np.shape(x)[0]
    timepoints = np.sort(z[d == 1])
    nFail = len(timepoints)
    inRisk = np.zeros((n, nFail))
    for i in range(nFail):
        inRisk[:, i] = (z >= timepoints[i])
    Failure = np.zeros((n, nFail))

    fc = bandwidth
    CC = 1 / (fc * np.sqrt(2 * math.pi))
    for i in range(nFail):
        # Failure[:, i] = 0.75*( 1-( (z - timepoints[i])/fc )**2 )/fc *( np.absolute( (z - timepoints[i])/fc)<=1) * (
        # d == 1)
        Failure[:, i] = CC * np.exp((-(z - timepoints[i]) ** 2) / (2 * fc ** 2)) * (d == 1)

    # L-2norm
    if method == 'SDCOV':
        X_gram = cdist(x, x, metric='euclidean')
        # X_gram = squareform(X_gram)

    # guass kenerl
    if method == 'Guass':
        X_LA = pdist(x, metric='euclidean')
        X_LB = squareform(X_LA)
        sigma = np.sqrt(0.5 * np.median(np.square(X_LA)))
        X_gram = -np.exp(- X_LB ** 2 / (2 * sigma ** 2))
        # -st.norm.pdf(X_LB, loc=0, scale=sigma)

    # laplace kenerl
    if method == 'laplace':
        X_LA = pdist(x, metric='euclidean')
        X_LB = squareform(X_LA)
        sigma = np.sqrt(0.5 * np.median(np.square(X_LA)))
        X_gram = -np.exp(-X_LB / sigma)

    M_1 = inRisk.T.dot(np.ones((n, n))).dot(inRisk)
    M_2 = np.matmul(np.matmul(np.transpose(Failure), np.ones((n, n))), inRisk)
    M_3 = np.matmul(np.matmul(np.transpose(Failure), np.ones((n, n))), Failure)
    # S1n_mat = np.matmul(np.matmul(np.transpose(Failure), X_gram), Failure) / np.matmul(
    #     np.matmul(np.transpose(Failure), np.ones((n, n))),
    #     Failure)
    S1n_mat = 1 / (n ** 4) * (np.matmul(np.matmul(np.transpose(Failure), X_gram), Failure) *
                              M_1)
    # S2n_mat = np.matmul(np.matmul(np.transpose(Failure), X_gram), inRisk) / np.matmul(
    #     np.matmul(np.transpose(Failure), np.ones((n, n))),
    #     inRisk)
    S2n_mat = 1 / (n ** 4) * (np.matmul(np.matmul(np.transpose(Failure), X_gram), inRisk) * M_2)
    # S3n_mat = np.matmul(np.matmul(np.transpose(inRisk), X_gram), inRisk) / np.matmul(
    #     np.matmul(np.transpose(inRisk), np.ones((n, n))),
    #     inRisk)
    S3n_mat = 1 / (n ** 4) * ((inRisk.T.dot(X_gram).dot(inRisk)) * M_3)

    SD = np.sum(np.diagonal(-S1n_mat + 2 * S2n_mat - S3n_mat)[range(int(0.00 * nFail), int(1 * nFail))]) / n
    local_state = np.random.RandomState(seed)
    samp = list(map(lambda x: list(local_state.choice(range(n), n, replace=False)), range(B)))
    reps = np.zeros(B + 1)
    reps[0] = SD
    for i in range(B):
        # S1n_mat = np.matmul(np.matmul(np.transpose(Failure), X_gram[samp[i]][:, samp[i]]), Failure) / np.matmul(
        #     np.matmul(np.transpose(Failure), np.ones((n, n))),
        #     Failure)
        S1n_mat = 1 / (n ** 4) * (np.matmul(np.matmul(np.transpose(Failure), X_gram[samp[i]][:, samp[i]]), Failure) *
                                  M_1)

        # S2n_mat = np.matmul(np.matmul(np.transpose(Failure), X_gram[samp[i]][:, samp[i]]), inRisk) / np.matmul(
        #     np.matmul(np.transpose(Failure), np.ones((n, n))),
        #     inRisk)

        S2n_mat = 1 / (n ** 4) * (np.matmul(np.matmul(np.transpose(Failure), X_gram[samp[i]][:, samp[i]]), inRisk) *
                                  M_2)

        S3n_mat = 1 / (n ** 4) * (np.matmul(np.matmul(np.transpose(inRisk), X_gram[samp[i]][:, samp[i]]), inRisk) *
                                  M_3)
        reps[i + 1] = np.sum(
            np.diagonal(-S1n_mat + 2 * S2n_mat - S3n_mat)[range(int(0.00 * nFail), int(1 * nFail))]) / n
    vec = pd.Series(reps)
    vec = vec.sample(frac=1).rank(method='first')
    k = vec[0]
    return (B - k + 2) / (B + 1)


def SDCov_gauss_wild_bootstrap(x, z, d, bandwidth, method, alpha, B=1999, seed=1):
    n = np.shape(x)[0]
    timepoints = np.sort(z[d == 1])
    nFail = len(timepoints)
    inRisk = np.zeros((n, nFail))
    for i in range(nFail):
        inRisk[:, i] = (z >= timepoints[i])
    Failure = np.zeros((n, nFail))

    fc = bandwidth
    CC = 1 / (fc * np.sqrt(2 * math.pi))
    for i in range(nFail):
        Failure[:, i] = CC * np.exp((-(z - timepoints[i]) ** 2) / (2 * fc ** 2)) * (d == 1)

    # L-2norm
    if method == 'SDCOV':
        X_gram = cdist(x, x, metric='euclidean')
        X_gram = X_gram ** alpha
        # X_gram = squareform(X_gram)

    # guass kenerl
    if method == 'Guass':
        X_LA = pdist(x, metric='euclidean')
        X_LB = squareform(X_LA)
        sigma = np.sqrt(0.5 * np.median(np.square(X_LA)))  # np.sqrt(0.5 * np.median(np.square(X_LA))
        X_gram = -np.exp(- X_LB ** 2 / (2 * sigma ** 2))
        # -st.norm.pdf(X_LB, loc=0, scale=sigma)

    # laplace kenerl
    if method == 'laplace':
        X_LA = pdist(x, metric='euclidean')
        X_LB = squareform(X_LA)
        sigma = np.median(X_LB[X_LB > 0])  # np.sqrt(0.5 * np.median(np.square(X_LA)))
        X_gram = -np.exp(-X_LB / sigma)

    M_1 = inRisk.T.dot(np.ones((n, n))).dot(inRisk)
    M_2 = Failure.T.dot(np.ones((n, n)).dot(inRisk))
    M_3 = Failure.T.dot(np.ones((n, n)).dot(Failure))
    S1n_mat = Failure.T.dot(X_gram.dot(Failure)) * M_1
    S2n_mat = Failure.T.dot(X_gram.dot(inRisk)) * M_2
    S3n_mat = inRisk.T.dot(X_gram).dot(inRisk) * M_3

    SD = np.sum(np.diagonal(-S1n_mat + 2 * S2n_mat - S3n_mat)) / n ** 5

    # if method == 'SDCOV':
    #     X_gram_n2 = X_gram
    # else:
    X_gram_n2 = -X_gram

    h_n2 = np.zeros((n, n, nFail))
    One = np.ones((n, 1))
    S_hat = np.sum(inRisk, axis=0) / n
    F_hat = np.sum(Failure, axis=0) / n
    for i in range(nFail):
        U_cross_term = One.dot(inRisk[:, i].reshape(1, n).dot(X_gram_n2)) / np.sum(inRisk[:, i])

        temp_1 = X_gram_n2.dot(inRisk[:, i].reshape(n, 1).dot(One.reshape(1, n)))
        U_third_term = One.dot(inRisk[:, i].reshape(1, n).dot(temp_1)) / np.sum(inRisk[:, i]) ** 2

        U_hat = X_gram_n2 - (U_cross_term + U_cross_term.T) + U_third_term

        temp_2 = (S_hat[i] * Failure[:, i] - inRisk[:, i] * F_hat[i]).reshape(n, 1)
        V_hat = temp_2.dot(temp_2.reshape(1, n))

        h_n2[:, :, i] = U_hat * V_hat

    h_2 = np.sum(h_n2, 2) / n
    # h_2 = np.triu(h_2, k=1)
    local_state = np.random.RandomState(seed)
    reps = np.zeros(B + 1)
    reps[0] = SD
    for i in range(B):
        # W = local_state.normal(0, 1, size=n)
        W = local_state.binomial(1, 1 / 2, size=n) * 2 - 1
        # W = local_state.binomial(1, (1 + np.sqrt(5)) / (2 * np.sqrt(5)), size=n)
        # W[W == 1] = (1 - np.sqrt(5)) / 2
        # W[W == 0] = (1 + np.sqrt(5)) / 2
        # reps[i + 1] = 2 / (n*(n-1)) * W.T.dot(h_2.dot(W))
        # W = W - np.mean(W)
        reps[i + 1] = 1 / (n ** 2) * W.T.dot(h_2.dot(W))

    vec = pd.Series(reps)
    vec = vec.sample(frac=1).rank(method='first')
    k = vec[0]
    return (B - k + 2) / (B + 1)
