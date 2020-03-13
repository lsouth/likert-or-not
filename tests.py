import numpy as np
import pandas as pd
from samplingFunctions import *
import scipy.stats
import matplotlib.pyplot as plt
import sys, random, statistics
from statsmodels.stats.anova import AnovaRM

h0_median = 4

def signTest(sample, etaNull):
    alpha = 0.05
    b = sum(sample > etaNull)
    nStar = sum(sample != etaNull)
    p = stats.binom.cdf(nStar - b, nStar, 0.5)
    return b,p

def to_likert_range(x, x_min, x_max):
    x_min <- min(x)
    x_max <- max(x)
    likert_start = 1
    likert_end = 7
    likert <- (((x - x_min) * (likert_end - likert_start)) / (x_max - x_min)) + likert_start
    return round(likert)

def test_existing_data(file):
    df = pd.read_csv(file)

    for q in df.columns:
        if q == "participant":
            continue
        print("%s: " % q)
        data = df[q].values
        test_result = scipy.stats.ttest_1samp(data,popmean=h0_median)
        t_stat = test_result[0]
        two_sided_p_val = test_result[1]
        one_sided_p_val = two_sided_p_val / 2
        p_val = one_sided_p_val if t_stat > 0 else 1 - one_sided_p_val
        print("H0: eta = %d vs. HA: eta > %d" % (h0_median,h0_median))
        print("One-sided T-Test: \t\t", ("Reject H0" if p_val < 0.05 else "Fail to reject H0"), "(p-val: %f)" % p_val)

        # sign_result = scipy.stats.mannwhitneyu(x=data-h0_median,alternative="greater")
        # print("Mann-Whitney U Test: \t",("Reject H0" if mann_whitney_result[1] < 0.05 else "Fail to reject H0"), "(p-val: %f)" % mann_whitney_result[1])
        #
        # test_result = scipy.stats.wilcoxon(x=data - h0_median,alternative="greater")
        # t_stat = test_result[0]
        # p_val = test_result[1]
        # print("Wilcoxon Signed Rank Test: \t",("Reject H0" if p_val < 0.05 else "Fail to reject H0"), "(p-val: %f)" % p_val)

        test_result, p_val = signTest(data,h0_median)
        print("Sign test: \t\t\t", ("Reject H0" if p_val < 0.05 else "Fail to reject H0"), "(p-val: %f)" % p_val)

        print()

def generate_data(true_median=4,sample_size=30,treatment="control",debug=False):
    if debug:
        print("Generating dataset of size %d centered around median %d with treatment %s." % (sample_size, true_median, treatment))
    data = None
    if treatment == "control":
        data = sampleIndependentNormal(numSamples=sample_size, offset=true_median, discrete=True)
    elif treatment == "uniform":
        # Errors from independent Normal not centered at zero, still constant variance.
        data = sampleUniform(numSamples=sample_size, discrete=True)
    elif treatment == "dependent":
        # Errors are dependent and drawn from a Normal distribution.
        data = generateDependentSamplesLatentNormal(numSamples=sample_size,offset=true_median, discrete=True)
    elif treatment == "asymmetric":
        data = sampleIndependentContinuousAsymmetric(numSamples=sample_size, offset=true_median, discrete=True)
    elif treatment == "bimodal":
        data = sampleBeta(numSamples=sample_size)
    else:
        print("Did not recognize treatment (%s)." % treatment)
    return data

def one_sample_test(sample,h0_median=4,test="sign", debug=False):
    if debug:
        print("Running one sample %s test for significance." % (test))
    test_stat = 0
    p_val = 0
    try:
        if test == "sign":
            test_stat, p_val = signTest(sample,h0_median)
        elif test == "wilcoxon":
            test_stat, p_val = scipy.stats.wilcoxon(x=sample - h0_median,alternative="greater")
        elif test == "student":
            test_stat, two_sided_p_val = scipy.stats.ttest_1samp(sample,popmean=h0_median)
            one_sided_p_val = two_sided_p_val / 2
            p_val = one_sided_p_val if test_stat > 0 else 1 - one_sided_p_val
    except Exception as e:
        print(sample)
        print(e)
    return test_stat, p_val

def two_sample_test(sample1, sample2, test):
    if test == "student": #parametric, between-subjects.
        test_stat, two_sided_p_val = scipy.stats.ttest_ind(sample1,sample2)
        one_sided_p_val = two_sided_p_val / 2
        p_val = one_sided_p_val if test_stat > 0 else 1 - one_sided_p_val
    elif test == "paired-student": #parametric, within-subjects.
        test_stat, two_sided_p_val = scipy.stats.ttest_rel(sample1,sample2)
        one_sided_p_val = two_sided_p_val / 2
        p_val = one_sided_p_val if test_stat > 0 else 1 - one_sided_p_val
    elif test == "mann-whitney":   #nonparametric, between-subjects.
        test_stat, p_val = scipy.stats.mannwhitneyu(sample1,sample2,alternative="greater")
    elif test == "wilcoxon":    #nonparametric, within-subjects.
        test_stat, p_val = test_stat, p_val = scipy.stats.wilcoxon(sample1,sample2,alternative="greater")
    return test_stat,p_val

def three_sample_test(sample1, sample2, sample3, test):
    if test == "anova": #parametric, between-subjects.
        test_stat, p_val = scipy.stats.f_oneway(sample1, sample2, sample3)
    elif test =="rm-anova": #parametric, within-subjects.
        data = {"response": [], "id": [], "group": []}
        for i in range(len(sample1)):
            data["response"].append(sample1[i])
            data["id"].append(i)
            data["group"].append("A")

            data["response"].append(sample2[i])
            data["id"].append(i)
            data["group"].append("B")

            data["response"].append(sample3[i])
            data["id"].append(i)
            data["group"].append("C")

        df = pd.DataFrame(data=data)
        anova_rm = AnovaRM(df,depvar="response",subject="id",within=["group"])
        res = anova_rm.fit()
        test_stat = res.anova_table['F Value'][0]
        p_val = res.anova_table['Pr > F'][0]
    elif test == "kruskal-wallis": #nonparametric, between-subjects.
        test_stat, p_val = scipy.stats.kruskal(sample1, sample2, sample3)
    elif test == "friedman": #nonparametric, within-subjects.
        test_stat, p_val = scipy.stats.friedmanchisquare(sample1, sample2, sample3)
    return test_stat,p_val

def run_one_sample_tests(num_iterations=10):
    print("Running one sample tests with %d iterations." % num_iterations)
    medians = range(1,8)
    h0_median = 4
    tests = ["sign","wilcoxon","student"]
    sample_sizes = [10,30,100]
    treatments = ["control","uniform","asymmetric","bimodal","dependent"]
    with open("one-sample.csv","w") as f_summary:
        f_summary.write("id,treatment,sample_size,median,test,type1_error_rate,type2_error_rate\n")
        id = 0
        for median in medians:
            for sample_size in sample_sizes:
                for treatment in treatments:
                    type1_error_counts = [0 for test in tests]
                    type2_error_counts = [0 for test in tests]
                    if treatment == "uniform" or treatment == "bimodal":
                        median = 4
                    with open("one_sample/diff-%d-m1-%d-n-%d-treatment-%s.csv" % (abs(median - h0_median),median,sample_size,treatment), "w+") as f:
                        f.write("iteration,response,id\n")
                        for i in range(num_iterations):
                            sample = generate_data(true_median=median,sample_size=sample_size,treatment=treatment)
                            if median != statistics.median(sample):
                                print("Treatment: %s Expected median: %d Actual median: %d" % (treatment, median, statistics.median(sample)))
                            for obs in range(len(sample)):
                                f.write("%d,%.1f,%d\n" % (i,sample[obs], obs))
                            for i in range(len(tests)):
                                test = tests[i]
                                test_stat, p_val = one_sample_test(sample,test=test)
                                type1_error_counts[i] += 1 if p_val < 0.05 and median < h0_median else 0
                                type2_error_counts[i] += 1 if p_val > 0.05 and median > h0_median else 0
                    for i in range(len(tests)):
                        test = tests[i]
                        type1_error_rate = type1_error_counts[i] / num_iterations
                        type2_error_rate = type2_error_counts[i] / num_iterations
                        f_summary.write("%d,%s,%d,%d,%s,%.4f,%.4f\n" % (id,treatment,sample_size,median,test,type1_error_rate,type2_error_rate))
                        id += 1

def run_two_sample_tests(num_iterations=100):
    print("Running two sample tests with %d iterations." % num_iterations)
    medians = range(1,8)
    differences = range(0,7)
    h0_median = 4
    tests = ["paired-student","wilcoxon","student","mann-whitney"]
    sample_sizes = [10,30,100]
    treatments = ["control","uniform","asymmetric","bimodal","dependent"]
    rhos = [0]
    with open("two-sample.csv","w") as f_summary:
        f_summary.write("id,treatment,sample_size,median1,median2,rho,test,type1_error_rate,type2_error_rate\n")
        id = 0
        for difference in differences:
            median1 = random.randint(1,7)
            median2 = random.randint(max(1,median1 - difference), min(7,median1 + difference))
            while(abs(median1 - median2) != difference):
                median1 = random.randint(1,7)
                median2 = random.randint(max(1,median1 - difference), min(7,median1 + difference))
            for sample_size in sample_sizes:
                for treatment in treatments:
                    rhos = [0,0.1,0.5,0.8] if treatment == "dependent" else [0]
                    for rho in rhos:
                        type1_error_counts = [0 for test in tests]
                        type2_error_counts = [0 for test in tests]
                        with open("two_sample/diff-%d-m1-%d-m2-%d-n-%d-rho-%d-treatment-%s.csv" % (difference,median1, median2,sample_size,rho * 100, treatment), "w+") as f:
                            f.write("iteration,response,id,group\n")
                            for i in range(num_iterations):
                                if treatment == "dependent":
                                    samples = sampleCorrelatedMultivariateNormal([median1, median2], sample_size, rho)
                                    sample1 = categorize(samples.T[0])
                                    sample2 = categorize(samples.T[1])
                                else:
                                    sample1 = generate_data(true_median=median1,sample_size=sample_size,treatment=treatment)
                                    sample2 = generate_data(true_median=median2,sample_size=sample_size,treatment=treatment)

                                    for obs in range(len(sample1)):
                                        f.write("%d,%.1f,%d,%s\n" % (i,sample1[obs], obs, "1"))
                                        f.write("%d,%.1f,%d,%s\n" % (i,sample2[obs], obs, "2"))
                                for i in range(len(tests)):
                                    test = tests[i]
                                    test_stat, p_val = two_sample_test(sample1, sample2, test=test)
                                    # Improperly reject the null if no true difference exists between the populations.
                                    type1_error_counts[i] += 1 if p_val < 0.05 and median1 == median2 else 0
                                    # Improperly fail to reject the null if a true difference exists between the populations.
                                    type2_error_counts[i] += 1 if p_val > 0.05 and median1 != median2 else 0
                        for i in range(len(tests)):
                            test = tests[i]
                            type1_error_rate = type1_error_counts[i] / num_iterations
                            type2_error_rate = type2_error_counts[i] / num_iterations
                            f_summary.write("%d,%s,%d,%d,%d,%.1f,%s,%.4f,%.4f\n" % (id,treatment,sample_size,median1,median2,rho,test,type1_error_rate,type2_error_rate))
                            id += 1

def run_three_sample_tests(num_iterations=10, within_subjects=False):
    print("Running three sample tests (%s) with %d iterations." % ("within-subjects" if within_subjects else "between-subjects", num_iterations), flush=True)
    medians = range(1,8)
    differences = range(0,13)#[0,2,4,6,8,10,12]
    h0_median = 4
    tests = ["friedman","anova","kruskal-wallis","rm-anova"]
    sample_sizes = [10,30,100]
    treatments = ["control","uniform","asymmetric","bimodal","dependent"]
    rhos = [0]
    with open("three-sample.csv","w") as f_summary:
        f_summary.write("id,treatment,sample_size,median1,median2,median3,rho,test,type1_error_rate,type2_error_rate\n")
        id = 0
        max_diff = 0
        for difference in differences:
            median1 = random.randint(1,7)
            median2 = random.randint(max(1,median1 - difference), min(7,median1 + difference))
            remaining_difference = difference - abs(median1 - median2)
            median3 = max(1,median1 - remaining_difference) if random.random() > 0.5 else min(7,median1 + remaining_difference)
            while(absolute_difference(median1,median2, median3) != difference):
                median1 = random.randint(1,7)
                median2 = random.randint(max(1,median1 - difference), min(7,median1 + difference))
                remaining_difference = difference - abs(median1 - median2)
                median3 = max(1,median1 - remaining_difference) if random.random() > 0.5 else min(7,median1 + remaining_difference)
            for sample_size in sample_sizes:
                for treatment in treatments:
                    if treatment == "dependent":
                        rhos = [0,0.2,0.5,0.8]
                    for rho in rhos:
                        type1_error_counts = [0 for test in tests]
                        type2_error_counts = [0 for test in tests]
                        with open("three_sample/diff-%d-m1-%d-m2-%d-m3-%d-n-%d-rho-%d-treatment-%s.csv" % (difference,median1, median2,median3,sample_size,rho * 100, treatment), "w+") as f:
                            f.write("iteration,response,id,group\n")
                            for i in range(num_iterations):
                                if treatment == "dependent":
                                    samples = sampleCorrelatedMultivariateNormal([median1, median2, median3], sample_size, rho)
                                    sample1 = categorize(samples.T[0])
                                    sample2 = categorize(samples.T[1])
                                    sample3 = categorize(samples.T[2])
                                else:
                                    sample1 = generate_data(true_median=median1,sample_size=sample_size,treatment=treatment)
                                    sample2 = generate_data(true_median=median2,sample_size=sample_size,treatment=treatment)
                                    sample3 = generate_data(true_median=median3,sample_size=sample_size,treatment=treatment)

                                for obs in range(len(sample1)):
                                    f.write("%d,%.1f,%d,%s\n" % (i,sample1[obs], obs, "1"))
                                    f.write("%d,%.1f,%d,%s\n" % (i,sample2[obs], obs, "2"))
                                    f.write("%d,%.1f,%d,%s\n" % (i,sample3[obs], obs, "3"))

                                for i in range(len(tests)):
                                    test = tests[i]
                                    test_stat, p_val = three_sample_test(sample1, sample2, sample3, test=test)
                                    # Improperly reject the null if no true difference exists between the populations.
                                    type1_error_counts[i] += 1 if p_val < 0.05 and median1 == median2 and median2 == median3 else 0
                                    # Improperly fail to reject the null if a true difference exists between the populations.
                                    type2_error_counts[i] += 1 if p_val > 0.05 and not (median1 == median2 and median2 == median3) else 0
                        for i in range(len(tests)):
                            test = tests[i]
                            type1_error_rate = type1_error_counts[i] / num_iterations
                            type2_error_rate = type2_error_counts[i] / num_iterations
                            f_summary.write("%d,%s,%d,%d,%d,%d,%.1f,%s,%.4f,%.4f\n" % (id,treatment,sample_size,median1,median2,median3,rho,test,type1_error_rate,type2_error_rate))

def absolute_difference(a,b,c):
    return abs(a-b) + abs(b-c)

if __name__ == "__main__":
    if sys.argv[1] == "existing":
        test_existing_data(sys.argv[2])
    if sys.argv[1] == "one":
        run_one_sample_tests()
        print("Finished one sample tests.")
    if sys.argv[1] == "two":
        run_two_sample_tests()
        print("Finished two sample tests.")
    if sys.argv[1] == "three":
        run_three_sample_tests()
        print("Finished three sample tests.")
    if sys.argv[1] == "all":
        run_one_sample_tests()
        print("Finished one sample tests.")
        run_two_sample_tests()
        print("Finished two sample tests.")
        run_three_sample_tests()
        print("Finished three sample tests.")
