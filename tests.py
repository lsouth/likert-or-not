import numpy as np
import pandas as pd
from samplingFunctions import *
import scipy.stats
import matplotlib.pyplot as plt
import sys, random, statistics, json, math
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
        data = df[q].values
        test_result = scipy.stats.ttest_1samp(data,popmean=h0_median)
        t_stat = test_result[0]
        two_sided_p_val = test_result[1]
        one_sided_p_val = two_sided_p_val / 2
        p_val = one_sided_p_val if t_stat > 0 else 1 - one_sided_p_val
        print("H0: eta = %d vs. HA: eta > %d" % (h0_median,h0_median))
        print("One-sided T-Test: \t\t", ("Reject H0" if p_val < 0.05 else "Fail to reject H0"), "(p-val: %f)" % p_val)

        test_result, p_val = signTest(data,h0_median)
        print("Sign test: \t\t\t", ("Reject H0" if p_val < 0.05 else "Fail to reject H0"), "(p-val: %f)" % p_val)

        print()

def test_existing_two_sample(file):
    data = pd.read_csv(file)
    sample1 = data[data["sample"] == "sample1"].response
    sample2 = data[data["sample"] == "sample2"].response
    print(sample1)
    print(sample2)
    test_stat, p_val = scipy.stats.ttest_ind(sample1,sample2)
    print("Independent samples T-Test: \t\t", ("Reject H0" if p_val < 0.05 else "Fail to reject H0"), "(p-val: %f)" % p_val)

    test_stat, p_val = scipy.stats.ttest_rel(sample1,sample2)
    print("Paired samples T-Test: \t\t", ("Reject H0" if p_val < 0.05 else "Fail to reject H0"), "(p-val: %f)" % p_val)

    test_stat, p_val = scipy.stats.mannwhitneyu(x=sample1,y=sample2,alternative="greater")
    print("Mann-Whitney U Test: \t",("Reject H0" if p_val < 0.05 else "Fail to reject H0"), "(p-val: %f)" % p_val)

    test_stat, p_val = scipy.stats.wilcoxon(x=sample1,y=sample2,alternative="greater")
    print("Wilcoxon Signed Rank Test: \t",("Reject H0" if p_val < 0.05 else "Fail to reject H0"), "(p-val: %f)" % p_val)

def generate_data(true_median=4,sample_size=30,treatment="control",debug=False):
    if debug:
        print("Generating dataset of size %d centered around median %d with treatment %s." % (sample_size, true_median, treatment))
    data = None
    if treatment == "normal":
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
    elif treatment == "logistic":
        data = sampleIndependentContinuousSymmetric(numSamples=sample_size, offset=true_median,discrete=True)
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
            test_stat, p_val = scipy.stats.wilcoxon(x=sample - h0_median)
        elif test == "student":
            test_stat, p_val = scipy.stats.ttest_1samp(sample,popmean=h0_median)
    except Exception as e:
        print("Error running one sample tests.")
        print(e)
    return test_stat, p_val

def two_sample_test(sample1, sample2, test):
    try:
        if test == "student": #parametric, between-subjects.
            test_stat, p_val = scipy.stats.ttest_ind(sample1,sample2)
        elif test == "paired-student": #parametric, within-subjects.
            test_stat, p_val = scipy.stats.ttest_rel(sample1,sample2)
        elif test == "mann-whitney":   #nonparametric, between-subjects.
            test_stat, p_val = scipy.stats.mannwhitneyu(sample1,sample2)
        elif test == "wilcoxon":    #nonparametric, within-subjects.
            test_stat, p_val = scipy.stats.wilcoxon(sample1,sample2)
        return test_stat,p_val
    except ValueError as e:
        print(e)
        return -1,-1

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

def run_one_sample_tests(num_iterations=1000):
    print("Running one sample tests with %d iterations." % num_iterations)
    medians = range(5,8)
    h0_median = 4
    tests = ["sign","wilcoxon","student"]
    sample_sizes = [10,30,100]
    treatments = ["normal","asymmetric","logistic","dependent"]
    results = {"samples": []}
    results_with_iterations = {"samples": []}

    for median in medians:
        for sample_size in sample_sizes:
            for treatment in treatments:
                rhos = [0,0.2,0.5,0.8] if treatment == "dependent" else [0]
                for rho in rhos:
                    result = {"sample_size": sample_size, "median": median, "effect_size": median - h0_median, "treatment": treatment, "rho": rho, "sign": 0,"wilcoxon": 0,"student": 0}
                    result_with_iterations = {"sample_size": sample_size, "median": median, "effect_size": median - h0_median, "treatment": treatment, "rho": rho, "sign": 0,"wilcoxon": 0,"student": 0, "iterations": []}
                    type1_error_counts = [0 for test in tests]
                    type2_error_counts = [0 for test in tests]


                    for i in range(num_iterations):
                        if treatment == "dependent":
                            sample = sampleCorrelatedMultivariateNormal([median for i in range(sample_size)], sample_size, between_rho=rho, paired="none")
                        else:
                            sample = generate_data(true_median=median,sample_size=sample_size,treatment=treatment)
                        iteration_result = {"iteration": i, "tests": [], "data": []}
                        iteration_result["data"] = sample.tolist()

                        for i in range(len(tests)):
                            test = tests[i]
                            test_stat, p_val = one_sample_test(sample,test=test)
                            type1_error = True if p_val < 0.05 and median == h0_median else False
                            type2_error = True if p_val > 0.05 and median != h0_median else False
                            iteration_result["tests"].append({"name": test, "p-value": float(p_val), "test_stat": float(test_stat), "type1_error": type1_error, "type2_error": type2_error})
                            type1_error_counts[i] += 1 if type1_error else 0
                            type2_error_counts[i] += 1 if type2_error else 0
                        result_with_iterations["iterations"].append(iteration_result)
                    for i in range(len(tests)):
                        test = tests[i]
                        type1_error_rate = type1_error_counts[i] / num_iterations
                        type2_error_rate = type2_error_counts[i] / num_iterations
                        result[test] = type2_error_rate
                        result_with_iterations[test] = type2_error_rate
                        print(test, "\t\t" if test == "sign" else "\t", treatment, "\t", "n = ", sample_size, "\t","median = ",median,"\t", type2_error_counts[i], " errors made over ", num_iterations, "iterations",flush=True)
                        id += 1
                    results["samples"].append(result)
                    results_with_iterations["samples"].append(result_with_iterations)
    with open("one-sample.json", "w") as f:
        json.dump(results,f)
    with open("one-sample-with-iteration-data.json", "w") as f:
        json.dump(results_with_iterations,f)

def run_two_sample_tests(num_iterations=1000,paired=True):
    print("Running two sample tests with %d iterations." % num_iterations)
    medians = range(1,8)
    differences = range(1,7)
    h0_median = 4
    tests = ["paired-student","wilcoxon"] if paired else ["student","mann-whitney"]
    sample_sizes = [10,30,100]
    treatments = ["normal","asymmetric","logistic","dependent"]
    rhos = [0]
    results = {"samples": []}
    results_with_iterations = {"samples": []}

    for difference in differences:
        median1 = random.randint(1,7)
        median2 = random.randint(max(1,median1 - difference), min(7,median1 + difference))
        while(abs(median1 - median2) != difference):
            median1 = random.randint(1,7)
            median2 = random.randint(max(1,median1 - difference), min(7,median1 + difference))
        for sample_size in sample_sizes:
            for treatment in treatments:
                rhos = [0,0.2,0.5,0.8] if treatment == "dependent" else [0]
                for rho in rhos:
                    type1_error_counts = [0 for test in tests]
                    type2_error_counts = [0 for test in tests]
                    median_differences = 0
                    result = {"sample_size": sample_size, "median1": median1, "median2": median2, "effect_size": difference, "treatment": treatment, "rho": rho, "paired-student": 0,"wilcoxon": 0,"student": 0, "mann-whitney": 0}
                    result_with_iterations = {"sample_size": sample_size, "median1": median1, "median2": median2, "effect_size": difference, "treatment": treatment, "rho": rho, "paired-student": 0,"wilcoxon": 0,"student": 0, "mann-whitney": 0, "iterations": []}
                    if treatment == "bimodal" or treatment == "uniform":
                        result["effect_size"] = 0
                        result_with_iterations["effect_size"] = 0
                    for i in range(num_iterations):
                        if treatment == "dependent":
                            medians = [0 for m in range(sample_size * 2)]
                            for k in range(len(medians)):
                                if k % 2 == 0:
                                    medians[k] = median1
                                else:
                                    medians[k] = median2
                            samples = sampleCorrelatedMultivariateNormal(medians, sample_size*2, between_rho=rho, paired="two" if paired else "none")
                            sample1 = np.zeros(sample_size)
                            sample2 = np.zeros(sample_size)
                            for k in range(len(medians)):
                                if k % 2 == 0:
                                    sample1[int(k/2)] = samples[k]
                                else:
                                    sample2[int(k/2)] = samples[k]
                            while (sample1 == sample2).all():
                                # Rank-based tests can't run if all values are identical, so keep re-drawing until at least one is different.
                                print("Samples are identical. Redrawing...")
                                samples = sampleCorrelatedMultivariateNormal(medians, sample_size*2, between_rho=rho, paired="two" if paired else "none")
                                sample1 = np.zeros(sample_size)
                                sample2 = np.zeros(sample_size)
                                for k in range(len(medians)):
                                    if k % 2 == 0:
                                        sample1[int(k/2)] = samples[k]
                                    else:
                                        sample2[int(k/2)] = samples[k]
                            median_differences += (statistics.median(sample1) - median1) + (statistics.median(sample2) - median2)

                        else:
                            sample1 = generate_data(true_median=median1,sample_size=sample_size,treatment=treatment)
                            sample2 = generate_data(true_median=median2,sample_size=sample_size,treatment=treatment)

                        iteration_result = {"iteration": i, "tests": [], "data": []}
                        iteration_result["data"].append({"sample1": sample1.tolist(), "sample2": sample2.tolist()})
                        for i in range(len(tests)):
                            test = tests[i]
                            test_stat, p_val = two_sample_test(sample1, sample2, test=test)

                            type1_error = True if p_val < 0.05 and median1 == median2 else False
                            type2_error = True if p_val > 0.05 and median1 != median2 else False
                            # Improperly reject the null if no true difference exists between the populations.
                            type1_error_counts[i] += 1 if type1_error else 0
                            # Improperly fail to reject the null if a true difference exists between the populations.
                            type2_error_counts[i] += 1 if type2_error else 0
                            iteration_result["tests"].append({"name": test, "p-value": float(p_val), "test_stat": float(test_stat), "type1_error": type1_error, "type2_error": type2_error})
                        result_with_iterations["iterations"].append(iteration_result)
                    for i in range(len(tests)):
                        test = tests[i]
                        type1_error_rate = float(type1_error_counts[i] / num_iterations)
                        type2_error_rate = float(type2_error_counts[i] / num_iterations)
                        print(test, "\t", "n = ", sample_size, "\t", rho, "\t", "diff = ", difference, "\t", type2_error_counts[i], " errors made over ", num_iterations, "iterations",flush=True)
                        result[test] = type2_error_rate
                        result_with_iterations[test] = type2_error_rate
                    results["samples"].append(result)
                    results_with_iterations["samples"].append(result_with_iterations)

    with open("two-sample" + ("-paired" if paired else "") + ".json", "w") as f:
        json.dump(results,f)
    with open("two-sample" + ("-paired" if paired else "") + "-with-iteration-data.json", "w") as f:
        json.dump(results_with_iterations,f)

def run_three_sample_tests(num_iterations=1000, paired=True):
    print("Running three sample tests (%s) with %d iterations." % ("paired" if paired else "", num_iterations), flush=True)
    medians = range(1,8)
    differences = range(1,13)
    h0_median = 4
    tests = ["kruskal-wallis","rm-anova","friedman","anova"]
    sample_sizes = [10,30,100]
    treatments = ["normal","dependent","asymmetric","logistic"]
    rhos = [0]
    results = {"samples": []}
    results_with_iterations = {"samples": []}
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
                rhos = [0,0.2,0.5,0.8] if treatment == "dependent" else [0]
                for rho in rhos:
                    type1_error_counts = [0 for test in tests]
                    type2_error_counts = [0 for test in tests]

                    result = {"sample_size": sample_size, "median1": median1, "median2": median2,"median3": median3, "effect_size": difference, "treatment": treatment, "rho": rho, "friedman": 0,"anova": 0,"kruskal-wallis": 0, "rm-anova": 0}
                    result_with_iterations = {"sample_size": sample_size, "median1": median1, "median2": median2,"median3": median3, "effect_size": difference, "treatment": treatment, "rho": rho, "friedman": 0,"anova": 0,"kruskal-wallis": 0, "rm-anova": 0, "iterations": []}

                    for i in range(num_iterations):
                        if treatment == "dependent":
                            medians = [0 for m in range(sample_size * 3)]
                            for k in range(len(medians)-2):
                                if k % 3 == 0:
                                    medians[k] = median1
                                    medians[k+1] = median2
                                    medians[k+2] = median3
                                else: continue
                            samples = sampleCorrelatedMultivariateNormal(medians, sample_size*3, between_rho=rho, paired="three" if paired else "none")
                            sample1 = np.zeros(sample_size)
                            sample2 = np.zeros(sample_size)
                            sample3 = np.zeros(sample_size)
                            for k in range(len(medians)-2):
                                if k % 3 == 0:
                                    sample1[int(k/3)] = samples[k]
                                    sample2[int(k/3)] = samples[k+1]
                                    sample3[int(k/3)] = samples[k+2]
                                else: continue

                            while (sample1 == sample2).all() or (sample2 == sample3).all() or (sample1 == sample3).all():
                                # Rank-based tests can't run if all values are identical, so keep re-drawing until at least one is different.
                                samples = sampleCorrelatedMultivariateNormal(medians, sample_size*3, between_rho=rho, paired="three" if paired else "none")
                                sample1 = np.zeros(sample_size)
                                sample2 = np.zeros(sample_size)
                                sample3 = np.zeros(sample_size)
                                for k in range(len(medians)-2):
                                    if k % 3 == 0:
                                        sample1[int(k/3)] = samples[k]
                                        sample2[int(k/3)] = samples[k+1]
                                        sample3[int(k/3)] = samples[k+2]
                                    else: continue
                        else:
                            sample1 = generate_data(true_median=median1,sample_size=sample_size,treatment=treatment)
                            sample2 = generate_data(true_median=median2,sample_size=sample_size,treatment=treatment)
                            sample3 = generate_data(true_median=median3,sample_size=sample_size,treatment=treatment)

                        iteration_result = {"iteration": i, "tests": [], "data": []}
                        iteration_result["data"].append({"sample1": sample1.tolist(), "sample2": sample2.tolist(), "sample3": sample3.tolist()})
                        for i in range(len(tests)):
                            test = tests[i]
                            test_stat, p_val = three_sample_test(sample1, sample2, sample3, test=test)
                            type1_error = True if p_val < 0.05 and median1 == median2 and median2 == median3 else False
                            type2_error = True if p_val > 0.05 and not (median1 == median2 and median2 == median3) else False
                            # Improperly reject the null if no true difference exists between the populations.
                            type1_error_counts[i] += 1 if type1_error else 0
                            # Improperly fail to reject the null if a true difference exists between the populations.
                            type2_error_counts[i] += 1 if type2_error else 0
                            iteration_result["tests"].append({"name": test, "p-value": float(p_val), "test_stat": float(test_stat), "type1_error": type1_error, "type2_error": type2_error})
                        result_with_iterations["iterations"].append(iteration_result)
                    for i in range(len(tests)):
                        test = tests[i]
                        type1_error_rate = float(type1_error_counts[i] / num_iterations)
                        type2_error_rate = float(type2_error_counts[i] / num_iterations)
                        result[test] = type2_error_rate
                        print(test, ("\t\t" if test == "anova" else "\t"), "n = ", sample_size, "\t", rho, "\t", "diff = ", difference, "\t", type2_error_counts[i], " errors made over ", num_iterations, "iterations",flush=True)
                    results["samples"].append(result)
                    results_with_iterations["samples"].append(result_with_iterations)
    with open("three-sample" + ("-paired" if paired else "") + ".json", "w") as f:
        json.dump(results,f)
    with open("three-sample" + ("-paired" if paired else "") + "-with-iteration-data.json", "w") as f:
        json.dump(results_with_iterations,f)

def absolute_difference(a,b,c):
    return abs(a-b) + abs(b-c)

if __name__ == "__main__":
    print(sys.argv,flush=True)
    paired = len(sys.argv) > 2 and sys.argv[2] == "paired"
    if sys.argv[1] == "existing":
        print(sys.argv,flush=True)
        test_existing_two_sample(sys.argv[2])
    if sys.argv[1] == "one":
        run_one_sample_tests()
        print("Finished one sample tests.")
    if sys.argv[1] == "two":
        run_two_sample_tests(paired=paired)
        print("Finished two sample tests.")
    if sys.argv[1] == "three":
        run_three_sample_tests(paired=paired)
        print("Finished three sample tests.")
    if sys.argv[1] == "all":
        run_one_sample_tests()
        print("Finished one sample tests.")
        run_two_sample_tests()
        print("Finished two sample tests.")
        run_three_sample_tests()
        print("Finished three sample tests.")
