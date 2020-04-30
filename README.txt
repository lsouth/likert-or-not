Supplementary Material for IEEE VIS 2020 submission #1179
Title: Likert or Not: Effective Use of Likert Scale Data in Visualization Evaluations
Authors: Laura South, David Saffo, Daniel Zeiberg, Cody Dunne, Olga Vitek, Michelle A. Borkin

Contents of Supplementary Materials:
-Literature survey datasets: one CSV file with all metadata collected during a survey of 137 visualization papers that used Likert scale data for evaluations. 
-Simulation code: two Python files (tests.py and samplingFunctions.py) used to simulate Likert scale datasets. 
-Simulation datasets: 10 JSON files containing simulation results for five experimental designs described in the paper. Each experimental design has two labelled JSON files. The first (named "[experimental design]-with-iterations-data.json") includes all simulated datasets across 1000 iterations and the results of all statistical tests run on each dataset (including p-value, test statistic, and statistical power accumulated across all iterations). The second (named "[experimental design].json") includes statistical power aggregated across all iterations for each simulation, but does not include the raw generated datasets, and is much smaller and easier to work with for viewing/visualizing simulation trends. 
