from scipy import stats

result = stats.binom_test(527, n=1000, p=0.5, alternative='greater')

print(result)
