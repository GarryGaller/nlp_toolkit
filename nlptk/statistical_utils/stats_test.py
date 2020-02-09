import stats
from stats import euclidean_distance,euclidean_distance1,euclidean_distance_array,euclidean_distance_array_norm
from stats import cosine_dist_fast, cosine_dist, cosine_dist_from_scipy

v1 = [0,0,0.77,0.92,0]
v2 = [0,0,0.32,0.74,0]


print(cosine_dist_fast(v1,v2)) 
print(cosine_dist(v1,v2))
print(cosine_dist_from_scipy(v1,v2))
#0.0413919274008
#0.04139192740081932
#0.0413919274008



quit()
v1 = [1,0,1,0,1,0]
v2 = [1,0,1,0,1,1]

print(euclidean_distance(v1,v2))
print(euclidean_distance1(v1,v2))
print(euclidean_distance_array(v1,v2))
print(euclidean_distance_array_norm(v1,v2))



import scipy.stats

table = [1,2,3,4,5,6]

chi2, prob, df, expected = scipy.stats.chi2_contingency(table)

output = "test Statistics: {}\ndegrees of freedom: {}\np-value: {}\nexpected:{}"

print(output.format( chi2, df, prob,expected))



print(stats.chi2(table))


from stats import variation
import scipy.stats
samples = [1, 2, 3, 4, 5]
print(scipy.stats.variation(samples))
print(variation(samples))

#0.47140452079103173
#0.47140452079103173

