# calculate edit distance as part of reward scores

from multiprocessing import Pool

def word_level_edit_distance(pair):

    str1, str2 = pair
    words1, words2 = str1.split(), str2.split()

    # Initialize matrix of zeros
    len_str1 = len(words1) + 1
    len_str2 = len(words2) + 1
    distance_matrix = [[0] * len_str2 for _ in range(len_str1)]

    # Initialize the matrix edge values
    for i in range(len_str1):
        distance_matrix[i][0] = i
    for j in range(len_str2):
        distance_matrix[0][j] = j

    # Compute word-level edit distance
    for i in range(1, len_str1):
        for j in range(1, len_str2):
            if words1[i-1] == words2[j-1]:
                cost = 0
            else:
                cost = 1
            distance_matrix[i][j] = min(
                distance_matrix[i-1][j] + 1,  # Deletion
                distance_matrix[i][j-1] + 1,  # Insertion
                distance_matrix[i-1][j-1] + cost  # Substitution
            )
    
    return distance_matrix[-1][-1] / max(len(words1), len(words2))


# Use multiprocessing to calculate distances in parallel
if __name__ == '__main__':

    # Example pairs of strings
    pairs = [
        ("The quick brown fox", "The fast brown dog"),
        ("Hello World", "Hello World!"),
        # Add more pairs as needed
    ]
    
    with Pool() as pool:
        distances = pool.map(word_level_edit_distance, pairs)
    print(distances)
