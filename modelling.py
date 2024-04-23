from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import networkx as nx
from sklearn.impute import SimpleImputer

# Function to calculate distances between nodes in a graph
def calculate_distances(graph):
    all_pairs_shortest_path = dict(nx.all_pairs_shortest_path_length(graph))
    distances = {}
    for node, paths in all_pairs_shortest_path.items():
        total_distance = sum(paths.values())
        avg_distance = total_distance / len(paths)
        distances[node] = avg_distance
    return distances

# Function to calculate the maximal common subgraph size between two graphs
def calculate_maximal_common_subgraph_size(graph_1, graph_2):
    num_nodes_g1 = len(graph_1)
    num_nodes_g2 = len(graph_2)
    common_nodes = set(graph_1.nodes()) & set(graph_2.nodes())
    common_subgraph_size = len(graph_1.subgraph(common_nodes))
    max_size = max(num_nodes_g1, num_nodes_g2)
    distance = 1 - (common_subgraph_size / max_size)
    return distance

# Function to preprocess the content of a document
def preprocess_document(file_path):
    with open(file_path, 'r', encoding='latin-1') as file:
        content = file.read()
    return preprocess_content(content)

# Function to create a graph from a list of words
def create_graph(words):
    graph = nx.Graph()
    graph.add_nodes_from(words)
    for i in range(len(words)-1):
        word1 = words[i]
        word2 = words[i+1]
        graph.add_edge(word1, word2)
    return graph

# Function to preprocess content (split into words)
def preprocess_content(content):
    return content.split()

# Initialize lists to store features and labels
features = []
labels = []
topics = ['Fashion', 'Disease', 'Sports']

# Process training data (36 documents)
for i, topic in enumerate(topics, start=1):
    for doc_num in range(1, 13):
        file_path = f"C:\\Users\\DELL\\Desktop\\GTProject\\Preprocessed\\{topic}\\File{doc_num}.txt"
        words = preprocess_document(file_path)
        graph = create_graph(words)
        distances = calculate_distances(graph)
        
        features_for_document = list(distances.values())  # Extract distances as a list
        
        for other_topic in topics:
            if other_topic != topic:
                for other_doc_num in range(1, 13):
                    other_file_path = f"C:\\Users\\DELL\\Desktop\\GTProject\\Preprocessed\\{other_topic}\\File{other_doc_num}.txt"
                    other_words = preprocess_document(other_file_path)
                    other_graph = create_graph(other_words)
                    distance = calculate_maximal_common_subgraph_size(graph, other_graph)
                    features_for_document.append(distance)  # Append graph distance as feature
        
        features.append(features_for_document)
        labels.append(i)

# Determine the maximum length of feature vectors
max_length = max(len(f) for f in features)

# Pad shorter feature vectors with zeros
for f in features:
    while len(f) < max_length:
        f.append(0)  # Pad with zeros until all feature vectors have the same length

# Convert lists to numpy arrays
X_train = np.array(features)
y_train = np.array(labels)

# Process testing data (9 documents)
test_features = []
test_labels = []
for i, topic in enumerate(topics, start=1):
    for doc_num in range(13, 16):  # 3 documents per topic for testing
        file_path = f"C:\\Users\\DELL\\Desktop\\GTProject\\Preprocessed\\{topic}\\File{doc_num}.txt"
        words = preprocess_document(file_path)
        graph = create_graph(words)
        distances = calculate_distances(graph)
        
        features_for_document = list(distances.values())  # Extract distances as a list
        
        for other_topic in topics:
            if other_topic != topic:
                for other_doc_num in range(1, 13):
                    other_file_path = f"C:\\Users\\DELL\\Desktop\\GTProject\\Preprocessed\\{other_topic}\\File{other_doc_num}.txt"
                    other_words = preprocess_document(other_file_path)
                    other_graph = create_graph(other_words)
                    distance = calculate_maximal_common_subgraph_size(graph, other_graph)
                    features_for_document.append(distance)  # Append graph distance as feature
        
        test_features.append(features_for_document)
        test_labels.append(i)

# Pad shorter feature vectors with zeros for testing data
for f in test_features:
    while len(f) < max_length:
        f.append(0)  # Pad with zeros until all feature vectors have the same length

# Convert lists to numpy arrays
X_test = np.array(test_features)
y_test = np.array(test_labels)

# Handle NaN values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Train the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_imputed, y_train)

# Predict labels for the test set
y_pred = knn_model.predict(X_test_imputed)

# Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
# Calculate accuracy
accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)

print("Accuracy:", accuracy)