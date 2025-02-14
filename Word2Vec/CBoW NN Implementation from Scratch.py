#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[3]:


text = "Continuous Bag of Words (CBOW) is one of the architectures used in the Word2Vec framework for learning word embeddings. CBOW is designed to predict a target word based on its context which consists of surrounding words within a fixed window size"
text = text.split(" ")  # Keep the order intact
vocab = dict()
VOCABULARY_SIZE = len(text)

for i, word in enumerate(text):
    # Create a one-hot vector where the index corresponding to the word is set to 1
    one_hot_vector = [0] * VOCABULARY_SIZE
    one_hot_vector[i] = 1
    vocab[word] = one_hot_vector

# Print the vocabulary with one-hot vectors
for word, encoding in vocab.items():
    print(f'{word}: {encoding}\n\n')


# In[4]:


VOCABULARY_SIZE


# In[5]:


CONTEXT_WINDOW = 1


# In[61]:


def init_params():
    W1 = np.random.rand(5, 82) - 0.5
    b1 = np.random.rand(5, 1) - 0.5
    W2 = np.random.rand(41, 5) - 0.5
    b2 = np.random.rand(41, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1  # Shape: (5, m)
    A1 = ReLU(Z1)  # Shape: (5, m)
    Z2 = W2.dot(A1) + b2  # Shape: (41, m)
    A2 = softmax(Z2)  # Shape: (41, m)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((41, 1 + 1))
    one_hot_Y[np.arange(41), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y  # Shape: (41, m)
    dW2 = 1 / m * dZ2.dot(A1.T)  # Shape: (41, 5)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)  # Shape: (41, 1)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)  # Shape: (5, m)
    dW1 = 1 / m * dZ1.dot(X.T)  # Shape: (5, 82)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)  # Shape: (5, 1)
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


# In[39]:


def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
    return W1, b1, W2, b2


# In[27]:


def create_input_vectors(text, vocab, VOCABULARY_SIZE, context_window_size=1):
    input_vectors = []
    
    # Loop through each word in the text
    for i, word in enumerate(text):
        left_context = []
        right_context = []
        
        # Get the left context word (if exists)
        if i - 1 >= 0:
            left_context = vocab[text[i - 1]]
        else:
            left_context = [0] * VOCABULARY_SIZE  # Add zero vector if no left context
        
        # Get the right context word (if exists)
        if i + 1 < len(text):
            right_context = vocab[text[i + 1]]
        else:
            right_context = [0] * VOCABULARY_SIZE  # Add zero vector if no right context
        
        # Combine left context and right context to create the input vector
        input_vector = left_context + right_context
        input_vectors.append(input_vector)
    
    return input_vectors

matrix = create_input_vectors(text, vocab, VOCABULARY_SIZE, CONTEXT_WINDOW)


# In[35]:


def transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]


# In[62]:


W1, b1, W2, b2 = gradient_descent(transpose(matrix), vocab, 0.10, 500)


# In[9]:


len(vocab)

