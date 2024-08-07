### torch 2.3.1 introduced a Numpy version incompatiblity with numpy 2.0.1 with a simple warning.
### as to date of  7/30/2024, it was still able to run. However 2.4.0 explicitely raised the error.
### the versions compatibility test did not go further than testing torch v2.3.1 & v2.4.0 against numpy v2.0.1 & v1.24.6
### downgrading numpy to 1.24.6 solved the problem =>{numpy: 1.24.6, torch: 2.3.1}

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
from typing import List

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure nltk resources are available
import nltk
nltk.download('punkt')
nltk.download('stopwords')

class OptimizedHashEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        """
        Initializes the OptimizedHashEmbedding class.

        Args:
            num_embeddings (int): The number of embeddings.
            embedding_dim (int): The dimension of each embedding.

        Initializes the following instance variables:
            - device (torch.device): The device to use for computation ('cuda' if CUDA is available, otherwise 'cpu').
            - num_embeddings (int): The number of embeddings.
            - embedding_dim (int): The dimension of each embedding.
            - embeddings (torch.nn.Embedding): The embedding layer with the specified number of embeddings and embedding dimension.

        Returns:
            None
        """
        super(OptimizedHashEmbedding, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim).to(self.device)

        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text: str):
        """
        Cleans the given text by removing specific unwanted characters, removing non-ASCII special characters, and replacing multiple spaces or newlines with a single space.

        Args:
            text (str): The text to be cleaned.

        Returns:
            str: The cleaned text.
        """
        # Remove specific unwanted characters
        text = re.sub(r'[\n\b]', ' ', text)
        # Remove non-ASCII special characters
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        # Replace multiple spaces or newlines with a single space
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def ensure_utf8_encoding(self, text):
        """
        Ensures that the given text is encoded in UTF-8 format.

        Parameters:
            text (Union[str, bytes]): The text to be encoded. It can be either a string or bytes object.

        Returns:
            str: The encoded text in UTF-8 format. If the input is already in UTF-8 format, it is returned as is.

        Raises:
            None
        """
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='ignore')
        else: 
            text = str(text)
        return text
    
    def shrink_text(self, text):
        """
        Preprocesses the given text by cleaning it, tokenizing, and removing stopwords.

        Args:
            text (str): The text to be preprocessed.

        Returns:
            str: The preprocessed text.
        """
        if not text:
            raise ValueError("Input text must not be empty.")
        text = self.clean_text(text)
        tokens = word_tokenize(text.lower())
        filtered_tokens = [word for word in tokens if word.isalnum() and word not in self.stop_words]
        return ' '.join(filtered_tokens)


    def preprocess_text(self, text: str):
        """
        Preprocesses the given text by ensuring it is in UTF-8 encoding, cleaning it, and splitting it into words.

        Args:
            text (str): The text to be preprocessed.

        Returns:
            List[str]: A list of words extracted from the preprocessed text.
        """
        # Ensure the text is in UTF-8 encoding
        text = self.ensure_utf8_encoding(text)
        # Clean the text and split it into words
        words = self.shrink_text(text).split()
        return words

    def hash_function(self, word: str):
        """
        Use a hash function to map the word to an index.

        Args:
            word (str): The word to be hashed.

        Returns:
            int: The index that the word is mapped to.
        """
        # Use a hash function to map the word to an index
        return int(hashlib.md5(word.encode(errors='ignore')).hexdigest(), 16) % self.num_embeddings

    def forward(self, words: List[str]):
        """
        Converts a list of words to indices using the hash function and returns the embeddings.

        Args:
            words (List[str]): The list of words to be converted.

        Returns:
            torch.Tensor: The embeddings of the words.
        """
        # Convert words to indices using the hash function
        indices = torch.tensor([self.hash_function(word) for word in words], device=self.device)
        return self.embeddings(indices)

    def cosine_similarity(self, embedding1, embedding2):
        """
        Calculates the cosine similarity between two embeddings.

        Args:
            embedding1 (torch.Tensor): The first embedding tensor.
            embedding2 (torch.Tensor): The second embedding tensor.

        Returns:
            float: The cosine similarity between the two embeddings.

        Normalizes the embeddings by dividing each element by its Euclidean norm. Then calculates the cosine similarity between the two normalized embeddings. The cosine similarity is a measure of similarity between two vectors that is calculated by taking the dot product of the two vectors and dividing it by the product of their magnitudes. The result is a value between -1 and 1, where 1 indicates strong similarity and -1 indicates strong dissimilarity. The function returns the cosine similarity as a float.
        """
        # Normalize the embeddings
        embedding1 = F.normalize(embedding1, p=2, dim=0)
        embedding2 = F.normalize(embedding2, p=2, dim=0)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
        return cos_sim.item()

    def euclidean_distance(self, embedding1, embedding2):
        """
        Calculates the Euclidean distance between two embeddings.

        Args:
            embedding1 (torch.Tensor): The first embedding tensor.
            embedding2 (torch.Tensor): The second embedding tensor.

        Returns:
            float: The Euclidean distance between the two embeddings.

        Calculates the Euclidean distance between two embeddings by computing the Euclidean norm of their difference. The Euclidean distance is a measure of dissimilarity between two vectors that is calculated by taking the square root of the sum of the squared differences between the corresponding elements of the vectors. The function returns the Euclidean distance as a float.
        """
        return torch.dist(embedding1, embedding2).item()

    def dot_product(self, embedding1, embedding2):
        """
        Calculates the dot product between two embedding tensors.

        Args:
            embedding1 (torch.Tensor): The first embedding tensor.
            embedding2 (torch.Tensor): The second embedding tensor.

        Returns:
            float: The dot product between the two embedding tensors.

        This function calculates the dot product between two embedding tensors by taking the dot product of the two tensors and converting the result to a float. The dot product is a measure of similarity between two vectors that is calculated by taking the sum of the products of corresponding elements of the vectors. The function returns the dot product as a float.
        """
        return torch.dot(embedding1, embedding2).item()
    
    def compute_embeddings_from_str(self, words: str):
        """
        Computes the average embedding for a given string of words.

        Args:
            words (str): A string of words.

        Returns:
            torch.Tensor: The average embedding tensor for the given string of words.
        """
        return self.forward(self.preprocess_text(words)).mean(dim=0)
    
    def compute_similarity_from_str(self, words: str, words2: str, method: str ="cosine"):
        """
        Compute the similarity between two strings of words using the specified method.

        Args:
            words (str): The first string of words.
            words2 (str): The second string of words.
            method (str, optional): The method to use for computing similarity. Defaults to "cosine".

        Returns:
            float: The similarity score between the two strings of words.

        Raises:
            ValueError: If the specified method is invalid.

        This function takes two strings of words as input and computes their similarity using the specified method. The method can be one of "cosine", "euclidean", or "dot". The similarity score is returned as a float. If an invalid method is specified, a ValueError is raised.

        Example:
            >>> embeddings = OptimizedHashEmbedding(num_embeddings=10000, embedding_dim=128)
            >>> similarity = embeddings.compute_similarity_from_str("apple banana", "banana orange", method="cosine")
            >>> print(similarity)
            0.7071067811865476

        """
        embed1 = self.compute_embeddings_from_str(words)
        embed2 = self.compute_embeddings_from_str(words2)

        if method == "cosine":
            return self.cosine_similarity(embed1, embed2)
        
        elif method == "euclidean":
            return self.euclidean_distance(embed1, embed2)
        
        elif method == "dot":
            return self.dot_product(embed1, embed2)

        else:
            raise ValueError(f"Invalid method: {method}")
        

    def compute_similarity_from_embeddings(self, words: List[str], words2: List[str], method: str ="cosine"):
        """
        Compute the similarity between two lists of words using the specified method.

        Args:
            words (List[str]): The first list of words.
            words2 (List[str]): The second list of words.
            method (str, optional): The method to use for computing similarity. Defaults to "cosine".

        Returns:
            float: The similarity score between the two lists of words.

        Raises:
            ValueError: If the specified method is invalid.

        This function takes two lists of words as input and computes their similarity using the specified method. The method can be one of "cosine", "euclidean", or "dot". The similarity score is returned as a float. If an invalid method is specified, a ValueError is raised.

        Example:
            >>> embeddings = OptimizedHashEmbedding(num_embeddings=10000, embedding_dim=128)
            >>> similarity = embeddings.compute_similarity_from_embeddings(["apple", "banana"], ["banana", "orange"], method="cosine")
            >>> print(similarity)
            0.7071067811865476
        """
        if method == "cosine":
            return self.cosine_similarity(words, words2)
            
        elif method == "euclidean":
            return self.euclidean_distance(words, words2)

        elif method == "dot":
            return self.dot_product(words, words2)
        
        else:
            raise ValueError(f"Invalid method: {method}")
        

if __name__ == "__main__":
    import time

    embedding = OptimizedHashEmbedding(num_embeddings=10000, embedding_dim=128)

    start = time.time()
    similarity = embedding.compute_similarity_from_str("apple banana", "banana orange", method="cosine")
    end = time.time()
    print(f"Time taken: {end - start}")
    print(similarity)

    start = time.time()
    embed1 = embedding.compute_embeddings_from_str("apple banana")
    embed2 = embedding.compute_embeddings_from_str("banana orange")
    similarity = embedding.cosine_similarity(embed1, embed2)
    end = time.time()
    print(f"Time taken: {end - start}")
    print(similarity)

            

