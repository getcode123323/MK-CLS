# What is this?
To the best of our knowledge, there's no publicly available word embedding model that was trained on app reviews. For that reason, we decided to create such a model and publicly provide it as part of our replication package. We trained a FastText model on 1,673,672 app reviews using the following parameters:
- Skipgram approach (not cbow).
- Vector length of 100.
- Min length of subword is 3.
- Max length of subword is 6. 
- Epoch is 50.
- Learning rate is 0.05.
Those parameters were selected based on manual analysis of different choices. We manually compared the representation provided by different parameters, and found this to provide a reasonable representation that captured most of the semantics we hoped it can capture.

# How can I use this model?
Create an array of all unique terms in your dataset, where each row is a word. Next, you can use the model to get a vector of length 100 (fixed size) for each term. The vector will tell you where the term exists in the learned space. If you provide the model with two terms that are semantically close, they should also have vector values are numerically close. To test this, you can measure the cosine similarity between the two vectors, e.g., "love" and "like".  To get the vector for a term, use the following command: 
`$ path_to_fasttext_executable print-word-vectors path_to_model < path_to_dict_of_unique_terms.txt >> outputvects/path_to_output_file.vec`

For example, given that fastext is installed (which can be downloaded from https://fasttext.cc/), and:
- Path to fasttext executable (./models/fastText/fasttext)
- Path to the provided pre-trained model is ./result/train_50_single_epoch50.bin
- Path to dictionry of unique terms is ./fse_reviews_dict.txt
- Path to desired outputfile is outputvects/fse_dim50_dict.vec
The command would be:
`$ ./models/fastText/fasttext print-word-vectors ./result/train_100_single_epoch50.bin < fse_reviews_dict.txt >> outputvects/fse_dim100_dict.vec`
