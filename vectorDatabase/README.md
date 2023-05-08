OpenAI's text embeddings, like those produced by the GPT models, can be used to create a collection of your resumes. In order to do this, you would typically follow these steps:

1. Preprocess your resumes: Ensure that your resume data is clean and well-formatted. Remove any unnecessary information, correct spelling and grammatical errors, and standardize the format across all resumes.

2. Tokenize and embed your resumes: Using OpenAI's API or another GPT-based model, convert each resume into a sequence of tokens. Then, use the model to generate embeddings for each token. These embeddings are high-dimensional vectors that represent the semantic meaning of the text.

3. Group similar resumes: To create a collection, you'll want to group similar resumes together. One way to do this is by using a clustering algorithm, such as k-means or hierarchical clustering, on the embeddings. This will group resumes based on the similarity of their embeddings.

4. Visualize the collection: To better understand the relationships between your resumes, you may want to visualize the embeddings using a dimensionality reduction technique like t-SNE or UMAP. This will allow you to see how the resumes are organized in the embedding space and identify any trends or patterns.

5. Annotate and organize: Once you have identified clusters or groups of similar resumes, you can manually review and annotate them based on the common themes or features. This will help you create a well-organized collection of your resumes.

Here's a simple example in Python, using the Hugging Face Transformers library to obtain embeddings from your resumes:

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load GPT model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Assuming your resumes are stored as a list of strings
resumes = ["Resume 1 text...", "Resume 2 text...", ...]

# Tokenize and embed resumes
resume_embeddings = []
for resume in resumes:
    tokens = tokenizer(resume, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**tokens)
        # Use mean pooling to obtain a single vector for the entire resume
        resume_embedding = torch.mean(outputs.last_hidden_state, dim=1)
        resume_embeddings.append(resume_embedding)

# Use clustering algorithms and dimensionality reduction techniques to organize and visualize your collection
# ...
```

This example uses GPT-2, but you can replace it with GPT-3 or any other GPT-based model available in the Hugging Face Model Hub.