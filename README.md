# RAG System for Legal Document Analysis

## Deployed Application
The application is deployed at the following URL:
https://assessment1py-xgxeqhpvv2erzjvtasti2q.streamlit.app/

## How to Use
1. Go to the above URL.
2. Upload a legal document (PDF).
3. Enter a legal question to analyze the document.
4. Get the answer generated based on the document.

## Test Case

### Objective:
The test case verifies the system's ability to:

1. Upload and process a legal document (PDF).
2. Retrieve relevant information from the document based on a legal query.
2. Use AI to generate an answer based on the document contents.

### Test File:
Use the provided sample PDF to test the application. Upload it in Step 1 of the app.

### Test Question:
Ask the following legal question in Step 2: 
- "what is the ratio of the case?"

### Submit and Generate the Answer
Action: Click the Submit Query button to submit your question.
AI Analysis: The system will analyze the document and attempt to answer the question using GPT-4o, based on the document's content.

### Expected Output:
Answer: The system should return a legal reasoning based on the case in the document, similar to:
"The ratio of the case is that contributory negligence cannot be used as a complete defense in personal injury claims."

## Test Artefacts

To run the test case, the following artefact is provided in the repository:
- `sample_case3.2 Lynch v Lynch (1991) 14 MVR 512.pdf`: This PDF file is required for testing the file upload functionality. Please upload this file in Step 1 when using the application.


## Setup

Setup your repository by following these steps. In the terminal:

> conda create -n . python=3.12

> conda init

--> close and reopen terminal

> conda activate .

> conda install -c conda-forge sqlite

> pip3 install -r requirements.txt

--> run `streamlit run Home.py` to confirm everything is working

> create .env file and add OPENAI_API_KEY="" with your API key

If you can't see the following already:

> create a .streamlit folder and inside this folder create a file named config.toml

> inside of the config.toml file add the following text:

```
[server]
enableXsrfProtection = false
enableCORS = false
```

## Save Your Code

> git add -A

> git commit -m "update"

> git push

Once this is done you can delete the codespaces to free up new space.

## Reopen Terminal

If you don't see the (/opt/conda/envs), use:

`conda activate .`