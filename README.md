# Evaluation of Named Entity Recognition Models for Russian News Texts in the Cultural Domain

## Introduction

This project aims to compare the effectiveness of different Named Entity Recognition (NER) transformer-based models in processing Russian news texts. The ultimate goal is to determine the most effective strategies for real-world applications in the cultural heritage domain.

## Data

The dataset used for this project was created by parsing and cleaning raw text from electronic newsletters sent within [the SPbLitGuide (Saint Petersburg Literary Guide)](https://isvoe.ru/spblitgid/) project between 1999 and 2019. These newsletters detailed upcoming cultural events in St. Petersburg, providing a substantial source of information for analyzing the city's literary landscape over two decades. See the [Streamlit App](https://spblitguide.streamlit.app/) for the principles of word processing and a description of the data model.

- **Dataset Size**: 15,012 records
- **Attributes**: Event ID, event description, date/time, location, address, and geographic coordinates (latitude/longitude)
- **Annotations**: Person names labeled using Doccano text annotation framework
- **Annotated Sample Size**: 1000 records

## Models

The following NER models were evaluated in this study:

- `mult_model`: Advanced BERT architecture for deep contextual learning
- `rus_ner_model`: Rule-based system for precision in controlled linguistic environments
- `roberta_large`: Optimized for the Russian language
- `spacy`: Efficient conventional NLP approach, suitable for CPU usage
- `gpt-3.5`: Advanced transformer-based model by OpenAI
- `gpt-4`: Enhanced transformer-based model with refined training methodologies
- `gpt-4o`: Optimized variant of GPT-4 with improved efficiency
- `gpt4o_json`: Variant of GPT-4o with structured output

## Evaluation

The models were evaluated based on their precision, recall, and F1 scores. Tokenizers from the respective models were used, resulting in different token counts for the same sample, highlighting the differences in tokenization approaches.

## Results


The evaluation results are summarized in the table below:

| Model           | Precision | Recall | F1 Score |
|-----------------|-----------|--------|----------|
| rus_ner_model   | 0.96      | 0.65   | 0.78     |
| mult_model      | 0.94      | 0.71   | 0.81     |
| roberta_large   | 0.92      | 0.77   | 0.84     |
| spacy           | 0.84      | 0.81   | 0.83     |
| gpt-3.5         | 0.95      | 0.71   | 0.81     |
| gpt-4           | 0.99      | 0.69   | 0.81     |
| gpt-4o          | 0.96      | 0.86   | 0.91     |
| gpt4o (json)    | 0.96      | 0.90   | 0.93     |

The evaluation results showed that:

- `gpt4o_json` achieved the highest F1 Score of 0.93, making it the most balanced and accurate model.
- `gpt-4` had the highest precision (0.99), suitable for applications where minimizing false positives is critical.
- `roberta_large` and `spacy` provided balanced performance with high efficiency, making them reliable for general use.

## Citation

If you use this repository in your research, please cite it as follows:

```bibtex
@misc{lev2024ner,
  author = {Maria Levchenko},
  title = {Evaluation of Named Entity Recognition Models for Russian News Texts in the Cultural Domain},
  year = {2024},
  howpublished = {\url{https://github.com/mary-lev/NER}},
  note = {Accessed: 2024-06-01}
}