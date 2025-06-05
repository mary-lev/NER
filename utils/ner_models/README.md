# 🤖 Simplified NER Models Infrastructure

This directory contains a **simplified NER system** providing a unified interface for various Named Entity Recognition models including OpenAI API, spaCy, and DeepPavlov.

## 🏗️ **Architecture**

### **Core Components**
- **`base.py`** - Abstract `BaseNERModel` class and `NERPrediction` data structure
- **`openai_model.py`** - OpenAI API integration (GPT-4o, GPT-4, GPT-3.5-turbo)
- **`spacy_model.py`** - spaCy model wrapper with Russian language support
- **`deeppavlov_model.py`** - DeepPavlov BERT-based models for Russian NER

### **Design Principles**
- **Direct Instantiation**: Simple, clear model creation without factory complexity
- **Abstract Base Class**: All models implement `BaseNERModel` interface
- **Unified Output**: Standardized `NERPrediction` objects with character positions
- **Error Handling**: Graceful degradation and informative error messages

## 🚀 **Quick Start**

```python
from utils.ner_models import SpacyModel

# Create and use a model directly
model = SpacyModel('ru_core_news_sm')
model.initialize()
predictions = model.predict("Встреча с писателем Иваном Петровым")

# Print results
for pred in predictions:
    print(f"'{pred.text}' ({pred.entity_type}) [{pred.start}:{pred.end}]")
```

## 📋 **Available Models**

### **spaCy Models** 🔤
- `SpacyModel('ru_spacy_ru_updated')` - Custom Russian spaCy model
- `SpacyModel('ru_core_news_sm')` - Standard Russian spaCy model
- `SpacyModel('en_core_web_sm')` - English spaCy model

### **OpenAI Models** 🤖 *(Requires API key)*
- `OpenAIModel('gpt-4o')` - GPT-4o with list output format
- `OpenAIModel('gpt-4o', output_format='json')` - GPT-4o with structured JSON output
- `OpenAIModel('gpt-4')` - GPT-4 model
- `OpenAIModel('gpt-3.5-turbo')` - GPT-3.5 Turbo model

### **DeepPavlov Models** 🧠 *(Requires installation)*
- `DeepPavlovModel('ner_collection3_bert')` - Russian BERT NER
- `DeepPavlovModel('ner_ontonotes_bert_mult')` - Multi-language BERT NER

## 🔧 **Usage Examples**

### **Basic Usage**
```python
# Direct model instantiation
from utils.ner_models import SpacyModel, OpenAIModel, DeepPavlovModel

# Create and use a spaCy model
model = SpacyModel('ru_core_news_sm')
model.initialize()
predictions = model.predict(text)
```

### **Custom Configuration**
```python
# Custom OpenAI model
custom_gpt = OpenAIModel(
    model_name='gpt-4',
    temperature=0.3,
    output_format='json'
)
custom_gpt.initialize()

# Custom spaCy model with specific entity types
custom_spacy = SpacyModel(
    model_name='ru_core_news_sm',
    entity_types={'PERSON', 'ORG'}
)
custom_spacy.initialize()
```

### **Multiple Text Processing**
```python
# Process multiple texts with simple iteration
texts = ["Text 1", "Text 2", "Text 3"]

for i, text in enumerate(texts):
    predictions = model.predict(text)
    print(f"Text {i+1}: {len(predictions)} entities")
```

### **Integration with Evaluation**
```python
# Convert to evaluation format
eval_tuples = [pred.to_tuple() for pred in predictions]
# Returns: [(start, end, entity_type), ...]

# Use with NEREvaluator
from utils import NEREvaluator
evaluator = NEREvaluator()
metrics = evaluator.calculate_metrics(eval_tuples, ground_truth)
```

## 📦 **Installation Requirements**

### **Core Dependencies**
```bash
pip install numpy pandas matplotlib
```

### **spaCy Models**
```bash
pip install spacy
python -m spacy download ru_core_news_sm
# Or custom Russian model:
pip install https://huggingface.co/Dessan/ru_spacy_ru_updated/resolve/main/ru_spacy_ru_updated-any-py3-none-any.whl
```

### **OpenAI Models**
```bash
pip install openai
export OPENAI_API_KEY="your_api_key_here"
```

### **DeepPavlov Models**
```bash
pip install deeppavlov
# First run downloads BERT models (takes 5-10 minutes)
```

## 🎯 **Model Performance**

### **Expected F1 Scores on Russian Cultural Texts**
- **DeepPavlov rus_ner_model**: ~0.78
- **DeepPavlov mult_model**: ~0.81
- **spaCy ru_core_news_sm**: ~0.65
- **OpenAI GPT-4o**: ~0.85+ *(varies by prompt)*

### **Model Characteristics**
- **spaCy**: Fast, local, good for development
- **DeepPavlov**: High accuracy, BERT-based, local inference
- **OpenAI**: Highest flexibility, requires API key, rate limited

## 🔍 **Error Handling**

```python
try:
    model = NERModelFactory.create_and_initialize('model_name')
    predictions = model.predict(text)
except ModelInitializationError as e:
    print(f"Model setup failed: {e}")
except ModelPredictionError as e:
    print(f"Prediction failed: {e}")
except UnsupportedModelError as e:
    print(f"Model not supported: {e}")
```

## 🧪 **Testing**

```python
# Test model creation
model = NERModelFactory.create_model('spacy-ru-core')
assert model.model_name == 'ru_core_news_sm'

# Test prediction format
predictions = model.predict("Test text")
assert all(isinstance(p, NERPrediction) for p in predictions)
```

## 🚀 **Extending the System**

### **Adding New Model Types**
1. Create class inheriting from `BaseNERModel`
2. Implement `initialize()` and `predict()` methods
3. Register with `NERModelFactory.register_model_type()`

### **Adding New Configurations**
```python
NERModelFactory.add_model_config('my_model', {
    'type': 'spacy',
    'model_name': 'my_custom_model',
    'entity_types': {'PERSON', 'ORG'}
})
```

## 📊 **Usage in Notebooks**

- **`NER_Models_Demo.ipynb`** - Interactive demonstration of all models
- **`Local_NER_Processing.ipynb`** - Legacy notebook with mixed usage
- **`Evaluation_Analysis.ipynb`** - Uses evaluation framework (not this infrastructure)

## 🎯 **Design Goals**

1. **Simplicity**: Direct model instantiation without unnecessary abstractions
2. **Unified Interface**: All models work through same `predict()` API
3. **Type Safety**: Clear, type-safe model creation
4. **Error Resilience**: Graceful handling of failures
5. **Single Responsibility**: Models focus only on individual text prediction
6. **Flexibility**: Customizable configurations
7. **Integration**: Compatible with existing evaluation framework

## ✅ **Benefits of Simplified Design**

- **Clearer code**: No factory or batch processing abstractions
- **Better debugging**: Direct stack traces, no complex call paths
- **Type safety**: IDE autocomplete and type checking
- **Minimal complexity**: 436+ fewer lines of unnecessary code
- **Easier testing**: Test single `predict()` method
- **Single responsibility**: Models focus only on individual text prediction

---

**🎉 This simplified system provides a clean, maintainable foundation for NER model experimentation and production use!**