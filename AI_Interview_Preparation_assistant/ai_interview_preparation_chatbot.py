import gradio as gr
from transformers import pipeline

pipe = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct")

few_shot_examples = """
Question: What is Machine Learning?
Answer: Machine Learning is a technology where computers learn from data and improve automatically.

Question: What is Artificial Intelligence?
Answer: Artificial Intelligence is a technology that allows machines to think and make decisions like humans.

Question: What is Overfitting?
Answer: Overfitting happens when a model learns the training data too much and performs badly on new data.

Question: What is Underfitting?
Answer: Underfitting happens when a model cannot learn the data properly because it is too simple.

Question: What is Deep Learning?
Answer: Deep Learning is a part of Machine Learning that uses neural networks to solve complex problems.

Question: What is NLP?
Answer: NLP stands for Natural Language Processing. It helps computers understand human language.

Question: What is a Neural Network?
Answer: A Neural Network is a system inspired by the human brain that helps computers recognize patterns.

Question: What is Supervised Learning?
Answer: Supervised Learning is a type of learning where the model is trained using labeled data.

Question: What is Unsupervised Learning?
Answer: Unsupervised Learning is a type of learning where the model finds patterns without labeled data.

Question: What is Generative AI?
Answer: Generative AI is a type of AI that can create text, images, code, and other content.
"""

def chatBot(message, history):

  system_prompt = """
  You are an Artificial Inteligence Interview preparation Assistant.

  Rules:
  - Give short and clear answers
  - Explain in simple language
  - Answer only the asked question
  """
  final_prompt = f"""
  {system_prompt}

  {few_shot_examples}

  User: {message}
  Answer: """

  result = pipe(
    final_prompt,
    max_new_tokens=200,
    truncation=True,
    temperature=0.3,
    max_length=None
  )

  response = result[0]['generated_text']

  cleaned_response = response.split("Answer:")[-1].strip()

  return cleaned_response

app = gr.ChatInterface(
    fn=chatBot,
    title="AI Interview Preparation Assistant 🤖",
    description="Ask anything about AI"
)    

app.launch(share=True)